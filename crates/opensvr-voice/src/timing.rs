//! Neural phoneme duration prediction (`PhonemeTiming`).
//!
//! Ports OpenSV's `src/synthesis/PhonemeTiming.{h,cpp}` as of the
//! voice-compatible upstream design (the `gen2a` loader): syllables are
//! mapped to timing intervals through the voice's per-language [`PhoneSet`]s
//! (a syllable language matches by substring), consonant onsets are shifted
//! onto the beat ([`align_intervals`]), a 272-channel feature tensor is built
//! per phoneme (column-copied embeddings plus sinusoidal [`encode_scalar`]
//! expansions), one full [`DnniInference`] pass predicts log-durations, and
//! the results are denormalized and distributed across each interval
//! ([`PhonemeTiming::predict`]). [`quantize_phoneme_durations`] then snaps
//! the continuous durations onto acoustic frames with int32 boundaries.
//!
//! Provenance note: the pinned OpenSV revision's `PhonemeTiming::load`
//! expects an 8-child frontend that the available real voice does not have;
//! its duration model carries a 4-child frontend with the embedding tables in
//! `modl4` wrappers at root level instead. This crate ports the loader that
//! reads the real voice (same error strings, same feature math), so golden
//! tests against the C++ build stay meaningful. `FeatureNormalizer` is
//! unaffected (identical logic in both revisions).

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniInference, DnniMatrix, DnniReader, DnniTensor};
use opensvr_g2p::{PhoneSet, PhonemeDuration, TimingSyllable, read_phone_set};

use crate::normalizer::FeatureNormalizer;
use crate::{Result, invalid};

// Total input feature dimension of the duration network:
// 32 (phoneme) + 32 (language) + 16 (forward pos) + 16 (reverse pos)
// + 32 (encoded duration) + 16 (encoded pitch) + 128 (speaker/voice)
// = 272 channels.
const FEATURE_CHANNELS: usize = 272;
const MAXIMUM_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;
const PHRASE_LIMIT: usize = MAXIMUM_TENSOR_ELEMENTS / FEATURE_CHANNELS;

// Model architecture type hashes in DNNI.
const DURATION_TYPES: [u64; 12] = [
    0x4c30_eaff_390f_599a,
    0x210a_e732_bcda_3950,
    0xedb2_643e_c310_49d7,
    0x76b7_7e5b_f8d6_500c,
    0x8e0f_0182_b380_a860,
    0xd4c2_d794_6f91_e167,
    0x135e_e088_9750_1166,
    0x71a4_b344_df3f_7ece,
    0xa4f6_23b0_0c5a_97c0,
    0x1fad_6bd2_dfa1_a17f,
    0x5d5d_600f_0e54_53e9,
    0x1db5_5850_4d28_a4ad,
];
const FRONTEND_TYPES: [u64; 12] = [
    0xff36_9e74_e595_e206,
    0xea62_ca7c_31e3_7ca0,
    0x319b_1e6e_3791_7bd3,
    0x0590_9feb_a966_bb54,
    0x18b5_591e_22ab_a550,
    0x825a_3110_2572_a303,
    0xf0c8_bdd6_e4fe_863a,
    0x77f5_f7f5_51d9_6e82,
    0xd917_7b50_6384_3fb0,
    0xa097_f32e_2e5f_e07b,
    0xc6ea_192c_f9c0_95d1,
    0x3287_0689_962b_ff1d,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Phoneme timing: {}", reason.into()))
}

/// One phoneme with its alignment metadata during interval building.
#[derive(Debug, Clone)]
struct TimingPhone {
    syllable_index: usize,
    phoneme_index: usize,
    language_index: usize,
    unified_index: usize,
    is_nucleus: bool,
}

/// One musical timing interval: the phones sounding within a note boundary.
#[derive(Debug, Clone)]
struct TimingInterval {
    phones: Vec<TimingPhone>,
    duration_seconds: f64,
    midi_pitch: i32,
}

/// Aligns phonemes to musical note boundaries for natural singing synthesis.
///
/// Step 1 (onset shift): leading consonants before the first nucleus move into
/// the preceding interval so the vowel nucleus lands on the note onset; index
/// 0 has no predecessor and is left alone, as is an interval without a
/// nucleus. Step 2 (multi-nuclei split): intervals with more than one nucleus
/// split into equal-duration pieces cut right before each subsequent nucleus,
/// carrying the MIDI pitch over.
fn align_intervals(mut intervals: Vec<TimingInterval>) -> Vec<TimingInterval> {
    for index in 1..intervals.len() {
        let nucleus = intervals[index]
            .phones
            .iter()
            .position(|phone| phone.is_nucleus);
        if let Some(nucleus) = nucleus {
            let moved: Vec<TimingPhone> = intervals[index].phones.drain(..nucleus).collect();
            intervals[index - 1].phones.extend(moved);
        }
    }

    let mut result = Vec::with_capacity(intervals.len());
    for interval in intervals {
        let nuclei = interval
            .phones
            .iter()
            .filter(|phone| phone.is_nucleus)
            .count();
        if nuclei <= 1 {
            result.push(interval);
            continue;
        }
        // Split the duration evenly across the vowel nuclei, cutting right
        // before each subsequent nucleus.
        let seconds = interval.duration_seconds / nuclei as f64;
        let mut begin = 0_usize;
        let mut found_nucleus = false;
        for (index, phone) in interval.phones.iter().enumerate() {
            if phone.is_nucleus {
                if found_nucleus {
                    result.push(TimingInterval {
                        phones: interval.phones[begin..index].to_vec(),
                        duration_seconds: seconds,
                        midi_pitch: interval.midi_pitch,
                    });
                    begin = index;
                }
                found_nucleus = true;
            }
        }
        result.push(TimingInterval {
            phones: interval.phones[begin..].to_vec(),
            duration_seconds: seconds,
            midi_pitch: interval.midi_pitch,
        });
    }
    result
}

/// Encodes one continuous scalar into sinusoidal position channels.
///
/// The destination splits into two halves: sines in the first, cosines in the
/// second, with exponentially spaced frequencies (`exp2` steps of
/// `-26.575424 / len`). A zero value yields all-zero sines and all-one
/// cosines.
fn encode_scalar(value: f32, destination: &mut [f32]) {
    let half = destination.len() / 2;
    for index in 0..half {
        let angle = value * (index as f32 * -26.575_424_f32 / destination.len() as f32).exp2();
        destination[index] = angle.sin();
        destination[index + half] = angle.cos();
    }
}

/// Copies one embedding column into the feature buffer.
///
/// Embeddings are stored channels-by-entries (`rows` channels,
/// `columns` entries); the `column`-th entry fills the whole destination,
/// which must hold exactly `rows` channels.
fn copy_embedding(matrix: &DnniMatrix, column: usize, destination: &mut [f32]) {
    for (row, slot) in destination.iter_mut().enumerate() {
        *slot = matrix.values[row * matrix.columns as usize + column];
    }
}

/// One discrete acoustic frame span assigned to a phoneme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedPhoneme {
    pub language: String,
    pub symbol: String,
    pub frame_count: usize,
}

/// Quantizes continuous phoneme durations into discrete acoustic frames.
///
/// Boundaries accumulate from zero (`accumulated * framesPerSecond`,
/// truncated toward zero like the C++ `int32` cast) so rounding never
/// drifts; every phoneme receives at least one frame. Empty inputs, invalid
/// intervals and non-positive durations are rejected, and `output` is only
/// assigned on success.
pub fn quantize_phoneme_durations(
    durations: &[PhonemeDuration],
    frame_interval_seconds: f32,
    output: &mut Vec<TimedPhoneme>,
) -> Result<()> {
    if durations.is_empty()
        || durations.len() > PHRASE_LIMIT
        || !frame_interval_seconds.is_finite()
        || frame_interval_seconds <= 0.0
    {
        return Err(failure(
            "invalid phoneme sequence or acoustic frame interval for time quantization.",
        ));
    }
    let frames_per_second = 1.0 / f64::from(frame_interval_seconds);
    let mut quantized = Vec::with_capacity(durations.len());
    let mut accumulated_seconds = 0.0_f64;
    let mut previous_boundary = 0_i32;
    for phoneme in durations {
        if !phoneme.duration_seconds.is_finite()
            || phoneme.duration_seconds <= 0.0
            || phoneme.language.is_empty()
            || phoneme.symbol.is_empty()
        {
            return Err(failure(
                "time quantization requires positive finite durations and explicit phonemes.",
            ));
        }
        accumulated_seconds += phoneme.duration_seconds;
        let boundary = accumulated_seconds * frames_per_second;
        if !boundary.is_finite() || boundary > f64::from(i32::MAX) || previous_boundary == i32::MAX
        {
            return Err(failure(
                "quantized phoneme boundaries exceed the supported int32 frame range.",
            ));
        }
        // Truncation toward zero matches the C++ `int32` cast; the accumulated
        // boundary is always positive here.
        let next_boundary = (previous_boundary + 1).max(boundary as i32);
        quantized.push(TimedPhoneme {
            language: phoneme.language.clone(),
            symbol: phoneme.symbol.clone(),
            frame_count: (next_boundary - previous_boundary) as usize,
        });
        previous_boundary = next_boundary;
    }
    *output = quantized;
    Ok(())
}

fn read_word(bytes: &[u8], offset: usize) -> i32 {
    // Little-endian word; the bit pattern matches C++'s
    // `std::bit_cast<std::int32_t>(readWord(...))`. Callers guarantee
    // `offset + 4` fits.
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Reads an embedding matrix out of its `modl4` wrapper and verifies dims.
///
/// The wrapper must be an empty `modl4` node with exactly one matrix child
/// holding `rows` channels by `columns` entries.
fn read_embedding(
    reader: &DnniReader,
    node_index: usize,
    rows: usize,
    columns: usize,
    matrix: &mut DnniMatrix,
) -> Result<()> {
    let wrapper = reader.nodes().get(node_index).filter(|node| {
        node.name == "modl4" && reader.payload(node_index).is_empty() && node.children.len() == 1
    });
    let Some(wrapper) = wrapper else {
        return Err(failure(
            "an embedding must be an empty modl4 node with one matrix.",
        ));
    };
    *matrix = reader.read_float_matrix(wrapper.children[0])?;
    if matrix.rows as usize != rows || matrix.columns as usize != columns {
        return Err(failure("an embedding matrix has unsupported dimensions."));
    }
    Ok(())
}

/// Frontend tables loaded from the 4-child descriptor node.
struct FrontendTables {
    phone_sets: Vec<PhoneSet>,
    unified_phone_set: PhoneSet,
    input_normalization: FeatureNormalizer,
    output_normalization: FeatureNormalizer,
    minimum_pitch: i32,
    maximum_pitch: i32,
}

/// Loads the phone sets, normalizers and pitch bounds from the frontend.
fn load_frontend(reader: &DnniReader, frontend_node: usize) -> Result<FrontendTables> {
    let nodes = reader.nodes();
    let frontend_payload = reader.payload(frontend_node);
    let frontend_type = nodes.get(frontend_node).map_or(0, |node| node.type_id);
    if !FRONTEND_TYPES.contains(&frontend_type)
        || nodes
            .get(frontend_node)
            .is_none_or(|node| node.children.len() != 4)
        || frontend_payload.len() != 8
    {
        return Err(failure("unsupported duration feature frontend."));
    }
    let minimum_pitch = read_word(frontend_payload, 0);
    let maximum_pitch = read_word(frontend_payload, 4);
    if minimum_pitch < 0 || maximum_pitch > 127 || minimum_pitch >= maximum_pitch {
        return Err(failure("invalid MIDI pitch normalization range."));
    }
    let frontend_children = nodes[frontend_node].children.clone();

    // Read the phone sets and the unified phone set.
    let group = &nodes[frontend_children[0]];
    if group.name != "cmpg1"
        || !reader.payload(frontend_children[0]).is_empty()
        || group.children.is_empty()
    {
        return Err(failure(
            "the duration frontend requires a nonempty language phone-set group.",
        ));
    }
    let mut phone_sets = Vec::with_capacity(group.children.len());
    for &child in &group.children.clone() {
        phone_sets.push(read_phone_set(reader, child)?);
    }
    let unified_phone_set = read_phone_set(reader, frontend_children[1])?;
    let unified_symbols: std::collections::HashSet<&str> = unified_phone_set
        .symbols
        .iter()
        .map(String::as_str)
        .collect();
    for phone_set in &phone_sets {
        for symbol in &phone_set.unified_symbols[..phone_set.symbols.len()] {
            if !unified_symbols.contains(symbol.as_str()) {
                return Err(failure(
                    "a language phoneme maps to an unknown unified symbol.",
                ));
            }
        }
    }

    // Normalizers (exactly one channel each).
    let mut input_normalization = FeatureNormalizer::default();
    input_normalization.load(reader, frontend_children[2])?;
    let mut output_normalization = FeatureNormalizer::default();
    output_normalization.load(reader, frontend_children[3])?;
    if input_normalization.channel_count() != 1 || output_normalization.channel_count() != 1 {
        return Err(failure(
            "duration normalizers must each contain one channel.",
        ));
    }
    Ok(FrontendTables {
        phone_sets,
        unified_phone_set,
        input_normalization,
        output_normalization,
        minimum_pitch,
        maximum_pitch,
    })
}
///
/// Predicts the continuous time duration (in seconds) of each phoneme within
/// a singing phrase; see the module docs for the pipeline.
/// Neural phoneme duration prediction model (`gen2a` duration model).
///
/// Predicts the continuous time duration (in seconds) of each phoneme within
/// a singing phrase; see the module docs for the pipeline.
#[derive(Debug)]
pub struct PhonemeTiming {
    phone_sets: Vec<PhoneSet>,
    unified_phone_set: PhoneSet,
    input_normalization: FeatureNormalizer,
    output_normalization: FeatureNormalizer,
    language_embedding: DnniMatrix,
    phoneme_embedding: DnniMatrix,
    position_embedding: DnniMatrix,
    voice_embedding: Vec<f32>,
    network: DnniInference,
    minimum_pitch: i32,
    maximum_pitch: i32,
}

impl PhonemeTiming {
    /// Creates an unloaded model; [`predict`](Self::predict) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self {
            phone_sets: Vec::new(),
            unified_phone_set: PhoneSet::default(),
            input_normalization: FeatureNormalizer::default(),
            output_normalization: FeatureNormalizer::default(),
            language_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            phoneme_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            position_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            voice_embedding: Vec::new(),
            network: DnniInference::new(),
            minimum_pitch: 0,
            maximum_pitch: 0,
        }
    }

    /// Loads the verified `gen2a` duration model weights and embeddings.
    ///
    /// The root carries six children (frontend, network, three `modl4`
    /// embeddings, voice vector) and an 8-byte `[32, 16]` feature-layout
    /// payload; the frontend carries four children (phone-set group, unified
    /// set, two 1-channel normalizers) and the MIDI pitch bounds. Everything
    /// is staged into a candidate first: a failed load preserves the current
    /// model.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(root) = nodes
            .get(root_node)
            .filter(|root| DURATION_TYPES.contains(&root.type_id))
        else {
            return Err(failure(
                "the selected node is not the verified gen2a duration model.",
            ));
        };
        let payload = reader.payload(root_node);
        if root.children.len() != 6
            || payload.len() != 8
            || read_word(payload, 0) != 32
            || read_word(payload, 4) != 16
        {
            return Err(failure("unsupported gen2a feature layout."));
        }
        let children = root.children.clone();
        let frontend = load_frontend(reader, children[0])?;

        // Embeddings live at root level: language (32 x languages), phoneme
        // (32 x unified symbols), position (16 x 8).
        let mut language_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(
            reader,
            children[2],
            32,
            frontend.phone_sets.len(),
            &mut language_embedding,
        )?;
        let mut phoneme_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(
            reader,
            children[3],
            32,
            frontend.unified_phone_set.symbols.len(),
            &mut phoneme_embedding,
        )?;
        let mut position_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(reader, children[4], 16, 8, &mut position_embedding)?;

        // Speaker / voice embedding vector (128 floats).
        let voice_embedding = reader.read_float_vector(children[5])?;
        if voice_embedding.len() != 128 {
            return Err(failure(
                "the duration voice embedding must contain 128 channels.",
            ));
        }

        // Neural network.
        let mut network = DnniInference::new();
        network.load(reader, children[1])?;

        self.phone_sets = frontend.phone_sets;
        self.unified_phone_set = frontend.unified_phone_set;
        self.input_normalization = frontend.input_normalization;
        self.output_normalization = frontend.output_normalization;
        self.language_embedding = language_embedding;
        self.phoneme_embedding = phoneme_embedding;
        self.position_embedding = position_embedding;
        self.voice_embedding = voice_embedding;
        self.network = network;
        self.minimum_pitch = frontend.minimum_pitch;
        self.maximum_pitch = frontend.maximum_pitch;
        Ok(())
    }

    /// Predicts the duration in seconds of each phoneme in a syllable phrase.
    ///
    /// Include an explicit leading rest if the first syllable's consonants
    /// should start early; the whole phrase is processed together. `output`
    /// is only assigned on success. This allocates and runs neural inference.
    pub fn predict(
        &self,
        syllables: &[TimingSyllable],
        output: &mut Vec<PhonemeDuration>,
    ) -> Result<()> {
        if self.phone_sets.is_empty() {
            return Err(failure("no duration model has been loaded."));
        }
        if syllables.is_empty() || syllables.len() > PHRASE_LIMIT {
            return Err(failure(
                "the phrase is empty or exceeds the tensor size limit.",
            ));
        }

        let (intervals, phoneme_count) = self.build_intervals(syllables)?;
        let (features, mut durations) =
            self.render_features(syllables, &intervals, phoneme_count)?;

        // 5. Neural network inference (single full pass, no cache).
        let mut predicted = DnniTensor::default();
        self.network
            .run(&features, &mut predicted, None, &CancelToken::new())?;
        if predicted.frames != phoneme_count
            || predicted.channels != 1
            || predicted.values.len() != phoneme_count
        {
            return Err(failure(
                "the duration network must return one value per phoneme.",
            ));
        }

        // 6. Denormalize the predicted log-durations.
        let mut log_durations = DnniTensor::default();
        self.output_normalization
            .denormalize(&predicted, &mut log_durations)?;

        // 7. Softmax-like proportional distribution of each note duration.
        distribute_durations(&intervals, &mut log_durations.values, &mut durations)?;

        *output = durations;
        Ok(())
    }

    /// Maps syllables to aligned timing intervals, validating every field.
    ///
    /// Returns the intervals plus the total phoneme count.
    fn build_intervals(
        &self,
        syllables: &[TimingSyllable],
    ) -> Result<(Vec<TimingInterval>, usize)> {
        let mut intervals = Vec::with_capacity(syllables.len());
        let mut phoneme_count = 0_usize;
        for (syllable_index, syllable) in syllables.iter().enumerate() {
            if syllable.phonemes.is_empty()
                || syllable.phonemes.len() > PHRASE_LIMIT.saturating_sub(phoneme_count)
            {
                return Err(failure(
                    "a syllable is empty or exceeds the tensor size limit.",
                ));
            }
            if !syllable.duration_seconds.is_finite()
                || syllable.duration_seconds <= 0.0
                || syllable.duration_seconds > f64::from(f32::MAX)
                || syllable.midi_pitch < 0
                || syllable.midi_pitch > 127
            {
                return Err(failure(
                    "syllable duration must be finite and positive, and pitch must be a MIDI note.",
                ));
            }
            phoneme_count += syllable.phonemes.len();
            if syllable.language.is_empty() {
                return Err(failure("an explicit phoneme language is required."));
            }
            // The syllable language matches by nonempty substring, exactly once.
            let mut language_index = self.phone_sets.len();
            for (index, phone_set) in self.phone_sets.iter().enumerate() {
                if phone_set.name.contains(syllable.language.as_str()) {
                    if language_index != self.phone_sets.len() {
                        return Err(failure("the language matches more than one phone set."));
                    }
                    language_index = index;
                }
            }
            if language_index == self.phone_sets.len() {
                return Err(failure(format!(
                    "unknown phoneme language: {}.",
                    syllable.language
                )));
            }
            let phone_set = &self.phone_sets[language_index];
            let mut interval = TimingInterval {
                phones: Vec::with_capacity(syllable.phonemes.len()),
                duration_seconds: syllable.duration_seconds,
                midi_pitch: syllable.midi_pitch,
            };
            for (position, symbol) in syllable.phonemes.iter().enumerate() {
                let Some(source_index) = phone_set
                    .symbols
                    .iter()
                    .position(|candidate| candidate == symbol)
                else {
                    return Err(failure(format!(
                        "unknown phoneme '{symbol}' for {}.",
                        phone_set.name
                    )));
                };
                // The loader verified every mapping, so this always resolves.
                let unified = &phone_set.unified_symbols[source_index];
                let Some(target_index) = self
                    .unified_phone_set
                    .symbols
                    .iter()
                    .position(|candidate| candidate == unified)
                else {
                    return Err(failure(
                        "a language phoneme maps to an unknown unified symbol.",
                    ));
                };
                let category = &phone_set.categories[source_index];
                interval.phones.push(TimingPhone {
                    syllable_index,
                    phoneme_index: position,
                    language_index,
                    unified_index: target_index,
                    is_nucleus: category == "vowel" || category == "diphthong",
                });
            }
            intervals.push(interval);
        }
        Ok((align_intervals(intervals), phoneme_count))
    }

    /// Builds the 272-channel feature tensor plus duration shells.
    fn render_features(
        &self,
        syllables: &[TimingSyllable],
        intervals: &[TimingInterval],
        phoneme_count: usize,
    ) -> Result<(DnniTensor, Vec<PhonemeDuration>)> {
        // Normalized log-duration input features.
        let mut raw_values = Vec::with_capacity(intervals.len());
        for interval in intervals {
            // The reference uses log(durationSeconds + 0.01) to prevent the
            // log(0) singularity; 0.01f widens to double exactly as in C++.
            raw_values.push(((interval.duration_seconds + f64::from(0.01_f32)).ln()) as f32);
        }
        let raw_duration = DnniTensor {
            frames: intervals.len(),
            channels: 1,
            values: raw_values,
        };
        let mut normalized_duration = DnniTensor::default();
        self.input_normalization
            .normalize(&raw_duration, &mut normalized_duration, true)?;

        let mut features = DnniTensor {
            frames: phoneme_count,
            channels: FEATURE_CHANNELS,
            values: vec![0.0; phoneme_count * FEATURE_CHANNELS],
        };
        let mut durations = Vec::with_capacity(phoneme_count);
        let mut frame = 0_usize;
        for (interval_index, interval) in intervals.iter().enumerate() {
            let pitch = (interval.midi_pitch - self.minimum_pitch) as f32
                / (self.maximum_pitch - self.minimum_pitch) as f32;
            for (position, phone) in interval.phones.iter().enumerate() {
                let syllable = &syllables[phone.syllable_index];
                let destination =
                    &mut features.values[frame * FEATURE_CHANNELS..(frame + 1) * FEATURE_CHANNELS];
                // Feature layout (column-copied embeddings):
                // [0..32): phoneme embedding (32 channels).
                copy_embedding(
                    &self.phoneme_embedding,
                    phone.unified_index,
                    &mut destination[..32],
                );
                // [32..64): language embedding (32 channels).
                copy_embedding(
                    &self.language_embedding,
                    phone.language_index,
                    &mut destination[32..64],
                );
                // [64..80): forward position in syllable, clamped to 7.
                copy_embedding(
                    &self.position_embedding,
                    position.min(7),
                    &mut destination[64..80],
                );
                // [80..96): reverse position from the end, clamped to 7.
                copy_embedding(
                    &self.position_embedding,
                    (interval.phones.len() - position - 1).min(7),
                    &mut destination[80..96],
                );
                // [96..128): sinusoidal expansion of the normalized duration.
                encode_scalar(
                    normalized_duration.values[interval_index],
                    &mut destination[96..128],
                );
                // [128..144): sinusoidal expansion of the normalized pitch.
                encode_scalar(pitch, &mut destination[128..144]);
                // [144..272): voice / speaker identity embedding.
                destination[144..272].copy_from_slice(&self.voice_embedding);

                durations.push(PhonemeDuration {
                    language: syllable.language.clone(),
                    symbol: syllable.phonemes[phone.phoneme_index].clone(),
                    syllable_index: phone.syllable_index,
                    timing_interval_index: interval_index,
                    duration_seconds: 0.0,
                });
                frame += 1;
            }
        }
        Ok((features, durations))
    }
}

impl Default for PhonemeTiming {
    fn default() -> Self {
        Self::new()
    }
}

/// Distributes each interval duration across its phonemes.
///
/// The denormalized network outputs are exponentiated and each phoneme takes
/// its softmax-like share of the musical interval.
fn distribute_durations(
    intervals: &[TimingInterval],
    mut values: &mut [f32],
    mut durations: &mut [PhonemeDuration],
) -> Result<()> {
    for interval in intervals {
        let len = interval.phones.len();
        let (head_values, tail_values) = values.split_at_mut(len);
        let (head_durations, tail_durations) = durations.split_at_mut(len);
        values = tail_values;
        durations = tail_durations;
        let mut sum = 0.0_f32;
        for slot in head_values.iter_mut() {
            let value = slot.exp();
            if !value.is_finite() || value <= 0.0 {
                return Err(failure(
                    "a predicted duration is not a finite positive float.",
                ));
            }
            *slot = value;
            sum += value;
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(failure(
                "the predicted timing interval duration sum is invalid.",
            ));
        }
        let reciprocal = 1.0 / sum;
        for (slot, duration) in head_values.iter().zip(head_durations.iter_mut()) {
            // Allocate this fraction of the musical interval to the phoneme.
            let fraction = *slot * reciprocal;
            let seconds = (f64::from(fraction) * interval.duration_seconds) as f32;
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(failure(
                    "the normalized duration cannot be represented as a positive float.",
                ));
            }
            duration.duration_seconds = f64::from(seconds);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_dnni::DnniReader;

    // --- Pure alignment tests (unchanged algorithm) ---

    fn phone(
        syllable_index: usize,
        phoneme_index: usize,
        language_index: usize,
        unified_index: usize,
        is_nucleus: bool,
    ) -> TimingPhone {
        TimingPhone {
            syllable_index,
            phoneme_index,
            language_index,
            unified_index,
            is_nucleus,
        }
    }

    fn interval(
        phones: Vec<TimingPhone>,
        duration_seconds: f64,
        midi_pitch: i32,
    ) -> TimingInterval {
        TimingInterval {
            phones,
            duration_seconds,
            midi_pitch,
        }
    }

    fn nuclei(intervals: &[TimingInterval]) -> Vec<Vec<bool>> {
        intervals
            .iter()
            .map(|interval| {
                interval
                    .phones
                    .iter()
                    .map(|phone| phone.is_nucleus)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn onset_shift_moves_leading_consonants_to_the_previous_interval() {
        let intervals = vec![
            interval(vec![phone(0, 0, 0, 0, false)], 0.4, 60),
            interval(
                vec![
                    phone(1, 0, 0, 1, false),
                    phone(1, 1, 0, 2, false),
                    phone(1, 2, 0, 3, true),
                ],
                0.6,
                62,
            ),
        ];
        let aligned = align_intervals(intervals);
        assert_eq!(aligned.len(), 2);
        assert_eq!(aligned[0].phones.len(), 3);
        assert_eq!(aligned[1].phones.len(), 1);
        assert!(aligned[1].phones[0].is_nucleus);
        assert_eq!(aligned[0].duration_seconds, 0.4);
        assert_eq!(aligned[1].duration_seconds, 0.6);
        assert_eq!(aligned[0].midi_pitch, 60);
        assert_eq!(aligned[1].midi_pitch, 62);
        assert_eq!(aligned[0].phones[1].syllable_index, 1);
        assert_eq!(aligned[0].phones[1].phoneme_index, 0);
    }

    #[test]
    fn first_interval_has_no_predecessor_to_shift_into() {
        let intervals = vec![interval(
            vec![phone(0, 0, 0, 1, false), phone(0, 1, 0, 3, true)],
            0.5,
            60,
        )];
        let aligned = align_intervals(intervals);
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned[0].phones.len(), 2);
    }

    #[test]
    fn interval_without_nucleus_shifts_nothing() {
        let intervals = vec![
            interval(vec![phone(0, 0, 0, 0, true)], 0.4, 60),
            interval(vec![phone(1, 0, 0, 4, false)], 0.3, 62),
        ];
        let aligned = align_intervals(intervals);
        assert_eq!(nuclei(&aligned), vec![vec![true], vec![false]]);
    }

    #[test]
    fn multi_nuclei_interval_splits_evenly_before_each_later_nucleus() {
        let intervals = vec![interval(
            vec![
                phone(0, 0, 0, 0, true),
                phone(0, 1, 0, 1, false),
                phone(0, 2, 0, 2, true),
                phone(0, 3, 0, 3, true),
            ],
            0.9,
            64,
        )];
        let aligned = align_intervals(intervals);
        assert_eq!(aligned.len(), 3);
        assert_eq!(
            nuclei(&aligned),
            vec![vec![true, false], vec![true], vec![true]]
        );
        for piece in &aligned {
            assert_eq!(piece.duration_seconds, 0.3);
            assert_eq!(piece.midi_pitch, 64);
        }
    }

    #[test]
    fn single_nucleus_interval_is_unchanged() {
        let intervals = vec![interval(
            vec![
                phone(0, 0, 0, 1, false),
                phone(0, 1, 0, 0, true),
                phone(0, 2, 0, 1, false),
            ],
            0.5,
            60,
        )];
        let aligned = align_intervals(intervals);
        assert_eq!(aligned.len(), 1);
        assert_eq!(aligned[0].phones.len(), 3);
        assert_eq!(aligned[0].duration_seconds, 0.5);
    }

    // --- Sinusoidal scalar encoding ---

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= allowed,
                "channel {index}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn encode_scalar_zero_yields_zero_sines_and_unit_cosines() {
        let mut out = [9.0_f32; 4];
        encode_scalar(0.0, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn encode_scalar_matches_hand_computed_sinusoids() {
        // angle(i) = value * exp2(i * -26.575424 / len); channel i holds
        // sin(angle), channel i + half holds cos(angle).
        let mut out = [0.0_f32; 4];
        encode_scalar(1.0, &mut out);
        assert_close(
            &out,
            &[0.841_471, 0.009_999_83, 0.540_302_3, 0.999_95],
            1e-6,
        );
    }

    #[test]
    fn encode_scalar_pairs_stay_on_the_unit_circle() {
        let mut wide = [0.0_f32; 32];
        encode_scalar(0.43, &mut wide);
        assert!(wide.iter().all(|value| value.is_finite()));
        for pair in 0..16 {
            let (sine, cosine) = (wide[pair], wide[pair + 16]);
            assert!(sine.abs() <= 1.0 && cosine.abs() <= 1.0);
            assert!((sine * sine + cosine * cosine - 1.0).abs() < 1e-6);
        }
        let mut narrow = [0.0_f32; 16];
        encode_scalar(-2.0, &mut narrow);
        assert!(narrow.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn copy_embedding_copies_the_indexed_column() {
        // 2 channels x 3 entries; column 1 is [2, 5].
        let matrix = DnniMatrix {
            rows: 2,
            columns: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let mut destination = [0.0_f32; 2];
        copy_embedding(&matrix, 1, &mut destination);
        assert_eq!(destination, [2.0, 5.0]);
    }

    // --- int32-boundary quantization ---

    fn duration(
        language: &str,
        symbol: &str,
        syllable_index: usize,
        timing_interval_index: usize,
        duration_seconds: f64,
    ) -> PhonemeDuration {
        PhonemeDuration {
            language: language.to_owned(),
            symbol: symbol.to_owned(),
            syllable_index,
            timing_interval_index,
            duration_seconds,
        }
    }

    fn frame_counts(output: &[TimedPhoneme]) -> Vec<usize> {
        output.iter().map(|phoneme| phoneme.frame_count).collect()
    }

    #[test]
    fn quantize_splits_cumulative_boundaries_into_exact_frames() {
        let durations = vec![
            duration("japanese", "k", 0, 0, 0.25),
            duration("japanese", "a", 0, 0, 0.25),
            duration("japanese", "sil", 1, 1, 0.5),
        ];
        let mut output = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut output).expect("must quantize");
        assert_eq!(
            output,
            vec![
                TimedPhoneme {
                    language: "japanese".to_owned(),
                    symbol: "k".to_owned(),
                    frame_count: 50,
                },
                TimedPhoneme {
                    language: "japanese".to_owned(),
                    symbol: "a".to_owned(),
                    frame_count: 50,
                },
                TimedPhoneme {
                    language: "japanese".to_owned(),
                    symbol: "sil".to_owned(),
                    frame_count: 100,
                },
            ]
        );
    }

    #[test]
    fn quantize_truncates_boundaries_toward_zero() {
        // 3 x 0.006 s at 5 ms frames: f64 boundaries 1.2000/2.4000/3.6000
        // truncate to 1/2/3 (the float-cumulative variant would round up).
        let durations = vec![
            duration("english", "s", 0, 0, 0.006),
            duration("english", "ih", 0, 0, 0.006),
            duration("english", "l", 0, 0, 0.006),
        ];
        let mut output = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut output).expect("must quantize");
        assert_eq!(frame_counts(&output), vec![1, 1, 1]);
        assert_eq!(frame_counts(&output).iter().sum::<usize>(), 3);
    }

    #[test]
    fn quantize_gives_every_phoneme_at_least_one_frame() {
        let durations = vec![
            duration("japanese", "k", 0, 0, 1e-9),
            duration("japanese", "a", 0, 0, 1e-9),
        ];
        let mut output = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut output).expect("must quantize");
        assert_eq!(frame_counts(&output), vec![1, 1]);
    }

    #[test]
    fn quantize_rejects_empty_input_and_bad_intervals() {
        let mut output = vec![TimedPhoneme {
            language: "sentinel".to_owned(),
            symbol: "x".to_owned(),
            frame_count: 7,
        }];
        assert_eq!(
            quantize_phoneme_durations(&[], 0.005, &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: invalid phoneme sequence or acoustic frame interval for time quantization."
        );
        for interval in [0.0, -0.005, f32::INFINITY, f32::NAN] {
            assert_eq!(
                quantize_phoneme_durations(
                    &[duration("japanese", "a", 0, 0, 0.25)],
                    interval,
                    &mut output
                )
                .unwrap_err()
                .to_string(),
                "Phoneme timing: invalid phoneme sequence or acoustic frame interval for time quantization."
            );
        }
        // The output is untouched on failure.
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].symbol, "x");
    }

    #[test]
    fn quantize_rejects_non_positive_durations_and_blank_labels() {
        let bad = [
            duration("japanese", "a", 0, 0, 0.0),
            duration("japanese", "a", 0, 0, -0.1),
            duration("japanese", "a", 0, 0, f64::INFINITY),
            duration("", "a", 0, 0, 0.1),
            duration("japanese", "", 0, 0, 0.1),
        ];
        for durations in bad.chunks(1) {
            let mut output = Vec::new();
            assert_eq!(
                quantize_phoneme_durations(durations, 0.005, &mut output)
                    .unwrap_err()
                    .to_string(),
                "Phoneme timing: time quantization requires positive finite durations and explicit phonemes."
            );
            assert!(output.is_empty());
        }
    }

    #[test]
    fn quantize_rejects_boundaries_past_int32() {
        // 1e8 s at 5 ms frames: the boundary (~2e10) exceeds i32::MAX.
        let durations = vec![duration("japanese", "a", 0, 0, 1e8)];
        let mut output = Vec::new();
        assert_eq!(
            quantize_phoneme_durations(&durations, 0.005, &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: quantized phoneme boundaries exceed the supported int32 frame range."
        );
    }

    // --- Synthetic DNNI blob builders (v2 object trees, gen2a layout) ---

    const FNV_PRIME_64: u64 = 0x0100_0000_01b3;
    const CHILD_XOR_MASK: u32 = 0xac5e_7bd5;

    fn fnv(seed: u64, name: &str) -> u64 {
        let mut hash = seed;
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME_64);
        }
        hash
    }

    fn named_id(name: &str) -> u64 {
        fnv(0x0dcd_5918_9d5a_0f24, name)
    }

    fn v2_node(type_id: u64, payload: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
        let low = type_id as u32;
        let high = (type_id >> 32) as u32;
        let encoded = (children.len() as u32).wrapping_mul(3) ^ low ^ high ^ CHILD_XOR_MASK;
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_41ff_u32.to_le_bytes());
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&encoded.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        for child in children {
            out.extend_from_slice(child);
        }
        out
    }

    fn v2_file(root: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        out.extend_from_slice(&2_u32.to_le_bytes());
        out.extend_from_slice(root);
        out
    }

    fn encode_string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    fn encode_list(out: &mut Vec<u8>, items: &[&str]) {
        out.extend_from_slice(&(items.len() as u32).to_le_bytes());
        for item in items {
            encode_string(out, item);
        }
    }

    fn phone_set_payload(
        name: &str,
        symbols: &[&str],
        categories: &[&str],
        unified: &[&str],
        classes: &[&str],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        encode_string(&mut out, name);
        encode_list(&mut out, symbols);
        encode_list(&mut out, categories);
        encode_list(&mut out, unified);
        encode_list(&mut out, classes);
        out
    }

    fn phone_set_node(name: &str) -> Vec<u8> {
        v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                name,
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
            ),
            &[],
        )
    }

    fn unified_node() -> Vec<u8> {
        v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "multiphone",
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
            ),
            &[],
        )
    }

    fn dense_payload(rows: u32, columns: u32, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&columns.to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn vector_payload(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn cmpu_node(lower: &[f32], upper: &[f32], range: &[f32]) -> Vec<u8> {
        v2_node(
            named_id("cmpu0"),
            &[],
            &[
                v2_node(named_id("prim1"), &vector_payload(lower), &[]),
                v2_node(named_id("prim1"), &vector_payload(upper), &[]),
                v2_node(named_id("prim1"), &vector_payload(range), &[]),
            ],
        )
    }

    fn embedding_node(rows: u32, columns: u32, fill: f32) -> Vec<u8> {
        let values = vec![fill; rows as usize * columns as usize];
        v2_node(
            named_id("modl4"),
            &[],
            &[v2_node(
                named_id("prim0"),
                &dense_payload(rows, columns, &values),
                &[],
            )],
        )
    }

    fn pitch_payload(minimum: u32, maximum: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&minimum.to_le_bytes());
        out.extend_from_slice(&maximum.to_le_bytes());
        out
    }

    /// Valid frontend: group[1 phone set], unified, two 1-channel normalizers.
    fn frontend_node() -> Vec<u8> {
        let group = v2_node(
            named_id("cmpg1"),
            &[],
            &[phone_set_node("japanese-romaji-phones")],
        );
        v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                group,
                unified_node(),
                cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        )
    }

    /// Zero 272 -> 1 dense network: every phoneme scores 0, so each interval
    /// splits its duration evenly across its phonemes after denormalization.
    fn zero_dense_network() -> Vec<u8> {
        v2_node(
            named_id("modl0"),
            &[],
            &[
                v2_node(named_id("prim0"), &dense_payload(1, 272, &[0.0; 272]), &[]),
                v2_node(named_id("prim1"), &vector_payload(&[0.0]), &[]),
            ],
        )
    }

    fn timing_reader(
        frontend: Vec<u8>,
        network: Vec<u8>,
        language: Vec<u8>,
        phoneme: Vec<u8>,
        position: Vec<u8>,
        voice: Vec<u8>,
    ) -> DnniReader {
        let mut payload = Vec::new();
        payload.extend_from_slice(&32_u32.to_le_bytes());
        payload.extend_from_slice(&16_u32.to_le_bytes());
        let root = v2_node(
            DURATION_TYPES[0],
            &payload,
            &[frontend, network, language, phoneme, position, voice],
        );
        DnniReader::from_bytes(v2_file(&root)).expect("synthetic timing blob must parse")
    }

    /// Six v2 blobs in root-child order: frontend, network, language
    /// embedding, phoneme embedding, position embedding, voice vector.
    type ModelParts = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

    fn valid_parts() -> ModelParts {
        let voice = v2_node(
            named_id("prim1"),
            &vector_payload(
                &(0..128)
                    .map(|index| index as f32 * 0.001)
                    .collect::<Vec<_>>(),
            ),
            &[],
        );
        (
            frontend_node(),
            zero_dense_network(),
            embedding_node(32, 1, 0.5),
            embedding_node(32, 3, 1.0),
            embedding_node(16, 8, 0.1),
            voice,
        )
    }

    fn valid_reader() -> DnniReader {
        let (frontend, network, language, phoneme, position, voice) = valid_parts();
        timing_reader(frontend, network, language, phoneme, position, voice)
    }

    fn loaded_model() -> PhonemeTiming {
        let reader = valid_reader();
        let mut model = PhonemeTiming::new();
        model.load(&reader, 0).expect("valid blob must load");
        model
    }

    fn syllable(
        language: &str,
        phonemes: &[&str],
        duration_seconds: f64,
        midi_pitch: i32,
    ) -> TimingSyllable {
        TimingSyllable {
            language: language.to_owned(),
            phonemes: phonemes.iter().map(|symbol| (*symbol).to_owned()).collect(),
            duration_seconds,
            midi_pitch,
            is_continuation: false,
        }
    }

    #[test]
    fn synthetic_model_predicts_even_splits() {
        let model = loaded_model();
        // "japanese" matches "japanese-romaji-phones" by substring.
        let syllables = vec![
            syllable("japanese", &["k", "a"], 0.5, 60),
            syllable("japanese", &["sil"], 0.5, 62),
        ];
        let mut output = Vec::new();
        model
            .predict(&syllables, &mut output)
            .expect("must predict");
        assert_eq!(output.len(), 3);
        assert_eq!(
            output
                .iter()
                .map(|duration| duration.symbol.clone())
                .collect::<Vec<_>>(),
            vec!["k", "a", "sil"]
        );
        assert_eq!(
            output
                .iter()
                .map(|duration| duration.syllable_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        assert_eq!(
            output
                .iter()
                .map(|duration| duration.timing_interval_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 1]
        );
        // The zero network scores every phoneme identically, so interval 0
        // (two phonemes) splits 0.5 s evenly and interval 1 keeps 0.5 s.
        // Identical inputs take identical f32 paths, so the pair is bitwise equal.
        assert_eq!(output[0].duration_seconds, output[1].duration_seconds);
        for duration in &output {
            assert!(duration.duration_seconds > 0.0);
        }
        let first: f64 = output[0].duration_seconds + output[1].duration_seconds;
        assert!((first - 0.5).abs() < 1e-6, "interval 0 sum {first}");
        assert!((output[2].duration_seconds - 0.5).abs() < 1e-9);
    }

    #[test]
    fn synthetic_model_quantizes_to_exact_frames() {
        let model = loaded_model();
        let syllables = vec![
            syllable("japanese", &["k", "a"], 0.5, 60),
            syllable("japanese", &["sil"], 0.5, 62),
        ];
        let mut durations = Vec::new();
        model
            .predict(&syllables, &mut durations)
            .expect("must predict");
        let mut timed = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut timed).expect("must quantize");
        assert_eq!(frame_counts(&timed), vec![50, 50, 100]);
    }

    #[test]
    fn unloaded_model_rejects_predict() {
        let model = PhonemeTiming::new();
        let mut output = vec![duration("japanese", "a", 0, 0, 1.0)];
        assert_eq!(
            model
                .predict(&[syllable("japanese", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: no duration model has been loaded."
        );
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn predict_rejects_empty_and_invalid_phrases() {
        let model = loaded_model();
        let mut output = Vec::new();
        assert_eq!(
            model.predict(&[], &mut output).unwrap_err().to_string(),
            "Phoneme timing: the phrase is empty or exceeds the tensor size limit."
        );
        // Syllables with no phonemes are rejected.
        assert_eq!(
            model
                .predict(&[syllable("japanese", &[], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: a syllable is empty or exceeds the tensor size limit."
        );
        // Bad durations and out-of-range pitches are rejected.
        for (duration_seconds, midi_pitch) in [
            (0.0, 60),
            (-0.5, 60),
            (f64::INFINITY, 60),
            (1e39, 60),
            (0.5, -1),
            (0.5, 128),
        ] {
            assert_eq!(
                model
                    .predict(
                        &[syllable("japanese", &["a"], duration_seconds, midi_pitch)],
                        &mut output
                    )
                    .unwrap_err()
                    .to_string(),
                "Phoneme timing: syllable duration must be finite and positive, and pitch must be a MIDI note.",
                "duration {duration_seconds} pitch {midi_pitch}"
            );
        }
        // A blank language is rejected before matching.
        assert_eq!(
            model
                .predict(&[syllable("", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: an explicit phoneme language is required."
        );
        assert!(output.is_empty());
    }

    #[test]
    fn predict_rejects_unknown_language_and_phoneme() {
        let model = loaded_model();
        let mut output = Vec::new();
        assert_eq!(
            model
                .predict(&[syllable("klingon", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: unknown phoneme language: klingon."
        );
        assert_eq!(
            model
                .predict(&[syllable("japanese", &["xyz"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: unknown phoneme 'xyz' for japanese-romaji-phones."
        );
        assert!(output.is_empty());
    }

    #[test]
    fn predict_rejects_ambiguous_language_match() {
        // Both phone-set names contain "lang", so the lookup is ambiguous.
        let group = v2_node(
            named_id("cmpg1"),
            &[],
            &[
                phone_set_node("lang-a-phones"),
                phone_set_node("lang-ab-phones"),
            ],
        );
        let frontend = v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                group,
                unified_node(),
                cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        );
        let (_, _, _, _, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            zero_dense_network(),
            embedding_node(32, 2, 0.0),
            embedding_node(32, 3, 0.0),
            position,
            voice,
        );
        let mut model = PhonemeTiming::new();
        model.load(&reader, 0).expect("ambiguous blob must load");
        let mut output = Vec::new();
        assert_eq!(
            model
                .predict(&[syllable("lang", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: the language matches more than one phone set."
        );
    }

    #[test]
    fn predict_rejects_wrong_shaped_network_output() {
        // Identity network passes 272 channels through: not one value per phoneme.
        let (frontend, _, language, phoneme, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            v2_node(named_id("moda5"), &[], &[]),
            language,
            phoneme,
            position,
            voice,
        );
        let mut model = PhonemeTiming::new();
        model.load(&reader, 0).expect("identity blob must load");
        let mut output = Vec::new();
        assert_eq!(
            model
                .predict(&[syllable("japanese", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: the duration network must return one value per phoneme."
        );
    }

    #[test]
    fn load_rejects_bad_root() {
        let mut tag = [0_u8; 8];
        tag[.."prim0".len()].copy_from_slice(b"prim0");
        let mut node = Vec::new();
        node.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        node.extend_from_slice(&tag);
        node.extend_from_slice(&0_u32.to_le_bytes());
        node.extend_from_slice(&0_u32.to_le_bytes());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&node);
        let reader = DnniReader::from_bytes(bytes).expect("v1 blob must parse");
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: the selected node is not the verified gen2a duration model."
        );
        assert_eq!(
            model.load(&reader, 99).unwrap_err().to_string(),
            "Phoneme timing: the selected node is not the verified gen2a duration model."
        );
    }

    #[test]
    fn load_rejects_bad_feature_layout() {
        let (frontend, network, _, _, _, _) = valid_parts();
        // Root with two children instead of six.
        let short = v2_node(
            DURATION_TYPES[3],
            &pitch_payload(32, 16),
            &[frontend, network],
        );
        let reader = DnniReader::from_bytes(v2_file(&short)).expect("must parse");
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: unsupported gen2a feature layout."
        );
        // Six children but the wrong layout payload.
        let (frontend, network, language, phoneme, position, voice) = valid_parts();
        let bad_payload = v2_node(
            DURATION_TYPES[3],
            &pitch_payload(31, 16),
            &[frontend, network, language, phoneme, position, voice],
        );
        let reader = DnniReader::from_bytes(v2_file(&bad_payload)).expect("must parse");
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: unsupported gen2a feature layout."
        );
    }

    #[test]
    fn load_rejects_bad_frontend() {
        let (_, network, language, phoneme, position, voice) = valid_parts();
        let mut payload = Vec::new();
        payload.extend_from_slice(&32_u32.to_le_bytes());
        payload.extend_from_slice(&16_u32.to_le_bytes());
        // Frontend of the wrong type.
        let wrong = v2_node(
            DURATION_TYPES[3],
            &payload,
            &[
                v2_node(named_id("prim0"), &dense_payload(1, 1, &[1.0]), &[]),
                network.clone(),
                language.clone(),
                phoneme.clone(),
                position.clone(),
                voice.clone(),
            ],
        );
        let reader = DnniReader::from_bytes(v2_file(&wrong)).expect("must parse");
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: unsupported duration feature frontend."
        );
        // Frontend with two children instead of four.
        let short = v2_node(
            DURATION_TYPES[3],
            &payload,
            &[
                v2_node(
                    FRONTEND_TYPES[7],
                    &pitch_payload(0, 127),
                    &[v2_node(named_id("cmpg1"), &[], &[phone_set_node("x")])],
                ),
                network,
                language,
                phoneme,
                position,
                voice,
            ],
        );
        let reader = DnniReader::from_bytes(v2_file(&short)).expect("must parse");
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: unsupported duration feature frontend."
        );
    }

    #[test]
    fn load_rejects_bad_pitch_range() {
        for (minimum, maximum) in [(90_u32, 60_u32), (0, 200), (60, 60)] {
            let group = v2_node(
                named_id("cmpg1"),
                &[],
                &[phone_set_node("japanese-romaji-phones")],
            );
            let frontend = v2_node(
                FRONTEND_TYPES[0],
                &pitch_payload(minimum, maximum),
                &[
                    group,
                    unified_node(),
                    cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                    cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
                ],
            );
            let (_, _, language, phoneme, position, voice) = valid_parts();
            let reader = timing_reader(
                frontend,
                zero_dense_network(),
                language,
                phoneme,
                position,
                voice,
            );
            let mut model = loaded_model();
            assert_eq!(
                model.load(&reader, 0).unwrap_err().to_string(),
                "Phoneme timing: invalid MIDI pitch normalization range.",
                "pitch [{minimum}, {maximum}]"
            );
        }
    }

    #[test]
    fn load_rejects_bad_phone_set_group() {
        // Group node of the wrong type.
        let frontend = v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                phone_set_node("japanese-romaji-phones"),
                unified_node(),
                cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        );
        let (_, _, language, phoneme, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            zero_dense_network(),
            language,
            phoneme,
            position,
            voice,
        );
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: the duration frontend requires a nonempty language phone-set group."
        );
        // Empty group.
        let frontend = v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                v2_node(named_id("cmpg1"), &[], &[]),
                unified_node(),
                cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        );
        let (_, _, language, phoneme, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            zero_dense_network(),
            language,
            phoneme,
            position,
            voice,
        );
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: the duration frontend requires a nonempty language phone-set group."
        );
    }

    #[test]
    fn load_rejects_unmapped_unified_symbol() {
        // The language set maps "k" to "zzz", outside the unified inventory.
        let rogue = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "japanese-romaji-phones",
                &["k", "a"],
                &["consonant", "vowel"],
                &["zzz", "a"],
                &["consonant", "vowel"],
            ),
            &[],
        );
        let frontend = v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                v2_node(named_id("cmpg1"), &[], &[rogue]),
                unified_node(),
                cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        );
        let (_, _, language, phoneme, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            zero_dense_network(),
            language,
            phoneme,
            position,
            voice,
        );
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: a language phoneme maps to an unknown unified symbol."
        );
    }

    #[test]
    fn load_rejects_multi_channel_normalizers() {
        let frontend = v2_node(
            FRONTEND_TYPES[0],
            &pitch_payload(0, 127),
            &[
                v2_node(
                    named_id("cmpg1"),
                    &[],
                    &[phone_set_node("japanese-romaji-phones")],
                ),
                unified_node(),
                cmpu_node(&[-5.0, 0.0], &[5.0, 1.0], &[0.0, 1.0]),
                cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]),
            ],
        );
        let (_, _, language, phoneme, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            zero_dense_network(),
            language,
            phoneme,
            position,
            voice,
        );
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: duration normalizers must each contain one channel."
        );
    }

    #[test]
    fn load_rejects_bad_embeddings() {
        // Raw matrix instead of a modl4 wrapper.
        let (frontend, network, _, _, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            network,
            v2_node(named_id("prim0"), &dense_payload(32, 1, &[0.0; 32]), &[]),
            embedding_node(32, 3, 0.0),
            position,
            voice,
        );
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: an embedding must be an empty modl4 node with one matrix."
        );
        // Wrapper present but the dimensions are wrong (32x2 for one language).
        let (frontend, network, _, _, position, voice) = valid_parts();
        let reader = timing_reader(
            frontend,
            network,
            embedding_node(32, 2, 0.0),
            embedding_node(32, 3, 0.0),
            position,
            voice,
        );
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: an embedding matrix has unsupported dimensions."
        );
    }

    #[test]
    fn load_rejects_short_voice_embedding() {
        let (frontend, network, language, phoneme, position, _) = valid_parts();
        let reader = timing_reader(
            frontend,
            network,
            language,
            phoneme,
            position,
            v2_node(named_id("prim1"), &vector_payload(&[0.0; 4]), &[]),
        );
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: the duration voice embedding must contain 128 channels."
        );
    }
}
