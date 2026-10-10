//! Neural phoneme duration prediction (`PhonemeTiming`).
//!
//! Ports OpenSV's `src/synthesis/PhonemeTiming.{h,cpp}`: syllables are mapped
//! to timing intervals through the voice's per-language [`PhoneSet`]s,
//! consonant onsets are shifted onto the beat ([`align_intervals`]), a
//! 272-channel feature tensor is built per phoneme ([`encode_scalar`] RBF
//! expansions plus learned embeddings), one full [`DnniInference`] pass
//! predicts log-durations, and the results are denormalized and distributed
//! across each interval ([`PhonemeTiming::predict`]). [`quantize_phoneme_durations`]
//! then snaps the continuous durations onto acoustic frames.

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniInference, DnniMatrix, DnniReader, DnniTensor};
use opensvr_g2p::{PhonemeDuration, PhoneSet, TimingSyllable, read_phone_set};

use crate::normalizer::FeatureNormalizer;
use crate::{Result, invalid};

// Total input feature dimension of the duration network:
// 32 (phoneme) + 32 (language) + 16 (forward pos) + 16 (reverse pos)
// + 32 (encoded duration) + 16 (encoded pitch) + 128 (speaker/voice)
// = 272 channels.
const FEATURE_CHANNELS: usize = 272;
const MAXIMUM_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;

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
/// 0 has no predecessor and is left alone. Step 2 (multi-nuclei split):
/// intervals with more than one nucleus split into equal-duration pieces cut
/// right before each subsequent nucleus, carrying the MIDI pitch over.
fn align_intervals(mut intervals: Vec<TimingInterval>) -> Vec<TimingInterval> {
    for index in 1..intervals.len() {
        let nucleus = intervals[index]
            .phones
            .iter()
            .position(|phone| phone.is_nucleus);
        if let Some(nucleus) = nucleus {
            let moved: Vec<TimingPhone> =
                intervals[index].phones.drain(..nucleus).collect();
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

/// Encodes one continuous scalar into a multi-channel Gaussian RBF expansion.
fn encode_scalar(value: f32, destination: &mut [f32]) {
    let channels = destination.len();
    for (channel, slot) in destination.iter_mut().enumerate() {
        let center = channel as f32 / (channels - 1) as f32;
        let delta = value - center;
        // Standard Gaussian RBF expansion.
        *slot = (-delta * delta * channels as f32).exp();
    }
}

/// Copies one embedding row into the feature buffer.
fn copy_embedding(matrix: &DnniMatrix, index: usize, destination: &mut [f32]) {
    let begin = index * matrix.columns as usize;
    destination.copy_from_slice(&matrix.values[begin..begin + matrix.columns as usize]);
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
/// Every phoneme receives at least one frame; cumulative boundaries accumulate
/// from zero so rounding never drifts. `output` is cleared first and only
/// assigned on success.
pub fn quantize_phoneme_durations(
    durations: &[PhonemeDuration],
    frame_interval_seconds: f32,
    output: &mut Vec<TimedPhoneme>,
) -> Result<()> {
    if frame_interval_seconds <= 0.0 || !frame_interval_seconds.is_finite() {
        return Err(failure("invalid acoustic frame interval."));
    }
    let mut quantized = Vec::with_capacity(durations.len());
    let mut cumulative_seconds = 0.0_f64;
    let mut previous_frame = 0_usize;
    for duration in durations {
        cumulative_seconds += duration.duration_seconds;
        // Round the cumulative time to the nearest acoustic frame boundary,
        // keeping at least one frame per phoneme.
        let target_frame = (previous_frame as f64 + 1.0)
            .max((cumulative_seconds / f64::from(frame_interval_seconds)).round())
            as usize;
        quantized.push(TimedPhoneme {
            language: duration.language.clone(),
            symbol: duration.symbol.clone(),
            frame_count: target_frame - previous_frame,
        });
        previous_frame = target_frame;
    }
    *output = quantized;
    Ok(())
}

fn read_word(bytes: &[u8], offset: usize) -> i32 {
    // Little-endian pitch bound; the wrapping bit pattern matches C++'s
    // `static_cast<int>(readWord(...))`. Callers guarantee `offset + 4` fits.
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Reads an embedding matrix and verifies its declared dimensions.
fn read_embedding(
    reader: &DnniReader,
    node_index: usize,
    rows: usize,
    columns: usize,
    matrix: &mut DnniMatrix,
) -> Result<()> {
    *matrix = reader.read_float_matrix(node_index)?;
    if matrix.rows as usize != rows || matrix.columns as usize != columns {
        return Err(failure(
            "embedding matrix dimensions do not match the declared channels.",
        ));
    }
    Ok(())
}

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

impl Default for PhonemeTiming {
    fn default() -> Self {
        Self::new()
    }
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

    /// Loads the duration model weights and embeddings from a DNNI file.
    ///
    /// The root must be a duration model container with a frontend feature
    /// extractor child and an inference network child. On failure the model
    /// keeps its previous state.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(root) = nodes.get(root_node) else {
            return Err(failure("root node index out of bounds."));
        };
        if !DURATION_TYPES.contains(&root.type_id) {
            return Err(failure("unsupported duration model root type."));
        }
        // Child 0: frontend feature descriptors, embeddings, normalizers.
        // Child 1: inference network.
        if root.children.len() < 2 {
            return Err(failure("duration model missing required child components."));
        }
        let frontend_node = root.children[0];
        let network_node = root.children[1];
        let Some(frontend_type) = nodes.get(frontend_node).map(|node| node.type_id) else {
            return Err(failure("duration model missing required child components."));
        };
        if !FRONTEND_TYPES.contains(&frontend_type) {
            return Err(failure("unsupported frontend feature descriptor type."));
        }
        let Some(frontend) = nodes.get(frontend_node) else {
            return Err(failure("duration model missing required child components."));
        };
        if frontend.children.len() < 8 {
            return Err(failure(
                "frontend descriptor missing required embedding tables.",
            ));
        }
        let frontend_children = frontend.children.clone();

        // Read the phone sets and the unified phone set.
        let mut phone_sets = Vec::new();
        let Some(group) = nodes.get(frontend_children[0]) else {
            return Err(failure(
                "frontend descriptor missing required embedding tables.",
            ));
        };
        for &child in &group.children.clone() {
            phone_sets.push(read_phone_set(reader, child)?);
        }
        let unified_phone_set = read_phone_set(reader, frontend_children[1])?;

        // Load the embeddings: language (32), phoneme (32), position (16).
        let mut language_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(
            reader,
            frontend_children[2],
            phone_sets.len(),
            32,
            &mut language_embedding,
        )?;
        let mut phoneme_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(
            reader,
            frontend_children[3],
            unified_phone_set.symbols.len(),
            32,
            &mut phoneme_embedding,
        )?;
        let mut position_embedding = DnniMatrix {
            rows: 0,
            columns: 0,
            values: Vec::new(),
        };
        read_embedding(reader, frontend_children[4], 8, 16, &mut position_embedding)?;

        // Speaker / voice embedding vector (128 floats).
        let voice_embedding = reader.read_float_vector(frontend_children[5])?;
        if voice_embedding.len() != 128 {
            return Err(failure("voice embedding must be 128 elements."));
        }

        // Normalizers.
        let mut input_normalization = FeatureNormalizer::default();
        input_normalization.load(reader, frontend_children[6])?;
        let mut output_normalization = FeatureNormalizer::default();
        output_normalization.load(reader, frontend_children[7])?;

        // Pitch bounds from the frontend payload.
        let payload = reader.payload(frontend_node);
        if payload.len() < 8 {
            return Err(failure("frontend payload truncated."));
        }
        let minimum_pitch = read_word(payload, 0);
        let maximum_pitch = read_word(payload, 4);

        // Neural network.
        let mut network = DnniInference::new();
        network.load(reader, network_node)?;

        self.phone_sets = phone_sets;
        self.unified_phone_set = unified_phone_set;
        self.input_normalization = input_normalization;
        self.output_normalization = output_normalization;
        self.language_embedding = language_embedding;
        self.phoneme_embedding = phoneme_embedding;
        self.position_embedding = position_embedding;
        self.voice_embedding = voice_embedding;
        self.network = network;
        self.minimum_pitch = minimum_pitch;
        self.maximum_pitch = maximum_pitch;
        Ok(())
    }

    /// Predicts the duration in seconds of each phoneme in a syllable phrase.
    ///
    /// `output` is cleared first (including for empty input) and only
    /// assigned on success.
    pub fn predict(
        &self,
        syllables: &[TimingSyllable],
        output: &mut Vec<PhonemeDuration>,
    ) -> Result<()> {
        if syllables.is_empty() {
            output.clear();
            return Ok(());
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

    /// Maps syllables to aligned timing intervals (steps 1-2).
    ///
    /// Returns the intervals plus the total phoneme count.
    fn build_intervals(
        &self,
        syllables: &[TimingSyllable],
    ) -> Result<(Vec<TimingInterval>, usize)> {
        // 1. Map input syllables and phonemes into timing intervals.
        let mut intervals = Vec::with_capacity(syllables.len());
        let mut phoneme_count = 0_usize;
        for (syllable_index, syllable) in syllables.iter().enumerate() {
            phoneme_count += syllable.phonemes.len();

            // Match the syllable language to the known phone sets.
            let mut language_index = self.phone_sets.len();
            for (index, phone_set) in self.phone_sets.iter().enumerate() {
                if phone_set.name == syllable.language {
                    if language_index != self.phone_sets.len() {
                        return Err(failure(
                            "the language matches more than one phone set.",
                        ));
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
                let Some(source_index) =
                    phone_set.symbols.iter().position(|candidate| candidate == symbol)
                else {
                    return Err(failure(format!(
                        "unknown phoneme '{symbol}' for {}.",
                        phone_set.name
                    )));
                };
                let unified = &phone_set.unified_symbols[source_index];
                let Some(target_index) = self
                    .unified_phone_set
                    .symbols
                    .iter()
                    .position(|candidate| candidate == unified)
                else {
                    return Err(failure(format!(
                        "unknown unified phoneme '{unified}' for {}.",
                        phone_set.name
                    )));
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

        // 2. Singing timing alignment (leading consonant shift, vowel splits).
        Ok((align_intervals(intervals), phoneme_count))
    }

    /// Builds the 272-channel feature tensor plus duration shells (steps 3-4).
    fn render_features(
        &self,
        syllables: &[TimingSyllable],
        intervals: &[TimingInterval],
        phoneme_count: usize,
    ) -> Result<(DnniTensor, Vec<PhonemeDuration>)> {
        // 3. Normalized log-duration input features.
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
        self.input_normalization.normalize(
            &raw_duration,
            &mut normalized_duration,
            true,
        )?;

        // 4. Construct the 272-channel input feature tensor.
        let total = phoneme_count
            .checked_mul(FEATURE_CHANNELS)
            .filter(|&elements| elements <= MAXIMUM_TENSOR_ELEMENTS)
            .ok_or_else(|| failure("feature tensor exceeds the memory limit."))?;
        let mut features = DnniTensor {
            frames: phoneme_count,
            channels: FEATURE_CHANNELS,
            values: vec![0.0; total],
        };
        let mut durations = Vec::with_capacity(phoneme_count);
        let mut frame = 0_usize;
        for (interval_index, interval) in intervals.iter().enumerate() {
            let pitch = (interval.midi_pitch.wrapping_sub(self.minimum_pitch)) as f32
                / (self.maximum_pitch.wrapping_sub(self.minimum_pitch)) as f32;
            for (position, phone) in interval.phones.iter().enumerate() {
                let syllable = &syllables[phone.syllable_index];
                let destination =
                    &mut features.values[frame * FEATURE_CHANNELS..(frame + 1) * FEATURE_CHANNELS];
                // Feature layout:
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
                // [96..128): RBF expansion of the normalized interval duration.
                encode_scalar(
                    normalized_duration.values[interval_index],
                    &mut destination[96..128],
                );
                // [128..144): RBF expansion of the normalized pitch.
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

/// Distributes each interval duration across its phonemes (step 7).
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

    fn interval(phones: Vec<TimingPhone>, duration_seconds: f64, midi_pitch: i32) -> TimingInterval {
        TimingInterval {
            phones,
            duration_seconds,
            midi_pitch,
        }
    }

    fn symbols(intervals: &[TimingInterval]) -> Vec<Vec<bool>> {
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
        // Durations and pitches stay with their own intervals.
        assert_eq!(aligned[0].duration_seconds, 0.4);
        assert_eq!(aligned[1].duration_seconds, 0.6);
        assert_eq!(aligned[0].midi_pitch, 60);
        assert_eq!(aligned[1].midi_pitch, 62);
        // The moved phones keep their source syllable / phoneme indices.
        assert_eq!(aligned[0].phones[1].syllable_index, 1);
        assert_eq!(aligned[0].phones[1].phoneme_index, 0);
    }

    #[test]
    fn first_interval_has_no_predecessor_to_shift_into() {
        let intervals = vec![interval(
            vec![
                phone(0, 0, 0, 1, false),
                phone(0, 1, 0, 3, true),
            ],
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
        assert_eq!(symbols(&aligned), vec![vec![true], vec![false]]);
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
        assert_eq!(symbols(&aligned), vec![vec![true, false], vec![true], vec![true]]);
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
    fn encode_scalar_matches_hand_computed_rbf_values() {
        // Centers for 4 channels are 0, 1/3, 2/3, 1 (f32); value 0 peaks at 1.
        // Expectations keep 7 significant digits (what f32 resolves) and the
        // comparison allows 1e-6 relative slack for the last-ulp rounding.
        let mut out = [0.0_f32; 4];
        encode_scalar(0.0, &mut out);
        assert_close(&out, &[1.0, 0.641_180_4, 0.169_013_3, 0.018_315_64], 1e-6);
        // Value 0.5 sits between the middle centers; the RBF is symmetric up
        // to f32 center rounding (channels 1 and 2 differ in the last ulp).
        encode_scalar(0.5, &mut out);
        assert_close(&out, &[0.367_879_5, 0.894_839_3, 0.894_839_3, 0.367_879_5], 1e-6);
        let mut pair = [0.0_f32; 2];
        encode_scalar(1.0, &mut pair);
        assert_close(&pair, &[0.135_335_28, 1.0], 1e-6);
        encode_scalar(0.25, &mut pair);
        assert_close(&pair, &[0.882_496_9, 0.324_652_5], 1e-6);
    }

    #[test]
    fn encode_scalar_into_32_and_16_channels_is_finite_and_peaked() {
        let mut wide = [0.0_f32; 32];
        encode_scalar(0.43, &mut wide);
        assert!(wide.iter().all(|value| value.is_finite()));
        let peak = wide
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("nonempty")
            .0;
        // 0.43 * 31 = 13.3, so channel 13 must be the peak (off-center, < 1).
        assert_eq!(peak, 13);
        assert!(wide[peak] < 1.0 && wide[peak] > 0.99);
        // Exactly on a center the RBF peaks at exactly 1.
        let mut centered = [0.0_f32; 32];
        encode_scalar(13.0 / 31.0, &mut centered);
        assert_eq!(centered[13], 1.0);
        let mut narrow = [0.0_f32; 16];
        encode_scalar(0.0, &mut narrow);
        assert_eq!(narrow[0], 1.0);
        assert!(narrow[1..].iter().all(|&value| value < 1.0));
    }

    #[test]
    fn copy_embedding_copies_the_indexed_row() {
        let matrix = DnniMatrix {
            rows: 2,
            columns: 3,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let mut destination = [0.0_f32; 3];
        copy_embedding(&matrix, 1, &mut destination);
        assert_eq!(destination, [4.0, 5.0, 6.0]);
    }

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
    fn quantize_gives_every_phoneme_at_least_one_frame() {
        let durations = vec![
            duration("japanese", "k", 0, 0, 1e-9),
            duration("japanese", "a", 0, 0, 1e-9),
        ];
        let mut output = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut output).expect("must quantize");
        assert_eq!(
            output.iter().map(|phoneme| phoneme.frame_count).collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn quantize_accumulates_from_zero_without_drift() {
        // 3 x 0.006 s at 5 ms frames: cumulative boundaries fall at frames
        // round(1.2) = 1, round(2.4) = 2, round(3.6) = 4.
        let durations = vec![
            duration("english", "s", 0, 0, 0.006),
            duration("english", "ih", 0, 0, 0.006),
            duration("english", "l", 0, 0, 0.006),
        ];
        let mut output = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut output).expect("must quantize");
        assert_eq!(
            output.iter().map(|phoneme| phoneme.frame_count).collect::<Vec<_>>(),
            vec![1, 1, 2]
        );
        // The frame sum covers the whole phrase: 4 frames = 0.02 s >= 0.018 s.
        assert_eq!(
            output.iter().map(|phoneme| phoneme.frame_count).sum::<usize>(),
            4
        );
    }

    #[test]
    fn quantize_rejects_non_positive_or_non_finite_frame_intervals() {
        let durations = vec![duration("japanese", "a", 0, 0, 0.25)];
        let mut output = vec![TimedPhoneme {
            language: "sentinel".to_owned(),
            symbol: "x".to_owned(),
            frame_count: 7,
        }];
        for interval in [0.0, -0.005, f32::INFINITY, f32::NAN] {
            assert_eq!(
                quantize_phoneme_durations(&durations, interval, &mut output)
                    .unwrap_err()
                    .to_string(),
                "Phoneme timing: invalid acoustic frame interval."
            );
            // The output is untouched on failure.
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].symbol, "x");
        }
    }

    // --- Synthetic DNNI blob builders (v2 object trees) ---

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

    /// Minimal valid timing-model frontend: one phone set plus embeddings.
    fn frontend_node() -> Vec<u8> {
        let phone_set = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "japanese",
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
            ),
            &[],
        );
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set]);
        let unified = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "unified",
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
                &["k", "a", "sil"],
                &["consonant", "vowel", "silence"],
            ),
            &[],
        );
        let language = v2_node(named_id("prim0"), &dense_payload(1, 32, &[0.5; 32]), &[]);
        let phoneme = v2_node(
            named_id("prim0"),
            &dense_payload(
                3,
                32,
                &[
                    vec![0.0; 32],
                    vec![1.0; 32],
                    vec![2.0; 32],
                ]
                .concat(),
            ),
            &[],
        );
        let position = v2_node(
            named_id("prim0"),
            &dense_payload(
                8,
                16,
                &(0..8)
                    .flat_map(|row| vec![row as f32 * 0.1; 16])
                    .collect::<Vec<_>>(),
            ),
            &[],
        );
        let voice = v2_node(
            named_id("prim1"),
            &vector_payload(&(0..128).map(|index| index as f32 * 0.001).collect::<Vec<_>>()),
            &[],
        );
        let input_norm = cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]);
        let output_norm = cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]);
        let mut payload = Vec::new();
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&127_u32.to_le_bytes());
        v2_node(
            FRONTEND_TYPES[0],
            &payload,
            &[
                group, unified, language, phoneme, position, voice, input_norm, output_norm,
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

    fn timing_reader(frontend: Vec<u8>, network: Vec<u8>) -> DnniReader {
        let root = v2_node(DURATION_TYPES[0], &[], &[frontend, network]);
        DnniReader::from_bytes(v2_file(&root)).expect("synthetic timing blob must parse")
    }

    fn valid_reader() -> DnniReader {
        timing_reader(frontend_node(), zero_dense_network())
    }

    fn loaded_model() -> PhonemeTiming {
        let reader = valid_reader();
        let mut model = PhonemeTiming::new();
        model.load(&reader, 0).expect("valid blob must load");
        model
    }

    fn syllable(language: &str, phonemes: &[&str], duration_seconds: f64, midi_pitch: i32) -> TimingSyllable {
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
        let syllables = vec![
            syllable("japanese", &["k", "a"], 0.5, 60),
            syllable("japanese", &["sil"], 0.5, 62),
        ];
        let mut output = Vec::new();
        model.predict(&syllables, &mut output).expect("must predict");
        assert_eq!(output.len(), 3);
        assert_eq!(
            output.iter().map(|duration| duration.symbol.clone()).collect::<Vec<_>>(),
            vec!["k", "a", "sil"]
        );
        assert_eq!(
            output.iter().map(|duration| duration.syllable_index).collect::<Vec<_>>(),
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
        model.predict(&syllables, &mut durations).expect("must predict");
        let mut timed = Vec::new();
        quantize_phoneme_durations(&durations, 0.005, &mut timed).expect("must quantize");
        assert_eq!(
            timed.iter().map(|phoneme| phoneme.frame_count).collect::<Vec<_>>(),
            vec![50, 50, 100]
        );
    }

    #[test]
    fn empty_phrase_predicts_nothing() {
        let model = loaded_model();
        let mut output = vec![duration("japanese", "a", 0, 0, 1.0)];
        model.predict(&[], &mut output).expect("empty must succeed");
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
            "Phoneme timing: unknown phoneme 'xyz' for japanese."
        );
        assert!(output.is_empty());
    }

    #[test]
    fn predict_rejects_ambiguous_language_match() {
        // Two phone sets share one name, so the lookup is ambiguous.
        let phone_set = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "japanese",
                &["a"],
                &["vowel"],
                &["a"],
                &["vowel"],
            ),
            &[],
        );
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set.clone(), phone_set]);
        let reader = {
            let unified = v2_node(
                named_id("_psv2"),
                &phone_set_payload("unified", &["a"], &["vowel"], &["a"], &["vowel"]),
                &[],
            );
            let language = v2_node(named_id("prim0"), &dense_payload(2, 32, &[0.0; 64]), &[]);
            let phoneme =
                v2_node(named_id("prim0"), &dense_payload(1, 32, &[0.0; 32]), &[]);
            let position = v2_node(
                named_id("prim0"),
                &dense_payload(8, 16, &[0.0; 128]),
                &[],
            );
            let voice = v2_node(named_id("prim1"), &vector_payload(&[0.0; 128]), &[]);
            let input_norm = cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]);
            let output_norm = cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]);
            let mut payload = Vec::new();
            payload.extend_from_slice(&0_u32.to_le_bytes());
            payload.extend_from_slice(&127_u32.to_le_bytes());
            let frontend = v2_node(
                FRONTEND_TYPES[0],
                &payload,
                &[
                    group, unified, language, phoneme, position, voice, input_norm,
                    output_norm,
                ],
            );
            timing_reader(frontend, zero_dense_network())
        };
        let mut model = PhonemeTiming::new();
        model.load(&reader, 0).expect("ambiguous blob must load");
        let mut output = Vec::new();
        assert_eq!(
            model
                .predict(&[syllable("japanese", &["a"], 0.5, 60)], &mut output)
                .unwrap_err()
                .to_string(),
            "Phoneme timing: the language matches more than one phone set."
        );
    }

    #[test]
    fn predict_rejects_wrong_shaped_network_output() {
        // Identity network passes 272 channels through: not one value per phoneme.
        let reader = timing_reader(
            frontend_node(),
            v2_node(named_id("moda5"), &[], &[]),
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
    fn load_rejects_bad_root_type() {
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
            "Phoneme timing: unsupported duration model root type."
        );
        assert_eq!(
            model.load(&reader, 99).unwrap_err().to_string(),
            "Phoneme timing: root node index out of bounds."
        );
    }

    #[test]
    fn load_rejects_missing_children_and_tables() {
        let mut model = loaded_model();
        // Root with a single child: no network component.
        let lonely = v2_node(DURATION_TYPES[3], &[], &[frontend_node()]);
        let reader = DnniReader::from_bytes(v2_file(&lonely)).expect("must parse");
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: duration model missing required child components."
        );
        // Frontend of the wrong type.
        let wrong = v2_node(
            DURATION_TYPES[3],
            &[],
            &[
                v2_node(named_id("prim0"), &dense_payload(1, 1, &[1.0]), &[]),
                zero_dense_network(),
            ],
        );
        let reader = DnniReader::from_bytes(v2_file(&wrong)).expect("must parse");
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: unsupported frontend feature descriptor type."
        );
        // Frontend with too few children.
        let short = v2_node(
            DURATION_TYPES[3],
            &[],
            &[
                v2_node(FRONTEND_TYPES[7], &[], &[frontend_node()]),
                zero_dense_network(),
            ],
        );
        let reader = DnniReader::from_bytes(v2_file(&short)).expect("must parse");
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: frontend descriptor missing required embedding tables."
        );
    }

    #[test]
    fn load_rejects_bad_voice_embedding_length() {
        // Frontend identical to the valid one except the voice vector is short.
        let phone_set = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "japanese",
                &["a"],
                &["vowel"],
                &["a"],
                &["vowel"],
            ),
            &[],
        );
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set]);
        let unified = v2_node(
            named_id("_psv2"),
            &phone_set_payload("unified", &["a"], &["vowel"], &["a"], &["vowel"]),
            &[],
        );
        let language = v2_node(named_id("prim0"), &dense_payload(1, 32, &[0.0; 32]), &[]);
        let phoneme = v2_node(named_id("prim0"), &dense_payload(1, 32, &[0.0; 32]), &[]);
        let position =
            v2_node(named_id("prim0"), &dense_payload(8, 16, &[0.0; 128]), &[]);
        let voice = v2_node(named_id("prim1"), &vector_payload(&[0.0; 4]), &[]);
        let input_norm = cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]);
        let output_norm = cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]);
        let mut payload = Vec::new();
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&127_u32.to_le_bytes());
        let frontend = v2_node(
            FRONTEND_TYPES[2],
            &payload,
            &[
                group, unified, language, phoneme, position, voice, input_norm,
                output_norm,
            ],
        );
        let reader = timing_reader(frontend, zero_dense_network());
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: voice embedding must be 128 elements."
        );
    }

    #[test]
    fn load_rejects_embedding_shape_mismatch() {
        // Language embedding claims 2 rows for a single phone set.
        let phone_set = v2_node(
            named_id("_psv2"),
            &phone_set_payload(
                "japanese",
                &["a"],
                &["vowel"],
                &["a"],
                &["vowel"],
            ),
            &[],
        );
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set]);
        let unified = v2_node(
            named_id("_psv2"),
            &phone_set_payload("unified", &["a"], &["vowel"], &["a"], &["vowel"]),
            &[],
        );
        let language = v2_node(named_id("prim0"), &dense_payload(2, 32, &[0.0; 64]), &[]);
        let phoneme = v2_node(named_id("prim0"), &dense_payload(1, 32, &[0.0; 32]), &[]);
        let position =
            v2_node(named_id("prim0"), &dense_payload(8, 16, &[0.0; 128]), &[]);
        let voice = v2_node(named_id("prim1"), &vector_payload(&[0.0; 128]), &[]);
        let input_norm = cmpu_node(&[-5.0], &[5.0], &[0.0, 1.0]);
        let output_norm = cmpu_node(&[-10.0], &[10.0], &[0.0, 1.0]);
        let mut payload = Vec::new();
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&127_u32.to_le_bytes());
        let frontend = v2_node(
            FRONTEND_TYPES[2],
            &payload,
            &[
                group, unified, language, phoneme, position, voice, input_norm,
                output_norm,
            ],
        );
        let reader = timing_reader(frontend, zero_dense_network());
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: embedding matrix dimensions do not match the declared channels."
        );
    }

    #[test]
    fn load_rejects_truncated_frontend_payload() {
        // Rebuild the valid blob, then shrink the frontend payload from 8 to
        // 3 bytes in place (payload bytes drain with it, so the tree still
        // parses but the pitch bounds no longer fit).
        let bare = {
            let valid = v2_file(&v2_node(
                DURATION_TYPES[0],
                &[],
                &[frontend_node(), zero_dense_network()],
            ));
            // Frontend payload starts at the frontend header + 20; find it by
            // re-reading the node table instead of hardcoding offsets.
            let parsed = DnniReader::from_bytes(valid).expect("must parse");
            let front = parsed.nodes()[0].children[0];
            let offset = parsed.nodes()[front].payload_offset;
            let mut bytes = {
                let root = v2_node(DURATION_TYPES[0], &[], &[frontend_node(), zero_dense_network()]);
                v2_file(&root)
            };
            // Zero the 8 payload bytes in place.
            for byte in &mut bytes[offset..offset + 8] {
                *byte = 0;
            }
            // Now shrink the payload size field (header + 16) from 8 to 3 and
            // drop 5 bytes so the file still parses but the payload is short.
            let header = parsed.nodes()[front].offset;
            bytes[header + 16] = 3;
            bytes.drain(offset + 3..offset + 8);
            bytes
        };
        let reader = DnniReader::from_bytes(bare).expect("short payload blob must parse");
        let mut model = loaded_model();
        assert_eq!(
            model.load(&reader, 0).unwrap_err().to_string(),
            "Phoneme timing: frontend payload truncated."
        );
    }
}
