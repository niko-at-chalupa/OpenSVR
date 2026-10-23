//! Frontend feature encoder for the neural pitch model (`gen5`).
//!
//! Ports OpenSV's `src/synthesis/PitchFeatures.{h,cpp}` (namespace
//! `sv::synthesis`): projects discrete notes and phoneme durations onto the
//! acoustic frame grid (typically 5 ms), extracts the 10-channel continuous
//! pitch/timing context, z-score normalizes it, and denormalizes predicted
//! pitch back to absolute MIDI semitones. The z-score divides by the stored
//! per-channel standard deviation, matching OpenSV+ `PitchFeatures.cpp`; the
//! fork multiplied by it, which pushed every feature far outside the trained
//! range and pinned the network output to the mean pitch.
//!
//! Score feature layout (`featureChannels = 10`):
//! `[0]` rest/breath flag, `[1]` continuation (slur) flag, `[2]` MIDI pitch,
//! `[3]` `log(min(duration + 0.001, 5))`, `[4]` rap flag, `[5..10]` reserved
//! tone/accent channels (validated but left at zero, exactly as in C++).

use opensvr_dnni::{DnniReader, DnniTensor};
use opensvr_g2p::{PhoneSet, read_phone_set};

use crate::{Result, invalid};

/// Note feature channels per score row.
const FEATURE_CHANNELS: usize = 10;
/// Maximum notes, phonemes or frames in one prediction interval.
const MAXIMUM_FRAMES: usize = 4 * 1024 * 1024;

/// Verified `gen5` frontend node type hashes.
const FRONTEND_TYPES: [u64; 24] = [
    0x3269_15e5_8397_9c55,
    0x41cc_3de4_18da_4ec3,
    0xeb26_790a_fe5f_ca94,
    0x4295_0b54_a390_f7a7,
    0x9d63_afbd_1522_17d3,
    0x1312_9719_647e_2ca4,
    0x8614_9d20_3f18_6f29,
    0x1051_cecb_9d99_60d1,
    0x5dc5_51ef_55f9_d8b3,
    0x7430_fc6f_1646_a1dc,
    0xf35e_015c_07d8_d996,
    0xc6df_474d_8865_8ae2,
    0x3269_12e5_8397_973c,
    0x41cc_3ee4_18da_5076,
    0xeb26_7c0a_fe5f_cfad,
    0x4295_0c54_a390_f95a,
    0x9d63_b0bd_1522_1986,
    0x1312_9a19_647e_31bd,
    0x8614_9a20_3f18_6a10,
    0x1051_cbcb_9d99_5bb8,
    0x5dc5_52ef_55f9_da66,
    0x7430_ff6f_1646_a6f5,
    0xf35e_005c_07d8_d7e3,
    0xc6df_464d_8865_892f,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Pitch features: {}", reason.into()))
}

/// Input musical note for the pitch prediction pipeline.
///
/// Mirrors `PitchNote` (`PitchFeatures.h`): the note syllable plus
/// breath/silence/continuation/rap flags, a tone register offset in
/// `0..=5`, and the vibrato modulation scaler.
/// `struct_excessive_bools`: the four mode flags mirror the C++ `PitchNote`
/// bit-for-bit; grouping them would diverge from the ported layout.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct PitchNote {
    /// Note syllable with timing, pitch and lyric phonemes.
    pub syllable: opensvr_g2p::TimingSyllable,
    /// True for an aspiration / breath intake.
    pub is_breath: bool,
    /// True for a silence / rest separator.
    pub is_silence: bool,
    /// True when continuing the pitch/phoneme of the previous note.
    pub is_continuation: bool,
    /// True for rap intonation rather than melodic singing.
    pub is_rap: bool,
    /// Tone modifier / pitch register offset (`0..=5`).
    pub tone: i32,
    /// Scaling factor applied to vibrato speed and depth.
    pub vibrato_modulation: f32,
}

impl Default for PitchNote {
    fn default() -> Self {
        Self {
            syllable: opensvr_g2p::TimingSyllable {
                language: String::new(),
                phonemes: Vec::new(),
                duration_seconds: 0.0,
                midi_pitch: 60,
                is_continuation: false,
            },
            is_breath: false,
            is_silence: false,
            is_continuation: false,
            is_rap: false,
            tone: 0,
            vibrato_modulation: 1.0,
        }
    }
}

/// Encoded note/phoneme features on the frame grid.
#[derive(Debug, Clone, Default)]
pub struct PitchFeatureOutput {
    /// Normalized score rows, one per note (`[notes, 10]`).
    pub note_features: DnniTensor,
    /// 5 ms frames spanned by each note.
    pub note_frame_counts: Vec<usize>,
    /// 5 ms frames spanned by each phoneme.
    pub phoneme_frame_counts: Vec<usize>,
    /// Phonetic category id of each phoneme.
    pub phoneme_categories: Vec<usize>,
    /// Language id of each phoneme.
    pub phoneme_languages: Vec<usize>,
    /// 1 for vowel/nucleus phonemes, 0 for consonants and silence.
    pub phoneme_vowels: Vec<u8>,
}

/// Frontend encoder: phone tables plus z-score statistics.
#[derive(Debug, Default)]
pub struct PitchFeatures {
    phone_sets: Vec<PhoneSet>,
    unified_phone_set: PhoneSet,
    means: Vec<f32>,
    scales: Vec<f32>,
    frame_interval_seconds: f32,
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn find_language(phone_sets: &[PhoneSet], language: &str) -> Result<usize> {
    if language.is_empty() {
        return Err(failure("an explicit phoneme language is required."));
    }
    let mut found = phone_sets.len();
    for (index, phone_set) in phone_sets.iter().enumerate() {
        if phone_set.name.contains(language) {
            if found != phone_sets.len() {
                return Err(failure("the language matches multiple phone sets."));
            }
            found = index;
        }
    }
    if found == phone_sets.len() {
        return Err(failure(format!("unknown phoneme language: {language}.")));
    }
    Ok(found)
}

impl PitchFeatures {
    /// Creates an unloaded frontend; [`encode`](Self::encode) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the frontend descriptor and normalization statistics.
    ///
    /// The node carries an 8-byte payload (`[1, frameIntervalSeconds]`) and
    /// three children: the language phone-set group, the unified phone set,
    /// and a two-vector `cmpu1` normalizer with ten channels each. Everything
    /// is staged into a candidate first: a failed load preserves the current
    /// frontend.
    pub fn load(&mut self, reader: &DnniReader, node_index: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes
            .get(node_index)
            .filter(|node| FRONTEND_TYPES.contains(&node.type_id))
        else {
            return Err(failure(
                "the selected node is not the verified gen5 frontend.",
            ));
        };
        let payload = reader.payload(node_index);
        if payload.len() != 8 || node.children.len() != 3 || read_word(payload, 0) != 1 {
            return Err(failure("unsupported gen5 frontend configuration."));
        }
        let frame_interval_seconds = f32::from_bits(read_word(payload, 4));
        if !frame_interval_seconds.is_finite() || frame_interval_seconds <= 0.0 {
            return Err(failure("the frame interval must be finite and positive."));
        }
        let children = node.children.clone();

        let mut candidate = Self {
            frame_interval_seconds,
            ..Self::default()
        };
        let group = &nodes[children[0]];
        if group.name != "cmpg1"
            || !reader.payload(children[0]).is_empty()
            || group.children.is_empty()
        {
            return Err(failure("the language phone-set group is invalid."));
        }
        candidate.phone_sets.reserve(group.children.len());
        for &child in &group.children.clone() {
            candidate.phone_sets.push(read_phone_set(reader, child)?);
        }
        candidate.unified_phone_set = read_phone_set(reader, children[1])?;
        for phone_set in &candidate.phone_sets {
            for category in &phone_set.categories {
                if !candidate.unified_phone_set.classes.contains(category) {
                    return Err(failure(
                        "a phoneme category is absent from the unified class table.",
                    ));
                }
            }
            for symbol in &phone_set.unified_symbols {
                if !symbol.is_empty() && !candidate.unified_phone_set.symbols.contains(symbol) {
                    return Err(failure(
                        "a phoneme mapping is absent from the unified symbol table.",
                    ));
                }
            }
        }

        let normalization = &nodes[children[2]];
        if normalization.name != "cmpu1"
            || !reader.payload(children[2]).is_empty()
            || normalization.children.len() != 2
        {
            return Err(failure(
                "the frontend requires a two-vector cmpu1 normalization.",
            ));
        }
        candidate.means = reader.read_float_vector(normalization.children[0])?;
        candidate.scales = reader.read_float_vector(normalization.children[1])?;
        if candidate.means.len() != FEATURE_CHANNELS || candidate.scales.len() != FEATURE_CHANNELS {
            return Err(failure("score normalization must contain ten channels."));
        }
        *self = candidate;
        Ok(())
    }

    /// Acoustic frame interval in seconds (typically 0.005).
    pub fn frame_interval_seconds(&self) -> f32 {
        self.frame_interval_seconds
    }

    /// Phonetic category count from the unified class table.
    pub fn phoneme_category_count(&self) -> usize {
        self.unified_phone_set.classes.len()
    }

    /// Language count from the phone-set group.
    pub fn language_count(&self) -> usize {
        self.phone_sets.len()
    }

    /// Encodes notes and phonemes into normalized feature frames.
    ///
    /// Note durations project with mixed precision (`round(noteSeconds *
    /// (1/interval))` in `f32`, `round(endSeconds * (1/interval))` in `f64`);
    /// phoneme boundaries use `f32` throughout. `output` is only assigned on
    /// success.
    #[allow(clippy::too_many_lines)]
    pub fn encode(
        &self,
        notes: &[PitchNote],
        phonemes: &[opensvr_g2p::PhonemeDuration],
        output: &mut PitchFeatureOutput,
    ) -> Result<()> {
        if self.means.len() != FEATURE_CHANNELS || self.frame_interval_seconds <= 0.0 {
            return Err(failure("no frontend has been loaded."));
        }
        if notes.is_empty()
            || phonemes.is_empty()
            || notes.len() > MAXIMUM_FRAMES
            || phonemes.len() > MAXIMUM_FRAMES
        {
            return Err(failure(
                "the note or phoneme sequence is empty or exceeds the size limit.",
            ));
        }
        let mut candidate = PitchFeatureOutput {
            note_features: DnniTensor {
                frames: notes.len(),
                channels: FEATURE_CHANNELS,
                values: vec![0.0; notes.len() * FEATURE_CHANNELS],
            },
            ..PitchFeatureOutput::default()
        };
        candidate.note_frame_counts.reserve(notes.len());
        candidate.phoneme_frame_counts.reserve(phonemes.len());
        candidate.phoneme_categories.reserve(phonemes.len());
        candidate.phoneme_languages.reserve(phonemes.len());
        candidate.phoneme_vowels.reserve(phonemes.len());

        // 1. Note durations onto frame counts plus raw 10-channel features.
        let mut note_seconds = 0.0_f32;
        let inverse_interval = 1.0_f32 / self.frame_interval_seconds;
        let inverse_note_interval = 1.0_f64 / f64::from(self.frame_interval_seconds);
        for note in notes {
            let seconds = note.syllable.duration_seconds;
            if !seconds.is_finite() || seconds <= 0.0 || note.tone < 0 || note.tone > 5 {
                return Err(failure(
                    "notes require positive finite durations and a tone in the range 0 to 5.",
                ));
            }
            let end_seconds = f64::from(note_seconds) + seconds;
            let start_frame = (note_seconds * inverse_interval).round();
            let end_frame = (end_seconds * inverse_note_interval).round();
            if !end_frame.is_finite()
                || end_frame > MAXIMUM_FRAMES as f64
                || end_frame < f64::from(start_frame)
            {
                return Err(failure("note frame boundaries exceed the supported range."));
            }
            candidate
                .note_frame_counts
                .push((end_frame - f64::from(start_frame)) as usize);
            note_seconds = end_seconds as f32;

            let base = candidate.note_frame_counts.len() - 1;
            let row = &mut candidate.note_features.values
                [base * FEATURE_CHANNELS..(base + 1) * FEATURE_CHANNELS];
            row[0] = f32::from(note.is_silence || note.is_breath);
            row[1] = f32::from(note.is_continuation);
            row[2] = note.syllable.midi_pitch as f32;
            row[3] = (seconds + 0.001).min(5.0).ln() as f32;
            row[4] = f32::from(note.is_rap);
        }

        // 2. Phoneme durations onto frame counts plus category/language ids.
        let mut phoneme_seconds = 0.0_f32;
        for phoneme in phonemes {
            if !phoneme.duration_seconds.is_finite()
                || phoneme.duration_seconds <= 0.0
                || phoneme.syllable_index >= notes.len()
            {
                return Err(failure(
                    "phonemes require positive finite durations and a valid source note.",
                ));
            }
            let seconds = phoneme.duration_seconds as f32;
            let end_seconds = phoneme_seconds + seconds;
            let start_frame = (phoneme_seconds / self.frame_interval_seconds).round();
            let end_frame = (end_seconds / self.frame_interval_seconds).round();
            if !end_frame.is_finite()
                || end_frame > MAXIMUM_FRAMES as f32
                || end_frame < start_frame
            {
                return Err(failure(
                    "phoneme frame boundaries exceed the supported range.",
                ));
            }
            candidate
                .phoneme_frame_counts
                .push((end_frame - start_frame) as usize);
            phoneme_seconds = end_seconds;

            let language = find_language(&self.phone_sets, &phoneme.language)?;
            let phone_set = &self.phone_sets[language];
            let Some(source_index) = phone_set
                .symbols
                .iter()
                .position(|symbol| symbol == &phoneme.symbol)
            else {
                return Err(failure(format!(
                    "unknown phoneme '{}' for {}.",
                    phoneme.symbol, phone_set.name
                )));
            };
            let category = &phone_set.categories[source_index];
            let category_index = self
                .unified_phone_set
                .classes
                .iter()
                .position(|class| class == category)
                .unwrap_or(self.unified_phone_set.classes.len());
            candidate.phoneme_categories.push(category_index);
            candidate.phoneme_languages.push(language);
            candidate
                .phoneme_vowels
                .push(u8::from(category == "vowel" || category == "diphthong"));
        }

        // 3. Z-score normalization: `(feature - mean) / scale`, with the
        //    training standard deviation floored at 1e-5 (OpenSV+
        //    `PitchFeatures.cpp`). The fork multiplies by the deviation
        //    instead, which pushes every feature far outside the trained
        //    range and pins the network output to the mean pitch.
        for index in 0..notes.len() {
            let (row, (means, scales)) = (
                &mut candidate.note_features.values
                    [index * FEATURE_CHANNELS..(index + 1) * FEATURE_CHANNELS],
                (&self.means, &self.scales),
            );
            for ((slot, mean), scale) in row.iter_mut().zip(means).zip(scales) {
                *slot = (*slot - *mean) / scale.max(0.000_01);
            }
        }
        *output = candidate;
        Ok(())
    }

    /// Converts normalized pitch predictions to MIDI semitones (`norm * scale + mean`).
    pub fn denormalize_pitch(
        &self,
        normalized_pitch: &[f32],
        midi_pitch: &mut Vec<f32>,
    ) -> Result<()> {
        if self.means.len() != FEATURE_CHANNELS
            || self.scales.len() != FEATURE_CHANNELS
            || self.scales[2] == 0.0
        {
            return Err(failure("no frontend loaded or zero pitch scale."));
        }
        midi_pitch.resize(normalized_pitch.len(), 0.0);
        let pitch_scale = self.scales[2];
        let pitch_mean = self.means[2];
        for (frame, value) in normalized_pitch.iter().enumerate() {
            midi_pitch[frame] = value * pitch_scale + pitch_mean;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_g2p::{PhonemeDuration, TimingSyllable};

    const FNV_PRIME_64: u64 = 0x0100_0000_01b3;

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
        let encoded = (children.len() as u32).wrapping_mul(3) ^ low ^ high ^ 0xac5e_7bd5;
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

    fn phone_set_node(name: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        encode_string(&mut payload, name);
        encode_list(&mut payload, &["k", "a"]);
        encode_list(&mut payload, &["consonant", "vowel"]);
        encode_list(&mut payload, &["k", "a"]);
        encode_list(&mut payload, &["consonant", "vowel"]);
        v2_node(named_id("_psv2"), &payload, &[])
    }

    fn unified_node() -> Vec<u8> {
        let mut payload = Vec::new();
        encode_string(&mut payload, "unified");
        encode_list(&mut payload, &["k", "a"]);
        encode_list(&mut payload, &["consonant", "vowel"]);
        encode_list(&mut payload, &["k", "a"]);
        encode_list(&mut payload, &["consonant", "vowel"]);
        v2_node(named_id("_psv2"), &payload, &[])
    }

    fn vector_payload(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn frontend_reader() -> DnniReader {
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set_node("japanese")]);
        let norm = v2_node(
            named_id("cmpu1"),
            &[],
            &[
                v2_node(named_id("prim1"), &vector_payload(&[0.0; 10]), &[]),
                v2_node(named_id("prim1"), &vector_payload(&[1.0; 10]), &[]),
            ],
        );
        let mut payload = Vec::new();
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&0.005_f32.to_le_bytes());
        let root = v2_node(FRONTEND_TYPES[0], &payload, &[group, unified_node(), norm]);
        DnniReader::from_bytes(v2_file(&root)).expect("synthetic frontend must parse")
    }

    fn loaded() -> PitchFeatures {
        let reader = frontend_reader();
        let mut features = PitchFeatures::new();
        features.load(&reader, 0).expect("valid frontend must load");
        features
    }

    fn note(midi: i32, seconds: f64) -> PitchNote {
        PitchNote {
            syllable: TimingSyllable {
                language: "japanese".to_owned(),
                phonemes: vec!["a".to_owned()],
                duration_seconds: seconds,
                midi_pitch: midi,
                is_continuation: false,
            },
            ..PitchNote::default()
        }
    }

    fn phoneme(symbol: &str, seconds: f64, syllable: usize) -> PhonemeDuration {
        PhonemeDuration {
            language: "japanese".to_owned(),
            symbol: symbol.to_owned(),
            syllable_index: syllable,
            timing_interval_index: 0,
            duration_seconds: seconds,
        }
    }

    #[test]
    fn loader_rejects_an_unknown_frontend() {
        let reader = DnniReader::from_bytes(v2_file(&v2_node(named_id("modl0"), &[], &[])))
            .expect("blob must parse");
        let mut features = PitchFeatures::new();
        assert_eq!(
            features.load(&reader, 0).unwrap_err().to_string(),
            "Pitch features: the selected node is not the verified gen5 frontend."
        );
    }

    #[test]
    fn unloaded_frontend_rejects_encode() {
        let features = PitchFeatures::new();
        let mut output = PitchFeatureOutput::default();
        assert_eq!(
            features
                .encode(&[note(60, 0.4)], &[phoneme("a", 0.4, 0)], &mut output)
                .unwrap_err()
                .to_string(),
            "Pitch features: no frontend has been loaded."
        );
    }

    #[test]
    fn encode_projects_frames_and_raw_channels() {
        // Identity normalization: raw channels survive untouched.
        let features = loaded();
        let notes = vec![note(60, 0.4), note(62, 0.6)];
        let phonemes = vec![
            phoneme("k", 0.1, 0),
            phoneme("a", 0.3, 0),
            phoneme("a", 0.6, 1),
        ];
        let mut output = PitchFeatureOutput::default();
        features
            .encode(&notes, &phonemes, &mut output)
            .expect("must encode");
        assert_eq!(output.note_frame_counts, vec![80, 120]);
        assert_eq!(output.phoneme_frame_counts, vec![20, 60, 120]);
        assert_eq!(output.phoneme_categories, vec![0, 1, 1]);
        assert_eq!(output.phoneme_languages, vec![0, 0, 0]);
        assert_eq!(output.phoneme_vowels, vec![0, 1, 1]);
        // Note 0: rest 0, continuation 0, pitch 60, log duration, rap 0.
        let row = &output.note_features.values[..10];
        assert_eq!(&row[..3], &[0.0, 0.0, 60.0]);
        assert!(
            (row[3] - (0.401_f64).ln() as f32).abs() < 1e-7,
            "got {}",
            row[3]
        );
        assert_eq!(&row[4..], &[0.0; 6]);
    }

    #[test]
    fn denormalize_inverts_channel_two() {
        let features = loaded();
        let mut midi = Vec::new();
        features
            .denormalize_pitch(&[60.0, 62.5], &mut midi)
            .expect("must denormalize");
        assert_eq!(midi, vec![60.0, 62.5]);
    }

    #[test]
    fn encode_validates_notes_and_phonemes() {
        let features = loaded();
        let mut output = PitchFeatureOutput::default();
        assert_eq!(
            features
                .encode(&[], &[phoneme("a", 0.4, 0)], &mut output)
                .unwrap_err()
                .to_string(),
            "Pitch features: the note or phoneme sequence is empty or exceeds the size limit."
        );
        let mut bad_tone = note(60, 0.4);
        bad_tone.tone = 6;
        assert_eq!(
            features
                .encode(&[bad_tone], &[phoneme("a", 0.4, 0)], &mut output)
                .unwrap_err()
                .to_string(),
            "Pitch features: notes require positive finite durations and a tone in the range 0 to 5."
        );
        assert_eq!(
            features
                .encode(&[note(60, 0.4)], &[phoneme("zz", 0.4, 0)], &mut output)
                .unwrap_err()
                .to_string(),
            "Pitch features: unknown phoneme 'zz' for japanese."
        );
    }
}
