//! Acoustic frontend: phoneme/duration features, log-F0 and output decoding.
//!
//! Ports OpenSV's `src/synthesis/AcousticFeatures.{h,cpp}` (namespace
//! `sv::synthesis`). The `_ftmfv2`/`_ftmfv3` node carries a 32-byte payload
//! (`[phonemeChannels, pitchChannels, acousticChannels, 2, 66, 0,
//! frameIntervalSeconds, 0]`) and six children: the language phone-set
//! group, the unified phone set, the `_dctov0` matrices (validated 64x64,
//! otherwise unused), and the phoneme/pitch/acoustic `cmpu0` normalizers.
//!
//! Error strings reproduce the C++ `juce::Result` failure text verbatim, so
//! golden tests can compare messages.

use opensvr_dnni::{DnniReader, DnniTensor};
use opensvr_g2p::{PhoneSet, read_phone_set};

use crate::normalizer::FeatureNormalizer;
use crate::timing::TimedPhoneme;
use crate::{Result, invalid};

/// Maximum tensor elements in one call (128 Mi, as in the C++ frontend).
const MAXIMUM_TENSOR_ELEMENTS: usize = 128 * 1024 * 1024;

/// Frontend dimensions loaded from the `_ftmfv2`/`_ftmfv3` payload.
#[derive(Debug, Clone, Default)]
pub struct AcousticFeatureConfig {
    /// One-hot phoneme width plus the trailing log-duration channel.
    pub phoneme_channels: usize,
    /// Always 1 (a single normalized log-F0 channel).
    pub pitch_channels: usize,
    /// Always 71 (voiced flag, spectral and auxiliary channels).
    pub acoustic_channels: usize,
    /// Acoustic frame grid in seconds (the real voice uses 5 ms).
    pub frame_interval_seconds: f32,
}

/// Acoustic frontend: phone tables plus the three feature normalizers.
#[derive(Debug, Default)]
pub struct AcousticFeatures {
    config: AcousticFeatureConfig,
    phone_sets: Vec<PhoneSet>,
    unified_phone_set: PhoneSet,
    phoneme_normalization: FeatureNormalizer,
    pitch_normalization: FeatureNormalizer,
    acoustic_normalization: FeatureNormalizer,
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

impl AcousticFeatures {
    /// Creates an unloaded frontend; every method fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the `_ftmfv2`/`_ftmfv3` configuration at `node_index`.
    ///
    /// Everything is staged into a candidate first: a failed load preserves
    /// the current frontend.
    pub fn load(&mut self, reader: &DnniReader, node_index: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes.get(node_index).filter(|node| {
            (node.name == "_ftmfv2" || node.name == "_ftmfv3")
                && reader.payload(node_index).len() == 32
                && node.children.len() == 6
        }) else {
            return Err(invalid(
                "Acoustic features require a 32-byte _ftmfv2 or _ftmfv3 with six children.",
            ));
        };
        let children = node.children.clone();
        let payload = reader.payload(node_index);
        let mut candidate = Self::default();
        candidate.config.phoneme_channels = read_word(payload, 0) as usize;
        candidate.config.pitch_channels = read_word(payload, 4) as usize;
        candidate.config.acoustic_channels = read_word(payload, 8) as usize;
        candidate.config.frame_interval_seconds = f32::from_bits(read_word(payload, 24));
        if candidate.config.phoneme_channels < 2
            || candidate.config.pitch_channels != 1
            || candidate.config.acoustic_channels != 71
            || read_word(payload, 12) != 2
            || read_word(payload, 16) != 66
            || read_word(payload, 20) != 0
            || read_word(payload, 28) != 0
            || !candidate.config.frame_interval_seconds.is_finite()
            || candidate.config.frame_interval_seconds <= 0.0
        {
            return Err(invalid(
                "Unsupported acoustic feature configuration; expected phoneme/duration features, one F0 channel and 71 output channels without DCT postprocessing.",
            ));
        }

        let group = &nodes[children[0]];
        if group.name != "cmpg1"
            || !reader.payload(children[0]).is_empty()
            || group.children.is_empty()
        {
            return Err(invalid(
                "Acoustic features have no valid language phone-set group.",
            ));
        }
        candidate.phone_sets.reserve(group.children.len());
        for &child in &group.children.clone() {
            candidate.phone_sets.push(read_phone_set(reader, child)?);
        }
        candidate.unified_phone_set = read_phone_set(reader, children[1])?;
        if candidate.unified_phone_set.symbols.len() + 1 != candidate.config.phoneme_channels {
            return Err(invalid(
                "Unified acoustic phoneme count does not match the feature width.",
            ));
        }
        for phone_set in &candidate.phone_sets {
            for unified in &phone_set.unified_symbols[..phone_set.symbols.len()] {
                if !candidate.unified_phone_set.symbols.contains(unified) {
                    return Err(invalid(
                        "A language phoneme maps to a missing unified acoustic symbol.",
                    ));
                }
            }
        }

        let dct = &nodes[children[2]];
        if dct.name != "_dctov0"
            || !reader.payload(children[2]).is_empty()
            || dct.children.len() != 2
        {
            return Err(invalid(
                "Acoustic features have an invalid DCT configuration.",
            ));
        }
        for &child in &dct.children.clone() {
            let matrix = reader.read_float_matrix(child)?;
            if matrix.rows != 64 || matrix.columns != 64 {
                return Err(invalid(
                    "Acoustic DCT matrices must have 64 rows and columns.",
                ));
            }
        }

        candidate.phoneme_normalization.load(reader, children[3])?;
        candidate.pitch_normalization.load(reader, children[4])?;
        candidate.acoustic_normalization.load(reader, children[5])?;
        if candidate.phoneme_normalization.channel_count() != candidate.config.phoneme_channels
            || candidate.pitch_normalization.channel_count() != candidate.config.pitch_channels
            || candidate.acoustic_normalization.channel_count()
                != candidate.config.acoustic_channels
        {
            return Err(invalid(
                "Acoustic normalization dimensions do not match the feature configuration.",
            ));
        }
        *self = candidate;
        Ok(())
    }

    /// Returns the loaded frontend dimensions (zeros when unloaded).
    pub fn config(&self) -> &AcousticFeatureConfig {
        &self.config
    }

    /// Acoustic frame interval in seconds (0.0 when unloaded).
    pub fn frame_interval_seconds(&self) -> f32 {
        self.config.frame_interval_seconds
    }

    /// Number of language phone sets in the frontend.
    pub fn language_count(&self) -> usize {
        self.phone_sets.len()
    }

    /// Resolves a language to its phone-set index by nonempty substring.
    ///
    /// Mirrors the C++ `findLanguageIndex`: exactly one match is required.
    /// `AcousticModel::makeContext` reuses this so both paths agree.
    pub(crate) fn find_language_index(&self, language: &str) -> Result<usize> {
        if self.config.phoneme_channels == 0 {
            return Err(invalid("Acoustic features have not been loaded."));
        }
        if language.is_empty() {
            return Err(invalid(
                "An explicit acoustic phoneme language is required.",
            ));
        }
        let mut found = self.phone_sets.len();
        for (index, phone_set) in self.phone_sets.iter().enumerate() {
            if phone_set.name.contains(language) {
                if found != self.phone_sets.len() {
                    return Err(invalid(
                        "Acoustic phoneme language matches more than one phone set.",
                    ));
                }
                found = index;
            }
        }
        if found == self.phone_sets.len() {
            return Err(invalid(format!(
                "Unknown acoustic phoneme language: {language}"
            )));
        }
        Ok(found)
    }

    /// Encodes timed phonemes into normalized one-hot plus log-duration rows.
    ///
    /// Each row holds a one-hot unified symbol plus
    /// `log(frameCount * interval + 0.05)` in the trailing channel, mapped
    /// through the phoneme normalizer (clamped, the C++ default). `output`
    /// is only assigned on success.
    pub fn encode_phonemes(
        &self,
        phonemes: &[TimedPhoneme],
        output: &mut DnniTensor,
    ) -> Result<()> {
        if self.config.phoneme_channels == 0 {
            return Err(invalid("Acoustic features have not been loaded."));
        }
        if phonemes.is_empty()
            || phonemes.len() > MAXIMUM_TENSOR_ELEMENTS / self.config.phoneme_channels
        {
            return Err(invalid(
                "Acoustic phoneme sequence is empty or exceeds the supported tensor size.",
            ));
        }
        let channels = self.config.phoneme_channels;
        let mut raw = DnniTensor {
            frames: phonemes.len(),
            channels,
            values: vec![0.0; phonemes.len() * channels],
        };
        for (index, phoneme) in phonemes.iter().enumerate() {
            if phoneme.frame_count == 0 || phoneme.frame_count > i32::MAX as usize {
                return Err(invalid(
                    "Acoustic phoneme frame counts must be positive int32 values.",
                ));
            }
            let language_index = self.find_language_index(&phoneme.language)?;
            let phone_set = &self.phone_sets[language_index];
            let Some(source_index) = phone_set
                .symbols
                .iter()
                .position(|symbol| symbol == &phoneme.symbol)
            else {
                return Err(invalid(format!(
                    "Unknown acoustic phoneme '{}' for {}.",
                    phoneme.symbol, phone_set.name
                )));
            };
            let unified = &phone_set.unified_symbols[source_index];
            let Some(target_index) = self
                .unified_phone_set
                .symbols
                .iter()
                .position(|symbol| symbol == unified)
            else {
                return Err(invalid(
                    "A language phoneme maps to a missing unified acoustic symbol.",
                ));
            };
            raw.values[index * channels + target_index] = 1.0;
            let duration_seconds = phoneme.frame_count as f32 * self.config.frame_interval_seconds;
            raw.values[(index + 1) * channels - 1] = (duration_seconds + 0.05).ln();
        }
        self.phoneme_normalization.normalize(&raw, output, true)
    }

    /// Normalizes per-frame log-F0 through the pitch normalizer (clamped).
    ///
    /// `output` is only assigned on success.
    pub fn normalize_log_f0(&self, log_f0: &[f32], output: &mut DnniTensor) -> Result<()> {
        if log_f0.is_empty() || log_f0.len() > MAXIMUM_TENSOR_ELEMENTS {
            return Err(invalid(
                "Acoustic log-F0 sequence is empty or exceeds the supported tensor size.",
            ));
        }
        let input = DnniTensor {
            frames: log_f0.len(),
            channels: 1,
            values: log_f0.to_vec(),
        };
        self.pitch_normalization.normalize(&input, output, true)
    }

    /// Denormalizes 71-channel network output into vocoder-ready frames.
    ///
    /// The voiced flag is quantized (`> 0.5`), the energy channel clamped to
    /// `[0.01, 3.0]`, the last five auxiliary channels to `[0, 1]`, and
    /// `exp(logF0)` is inserted at channel 1, widening 71 channels to 72.
    /// `output` is only assigned on success.
    pub fn decode_acoustic_71(
        &self,
        normalized: &DnniTensor,
        log_f0: &[f32],
        output: &mut DnniTensor,
    ) -> Result<()> {
        if self.config.acoustic_channels == 0 {
            return Err(invalid("Acoustic features have not been loaded."));
        }
        if normalized.frames != log_f0.len()
            || normalized.frames > MAXIMUM_TENSOR_ELEMENTS / (self.config.acoustic_channels + 1)
        {
            return Err(invalid(
                "Acoustic output and log-F0 frame counts do not match or exceed the supported tensor size.",
            ));
        }
        let mut decoded = DnniTensor::default();
        self.acoustic_normalization
            .denormalize(normalized, &mut decoded)?;
        let channels = decoded.channels;
        let mut values = vec![0.0; decoded.frames * (channels + 1)];
        for frame in 0..decoded.frames {
            let pitch = log_f0[frame];
            let frequency = pitch.exp();
            if !pitch.is_finite() || !frequency.is_finite() || frequency <= 0.0 {
                return Err(invalid(
                    "Acoustic log-F0 cannot be represented as a finite positive frequency.",
                ));
            }
            let source = &mut decoded.values[frame * channels..(frame + 1) * channels];
            source[0] = if source[0] > 0.5 { 1.0 } else { 0.0 };
            source[1] = source[1].clamp(0.01, 3.0);
            for slot in &mut source[66..71] {
                *slot = slot.clamp(0.0, 1.0);
            }
            let destination = &mut values[frame * (channels + 1)..(frame + 1) * (channels + 1)];
            destination[0] = source[0];
            destination[1] = frequency;
            destination[2..].copy_from_slice(&source[1..]);
        }
        *output = DnniTensor {
            frames: decoded.frames,
            channels: channels + 1,
            values,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_dnni::DnniReader;

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

    fn dense_payload(rows: u32, columns: u32, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&columns.to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Two phoneme symbols plus the trailing duration channel: 3 wide.
    /// Unified set holds the same two symbols, so `2 + 1 == 3`.
    fn frontend_node(payload_words: [u32; 8]) -> Vec<u8> {
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set_node("japanese-test")]);
        let dct = v2_node(
            named_id("_dctov0"),
            &[],
            &[
                v2_node(
                    named_id("prim0"),
                    &dense_payload(64, 64, &vec![0.5; 64 * 64]),
                    &[],
                ),
                v2_node(
                    named_id("prim0"),
                    &dense_payload(64, 64, &vec![-0.25; 64 * 64]),
                    &[],
                ),
            ],
        );
        let mut payload = Vec::new();
        for word in payload_words {
            payload.extend_from_slice(&word.to_le_bytes());
        }
        v2_node(
            named_id("_ftmfv2"),
            &payload,
            &[
                group,
                phone_set_node("multiphone"),
                dct,
                cmpu_node(&[0.0, 0.0, -4.0], &[1.0, 1.0, 4.0], &[0.0, 1.0]),
                cmpu_node(&[-8.0], &[8.0], &[0.0, 1.0]),
                cmpu_node(&vec![0.0; 71], &vec![1.0; 71], &[0.0, 1.0]),
            ],
        )
    }

    fn valid_payload() -> [u32; 8] {
        [3, 1, 71, 2, 66, 0, 0.005_f32.to_bits(), 0]
    }

    fn valid_reader() -> DnniReader {
        DnniReader::from_bytes(v2_file(&frontend_node(valid_payload())))
            .expect("synthetic frontend must parse")
    }

    fn loaded() -> AcousticFeatures {
        let reader = valid_reader();
        let mut features = AcousticFeatures::new();
        features.load(&reader, 0).expect("valid frontend must load");
        assert_eq!(features.config().phoneme_channels, 3);
        assert_eq!(features.frame_interval_seconds(), 0.005);
        features
    }

    fn timed(language: &str, symbol: &str, frame_count: usize) -> TimedPhoneme {
        TimedPhoneme {
            language: language.to_owned(),
            symbol: symbol.to_owned(),
            frame_count,
        }
    }

    #[test]
    fn encode_phonemes_builds_one_hot_plus_log_duration_rows() {
        let features = loaded();
        let mut output = DnniTensor::default();
        // 3 frames at 5 ms: duration 0.015 s, log(0.015 + 0.05) = log(0.065).
        features
            .encode_phonemes(&[timed("japanese", "a", 3)], &mut output)
            .expect("must encode");
        assert_eq!((output.frames, output.channels), (1, 3));
        // One-hot "a" sits at unified index 1; the normalizer maps
        // [0, 1] -> [0, 1] on channels 0..2 and [-4, 4] -> [0, 1] on the
        // duration channel: (ln(0.065) + 4) / 8.
        let expected_duration = (0.065_f32.ln() + 4.0) / 8.0;
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1], 1.0);
        assert!((output.values[2] - expected_duration).abs() < 1e-6);
    }

    #[test]
    fn encode_phonemes_rejects_zero_frame_counts() {
        let features = loaded();
        let mut output = DnniTensor::default();
        assert_eq!(
            features
                .encode_phonemes(&[timed("japanese", "a", 0)], &mut output)
                .unwrap_err()
                .to_string(),
            "Acoustic phoneme frame counts must be positive int32 values."
        );
    }

    #[test]
    fn encode_phonemes_rejects_unknown_symbols_and_languages() {
        let features = loaded();
        let mut output = DnniTensor::default();
        assert_eq!(
            features
                .encode_phonemes(&[timed("japanese", "zzz", 2)], &mut output)
                .unwrap_err()
                .to_string(),
            "Unknown acoustic phoneme 'zzz' for japanese-test."
        );
        assert_eq!(
            features
                .encode_phonemes(&[timed("klingon", "a", 2)], &mut output)
                .unwrap_err()
                .to_string(),
            "Unknown acoustic phoneme language: klingon"
        );
        assert_eq!(
            features
                .encode_phonemes(&[timed("", "a", 2)], &mut output)
                .unwrap_err()
                .to_string(),
            "An explicit acoustic phoneme language is required."
        );
    }

    #[test]
    fn normalize_log_f0_maps_through_the_pitch_normalizer() {
        let features = loaded();
        let mut output = DnniTensor::default();
        features
            .normalize_log_f0(&[0.0, 8.0, -8.0], &mut output)
            .expect("must normalize");
        // [-8, 8] -> [0, 1]: 0 maps to 0.5, ends clamp to 0 and 1.
        assert_eq!(output.values, vec![0.5, 1.0, 0.0]);
    }

    #[test]
    fn decode_acoustic_71_quantizes_clamps_and_inserts_f0() {
        let features = loaded();
        // 71 channels of 0.5 denormalize to 0.5 (identity [0, 1] bounds).
        let normalized = DnniTensor {
            frames: 1,
            channels: 71,
            values: vec![0.5; 71],
        };
        let mut output = DnniTensor::default();
        features
            .decode_acoustic_71(&normalized, &[0.0], &mut output)
            .expect("must decode");
        assert_eq!((output.frames, output.channels), (1, 72));
        // Voiced flag 0.5 is not > 0.5, so unvoiced; exp(0) = 1 Hz.
        assert_eq!(output.values[0], 0.0);
        assert_eq!(output.values[1], 1.0);
        // Energy 0.5 survives the [0.01, 3.0] clamp; the tail copies through.
        assert_eq!(output.values[2], 0.5);
        assert_eq!(output.values[71], 0.5);
    }

    #[test]
    fn decode_acoustic_71_quantizes_the_voiced_flag_up() {
        let features = loaded();
        let mut values = vec![0.5; 71];
        values[0] = 0.9;
        values[1] = 99.0;
        values[66] = -3.0;
        values[70] = 42.0;
        let normalized = DnniTensor {
            frames: 1,
            channels: 71,
            values,
        };
        let mut output = DnniTensor::default();
        features
            .decode_acoustic_71(&normalized, &[1.0], &mut output)
            .expect("must decode");
        assert_eq!(output.values[0], 1.0);
        assert_eq!(output.values[1], 1.0_f32.exp());
        assert_eq!(output.values[2], 3.0);
        assert_eq!(output.values[67], 0.0);
        assert_eq!(output.values[71], 1.0);
    }

    #[test]
    fn decode_acoustic_71_rejects_non_finite_f0() {
        let features = loaded();
        let normalized = DnniTensor {
            frames: 1,
            channels: 71,
            values: vec![0.5; 71],
        };
        let mut output = DnniTensor::default();
        for pitch in [f32::INFINITY, 100.0] {
            // exp(100) overflows to infinity in f32.
            assert_eq!(
                features
                    .decode_acoustic_71(&normalized, &[pitch], &mut output)
                    .unwrap_err()
                    .to_string(),
                "Acoustic log-F0 cannot be represented as a finite positive frequency."
            );
        }
    }

    #[test]
    fn loader_rejects_a_bad_feature_width() {
        let mut payload = valid_payload();
        payload[2] = 70;
        let reader =
            DnniReader::from_bytes(v2_file(&frontend_node(payload))).expect("blob must parse");
        let mut features = AcousticFeatures::new();
        assert_eq!(
            features.load(&reader, 0).unwrap_err().to_string(),
            "Unsupported acoustic feature configuration; expected phoneme/duration features, one F0 channel and 71 output channels without DCT postprocessing."
        );
        assert_eq!(features.frame_interval_seconds(), 0.0);
    }

    #[test]
    fn loader_rejects_a_unified_width_mismatch() {
        // Unified set holds 2 symbols but the payload claims 4 phoneme channels.
        let mut payload = valid_payload();
        payload[0] = 4;
        let mut reader_payload = Vec::new();
        for word in payload {
            reader_payload.extend_from_slice(&word.to_le_bytes());
        }
        let group = v2_node(named_id("cmpg1"), &[], &[phone_set_node("japanese-test")]);
        let dct = v2_node(
            named_id("_dctov0"),
            &[],
            &[
                v2_node(
                    named_id("prim0"),
                    &dense_payload(64, 64, &vec![0.0; 64 * 64]),
                    &[],
                ),
                v2_node(
                    named_id("prim0"),
                    &dense_payload(64, 64, &vec![0.0; 64 * 64]),
                    &[],
                ),
            ],
        );
        let root = v2_node(
            named_id("_ftmfv3"),
            &reader_payload,
            &[
                group,
                phone_set_node("multiphone"),
                dct,
                cmpu_node(&[0.0; 4], &[1.0; 4], &[0.0, 1.0]),
                cmpu_node(&[-8.0], &[8.0], &[0.0, 1.0]),
                cmpu_node(&vec![0.0; 71], &vec![1.0; 71], &[0.0, 1.0]),
            ],
        );
        let reader = DnniReader::from_bytes(v2_file(&root)).expect("blob must parse");
        let mut features = AcousticFeatures::new();
        assert_eq!(
            features.load(&reader, 0).unwrap_err().to_string(),
            "Unified acoustic phoneme count does not match the feature width."
        );
    }

    #[test]
    fn unloaded_frontend_rejects_everything() {
        let features = AcousticFeatures::new();
        let mut tensor = DnniTensor::default();
        assert_eq!(
            features
                .encode_phonemes(&[timed("japanese", "a", 2)], &mut tensor)
                .unwrap_err()
                .to_string(),
            "Acoustic features have not been loaded."
        );
        assert_eq!(
            features
                .find_language_index("japanese")
                .unwrap_err()
                .to_string(),
            "Acoustic features have not been loaded."
        );
        assert_eq!(
            features
                .decode_acoustic_71(
                    &DnniTensor {
                        frames: 1,
                        channels: 71,
                        values: vec![0.0; 71]
                    },
                    &[0.0],
                    &mut tensor
                )
                .unwrap_err()
                .to_string(),
            "Acoustic features have not been loaded."
        );
    }
}
