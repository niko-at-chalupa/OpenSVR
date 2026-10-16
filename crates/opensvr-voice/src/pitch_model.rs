//! Automatic F0 prediction: the `gen5` pitch model.
//!
//! Ports OpenSV's `src/synthesis/PitchModel.{h,cpp}` (namespace
//! `sv::synthesis`): loads the `f0model-dds` wrapper (expression controls,
//! frontend, speaker vector, enhancement projection, context, decoder),
//! predicts absolute MIDI pitch on the 5 ms grid from score notes plus
//! phoneme durations, then smooths, denormalizes, postprocesses and restores
//! the transposed register.
//!
//! The phrase is transposed into the trained `[36, 79]` register before
//! prediction and shifted back afterwards. Noise comes from a ported
//! `std::mt19937` (seed 123, reseeded at phoneme/note boundaries) feeding the
//! hand-written Box-Muller rejection loop — never `std::normal_distribution`
//! equivalents — so the stream matches the C++ engine.

use std::sync::atomic::{AtomicU64, Ordering};

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniInference, DnniReader, DnniRunStatistics, DnniTensor};
use opensvr_g2p::PhonemeDuration;

use crate::noise::GaussianNoise;
use crate::pitch_context::{PitchContext, PitchContextState};
use crate::pitch_decoder::{PitchDecoder, PitchDecoderNotes};
use crate::pitch_features::{PitchFeatureOutput, PitchFeatures, PitchNote};
use crate::pitch_postprocess::{PitchFrameInterval, apply_pitch_postprocess};
use crate::{Result, invalid};

/// Verified `f0model-dds` wrapper type hashes (root, one child).
const WRAPPER_TYPES: [u64; 12] = [
    0xcce2_240f_4753_c837,
    0x0bc3_c8e1_1c36_8859,
    0xb7d7_6850_e579_425a,
    0xa81b_4e0a_3424_31ed,
    0xfd1c_9123_b5cc_f269,
    0x736a_014c_0a00_406a,
    0x933c_d542_e8c3_839b,
    0xdb0b_e7fe_929c_02e3,
    0xbbcb_3355_f533_1ec9,
    0xfdf9_9c04_dc49_6ac2,
    0x8cd9_01ed_074d_a870,
    0x3b69_b825_e5c1_bb3c,
];

/// Verified `gen5` model node type hashes (model child, five children).
const MODEL_TYPES: [u64; 12] = [
    0x5b85_62d1_8c97_8b91,
    0xe781_be47_3240_733f,
    0xa8ad_7b27_15fc_5e90,
    0xef13_3a93_0d94_f0fb,
    0x0461_232f_81e7_260f,
    0x6715_e413_10bb_7ce0,
    0xbcd2_df86_9cfd_154d,
    0xa7dc_6559_d4dc_f415,
    0xc310_c561_c14f_056f,
    0xc58a_4d93_dd62_3a98,
    0xe327_97f2_732f_379a,
    0x1bd8_f79c_0eac_a55e,
];

/// Expected pitch prediction grid in seconds.
const PREDICTION_INTERVAL_SECONDS: f32 = 0.005;

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Pitch model: {}", reason.into()))
}

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_model_identity() -> u64 {
    NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

// --- Deterministic noise lives in `crate::noise` (shared with the acoustic
// stage: exact `std::mt19937` plus the hand-written Box-Muller rejection
// loop — never substitute `rand_distr`) ---

/// 32-bit FNV-1a over the sign-extended bytes, matching the C++ hasher.
fn hash_seed_text(text: &str) -> u32 {
    let mut hash = 0x811c_9dc5_u32;
    for byte in text.bytes() {
        hash = (hash ^ (i8::from_ne_bytes([byte]) as u32)).wrapping_mul(0x0100_0193);
    }
    hash
}

/// Boundary seed from the phoneme trigram plus the global suffix.
///
/// The editor's default suffix is zero; a zero digest falls back to the hash
/// of `" "`.
fn phoneme_noise_seed(phonemes: &[PhonemeDuration], index: usize, global_seed: u32) -> u32 {
    let mut text = String::from("magic");
    if index > 0 {
        text.push_str(&phonemes[index - 1].symbol);
    }
    text.push('_');
    text.push_str(&phonemes[index].symbol);
    text.push('_');
    if index + 1 < phonemes.len() {
        text.push_str(&phonemes[index + 1].symbol);
    }
    text.push('_');
    text.push_str(&global_seed.to_string());
    let seed = hash_seed_text(&text);
    if seed != 0 { seed } else { hash_seed_text(" ") }
}

fn same_note(first: &PitchNote, second: &PitchNote) -> bool {
    first.syllable.language == second.syllable.language
        && first.syllable.phonemes == second.syllable.phonemes
        && first.syllable.duration_seconds == second.syllable.duration_seconds
        && first.syllable.midi_pitch == second.syllable.midi_pitch
        && first.is_breath == second.is_breath
        && first.is_silence == second.is_silence
        && first.is_continuation == second.is_continuation
        && first.is_rap == second.is_rap
        && first.tone == second.tone
        && first.vibrato_modulation == second.vibrato_modulation
}

fn same_phoneme(first: &PhonemeDuration, second: &PhonemeDuration) -> bool {
    first.language == second.language
        && first.symbol == second.symbol
        && first.syllable_index == second.syllable_index
        && first.timing_interval_index == second.timing_interval_index
        && first.duration_seconds == second.duration_seconds
}

/// Editor expression smoothing: forward then backward single-pole pass.
fn smooth_control(values: &mut [f32]) {
    for frame in 1..values.len() {
        values[frame] = values[frame] * 0.95 + values[frame - 1] * 0.050_000_012_f32;
    }
    for end in (2..=values.len()).rev() {
        let frame = end - 2;
        values[frame] = values[frame] * 0.95 + values[frame + 1] * 0.050_000_012_f32;
    }
}

/// Cached phrase prediction for incremental re-synthesis.
#[derive(Debug, Default)]
pub struct PitchModelState {
    /// Context inference snapshots.
    pub context: PitchContextState,
    /// True when the cached curve was reused verbatim.
    pub reused_prediction: bool,
    notes: Vec<PitchNote>,
    phonemes: Vec<PhonemeDuration>,
    midi_pitch: Vec<f32>,
    vibrato_envelope: Vec<f32>,
    model_identity: u64,
    noise_seed: u32,
}

impl PitchModelState {
    /// Byte size of the retained caches and phrase snapshots.
    pub fn get_bytes(&self) -> usize {
        let mut bytes = self.context.get_bytes()
            + self.midi_pitch.capacity() * size_of::<f32>()
            + self.vibrato_envelope.capacity() * size_of::<f32>()
            + self.notes.capacity() * size_of::<PitchNote>()
            + self.phonemes.capacity() * size_of::<PhonemeDuration>();
        for note in &self.notes {
            bytes += note.syllable.language.capacity()
                + note.syllable.phonemes.capacity() * size_of::<String>();
            for phoneme in &note.syllable.phonemes {
                bytes += phoneme.capacity();
            }
        }
        for phoneme in &self.phonemes {
            bytes += phoneme.language.capacity() + phoneme.symbol.capacity();
        }
        bytes
    }

    /// Resets the phrase snapshot and context caches.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Automatic F0 prediction model: frontend, speaker, projection, context, decoder.
#[derive(Debug, Default)]
pub struct PitchModel {
    features: PitchFeatures,
    context: PitchContext,
    decoder: PitchDecoder,
    enhancement_projection: DnniInference,
    speaker: Vec<f32>,
    expression: f32,
    expression_strength: f32,
    rap_expression: f32,
    model_identity: u64,
    loaded: bool,
}

impl PitchModel {
    /// Creates an unloaded model; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self {
            expression: 0.8,
            expression_strength: 1.0,
            rap_expression: 0.7,
            ..Self::default()
        }
    }

    /// Loads the `f0model-dds` wrapper: 16-byte expression controls plus the
    /// five model children (frontend, 32-channel speaker, enhancement
    /// projection, context, decoder). A failed load preserves the model.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<()> {
        self.load_with_speaker_override(reader, root_node, &[])
    }

    /// Loads the pitch model with a Flat voice override for the 32-channel
    /// speaker vector (`f0_style` from the voice document). An empty override
    /// keeps the model-embedded defaults, exactly as for container voices; a
    /// non-empty override replaces them before validation.
    pub fn load_with_speaker_override(
        &mut self,
        reader: &DnniReader,
        root_node: usize,
        speaker_override: &[f32],
    ) -> Result<()> {
        let nodes = reader.nodes();
        let Some(root) = nodes
            .get(root_node)
            .filter(|root| WRAPPER_TYPES.contains(&root.type_id))
        else {
            return Err(failure("unsupported F0 model wrapper"));
        };
        if root.children.len() != 1 {
            return Err(failure("unsupported F0 model wrapper"));
        }
        let payload = reader.payload(root_node);
        let Some(model) = nodes.get(root.children[0]) else {
            return Err(failure("unsupported gen5 model layout"));
        };
        if payload.len() != 16 || !MODEL_TYPES.contains(&model.type_id) || model.children.len() != 5
        {
            return Err(failure("unsupported gen5 model layout"));
        }
        let children = model.children.clone();
        let mut candidate = Self::new();
        candidate.expression = f32::from_bits(read_word(payload, 0));
        candidate.expression_strength = f32::from_bits(read_word(payload, 4));
        candidate.rap_expression = f32::from_bits(read_word(payload, 8));
        if !candidate.expression.is_finite()
            || !candidate.expression_strength.is_finite()
            || !candidate.rap_expression.is_finite()
            || candidate.expression < 0.0
            || candidate.expression_strength < 0.0
            || candidate.rap_expression < 0.0
        {
            return Err(failure("invalid model expression controls"));
        }
        candidate.features.load(reader, children[0])?;
        candidate.speaker = reader.read_float_vector(children[1])?;
        if !speaker_override.is_empty() {
            candidate.speaker = speaker_override.to_vec();
        }
        if candidate.speaker.len() != 32
            || !candidate.speaker.iter().all(|value| value.is_finite())
            || candidate.features.frame_interval_seconds() != PREDICTION_INTERVAL_SECONDS
        {
            return Err(failure(
                "unsupported speaker dimensions or prediction interval",
            ));
        }
        candidate.enhancement_projection.load(reader, children[2])?;
        candidate.context.load(
            reader,
            children[3],
            candidate.features.phoneme_category_count(),
            candidate.features.language_count(),
        )?;
        candidate.decoder.load(reader, children[4])?;
        candidate.loaded = true;
        candidate.model_identity = next_model_identity();
        *self = candidate;
        Ok(())
    }

    /// Frame interval in seconds (0.0 when unloaded).
    pub fn frame_interval_seconds(&self) -> f32 {
        if self.loaded {
            self.features.frame_interval_seconds()
        } else {
            0.0
        }
    }

    /// Predicts absolute MIDI pitch per 5 ms frame from time zero.
    ///
    /// `notes` and `phonemes` describe the same complete prediction interval
    /// (including any caller-prepared silence context); all frames are
    /// returned in `midi_pitch`. `vibrato_envelope` is sampled per frame
    /// (empty means unity) and `noise_seed` is the global suffix of the
    /// phoneme-context seed events. `output` is only assigned on success.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn run(
        &self,
        notes: &[PitchNote],
        phonemes: &[PhonemeDuration],
        midi_pitch: &mut Vec<f32>,
        noise_seed: u32,
        mut state: Option<&mut PitchModelState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
        vibrato_envelope: &[f32],
    ) -> Result<()> {
        if !self.loaded {
            return Err(failure("no model has been loaded"));
        }
        if let Some(state) = state.as_mut() {
            state.reused_prediction = false;
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        if notes.is_empty() || phonemes.is_empty() {
            return Err(failure("a complete score and phoneme sequence is required"));
        }
        if let Some(state) = state.as_mut() {
            if state.model_identity == self.model_identity
                && state.noise_seed == noise_seed
                && state.notes.len() == notes.len()
                && state.phonemes.len() == phonemes.len()
                && notes.iter().zip(&state.notes).all(|(a, b)| same_note(a, b))
                && phonemes
                    .iter()
                    .zip(&state.phonemes)
                    .all(|(a, b)| same_phoneme(a, b))
                && vibrato_envelope == state.vibrato_envelope.as_slice()
            {
                midi_pitch.clone_from(&state.midi_pitch);
                state.reused_prediction = true;
                return Ok(());
            }
        }
        let mut minimum_pitch = 128_i32;
        let mut maximum_pitch = 0_i32;
        for note in notes {
            if note.syllable.midi_pitch < 0
                || note.syllable.midi_pitch > 127
                || !note.vibrato_modulation.is_finite()
            {
                return Err(failure("note pitch or vibrato modulation is invalid"));
            }
            minimum_pitch = minimum_pitch.min(note.syllable.midi_pitch);
            maximum_pitch = maximum_pitch.max(note.syllable.midi_pitch);
        }
        // The wrapper transposes the phrase into its trained range, then
        // restores the requested register after prediction.
        let mut transpose = if maximum_pitch > 79 {
            79 - maximum_pitch
        } else {
            0
        };
        if minimum_pitch < 36 {
            transpose = 36 - minimum_pitch;
        }
        let mut model_notes: Vec<PitchNote> = notes.to_vec();
        for note in &mut model_notes {
            note.syllable.midi_pitch += transpose;
        }
        let mut encoded = PitchFeatureOutput::default();
        self.features.encode(&model_notes, phonemes, &mut encoded)?;

        let mut enhanced_speaker = DnniTensor::default();
        self.enhancement_projection.run_with_cache(
            &DnniTensor {
                frames: 1,
                channels: 1,
                values: vec![1.0],
            },
            &mut enhanced_speaker,
            None,
            None,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if enhanced_speaker.frames != 1
            || enhanced_speaker.channels != self.speaker.len()
            || enhanced_speaker.values.len() != self.speaker.len()
        {
            return Err(failure(
                "enhancement projection returned an invalid speaker shape",
            ));
        }
        for (channel, value) in self.speaker.iter().enumerate() {
            enhanced_speaker.values[channel] += value;
        }

        let mut feed_forward = DnniTensor::default();
        let mut residual = DnniTensor::default();
        {
            let mut context_state = None;
            if let Some(state) = state.as_mut() {
                context_state = Some(&mut state.context);
            }
            self.context.run(
                &encoded,
                &enhanced_speaker.values,
                &mut feed_forward,
                &mut residual,
                context_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
        }
        let frames = residual.frames;
        if frames == 0 || feed_forward.frames != frames {
            return Err(failure("the score context has an invalid frame count"));
        }
        if !vibrato_envelope.is_empty()
            && (vibrato_envelope.len() < frames
                || !vibrato_envelope[..frames]
                    .iter()
                    .all(|value| value.is_finite()))
        {
            return Err(failure(
                "the vibrato envelope must contain a finite value for each prediction frame",
            ));
        }
        let mut expressive_frames = vec![self.expression; frames];
        let mut vibrato_frames = vec![0.0_f32; frames];
        let mut score_pitch = Vec::with_capacity(model_notes.len());
        let mut frame = 0_usize;
        for (note_index, note) in model_notes.iter().enumerate() {
            score_pitch.push(note.syllable.midi_pitch as f32);
            let count = encoded.note_frame_counts[note_index];
            if count > frames - frame {
                return Err(failure("note durations exceed the context frame count"));
            }
            let note_expression = if note.is_rap {
                self.expression * self.rap_expression
            } else {
                self.expression
            };
            expressive_frames[frame..frame + count].fill(note_expression);
            for within in 0..count {
                let envelope = if vibrato_envelope.is_empty() {
                    1.0
                } else {
                    vibrato_envelope[frame + within]
                };
                vibrato_frames[frame + within] =
                    (note.vibrato_modulation * envelope).clamp(0.0, 2.0) - 1.0;
            }
            frame += count;
        }
        smooth_control(&mut expressive_frames);
        let mut controls = DnniTensor {
            frames,
            channels: 3,
            values: vec![0.0; frames * 3],
        };
        let mut noise = vec![0.0_f32; frames];
        let mut noise_source = GaussianNoise::new(123);
        let mut next_phoneme = 0_usize;
        let mut next_phoneme_frame = 0_usize;
        let mut next_note = 0_usize;
        let mut next_note_frame = 0_usize;
        let mut active_phoneme_seed = 0_u32;
        for frame in 0..frames {
            let mut reseed = false;
            // Each event stream consumes at most one boundary per frame,
            // including zero-frame intervals. A note boundary emits an event
            // even when its default take id is zero and leaves the phoneme
            // seed intact.
            if next_phoneme < phonemes.len() && frame >= next_phoneme_frame {
                active_phoneme_seed = phoneme_noise_seed(phonemes, next_phoneme, noise_seed);
                next_phoneme_frame += encoded.phoneme_frame_counts[next_phoneme];
                next_phoneme += 1;
                reseed = true;
            }
            if next_note < model_notes.len() && frame >= next_note_frame {
                next_note_frame += encoded.note_frame_counts[next_note];
                next_note += 1;
                reseed = true;
            }
            if reseed {
                noise_source.reseed(active_phoneme_seed);
            }
            let expressive = expressive_frames[frame];
            let mut strength = self.expression_strength;
            if expressive > 0.8 {
                strength =
                    (f64::from(strength) * (0.0_f64).max(3.0 - 2.5 * f64::from(expressive))) as f32;
            }
            controls.values[frame * 3] = 0.995;
            controls.values[frame * 3 + 1] = strength;
            controls.values[frame * 3 + 2] = 0.0;
            noise[frame] = noise_source.next() * (expressive * 2.0).min(1.0);
        }
        let mut vowel_frames = vec![0_u8; frames];
        frame = 0;
        for (phone, &count) in encoded.phoneme_frame_counts.iter().enumerate() {
            if count > frames - frame {
                return Err(failure("phoneme durations exceed the context frame count"));
            }
            vowel_frames[frame..frame + count].fill(encoded.phoneme_vowels[phone]);
            frame += count;
        }
        let mut decoder_notes = PitchDecoderNotes {
            frame_counts: &encoded.note_frame_counts,
            midi_pitch: &score_pitch,
            vowel_frames: &vowel_frames,
            ..PitchDecoderNotes::default()
        };
        let mut normalized = DnniTensor::default();
        self.decoder.run(
            &residual,
            &feed_forward,
            &noise,
            &controls,
            &vibrato_frames,
            &self.features,
            &mut decoder_notes,
            &mut normalized,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if normalized.frames != frames
            || normalized.channels != 1
            || normalized.values.len() != frames
        {
            return Err(failure("pitch decoder returned an invalid output shape"));
        }
        let mut smoothed = vec![0.0_f32; frames];
        for (frame, slot) in smoothed.iter_mut().enumerate() {
            let mut value = if frame > 1 {
                normalized.values[frame - 1] * 0.25
            } else {
                0.0
            };
            if frame > 0 {
                value += normalized.values[frame] * 0.5;
            }
            if frame + 1 < frames {
                value += normalized.values[frame + 1] * 0.25;
            }
            *slot = value;
        }
        let mut result_pitch = Vec::new();
        self.features
            .denormalize_pitch(&smoothed, &mut result_pitch)?;
        apply_pitch_postprocess(
            &model_notes,
            PitchFrameInterval::new(self.features.frame_interval_seconds())?,
            &mut result_pitch,
            cancel,
        )?;
        for pitch in &mut result_pitch {
            *pitch -= transpose as f32;
            if !pitch.is_finite() {
                return Err(failure("the predicted pitch contains a non-finite value"));
            }
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        if let Some(state) = state.as_mut() {
            state.model_identity = 0;
            state.notes = notes.to_vec();
            state.phonemes = phonemes.to_vec();
            state.midi_pitch.clone_from(&result_pitch);
            state.vibrato_envelope = vibrato_envelope.to_vec();
            state.noise_seed = noise_seed;
            state.model_identity = self.model_identity;
        }
        *midi_pitch = result_pitch;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_seed_text_matches_fnv1a() {
        assert_eq!(hash_seed_text("magic"), 0xe4eb_258c);
        assert_eq!(hash_seed_text("magic_a_b_0"), 0xe0aa_ae9a);
        assert_eq!(hash_seed_text(" "), hash_seed_text(" "));
    }

    #[test]
    fn phoneme_noise_seed_formats_the_trigram() {
        let phonemes = vec![
            PhonemeDuration {
                language: "japanese".to_owned(),
                symbol: "a".to_owned(),
                syllable_index: 0,
                timing_interval_index: 0,
                duration_seconds: 0.1,
            },
            PhonemeDuration {
                language: "japanese".to_owned(),
                symbol: "b".to_owned(),
                syllable_index: 0,
                timing_interval_index: 0,
                duration_seconds: 0.1,
            },
        ];
        assert_eq!(
            phoneme_noise_seed(&phonemes, 0, 0),
            hash_seed_text("magic_a_b_0")
        );
        assert_ne!(
            phoneme_noise_seed(&phonemes, 0, 0),
            phoneme_noise_seed(&phonemes, 0, 1)
        );
    }

    #[test]
    fn smooth_control_matches_the_two_pass_smoother() {
        // Forward: v[1] = 2 * 0.95 + 1 * 0.05; backward folds v[1] back.
        let mut values = vec![1.0_f32, 2.0_f32];
        smooth_control(&mut values);
        let forward = 2.0_f32 * 0.95 + 1.0 * 0.050_000_012_f32;
        assert!((values[1] - forward).abs() < 1e-7, "got {}", values[1]);
        let backward = 1.0_f32 * 0.95 + forward * 0.050_000_012_f32;
        assert!((values[0] - backward).abs() < 1e-7, "got {}", values[0]);
    }

    #[test]
    fn unloaded_model_rejects_run() {
        let mut pitch = Vec::new();
        assert_eq!(
            PitchModel::new()
                .run(
                    &[],
                    &[],
                    &mut pitch,
                    0,
                    None,
                    &CancelToken::new(),
                    None,
                    &[],
                )
                .unwrap_err()
                .to_string(),
            "Pitch model: no model has been loaded"
        );
    }

    #[test]
    fn loader_rejects_a_non_wrapper_node() {
        let reader = DnniReader::from_bytes(vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .expect("blob must parse");
        assert_eq!(
            PitchModel::new().load(&reader, 0).unwrap_err().to_string(),
            "Pitch model: unsupported F0 model wrapper"
        );
    }
}
