//! Neural context encoder for the pitch prediction model.
//!
//! Ports OpenSV's `src/synthesis/PitchContext.{h,cpp}` (namespace
//! `sv::synthesis`): 32-channel categorical embeddings for phoneme categories
//! and languages, bidirectional note/phoneme recurrent encoders, expansion of
//! both streams onto the 5 ms frame grid (with 8-phase cosine intraphone
//! position and speaker/language conditioning, 328 channels total), then a
//! shared projection splitting into feed-forward and residual conditioning
//! branches for the pitch decoder.

use opensvr_core::CancelToken;
use opensvr_dnni::{
    DnniCache, DnniInference, DnniMatrix, DnniReader, DnniRunStatistics, DnniTensor,
};

use crate::pitch_features::PitchFeatureOutput;
use crate::{Result, invalid};

/// Expanded frame feature channels: 128 note + 128 phoneme + 8 cosine phase
/// + 32 speaker + 32 language.
const CONTEXT_CHANNELS: usize = 328;
/// Maximum frames in one prediction interval (64 Mi elements / channels).
const MAXIMUM_FRAMES: usize = 64 * 1024 * 1024 / CONTEXT_CHANNELS;
/// Embedding width of the categorical phoneme/language tables.
const EMBEDDING_CHANNELS: usize = 32;

/// Verified `gen5` context node type hashes.
const CONTEXT_TYPES: [u64; 12] = [
    0xa5e3_2125_1caa_33ed,
    0xef76_985a_fa86_ddf3,
    0xad6e_ce19_8010_cb64,
    0x2b04_7742_9820_6d1f,
    0xf396_d3b2_a858_2243,
    0x0177_1a1f_6fab_6574,
    0xb1ec_6d61_28b1_4ae9,
    0xfa28_a01c_d2e6_d271,
    0xb092_f5e4_e64d_93a3,
    0x470d_b6b4_1577_b68c,
    0x8ef5_b034_49c5_4d0e,
    0x3ff6_7ea0_5f32_0122,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Pitch context: {}", reason.into()))
}

/// Retained inference snapshots for phrase-level incremental caching.
///
/// Each cache belongs to one phrase on the synthesis worker thread and is
/// never shared across threads.
#[derive(Debug, Default)]
pub struct PitchContextState {
    /// Shared projection cache.
    pub projection: DnniCache,
    /// Feed-forward branch cache.
    pub feed_forward: DnniCache,
    /// Residual branch cache.
    pub residual: DnniCache,
}

impl PitchContextState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.projection.get_bytes() + self.feed_forward.get_bytes() + self.residual.get_bytes()
    }

    /// Resets all branch caches.
    pub fn clear(&mut self) {
        self.projection.clear();
        self.feed_forward.clear();
        self.residual.clear();
    }
}

fn check_shape(tensor: &DnniTensor, frames: usize, channels: usize) -> Result<()> {
    if tensor.frames != frames
        || tensor.channels != channels
        || tensor.values.len() != frames * channels
    {
        return Err(failure("unexpected tensor dimensions"));
    }
    if !tensor.values.iter().all(|value| value.is_finite()) {
        return Err(failure("non-finite tensor value"));
    }
    Ok(())
}

fn count_frames(durations: &[usize], total: &mut usize) -> Result<()> {
    *total = 0;
    for &duration in durations {
        if duration > MAXIMUM_FRAMES - *total {
            return Err(failure("frame sequence exceeds the memory limit"));
        }
        *total += duration;
    }
    Ok(())
}

/// 8-phase cosine harmonic basis of the relative position inside a phoneme.
///
/// `angle = (position + (1 - phase) * 0.125) * 2pi`, stored as
/// `cos(angle) * 0.5 + 0.5`; a single-frame phoneme sits at position 0.5.
fn cosine_phase(position: f32) -> [f32; 8] {
    let mut phases = [0.0_f32; 8];
    for (phase, slot) in phases.iter_mut().enumerate() {
        let angle = (position + (1.0 - phase as f32) * 0.125) * 6.283_185_5_f32;
        *slot = (f64::from(angle.cos()) * 0.5 + 0.5) as f32;
    }
    phases
}

/// Context encoder: categorical embeddings plus five sub-networks.
#[derive(Debug)]
pub struct PitchContext {
    phoneme_embedding: DnniMatrix,
    language_embedding: DnniMatrix,
    note_encoder: DnniInference,
    phoneme_encoder: DnniInference,
    projection: DnniInference,
    feed_forward: DnniInference,
    residual: DnniInference,
    loaded: bool,
}

impl Default for PitchContext {
    fn default() -> Self {
        Self {
            phoneme_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            language_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            note_encoder: DnniInference::new(),
            phoneme_encoder: DnniInference::new(),
            projection: DnniInference::new(),
            feed_forward: DnniInference::new(),
            residual: DnniInference::new(),
            loaded: false,
        }
    }
}

impl PitchContext {
    /// Creates an unloaded encoder; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads embeddings and the five sub-networks from the context node.
    ///
    /// The node carries a 4-byte payload and seven children: the two `modl4`
    /// categorical embeddings (32 channels by frontend category/language
    /// counts), then note encoder, phoneme encoder, shared projection,
    /// feed-forward branch and residual branch networks. A failed load
    /// preserves the current encoder.
    pub fn load(
        &mut self,
        reader: &DnniReader,
        node_index: usize,
        phoneme_category_count: usize,
        language_count: usize,
    ) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes
            .get(node_index)
            .filter(|node| CONTEXT_TYPES.contains(&node.type_id))
        else {
            return Err(failure("unsupported gen5 context layout"));
        };
        if node.children.len() != 7 || reader.payload(node_index).len() != 4 {
            return Err(failure("unsupported gen5 context layout"));
        }
        let children = node.children.clone();
        let valid_embedding = |index: usize| {
            nodes.get(index).is_some_and(|node| {
                node.name == "modl4" && node.children.len() == 1 && reader.payload(index).is_empty()
            })
        };
        if !valid_embedding(children[0]) || !valid_embedding(children[1]) {
            return Err(failure("missing categorical embeddings"));
        }
        let phoneme_matrix = reader.read_float_matrix(nodes[children[0]].children[0])?;
        let language_matrix = reader.read_float_matrix(nodes[children[1]].children[0])?;
        if phoneme_category_count == 0
            || language_count == 0
            || phoneme_matrix.rows as usize != EMBEDDING_CHANNELS
            || phoneme_matrix.columns as usize != phoneme_category_count
            || language_matrix.rows as usize != EMBEDDING_CHANNELS
            || language_matrix.columns as usize != language_count
        {
            return Err(failure(
                "32-channel categorical embeddings must match the frontend's category and language tables",
            ));
        }
        let mut note_encoder = DnniInference::new();
        note_encoder.load(reader, children[2])?;
        let mut phoneme_encoder = DnniInference::new();
        phoneme_encoder.load(reader, children[3])?;
        let mut projection = DnniInference::new();
        projection.load(reader, children[4])?;
        let mut feed_forward = DnniInference::new();
        feed_forward.load(reader, children[5])?;
        let mut residual = DnniInference::new();
        residual.load(reader, children[6])?;
        *self = Self {
            phoneme_embedding: phoneme_matrix,
            language_embedding: language_matrix,
            note_encoder,
            phoneme_encoder,
            projection,
            feed_forward,
            residual,
            loaded: true,
        };
        Ok(())
    }

    /// Encodes features plus speaker into dual-branch conditioning tensors.
    ///
    /// Evaluates the note and phoneme recurrent encoders without caching,
    /// expands both onto the frame grid with cosine phase and embedding
    /// conditioning, then runs the cached shared projection and its two
    /// branches. Outputs are only assigned on success.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn run(
        &self,
        features: &PitchFeatureOutput,
        speaker: &[f32],
        feed_forward_output: &mut DnniTensor,
        residual_output: &mut DnniTensor,
        state: Option<&mut PitchContextState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        if !self.loaded {
            return Err(failure("model has not been loaded"));
        }
        let note_count = features.note_frame_counts.len();
        let phone_count = features.phoneme_frame_counts.len();
        if note_count == 0
            || phone_count == 0
            || note_count > MAXIMUM_FRAMES
            || phone_count > MAXIMUM_FRAMES
            || features.phoneme_categories.len() != phone_count
            || features.phoneme_languages.len() != phone_count
        {
            return Err(failure("inconsistent score or phoneme sequences"));
        }
        check_shape(&features.note_features, note_count, 10)?;
        if speaker.len() != EMBEDDING_CHANNELS || !speaker.iter().all(|value| value.is_finite()) {
            return Err(failure("expected a finite 32-channel speaker vector"));
        }
        let mut note_frames = 0_usize;
        let mut phone_frames = 0_usize;
        count_frames(&features.note_frame_counts, &mut note_frames)?;
        count_frames(&features.phoneme_frame_counts, &mut phone_frames)?;
        let frame_count = note_frames.max(phone_frames);
        if frame_count == 0 {
            return Err(failure("empty prediction interval"));
        }
        if cancel.is_cancelled() {
            return Err(failure("cancelled"));
        }

        // 1. Recurrent note encoder (no cache).
        let mut note_context = DnniTensor::default();
        self.note_encoder.run_with_cache(
            &features.note_features,
            &mut note_context,
            None,
            None,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&note_context, note_count, 128)?;

        // 2. Phoneme categorical embedding input `[phones, 32]`.
        let mut phone_input = DnniTensor {
            frames: phone_count,
            channels: EMBEDDING_CHANNELS,
            values: vec![0.0; phone_count * EMBEDDING_CHANNELS],
        };
        for phone in 0..phone_count {
            let category = features.phoneme_categories[phone];
            let language = features.phoneme_languages[phone];
            if category >= self.phoneme_embedding.columns as usize
                || language >= self.language_embedding.columns as usize
            {
                return Err(failure(
                    "categorical index exceeds the embedding dimensions",
                ));
            }
            for channel in 0..EMBEDDING_CHANNELS {
                phone_input.values[phone * EMBEDDING_CHANNELS + channel] = self
                    .phoneme_embedding
                    .values[channel * self.phoneme_embedding.columns as usize + category];
            }
        }

        // 3. Recurrent phoneme encoder (no cache).
        let mut phone_context = DnniTensor::default();
        self.phoneme_encoder.run_with_cache(
            &phone_input,
            &mut phone_context,
            None,
            None,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&phone_context, phone_count, 128)?;

        // 4. Expanded frame context `[frames, 328]`, zero-initialized.
        let mut context = DnniTensor {
            frames: frame_count,
            channels: CONTEXT_CHANNELS,
            values: vec![0.0; frame_count * CONTEXT_CHANNELS],
        };
        let mut frame = 0_usize;
        for note in 0..note_count {
            for _ in 0..features.note_frame_counts[note] {
                if frame % 256 == 0 && cancel.is_cancelled() {
                    return Err(failure("cancelled"));
                }
                context.values[frame * CONTEXT_CHANNELS..frame * CONTEXT_CHANNELS + 128]
                    .copy_from_slice(&note_context.values[note * 128..note * 128 + 128]);
                frame += 1;
            }
        }
        frame = 0;
        let mut last_language = 0_usize;
        for phone in 0..phone_count {
            let duration = features.phoneme_frame_counts[phone];
            for within in 0..duration {
                if frame % 256 == 0 && cancel.is_cancelled() {
                    return Err(failure("cancelled"));
                }
                let destination =
                    &mut context.values[frame * CONTEXT_CHANNELS..(frame + 1) * CONTEXT_CHANNELS];
                destination[128..256]
                    .copy_from_slice(&phone_context.values[phone * 128..phone * 128 + 128]);
                let position = if duration == 1 {
                    0.5
                } else {
                    within as f32 / (duration - 1) as f32
                };
                destination[256..264].copy_from_slice(&cosine_phase(position));
                last_language = features.phoneme_languages[phone];
                for channel in 0..EMBEDDING_CHANNELS {
                    destination[296 + channel] = self.language_embedding.values
                        [channel * self.language_embedding.columns as usize + last_language];
                }
                frame += 1;
            }
        }
        // Frames past the phoneme stream (longer note tail) keep the last
        // language; the recurrent and phase channels stay zero.
        for rest in frame..frame_count {
            for channel in 0..EMBEDDING_CHANNELS {
                context.values[rest * CONTEXT_CHANNELS + 296 + channel] = self
                    .language_embedding
                    .values[channel * self.language_embedding.columns as usize + last_language];
            }
        }
        // Broadcast the 32-channel speaker embedding.
        for rest in 0..frame_count {
            context.values[rest * CONTEXT_CHANNELS + 264..rest * CONTEXT_CHANNELS + 296]
                .copy_from_slice(speaker);
        }

        // 5. Shared projection, then the two conditioning branches.
        let (mut projection_state, mut feed_forward_state, mut residual_state) = (None, None, None);
        if let Some(state) = state {
            projection_state = Some(&mut state.projection);
            feed_forward_state = Some(&mut state.feed_forward);
            residual_state = Some(&mut state.residual);
        }
        let mut projected = DnniTensor::default();
        self.projection.run_with_cache(
            &context,
            &mut projected,
            None,
            projection_state,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&projected, frame_count, 128)?;
        let mut feed_forward_result = DnniTensor::default();
        let mut residual_result = DnniTensor::default();
        self.feed_forward.run_with_cache(
            &projected,
            &mut feed_forward_result,
            None,
            feed_forward_state,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        self.residual.run_with_cache(
            &projected,
            &mut residual_result,
            None,
            residual_state,
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&feed_forward_result, frame_count, 128)?;
        check_shape(&residual_result, frame_count, 128)?;
        *feed_forward_output = feed_forward_result;
        *residual_output = residual_result;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_phase_matches_hand_computed_harmonics() {
        // position 0.5, phase 0: angle = 0.625 * 2pi, cos = -sqrt(2)/2,
        // stored as cos * 0.5 + 0.5 ~= 0.1464466.
        let phases = cosine_phase(0.5);
        assert!((phases[0] - 0.146_446_6).abs() < 1e-6, "got {}", phases[0]);
        // Every pair sits on the shifted unit circle: values in [0, 1].
        for (phase, value) in phases.iter().enumerate() {
            let angle = (0.5 + (1.0 - phase as f32) * 0.125) * 6.283_185_5_f32;
            assert!((value - (angle.cos() * 0.5 + 0.5)).abs() < 1e-7);
            assert!((0.0..=1.0).contains(value));
        }
    }

    #[test]
    fn unloaded_encoder_rejects_run() {
        let features = PitchFeatureOutput::default();
        let mut feed_forward = DnniTensor::default();
        let mut residual = DnniTensor::default();
        assert_eq!(
            PitchContext::new()
                .run(
                    &features,
                    &[0.0; 32],
                    &mut feed_forward,
                    &mut residual,
                    None,
                    &CancelToken::new(),
                    None,
                )
                .unwrap_err()
                .to_string(),
            "Pitch context: model has not been loaded"
        );
    }

    #[test]
    fn loader_rejects_an_unknown_layout() {
        let reader = DnniReader::from_bytes(vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .expect("blob must parse");
        assert_eq!(
            PitchContext::new()
                .load(&reader, 0, 2, 1)
                .unwrap_err()
                .to_string(),
            "Pitch context: unsupported gen5 context layout"
        );
    }
}
