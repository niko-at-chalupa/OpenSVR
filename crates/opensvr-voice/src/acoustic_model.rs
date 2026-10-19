//! Vocoder-ready acoustic frame prediction (`AcousticModel`).
//!
//! Ports OpenSV's `src/synthesis/AcousticModel.{h,cpp}` (namespace
//! `sv::synthesis`): the `_rldtg0` root (model + style children, architecture
//! `"dds"` via the voice config's `model_timbre_pred` entry) predicts
//! 71-channel normalized acoustic frames from [`TimedPhoneme`]s plus the
//! per-frame log-F0 curve, then [`AcousticFeatures::decode_acoustic_71`]
//! widens them to 72 vocoder-ready channels (voiced flag, F0 in Hz, spectral
//! and auxiliary features).
//!
//! Pipeline per run: phoneme features -> phoneme encoder -> 144-channel
//! frame context (encoded phoneme, 8 cosine position/pitch phases, speaker,
//! language, vocal-mode style) -> context network + projection (256 ch each)
//! -> diffusion-style latent sampler (128 control channels plus fresh
//! Gaussian noise, seed-selected) -> spectral head (64 ch) + auxiliary head
//! (7 ch) -> interleaved 71 channels -> decoded output.
//!
//! Noise comes from the shared [`crate::noise`] `std::mt19937` Box-Muller
//! stream (one contiguous draw sequence per run, default seed 5489), never
//! `rand_distr` substitutes.

use std::collections::BTreeMap;

use opensvr_core::CancelToken;
use opensvr_dnni::{
    DnniCache, DnniInference, DnniMatrix, DnniReader, DnniRunStatistics, DnniTensor,
};

use crate::acoustic_features::AcousticFeatures;
use crate::noise::GaussianNoise;
use crate::timing::TimedPhoneme;
use crate::{Result, invalid};

/// Maximum phrase frames (64 Mi elements over the 388-wide latent condition).
const MAX_FRAMES: usize = 64 * 1024 * 1024 / 388;
/// Frame context width: 64 encoded + 8 phase + 32 speaker + 8 language + 32 style.
const CONTEXT_CHANNELS: usize = 144;
/// Latent condition width: 256 context + 128 controls + 4 noise.
const CONDITION_CHANNELS: usize = 388;
/// Latent channels sampled per frame.
const LATENT_CHANNELS: usize = 4;
/// Expected acoustic frame interval in seconds.
const FRAME_INTERVAL_SECONDS: f32 = 0.005;

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Acoustic model: {}", reason.into()))
}

/// Resolves document vocal modes to absolute style vectors in sorted-name
/// order: each offset added onto the (possibly overridden) default style.
/// The run-path mixing (`weight * (absolute - default)`) then reproduces the
/// reference (`weight * offset`) exactly.
fn absolute_document_styles(
    default_style: &[f32],
    vocal_modes: &BTreeMap<String, Vec<f32>>,
) -> Result<Vec<Vec<f32>>> {
    let mut styles = Vec::with_capacity(vocal_modes.len());
    for offset in vocal_modes.values() {
        if offset.len() != default_style.len() {
            return Err(failure(
                "a vocal-mode style vector has an unsupported dimension",
            ));
        }
        let absolute: Vec<f32> = default_style
            .iter()
            .zip(offset.iter())
            .map(|(base, delta)| base + delta)
            .collect();
        if !absolute.iter().all(|value| value.is_finite()) {
            return Err(failure("voice speaker/style embeddings must be finite"));
        }
        styles.push(absolute);
    }
    Ok(styles)
}

fn check_shape(tensor: &DnniTensor, frames: usize, channels: usize, name: &str) -> Result<()> {
    if tensor.frames != frames
        || tensor.channels != channels
        || tensor.values.len() != frames * channels
    {
        return Err(failure(format!(
            "{name} returned an unexpected tensor shape"
        )));
    }
    if !tensor.values.iter().all(|value| value.is_finite()) {
        return Err(failure(format!("{name} returned a non-finite value")));
    }
    Ok(())
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Cosine phases for the scalar position/pitch expansion in
/// [`AcousticModel::make_context`]: `[pi/2, 0, -pi/2, -pi]`, matching the
/// C++ literals bit for bit (see `cosine_constants_match_the_cpp_float_bits`).
const COSINE_PHASES: [f32; 4] = [
    std::f32::consts::FRAC_PI_2,
    0.0,
    -std::f32::consts::FRAC_PI_2,
    -std::f32::consts::PI,
];
/// Editor normal-take diffusion controls (the low-level empty-queue
/// defaults are not the editor defaults).
const DIFFUSION_TEMPERATURE: f32 = 0.995;
const SOURCE_TEMPERATURE: f32 = 0.5;
const GUIDANCE: f32 = 1.0;

/// Frame interval of the acoustic model grid (typically 5 ms).
///
/// Kept distinct from [`crate::PitchFrameInterval`] so the two grids can
/// never be mixed up at a call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticFrameInterval(f32);

impl AcousticFrameInterval {
    /// Builds an interval; rejects non-finite and non-positive values.
    pub fn new(seconds: f32) -> Result<Self> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(invalid("Acoustic model: invalid frame interval."));
        }
        Ok(Self(seconds))
    }

    /// Interval in seconds.
    pub fn seconds(self) -> f32 {
        self.0
    }
}

/// Retained inference snapshots for phrase-level incremental caching.
///
/// Each cache belongs to one phrase on the synthesis worker thread and is
/// never shared across threads.
#[derive(Debug, Default)]
pub struct AcousticModelState {
    /// Phoneme encoder cache.
    pub phoneme_encoder: DnniCache,
    /// Frame-context projection cache.
    pub context_projection: DnniCache,
    /// Frame-context network cache.
    pub context_network: DnniCache,
    /// Latent projection cache.
    pub latent_projection: DnniCache,
    /// Latent diffusion network cache.
    pub latent_network: DnniCache,
    /// Latent head cache.
    pub latent_head: DnniCache,
    /// Spectral network cache.
    pub spectral_network: DnniCache,
    /// Spectral head cache.
    pub spectral_head: DnniCache,
    /// Auxiliary network cache.
    pub auxiliary_network: DnniCache,
    /// Auxiliary head cache.
    pub auxiliary_head: DnniCache,
}

impl AcousticModelState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.phoneme_encoder.get_bytes()
            + self.context_projection.get_bytes()
            + self.context_network.get_bytes()
            + self.latent_projection.get_bytes()
            + self.latent_network.get_bytes()
            + self.latent_head.get_bytes()
            + self.spectral_network.get_bytes()
            + self.spectral_head.get_bytes()
            + self.auxiliary_network.get_bytes()
            + self.auxiliary_head.get_bytes()
    }

    /// Resets all branch caches.
    pub fn clear(&mut self) {
        self.phoneme_encoder.clear();
        self.context_projection.clear();
        self.context_network.clear();
        self.latent_projection.clear();
        self.latent_network.clear();
        self.latent_head.clear();
        self.spectral_network.clear();
        self.spectral_head.clear();
        self.auxiliary_network.clear();
        self.auxiliary_head.clear();
    }
}

/// Acoustic model: frontend, speaker/style embeddings and ten networks.
#[derive(Debug)]
pub struct AcousticModel {
    features: AcousticFeatures,
    phoneme_encoder: DnniInference,
    context_projection: DnniInference,
    context_network: DnniInference,
    latent_projection: DnniInference,
    latent_network: DnniInference,
    latent_head: DnniInference,
    spectral_network: DnniInference,
    spectral_head: DnniInference,
    auxiliary_network: DnniInference,
    auxiliary_head: DnniInference,
    language_embedding: DnniMatrix,
    speaker: Vec<f32>,
    default_style: Vec<f32>,
    vocal_mode_styles: Vec<Vec<f32>>,
    loaded: bool,
}

impl AcousticModel {
    /// Creates an unloaded model; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self {
            features: AcousticFeatures::new(),
            phoneme_encoder: DnniInference::new(),
            context_projection: DnniInference::new(),
            context_network: DnniInference::new(),
            latent_projection: DnniInference::new(),
            latent_network: DnniInference::new(),
            latent_head: DnniInference::new(),
            spectral_network: DnniInference::new(),
            spectral_head: DnniInference::new(),
            auxiliary_network: DnniInference::new(),
            auxiliary_head: DnniInference::new(),
            language_embedding: DnniMatrix {
                rows: 0,
                columns: 0,
                values: Vec::new(),
            },
            speaker: Vec::new(),
            default_style: Vec::new(),
            vocal_mode_styles: Vec::new(),
            loaded: bool::default(),
        }
    }

    /// Loads the `_rldtg0` acoustic model with the database default style.
    ///
    /// A failed load preserves the current model.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<()> {
        self.load_with_vocal_modes(reader, &[], root_node)
    }

    /// Loads the `_rldtg0` acoustic model with named vocal-mode styles.
    ///
    /// `vocal_mode_names` must pair one-to-one with the style vectors after
    /// the default (the voice metadata's timbre styles); pass an empty slice
    /// for the default style only. A failed load preserves the current model.
    #[allow(clippy::too_many_lines)]
    pub fn load_with_vocal_modes(
        &mut self,
        reader: &DnniReader,
        vocal_mode_names: &[String],
        root_node: usize,
    ) -> Result<()> {
        self.load_inner(reader, vocal_mode_names, &[], &[], root_node, None)
    }

    /// Loads the acoustic model with Flat voice overrides for the 32-channel
    /// speaker vector (`(base)` style) and default style vector (`(default)`
    /// style), plus the named document vocal modes as offsets from the
    /// default style. Empty overrides keep the model-embedded defaults,
    /// exactly as for container voices; non-empty overrides replace them
    /// before validation. `vocal_modes` iterates in sorted-name order, which
    /// matches the synthesizer's vocal-mode names from the same table, so
    /// the existing positional weight mixing in `run` matches the reference
    /// (`default + weight * offset`) exactly.
    pub fn load_with_overrides(
        &mut self,
        reader: &DnniReader,
        root_node: usize,
        speaker_override: &[f32],
        style_override: &[f32],
        vocal_modes: &BTreeMap<String, Vec<f32>>,
    ) -> Result<()> {
        self.load_inner(
            reader,
            &[],
            speaker_override,
            style_override,
            root_node,
            Some(vocal_modes),
        )
    }

    /// Shared loader: `load`/`load_with_vocal_modes` (container voices, model
    /// styles) and `load_with_overrides` (Flat voices, document styles).
    #[allow(clippy::too_many_lines)]
    fn load_inner(
        &mut self,
        reader: &DnniReader,
        vocal_mode_names: &[String],
        speaker_override: &[f32],
        style_override: &[f32],
        root_node: usize,
        document_modes: Option<&BTreeMap<String, Vec<f32>>>,
    ) -> Result<()> {
        let nodes = reader.nodes();
        let Some(root) = nodes
            .get(root_node)
            .filter(|root| root.name == "_rldtg0" && root.children.len() == 2)
        else {
            return Err(failure(
                "expected an _rldtg0 root with model and style children",
            ));
        };
        let children = root.children.clone();
        let model = &nodes[children[0]];
        let style = &nodes[children[1]];
        if model.name != "_rldms0"
            || model.children.len() != 8
            || style.name != "_stbkv1"
            || style.children.len() != 2
        {
            return Err(failure("unsupported acoustic model or style layout"));
        }
        let model_children = model.children.clone();
        let style_children = style.children.clone();
        let context = &nodes[model_children[2]];
        let latent = &nodes[model_children[3]];
        if context.name != "_vqctx1"
            || context.children.len() != 4
            || latent.name != "_didsv0"
            || latent.children.len() != 3
        {
            return Err(failure("unsupported context or latent model layout"));
        }
        let latent_payload = reader.payload(model_children[3]);
        if latent_payload.len() != 12
            || read_word(latent_payload, 0) != 4
            || read_word(latent_payload, 4) != 64
            || f32::from_bits(read_word(latent_payload, 8)) != 64.0
        {
            return Err(failure("unsupported latent feature configuration"));
        }
        let context_children = context.children.clone();
        let latent_children = latent.children.clone();
        let embedding = &nodes[context_children[3]];
        let style_vectors = &nodes[style_children[0]];
        if embedding.name != "modl4"
            || embedding.children.len() != 1
            || style_vectors.name != "cmpg1"
            || style_vectors.children.is_empty()
        {
            return Err(failure(
                "missing language embedding or default style vector",
            ));
        }

        let mut candidate = Self::new();
        candidate.features.load(reader, model_children[0])?;
        if candidate.features.config().pitch_channels != 1
            || candidate.features.config().acoustic_channels != 71
            || candidate.features.frame_interval_seconds() != FRAME_INTERVAL_SECONDS
        {
            return Err(failure(
                "unsupported acoustic feature dimensions or frame interval",
            ));
        }
        candidate.speaker = reader.read_float_vector(model_children[1])?;
        candidate.default_style = reader.read_float_vector(style_vectors.children[0])?;
        if !speaker_override.is_empty() {
            candidate.speaker = speaker_override.to_vec();
        }
        if !style_override.is_empty() {
            candidate.default_style = style_override.to_vec();
        }
        if !candidate.speaker.iter().all(|value| value.is_finite())
            || !candidate
                .default_style
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(failure("voice speaker/style embeddings must be finite"));
        }
        if !vocal_mode_names.is_empty()
            && style_vectors.children.len() - 1 != vocal_mode_names.len()
        {
            return Err(failure(
                "voice timbre-style metadata does not match the acoustic model style vectors",
            ));
        }
        candidate.vocal_mode_styles.reserve(vocal_mode_names.len());
        if let Some(document_modes) = document_modes {
            // Document styles are offsets from the default style, stored here
            // as absolute vectors in sorted-name order (the map's iteration
            // order, matching the synthesizer's vocal-mode names), so the
            // positional mixing in `run` matches the reference exactly.
            candidate.vocal_mode_styles =
                absolute_document_styles(&candidate.default_style, document_modes)?;
        } else {
            for index in 0..vocal_mode_names.len() {
                let mode_style = reader.read_float_vector(style_vectors.children[index + 1])?;
                if mode_style.len() != candidate.default_style.len() {
                    return Err(failure(
                        "a vocal-mode style vector has an unsupported dimension",
                    ));
                }
                candidate.vocal_mode_styles.push(mode_style);
            }
        }
        candidate.language_embedding = reader.read_float_matrix(embedding.children[0])?;
        if candidate.speaker.len() != 32
            || candidate.default_style.len() != 32
            || candidate.language_embedding.rows != 8
            || candidate.language_embedding.columns as usize != candidate.features.language_count()
        {
            return Err(failure(
                "unsupported speaker, style or language embedding dimensions",
            ));
        }
        let networks = [
            (&mut candidate.phoneme_encoder, context_children[0]),
            (&mut candidate.context_projection, context_children[1]),
            (&mut candidate.context_network, context_children[2]),
            (&mut candidate.latent_projection, latent_children[0]),
            (&mut candidate.latent_network, latent_children[1]),
            (&mut candidate.latent_head, latent_children[2]),
            (&mut candidate.spectral_network, model_children[4]),
            (&mut candidate.spectral_head, model_children[5]),
            (&mut candidate.auxiliary_network, model_children[6]),
            (&mut candidate.auxiliary_head, model_children[7]),
        ];
        for (network, node) in networks {
            network.load(reader, node)?;
        }
        candidate.loaded = true;
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

    /// Typed acoustic frame interval; fails when unloaded.
    pub fn frame_interval(&self) -> Result<AcousticFrameInterval> {
        if !self.loaded {
            return Err(failure("no model has been loaded"));
        }
        AcousticFrameInterval::new(self.features.frame_interval_seconds())
    }

    /// Vocal-mode style count (0 when only the default style is loaded).
    pub fn vocal_mode_count(&self) -> usize {
        self.vocal_mode_styles.len()
    }
}

impl Default for AcousticModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AcousticModel {
    /// Predicts vocoder-ready acoustic frames (72 channels per frame).
    ///
    /// `phonemes` carry per-phoneme frame counts on the acoustic grid and
    /// `log_f0` holds one natural-log Hz value per frame; `output` receives
    /// `[frames, 72]` (voiced flag, F0 Hz, spectral + auxiliary channels).
    /// `noise_seed` selects the deterministic noise stream (the C++ default
    /// is 5489) and `vocal_mode_weights` holds one weight per frame per
    /// vocal mode (empty means the database default style). `output` is
    /// only assigned on success.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn run(
        &self,
        phonemes: &[TimedPhoneme],
        log_f0: &[f32],
        output: &mut DnniTensor,
        noise_seed: u32,
        mut state: Option<&mut AcousticModelState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
        vocal_mode_weights: &[f32],
    ) -> Result<()> {
        if !self.loaded {
            return Err(failure("no model has been loaded"));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        let mut frames = 0_usize;
        for phoneme in phonemes {
            if phoneme.frame_count == 0
                || frames >= MAX_FRAMES
                || phoneme.frame_count > MAX_FRAMES - frames
            {
                return Err(failure(
                    "phoneme frame counts must be positive and fit the inference memory limit",
                ));
            }
            frames += phoneme.frame_count;
        }
        if frames == 0 || log_f0.len() != frames {
            return Err(failure(
                "one log-F0 value is required for each phoneme frame",
            ));
        }
        if !vocal_mode_weights.is_empty()
            && vocal_mode_weights.len() != frames * self.vocal_mode_styles.len()
        {
            return Err(failure(
                "vocal-mode automation dimensions do not match the phrase frame count",
            ));
        }
        if !vocal_mode_weights.iter().all(|weight| weight.is_finite()) {
            return Err(failure(
                "vocal-mode automation contains a non-finite amount",
            ));
        }
        let mut pitch = DnniTensor::default();
        self.features.normalize_log_f0(log_f0, &mut pitch)?;
        check_shape(&pitch, frames, 1, "pitch normalization")?;

        let mut frame_features = DnniTensor::default();
        {
            let mut context_state = None;
            if let Some(state) = state.as_mut() {
                context_state = Some(&mut **state);
            }
            self.make_context(
                phonemes,
                &pitch,
                &mut frame_features,
                context_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
                vocal_mode_weights,
            )?;
        }
        let mut context = DnniTensor::default();
        let mut projected = DnniTensor::default();
        {
            let (mut network_state, mut projection_state) = (None, None);
            if let Some(state) = state.as_mut() {
                network_state = Some(&mut state.context_network);
                projection_state = Some(&mut state.context_projection);
            }
            self.context_network.run_with_cache(
                &frame_features,
                &mut context,
                None,
                network_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
            self.context_projection.run_with_cache(
                &frame_features,
                &mut projected,
                None,
                projection_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
        }
        check_shape(&context, frames, 256, "context network")?;
        check_shape(&projected, frames, 256, "context projection")?;

        let mut latent = DnniTensor::default();
        {
            let mut latent_state = None;
            if let Some(state) = state.as_mut() {
                latent_state = Some(&mut **state);
            }
            self.sample_latent(
                &context,
                &projected,
                noise_seed,
                &mut latent,
                latent_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
        }
        check_shape(&latent, frames, LATENT_CHANNELS, "latent head")?;

        let mut spectral_hidden = DnniTensor::default();
        let mut spectrum = DnniTensor::default();
        {
            let (mut network_state, mut head_state) = (None, None);
            if let Some(state) = state.as_mut() {
                network_state = Some(&mut state.spectral_network);
                head_state = Some(&mut state.spectral_head);
            }
            self.spectral_network.run_with_cache(
                &context,
                &mut spectral_hidden,
                Some(&latent),
                network_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
            self.spectral_head.run_with_cache(
                &spectral_hidden,
                &mut spectrum,
                None,
                head_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
        }
        check_shape(&spectrum, frames, 64, "64-channel head")?;

        let mut auxiliary_hidden = DnniTensor::default();
        let mut auxiliary = DnniTensor::default();
        {
            let (mut network_state, mut head_state) = (None, None);
            if let Some(state) = state.as_mut() {
                network_state = Some(&mut state.auxiliary_network);
                head_state = Some(&mut state.auxiliary_head);
            }
            self.auxiliary_network.run_with_cache(
                &context,
                &mut auxiliary_hidden,
                Some(&spectrum),
                network_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
            self.auxiliary_head.run_with_cache(
                &auxiliary_hidden,
                &mut auxiliary,
                None,
                head_state,
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
        }
        check_shape(&auxiliary, frames, 7, "7-channel head")?;

        // Interleave: 2 auxiliary, 64 spectral, 5 auxiliary.
        let mut normalized = DnniTensor {
            frames,
            channels: 71,
            values: vec![0.0; frames * 71],
        };
        for frame in 0..frames {
            let destination = &mut normalized.values[frame * 71..(frame + 1) * 71];
            let auxiliary_source = &auxiliary.values[frame * 7..(frame + 1) * 7];
            let spectrum_source = &spectrum.values[frame * 64..(frame + 1) * 64];
            destination[..2].copy_from_slice(&auxiliary_source[..2]);
            destination[2..66].copy_from_slice(spectrum_source);
            destination[66..].copy_from_slice(&auxiliary_source[2..]);
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        self.features
            .decode_acoustic_71(&normalized, log_f0, output)
    }

    /// Expands phoneme encodings onto the frame grid with position, speaker,
    /// language and style conditioning (144 channels per frame).
    #[allow(clippy::too_many_arguments)]
    fn make_context(
        &self,
        phonemes: &[TimedPhoneme],
        pitch: &DnniTensor,
        output: &mut DnniTensor,
        mut state: Option<&mut AcousticModelState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
        vocal_mode_weights: &[f32],
    ) -> Result<()> {
        let mut phone_features = DnniTensor::default();
        self.features
            .encode_phonemes(phonemes, &mut phone_features)?;
        let mut encoded = DnniTensor::default();
        self.phoneme_encoder.run_with_cache(
            &phone_features,
            &mut encoded,
            None,
            state.as_mut().map(|state| &mut state.phoneme_encoder),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&encoded, phonemes.len(), 64, "phoneme encoder")?;

        let frames = pitch.frames;
        let mut context = DnniTensor {
            frames,
            channels: CONTEXT_CHANNELS,
            values: vec![0.0; frames * CONTEXT_CHANNELS],
        };
        let mut frame = 0_usize;
        for (phone, phoneme) in phonemes.iter().enumerate() {
            let language = self.features.find_language_index(&phoneme.language)?;
            if language >= self.language_embedding.columns as usize {
                return Err(failure(
                    "language embedding index exceeds the stored matrix",
                ));
            }
            for within_phone in 0..phoneme.frame_count {
                let destination =
                    &mut context.values[frame * CONTEXT_CHANNELS..(frame + 1) * CONTEXT_CHANNELS];
                destination[..64].copy_from_slice(&encoded.values[phone * 64..(phone + 1) * 64]);
                let position = if phoneme.frame_count == 1 {
                    0.5
                } else {
                    within_phone as f32 / (phoneme.frame_count - 1) as f32
                };
                let scalars = [position, pitch.values[frame]];
                for (scalar_index, scalar) in scalars.iter().enumerate() {
                    let angle = scalar * 6.283_185_5_f32;
                    for (phase_index, phase) in COSINE_PHASES.iter().enumerate() {
                        // `cosf` in double, then the half-scale in double,
                        // narrowed once — exactly as the C++ expression.
                        let cosine = f64::from((angle + phase).cos());
                        destination[64 + scalar_index * 4 + phase_index] =
                            (cosine * 0.5 + 0.5) as f32;
                    }
                }
                destination[72..104].copy_from_slice(&self.speaker);
                for channel in 0..8 {
                    destination[104 + channel] = self.language_embedding.values
                        [channel * self.language_embedding.columns as usize + language];
                }
                let style = &mut destination[112..144];
                style.copy_from_slice(&self.default_style);
                for (mode, mode_style) in self.vocal_mode_styles.iter().enumerate() {
                    let weight = if vocal_mode_weights.is_empty() {
                        0.0
                    } else {
                        vocal_mode_weights[frame * self.vocal_mode_styles.len() + mode]
                    };
                    for (channel, slot) in style.iter_mut().enumerate() {
                        *slot += weight * (mode_style[channel] - self.default_style[channel]);
                    }
                }
                frame += 1;
            }
        }
        *output = context;
        Ok(())
    }

    /// Samples the 4-channel latent from the projected context plus diffusion
    /// controls and fresh Gaussian noise (one stream per run).
    #[allow(clippy::too_many_arguments)]
    fn sample_latent(
        &self,
        context: &DnniTensor,
        projected: &DnniTensor,
        noise_seed: u32,
        output: &mut DnniTensor,
        mut state: Option<&mut AcousticModelState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        let mut latent_input = DnniTensor::default();
        self.latent_projection.run_with_cache(
            projected,
            &mut latent_input,
            None,
            state.as_mut().map(|state| &mut state.latent_projection),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(
            &latent_input,
            context.frames,
            LATENT_CHANNELS,
            "latent projection",
        )?;

        // Editor normal-take diffusion controls (the low-level empty-queue
        // defaults are not the editor defaults).
        let mut controls = [0.0_f32; 128];
        for index in 0..32 {
            let frequency = ((index as f32) * (-26.575_424_f32 / 64.0)).exp2();
            let diffusion_angle = 64.0 * DIFFUSION_TEMPERATURE * frequency;
            let guidance_angle = 64.0 * GUIDANCE * frequency;
            controls[index] = diffusion_angle.sin();
            controls[index + 32] = diffusion_angle.cos();
            controls[index + 64] = guidance_angle.sin();
            controls[index + 96] = guidance_angle.cos();
        }
        // This seed selects the reproducible noise sequence, not a retake.
        let mut noise = GaussianNoise::new(noise_seed);
        let mut condition = DnniTensor {
            frames: context.frames,
            channels: CONDITION_CHANNELS,
            values: vec![0.0; context.frames * CONDITION_CHANNELS],
        };
        for frame in 0..context.frames {
            let destination =
                &mut condition.values[frame * CONDITION_CHANNELS..(frame + 1) * CONDITION_CHANNELS];
            destination[..256].copy_from_slice(&context.values[frame * 256..(frame + 1) * 256]);
            destination[256..384].copy_from_slice(&controls);
            for channel in 0..LATENT_CHANNELS {
                destination[384 + channel] = noise.next() * SOURCE_TEMPERATURE;
            }
        }
        let mut hidden = DnniTensor::default();
        self.latent_network.run_with_cache(
            &latent_input,
            &mut hidden,
            Some(&condition),
            state.as_mut().map(|state| &mut state.latent_network),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        check_shape(&hidden, context.frames, 128, "latent network")?;
        Ok(self.latent_head.run_with_cache(
            &hidden,
            output,
            None,
            state.as_mut().map(|state| &mut state.latent_head),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_dnni::DnniReader;

    #[test]
    fn cosine_constants_match_the_cpp_float_bits() {
        // The truncated literals must stay bitwise identical to the C++
        // `f` suffixed decimals (verified against the IEEE-754 patterns).
        assert_eq!(super::COSINE_PHASES[0].to_bits(), 0x3FC9_0FDB);
        assert_eq!(super::COSINE_PHASES[2].to_bits(), 0xBFC9_0FDB);
        assert_eq!(super::COSINE_PHASES[3].to_bits(), 0xC049_0FDB);
        assert_eq!((6.283_185_5_f32).to_bits(), 0x40C9_0FDB);
        assert_eq!((-26.575_424_f32).to_bits(), 0xC1D4_9A78);
    }

    #[test]
    fn document_styles_resolve_to_absolute_vectors_in_name_order() {
        let modes: BTreeMap<String, Vec<f32>> = [
            ("Joyful".to_owned(), vec![1.0, 0.0]),
            ("Cute".to_owned(), vec![0.0, 2.0]),
        ]
        .into_iter()
        .collect();
        // Sorted-name order (Cute, Joyful), each offset added onto default.
        assert_eq!(
            absolute_document_styles(&[10.0, 20.0], &modes).unwrap(),
            vec![vec![10.0, 22.0], vec![11.0, 20.0]]
        );
        assert_eq!(
            absolute_document_styles(&[10.0, 20.0], &BTreeMap::new())
                .unwrap()
                .len(),
            0
        );
        // Wrong-dimension offsets are errors, never panics.
        let bad: BTreeMap<String, Vec<f32>> =
            [("Cute".to_owned(), vec![1.0])].into_iter().collect();
        assert_eq!(
            absolute_document_styles(&[10.0, 20.0], &bad)
                .unwrap_err()
                .to_string(),
            "Acoustic model: a vocal-mode style vector has an unsupported dimension"
        );
    }

    #[test]
    fn typed_interval_rejects_non_positive_grids() {
        assert_eq!(
            AcousticFrameInterval::new(0.005).expect("valid").seconds(),
            0.005
        );
        for bad in [0.0, -0.005, f32::INFINITY, f32::NAN] {
            assert_eq!(
                AcousticFrameInterval::new(bad).unwrap_err().to_string(),
                "Acoustic model: invalid frame interval."
            );
        }
    }

    #[test]
    fn unloaded_model_rejects_run() {
        let mut output = DnniTensor::default();
        assert_eq!(
            AcousticModel::new()
                .run(
                    &[],
                    &[],
                    &mut output,
                    5489,
                    None,
                    &CancelToken::new(),
                    None,
                    &[],
                )
                .unwrap_err()
                .to_string(),
            "Acoustic model: no model has been loaded"
        );
        assert_eq!(
            AcousticModel::new()
                .frame_interval()
                .unwrap_err()
                .to_string(),
            "Acoustic model: no model has been loaded"
        );
        assert_eq!(AcousticModel::new().frame_interval_seconds(), 0.0);
    }

    #[test]
    fn loader_rejects_a_non_acoustic_root() {
        let reader = DnniReader::from_bytes(vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .expect("blob must parse");
        assert_eq!(
            AcousticModel::new()
                .load(&reader, 0)
                .unwrap_err()
                .to_string(),
            "Acoustic model: expected an _rldtg0 root with model and style children"
        );
    }

    #[test]
    fn fresh_state_holds_no_bytes() {
        assert_eq!(AcousticModelState::default().get_bytes(), 0);
        let mut state = AcousticModelState::default();
        state.clear();
        assert_eq!(state.get_bytes(), 0);
    }
}
