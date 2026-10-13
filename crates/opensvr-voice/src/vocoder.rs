//! Neural vocoder (`NeuralVocoder`): waveform synthesis from acoustic frames.
//!
//! Ports OpenSV's `src/synthesis/NeuralVocoder.{h,cpp}` (namespace
//! `sv::synthesis`): the 72-channel raw acoustic frames (voiced flag, F0 in
//! Hz, spectral + auxiliary features) are log-converted, normalized and run
//! through the shape/condition/modulation networks; the LF excitation,
//! periodic modulation and subband residual are rendered and shaped by the
//! three spectral filters into waveform samples plus the periodic/aperiodic
//! decomposition.

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniCache, DnniInference, DnniReader, DnniRunStatistics, DnniTensor};

use crate::excitation::{
    PeriodicExcitationFrame, render_periodic_excitation, render_periodic_modulation,
};
use crate::filter_bank::VocoderFilterBank;
use crate::normalizer::FeatureNormalizer;
use crate::residual::{VocoderResidual, VocoderResidualState};
use crate::spectral_heads::{VocoderSpectralHeads, VocoderSpectralHeadsState};
use crate::{Result, invalid};

/// The twelve vocoder-root aliases registered for the factory.
const VOCODER_TYPE_IDS: [u64; 12] = [
    0x6DFB_6790_9B04_14A1,
    0xBC0A_CD42_0E51_B743,
    0x1FD1_7498_26B0_CAEC,
    0x88F2_759E_BD08_5477,
    0x9B00_0CB7_74E2_A953,
    0x44D4_CFAB_EA5D_D27C,
    0x7963_F08A_BEB9_8D85,
    0x950E_9ACA_274E_E27D,
    0xA919_046C_636D_B7B3,
    0x7E6D_CC7C_7F9F_8224,
    0x9BC6_6704_6629_3322,
    0xB244_CC86_26A0_5CAE,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Neural vocoder: {}", reason.into()))
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_float(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn has_shape(tensor: &DnniTensor, frames: usize, channels: usize) -> bool {
    tensor.frames == frames
        && tensor.channels == channels
        && frames <= usize::MAX / channels.max(1)
        && tensor.values.len() == frames * channels
}

/// Waveform output: samples at the native sample rate plus the
/// periodic/aperiodic decomposition.
#[derive(Debug, Clone, Default)]
pub struct NeuralVocoderOutput {
    /// Native sample rate in Hz.
    pub sample_rate: f64,
    /// Samples per acoustic frame.
    pub hop_samples: usize,
    /// Mixed waveform samples.
    pub samples: Vec<f32>,
    /// Periodic component (filters 0 + 2).
    pub periodic: Vec<f32>,
    /// Aperiodic component (filter 1 output).
    pub aperiodic: Vec<f32>,
}

/// Retained inference snapshots for phrase-level incremental caching.
#[derive(Debug, Default)]
pub struct NeuralVocoderState {
    /// Shape-network cache.
    pub shape_network: DnniCache,
    /// Condition-network cache.
    pub condition_network: DnniCache,
    /// Modulation-network cache.
    pub modulation_network: DnniCache,
    /// Spectral-head caches.
    pub spectral_heads: VocoderSpectralHeadsState,
    /// Residual caches.
    pub residual: VocoderResidualState,
}

impl NeuralVocoderState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.shape_network.get_bytes()
            + self.condition_network.get_bytes()
            + self.modulation_network.get_bytes()
            + self.spectral_heads.get_bytes()
            + self.residual.get_bytes()
    }

    /// Resets all branch caches.
    pub fn clear(&mut self) {
        self.shape_network.clear();
        self.condition_network.clear();
        self.modulation_network.clear();
        self.spectral_heads.clear();
        self.residual.clear();
    }
}

/// Vocoder: acoustic frames in, waveform samples out.
#[derive(Debug, Default)]
pub struct NeuralVocoder {
    input_normalizer: FeatureNormalizer,
    shape_network: DnniInference,
    condition_network: DnniInference,
    modulation_network: DnniInference,
    spectral_heads: VocoderSpectralHeads,
    filters: [VocoderFilterBank; 3],
    residual: VocoderResidual,
    modulation_ratios: Vec<f32>,
    sample_rate: f64,
    frame_period_seconds: f64,
    hop_samples: usize,
    excitation_gain: f32,
    modulation_bias: f32,
    force_voiced_features: bool,
    loaded: bool,
}

impl NeuralVocoder {
    /// Creates an unloaded vocoder; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the vocoder model from the DNNI root node.
    ///
    /// A failed load preserves the current vocoder.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes.get(root_node) else {
            return Err(failure("root node is out of range"));
        };
        if !VOCODER_TYPE_IDS.contains(&node.type_id) {
            return Err(failure(format!(
                "unsupported root type at offset {}",
                node.offset
            )));
        }
        let children = node.children.clone();
        let payload = reader.payload(root_node);
        if payload.len() != 18 || children.len() != 13 || payload[12] & 1 == 0 {
            return Err(failure(
                "expected the 13-child architecture with a shape network",
            ));
        }
        let configuration_node = children[0];
        let Some(configuration) = nodes.get(configuration_node) else {
            return Err(failure("invalid _vocfv1/v2 feature configuration"));
        };
        let configuration_bytes = reader.payload(configuration_node);
        let feature_version2 = configuration.name == "_vocfv2";
        if (configuration.name != "_vocfv1" && !feature_version2)
            || configuration_bytes.len() != if feature_version2 { 52 } else { 48 }
            || configuration.children.len() < 4
        {
            return Err(failure("invalid _vocfv1/v2 feature configuration"));
        }
        if read_word(configuration_bytes, 0) != 72
            || read_word(configuration_bytes, 16) != 72
            || read_word(configuration_bytes, 32) != 1
        {
            return Err(failure("unsupported feature dimensions or F0 expansion"));
        }
        let configuration_children = configuration.children.clone();
        let mut candidate = Self::new();
        // _vocfv2 adds the input voicing override; the first twelve fields
        // retain the _vocfv1 layout.
        candidate.force_voiced_features =
            feature_version2 && read_word(configuration_bytes, 48) != 0;
        candidate.sample_rate = f64::from(read_word(configuration_bytes, 8));
        candidate.frame_period_seconds = f64::from(read_float(configuration_bytes, 12));
        let hop = (candidate.sample_rate * candidate.frame_period_seconds).round();
        candidate.excitation_gain = read_float(payload, 4).exp();
        candidate.modulation_bias = read_float(payload, 14);
        if !candidate.sample_rate.is_finite()
            || candidate.sample_rate <= 0.0
            || !candidate.frame_period_seconds.is_finite()
            || candidate.frame_period_seconds <= 0.0
            || !hop.is_finite()
            || hop < 1.0
            || hop > f64::from(i32::MAX)
            || !candidate.excitation_gain.is_finite()
            || candidate.excitation_gain <= 0.0
            || !candidate.modulation_bias.is_finite()
        {
            return Err(failure("invalid sample timing or gain configuration"));
        }
        candidate.hop_samples = hop as usize;
        candidate
            .input_normalizer
            .load(reader, configuration_children[0])?;
        if candidate.input_normalizer.channel_count() != 72 {
            return Err(failure("input normalization must have 72 channels"));
        }
        candidate.shape_network.load(reader, children[1])?;
        candidate.condition_network.load(reader, children[2])?;
        candidate.spectral_heads.load(reader, children[3])?;
        candidate.modulation_ratios = reader.read_float_vector(children[4])?;
        if candidate.modulation_ratios.is_empty()
            || !candidate
                .modulation_ratios
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(failure("invalid modulation frequency ratios"));
        }
        candidate.modulation_network.load(reader, children[5])?;
        for index in 0..3 {
            candidate.filters[index].load(reader, children[6 + index], candidate.hop_samples)?;
        }
        candidate.residual.load(reader, root_node)?;
        candidate.loaded = true;
        *self = candidate;
        Ok(())
    }

    /// Renders waveform samples from raw acoustic frames.
    ///
    /// `input` is a complete sequence of raw acoustic frames (before log-F0
    /// and normalization); each call starts fresh DSP state. Optional
    /// `shape_control` holds one value per frame. `output` is only assigned
    /// on success.
    #[allow(clippy::too_many_lines)]
    pub fn run(
        &self,
        input: &DnniTensor,
        output: &mut NeuralVocoderOutput,
        shape_control: &[f32],
        mut state: Option<&mut NeuralVocoderState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        if !self.loaded {
            return Err(failure("no model loaded"));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        if input.frames == 0
            || !has_shape(input, input.frames, 72)
            || input.frames > usize::MAX / self.hop_samples.max(1)
        {
            return Err(failure(
                "expected a nonempty frame-major tensor with 72 channels",
            ));
        }
        if (!shape_control.is_empty() && shape_control.len() != input.frames)
            || !shape_control.iter().all(|value| value.is_finite())
        {
            return Err(failure(
                "shape control must be empty or contain one finite value per frame",
            ));
        }
        if !input.values.iter().all(|value| value.is_finite()) {
            return Err(failure("input contains nonfinite values"));
        }
        let mut logarithmic = input.clone();
        for frame in 0..input.frames {
            let frequency = input.values[frame * 72 + 1];
            if frequency <= 0.0 {
                return Err(failure("F0 must be positive even for unvoiced frames"));
            }
            if self.force_voiced_features {
                logarithmic.values[frame * 72] = 1.0;
            }
            logarithmic.values[frame * 72 + 1] = frequency.ln();
        }
        let mut normalized = DnniTensor::default();
        self.input_normalizer
            .normalize(&logarithmic, &mut normalized, true)?;
        let mut shape = DnniTensor::default();
        self.shape_network.run_with_cache(
            &normalized,
            &mut shape,
            None,
            state.as_mut().map(|state| &mut state.shape_network),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if !has_shape(&shape, input.frames, 1) {
            return Err(failure(
                "shape network output does not align with input frames",
            ));
        }
        let mut source_frames = Vec::with_capacity(input.frames);
        for frame in 0..input.frames {
            // The shape head emits a logit; a larger value must increase Rd.
            let mut source_shape = 2.9 / ((-shape.values[frame]).exp() + 1.0) + 0.1;
            if !shape_control.is_empty() {
                let control = shape_control[frame];
                source_shape += control
                    * if control <= 0.0 {
                        source_shape - 0.1
                    } else {
                        3.0 - source_shape
                    };
            }
            source_frames.push(PeriodicExcitationFrame {
                fundamental_frequency_hz: input.values[frame * 72 + 1],
                shape: source_shape,
                voiced: self.force_voiced_features || input.values[frame * 72] >= 0.5,
            });
        }
        let mut condition = DnniTensor::default();
        self.condition_network.run_with_cache(
            &normalized,
            &mut condition,
            None,
            state.as_mut().map(|state| &mut state.condition_network),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        let mut spectra = [
            DnniTensor::default(),
            DnniTensor::default(),
            DnniTensor::default(),
        ];
        self.spectral_heads.run(
            &condition,
            &mut spectra,
            state.as_mut().map(|state| &mut state.spectral_heads),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        let mut modulation_parameters = DnniTensor::default();
        self.modulation_network.run_with_cache(
            &condition,
            &mut modulation_parameters,
            None,
            state.as_mut().map(|state| &mut state.modulation_network),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if self.modulation_ratios.len() > usize::MAX / 2
            || !has_shape(
                &modulation_parameters,
                input.frames,
                self.modulation_ratios.len() * 2,
            )
        {
            return Err(failure(
                "modulation network output does not align with input frames",
            ));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        let mut excitation = Vec::new();
        let mut phase_cycles = Vec::new();
        render_periodic_excitation(
            &source_frames,
            self.hop_samples,
            self.frame_period_seconds,
            self.excitation_gain,
            &mut excitation,
            &mut phase_cycles,
        )?;
        let mut modulation = Vec::new();
        render_periodic_modulation(
            &source_frames,
            &phase_cycles,
            &self.modulation_ratios,
            &modulation_parameters.values,
            self.hop_samples,
            self.frame_period_seconds,
            &mut modulation,
        )?;
        let sample_count = input.frames * self.hop_samples;
        if excitation.len() != sample_count || modulation.len() != sample_count {
            return Err(failure(
                "periodic source length does not match frame timing",
            ));
        }
        for (slot, driver) in modulation.iter_mut().zip(&excitation) {
            *slot = driver * (*slot + self.modulation_bias);
        }
        let mut residual_excitation = Vec::new();
        self.residual.render(
            &normalized,
            &excitation,
            &mut residual_excitation,
            state.as_mut().map(|state| &mut state.residual),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        let mut components = [Vec::new(), Vec::new(), Vec::new()];
        self.filters[0].render(&spectra[0], &excitation, &mut components[0])?;
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        self.filters[1].render(&spectra[1], &residual_excitation, &mut components[1])?;
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        self.filters[2].render(&spectra[2], &modulation, &mut components[2])?;
        for component in &components {
            if component.len() != sample_count {
                return Err(failure(
                    "filtered output length does not match frame timing",
                ));
            }
        }
        let mut candidate = NeuralVocoderOutput {
            sample_rate: self.sample_rate,
            hop_samples: self.hop_samples,
            samples: Vec::with_capacity(sample_count),
            periodic: Vec::with_capacity(sample_count),
            aperiodic: components[1].clone(),
        };
        for ((&c0, &c1), &c2) in components[0].iter().zip(&components[1]).zip(&components[2]) {
            candidate.periodic.push(c0 + c2);
            candidate.samples.push(c0 + c1 + c2);
        }
        for (sample, periodic) in candidate.samples.iter().zip(&candidate.periodic) {
            if !sample.is_finite() || !periodic.is_finite() {
                return Err(failure("nonfinite PCM generated by the model"));
            }
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        *output = candidate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_vocoder_rejects_run() {
        let mut output = NeuralVocoderOutput::default();
        assert_eq!(
            NeuralVocoder::new()
                .run(
                    &DnniTensor::default(),
                    &mut output,
                    &[],
                    None,
                    &CancelToken::new(),
                    None,
                )
                .unwrap_err()
                .to_string(),
            "Neural vocoder: no model loaded"
        );
    }

    #[test]
    fn loader_rejects_a_non_vocoder_root() {
        let bytes = vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let reader = DnniReader::from_bytes(bytes).expect("blob must parse");
        assert_eq!(
            NeuralVocoder::new()
                .load(&reader, 0)
                .unwrap_err()
                .to_string(),
            "Neural vocoder: unsupported root type at offset 8"
        );
    }

    #[test]
    fn fresh_state_holds_no_bytes() {
        assert_eq!(NeuralVocoderState::default().get_bytes(), 0);
        let mut state = NeuralVocoderState::default();
        state.clear();
        assert_eq!(state.get_bytes(), 0);
    }
}
