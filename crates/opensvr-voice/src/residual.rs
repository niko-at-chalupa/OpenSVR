//! Subband residual-noise renderer (`VocoderResidual`).
//!
//! Ports OpenSV's `src/synthesis/VocoderResidual.{h,cpp}` (namespace
//! `sv::synthesis`): the periodic excitation is analysed into subbands,
//! joined with gated Gaussian noise and the interpolated normalized
//! conditions, transformed by two networks, gated, and resynthesized.
//!
//! Noise note: this stage does NOT use the shared `mt19937` stream. It draws
//! from a `std::minstd_rand` engine seeded at 1, fresh on every render,
//! through the simpler rejection loop below — ported exactly, never
//! substituted.

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniCache, DnniInference, DnniReader, DnniRunStatistics, DnniTensor};

use crate::{Result, invalid};

const MAXIMUM_VALUES: usize = 64 * 1024 * 1024;

/// The twelve vocoder-root aliases shared with the neural vocoder.
const VOCODER_TYPE_IDS: [u64; 12] = [
    0x6dfb_6790_9b04_14a1,
    0xbc0a_cd42_0e51_b743,
    0x1fd1_7498_26b0_caec,
    0x88f2_759e_bd08_5477,
    0x9b00_0cb7_74e2_a953,
    0x44d4_cfab_ea5d_d27c,
    0x7963_f08a_beb9_8d85,
    0x950e_9aca_274e_e27d,
    0xa919_046c_636d_b7b3,
    0x7e6d_cc7c_7f9f_8224,
    0x9bc6_6704_6629_3322,
    0xb244_cc86_26a0_5cae,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!(
        "Vocoder residual: {reason}",
        reason = reason.into()
    ))
}

fn read_word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// `std::minstd_rand`: `(48271 * x) mod (2^31 - 1)`.
#[derive(Debug, Clone)]
struct MinstdRand(u32);

impl MinstdRand {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u32 {
        self.0 = ((u64::from(self.0) * 48271) % 2_147_483_647) as u32;
        self.0
    }
}

/// Deterministic Gaussian excitation over the minstd stream.
///
/// Fresh per render (seed 1); a cached spare is served first, exactly as the
/// C++ member does.
#[derive(Debug)]
struct GaussianExcitation {
    generator: MinstdRand,
    spare: f32,
    has_spare: bool,
}

impl GaussianExcitation {
    fn new() -> Self {
        Self {
            generator: MinstdRand::new(1),
            spare: 0.0,
            has_spare: false,
        }
    }

    fn next(&mut self) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            // `0x1p-30f`: exact via bits (hex float literals are unstable);
            // 2^-30 has exponent 97 (0x61), i.e. bits 0x30800000.
            let first =
                self.generator.next().wrapping_sub(1) as f32 * f32::from_bits(0x3080_0000) - 1.0;
            let second =
                self.generator.next().wrapping_sub(1) as f32 * f32::from_bits(0x3080_0000) - 1.0;
            let radius_squared = second * second + first * first;
            if radius_squared > 1.0 || radius_squared == 0.0 {
                continue;
            }
            let scale = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
            self.spare = second * scale;
            self.has_spare = true;
            return first * scale;
        }
    }
}

fn load_filters(
    reader: &DnniReader,
    node_index: usize,
    expected_type: &str,
    bands: usize,
    filters: &mut Vec<Vec<f32>>,
) -> Result<()> {
    let nodes = reader.nodes();
    let Some(node) = nodes.get(node_index) else {
        return Err(failure("filter-bank node is out of range."));
    };
    if node.name != expected_type || node.payload_size != 0 || node.children.len() != bands {
        return Err(failure(
            "resampler must contain one coefficient vector per band.",
        ));
    }
    let children = node.children.clone();
    let mut replacement: Vec<Vec<f32>> = Vec::with_capacity(bands);
    for child in children {
        let values = reader.read_float_vector(child)?;
        if values.len() < 3
            || values.len() > 4097
            || values.len() % 2 == 0
            || (!replacement.is_empty() && values.len() != replacement[0].len())
            || !values.iter().all(|value| value.is_finite())
        {
            return Err(failure(
                "expected finite, equal-length odd FIR coefficient vectors.",
            ));
        }
        replacement.push(values);
    }
    *filters = replacement;
    Ok(())
}

/// The `usize as i64` conversions below mirror the C++ `int64_t` history
/// arithmetic; all operands are validated far below the 63-bit range.
#[allow(clippy::cast_possible_wrap)]
fn analyse_excitation(
    input: &[f32],
    filters: &[Vec<f32>],
    expected_frames: usize,
) -> Result<DnniTensor> {
    let bands = filters.len();
    let filter_length = filters[0].len();
    let tail_samples = filter_length / 2;
    let discarded_frames = bands / 2 - 1;
    let first_sample = bands - 1 - tail_samples % bands;
    let sample_count = input.len() + tail_samples;
    let emitted_frames = if sample_count > first_sample {
        1 + (sample_count - first_sample - 1) / bands
    } else {
        0
    };
    if emitted_frames != expected_frames + discarded_frames + 1 {
        return Err(failure(
            "analysis-filter timing does not match the model's subband frames.",
        ));
    }
    let mut noise = GaussianExcitation::new();
    let mut values = vec![0.0f32; expected_frames * bands * 2];
    // The analysis operator starts with half a filter of zero history. Its
    // caller discards the first bands/2-1 frames and the final frame.
    for frame in 0..expected_frames {
        let sample = first_sample + (frame + discarded_frames) * bands;
        let history_start = sample as i64 - (filter_length - 1) as i64;
        let coefficient_begin = (0_i64.max(-history_start)) as usize;
        let coefficient_end =
            ((filter_length as i64).min(input.len() as i64 - history_start)) as usize;
        let source_begin = (history_start + coefficient_begin as i64) as usize;
        let destination = &mut values[frame * bands * 2..(frame + 1) * bands * 2];
        // Each band accumulates independently in the original tap order
        // (the C++ lane packing does not reorder the per-band sums).
        for band in 0..bands {
            let mut sum = 0.0f32;
            for coefficient in coefficient_begin..coefficient_end {
                sum += filters[band][coefficient]
                    * input[source_begin + (coefficient - coefficient_begin)];
            }
            destination[band] = sum;
        }
        for band in 0..bands {
            destination[bands + band] = noise.next();
        }
    }
    Ok(DnniTensor {
        frames: expected_frames,
        channels: bands * 2,
        values,
    })
}

fn interpolate_conditions(input: &DnniTensor, factor: usize) -> DnniTensor {
    let mut values = vec![0.0f32; input.frames * factor * input.channels];
    let leading_frames = factor / 2;
    let trailing_frames = factor - leading_frames;
    let offset = if factor % 2 == 0 { 0.5 } else { 0.0 };
    let inverse_factor = 1.0 / factor as f32;
    let mut destination = 0;
    for _ in 0..leading_frames {
        values[destination..destination + input.channels]
            .copy_from_slice(&input.values[..input.channels]);
        destination += input.channels;
    }
    for frame in 1..input.frames {
        for step in 0..factor {
            let position = step as f32 + offset;
            for channel in 0..input.channels {
                let previous = input.values[(frame - 1) * input.channels + channel];
                let current = input.values[frame * input.channels + channel];
                values[destination] = (current - previous) * position * inverse_factor + previous;
                destination += 1;
            }
        }
    }
    for _ in 0..trailing_frames {
        let start = input.values.len() - input.channels;
        values[destination..destination + input.channels].copy_from_slice(&input.values[start..]);
        destination += input.channels;
    }
    DnniTensor {
        frames: input.frames * factor,
        channels: input.channels,
        values,
    }
}

fn synthesise_subbands(input: &DnniTensor, filters: &[Vec<f32>]) -> Vec<f32> {
    let filter_length = filters[0].len();
    let latency_samples = (filter_length - 1) / 2;
    let mut ring = vec![0.0f32; 512.max(filter_length)];
    let mut output = Vec::with_capacity(input.frames * input.channels);
    let mut position = 0_usize;
    let mut pending_latency = 0_usize;
    let emit = |ring: &mut Vec<f32>, position: &mut usize, output: &mut Vec<f32>| {
        output.push(ring[*position]);
        ring[*position] = 0.0;
        *position += 1;
        if *position == ring.len() {
            *position = 0;
        }
    };
    for frame in 0..input.frames {
        let first_segment = filter_length.min(ring.len() - position);
        let second_segment = filter_length - first_segment;
        let row = &input.values[frame * input.channels..(frame + 1) * input.channels];
        for (filter, &amplitude) in filters.iter().zip(row) {
            for index in 0..first_segment {
                ring[position + index] += filter[index] * amplitude;
            }
            for index in 0..second_segment {
                ring[index] += filter[first_segment + index] * amplitude;
            }
        }
        for _ in 0..input.channels {
            // The reference increments this initial delay without advancing
            // the accumulation cursor; retaining that boundary behavior is
            // necessary for the serialized synthesis filters.
            if pending_latency < latency_samples {
                pending_latency += 1;
            } else {
                emit(&mut ring, &mut position, &mut output);
            }
        }
    }
    while pending_latency > 0 {
        emit(&mut ring, &mut position, &mut output);
        pending_latency -= 1;
    }
    output
}

/// Retained inference caches for one phrase.
#[derive(Debug, Default)]
pub struct VocoderResidualState {
    /// Subband network cache.
    pub network: DnniCache,
    /// Subband projection cache.
    pub projection: DnniCache,
}

impl VocoderResidualState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.network.get_bytes() + self.projection.get_bytes()
    }

    /// Resets both caches.
    pub fn clear(&mut self) {
        self.network.clear();
        self.projection.clear();
    }
}

/// Subband residual-noise renderer driven by two networks and FIR banks.
#[derive(Debug, Default)]
pub struct VocoderResidual {
    network: DnniInference,
    projection: DnniInference,
    analysis_filters: Vec<Vec<f32>>,
    synthesis_filters: Vec<Vec<f32>>,
    hop_samples: usize,
    bands: usize,
    feature_channels: usize,
    gate_noise: bool,
    loaded: bool,
}

impl VocoderResidual {
    /// Creates an unloaded renderer; [`render`](Self::render) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the residual banks and networks from the vocoder root node.
    ///
    /// A failed load preserves the current renderer.
    pub fn load(&mut self, reader: &DnniReader, vocoder_root: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(root) = nodes
            .get(vocoder_root)
            .filter(|root| VOCODER_TYPE_IDS.contains(&root.type_id))
        else {
            return Err(failure(
                "root node is out of range or has an unsupported type.",
            ));
        };
        let root_payload = reader.payload(vocoder_root);
        if root_payload.len() != 18 || root_payload[12] > 1 || root_payload[13] > 1 {
            return Err(failure("unsupported vocoder root configuration."));
        }
        let first_network = if root_payload[12] != 0 { 2 } else { 1 };
        let children = root.children.clone();
        if children.len() != first_network + 11 {
            return Err(failure("vocoder root has an unexpected child count."));
        }
        let configuration = reader.payload(children[0]);
        let configuration_type = nodes[children[0]].name.clone();
        let feature_version2 = configuration_type == "_vocfv2";
        if (configuration_type != "_vocfv1" && !feature_version2)
            || configuration.len() != if feature_version2 { 52 } else { 48 }
        {
            return Err(failure("missing vocoder feature configuration."));
        }
        let band_count = read_word(root_payload, 0) as usize;
        let channels = read_word(configuration, 0);
        let sample_rate = read_word(configuration, 8);
        let frame_period = f32::from_bits(read_word(configuration, 12));
        let hop = (sample_rate as f32 * frame_period).round();
        if !(2..=64).contains(&band_count)
            || band_count % 2 != 0
            || channels == 0
            || channels > 4096
            || !hop.is_finite()
            || hop < 1.0
            || hop > 65536.0
            || (hop as usize) % band_count != 0
        {
            return Err(failure("invalid sample rate, frame period, or band count."));
        }
        let mut candidate = Self::new();
        load_filters(
            reader,
            children[first_network + 7],
            "_ppusv0",
            band_count,
            &mut candidate.synthesis_filters,
        )?;
        load_filters(
            reader,
            children[first_network + 8],
            "_ppdsv0",
            band_count,
            &mut candidate.analysis_filters,
        )?;
        candidate
            .network
            .load(reader, children[first_network + 9])?;
        candidate
            .projection
            .load(reader, children[first_network + 10])?;
        candidate.hop_samples = hop as usize;
        candidate.bands = band_count;
        candidate.feature_channels = channels as usize;
        candidate.gate_noise = root_payload[13] != 0;
        candidate.loaded = true;
        *self = candidate;
        Ok(())
    }

    /// Whole-sequence offline rendering with deterministic excitation.
    ///
    /// Each call starts with the serialized model's initial state and fresh
    /// Gaussian excitation. The output is only assigned on success.
    pub fn render(
        &self,
        normalized_features: &DnniTensor,
        excitation: &[f32],
        output: &mut Vec<f32>,
        mut state: Option<&mut VocoderResidualState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        if !self.loaded {
            return Err(failure("no model has been loaded."));
        }
        if normalized_features.channels != self.feature_channels
            || normalized_features.frames > MAXIMUM_VALUES / self.feature_channels.max(1)
            || normalized_features.values.len()
                != normalized_features.frames * self.feature_channels
            || normalized_features.frames > MAXIMUM_VALUES / self.hop_samples.max(1)
            || excitation.len() != normalized_features.frames * self.hop_samples
        {
            return Err(failure(
                "condition frames and excitation length do not match the model.",
            ));
        }
        if !excitation.iter().all(|value| value.is_finite())
            || !normalized_features
                .values
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(failure("input contains a non-finite value."));
        }
        if normalized_features.frames == 0 {
            output.clear();
            return Ok(());
        }
        let interpolation_factor = self.hop_samples / self.bands;
        let subband_frames = normalized_features.frames * interpolation_factor;
        if self.feature_channels == 0
            || subband_frames > MAXIMUM_VALUES / self.feature_channels
            || subband_frames > MAXIMUM_VALUES / (self.bands * 2)
        {
            return Err(failure(
                "expanded condition tensor exceeds the memory limit.",
            ));
        }
        let analysis = analyse_excitation(excitation, &self.analysis_filters, subband_frames)?;
        let condition = interpolate_conditions(normalized_features, interpolation_factor);
        let mut hidden = DnniTensor::default();
        self.network.run_with_cache(
            &analysis,
            &mut hidden,
            Some(&condition),
            state.as_mut().map(|state| &mut state.network),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        let mut subbands = DnniTensor::default();
        self.projection.run_with_cache(
            &hidden,
            &mut subbands,
            None,
            state.as_mut().map(|state| &mut state.projection),
            cancel,
            crate::reborrow_stats(&mut statistics),
        )?;
        if subbands.frames != subband_frames || subbands.channels != self.bands {
            return Err(failure(
                "projected subband dimensions do not match the model.",
            ));
        }
        if self.gate_noise {
            for frame in 0..subband_frames {
                for band in 0..self.bands {
                    let index = frame * self.bands + band;
                    let noise = analysis.values[frame * self.bands * 2 + self.bands + band];
                    subbands.values[index] = noise / (1.0 + (-subbands.values[index]).exp());
                }
            }
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        let samples = synthesise_subbands(&subbands, &self.synthesis_filters);
        if samples.len() != excitation.len() || !samples.iter().all(|value| value.is_finite()) {
            return Err(failure(
                "synthesis produced an invalid sample count or a non-finite value.",
            ));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        *output = samples;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minstd_rand_matches_the_classic_stream() {
        // `std::minstd_rand` with the default seed: well-known first values.
        let mut generator = MinstdRand::new(1);
        assert_eq!(generator.next(), 48271);
        assert_eq!(generator.next(), 182_605_794);
        assert_eq!(generator.next(), 1_291_394_886);
    }

    #[test]
    fn gaussian_excitation_is_deterministic_with_a_spare() {
        // The rejection uniforms stay in [-1, 1): a wrong scale here hangs
        // the loop forever, so pin the first pair's range explicitly.
        let mut probe = MinstdRand::new(1);
        for _ in 0..16 {
            let uniform = probe.next().wrapping_sub(1) as f32 * f32::from_bits(0x3080_0000) - 1.0;
            assert!((-1.0..1.0).contains(&uniform));
        }
        let draw = || {
            let mut noise = GaussianExcitation::new();
            (0..8).map(|_| noise.next()).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
        for value in draw() {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn interpolation_matches_the_hand_computed_ramps() {
        // Factor 4 (even): two leading copies of the first frame, midpoint
        // ramps at 0.5/4 offsets, one trailing copy of the last frame.
        let input = DnniTensor {
            frames: 2,
            channels: 1,
            values: vec![0.0, 8.0],
        };
        let output = interpolate_conditions(&input, 4);
        assert_eq!(output.frames, 8);
        assert_eq!(output.values, vec![0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 8.0]);
        // Factor 3 (odd): one leading copy, integer ramps, two trailing.
        let output = interpolate_conditions(&input, 3);
        assert_eq!(
            output.values,
            vec![0.0, 0.0, 8.0 / 3.0, 16.0 / 3.0, 8.0, 8.0]
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
            VocoderResidual::new()
                .load(&reader, 0)
                .unwrap_err()
                .to_string(),
            "Vocoder residual: root node is out of range or has an unsupported type."
        );
        let mut output = Vec::new();
        assert_eq!(
            VocoderResidual::new()
                .render(
                    &DnniTensor::default(),
                    &[],
                    &mut output,
                    None,
                    &CancelToken::new(),
                    None
                )
                .unwrap_err()
                .to_string(),
            "Vocoder residual: no model has been loaded."
        );
    }

    #[test]
    fn fresh_state_holds_no_bytes() {
        assert_eq!(VocoderResidualState::default().get_bytes(), 0);
    }
}
