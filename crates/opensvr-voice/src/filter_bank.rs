//! Overlap-add cepstral filter bank (`VocoderFilterBank`).
//!
//! Ports OpenSV's `src/synthesis/VocoderFilterBank.{h,cpp}` (namespace
//! `sv::synthesis`): each of the three learned filters shapes one excitation
//! (periodic, residual-noise, modulated) by its per-frame weighted cepstral
//! coefficients. The quefrency placement, the Hann analysis window normalized
//! by its hop-sampled sum, the `10 * c / (k + 1)` weighting and the 16-sample
//! boundary crossfade are copied verbatim; both FFT directions run through
//! the shared [`crate::fft`] JUCE-compatible transforms.

use opensvr_dnni::{DnniReader, DnniTensor};
use rustfft::num_complex::Complex32;

use crate::fft::JuceFft;
use crate::{Result, invalid};

const MAXIMUM_FFT_SIZE: usize = 1024 * 1024;
const MAXIMUM_TENSOR_ELEMENTS: usize = 128 * 1024 * 1024;

/// Twelve seeded identifiers for the same filter implementation.
const FILTER_TYPE_IDS: [u64; 12] = [
    0x7106_3dac_a00b_5f25,
    0x452e_15e3_aced_1393,
    0x35bd_4340_52b6_3ee4,
    0xcb41_13ab_3ac4_b8e7,
    0x7380_a485_9db5_3c43,
    0x587c_9e37_3dc5_e0d4,
    0xfa92_9fbb_af89_ff69,
    0x0892_faf7_22f3_5001,
    0x33e2_c6b7_de8d_d6a3,
    0x2dcf_b47b_d6bb_27fc,
    0x8662_7a39_c3a1_7f9e,
    0xa1ce_8f87_c4fa_90ca,
];

fn boundary_window(sample: usize, fft_size: usize) -> f32 {
    if sample < 16 {
        (sample + 1) as f32 / 16.0
    } else if sample >= fft_size - 16 {
        (fft_size - 1 - sample) as f32 / 16.0
    } else {
        1.0
    }
}

/// One learned overlap-add spectral filter (two frames per window).
#[derive(Debug, Clone, Default)]
pub struct VocoderFilterBank {
    hop_samples: usize,
    fft_size: usize,
    analysis_window: Vec<f32>,
}

impl VocoderFilterBank {
    /// Loads the filter from its 13-byte leaf node for `requested_hop_samples`.
    ///
    /// The reference factory consumes only the first payload word (the window
    /// frame count, verified as 2); the remaining nine bytes do not
    /// configure the runtime filter. Nothing is stored on failure.
    pub fn load(
        &mut self,
        reader: &DnniReader,
        filter_node: usize,
        requested_hop_samples: usize,
    ) -> Result<()> {
        let nodes = reader.nodes();
        if nodes.get(filter_node).is_none_or(|node| {
            !FILTER_TYPE_IDS.contains(&node.type_id)
                || node.payload_size != 13
                || !node.children.is_empty()
        }) {
            return Err(invalid(
                "Vocoder filtering requires a supported 13-byte leaf filter node.",
            ));
        }
        let payload = reader.payload(filter_node);
        let window_frames = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        if window_frames != 2 || !(8..=MAXIMUM_FFT_SIZE / 4).contains(&requested_hop_samples) {
            return Err(invalid(
                "Unsupported vocoder filter window or sample hop; the verified filter uses two frames per window.",
            ));
        }
        let window_samples = requested_hop_samples * 2;
        let fft_size = 2 * (window_samples - 1).next_power_of_two();
        let mut window_sum = 0.0f32;
        let mut sample = 0;
        while sample < window_samples {
            let phase = sample as f32 * (2.0 * std::f32::consts::PI) / window_samples as f32;
            window_sum += 0.5 - 0.5 * phase.cos();
            sample += requested_hop_samples;
        }
        let mut analysis_window = vec![0.0f32; window_samples];
        for (index, slot) in analysis_window.iter_mut().enumerate() {
            let phase = index as f32 * (2.0 * std::f32::consts::PI) / window_samples as f32;
            *slot = (0.5 - 0.5 * phase.cos()) / window_sum;
        }
        self.hop_samples = requested_hop_samples;
        self.fft_size = fft_size;
        self.analysis_window = analysis_window;
        Ok(())
    }

    /// Whole-sequence filtering: `frames * hop_samples` samples out.
    ///
    /// `spectral_features` holds the network's weighted cepstral coefficients
    /// (not magnitude bins). The output is only assigned on success.
    ///
    /// The `usize as i64` conversions below mirror the C++ `int64_t` sample
    /// arithmetic; all operands are validated far below the 63-bit range.
    #[allow(clippy::cast_possible_wrap)]
    pub fn render(
        &self,
        spectral_features: &DnniTensor,
        excitation: &[f32],
        output: &mut Vec<f32>,
    ) -> Result<()> {
        if self.hop_samples == 0 {
            return Err(invalid("Vocoder filtering has not been loaded."));
        }
        let frames = spectral_features.frames;
        let channels = spectral_features.channels;
        if channels < 4
            || channels > self.fft_size
            || channels % 2 != 0
            || frames > MAXIMUM_TENSOR_ELEMENTS / channels.max(1)
            || frames * channels != spectral_features.values.len()
            || frames > MAXIMUM_TENSOR_ELEMENTS / self.hop_samples
            || frames * self.hop_samples != excitation.len()
        {
            return Err(invalid(
                "Vocoder filter features and excitation have incompatible shapes.",
            ));
        }
        if !spectral_features
            .values
            .iter()
            .all(|value| value.is_finite())
            || !excitation.iter().all(|value| value.is_finite())
        {
            return Err(invalid("Vocoder filter inputs contain a non-finite value."));
        }
        let fft = JuceFft::new(self.fft_size);
        let half_channels = channels / 2;
        let padding_samples = (self.fft_size - self.analysis_window.len()) / 2;
        let mut rendered = vec![0.0f32; excitation.len()];
        for frame in 0..frames {
            // Cepstrum (quefrency) -> log spectrum. Using JUCE's
            // negative-exponent convention for both transforms preserves the
            // reference's complex product.
            let mut quefrency = vec![0.0f32; self.fft_size];
            let features = &spectral_features.values[frame * channels..(frame + 1) * channels];
            for coefficient in 0..=half_channels {
                quefrency[coefficient] = 10.0 * features[coefficient] / (coefficient + 1) as f32;
            }
            for coefficient in 1..half_channels {
                quefrency[self.fft_size - half_channels + coefficient] = 10.0
                    * features[half_channels + coefficient]
                    / (half_channels + 1 - coefficient) as f32;
            }
            let log_spectrum = fft.real_forward(&quefrency);

            let mut segment = vec![0.0f32; self.fft_size];
            let frame_sample = frame as i64 * self.hop_samples as i64;
            let source_start = frame_sample - self.hop_samples as i64;
            for (index, window) in self.analysis_window.iter().enumerate() {
                let source_sample = source_start + index as i64;
                if source_sample >= 0 && source_sample < excitation.len() as i64 {
                    segment[padding_samples + index] = excitation[source_sample as usize] * window;
                }
            }
            let excitation_spectrum = fft.real_forward(&segment);
            let mut product = vec![Complex32::new(0.0, 0.0); self.fft_size];
            for bin in 0..=self.fft_size / 2 {
                let amplitude = log_spectrum[bin].re.exp();
                let phase = log_spectrum[bin].im;
                if !amplitude.is_finite() || !phase.is_finite() {
                    return Err(invalid(format!(
                        "Vocoder filter spectrum overflowed at frame {frame}."
                    )));
                }
                let filter_real = amplitude * phase.cos();
                let filter_imaginary = amplitude * phase.sin();
                let signal_real = excitation_spectrum[bin].re;
                let signal_imaginary = excitation_spectrum[bin].im;
                product[bin] = Complex32::new(
                    signal_real * filter_real - signal_imaginary * filter_imaginary,
                    signal_real * filter_imaginary + signal_imaginary * filter_real,
                );
            }
            // JUCE's inverse already applies 1/N, equivalent to the
            // reference's unnormalised real inverse followed by 2/N.
            let time = fft.real_inverse(&product);
            let destination_start = frame_sample - self.fft_size as i64 / 2;
            for (index, &value) in time.iter().enumerate() {
                let destination_sample = destination_start + index as i64;
                if destination_sample >= 0 && destination_sample < rendered.len() as i64 {
                    let slot = &mut rendered[destination_sample as usize];
                    *slot += value * boundary_window(index, self.fft_size);
                    if !slot.is_finite() {
                        return Err(invalid(format!(
                            "Vocoder filter produced a non-finite output at frame {frame}."
                        )));
                    }
                }
            }
        }
        *output = rendered;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_window_ramps_over_sixteen_samples() {
        assert_eq!(boundary_window(0, 1024), 1.0 / 16.0);
        assert_eq!(boundary_window(15, 1024), 1.0);
        assert_eq!(boundary_window(16, 1024), 1.0);
        assert_eq!(boundary_window(1007, 1024), 1.0);
        assert_eq!(boundary_window(1008, 1024), 15.0 / 16.0);
        assert_eq!(boundary_window(1023, 1024), 0.0);
    }

    #[test]
    fn unloaded_filter_rejects_render() {
        let mut output = Vec::new();
        assert_eq!(
            VocoderFilterBank::default()
                .render(&DnniTensor::default(), &[], &mut output)
                .unwrap_err()
                .to_string(),
            "Vocoder filtering has not been loaded."
        );
    }

    /// Builds a v2 blob whose root is a filter leaf with the given payload.
    fn filter_reader(payload: &[u8]) -> opensvr_dnni::DnniReader {
        const TYPE_ID: u64 = 0x7106_3dac_a00b_5f25;
        let type_low = TYPE_ID as u32;
        let type_high = (TYPE_ID >> 32) as u32;
        // v2 XORs the tripled child count; zero children need the raw mask.
        let child_raw = type_low ^ type_high ^ 0xac5e_7bd5;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        bytes.extend_from_slice(&type_low.to_le_bytes());
        bytes.extend_from_slice(&type_high.to_le_bytes());
        bytes.extend_from_slice(&child_raw.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(payload);
        opensvr_dnni::DnniReader::from_bytes(bytes).expect("synthetic filter must parse")
    }

    fn filter_payload() -> Vec<u8> {
        let mut payload = vec![0u8; 13];
        payload[0..4].copy_from_slice(&2_u32.to_le_bytes());
        payload
    }

    #[test]
    fn loader_rejects_a_non_filter_node() {
        // A synthetic non-filter leaf: the load must fail before touching
        // the hop size.
        let bytes = vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let reader = opensvr_dnni::DnniReader::from_bytes(bytes).expect("blob must parse");
        assert_eq!(
            VocoderFilterBank::default()
                .load(&reader, 0, 240)
                .unwrap_err()
                .to_string(),
            "Vocoder filtering requires a supported 13-byte leaf filter node."
        );
    }

    #[test]
    fn load_accepts_the_two_frame_window_and_rejects_other_hops() {
        let reader = filter_reader(&filter_payload());
        let mut filter = VocoderFilterBank::default();
        filter.load(&reader, 0, 240).expect("hop 240 must load");
        assert_eq!(filter.fft_size, 1024);
        // The analysis window is normalized by its hop-sampled sum, so the
        // hop-spaced taps add to exactly 1.
        let taps: f32 = (0..480)
            .step_by(240)
            .map(|index| filter.analysis_window[index])
            .sum();
        assert!((taps - 1.0).abs() < 1e-6);
        assert_eq!(
            VocoderFilterBank::default()
                .load(&reader, 0, 7)
                .unwrap_err()
                .to_string(),
            "Unsupported vocoder filter window or sample hop; the verified filter uses two frames per window."
        );
        let mut bad_window = filter_payload();
        bad_window[0] = 3;
        let bad = filter_reader(&bad_window);
        assert_eq!(
            VocoderFilterBank::default()
                .load(&bad, 0, 240)
                .unwrap_err()
                .to_string(),
            "Unsupported vocoder filter window or sample hop; the verified filter uses two frames per window."
        );
    }

    #[test]
    fn zero_cepstrum_passes_finite_energy() {
        // All-zero cepstral coefficients give amplitude 1, phase 0, so this
        // exercises the full forward/multiply/inverse chain without a model
        // and checks the overlap-add stays finite near the input energy.
        let reader = filter_reader(&filter_payload());
        let mut filter = VocoderFilterBank::default();
        filter.load(&reader, 0, 64).expect("hop 64 must load");
        let frames = 4;
        let features = DnniTensor {
            frames,
            channels: 8,
            values: vec![0.0; frames * 8],
        };
        let excitation: Vec<f32> = (0..frames * 64)
            .map(|index| (index as f32 * 0.1).sin())
            .collect();
        let mut output = Vec::new();
        filter
            .render(&features, &excitation, &mut output)
            .expect("identity must render");
        assert_eq!(output.len(), excitation.len());
        assert!(output.iter().all(|value| value.is_finite()));
        let energy_in: f32 = excitation.iter().map(|value| value * value).sum();
        let energy_out: f32 = output.iter().map(|value| value * value).sum();
        assert!(energy_out > 0.1 * energy_in && energy_out < 10.0 * energy_in);
    }
}
