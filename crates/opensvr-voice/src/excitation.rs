//! Periodic glottal excitation (`PeriodicExcitation`).
//!
//! Ports OpenSV's `src/synthesis/PeriodicExcitation.{h,cpp}` (namespace
//! `sv::synthesis`): the offline Liljencrants-Fant (LF) excitation and the
//! additional periodic modulation signal, both rendered with two-hop Hann
//! windows before the learned spectral filters.
//!
//! Numeric notes: the LF knot table is solved in `f64` exactly as the C++
//! bisection loops do; the chirp-Z harmonic sum runs through the shared
//! [`crate::fft`] JUCE-compatible transforms (unscaled forward, `1/N`
//! inverse); the float-quotient sample rate and Nyquist-bin count use `f32`
//! division before promotion, since promoting either division changes the
//! count at pitch boundaries.

use std::sync::OnceLock;

use rustfft::num_complex::{Complex32, ComplexFloat};

use crate::fft::JuceFft;
use crate::{Result, invalid};

type Complex64 = rustfft::num_complex::Complex<f64>;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const REFERENCE_PERIOD_SECONDS: f64 = 0.005;
const PARAMETER_KNOT_COUNT: usize = 256;
const MAXIMUM_FFT_SIZE: usize = 1024 * 1024;

/// One frame of excitation control: F0 in Hz, LF shape (Rd) and voicing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicExcitationFrame {
    /// Fundamental frequency in Hz (positive in voiced frames).
    pub fundamental_frequency_hz: f32,
    /// LF shape parameter (Rd), verified in `[0.01, 6]` when voiced.
    pub shape: f32,
    /// Whether the frame renders glottal pulses.
    pub voiced: bool,
}

/// Interpolated LF timing parameters on a unit-duration cycle.
#[derive(Debug, Clone, Copy)]
struct LfParameters {
    closing_time: f64,
    peak_time: f64,
    return_time: f64,
    decay: f64,
    growth: f64,
}

#[derive(Debug, Clone, Copy)]
struct LfKnot {
    shape: f32,
    closing_ratio: f32,
    peak_ratio: f32,
    return_ratio: f32,
    log_decay: f32,
    log_growth: f32,
}

/// Solves the LF continuity and zero-area equations on a unit-duration cycle.
///
/// The bisection midpoints keep the C++ `(lower + upper) * 0.5` form rather
/// than `f64::midpoint`, preserving the reference's rounding bit for bit.
#[allow(clippy::manual_midpoint)]
fn make_knot(index: usize) -> LfKnot {
    let rd = 0.01 * 600.0f64.powf(index as f64 / 255.0);
    let shape = rd as f32;
    let return_ratio = if rd < 0.21 {
        0.000_001
    } else if rd < 2.7 {
        (4.8 * rd - 1.0) / 100.0
    } else {
        0.323 / rd
    };
    let (closing_ratio, peak_ratio) = if index < 209 {
        let rk = (22.4 + 11.8 * rd) / 100.0;
        let rg = 0.25 * rk / (0.11 * rd / (0.5 + 1.2 * rk) - return_ratio);
        let peak = 0.5 / rg;
        (peak * (1.0 + rk), peak)
    } else {
        let open_quotient = 1.0 - 1.0 / (2.17 * rd);
        let rg = 0.009_355_2 + 5.96 / (7.92 - 2.0 * open_quotient);
        let rk = 2.0 * rg * open_quotient - 0.9572;
        let peak = 0.5 / rg;
        (peak * (1.0 + rk), peak)
    };
    let tail_length = 1.0 - closing_ratio;
    let mut lower = 0.0;
    let mut upper = 1.0 / return_ratio;
    for _ in 0..80 {
        let decay = (lower + upper) * 0.5;
        if -(-tail_length * decay).exp_m1() / decay > return_ratio {
            lower = decay;
        } else {
            upper = decay;
        }
    }
    let decay = (lower + upper) * 0.5;
    let tail_area = (-(-tail_length * decay).exp_m1()
        - tail_length * decay * (-tail_length * decay).exp())
        / (decay * decay * return_ratio);
    let angular_frequency = std::f64::consts::PI / peak_ratio;
    let closing_sine = (angular_frequency * closing_ratio).sin();
    let closing_cosine = (angular_frequency * closing_ratio).cos();
    let area_residual = |growth: f64| {
        tail_area * closing_sine * (growth * growth + angular_frequency * angular_frequency)
            + growth * closing_sine
            - angular_frequency * closing_cosine
            + angular_frequency * (-growth * closing_ratio).exp()
    };
    lower = 0.0;
    upper = 1.0;
    while area_residual(upper) > 0.0 {
        upper *= 2.0;
    }
    for _ in 0..80 {
        let growth = (lower + upper) * 0.5;
        if area_residual(growth) > 0.0 {
            lower = growth;
        } else {
            upper = growth;
        }
    }
    let growth = (lower + upper) * 0.5;
    LfKnot {
        shape,
        closing_ratio: closing_ratio as f32,
        peak_ratio: peak_ratio as f32,
        return_ratio: return_ratio as f32,
        log_decay: (decay / REFERENCE_PERIOD_SECONDS).ln() as f32,
        log_growth: (growth / REFERENCE_PERIOD_SECONDS).ln() as f32,
    }
}

fn knots() -> &'static [LfKnot; PARAMETER_KNOT_COUNT] {
    static KNOTS: OnceLock<[LfKnot; PARAMETER_KNOT_COUNT]> = OnceLock::new();
    KNOTS.get_or_init(|| {
        let mut values = [LfKnot {
            shape: 0.0,
            closing_ratio: 0.0,
            peak_ratio: 0.0,
            return_ratio: 0.0,
            log_decay: 0.0,
            log_growth: 0.0,
        }; PARAMETER_KNOT_COUNT];
        for (index, slot) in values.iter_mut().enumerate() {
            *slot = make_knot(index);
        }
        values
    })
}

fn lerp(left: f32, right: f32, position: f64) -> f64 {
    f64::from(left) + (f64::from(right) - f64::from(left)) * position
}

fn interpolate_parameters(shape: f32) -> LfParameters {
    let knots = knots();
    let upper = knots.partition_point(|knot| knot.shape <= shape);
    let right = if upper >= knots.len() {
        &knots[knots.len() - 1]
    } else {
        &knots[upper]
    };
    let left = if upper == 0 {
        &knots[0]
    } else {
        &knots[upper - 1]
    };
    let position = if left.shape == right.shape {
        0.0
    } else {
        (f64::from(shape) - f64::from(left.shape))
            / (f64::from(right.shape) - f64::from(left.shape))
    };
    LfParameters {
        closing_time: lerp(left.closing_ratio, right.closing_ratio, position),
        peak_time: lerp(left.peak_ratio, right.peak_ratio, position),
        return_time: lerp(left.return_ratio, right.return_ratio, position),
        decay: lerp(left.log_decay, right.log_decay, position).exp() * REFERENCE_PERIOD_SECONDS,
        growth: lerp(left.log_growth, right.log_growth, position).exp() * REFERENCE_PERIOD_SECONDS,
    }
}

fn get_lf_spectrum(parameters: &LfParameters, harmonic: usize) -> Complex64 {
    // The stored LF parameters feed a spectral variant whose opening
    // denominator subtracts sourceOmega^2; preserve the independently
    // observed transfer function here. Time is one normalized cycle.
    let omega = TWO_PI * harmonic as f64;
    let source_omega = std::f64::consts::PI / parameters.peak_time;
    let closing_angle = source_omega * parameters.closing_time;
    let closing_sine = closing_angle.sin();
    let s = Complex64::new(parameters.growth, -omega);
    let closing_rotation = Complex64::from_polar(1.0, -omega * parameters.closing_time);
    let opening = -(closing_rotation * (s * closing_sine - source_omega * closing_angle.cos())
        + source_omega * (-parameters.growth * parameters.closing_time).exp())
        / (closing_sine * (s * s - source_omega * source_omega));
    let tail_length = 1.0 - parameters.closing_time;
    let tail_rate = Complex64::new(parameters.decay, omega);
    let mut returning = -closing_rotation / tail_rate;
    let endpoint = 1.0 - parameters.decay * parameters.return_time;
    // The reference drops this endpoint correction below a decay-rate
    // threshold of 1/second, before converting the period to
    // unit-duration coordinates.
    if parameters.decay * endpoint >= REFERENCE_PERIOD_SECONDS {
        returning += closing_rotation
            * endpoint
            * (Complex64::new(1.0, 0.0) - Complex64::from_polar(1.0, -omega * tail_length))
            / (Complex64::new(0.0, omega) * parameters.return_time * tail_rate);
    }
    opening + returning
}

fn validate_render(
    frames: &[PeriodicExcitationFrame],
    hop_samples: usize,
    frame_period_seconds: f64,
) -> Result<()> {
    if hop_samples == 0
        || hop_samples > usize::MAX / 2
        || !frame_period_seconds.is_finite()
        || frame_period_seconds <= 0.0
    {
        return Err(invalid(
            "Periodic excitation requires a positive hop and finite positive frame period.",
        ));
    }
    if frames.len() > (isize::MAX as usize) / size_of::<f32>() / hop_samples.max(1) {
        return Err(invalid(
            "Periodic excitation sample count exceeds the container limit.",
        ));
    }
    let sample_rate = hop_samples as f64 / frame_period_seconds;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid(
            "Periodic excitation sample rate exceeds the numeric range.",
        ));
    }
    for frame in frames {
        if !frame.fundamental_frequency_hz.is_finite()
            || frame.fundamental_frequency_hz < 0.0
            || (frame.voiced && frame.fundamental_frequency_hz == 0.0)
        {
            return Err(invalid(
                "Periodic excitation requires finite non-negative F0 and positive F0 in voiced frames.",
            ));
        }
    }
    Ok(())
}

fn make_window(hop_samples: usize, gain: f64) -> Vec<f64> {
    let mut window = vec![0.0; hop_samples * 2];
    let length = window.len() as f64;
    for (index, slot) in window.iter_mut().enumerate() {
        *slot = gain * (0.5 - 0.5 * (TWO_PI * index as f64 / length).cos());
    }
    window
}

fn add_window(
    output: &mut [f32],
    frame_index: usize,
    hop_samples: usize,
    signal: &[f64],
    window: &[f64],
) -> Result<()> {
    let source_begin = if frame_index == 0 { hop_samples } else { 0 };
    let output_begin = if frame_index == 0 {
        0
    } else {
        (frame_index - 1) * hop_samples
    };
    for index in source_begin..signal.len() {
        let destination = output_begin + index - source_begin;
        let value = f64::from(output[destination]) + signal[index] * window[index];
        if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
            return Err(invalid(
                "LF excitation produced a sample outside the float range.",
            ));
        }
        output[destination] = value as f32;
    }
    Ok(())
}

/// Renders the offline LF excitation before the learned spectral filters.
///
/// Each frame contributes a two-hop Hann window; the first half-window is
/// discarded and the last retained, giving `frames.len() * hop_samples`
/// samples. `phase_cycles` holds the cumulative unwrapped phase at the end
/// of each frame, including unvoiced frames. Outputs are only assigned on
/// success.
pub fn render_periodic_excitation(
    frames: &[PeriodicExcitationFrame],
    hop_samples: usize,
    frame_period_seconds: f64,
    gain: f32,
    samples: &mut Vec<f32>,
    phase_cycles: &mut Vec<f64>,
) -> Result<()> {
    validate_render(frames, hop_samples, frame_period_seconds)?;
    if !gain.is_finite() || gain < 0.0 {
        return Err(invalid(
            "Periodic excitation gain must be finite and non-negative.",
        ));
    }
    if hop_samples > MAXIMUM_FFT_SIZE / 4 {
        return Err(invalid(
            "Periodic excitation hop exceeds the supported FFT size.",
        ));
    }
    for frame in frames {
        if frame.voiced && (!frame.shape.is_finite() || frame.shape < 0.01 || frame.shape > 6.0) {
            return Err(invalid(
                "LF shape must be finite and within the verified parameter range [0.01, 6].",
            ));
        }
    }
    // The float quotient and float Nyquist division determine the discrete
    // bin count. Promoting either division changes the count at pitch
    // boundaries.
    let sample_rate = hop_samples as f32 / frame_period_seconds as f32;
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid(
            "Periodic excitation sample rate exceeds the float range.",
        ));
    }
    let window = make_window(hop_samples, f64::from(gain));
    let mut rendered = vec![0.0f32; frames.len() * hop_samples];
    let mut phases = vec![0.0; frames.len()];
    let mut signal = vec![0.0; hop_samples * 2];
    let fft_size = (signal.len() * 2 - 1).next_power_of_two();
    let fft = JuceFft::new(fft_size);
    let mut chirp = vec![Complex64::new(0.0, 0.0); signal.len()];
    let mut spectrum = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut kernel = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut transformed_kernel = vec![Complex32::new(0.0, 0.0); fft_size];
    let mut accumulated_cycles = 0.0;
    for (frame_index, frame) in frames.iter().enumerate() {
        let increment = f64::from(frame.fundamental_frequency_hz * frame_period_seconds as f32);
        accumulated_cycles += increment;
        if !accumulated_cycles.is_finite() {
            return Err(invalid(
                "Periodic excitation phase exceeds the numeric range.",
            ));
        }
        phases[frame_index] = accumulated_cycles;
        if !frame.voiced {
            continue;
        }
        let harmonic_limit = 0.5f32 * sample_rate / frame.fundamental_frequency_hz;
        let harmonic_count = ((hop_samples * 2) as f64).min(f64::from(harmonic_limit)) as usize;
        if harmonic_count == 0 {
            continue;
        }
        let parameters = interpolate_parameters(frame.shape);
        let reference_magnitude = get_lf_spectrum(&parameters, 2).abs();
        let reference_phase = get_lf_spectrum(&parameters, 1).arg();
        let amplitude_scale = 0.1 * f64::from(sample_rate)
            / (f64::from(frame.fundamental_frequency_hz)
                * harmonic_count as f64
                * reference_magnitude);
        let frame_phase = TWO_PI * (accumulated_cycles % 1.0 - increment);
        // The reference's chirp-z transform consumes 2*hop bins, including
        // DC; the uppermost allocated bin is therefore outside that
        // transform.
        let last_harmonic = harmonic_count.min(hop_samples * 2 - 1);
        let half_step = std::f64::consts::PI * f64::from(frame.fundamental_frequency_hz)
            / f64::from(sample_rate);
        spectrum.fill(Complex32::new(0.0, 0.0));
        kernel.fill(Complex32::new(0.0, 0.0));
        for (index, slot) in chirp.iter_mut().enumerate() {
            let position = index as f64;
            *slot = Complex64::from_polar(1.0, half_step * position * position);
            kernel[index] = Complex32::new(slot.re as f32, -(slot.im as f32));
        }
        for harmonic in 1..=last_harmonic {
            let lf_spectrum = get_lf_spectrum(&parameters, harmonic);
            let amplitude = lf_spectrum.abs();
            let phase = lf_spectrum.arg() + harmonic as f64 * (frame_phase - reference_phase);
            let product = Complex64::from_polar(amplitude, phase) * chirp[harmonic];
            spectrum[harmonic] = Complex32::new(product.re as f32, product.im as f32);
            kernel[fft_size - harmonic] =
                Complex32::new(chirp[harmonic].re as f32, -(chirp[harmonic].im as f32));
        }
        // hn = (h^2 + n^2 - (n-h)^2) / 2 turns the harmonic sum into a
        // convolution without quantising F0 to FFT bins. The wrapped
        // negative kernel indices and zero padding preserve the finite sum's
        // boundaries.
        transformed_kernel.copy_from_slice(&fft.forward_complex(&kernel));
        kernel.copy_from_slice(&fft.forward_complex(&spectrum));
        for (slot, factor) in kernel.iter_mut().zip(&transformed_kernel) {
            *slot *= *factor;
        }
        // JUCE's inverse already includes the 1/N convolution
        // normalisation. Apply the common amplitude scale in double after
        // the float FFT so very small F0 does not overflow the transform's
        // intermediate values.
        spectrum.copy_from_slice(&fft.inverse_complex(&kernel));
        for (index, slot) in signal.iter_mut().enumerate() {
            let value =
                Complex64::new(f64::from(spectrum[index].re), f64::from(spectrum[index].im))
                    * chirp[index];
            *slot = value.re * amplitude_scale;
        }
        add_window(&mut rendered, frame_index, hop_samples, &signal, &window)?;
    }
    *samples = rendered;
    *phase_cycles = phases;
    Ok(())
}

/// Renders the additional periodic modulation signal.
///
/// `parameters` holds one frame-major row per frame: log amplitudes for all
/// ratios followed by their phase offsets in radians. Multiplication by the
/// excitation and the model's DC offset belongs to the caller. The output is
/// only assigned on success.
#[allow(clippy::too_many_arguments)]
pub fn render_periodic_modulation(
    frames: &[PeriodicExcitationFrame],
    phase_cycles: &[f64],
    frequency_ratios: &[f32],
    parameters: &[f32],
    hop_samples: usize,
    frame_period_seconds: f64,
    samples: &mut Vec<f32>,
) -> Result<()> {
    validate_render(frames, hop_samples, frame_period_seconds)?;
    if phase_cycles.len() != frames.len()
        || frequency_ratios.is_empty()
        || frequency_ratios.len() > usize::MAX / 2
    {
        return Err(invalid(
            "Periodic modulation requires one phase per frame and at least one frequency ratio.",
        ));
    }
    let channels = frequency_ratios.len() * 2;
    if frames.len() > usize::MAX / channels || parameters.len() != frames.len() * channels {
        return Err(invalid(
            "Periodic modulation parameters must contain log amplitudes followed by phase offsets for every frame.",
        ));
    }
    for ratio in frequency_ratios {
        if !ratio.is_finite() || *ratio <= 0.0 {
            return Err(invalid(
                "Periodic modulation frequency ratios must be finite and positive.",
            ));
        }
    }
    for value in parameters {
        if !value.is_finite() {
            return Err(invalid("Periodic modulation parameters must be finite."));
        }
    }
    let window = make_window(hop_samples, 1.0);
    let mut rendered = vec![0.0f32; frames.len() * hop_samples];
    let mut signal = vec![0.0; hop_samples * 2];
    for (frame_index, frame) in frames.iter().enumerate() {
        // The model pipeline converts its accumulated double phase to float here.
        let frame_cycles = phase_cycles[frame_index] as f32;
        if !frame_cycles.is_finite() {
            return Err(invalid(
                "Periodic modulation phase exceeds the float range.",
            ));
        }
        signal.fill(0.0);
        for (ratio_index, ratio) in frequency_ratios.iter().enumerate() {
            let amplitude = f64::from(parameters[frame_index * channels + ratio_index]).exp();
            let angular_ratio = TWO_PI * f64::from(*ratio);
            let phase = angular_ratio * f64::from(frame_cycles)
                + f64::from(
                    parameters[frame_index * channels + frequency_ratios.len() + ratio_index],
                );
            let step =
                angular_ratio * f64::from(frame.fundamental_frequency_hz) * frame_period_seconds
                    / hop_samples as f64;
            for (index, slot) in signal.iter_mut().enumerate() {
                *slot += amplitude * (phase + (index as f64 - hop_samples as f64) * step).cos();
            }
        }
        if add_window(&mut rendered, frame_index, hop_samples, &signal, &window).is_err() {
            return Err(invalid(
                "Periodic modulation produced a sample outside the float range.",
            ));
        }
    }
    *samples = rendered;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced(f0: f32, shape: f32) -> PeriodicExcitationFrame {
        PeriodicExcitationFrame {
            fundamental_frequency_hz: f0,
            shape,
            voiced: true,
        }
    }

    #[test]
    fn knots_are_ordered_and_cover_the_shape_range() {
        let knots = knots();
        assert_eq!(knots.len(), 256);
        assert!((knots[0].shape - 0.01).abs() < 1e-9);
        assert!(knots[255].shape > 5.9);
        for pair in knots.windows(2) {
            assert!(pair[0].shape < pair[1].shape);
        }
    }

    #[test]
    fn lf_spectrum_of_a_symmetric_pulse_is_real_at_dc_side() {
        // Hand-computed from the source: harmonic 0 of any parameter set is
        // the unit-cycle area (opening + returning at omega = 0).
        let parameters = interpolate_parameters(1.0);
        let dc = get_lf_spectrum(&parameters, 0);
        assert!(dc.re.is_finite() && dc.im.abs() < 1e-9);
    }

    #[test]
    fn window_matches_the_hand_computed_hann_shape() {
        // hop 4, gain 2: 2*(0.5 - 0.5*cos(2*pi*i/8)).
        let window = make_window(4, 2.0);
        assert_eq!(window.len(), 8);
        assert!((window[0] - 0.0).abs() < 1e-12);
        assert!((window[4] - 2.0).abs() < 1e-12);
        assert!((window[2] - 2.0 * (0.5 - 0.5 * (std::f64::consts::PI / 2.0).cos())).abs() < 1e-12);
    }

    #[test]
    fn unvoiced_frames_render_silence_with_running_phase() {
        let frames = vec![
            PeriodicExcitationFrame {
                fundamental_frequency_hz: 440.0,
                shape: 1.0,
                voiced: false,
            },
            PeriodicExcitationFrame {
                fundamental_frequency_hz: 440.0,
                shape: 1.0,
                voiced: false,
            },
        ];
        let mut samples = Vec::new();
        let mut phases = Vec::new();
        render_periodic_excitation(&frames, 64, 0.005, 1.0, &mut samples, &mut phases)
            .expect("unvoiced must render");
        assert_eq!(samples.len(), 128);
        assert!(samples.iter().all(|&value| value == 0.0));
        // Phase still accumulates through unvoiced frames (float period).
        let increment = f64::from(440.0f32 * 0.005f32);
        assert!((phases[0] - increment).abs() < 1e-12);
        assert!((phases[1] - 2.0 * increment).abs() < 1e-12);
    }

    #[test]
    fn single_voiced_frame_renders_nonzero_windowed_pulses() {
        let frames = vec![voiced(200.0, 1.0), voiced(200.0, 1.0), voiced(200.0, 1.0)];
        let mut samples = Vec::new();
        let mut phases = Vec::new();
        render_periodic_excitation(&frames, 64, 0.005, 1.0, &mut samples, &mut phases)
            .expect("voiced must render");
        assert_eq!(samples.len(), 192);
        assert!(samples.iter().any(|&value| value != 0.0));
        assert!(samples.iter().all(|value| value.is_finite()));
        // First half-window is discarded: the last frame's tail is retained.
        assert_eq!(phases.len(), 3);
    }

    #[test]
    fn modulation_of_a_single_ratio_matches_hand_computation() {
        // One frame, one ratio of 1.0, log-amplitude 0 and zero phase offset:
        // signal[i] = cos(2*pi*cycles + (i - hop)*step), Hann-windowed with
        // the second half kept for frame 0.
        let frames = vec![voiced(100.0, 1.0)];
        let phases = vec![0.25];
        let ratios = vec![1.0f32];
        let parameters = vec![0.0f32, 0.0f32];
        let mut samples = Vec::new();
        render_periodic_modulation(
            &frames,
            &phases,
            &ratios,
            &parameters,
            8,
            0.005,
            &mut samples,
        )
        .expect("modulation must render");
        assert_eq!(samples.len(), 8);
        let angular = TWO_PI;
        let step = angular * f64::from(100.0f32) * 0.005 / 8.0;
        for (index, &got) in samples.iter().enumerate() {
            let source = 8 + index; // second half of the two-hop window
            let wave = (angular * 0.25f64 + (source as f64 - 8.0) * step).cos();
            let window = 0.5 - 0.5 * (TWO_PI * source as f64 / 16.0).cos();
            assert!(
                (f64::from(got) - wave * window).abs() < 1e-5,
                "sample {index}"
            );
        }
    }

    #[test]
    fn fixed_voiced_sequence_matches_the_cpp_dump() {
        // Dumped from HEAD `renderPeriodicExcitation` (see
        // `do-not-distribute/vocoder-dump/excitation_probe.cc`): three voiced
        // 200 Hz frames, Rd 1.0, hop 64, gain 1. Covers the LF knot table,
        // the chirp-Z convolution and the FFT scaling in one shot.
        let frames = vec![voiced(200.0, 1.0), voiced(200.0, 1.0), voiced(200.0, 1.0)];
        let mut samples = Vec::new();
        let mut phases = Vec::new();
        render_periodic_excitation(&frames, 64, 0.005, 1.0, &mut samples, &mut phases)
            .expect("voiced must render");
        assert_eq!(samples.len(), 192);
        let want = [
            0.090_845_294_3f64,
            0.090_497_031_8,
            0.093_273_446,
            0.092_547_737_1,
            0.094_807_013_9,
            0.093_734_584_7,
            0.095_385_052_3,
            0.093_919_910_5,
        ];
        for (index, &want) in want.iter().enumerate() {
            let got = f64::from(samples[index]);
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                "sample {index}: {got} vs {want}"
            );
        }
        assert!((f64::from(samples[191]) - 5.284_370_85e-05).abs() < 1e-9);
        assert_eq!(phases, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn invalid_inputs_are_rejected_with_cpp_strings() {
        let mut samples = Vec::new();
        let mut phases = Vec::new();
        let frames = vec![voiced(100.0, 1.0)];
        assert_eq!(
            render_periodic_excitation(&frames, 0, 0.005, 1.0, &mut samples, &mut phases)
                .unwrap_err()
                .to_string(),
            "Periodic excitation requires a positive hop and finite positive frame period."
        );
        assert_eq!(
            render_periodic_excitation(&frames, 64, 0.005, -1.0, &mut samples, &mut phases)
                .unwrap_err()
                .to_string(),
            "Periodic excitation gain must be finite and non-negative."
        );
        assert_eq!(
            render_periodic_excitation(
                &[voiced(100.0, 100.0)],
                64,
                0.005,
                1.0,
                &mut samples,
                &mut phases,
            )
            .unwrap_err()
            .to_string(),
            "LF shape must be finite and within the verified parameter range [0.01, 6]."
        );
        let mut modulation = Vec::new();
        assert_eq!(
            render_periodic_modulation(
                &frames,
                &[],
                &[1.0],
                &[0.0, 0.0],
                64,
                0.005,
                &mut modulation
            )
            .unwrap_err()
            .to_string(),
            "Periodic modulation requires one phase per frame and at least one frequency ratio."
        );
    }
}
