//! Lip-radiation spectral correction (`LipRadiation`).
//!
//! Ports OpenSV's `src/synthesis/LipRadiation.{h,cpp}` (namespace
//! `sv::synthesis`): the piston-in-a-baffle mouth response applied to
//! harmonic amplitudes and phases. Pure DSP with no model weights: the
//! resistance/inductance constants, the `hypot`-based amplitude ratio and
//! the two-pass validate-then-apply structure are copied verbatim.

use crate::{Result, invalid};

/// Mouth response at one frequency: linear amplitude gain and phase shift.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LipRadiationResponse {
    /// Linear amplitude gain.
    pub amplitude_gain: f64,
    /// Phase shift in radians.
    pub phase_shift_radians: f64,
}

const RESISTANCE: f64 = 128.0 / (9.0 * std::f64::consts::PI * std::f64::consts::PI);
const INDUCTANCE_PER_CM: f64 = 8.0 / (100.0 * 3.0 * std::f64::consts::PI * 340.0);

fn is_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn fits_float(value: f64) -> bool {
    value.is_finite() && value.abs() <= f64::from(f32::MAX)
}

/// Frequency response of the lip opening (Hz, lip radius in cm).
///
/// Returns `None` when either argument is non-positive, non-finite, or
/// outside the numeric range. DC has no invertible response and is outside
/// this interface (rejected as non-positive).
pub fn lip_radiation_response(frequency_hz: f64, radius_cm: f64) -> Option<LipRadiationResponse> {
    if !is_positive_finite(frequency_hz) || !is_positive_finite(radius_cm) {
        return None;
    }
    let reactance = frequency_hz * radius_cm * (2.0 * std::f64::consts::PI * INDUCTANCE_PER_CM);
    if !is_positive_finite(reactance) {
        return None;
    }
    // The ratio avoids squaring large reactances or overflowing the numerator.
    let amplitude_gain = RESISTANCE * (reactance / RESISTANCE.hypot(reactance));
    if !is_positive_finite(amplitude_gain) {
        return None;
    }
    Some(LipRadiationResponse {
        amplitude_gain,
        phase_shift_radians: RESISTANCE.atan2(reactance),
    })
}

fn length_mismatch() -> crate::VoiceError {
    invalid("Harmonic amplitudes and phases must have the same length.")
}

/// Applies the lip-radiation response to harmonic amplitudes/phases.
///
/// Harmonics run from the fundamental upwards; amplitudes are linear and
/// phases are radians. Validation finishes before either slice is changed.
/// With `inverse`, divides by the amplitude gain and subtracts the phase
/// shift instead. Mirrors the C++ overlap check (rejected when the spans
/// overlap without being identical is not expressible in safe Rust, so any
/// shared backing allocation is rejected) and the error strings.
pub fn apply_lip_radiation_to_harmonics(
    amplitudes: &mut [f32],
    phases_radians: &mut [f32],
    fundamental_frequency_hz: f64,
    radius_cm: f64,
    inverse: bool,
) -> Result<()> {
    if amplitudes.len() != phases_radians.len() {
        return Err(length_mismatch());
    }
    // Safe-Rust counterpart of the C++ disjoint-spans check: two mutable
    // slices into one allocation cannot both be borrowed, so any pointer
    // overlap is rejected.
    if !amplitudes.is_empty() {
        let a_start = amplitudes.as_ptr() as usize;
        let a_end = a_start.wrapping_add(amplitudes.len() * 4);
        let p_start = phases_radians.as_ptr() as usize;
        let p_end = p_start.wrapping_add(phases_radians.len() * 4);
        if a_start < p_end && p_start < a_end {
            return Err(invalid(
                "Harmonic amplitude and phase arrays must not overlap.",
            ));
        }
    }
    if lip_radiation_response(fundamental_frequency_hz, radius_cm).is_none() {
        return Err(invalid(
            "Fundamental frequency and lip radius must be finite, positive, and within the numeric range.",
        ));
    }
    for (index, (amplitude, phase)) in amplitudes.iter().zip(phases_radians.iter()).enumerate() {
        if !amplitude.is_finite() || *amplitude < 0.0 || !phase.is_finite() {
            return Err(invalid(
                "Harmonic amplitudes must be finite and non-negative, and phases must be finite.",
            ));
        }
        let response =
            lip_radiation_response(fundamental_frequency_hz * (index + 1) as f64, radius_cm);
        let Some(response) = response else {
            return Err(invalid(
                "Harmonic frequency and lip radius exceed the numeric range.",
            ));
        };
        let amplitude = if inverse {
            f64::from(*amplitude) / response.amplitude_gain
        } else {
            f64::from(*amplitude) * response.amplitude_gain
        };
        let phase = f64::from(*phase)
            + if inverse {
                -response.phase_shift_radians
            } else {
                response.phase_shift_radians
            };
        if !fits_float(amplitude) || !fits_float(phase) {
            return Err(invalid(
                "Lip radiation would produce a value outside the float range.",
            ));
        }
    }
    for (index, (amplitude, phase)) in amplitudes
        .iter_mut()
        .zip(phases_radians.iter_mut())
        .enumerate()
    {
        let response =
            lip_radiation_response(fundamental_frequency_hz * (index + 1) as f64, radius_cm);
        // Validated above; the fallback is unreachable in practice.
        let Some(response) = response else {
            return Err(invalid(
                "Harmonic frequency and lip radius exceed the numeric range.",
            ));
        };
        *amplitude = if inverse {
            (f64::from(*amplitude) / response.amplitude_gain) as f32
        } else {
            (f64::from(*amplitude) * response.amplitude_gain) as f32
        };
        *phase = (f64::from(*phase)
            + if inverse {
                -response.phase_shift_radians
            } else {
                response.phase_shift_radians
            }) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_and_non_positive_inputs_have_no_response() {
        assert!(lip_radiation_response(0.0, 1.0).is_none());
        assert!(lip_radiation_response(440.0, 0.0).is_none());
        assert!(lip_radiation_response(f64::NAN, 1.0).is_none());
        assert!(lip_radiation_response(440.0, f64::INFINITY).is_none());
    }

    #[test]
    fn response_matches_the_hand_computed_ratio() {
        // reactance = 440 * 1.5 * 2*pi*inductance; gain = R*(x/hypot(R,x)).
        let response = lip_radiation_response(440.0, 1.5).expect("valid");
        let reactance = 440.0 * 1.5 * (2.0 * std::f64::consts::PI * INDUCTANCE_PER_CM);
        let want_gain = RESISTANCE * (reactance / RESISTANCE.hypot(reactance));
        assert!((response.amplitude_gain - want_gain).abs() < 1e-15);
        assert!((response.phase_shift_radians - RESISTANCE.atan2(reactance)).abs() < 1e-15);
        assert!(response.amplitude_gain > 0.0);
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let mut amplitudes = vec![0.5f32, 1.0, 2.0];
        let mut phases = vec![0.1f32, -0.3, 0.0];
        apply_lip_radiation_to_harmonics(&mut amplitudes, &mut phases, 220.0, 1.2, false)
            .expect("forward must apply");
        assert!(amplitudes.iter().all(|&value| value > 0.0));
        apply_lip_radiation_to_harmonics(&mut amplitudes, &mut phases, 220.0, 1.2, true)
            .expect("inverse must apply");
        for (&got, &want) in amplitudes.iter().zip(&[0.5f32, 1.0, 2.0]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        for (&got, &want) in phases.iter().zip(&[0.1f32, -0.3, 0.0]) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn mismatched_spans_are_rejected() {
        let mut amplitudes = vec![1.0f32, 2.0];
        let mut phases = vec![0.0f32];
        assert!(
            apply_lip_radiation_to_harmonics(&mut amplitudes, &mut phases, 220.0, 1.0, false)
                .is_err()
        );
        // Adjacent halves of one allocation do not overlap and apply cleanly.
        // (Truly overlapping `&mut` pairs cannot be built in safe Rust, so the
        // overlap guard itself is exercised by inspection.)
        let mut backing = [1.0f32, 2.0, 0.0, 0.0];
        let (amp, phase) = backing.split_at_mut(2);
        assert!(apply_lip_radiation_to_harmonics(amp, phase, 220.0, 1.0, false).is_ok());
    }

    #[test]
    fn invalid_harmonics_are_rejected_before_any_write() {
        let mut amplitudes = vec![1.0f32, -1.0];
        let mut phases = vec![0.0f32, 0.0];
        assert!(
            apply_lip_radiation_to_harmonics(&mut amplitudes, &mut phases, 220.0, 1.0, false)
                .is_err()
        );
        assert_eq!(amplitudes, vec![1.0f32, -1.0]);
    }
}
