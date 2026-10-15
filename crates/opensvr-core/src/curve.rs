//! Automation parameter curves (`pitchDelta`, `vibratoEnv`, ...).
//!
//! This is the Rust counterpart of OpenSV's `src/core/ParameterCurve.{h,cpp}`:
//! a piecewise curve over [`Blick`](crate::Blick) positions with `linear`,
//! `cosine` and `cubic` interpolation, sample-and-hold clamping outside the
//! point range, and a fallback default for empty curves.
//!
//! Control points must be sorted ascending by position with unique positions
//! (enforced upstream by project normalisation, as in the C++ code). All
//! evaluation is `f64`, in the same order as the C++ implementation.

use std::{fmt, str::FromStr};

use thiserror::Error;

use crate::{BLICKS_PER_QUARTER, Blick, Extra};

/// Interpolation mode of a [`Curve`]; the C++ default is `"cubic"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Interpolation {
    #[default]
    Cubic,
    Cosine,
    Linear,
}

impl Interpolation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cubic => "cubic",
            Self::Cosine => "cosine",
            Self::Linear => "linear",
        }
    }
}

impl fmt::Display for Interpolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when a string is not one of the supported interpolation modes.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("unsupported interpolation mode {0:?} (expected cubic, cosine or linear)")]
pub struct ParseInterpolationError(String);

impl FromStr for Interpolation {
    type Err = ParseInterpolationError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "cubic" => Ok(Self::Cubic),
            "cosine" => Ok(Self::Cosine),
            "linear" => Ok(Self::Linear),
            other => Err(ParseInterpolationError(other.to_owned())),
        }
    }
}

/// A single control point on a continuous parameter automation curve.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AutomationPoint {
    /// Time location of the point in blicks.
    pub position: Blick,
    /// Parameter value at this position.
    pub value: f64,
}

/// Piecewise parameter automation curve (pitch delta, vibrato envelope, ...).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Curve {
    pub mode: Interpolation,
    /// Ordered control points (sorted strictly by position).
    pub points: Vec<AutomationPoint>,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

/// Unsigned distance between two [`Blick`] positions as an `f64`.
///
/// Casts to `u64` before subtracting so positions spanning negative or large
/// signed offsets still measure correctly, exactly as the C++ `blickDistance`.
fn blick_distance(later: Blick, earlier: Blick) -> f64 {
    (later as u64).wrapping_sub(earlier as u64) as f64
}

/// Custom Hermite cubic interpolation between `points[right_index - 1]` and
/// `points[right_index]`, with `fraction` in `(0, 1)`.
///
/// Tangents blend Catmull-Rom-style central differences (closely spaced
/// points) into the chord slope (widely spaced points), so long spans decay
/// to a smoothstep S-curve instead of overshooting. Ported line-for-line from
/// the C++ `sampleCubic`.
fn sample_cubic(
    points: &[AutomationPoint],
    right_index: usize,
    interval: f64,
    fraction: f64,
) -> f64 {
    let left = &points[right_index - 1];
    let right = &points[right_index];

    let preceding_value = if right_index > 1 {
        points[right_index - 2].value
    } else {
        left.value
    };
    let following_value = if right_index + 1 < points.len() {
        points[right_index + 1].value
    } else {
        right.value
    };

    let preceding_interval = if right_index > 1 {
        blick_distance(right.position, points[right_index - 2].position)
    } else {
        interval + 1.0
    };
    let following_interval = if right_index + 1 < points.len() {
        blick_distance(points[right_index + 1].position, left.position)
    } else {
        interval + 1.0
    };

    // Adaptive tangent blending weight based on quarter-note beat distance.
    let spacing = (interval / (BLICKS_PER_QUARTER as f64) - 0.25).max(0.0);
    let weight = if spacing <= 4.0 {
        1.0 / (spacing * 10.0 + 1.0)
    } else {
        0.0
    };

    // Base segment slope (chord secant).
    let segment_slope = (right.value - left.value) / interval * (1.0 - weight);

    // Blended left and right tangents.
    let left_tangent =
        (right.value - preceding_value) / preceding_interval * interval * weight + segment_slope;
    let right_tangent =
        (following_value - left.value) / following_interval * interval * weight + segment_slope;

    // Hermite basis evaluation.
    let squared = fraction * fraction;
    let cubed = squared * fraction;
    (2.0 * cubed - 3.0 * squared + 1.0) * left.value
        + (fraction + cubed - 2.0 * squared) * left_tangent
        + (3.0 * squared - 2.0 * cubed) * right.value
        + (cubed - squared) * right_tangent
}

/// Evaluates an automation [`Curve`] at a musical position in blicks.
///
/// - Empty curves return `default`.
/// - Positions outside the point range clamp to the nearest endpoint
///   (sample-and-hold).
/// - Positions between points interpolate per `curve.mode`; `cubic` with
///   fewer than three points degrades to linear, exactly as the C++
///   `sampleParameterCurve`.
#[must_use]
pub fn sample_curve(curve: &Curve, position: Blick, default: f64) -> f64 {
    if curve.points.is_empty() {
        return default;
    }

    let points = &curve.points;

    if position <= points[0].position {
        return points[0].value;
    }
    if position >= points[points.len() - 1].position {
        return points[points.len() - 1].value;
    }

    // First point with position >= the query position.
    let right_index = points.partition_point(|point| point.position < position);
    let right = &points[right_index];
    if right.position == position {
        return right.value;
    }

    let left = &points[right_index - 1];
    let interval = blick_distance(right.position, left.position);
    let mut fraction = blick_distance(position, left.position) / interval;

    if curve.mode == Interpolation::Cubic && points.len() > 2 {
        return sample_cubic(points, right_index, interval, fraction);
    }

    if curve.mode == Interpolation::Cosine {
        fraction = (1.0 - (fraction * std::f64::consts::PI).cos()) * 0.5;
    }

    left.value * (1.0 - fraction) + right.value * fraction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BLICKS_PER_QUARTER as Q;

    fn curve(mode: Interpolation, points: &[(Blick, f64)]) -> Curve {
        Curve {
            mode,
            points: points
                .iter()
                .map(|&(position, value)| AutomationPoint { position, value })
                .collect(),
            extra: Extra::default(),
        }
    }

    #[test]
    fn empty_curves_return_the_default() {
        for mode in [
            Interpolation::Cubic,
            Interpolation::Cosine,
            Interpolation::Linear,
        ] {
            let empty = Curve {
                mode,
                points: Vec::new(),
                extra: Extra::default(),
            };
            assert_eq!(sample_curve(&empty, 0, 0.0), 0.0);
            assert_eq!(sample_curve(&empty, 42, 1.0), 1.0);
        }
    }

    #[test]
    fn positions_outside_the_range_clamp_to_the_endpoints() {
        let line = curve(Interpolation::Linear, &[(100, 2.0), (200, 4.0)]);
        assert_eq!(sample_curve(&line, 0, 0.0), 2.0);
        assert_eq!(sample_curve(&line, 99, 0.0), 2.0);
        assert_eq!(sample_curve(&line, 100, 0.0), 2.0);
        assert_eq!(sample_curve(&line, 200, 0.0), 4.0);
        assert_eq!(sample_curve(&line, 201, 0.0), 4.0);
        assert_eq!(sample_curve(&line, Blick::MAX, 0.0), 4.0);
    }

    #[test]
    fn linear_interpolates_proportionally() {
        let line = curve(Interpolation::Linear, &[(0, 0.0), (1000, 10.0)]);
        assert_eq!(sample_curve(&line, 250, 0.0), 2.5);
        assert_eq!(sample_curve(&line, 500, 0.0), 5.0);
        // Exact hits return the point value in every mode.
        for mode in [
            Interpolation::Cubic,
            Interpolation::Cosine,
            Interpolation::Linear,
        ] {
            let stepped = curve(mode, &[(0, 1.0), (100, 3.0), (200, 2.0)]);
            assert_eq!(sample_curve(&stepped, 100, 0.0), 3.0);
        }
    }

    #[test]
    fn cosine_eases_with_a_half_cosine_smoothstep() {
        let cosine = curve(Interpolation::Cosine, &[(0, 0.0), (1000, 8.0)]);
        // (1 - cos(pi/4)) / 2 of the way up at the quarter mark.
        let at_quarter = sample_curve(&cosine, 250, 0.0);
        assert!((at_quarter - 8.0 * (1.0 - std::f64::consts::FRAC_PI_4.cos()) * 0.5).abs() < 1e-12);
        // Midpoint agrees with linear (up to cos(pi/2) rounding); endpoints clamp.
        assert!((sample_curve(&cosine, 500, 0.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn cubic_with_two_points_degrades_to_linear() {
        let cubic = curve(Interpolation::Cubic, &[(0, 0.0), (1000, 10.0)]);
        assert_eq!(sample_curve(&cubic, 250, 0.0), 2.5);
        assert_eq!(sample_curve(&cubic, 750, 0.0), 7.5);
    }

    #[test]
    fn widely_spaced_cubic_is_a_smoothstep() {
        // Spans over 4.25 beats drive the tangent weight to zero, so both
        // tangents equal the chord slope and the Hermite basis collapses to
        // p0 + (p1 - p0) * (3t^2 - 2t^3).
        let span = 5 * Q;
        let cubic = curve(
            Interpolation::Cubic,
            &[(0, 0.0), (span, 4.0), (2 * span, 4.0)],
        );
        for &(blicks, t) in &[(span / 4, 0.25), (span / 2, 0.5), (3 * span / 4, 0.75)] {
            let smooth = 3.0 * t * t - 2.0 * t * t * t;
            let got = sample_curve(&cubic, blicks, 0.0);
            assert!(
                (got - 4.0 * smooth).abs() < 1e-9,
                "at t={t}: got {got}, want {}",
                4.0 * smooth
            );
        }
    }

    #[test]
    fn closely_spaced_cubic_passes_through_its_points() {
        // Intervals under a quarter beat use full Catmull-Rom-style tangents.
        let step = Q / 8;
        let cubic = curve(
            Interpolation::Cubic,
            &[(0, 0.0), (step, 1.0), (2 * step, 0.0), (3 * step, 1.0)],
        );
        for &(position, value) in &[(0, 0.0), (step, 1.0), (2 * step, 0.0), (3 * step, 1.0)] {
            assert_eq!(sample_curve(&cubic, position, 0.0), value);
        }
        // Symmetric peak: the midpoint between two equal neighbours of a
        // mirrored valley stays centered.
        let middle = sample_curve(&cubic, step / 2, 0.0);
        assert!(
            (middle - 0.5).abs() < 0.25,
            "overshooting midpoint: {middle}"
        );
    }

    #[test]
    fn interpolation_mode_round_trips_through_text() {
        for mode in [
            Interpolation::Cubic,
            Interpolation::Cosine,
            Interpolation::Linear,
        ] {
            assert_eq!(mode.to_string().parse(), Ok(mode));
        }
        assert!("smooth".parse::<Interpolation>().is_err());
        assert!("".parse::<Interpolation>().is_err());
    }

    #[test]
    fn fresh_curves_carry_no_preserved_fields() {
        assert!(Curve::default().extra.is_empty());
    }
}
