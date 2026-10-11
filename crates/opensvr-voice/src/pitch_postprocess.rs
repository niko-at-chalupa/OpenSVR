//! Stable-center pitch correction for long notes.
//!
//! Ports OpenSV's `applyPitchPostprocess` (`src/synthesis/PitchPostprocess.cpp`,
//! namespace `sv::synthesis`): per-note recentering of the predicted curve
//! toward the score pitch, weighted by local smoothness so vibrato survives
//! and transitions are preserved. Silence notes are excluded; the fade is ten
//! frames on each side with a 0.3 s duration bias and a 0.2 st threshold.
//!
//! The frame interval is a dedicated [`PitchFrameInterval`] (never a bare
//! float): pitch runs on the model's 5 ms grid while the acoustic stage may
//! use another.

use opensvr_core::CancelToken;

use crate::{Result, invalid};

/// Frame interval of the pitch model grid (typically 5 ms).
///
/// Kept distinct from the acoustic interval so the two grids can never be
/// mixed up at a call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrameInterval(f32);

impl PitchFrameInterval {
    /// Builds an interval; rejects non-finite and non-positive values.
    pub fn new(seconds: f32) -> Result<Self> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(invalid(
                "Pitch correction: invalid frame interval or sequence size.",
            ));
        }
        Ok(Self(seconds))
    }

    /// Interval in seconds.
    pub fn seconds(self) -> f32 {
        self.0
    }
}

/// Corrects the stable center of long notes in place, keeping vibrato.
///
/// `notes` and `midi_pitch` share the padded model timeline: each note's
/// span is `durationSeconds` accumulated from time zero at the interval grid
/// (`start = seconds * inverse`, truncated). The middle half of each
/// non-silence note votes for a smoothness-weighted mean pitch; when the
/// score pitch differs by more than 0.2 st, the note is pulled toward it with
/// a duration-scaled correction faded over ten frames at each edge.
pub fn apply_pitch_postprocess(
    notes: &[crate::PitchNote],
    frame_interval: PitchFrameInterval,
    midi_pitch: &mut [f32],
    cancel: &CancelToken,
) -> Result<()> {
    if midi_pitch.len() > i32::MAX as usize {
        return Err(invalid(
            "Pitch correction: invalid frame interval or sequence size.",
        ));
    }
    if !midi_pitch.iter().all(|pitch| pitch.is_finite()) {
        return Err(invalid("Pitch correction: non-finite predicted pitch."));
    }
    let inverse_interval = 1.0 / f64::from(frame_interval.seconds());
    let mut seconds = 0.0_f64;
    for note in notes {
        if cancel.is_cancelled() {
            return Err(invalid("Pitch correction: cancelled."));
        }
        let duration = note.syllable.duration_seconds;
        let end_seconds = seconds + duration;
        let end_position = end_seconds * inverse_interval;
        if !duration.is_finite()
            || duration < 0.0
            || !end_position.is_finite()
            || end_position > f64::from(i32::MAX)
        {
            return Err(invalid(
                "Pitch correction: note timing is outside the supported range.",
            ));
        }
        // Truncation matches the C++ `static_cast<std::size_t>` of the
        // non-negative frame positions.
        let start = (seconds * inverse_interval) as usize;
        let end = end_position as usize;
        seconds = end_seconds;
        if note.is_silence {
            continue;
        }
        let middle_start = start + (end - start) / 4;
        let middle_end = (start + (end - start) * 3 / 4).min(midi_pitch.len());
        if middle_start >= middle_end {
            continue;
        }
        let mut weighted_pitch = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for (offset, &pitch) in midi_pitch[middle_start..middle_end].iter().enumerate() {
            let frame = middle_start + offset;
            if frame % 256 == 0 && cancel.is_cancelled() {
                return Err(invalid("Pitch correction: cancelled."));
            }
            let pitch = f64::from(pitch);
            let previous = if frame < 2 { frame } else { frame - 1 };
            let next = (frame + 1).min(midi_pitch.len() - 1);
            let weight = 1.0
                / ((f64::from(midi_pitch[previous]) - pitch).abs()
                    + 0.1
                    + (f64::from(midi_pitch[next]) - pitch).abs());
            weight_sum += weight;
            weighted_pitch += weight * pitch;
        }
        let offset = f64::from(note.syllable.midi_pitch) - weighted_pitch / weight_sum;
        if offset.abs() <= f64::from(0.2_f32) {
            continue;
        }
        let correction = offset * ((duration - f64::from(0.3_f32)) * 10.0).clamp(0.0, 1.0);
        let limit = end.min(midi_pitch.len());
        for (offset, slot) in midi_pitch[start..limit].iter_mut().enumerate() {
            let frame = start + offset;
            if frame % 256 == 0 && cancel.is_cancelled() {
                return Err(invalid("Pitch correction: cancelled."));
            }
            let left = ((frame - start) as f64 * 0.1).min(1.0);
            let right = ((end - frame) as f64 * 0.1).min(1.0);
            *slot = (f64::from(*slot) + left * correction * right) as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_g2p::TimingSyllable;

    fn note(midi: i32, seconds: f64, silence: bool) -> crate::PitchNote {
        crate::PitchNote {
            syllable: TimingSyllable {
                language: "japanese".to_owned(),
                phonemes: vec!["a".to_owned()],
                duration_seconds: seconds,
                midi_pitch: midi,
                is_continuation: false,
            },
            is_silence: silence,
            ..crate::PitchNote::default()
        }
    }

    fn interval() -> PitchFrameInterval {
        PitchFrameInterval::new(0.005).expect("valid interval")
    }

    #[test]
    fn flat_curve_at_score_pitch_is_unchanged() {
        let notes = vec![note(60, 1.0, false)];
        let mut pitch = vec![60.0_f32; 200];
        apply_pitch_postprocess(&notes, interval(), &mut pitch, &CancelToken::new())
            .expect("must correct");
        assert_eq!(pitch, vec![60.0; 200]);
    }

    #[test]
    fn long_note_pulls_flat_curve_to_the_score() {
        // 1 s note at score 60, prediction flat at 62: offset 2 st with full
        // duration scaling; frame 100 sits mid-note where both fades are 1.
        let notes = vec![note(60, 1.0, false)];
        let mut pitch = vec![62.0_f32; 200];
        apply_pitch_postprocess(&notes, interval(), &mut pitch, &CancelToken::new())
            .expect("must correct");
        assert!((pitch[100] - 60.0).abs() < 1e-6, "got {}", pitch[100]);
        // Edge fades: frame 0 has left = 0, so it never moves.
        assert_eq!(pitch[0], 62.0);
    }

    #[test]
    fn small_offsets_and_silences_are_skipped() {
        // 0.1 st is under the 0.2 threshold; silence notes never correct.
        let notes = vec![note(60, 1.0, false), note(70, 0.5, true)];
        let mut pitch = vec![60.1_f32; 300];
        apply_pitch_postprocess(&notes, interval(), &mut pitch, &CancelToken::new())
            .expect("must correct");
        assert_eq!(pitch, vec![60.1; 300]);
    }

    #[test]
    fn rejects_bad_intervals_and_non_finite_pitch() {
        assert_eq!(
            PitchFrameInterval::new(0.0).unwrap_err().to_string(),
            "Pitch correction: invalid frame interval or sequence size."
        );
        assert_eq!(
            PitchFrameInterval::new(f32::NAN).unwrap_err().to_string(),
            "Pitch correction: invalid frame interval or sequence size."
        );
        let notes = vec![note(60, 1.0, false)];
        let mut pitch = vec![60.0_f32; 8];
        pitch[3] = f32::INFINITY;
        assert_eq!(
            apply_pitch_postprocess(&notes, interval(), &mut pitch, &CancelToken::new())
                .unwrap_err()
                .to_string(),
            "Pitch correction: non-finite predicted pitch."
        );
    }

    #[test]
    fn cancelled_token_aborts() {
        let token = CancelToken::new();
        token.cancel();
        let notes = vec![note(60, 1.0, false)];
        let mut pitch = vec![60.0_f32; 8];
        assert_eq!(
            apply_pitch_postprocess(&notes, interval(), &mut pitch, &token)
                .unwrap_err()
                .to_string(),
            "Pitch correction: cancelled."
        );
    }
}
