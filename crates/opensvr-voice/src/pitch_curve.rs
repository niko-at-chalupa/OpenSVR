//! Analytical pitch contour generation (portamento, vibrato, prediction base, auto mask).
//!
//! Ports OpenSV's `src/synthesis/PitchCurve.{h,cpp}` (namespace
//! `sv::synthesis`): the two-layer musical pitch model where an analytical
//! base curve (portamento transitions with Gaussian overshoot lobes, plus a
//! separate sinusoidal vibrato layer) underlies the neural residual predicted
//! by [`crate::PitchModel`]. All time parameters are in seconds, pitches in
//! MIDI semitones, phases in radians.
//!
//! Fidelity notes: the transition crossfade uses the legacy float-precision
//! pi (`3.14159`); only the automatic-mask raised cosine uses full `pi`.
//! Width floors (`secondsPerQuarter / 256` for transitions, `/ 128` for
//! vibrato fades) and the sigmoid `[-10, 10]` clamp are copied exactly.

use std::f64::consts::PI as F64_PI;

/// Legacy float-precision pi kept for analytical curve evaluation
/// (intentionally NOT `std::f64::consts::PI`: the C++ curves were tuned
/// against this literal).
#[allow(clippy::approx_constant)]
const CURVE_PI: f64 = 3.14159;

/// Transition duration scale relative to `t_f0_left` / `t_f0_right`.
const TRANSITION_WIDTH_SCALE: f64 = 0.3;

/// Offset of the Gaussian swing lobes in units of transition width.
const TRANSITION_SWING_OFFSET: f64 = 1.5;

/// Resolved per-note musical and phonetic attributes for curve evaluation.
///
/// Mirrors `PitchCurveNote` (`PitchCurve.h`): absolute note boundaries in
/// seconds, base pitch in semitones, tempo context, transition and vibrato
/// parameters. Field names follow the C++ originals.
#[derive(Debug, Clone)]
pub struct PitchCurveNote {
    /// Absolute start time of the note in seconds.
    pub start_seconds: f64,
    /// Absolute end time of the note in seconds (must exceed `start_seconds`).
    pub end_seconds: f64,
    /// Base musical pitch in semitones (60.0 = middle C).
    pub midi_pitch: f64,
    /// Tempo context: duration of one quarter note in seconds (`60 / BPM`).
    pub seconds_per_quarter: f64,
    /// Whether instant (AI automatic pitch) mode is enabled for this note.
    pub instant_mode: bool,
    /// Transition center offset in seconds relative to the note boundary.
    pub t_f0_offset: f64,
    /// Incoming transition (attack) width scale in seconds.
    pub t_f0_left: f64,
    /// Outgoing transition (release) width scale in seconds.
    pub t_f0_right: f64,
    /// Incoming transition overshoot depth in semitones.
    pub d_f0_left: f64,
    /// Outgoing transition overshoot depth in semitones.
    pub d_f0_right: f64,
    /// Vibrato start delay relative to note onset in seconds.
    pub t_f0_vbr_start: f64,
    /// Vibrato attack fade-in duration in seconds.
    pub t_f0_vbr_left: f64,
    /// Vibrato release fade-out duration before note offset in seconds.
    pub t_f0_vbr_right: f64,
    /// Peak-to-peak vibrato depth in semitones.
    pub d_f0_vbr: f64,
    /// Vibrato starting phase in radians.
    pub p_f0_vbr: f64,
    /// Vibrato oscillation frequency in Hertz.
    pub f_f0_vbr: f64,
    /// Rap intonation / speech inflection coefficient.
    pub rap_intonation: f64,
}

impl Default for PitchCurveNote {
    fn default() -> Self {
        Self {
            start_seconds: 0.0,
            end_seconds: 0.0,
            midi_pitch: 60.0,
            seconds_per_quarter: 0.5,
            instant_mode: true,
            t_f0_offset: 0.0,
            t_f0_left: 0.1,
            t_f0_right: 0.07,
            d_f0_left: 0.15,
            d_f0_right: 0.15,
            t_f0_vbr_start: 0.25,
            t_f0_vbr_left: 0.2,
            t_f0_vbr_right: 0.2,
            d_f0_vbr: 1.0,
            p_f0_vbr: 0.0,
            f_f0_vbr: 5.5,
            rap_intonation: 0.0,
        }
    }
}

/// Discrete temporal slice between notes or rests for piecewise interpolation.
#[derive(Debug, Clone)]
struct PitchSegment<'a> {
    start_seconds: f64,
    end_seconds: f64,
    midi_pitch: f64,
    /// `None` indicates a rest/silence segment.
    note: Option<&'a PitchCurveNote>,
}

/// Clamped logistic sigmoid: `1 / (1 + exp(-x))` on `[-10, 10]`.
fn sigmoid(value: f64) -> f64 {
    if value < -10.0 {
        return 0.0;
    }
    if value > 10.0 {
        return 1.0;
    }
    1.0 / (1.0 + (-value).exp())
}

/// Unnormalized symmetric Gaussian bell: `exp(-0.5 * (x / width)^2)`.
fn gaussian(position: f64, width: f64) -> f64 {
    let normalized = position / width;
    (-0.5 * normalized * normalized).exp()
}

/// Effective transition width with a quarter-note-relative floor.
fn transition_width(note: &PitchCurveNote, seconds: f64) -> f64 {
    (seconds * TRANSITION_WIDTH_SCALE).max(note.seconds_per_quarter / 256.0)
}

/// Partitions notes into alternating note/rest segments with quarter-note
/// silent padding before the first note and after the last, so attack and
/// release curves have defined boundary conditions in partial windows.
fn make_segments(notes: &[PitchCurveNote]) -> Vec<PitchSegment<'_>> {
    let mut segments = Vec::with_capacity(notes.len() * 2 + 1);
    let first = &notes[0];
    segments.push(PitchSegment {
        start_seconds: first.start_seconds - first.seconds_per_quarter,
        end_seconds: first.start_seconds,
        midi_pitch: first.midi_pitch,
        note: None,
    });
    for note in notes {
        if segments.last().expect("padding").end_seconds < note.start_seconds {
            let pitch = note.midi_pitch;
            let start = segments.last().expect("padding").end_seconds;
            segments.push(PitchSegment {
                start_seconds: start,
                end_seconds: note.start_seconds,
                midi_pitch: pitch,
                note: None,
            });
        }
        segments.push(PitchSegment {
            start_seconds: note.start_seconds,
            end_seconds: note.end_seconds,
            midi_pitch: note.midi_pitch,
            note: Some(note),
        });
    }
    let last = &notes[notes.len() - 1];
    segments.push(PitchSegment {
        start_seconds: last.end_seconds,
        end_seconds: last.end_seconds + last.seconds_per_quarter,
        midi_pitch: last.midi_pitch,
        note: None,
    });
    segments
}

/// Continuous pitch of one transition pair (rest attack, rest release, or
/// legato portamento with opposing Gaussian swing lobes) at `seconds`.
fn sample_pair(segments: &[PitchSegment<'_>], index: usize, seconds: f64) -> f64 {
    let current = &segments[index];
    if index + 1 == segments.len() {
        return current.midi_pitch;
    }
    let next = &segments[index + 1];
    let boundary_seconds = current.end_seconds;
    let pitch: f64;
    let mut outgoing_depth = 0.0;
    let mut incoming_depth = 0.0;
    let mut outgoing_width = 1.0;
    let mut incoming_width = 1.0;
    let mut offset = 0.0;

    match (current.note, next.note) {
        (None, _) => {
            let previous_pitch = if index == 0 {
                next.midi_pitch
            } else {
                segments[index - 1].midi_pitch
            };
            let progress = ((seconds - current.start_seconds)
                / (current.end_seconds - current.start_seconds))
                .clamp(0.0, 1.0);
            pitch = if seconds < boundary_seconds {
                previous_pitch
                    + (next.midi_pitch - previous_pitch) * sigmoid(20.0 * (progress - 0.5))
            } else {
                next.midi_pitch
            };
            if let Some(note) = next.note {
                incoming_width = transition_width(note, note.t_f0_left);
                incoming_depth = -note.d_f0_left;
                offset = TRANSITION_SWING_OFFSET * incoming_width;
            }
        }
        (Some(current_note), None) => {
            let following_pitch = if index + 2 < segments.len() {
                segments[index + 2].midi_pitch
            } else {
                current.midi_pitch
            };
            let progress = ((seconds - next.start_seconds)
                / (next.end_seconds - next.start_seconds))
                .clamp(0.0, 1.0);
            pitch = if seconds < boundary_seconds {
                current.midi_pitch
            } else {
                current.midi_pitch
                    + (following_pitch - current.midi_pitch) * sigmoid(20.0 * (progress - 0.5))
            };
            let outgoing_note = current_note;
            outgoing_width = transition_width(outgoing_note, outgoing_note.t_f0_right);
            outgoing_depth = -outgoing_note.d_f0_right;
            offset = -TRANSITION_SWING_OFFSET * outgoing_width;
        }
        (Some(current_note), Some(next_note)) => {
            outgoing_width = transition_width(current_note, current_note.t_f0_right);
            incoming_width = transition_width(next_note, next_note.t_f0_left);
            offset = -next_note.t_f0_offset;
            let relative_seconds = seconds - boundary_seconds + offset;
            pitch = current.midi_pitch
                + (next.midi_pitch - current.midi_pitch)
                    * sigmoid(3.0 * relative_seconds / (outgoing_width + incoming_width));
            let direction = if next.midi_pitch > current.midi_pitch {
                1.0
            } else {
                -1.0
            };
            outgoing_depth = -direction * current_note.d_f0_right;
            incoming_depth = direction * next_note.d_f0_left;
        }
    }

    let relative_seconds = seconds - boundary_seconds + offset;
    pitch
        + outgoing_depth
            * gaussian(
                relative_seconds + TRANSITION_SWING_OFFSET * outgoing_width,
                outgoing_width,
            )
        + incoming_depth
            * gaussian(
                relative_seconds - TRANSITION_SWING_OFFSET * incoming_width,
                incoming_width,
            )
}

/// Simplified pitch reference pair for neural prediction: no swing lobes or
/// offset nudges, and quarter-note-unclamped widths with a step limit at zero.
fn sample_prediction_pair(segments: &[PitchSegment<'_>], index: usize, seconds: f64) -> f64 {
    let current = &segments[index];
    if index + 1 == segments.len() {
        return current.midi_pitch;
    }
    let next = &segments[index + 1];
    if current.note.is_none() {
        let previous_pitch = if index == 0 {
            next.midi_pitch
        } else {
            segments[index - 1].midi_pitch
        };
        let progress = ((seconds - current.start_seconds)
            / (current.end_seconds - current.start_seconds))
            .clamp(0.0, 1.0);
        return previous_pitch + (next.midi_pitch - previous_pitch) * progress;
    }
    if next.note.is_none() {
        let following_pitch = if index + 2 < segments.len() {
            segments[index + 2].midi_pitch
        } else {
            current.midi_pitch
        };
        let progress = ((seconds - next.start_seconds) / (next.end_seconds - next.start_seconds))
            .clamp(0.0, 1.0);
        return current.midi_pitch + (following_pitch - current.midi_pitch) * progress;
    }
    let current_note = current.note.expect("note-note branch");
    let next_note = next.note.expect("note-note branch");
    let width = TRANSITION_WIDTH_SCALE * (current_note.t_f0_right + next_note.t_f0_left);
    let relative_seconds = seconds - current.end_seconds;
    let weight = if width > 0.0 {
        sigmoid(3.0 * relative_seconds / width)
    } else if relative_seconds < 0.0 {
        0.0
    } else if relative_seconds > 0.0 {
        1.0
    } else {
        0.5
    };
    current.midi_pitch + (next.midi_pitch - current.midi_pitch) * weight
}

/// Samples transition pairs over discrete frames with a raised-cosine
/// crossfade between the previous and current pair inside each segment.
fn render_pitch_base(
    notes: &[PitchCurveNote],
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
    prediction_reference: bool,
) -> Vec<f32> {
    let mut result = vec![0.0_f32; frame_count];
    if notes.is_empty() || frame_count == 0 {
        return result;
    }
    let segments = make_segments(notes);
    let sample: fn(&[PitchSegment<'_>], usize, f64) -> f64 = if prediction_reference {
        sample_prediction_pair
    } else {
        sample_pair
    };
    let mut segment_index = 0_usize;
    for (frame, slot) in result.iter_mut().enumerate() {
        let seconds = first_frame_seconds + frame as f64 * frame_interval_seconds;
        while segment_index + 1 < segments.len() && segments[segment_index].end_seconds <= seconds {
            segment_index += 1;
        }
        if segment_index == 0 {
            *slot = sample(&segments, segment_index, seconds) as f32;
            continue;
        }
        if segment_index + 1 == segments.len() {
            *slot = sample(&segments, segment_index - 1, seconds) as f32;
            continue;
        }
        let segment = &segments[segment_index];
        let progress = ((seconds - segment.start_seconds)
            / (segment.end_seconds - segment.start_seconds))
            .clamp(0.0, 1.0);
        let weight = 0.5 - 0.5 * (CURVE_PI * progress).cos();
        let incoming = sample(&segments, segment_index - 1, seconds);
        let outgoing = sample(&segments, segment_index, seconds);
        *slot = ((1.0 - weight) * incoming + weight * outgoing) as f32;
    }
    result
}

/// Renders analytical portamento transitions with Gaussian overshoot dynamics.
///
/// Evaluates sigmoid transitions between adjacent notes/rests plus swing
/// lobes scaled by `d_f0_left` / `d_f0_right`, in MIDI semitones per frame.
pub fn render_pitch_transitions(
    notes: &[PitchCurveNote],
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
) -> Vec<f32> {
    render_pitch_base(
        notes,
        first_frame_seconds,
        frame_interval_seconds,
        frame_count,
        false,
    )
}

/// Renders the neutral pitch reference contour for neural pitch prediction.
///
/// Unlike [`render_pitch_transitions`], this excludes vibrato and transition
/// overshoots: the clean baseline the neural model predicts residuals from.
pub fn render_pitch_prediction_base(
    notes: &[PitchCurveNote],
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
) -> Vec<f32> {
    render_pitch_base(
        notes,
        first_frame_seconds,
        frame_interval_seconds,
        frame_count,
        true,
    )
}

/// Earliest frame index whose timestamp is `>= seconds`.
fn frame_at_or_after(
    seconds: f64,
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
) -> usize {
    ((seconds - first_frame_seconds) / frame_interval_seconds)
        .ceil()
        .clamp(0.0, frame_count as f64) as usize
}

/// Renders the analytical vibrato modulation contour in semitones.
///
/// Each note contributes `0.5 * d_f0_vbr * envelope * sin(2*pi*f*t + phase)`
/// under a trapezoidal fade envelope, so user curves can scale vibrato depth
/// independently of portamento.
pub fn render_pitch_vibrato(
    notes: &[PitchCurveNote],
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
) -> Vec<f32> {
    let mut result = vec![0.0_f32; frame_count];
    for (index, note) in notes.iter().enumerate() {
        let delay = note.t_f0_vbr_start.max(0.0);
        if delay >= note.end_seconds - note.start_seconds {
            continue;
        }
        let start_seconds = note.start_seconds + delay;
        let mut end_seconds = note.end_seconds;
        if index + 1 < notes.len() && notes[index + 1].start_seconds == note.end_seconds {
            end_seconds += notes[index + 1].t_f0_offset;
        }
        let fade_in = note.t_f0_vbr_left.max(note.seconds_per_quarter / 128.0);
        let fade_out = note.t_f0_vbr_right.max(note.seconds_per_quarter / 128.0);
        let first_frame = frame_at_or_after(
            start_seconds,
            first_frame_seconds,
            frame_interval_seconds,
            frame_count,
        );
        let end_frame = frame_at_or_after(
            end_seconds,
            first_frame_seconds,
            frame_interval_seconds,
            frame_count,
        );
        for (offset, slot) in result[first_frame..end_frame].iter_mut().enumerate() {
            let frame = first_frame + offset;
            let seconds = first_frame_seconds + frame as f64 * frame_interval_seconds;
            let elapsed = seconds - start_seconds;
            let envelope = (elapsed / fade_in)
                .min((end_seconds - seconds) / fade_out)
                .clamp(0.0, 1.0);
            let phase = 2.0 * CURVE_PI * note.f_f0_vbr * elapsed + note.p_f0_vbr;
            *slot = (0.5 * note.d_f0_vbr * envelope * phase.sin()) as f32;
        }
    }
    result
}

/// Renders the smooth gating mask blending manual (`0.0`) and automatic
/// (`1.0`) pitch curves, with raised-cosine crossfades across note
/// boundaries and rests.
pub fn render_automatic_pitch_mask(
    notes: &[PitchCurveNote],
    first_frame_seconds: f64,
    frame_interval_seconds: f64,
    frame_count: usize,
) -> Vec<f32> {
    let mut result = vec![0.0_f32; frame_count];
    if notes.is_empty() {
        return result;
    }
    let raised_cosine = |position: f64| 0.5 - 0.5 * (F64_PI * position.clamp(0.0, 1.0)).cos();
    let mut note_index = 0_usize;
    for (frame, slot) in result.iter_mut().enumerate() {
        let seconds = first_frame_seconds + frame as f64 * frame_interval_seconds;
        while note_index + 1 < notes.len() && notes[note_index + 1].start_seconds <= seconds {
            note_index += 1;
        }
        let note = &notes[note_index];
        let current = f64::from(note.instant_mode);
        let mut mask = current;
        if note_index + 1 < notes.len()
            && note.end_seconds < seconds
            && note.end_seconds < notes[note_index + 1].start_seconds
        {
            let next = &notes[note_index + 1];
            let progress = (seconds - note.end_seconds) / (next.start_seconds - note.end_seconds);
            mask += (f64::from(next.instant_mode) - current) * raised_cosine(progress);
        } else {
            let width = ((note.end_seconds - note.start_seconds) * 0.5)
                .min(note.seconds_per_quarter * 0.125);
            if seconds < note.start_seconds + width {
                let mut previous = current;
                if note_index > 0 && notes[note_index - 1].end_seconds == note.start_seconds {
                    previous = f64::from(notes[note_index - 1].instant_mode);
                }
                mask = previous
                    + (current - previous)
                        * raised_cosine(0.5 + 0.5 * (seconds - note.start_seconds) / width);
            } else if seconds > note.end_seconds - width {
                let mut next = current;
                if note_index + 1 < notes.len()
                    && notes[note_index + 1].start_seconds == note.end_seconds
                {
                    next = f64::from(notes[note_index + 1].instant_mode);
                }
                mask += (next - current)
                    * raised_cosine(0.5 * (seconds - (note.end_seconds - width)) / width);
            }
        }
        *slot = mask as f32;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f64, end: f64, pitch: f64) -> PitchCurveNote {
        PitchCurveNote {
            start_seconds: start,
            end_seconds: end,
            midi_pitch: pitch,
            ..PitchCurveNote::default()
        }
    }

    #[test]
    fn sigmoid_clamps_and_centers() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(-11.0), 0.0);
        assert_eq!(sigmoid(11.0), 1.0);
        assert!((sigmoid(1.0) - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-15);
    }

    #[test]
    fn gaussian_peaks_at_zero() {
        assert_eq!(gaussian(0.0, 0.03), 1.0);
        assert!((gaussian(1.0, 1.0) - (-0.5_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn single_note_center_holds_its_pitch() {
        // One note [1, 2] at 60 st on a 5 ms grid from t = 0: the frame at
        // 1.5 s sits mid-note, past the attack swing and before release.
        let curve = render_pitch_transitions(&[note(1.0, 2.0, 60.0)], 0.0, 0.005, 400);
        assert!((curve[300] - 60.0).abs() < 1e-3, "got {}", curve[300]);
    }

    #[test]
    fn empty_notes_render_silence() {
        assert_eq!(render_pitch_transitions(&[], 0.0, 0.005, 8), vec![0.0; 8]);
        assert_eq!(
            render_pitch_prediction_base(&[], 0.0, 0.005, 8),
            vec![0.0; 8]
        );
        assert_eq!(render_pitch_vibrato(&[], 0.0, 0.005, 8), vec![0.0; 8]);
        assert_eq!(
            render_automatic_pitch_mask(&[], 0.0, 0.005, 8),
            vec![0.0; 8]
        );
        assert!(render_pitch_transitions(&[note(0.0, 1.0, 60.0)], 0.0, 0.005, 0).is_empty());
    }

    #[test]
    fn prediction_base_is_monotone_across_a_step() {
        // Notes 60 -> 64 with zero transition widths: the C++ step limit
        // holds 60 before the boundary, 64 at and after it.
        let first = PitchCurveNote {
            t_f0_right: 0.0,
            ..note(0.0, 1.0, 60.0)
        };
        let second = PitchCurveNote {
            t_f0_left: 0.0,
            ..note(1.0, 2.0, 64.0)
        };
        let curve = render_pitch_prediction_base(&[first, second], 0.0, 0.01, 200);
        assert_eq!(curve[99], 60.0);
        assert_eq!(curve[100], 62.0);
        assert_eq!(curve[101], 64.0);
    }

    #[test]
    fn vibrato_matches_hand_computed_peak() {
        // Note [0, 1], no start delay, 0.1 s fades, 1 st peak-to-peak at
        // 5.5 Hz: at t = 0.5 the envelope is 1 and the phase completes
        // 2.75 cycles, i.e. sin ~= -1, so the offset is ~= -0.5 st.
        let note = PitchCurveNote {
            t_f0_vbr_start: 0.0,
            t_f0_vbr_left: 0.1,
            t_f0_vbr_right: 0.1,
            ..note(0.0, 1.0, 60.0)
        };
        let curve = render_pitch_vibrato(&[note], 0.0, 0.005, 200);
        assert!((curve[100] + 0.5).abs() < 1e-4, "got {}", curve[100]);
        // Before the delay plus fade the envelope is partial; frame 0 is 0.
        assert_eq!(curve[0], 0.0);
    }

    #[test]
    fn vibrato_skips_notes_shorter_than_the_delay() {
        let note = PitchCurveNote {
            t_f0_vbr_start: 2.0,
            ..note(0.0, 1.0, 60.0)
        };
        assert_eq!(
            render_pitch_vibrato(&[note], 0.0, 0.01, 100),
            vec![0.0; 100]
        );
    }

    #[test]
    fn mask_is_uniform_for_uniform_modes() {
        let auto = note(1.0, 2.0, 60.0);
        assert_eq!(
            render_automatic_pitch_mask(&[auto], 0.0, 0.01, 300),
            vec![1.0; 300]
        );
        let manual = PitchCurveNote {
            instant_mode: false,
            ..note(1.0, 2.0, 60.0)
        };
        assert_eq!(
            render_automatic_pitch_mask(&[manual], 0.0, 0.01, 300),
            vec![0.0; 300]
        );
    }

    #[test]
    fn mask_blends_across_a_rest_gap() {
        // Auto note [0, 1], manual note [2, 3]: the rest gap crossfades with
        // a raised cosine, so the midpoint of the gap reads exactly 0.5.
        let first = note(0.0, 1.0, 60.0);
        let second = PitchCurveNote {
            instant_mode: false,
            ..note(2.0, 3.0, 64.0)
        };
        let mask = render_automatic_pitch_mask(&[first, second], 0.0, 0.01, 300);
        assert!((mask[150] - 0.5).abs() < 1e-6, "got {}", mask[150]);
        assert_eq!(mask[50], 1.0);
        assert_eq!(mask[250], 0.0);
    }
}
