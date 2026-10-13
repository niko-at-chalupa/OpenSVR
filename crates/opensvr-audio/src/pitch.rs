//! Phrase pitch assembly: continuation merging and log-F0 mapping.
//!
//! Ports two pieces of OpenSV's `src/audio/ProjectRenderer.cpp`:
//! [`merge_continuation_phones`] (legato `-` notes reusing the previous phoneme
//! extend its frames instead of starting new ones) and the frame loop of
//! [`build_log_f0`] (`buildPitch`: analytical transitions plus the neural
//! residual sampled onto the acoustic grid, converted to natural-log Hz).
//!
//! The project model has no parameter curves, vocal modes or per-note pitch
//! attributes yet (Phase 1), so every note resolves to the C++ defaults for a
//! default project: instant (neural) pitch mode, generated transition widths
//! (`seconds_per_quarter * 0.2`), no overshoot, no vibrato (`dF0Vbr` falls back
//! to its generated `0.0`), unity vibrato envelope and no cents or rap offsets.
//! The seam left for Phase 1 is per-note attribute and curve sampling inside
//! [`build_log_f0`]: the function already takes per-note detune and tempo, and
//! every other term is a marked constant below.

use opensvr_g2p::{PhonemeDuration, TimingSyllable};
use opensvr_voice::{
    PitchCurveNote, TimedPhoneme, render_automatic_pitch_mask, render_pitch_prediction_base,
    render_pitch_transitions, render_pitch_vibrato,
};

use crate::{BackendError, TimedNote};

/// Merges legato continuations into the previous identical phoneme.
///
/// Mirrors `mergeContinuationPhones`: when `durations[index]` belongs to a
/// continuation (`-`) syllable and its quantized phoneme repeats the previous
/// language and symbol, its frames extend the previous entry instead of
/// starting a new one. A durations/phonemes length mismatch leaves the
/// phonemes unchanged, exactly as in C++.
pub fn merge_continuation_phones(
    durations: &[PhonemeDuration],
    syllables: &[TimingSyllable],
    phonemes: &[TimedPhoneme],
) -> Vec<TimedPhoneme> {
    if durations.len() != phonemes.len() {
        return phonemes.to_vec();
    }
    let mut merged: Vec<TimedPhoneme> = Vec::with_capacity(phonemes.len());
    for (duration, phoneme) in durations.iter().zip(phonemes) {
        let is_continuation = syllables
            .get(duration.syllable_index)
            .is_some_and(|syllable| syllable.is_continuation);
        if is_continuation
            && let Some(back) = merged.last_mut()
            && back.language == phoneme.language
            && back.symbol == phoneme.symbol
            && let Some(sum) = back.frame_count.checked_add(phoneme.frame_count)
        {
            back.frame_count = sum;
            continue;
        }
        merged.push(phoneme.clone());
    }
    merged
}

/// Assembles per-frame log-F0 for one phrase's acoustic frames.
///
/// `notes` are the score notes (absolute seconds, transposed pitch plus detune);
/// `start_seconds` is the synthesis start (first note minus release context) and
/// `frame_interval_seconds` the acoustic grid. `automatic_pitch` is the padded
/// pitch-model curve starting at `pitch_start_seconds` on `pitch_frame_interval_seconds`.
/// An empty prediction contributes no residual, as in C++.
///
/// Errors when the assembled pitch is non-finite, mirroring `buildPitch`.
#[allow(clippy::too_many_arguments)]
pub fn build_log_f0(
    notes: &[TimedNote<'_>],
    start_seconds: f64,
    frame_interval_seconds: f32,
    frame_count: usize,
    automatic_pitch: &[f32],
    pitch_start_seconds: f64,
    pitch_frame_interval_seconds: f32,
) -> Result<Vec<f32>, BackendError> {
    let mut editor_notes = Vec::with_capacity(notes.len());
    let mut prediction_notes = Vec::with_capacity(notes.len());
    for note in notes {
        let quarter = note.seconds_per_quarter;
        // Generated transition widths for instant-mode notes without overrides.
        let width = quarter * 0.2;
        let editor = PitchCurveNote {
            start_seconds: note.start,
            end_seconds: note.end,
            midi_pitch: f64::from(note.pitch) + note.detune_cents / 100.0,
            seconds_per_quarter: quarter,
            instant_mode: true,
            t_f0_offset: 0.0,
            t_f0_left: width,
            t_f0_right: width,
            d_f0_left: 0.0,
            d_f0_right: 0.0,
            t_f0_vbr_start: 0.25,
            t_f0_vbr_left: 0.2,
            t_f0_vbr_right: 0.2,
            // Generated fallback for instant mode: no vibrato without curves.
            d_f0_vbr: 0.0,
            p_f0_vbr: 0.0,
            f_f0_vbr: 5.5,
            rap_intonation: 0.0,
        };
        let mut prediction = editor.clone();
        // The prediction reference hears the plain score pitch (no detune).
        prediction.midi_pitch = f64::from(note.pitch);
        editor_notes.push(editor);
        prediction_notes.push(prediction);
    }

    let interval = f64::from(frame_interval_seconds);
    let first_frame_seconds = start_seconds + 0.5 * interval;
    let transitions = render_pitch_transitions(&editor_notes, first_frame_seconds, interval, frame_count);
    // Unity vibrato envelope (no `vibratoEnv` curves yet); depth is zero above,
    // so this renders zeros while keeping the wiring faithful.
    let vibrato = render_pitch_vibrato(&editor_notes, first_frame_seconds, interval, frame_count);
    // All notes are instant-mode, so the mask renders all ones.
    let mask = render_automatic_pitch_mask(&editor_notes, first_frame_seconds, interval, frame_count);

    let mut residual = Vec::new();
    if !automatic_pitch.is_empty() {
        let base = render_pitch_prediction_base(
            &prediction_notes,
            pitch_start_seconds,
            f64::from(pitch_frame_interval_seconds),
            automatic_pitch.len(),
        );
        residual.reserve(automatic_pitch.len());
        for (predicted, reference) in automatic_pitch.iter().zip(&base) {
            residual.push(predicted - reference);
        }
    }

    let pitch_interval = f64::from(pitch_frame_interval_seconds);
    let last_model_frame = residual.len().saturating_sub(1) as f64;
    let mut log_f0 = Vec::with_capacity(frame_count);
    let mut note_index = 0_usize;
    for frame in 0..frame_count {
        // Sampled at the frame centre so float frame periods cannot push a
        // transition on an exact score boundary into the following frame.
        let seconds = start_seconds + (frame as f64 + 0.5) * interval;
        while note_index + 1 < notes.len() && seconds >= notes[note_index + 1].start {
            note_index += 1;
        }
        // No imported system contour and no group pitch correction yet.
        let mut automatic_delta = 0.0;
        if !residual.is_empty() {
            let model_frame =
                ((seconds - pitch_start_seconds) / pitch_interval).clamp(0.0, last_model_frame);
            let left = model_frame as usize;
            let right = (left + 1).min(residual.len() - 1);
            let fraction = model_frame - left as f64;
            automatic_delta = f64::from(residual[left])
                + f64::from(residual[right] - residual[left]) * fraction;
        }
        let pitch = f64::from(transitions[frame]) + f64::from(mask[frame]) * automatic_delta
            + f64::from(vibrato[frame]);
        let logarithmic = 440.0_f64.ln() + (pitch - 69.0) * std::f64::consts::LN_2 / 12.0;
        let frequency = (logarithmic as f32).exp();
        if !pitch.is_finite() || !frequency.is_finite() || frequency <= 0.0 {
            return Err(BackendError(
                "pitch automation produces an invalid fundamental frequency.".into(),
            ));
        }
        log_f0.push(logarithmic as f32);
    }
    Ok(log_f0)
}

#[cfg(test)]
mod tests {
    use opensvr_voice::midi_to_log_f0;

    use super::*;

    fn note(start: f64, end: f64, pitch: u8) -> TimedNote<'static> {
        TimedNote {
            start,
            end,
            pitch,
            detune_cents: 0.0,
            seconds_per_quarter: 0.5,
            lyrics: "la",
            phonemes: "",
        }
    }

    fn duration(
        language: &str,
        symbol: &str,
        syllable_index: usize,
        seconds: f64,
    ) -> PhonemeDuration {
        PhonemeDuration {
            language: language.to_owned(),
            symbol: symbol.to_owned(),
            syllable_index,
            timing_interval_index: 0,
            duration_seconds: seconds,
        }
    }

    fn timed(language: &str, symbol: &str, frame_count: usize) -> TimedPhoneme {
        TimedPhoneme {
            language: language.to_owned(),
            symbol: symbol.to_owned(),
            frame_count,
        }
    }

    fn syllable(continuation: bool) -> TimingSyllable {
        TimingSyllable {
            language: "japanese".to_owned(),
            phonemes: vec!["a".to_owned()],
            duration_seconds: 0.4,
            midi_pitch: 60,
            is_continuation: continuation,
        }
    }

    #[test]
    fn continuations_extend_matching_phonemes() {
        let syllables = vec![syllable(false), syllable(true), syllable(true)];
        let durations = vec![
            duration("japanese", "a", 0, 0.4),
            duration("japanese", "a", 1, 0.4),
            duration("japanese", "k", 2, 0.2),
        ];
        let phonemes = vec![timed("japanese", "a", 80), timed("japanese", "a", 80), timed("japanese", "k", 40)];
        let merged = merge_continuation_phones(&durations, &syllables, &phonemes);
        assert_eq!(
            merged,
            vec![timed("japanese", "a", 160), timed("japanese", "k", 40)],
            "the repeated legato vowel extends the previous frames; the new consonant does not"
        );
    }

    #[test]
    fn plain_notes_and_mismatches_pass_through() {
        let syllables = vec![syllable(false), syllable(false)];
        let durations = vec![
            duration("japanese", "a", 0, 0.4),
            duration("japanese", "a", 1, 0.4),
        ];
        let phonemes = vec![timed("japanese", "a", 80), timed("japanese", "a", 80)];
        assert_eq!(
            merge_continuation_phones(&durations, &syllables, &phonemes),
            phonemes,
            "no continuation, no merge"
        );
        // Length mismatch leaves the phonemes untouched.
        assert_eq!(
            merge_continuation_phones(&durations[..1], &syllables, &phonemes),
            phonemes
        );
    }

    #[test]
    fn silent_prediction_holds_the_score_pitch_mid_note() {
        // One note [1, 2] at MIDI 60 on the 5 ms grid from t = 0. With the
        // prediction equal to its own reference base the residual is zero, so
        // the centre frame reduces to the analytical transition (60 st).
        let notes = [note(1.0, 2.0, 60)];
        let prediction = render_pitch_prediction_base(
            &[PitchCurveNote {
                start_seconds: 1.0,
                end_seconds: 2.0,
                ..PitchCurveNote::default()
            }],
            0.0,
            0.005,
            400,
        );
        let log_f0 = build_log_f0(&notes, 0.0, 0.005, 400, &prediction, 0.0, 0.005).unwrap();
        assert_eq!(log_f0.len(), 400);
        assert!(log_f0.iter().all(|value| value.is_finite()));
        let expected = midi_to_log_f0(60.0);
        assert!(
            (log_f0[300] - expected).abs() < 1e-3,
            "got {}, want {expected}",
            log_f0[300]
        );
    }

    #[test]
    fn empty_prediction_contributes_no_residual() {
        let notes = [note(0.0, 0.5, 64)];
        let log_f0 = build_log_f0(&notes, 0.0, 0.005, 100, &[], 0.0, 0.005).unwrap();
        assert_eq!(log_f0.len(), 100);
        assert!(log_f0.iter().all(|value| value.is_finite()));
    }
}
