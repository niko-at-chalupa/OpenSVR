//! Phrase pitch assembly: continuation merging and log-F0 mapping.
//!
//! Ports four pieces of OpenSV's `src/audio/ProjectRenderer.cpp`:
//! [`merge_continuation_phones`] (legato `-` notes reusing the previous phoneme
//! extend its frames instead of starting new ones), [`local_curve_position`]
//! (group-local curve positions with saturating overflow guards),
//! [`build_vibrato_envelope`] (per-frame `vibratoEnv` sampling, default `1.0`),
//! [`build_vocal_mode_weights`] (per-frame vocal-mode style weights, empty
//! when the phrase has no modes) and the frame loop of [`build_log_f0`]
//! (`buildPitch`: analytical transitions plus the neural residual sampled onto
//! the acoustic grid, converted to natural-log Hz).
//!
//! Per-note attributes resolve exactly as in C++ (`resolvePitchAttribute` /
//! `resolveRapAttribute`): explicit note attributes win, instant-mode notes
//! fall back to generated widths (`seconds_per_quarter * 0.2`) or system
//! attributes for rap, and manual notes fall back to the placing reference's
//! `voicePitch` or the C++ manual default. Group `pitchDelta` curves add cents
//! offsets alongside the placing reference's imported `systemPitchDelta`
//! contour, exactly as in `buildPitch`: a non-empty system contour also gates
//! the neural residual off for its group, since the editor already generated
//! pitch motion there.

use opensvr_core::{Blick, GroupReference, Note, PitchAttribute, sample_curve};
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

/// Group-local curve position for a frame's absolute [`Blick`] position.
///
/// Mirrors `localCurvePosition`: parameter curves hold their endpoints outside
/// their stored range, and the subtraction saturates instead of overflowing
/// when the time offset would push the result past [`Blick::MIN`]/[`Blick::MAX`].
pub fn local_curve_position(absolute: Blick, time_offset: Blick) -> Blick {
    if time_offset > 0 && absolute < Blick::MIN + time_offset {
        return Blick::MIN;
    }
    if time_offset < 0 && absolute > Blick::MAX + time_offset {
        return Blick::MAX;
    }
    absolute - time_offset
}

/// Resolves one pitch attribute for a note, mirroring `resolvePitchAttribute`.
///
/// Explicit note attributes win; instant-mode notes fall back to the generated
/// value when one is given, otherwise to the placing reference's `voicePitch`
/// or the C++ manual default.
fn resolve_pitch(
    note: &Note,
    voice_pitch: &opensvr_core::PitchAttributes,
    field: PitchAttribute,
    manual_default: f64,
    generated: Option<f64>,
) -> f64 {
    if let Some(value) = note.attributes.get(field) {
        return value;
    }
    if note.instant_mode {
        if let Some(value) = generated {
            return value;
        }
    }
    voice_pitch.get(field).unwrap_or(manual_default)
}

/// Resolves one rap attribute for a note, mirroring `resolveRapAttribute`.
///
/// Explicit note attributes win; instant-mode notes fall back to the imported
/// system attributes, otherwise to `0.0`.
fn resolve_rap(note: &Note, field: PitchAttribute) -> f64 {
    if let Some(value) = note.attributes.get(field) {
        return value;
    }
    if note.instant_mode {
        if let Some(value) = note.system_attributes.get(field) {
            return value;
        }
    }
    0.0
}

/// Builds the analytical and prediction [`PitchCurveNote`]s for one score note.
///
/// The prediction reference hears the plain score pitch with generated widths
/// (explicit `tF0Left`/`tF0Right` only); the editor curve adds detune, rap
/// tone and every resolved attribute, exactly as in `buildPitch`.
fn curve_notes(note: &TimedNote<'_>) -> (PitchCurveNote, PitchCurveNote) {
    let score = note.note;
    let voice_pitch = &note.reference.voice_pitch;
    let quarter = note.seconds_per_quarter;
    let generated_width = quarter * 0.2;
    let prediction = PitchCurveNote {
        start_seconds: note.start,
        end_seconds: note.end,
        midi_pitch: f64::from(note.pitch),
        seconds_per_quarter: quarter,
        instant_mode: score.instant_mode,
        t_f0_left: score.attributes.t_f0_left.unwrap_or(generated_width),
        t_f0_right: score.attributes.t_f0_right.unwrap_or(generated_width),
        ..PitchCurveNote::default()
    };
    let mut editor = prediction.clone();
    editor.midi_pitch += note.detune_cents / 100.0;
    if score.musical_type == "rap" {
        editor.midi_pitch += resolve_rap(score, PitchAttribute::RTone);
        editor.rap_intonation = resolve_rap(score, PitchAttribute::RIntonation);
    }
    editor.t_f0_offset = resolve_pitch(
        score,
        voice_pitch,
        PitchAttribute::TF0Offset,
        0.0,
        Some(0.0),
    );
    editor.t_f0_left = resolve_pitch(
        score,
        voice_pitch,
        PitchAttribute::TF0Left,
        0.1,
        Some(generated_width),
    );
    editor.t_f0_right = resolve_pitch(
        score,
        voice_pitch,
        PitchAttribute::TF0Right,
        0.07,
        Some(generated_width),
    );
    editor.d_f0_left = resolve_pitch(score, voice_pitch, PitchAttribute::DF0Left, 0.15, Some(0.0));
    editor.d_f0_right = resolve_pitch(
        score,
        voice_pitch,
        PitchAttribute::DF0Right,
        0.15,
        Some(0.0),
    );
    editor.t_f0_vbr_start =
        resolve_pitch(score, voice_pitch, PitchAttribute::TF0VbrStart, 0.25, None);
    editor.t_f0_vbr_left = resolve_pitch(score, voice_pitch, PitchAttribute::TF0VbrLeft, 0.2, None);
    editor.t_f0_vbr_right =
        resolve_pitch(score, voice_pitch, PitchAttribute::TF0VbrRight, 0.2, None);
    editor.d_f0_vbr = resolve_pitch(score, voice_pitch, PitchAttribute::DF0Vbr, 1.0, Some(0.0));
    editor.p_f0_vbr = resolve_pitch(score, voice_pitch, PitchAttribute::PF0Vbr, 0.0, None);
    editor.f_f0_vbr = resolve_pitch(score, voice_pitch, PitchAttribute::FF0Vbr, 5.5, None);
    (editor, prediction)
}

/// Samples the per-frame vibrato depth envelope for a note phrase.
///
/// Mirrors `buildVibratoEnvelope`: each frame samples its note's `vibratoEnv`
/// curve at the group-local position (default `1.0`), advancing to the next
/// note at its start. Unlike [`build_log_f0`], frames sample at their start,
/// exactly as in C++.
pub fn build_vibrato_envelope(
    notes: &[TimedNote<'_>],
    start_seconds: f64,
    frame_interval_seconds: f32,
    frame_count: usize,
) -> Vec<f32> {
    let mut envelope = vec![1.0_f32; frame_count];
    let mut note_index = 0_usize;
    let interval = f64::from(frame_interval_seconds);
    for (frame, slot) in envelope.iter_mut().enumerate() {
        let seconds = start_seconds + frame as f64 * interval;
        while note_index + 1 < notes.len() && seconds >= notes[note_index + 1].start {
            note_index += 1;
        }
        let note = &notes[note_index];
        let position = local_curve_position(
            note.tempo_map.seconds_to_blick(seconds),
            note.reference.time_offset,
        );
        *slot = sample_curve(&note.group.vibrato_env, position, 1.0) as f32;
    }
    envelope
}

/// Whether any note of a phrase carries vocal-mode automation.
///
/// Mirrors `phraseHasVocalModes`: the track's main reference or any note's own
/// reference has a preset or param overrides, or any placing group defines
/// vocal modes. The track side arrives per note as [`TimedNote::main_reference`]
/// (all notes of a phrase share one track), so no wider trait is needed.
pub fn phrase_has_vocal_modes(notes: &[TimedNote<'_>]) -> bool {
    let Some(first) = notes.first() else {
        return false;
    };
    if reference_has_vocal_modes(first.main_reference) {
        return true;
    }
    notes
        .iter()
        .any(|note| reference_has_vocal_modes(note.reference) || !note.group.vocal_modes.is_empty())
}

fn reference_has_vocal_modes(reference: &GroupReference) -> bool {
    !reference.vocal_mode_preset.is_empty() || !reference.vocal_mode_params.is_empty()
}

/// Builds per-frame vocal-mode style weights for a note phrase.
///
/// Mirrors `buildVocalModeWeights`: an empty vector when the phrase has no
/// modes (the acoustic model then sings the database default style),
/// otherwise one clamped weight per mode name per frame — reference presets
/// and params first, then the placing group's per-mode curves or amounts at
/// the frame's group-local position. Vocal-mode inheritance resolves against
/// [`TimedNote::main_reference`] by pointer identity, exactly as C++ compares
/// against `&track.mainRef`.
pub fn build_vocal_mode_weights(
    notes: &[TimedNote<'_>],
    start_seconds: f64,
    frame_interval_seconds: f32,
    frame_count: usize,
    mode_names: &[String],
) -> Result<Vec<f32>, BackendError> {
    if !phrase_has_vocal_modes(notes) {
        return Ok(Vec::new());
    }
    if mode_names.is_empty() {
        return Err(BackendError(
            "The selected voice does not expose vocal-mode style vectors required by this project."
                .into(),
        ));
    }
    if mode_names.len() > (128 << 20) / size_of::<f32>() / frame_count.max(1) {
        return Err(BackendError(
            "Vocal-mode automation exceeds the phrase memory limit.".into(),
        ));
    }
    let mode_index = |name: &str| mode_names.iter().position(|mode| mode == name);
    let mut weights = vec![0.0_f32; frame_count * mode_names.len()];
    let mut note_index = 0_usize;
    let interval = f64::from(frame_interval_seconds);
    for frame in 0..frame_count {
        // Sampled at the frame centre, like the pitch builder.
        let seconds = start_seconds + (frame as f64 + 0.5) * interval;
        while note_index + 1 < notes.len() && seconds >= notes[note_index + 1].start {
            note_index += 1;
        }
        let note = &notes[note_index];
        let settings = {
            let settings = note.reference;
            if std::ptr::eq(settings, note.main_reference) {
                if settings.vocal_mode_inherited {
                    None
                } else {
                    Some(settings)
                }
            } else if settings.vocal_mode_inherited {
                if note.main_reference.vocal_mode_inherited {
                    None
                } else {
                    Some(note.main_reference)
                }
            } else {
                Some(settings)
            }
        };
        let row = &mut weights[frame * mode_names.len()..(frame + 1) * mode_names.len()];
        if let Some(settings) = settings {
            if !settings.vocal_mode_preset.is_empty()
                && settings.vocal_mode_preset != "Default"
                && settings.vocal_mode_preset != "default"
            {
                let Some(index) = mode_index(&settings.vocal_mode_preset) else {
                    return Err(BackendError(format!(
                        "Vocal-mode preset '{}' is not a style exposed by the selected voice.",
                        settings.vocal_mode_preset
                    )));
                };
                row[index] = 1.0;
            }
            for (name, amount) in &settings.vocal_mode_params {
                let Some(index) = mode_index(name) else {
                    return Err(BackendError(format!(
                        "Vocal mode '{name}' is not exposed by the selected voice."
                    )));
                };
                row[index] = amount.clamp(0.0, 1.0) as f32;
            }
        }
        let position = local_curve_position(
            note.tempo_map.seconds_to_blick(seconds),
            note.reference.time_offset,
        );
        for (name, setting) in &note.group.vocal_modes {
            let Some(index) = mode_index(name) else {
                return Err(BackendError(format!(
                    "Vocal-mode curve '{name}' is not exposed by the selected voice."
                )));
            };
            row[index] = if !setting.enabled {
                0.0
            } else if setting.has_curve {
                sample_curve(&setting.curve, position, setting.amount).clamp(0.0, 1.0) as f32
            } else {
                setting.amount.clamp(0.0, 1.0) as f32
            };
        }
    }
    Ok(weights)
}

/// Assembles per-frame log-F0 for one phrase's acoustic frames.
///
/// `notes` are the score notes (absolute seconds, transposed pitch plus detune);
/// `start_seconds` is the synthesis start (first note minus release context) and
/// `frame_interval_seconds` the acoustic grid. `automatic_pitch` is the padded
/// pitch-model curve starting at `pitch_start_seconds` on `pitch_frame_interval_seconds`.
/// An empty prediction contributes no residual, as in C++.
///
/// Per-note attributes, the group `pitchDelta` curve, the reference
/// `systemPitchDelta` contour and rap intonation feed
/// the assembly exactly as in `buildPitch`; the analytical vibrato layer is
/// scaled by [`build_vibrato_envelope`]. A non-empty system contour gates the
/// neural residual off for its group (the editor already generated pitch
/// motion there), so the prediction only applies where no contour was
/// imported.
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
        let (editor, prediction) = curve_notes(note);
        editor_notes.push(editor);
        prediction_notes.push(prediction);
    }

    let interval = f64::from(frame_interval_seconds);
    let first_frame_seconds = start_seconds + 0.5 * interval;
    let transitions =
        render_pitch_transitions(&editor_notes, first_frame_seconds, interval, frame_count);
    let vibrato = render_pitch_vibrato(&editor_notes, first_frame_seconds, interval, frame_count);
    let mask =
        render_automatic_pitch_mask(&editor_notes, first_frame_seconds, interval, frame_count);
    let envelope = build_vibrato_envelope(
        notes,
        first_frame_seconds,
        frame_interval_seconds,
        frame_count,
    );

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
        let note = &notes[note_index];
        let position = local_curve_position(
            note.tempo_map.seconds_to_blick(seconds),
            note.reference.time_offset,
        );
        // SVP's `systemPitchDelta` is already the source editor's generated
        // pitch contour: adding a second neural prediction over it
        // double-counts natural pitch motion, so the residual only applies
        // where the group imported no system contour, as in `buildPitch`.
        let mut automatic_delta = 0.0;
        if note.reference.system_pitch_delta.points.is_empty() && !residual.is_empty() {
            let model_frame =
                ((seconds - pitch_start_seconds) / pitch_interval).clamp(0.0, last_model_frame);
            let left = model_frame as usize;
            let right = (left + 1).min(residual.len() - 1);
            let fraction = model_frame - left as f64;
            automatic_delta =
                f64::from(residual[left]) + f64::from(residual[right] - residual[left]) * fraction;
        }
        // The imported system contour rides under the group's editable pitch
        // correction, in the C++ sampling order: both hold their endpoints
        // outside their stored range via the shared curve sampler.
        let cents = sample_curve(&note.reference.system_pitch_delta, position, 0.0)
            + sample_curve(&note.group.pitch_delta, position, 0.0);
        let note_progress = ((seconds - note.start) / (note.end - note.start)).clamp(0.0, 1.0);
        let rap_intonation = if note.note.musical_type == "rap" {
            editor_notes[note_index].rap_intonation * (note_progress - 0.5)
        } else {
            0.0
        };
        let pitch = f64::from(transitions[frame])
            + f64::from(mask[frame]) * automatic_delta
            + f64::from(envelope[frame]) * f64::from(vibrato[frame])
            + cents / 100.0
            + rap_intonation;
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
    use opensvr_core::{
        AutomationPoint, Curve, Extra, Interpolation, Note, NoteGroup, PitchAttributes, TempoMap,
        VocalModeSetting,
    };
    use opensvr_voice::midi_to_log_f0;

    use super::*;

    static SCORE: std::sync::LazyLock<Note> = std::sync::LazyLock::new(|| Note {
        onset: 0,
        duration: 1,
        pitch: 60,
        lyrics: "la".into(),
        phonemes: String::new(),
        detune: 0.0,
        instant_mode: true,
        musical_type: "singing".into(),
        accent: String::new(),
        attributes: PitchAttributes::default(),
        system_attributes: PitchAttributes::default(),
        extra: Extra::default(),
    });
    static GROUP: std::sync::LazyLock<NoteGroup> = std::sync::LazyLock::new(NoteGroup::default);
    static REFERENCE: std::sync::LazyLock<GroupReference> =
        std::sync::LazyLock::new(GroupReference::default);
    static TEMPO: std::sync::LazyLock<TempoMap> = std::sync::LazyLock::new(TempoMap::default);

    fn note(start: f64, end: f64, pitch: u8) -> TimedNote<'static> {
        timed_note(
            start, end, pitch, &SCORE, &GROUP, &REFERENCE, &REFERENCE, &TEMPO,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn timed_note<'a>(
        start: f64,
        end: f64,
        pitch: u8,
        score: &'a Note,
        group: &'a NoteGroup,
        reference: &'a GroupReference,
        main: &'a GroupReference,
        tempo: &'a TempoMap,
    ) -> TimedNote<'a> {
        TimedNote {
            start,
            end,
            pitch,
            detune_cents: 0.0,
            seconds_per_quarter: 0.5,
            lyrics: "la",
            phonemes: "",
            note: score,
            group,
            reference,
            main_reference: main,
            tempo_map: tempo,
            window_start: 0.0,
            window_end: None,
        }
    }

    fn linear_curve(points: &[(Blick, f64)]) -> Curve {
        Curve {
            mode: Interpolation::Linear,
            points: points
                .iter()
                .map(|&(position, value)| AutomationPoint { position, value })
                .collect(),
            extra: Extra::default(),
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
        let phonemes = vec![
            timed("japanese", "a", 80),
            timed("japanese", "a", 80),
            timed("japanese", "k", 40),
        ];
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

    #[test]
    fn local_positions_saturate_at_the_blick_edges() {
        assert_eq!(local_curve_position(100, 0), 100);
        assert_eq!(local_curve_position(100, 30), 70);
        assert_eq!(local_curve_position(100, -30), 130);
        // A positive offset past the minimum holds the minimum ...
        assert_eq!(local_curve_position(Blick::MIN + 5, 10), Blick::MIN);
        assert_eq!(local_curve_position(Blick::MIN + 10, 10), Blick::MIN);
        // ... and a negative offset past the maximum holds the maximum.
        assert_eq!(local_curve_position(Blick::MAX - 5, -10), Blick::MAX);
        assert_eq!(local_curve_position(Blick::MAX - 10, -10), Blick::MAX);
    }

    #[test]
    fn vibrato_envelopes_default_to_unity() {
        let notes = [note(0.0, 0.5, 60)];
        assert_eq!(
            build_vibrato_envelope(&notes, 0.0, 0.005, 100),
            vec![1.0; 100]
        );
    }

    #[test]
    fn vibrato_envelopes_sample_the_group_curve() {
        use opensvr_core::BLICKS_PER_QUARTER as Q;
        // At 120 BPM one quarter is 0.5 s: the note [0, 0.5] s spans [0, Q].
        let group = NoteGroup {
            vibrato_env: linear_curve(&[(0, 0.0), (Q, 2.0)]),
            ..NoteGroup::default()
        };
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        let envelope = build_vibrato_envelope(&notes, 0.0, 0.005, 100);
        assert_eq!(envelope.len(), 100);
        assert!((envelope[0] - 0.0).abs() < 1e-6, "got {}", envelope[0]);
        assert!(
            (envelope[99] - 1.98).abs() < 0.02,
            "got {}, want the curve near its end",
            envelope[99]
        );
        assert!(
            envelope.windows(2).all(|pair| pair[0] <= pair[1]),
            "a rising curve renders a rising envelope"
        );
    }

    #[test]
    fn vibrato_depth_changes_sung_pitch_deterministically() {
        use opensvr_core::BLICKS_PER_QUARTER as Q;
        let mut score = SCORE.clone();
        score.attributes.d_f0_vbr = Some(2.0);
        let group = NoteGroup {
            vibrato_env: linear_curve(&[(0, 0.0), (2 * Q, 1.0), (4 * Q, 0.0)]),
            ..NoteGroup::default()
        };
        let plain = [note(1.0, 2.0, 60)];
        let deep = [timed_note(
            1.0, 2.0, 60, &score, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        let plain_f0 = build_log_f0(&plain, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        let deep_f0 = build_log_f0(&deep, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        assert_ne!(plain_f0, deep_f0, "vibrato depth must reach the sung pitch");
        assert_eq!(
            deep_f0,
            build_log_f0(&deep, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap(),
            "envelope sampling must be deterministic"
        );
    }

    #[test]
    fn pitch_delta_cents_shift_the_sung_pitch() {
        use opensvr_core::BLICKS_PER_QUARTER as Q;
        let group = NoteGroup {
            pitch_delta: linear_curve(&[(0, 100.0), (4 * Q, 100.0)]),
            ..NoteGroup::default()
        };
        let plain = [note(1.0, 2.0, 60)];
        let bent = [timed_note(
            1.0, 2.0, 60, &SCORE, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        let plain_f0 = build_log_f0(&plain, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        let bent_f0 = build_log_f0(&bent, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        let semitone = std::f64::consts::LN_2 / 12.0;
        for (plain, bent) in plain_f0.iter().zip(&bent_f0) {
            assert!(
                (f64::from(*bent) - f64::from(*plain) - semitone).abs() < 1e-6,
                "got {bent}, want {plain} + one semitone"
            );
        }
    }

    #[test]
    fn system_pitch_delta_cents_add_to_the_group_curve() {
        // The imported system contour rides under the editable `pitchDelta`
        // correction in the C++ sampling order, so both shift the sung pitch.
        use opensvr_core::BLICKS_PER_QUARTER as Q;
        let group = NoteGroup {
            pitch_delta: linear_curve(&[(0, 50.0), (4 * Q, 50.0)]),
            ..NoteGroup::default()
        };
        let reference = GroupReference {
            system_pitch_delta: linear_curve(&[(0, 50.0), (4 * Q, 50.0)]),
            ..GroupReference::default()
        };
        let plain = [note(1.0, 2.0, 60)];
        let bent = [timed_note(
            1.0, 2.0, 60, &SCORE, &group, &reference, &REFERENCE, &TEMPO,
        )];
        let plain_f0 = build_log_f0(&plain, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        let bent_f0 = build_log_f0(&bent, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        let semitone = std::f64::consts::LN_2 / 12.0;
        for (plain, bent) in plain_f0.iter().zip(&bent_f0) {
            assert!(
                (f64::from(*bent) - f64::from(*plain) - semitone).abs() < 1e-6,
                "got {bent}, want {plain} + one semitone"
            );
        }
    }

    #[test]
    fn system_pitch_delta_gates_the_neural_residual() {
        // A non-empty system contour means the editor already generated pitch
        // motion there, so the neural residual must not apply on top of it.
        use opensvr_core::BLICKS_PER_QUARTER as Q;
        let base = render_pitch_prediction_base(
            &[PitchCurveNote {
                start_seconds: 1.0,
                end_seconds: 2.0,
                ..PitchCurveNote::default()
            }],
            1.0,
            0.005,
            200,
        );
        let prediction: Vec<f32> = base.iter().map(|value| value + 1.0).collect();
        let plain = [note(1.0, 2.0, 60)];
        let plain_f0 = build_log_f0(&plain, 1.0, 0.005, 200, &prediction, 1.0, 0.005).unwrap();
        let silent = build_log_f0(&plain, 1.0, 0.005, 200, &[], 1.0, 0.005).unwrap();
        assert_ne!(
            plain_f0, silent,
            "the residual must move the sung pitch without a system contour"
        );
        // A flat zero contour still counts as imported: it gates the residual
        // while adding no cents of its own.
        let reference = GroupReference {
            system_pitch_delta: linear_curve(&[(0, 0.0), (4 * Q, 0.0)]),
            ..GroupReference::default()
        };
        let gated = [timed_note(
            1.0, 2.0, 60, &SCORE, &GROUP, &reference, &REFERENCE, &TEMPO,
        )];
        assert_eq!(
            build_log_f0(&gated, 1.0, 0.005, 200, &prediction, 1.0, 0.005).unwrap(),
            silent,
            "an imported system contour must switch the residual off"
        );
    }

    #[test]
    fn explicit_attributes_override_generated_widths() {
        // Adjacent notes portamento through a sigmoid scaled by the transition
        // widths, so an explicit width reshapes the boundary frames.
        let mut score = SCORE.clone();
        score.attributes.t_f0_left = Some(0.5);
        let wide = [
            timed_note(0.5, 1.0, 60, &SCORE, &GROUP, &REFERENCE, &REFERENCE, &TEMPO),
            timed_note(1.0, 1.5, 62, &score, &GROUP, &REFERENCE, &REFERENCE, &TEMPO),
        ];
        let plain = [note(0.5, 1.0, 60), note(1.0, 1.5, 62)];
        let wide_f0 = build_log_f0(&wide, 0.0, 0.005, 300, &[], 0.0, 0.005).unwrap();
        let plain_f0 = build_log_f0(&plain, 0.0, 0.005, 300, &[], 0.0, 0.005).unwrap();
        assert_ne!(
            wide_f0, plain_f0,
            "an explicit width must reshape the transition"
        );
    }

    #[test]
    fn modeless_phrases_produce_empty_weights() {
        let notes = [note(0.0, 0.5, 60)];
        assert!(!phrase_has_vocal_modes(&notes));
        assert_eq!(
            build_vocal_mode_weights(&notes, 0.0, 0.005, 100, &["tender".to_owned()]).unwrap(),
            Vec::<f32>::new(),
            "empty means the database default style"
        );
    }

    fn modal_group() -> NoteGroup {
        let mut modes = std::collections::BTreeMap::new();
        modes.insert(
            "tender".to_owned(),
            VocalModeSetting {
                amount: 0.5,
                ..VocalModeSetting::default()
            },
        );
        NoteGroup {
            vocal_modes: modes,
            ..NoteGroup::default()
        }
    }

    #[test]
    fn group_modes_produce_per_frame_weights() {
        let group = modal_group();
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        assert!(phrase_has_vocal_modes(&notes));
        let weights = build_vocal_mode_weights(
            &notes,
            0.0,
            0.005,
            100,
            &["tender".to_owned(), "power".to_owned()],
        )
        .unwrap();
        assert_eq!(weights.len(), 200);
        for frame in weights.chunks_exact(2) {
            assert_eq!(frame, [0.5, 0.0]);
        }
    }

    #[test]
    fn reference_presets_and_params_resolve_by_name() {
        let reference = GroupReference {
            vocal_mode_inherited: false,
            vocal_mode_preset: "power".to_owned(),
            vocal_mode_params: [("tender".to_owned(), 0.25)].into_iter().collect(),
            ..GroupReference::default()
        };
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &GROUP, &reference, &REFERENCE, &TEMPO,
        )];
        let weights = build_vocal_mode_weights(
            &notes,
            0.0,
            0.005,
            10,
            &["tender".to_owned(), "power".to_owned()],
        )
        .unwrap();
        assert_eq!(weights.len(), 20);
        for frame in weights.chunks_exact(2) {
            assert_eq!(frame, [0.25, 1.0]);
        }
    }

    #[test]
    fn inherited_references_fall_back_to_the_main_reference() {
        let main = GroupReference {
            vocal_mode_inherited: false,
            vocal_mode_preset: "power".to_owned(),
            ..GroupReference::default()
        };
        let reference = GroupReference {
            vocal_mode_inherited: true,
            ..GroupReference::default()
        };
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &GROUP, &reference, &main, &TEMPO,
        )];
        assert!(phrase_has_vocal_modes(&notes));
        let weights =
            build_vocal_mode_weights(&notes, 0.0, 0.005, 4, &["power".to_owned()]).unwrap();
        assert_eq!(weights, vec![1.0; 4]);
        // A disabled group curve still counts as a phrase mode but writes
        // zeros: the vector stays non-empty so the cache key differs from a
        // modeless phrase.
        let mut modes = std::collections::BTreeMap::new();
        modes.insert(
            "power".to_owned(),
            VocalModeSetting {
                enabled: false,
                ..VocalModeSetting::default()
            },
        );
        let group = NoteGroup {
            vocal_modes: modes,
            ..NoteGroup::default()
        };
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &group, &reference, &REFERENCE, &TEMPO,
        )];
        assert_eq!(
            build_vocal_mode_weights(&notes, 0.0, 0.005, 4, &["power".to_owned()]).unwrap(),
            vec![0.0; 4]
        );
    }

    #[test]
    fn unknown_vocal_modes_fail_like_the_cxx_build() {
        let group = modal_group();
        let modal = [timed_note(
            0.0, 0.5, 60, &SCORE, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        // The voice exposes no style vectors at all.
        assert_eq!(
            build_vocal_mode_weights(&modal, 0.0, 0.005, 8, &[])
                .unwrap_err()
                .to_string(),
            "The selected voice does not expose vocal-mode style vectors required by this project."
        );
        // The voice exposes other styles than the automated curve.
        assert_eq!(
            build_vocal_mode_weights(&modal, 0.0, 0.005, 8, &["power".to_owned()])
                .unwrap_err()
                .to_string(),
            "Vocal-mode curve 'tender' is not exposed by the selected voice."
        );
        let mut reference = GroupReference {
            vocal_mode_inherited: false,
            vocal_mode_preset: "opera".to_owned(),
            ..GroupReference::default()
        };
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &GROUP, &reference, &REFERENCE, &TEMPO,
        )];
        assert_eq!(
            build_vocal_mode_weights(&notes, 0.0, 0.005, 8, &["power".to_owned()])
                .unwrap_err()
                .to_string(),
            "Vocal-mode preset 'opera' is not a style exposed by the selected voice."
        );
        reference.vocal_mode_preset.clear();
        reference.vocal_mode_params.insert("opera".to_owned(), 0.5);
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &GROUP, &reference, &REFERENCE, &TEMPO,
        )];
        assert_eq!(
            build_vocal_mode_weights(&notes, 0.0, 0.005, 8, &["power".to_owned()])
                .unwrap_err()
                .to_string(),
            "Vocal mode 'opera' is not exposed by the selected voice."
        );
    }

    #[test]
    fn modal_weights_miss_the_modeless_cache_entry() {
        use crate::cache::{FileStamp, PhraseCache};
        use opensvr_g2p::TimingSyllable;
        let dir = std::env::temp_dir().join(format!(
            "opensvr-audio-pitch-{}-weights",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let voice = dir.join("voice.nofs");
        std::fs::write(&voice, b"voice").unwrap();
        let stamp = FileStamp::read(&voice).unwrap();
        let syllables = vec![TimingSyllable {
            language: "japanese".to_owned(),
            phonemes: vec!["a".to_owned()],
            duration_seconds: 0.4,
            midi_pitch: 60,
            is_continuation: false,
        }];
        let log_f0 = vec![4.0_f32; 8];
        let mut cache = PhraseCache::default();
        cache.retain(
            stamp.clone(),
            syllables.clone(),
            log_f0.clone(),
            Vec::new(),
            48_000,
            vec![0.1; 8],
        );
        let group = modal_group();
        let notes = [timed_note(
            0.0, 0.5, 60, &SCORE, &group, &REFERENCE, &REFERENCE, &TEMPO,
        )];
        let weights =
            build_vocal_mode_weights(&notes, 0.0, 0.005, 8, &["tender".to_owned()]).unwrap();
        assert_ne!(weights.len(), 0);
        assert!(
            cache
                .lookup(&stamp, &syllables, &log_f0, &weights)
                .is_none(),
            "a mode change must miss the modeless cache entry"
        );
    }
}
