//! Phrase splitting: cutting a track's notes at rests.
//!
//! Ports the phrase loop of OpenSV's `src/audio/ProjectRenderer.cpp`: consecutive
//! notes whose gap is shorter than [`PHRASE_BREAK_SECONDS`] belong to one phrase
//! and are synthesized together (the duration model places onset consonants in
//! the preceding rest, so each phrase keeps [`REST_CONTEXT_SECONDS`] of leading
//! context even before zero). Splitting is pure timeline geometry over already
//! resolved [`TimedNote`]s; synthesis itself lives in the [`VoiceBackend`].

use crate::{REST_CONTEXT_SECONDS, TimedNote};

/// A gap of at least this long (twice the release context) starts a new phrase.
///
/// Mirrors `phraseBreakSeconds` (`2 * restContextSeconds`) in `ProjectRenderer.cpp`.
pub const PHRASE_BREAK_SECONDS: f64 = 2.0 * REST_CONTEXT_SECONDS;

/// Consecutive notes sung as one synthesis unit.
///
/// The backend renders leading release context before [`notes`][Self::notes]`[0].start`
/// (see [`PhraseAudio`][crate::PhraseAudio]), so phrases are always non-empty slices:
/// an empty track yields no phrases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phrase<'a> {
    pub notes: &'a [TimedNote<'a>],
}

/// Cuts time-ordered, non-overlapping `notes` into phrases at rests.
///
/// A rest of [`PHRASE_BREAK_SECONDS`] or more starts a new phrase; shorter gaps
/// (including exact adjacency) stay inside one phrase.
pub fn split_phrases<'a>(notes: &'a [TimedNote<'a>]) -> Vec<Phrase<'a>> {
    let mut phrases = Vec::new();
    let mut begin = 0_usize;
    for end in 1..=notes.len() {
        let boundary = end == notes.len()
            || notes[end].start - notes[end - 1].end >= PHRASE_BREAK_SECONDS;
        if boundary {
            phrases.push(Phrase {
                notes: &notes[begin..end],
            });
            begin = end;
        }
    }
    phrases
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensvr_core::{Extra, GroupReference, Note, NoteGroup, PitchAttributes, TempoMap};

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

    fn note(start: f64, end: f64) -> TimedNote<'static> {
        TimedNote {
            start,
            end,
            pitch: 60,
            detune_cents: 0.0,
            seconds_per_quarter: 0.5,
            lyrics: "la",
            phonemes: "",
            note: &SCORE,
            group: &GROUP,
            reference: &REFERENCE,
            main_reference: &REFERENCE,
            tempo_map: &TEMPO,
            window_start: 0.0,
            window_end: None,
        }
    }

    #[test]
    fn empty_tracks_yield_no_phrases() {
        assert!(split_phrases(&[]).is_empty());
    }

    #[test]
    fn adjacent_and_short_gaps_stay_in_one_phrase() {
        // Gaps of 0 s and 0.19 s: one phrase of three notes.
        let notes = [note(0.0, 0.5), note(0.5, 1.0), note(1.19, 1.5)];
        let phrases = split_phrases(&notes);
        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].notes.len(), 3);
    }

    #[test]
    fn gaps_near_the_threshold_follow_the_cxx_comparison() {
        // Gap just below 0.2 s stays; a clearly longer gap splits (< vs >=,
        // evaluated in f64 exactly as in C++).
        let notes = [note(0.0, 0.5), note(0.69, 1.0), note(1.31, 1.5)];
        let phrases = split_phrases(&notes);
        assert_eq!(phrases.len(), 2);
        assert_eq!(phrases[0].notes.len(), 2);
        assert_eq!(phrases[1].notes.len(), 1);
        assert_eq!(phrases[1].notes[0].start, 1.31);
    }

    #[test]
    fn every_rest_starts_a_new_phrase() {
        let notes = [note(0.0, 0.5), note(1.0, 1.5), note(2.0, 2.5)];
        let phrases = split_phrases(&notes);
        assert_eq!(phrases.len(), 3);
        for (phrase, expected) in phrases.iter().zip([0.0, 1.0, 2.0]) {
            assert_eq!(phrase.notes.len(), 1);
            assert_eq!(phrase.notes[0].start, expected);
        }
    }
}
