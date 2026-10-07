use std::{fmt, path::PathBuf, str::FromStr};

use thiserror::Error;

use crate::{Blick, TempoMap};

/// Errors raised while resolving a project's notes onto the timeline.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProjectError {
    #[error("a referenced note group ({0}) is missing")]
    MissingGroup(String),
    #[error("a note's time range overflows")]
    TimeOverflow,
    #[error("a transposed note falls outside the MIDI pitch range (pitch {0})")]
    PitchOutOfRange(i32),
}

/// A single sung note, positioned relative to its group's origin.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    pub onset: Blick,
    /// Length in blicks; always positive.
    pub duration: Blick,
    /// MIDI note number, `0..=127`.
    pub pitch: u8,
    pub lyrics: String,
    /// Optional space-separated phoneme override.
    pub phonemes: String,
    /// Microtonal offset in cents.
    pub detune: f64,
}

/// An ordered collection of notes, either a track's own group or a shared library group.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoteGroup {
    pub id: String,
    pub name: String,
    /// Sorted by onset.
    pub notes: Vec<Note>,
}

/// Placement of a [`NoteGroup`] on a track timeline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupReference {
    pub group_id: String,
    /// Where the group's origin sits on the track.
    pub time_offset: Blick,
    /// Semitone transposition.
    pub pitch_offset: i32,
    /// Left crop boundary on the track, `>= 0`.
    pub begin: Blick,
    /// Right crop boundary on the track; `None` means unbounded.
    pub end: Option<Blick>,
    /// An instrumental (audio backing) reference carries no singing notes.
    pub instrumental: bool,
}

/// Singing language used for grapheme-to-phoneme conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Language {
    #[default]
    Japanese,
    Mandarin,
    English,
    Cantonese,
    Spanish,
}

impl Language {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Japanese => "japanese",
            Self::Mandarin => "mandarin",
            Self::English => "english",
            Self::Cantonese => "cantonese",
            Self::Spanish => "spanish",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when a string is not one of the supported languages.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("unsupported language {0:?} (expected japanese, mandarin, english, cantonese or spanish)")]
pub struct ParseLanguageError(String);

impl FromStr for Language {
    type Err = ParseLanguageError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "japanese" => Ok(Self::Japanese),
            "mandarin" => Ok(Self::Mandarin),
            "english" => Ok(Self::English),
            "cantonese" => Ok(Self::Cantonese),
            "spanish" => Ok(Self::Spanish),
            other => Err(ParseLanguageError(other.to_owned())),
        }
    }
}

/// Which voice database and dictionary a track sings with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoiceSettings {
    /// Path to the `voice.nofs` database, if one is set.
    pub database: Option<PathBuf>,
    pub language: Language,
    /// Directory of pronunciation dictionaries (`clf-data`), if one is set.
    pub dictionary: Option<PathBuf>,
}

/// Per-track mixer state.
#[derive(Debug, Clone, PartialEq)]
pub struct Mixer {
    /// Linear gain; `1.0` is 0 dB.
    pub gain: f64,
    /// `-1.0` (left) to `1.0` (right).
    pub pan: f64,
    pub mute: bool,
    pub solo: bool,
}

impl Default for Mixer {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pan: 0.0,
            mute: false,
            solo: false,
        }
    }
}

/// A vocal track: its own note group plus any referenced library groups.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub name: String,
    pub main_group: NoteGroup,
    /// Always refers to `main_group`.
    pub main_ref: GroupReference,
    pub groups: Vec<GroupReference>,
    pub mixer: Mixer,
    pub voice: VoiceSettings,
}

impl Track {
    /// Creates a track whose main reference points at `main_group`.
    pub fn new(name: impl Into<String>, main_group: NoteGroup) -> Self {
        let main_ref = GroupReference {
            group_id: main_group.id.clone(),
            ..GroupReference::default()
        };
        Self {
            name: name.into(),
            main_group,
            main_ref,
            groups: Vec::new(),
            mixer: Mixer::default(),
            voice: VoiceSettings::default(),
        }
    }
}

/// A note resolved onto the track timeline: offset, transposed and cropped.
#[derive(Debug, Clone, Copy)]
pub struct PlacedNote<'a> {
    pub start: Blick,
    pub end: Blick,
    /// Pitch after the reference's transposition, `0..=127`.
    pub pitch: u8,
    pub note: &'a Note,
}

/// A whole Synthesizer V project.
#[derive(Debug, Clone, Default)]
pub struct Project {
    pub name: String,
    pub tracks: Vec<Track>,
    /// Shared note groups that tracks may reference.
    pub library: Vec<NoteGroup>,
    pub tempo_map: TempoMap,
}

impl Project {
    /// Finds a group by UUID among the library and every track's main group.
    pub fn find_group(&self, id: &str) -> Option<&NoteGroup> {
        self.library
            .iter()
            .chain(self.tracks.iter().map(|track| &track.main_group))
            .find(|group| group.id == id)
    }

    /// Resolves every singing note of `track` onto the timeline, sorted by start.
    ///
    /// Applies each reference's time offset, transposition and crop window, and
    /// drops notes that fall entirely outside the window.
    pub fn placed_notes<'a>(
        &'a self,
        track: &'a Track,
    ) -> Result<Vec<PlacedNote<'a>>, ProjectError> {
        let mut placed = Vec::new();
        place(&track.main_group, &track.main_ref, &mut placed)?;
        for reference in track
            .groups
            .iter()
            .filter(|reference| !reference.instrumental)
        {
            let group = self
                .find_group(&reference.group_id)
                .ok_or_else(|| ProjectError::MissingGroup(reference.group_id.clone()))?;
            place(group, reference, &mut placed)?;
        }
        placed.sort_by_key(|note| note.start);
        Ok(placed)
    }
}

fn place<'a>(
    group: &'a NoteGroup,
    reference: &GroupReference,
    placed: &mut Vec<PlacedNote<'a>>,
) -> Result<(), ProjectError> {
    if reference.instrumental {
        return Ok(());
    }
    for note in &group.notes {
        let onset = note
            .onset
            .checked_add(reference.time_offset)
            .ok_or(ProjectError::TimeOverflow)?;
        let natural_end = onset
            .checked_add(note.duration)
            .ok_or(ProjectError::TimeOverflow)?;
        let start = onset.max(reference.begin).max(0);
        let end = reference
            .end
            .map_or(natural_end, |limit| natural_end.min(limit));
        if end <= start {
            continue;
        }
        let pitch = i32::from(note.pitch) + reference.pitch_offset;
        let pitch = u8::try_from(pitch)
            .ok()
            .filter(|&pitch| pitch <= 127)
            .ok_or(ProjectError::PitchOutOfRange(pitch))?;
        placed.push(PlacedNote {
            start,
            end,
            pitch,
            note,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BLICKS_PER_QUARTER as Q;

    fn note(onset: Blick, duration: Blick, pitch: u8) -> Note {
        Note {
            onset,
            duration,
            pitch,
            lyrics: "la".into(),
            phonemes: String::new(),
            detune: 0.0,
        }
    }

    fn group(id: &str, notes: Vec<Note>) -> NoteGroup {
        NoteGroup {
            id: id.into(),
            name: id.into(),
            notes,
        }
    }

    #[test]
    fn main_group_notes_keep_their_timing() {
        let track = Track::new("t", group("main", vec![note(0, Q, 60), note(Q, Q, 62)]));
        let project = Project {
            tracks: vec![track.clone()],
            ..Project::default()
        };
        let placed = project.placed_notes(&track).unwrap();
        assert_eq!(placed.len(), 2);
        assert_eq!(
            (placed[1].start, placed[1].end, placed[1].pitch),
            (Q, 2 * Q, 62)
        );
    }

    #[test]
    fn references_offset_transpose_and_crop() {
        let mut track = Track::new("t", group("main", Vec::new()));
        track.groups.push(GroupReference {
            group_id: "lib".into(),
            time_offset: 4 * Q,
            pitch_offset: 2,
            begin: 4 * Q + Q / 2,
            end: Some(6 * Q),
            instrumental: false,
        });
        let library = vec![group(
            "lib",
            vec![note(0, Q, 60), note(Q, 4 * Q, 64), note(10 * Q, Q, 70)],
        )];
        let project = Project {
            tracks: vec![track.clone()],
            library,
            ..Project::default()
        };

        let placed = project.placed_notes(&track).unwrap();
        // The first note is cropped on the left, the second on the right, the third is dropped.
        let spans: Vec<_> = placed.iter().map(|n| (n.start, n.end, n.pitch)).collect();
        assert_eq!(spans, [(4 * Q + Q / 2, 5 * Q, 62), (5 * Q, 6 * Q, 66)]);
    }

    #[test]
    fn missing_groups_and_bad_pitches_are_errors() {
        let mut track = Track::new("t", group("main", vec![note(0, Q, 126)]));
        track.main_ref.pitch_offset = 2;
        let project = Project {
            tracks: vec![track.clone()],
            ..Project::default()
        };
        assert_eq!(
            project.placed_notes(&track).unwrap_err(),
            ProjectError::PitchOutOfRange(128)
        );

        let mut track = Track::new("t", group("main", Vec::new()));
        track.groups.push(GroupReference {
            group_id: "nope".into(),
            ..GroupReference::default()
        });
        let project = Project {
            tracks: vec![track.clone()],
            ..Project::default()
        };
        assert_eq!(
            project.placed_notes(&track).unwrap_err(),
            ProjectError::MissingGroup("nope".into())
        );
    }

    #[test]
    fn language_round_trips_through_text() {
        for language in [
            Language::Japanese,
            Language::Mandarin,
            Language::English,
            Language::Cantonese,
            Language::Spanish,
        ] {
            assert_eq!(language.to_string().parse(), Ok(language));
        }
        assert!("klingon".parse::<Language>().is_err());
    }
}
