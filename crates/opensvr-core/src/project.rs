use std::{collections::BTreeMap, fmt, path::PathBuf, str::FromStr};

use thiserror::Error;

use crate::{Blick, Extra, TempoMap, curve::Curve};

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
    /// Whether instant pitch mode (smooth neural transitions) is enabled.
    /// Mirrors OpenSV's `Note::instantMode` (defaults to true).
    pub instant_mode: bool,
    /// Musical style: `"singing"` or `"rap"`, mirroring OpenSV's
    /// `Note::musicalType` (any string is accepted on read).
    pub musical_type: String,
    /// Stress / emphasis markings, mirroring OpenSV's `Note::accent`.
    pub accent: String,
    /// User-specified manual pitch/vibrato overrides.
    pub attributes: PitchAttributes,
    /// Automatically calculated pitch/vibrato defaults.
    pub system_attributes: PitchAttributes,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

/// One of the 14 table-driven pitch/vibrato attribute keys, mirroring the
/// `pitchAttributeFields` table in OpenSV's `src/core/ProjectFile.cpp`.
/// The key list lives in [`PitchAttribute::ALL`]; the JSON names and the
/// [`PitchAttributes`] fields are mapped through the small macros below, so
/// readers loop over the table instead of hand-writing 14 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PitchAttribute {
    TF0Offset,
    TF0Left,
    TF0Right,
    DF0Left,
    DF0Right,
    TF0VbrStart,
    TF0VbrLeft,
    TF0VbrRight,
    DF0Vbr,
    PF0Vbr,
    FF0Vbr,
    DF0VbrMod,
    RTone,
    RIntonation,
}

/// Manual pitch/vibrato overrides, mirroring OpenSV's `PitchAttributes` in
/// `src/core/Project.h` (14 `std::optional<double>` plus the
/// preserved-fields JSON, carried here as [`Extra`]). Absent stays
/// `None`: `normaliseProject` never fills defaults. Inert until the pitch
/// builder lands.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PitchAttributes {
    pub t_f0_offset: Option<f64>,
    pub t_f0_left: Option<f64>,
    pub t_f0_right: Option<f64>,
    pub d_f0_left: Option<f64>,
    pub d_f0_right: Option<f64>,
    pub t_f0_vbr_start: Option<f64>,
    pub t_f0_vbr_left: Option<f64>,
    pub t_f0_vbr_right: Option<f64>,
    pub d_f0_vbr: Option<f64>,
    pub p_f0_vbr: Option<f64>,
    pub f_f0_vbr: Option<f64>,
    pub d_f0_vbr_mod: Option<f64>,
    pub r_tone: Option<f64>,
    pub r_intonation: Option<f64>,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

macro_rules! pitch_attribute_name {
    (TF0Offset) => {
        "tF0Offset"
    };
    (TF0Left) => {
        "tF0Left"
    };
    (TF0Right) => {
        "tF0Right"
    };
    (DF0Left) => {
        "dF0Left"
    };
    (DF0Right) => {
        "dF0Right"
    };
    (TF0VbrStart) => {
        "tF0VbrStart"
    };
    (TF0VbrLeft) => {
        "tF0VbrLeft"
    };
    (TF0VbrRight) => {
        "tF0VbrRight"
    };
    (DF0Vbr) => {
        "dF0Vbr"
    };
    (PF0Vbr) => {
        "pF0Vbr"
    };
    (FF0Vbr) => {
        "fF0Vbr"
    };
    (DF0VbrMod) => {
        "dF0VbrMod"
    };
    (RTone) => {
        "rTone"
    };
    (RIntonation) => {
        "rIntonation"
    };
}

macro_rules! pitch_attribute_field {
    ($self:expr, TF0Offset) => {
        $self.t_f0_offset
    };
    ($self:expr, TF0Left) => {
        $self.t_f0_left
    };
    ($self:expr, TF0Right) => {
        $self.t_f0_right
    };
    ($self:expr, DF0Left) => {
        $self.d_f0_left
    };
    ($self:expr, DF0Right) => {
        $self.d_f0_right
    };
    ($self:expr, TF0VbrStart) => {
        $self.t_f0_vbr_start
    };
    ($self:expr, TF0VbrLeft) => {
        $self.t_f0_vbr_left
    };
    ($self:expr, TF0VbrRight) => {
        $self.t_f0_vbr_right
    };
    ($self:expr, DF0Vbr) => {
        $self.d_f0_vbr
    };
    ($self:expr, PF0Vbr) => {
        $self.p_f0_vbr
    };
    ($self:expr, FF0Vbr) => {
        $self.f_f0_vbr
    };
    ($self:expr, DF0VbrMod) => {
        $self.d_f0_vbr_mod
    };
    ($self:expr, RTone) => {
        $self.r_tone
    };
    ($self:expr, RIntonation) => {
        $self.r_intonation
    };
}

impl PitchAttribute {
    /// Every key, in the C++ table order.
    pub const ALL: [PitchAttribute; 14] = [
        PitchAttribute::TF0Offset,
        PitchAttribute::TF0Left,
        PitchAttribute::TF0Right,
        PitchAttribute::DF0Left,
        PitchAttribute::DF0Right,
        PitchAttribute::TF0VbrStart,
        PitchAttribute::TF0VbrLeft,
        PitchAttribute::TF0VbrRight,
        PitchAttribute::DF0Vbr,
        PitchAttribute::PF0Vbr,
        PitchAttribute::FF0Vbr,
        PitchAttribute::DF0VbrMod,
        PitchAttribute::RTone,
        PitchAttribute::RIntonation,
    ];

    /// The SVP JSON key, e.g. `"tF0Offset"`.
    pub const fn name(self) -> &'static str {
        match self {
            PitchAttribute::TF0Offset => pitch_attribute_name!(TF0Offset),
            PitchAttribute::TF0Left => pitch_attribute_name!(TF0Left),
            PitchAttribute::TF0Right => pitch_attribute_name!(TF0Right),
            PitchAttribute::DF0Left => pitch_attribute_name!(DF0Left),
            PitchAttribute::DF0Right => pitch_attribute_name!(DF0Right),
            PitchAttribute::TF0VbrStart => pitch_attribute_name!(TF0VbrStart),
            PitchAttribute::TF0VbrLeft => pitch_attribute_name!(TF0VbrLeft),
            PitchAttribute::TF0VbrRight => pitch_attribute_name!(TF0VbrRight),
            PitchAttribute::DF0Vbr => pitch_attribute_name!(DF0Vbr),
            PitchAttribute::PF0Vbr => pitch_attribute_name!(PF0Vbr),
            PitchAttribute::FF0Vbr => pitch_attribute_name!(FF0Vbr),
            PitchAttribute::DF0VbrMod => pitch_attribute_name!(DF0VbrMod),
            PitchAttribute::RTone => pitch_attribute_name!(RTone),
            PitchAttribute::RIntonation => pitch_attribute_name!(RIntonation),
        }
    }
}

impl PitchAttributes {
    /// Reads one field by key.
    pub fn get(&self, field: PitchAttribute) -> Option<f64> {
        match field {
            PitchAttribute::TF0Offset => pitch_attribute_field!(self, TF0Offset),
            PitchAttribute::TF0Left => pitch_attribute_field!(self, TF0Left),
            PitchAttribute::TF0Right => pitch_attribute_field!(self, TF0Right),
            PitchAttribute::DF0Left => pitch_attribute_field!(self, DF0Left),
            PitchAttribute::DF0Right => pitch_attribute_field!(self, DF0Right),
            PitchAttribute::TF0VbrStart => {
                pitch_attribute_field!(self, TF0VbrStart)
            }
            PitchAttribute::TF0VbrLeft => pitch_attribute_field!(self, TF0VbrLeft),
            PitchAttribute::TF0VbrRight => {
                pitch_attribute_field!(self, TF0VbrRight)
            }
            PitchAttribute::DF0Vbr => pitch_attribute_field!(self, DF0Vbr),
            PitchAttribute::PF0Vbr => pitch_attribute_field!(self, PF0Vbr),
            PitchAttribute::FF0Vbr => pitch_attribute_field!(self, FF0Vbr),
            PitchAttribute::DF0VbrMod => pitch_attribute_field!(self, DF0VbrMod),
            PitchAttribute::RTone => pitch_attribute_field!(self, RTone),
            PitchAttribute::RIntonation => pitch_attribute_field!(self, RIntonation),
        }
    }

    /// Stores one field by key.
    pub fn set(&mut self, field: PitchAttribute, value: f64) {
        let slot = match field {
            PitchAttribute::TF0Offset => &mut self.t_f0_offset,
            PitchAttribute::TF0Left => &mut self.t_f0_left,
            PitchAttribute::TF0Right => &mut self.t_f0_right,
            PitchAttribute::DF0Left => &mut self.d_f0_left,
            PitchAttribute::DF0Right => &mut self.d_f0_right,
            PitchAttribute::TF0VbrStart => &mut self.t_f0_vbr_start,
            PitchAttribute::TF0VbrLeft => &mut self.t_f0_vbr_left,
            PitchAttribute::TF0VbrRight => &mut self.t_f0_vbr_right,
            PitchAttribute::DF0Vbr => &mut self.d_f0_vbr,
            PitchAttribute::PF0Vbr => &mut self.p_f0_vbr,
            PitchAttribute::FF0Vbr => &mut self.f_f0_vbr,
            PitchAttribute::DF0VbrMod => &mut self.d_f0_vbr_mod,
            PitchAttribute::RTone => &mut self.r_tone,
            PitchAttribute::RIntonation => &mut self.r_intonation,
        };
        *slot = Some(value);
    }
}

/// A vocal-mode (singing-style) setting on a note group, mirroring OpenSV's
/// `VocalModeSetting` in `src/core/Project.h`: `enabled` defaults to true,
/// `amount` is the static weight, and `has_curve` marks a dynamic automation
/// curve read through the same `readCurve` validation as the group curves.
/// Per-mode curves are stored as read: unlike `pitch_delta`/`vibrato_env`,
/// OpenSV's `normaliseProject` does not normalise them.
#[derive(Debug, Clone, PartialEq)]
pub struct VocalModeSetting {
    pub enabled: bool,
    pub amount: f64,
    pub has_curve: bool,
    pub curve: Curve,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

impl Default for VocalModeSetting {
    fn default() -> Self {
        Self {
            enabled: true,
            amount: 0.0,
            has_curve: false,
            curve: Curve::default(),
            extra: Extra::default(),
        }
    }
}

/// An ordered collection of notes, either a track's own group or a shared library group.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NoteGroup {
    pub id: String,
    pub name: String,
    /// Sorted by onset.
    pub notes: Vec<Note>,
    /// Continuous pitch-bend curve in semitones (offset from note pitch).
    pub pitch_delta: Curve,
    /// Dynamic vibrato depth envelope curve.
    pub vibrato_env: Curve,
    /// Vocal-style weights keyed by mode name (`std::map`, hence ordered).
    pub vocal_modes: BTreeMap<String, VocalModeSetting>,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

/// Placement of a [`NoteGroup`] on a track timeline.
#[derive(Debug, Clone, PartialEq)]
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
    /// Backing-track file for an instrumental reference, resolved against the
    /// project directory (`audioFile` in `src/core/Project.h`; `None` when
    /// the `audio` block is absent or its filename is empty).
    pub audio_file: Option<PathBuf>,
    /// Backing-track length in seconds (`audioDurationSeconds`, `0.0` by default).
    pub audio_duration_seconds: f64,
    /// If true, inherit vocal-mode settings from the parent track.
    pub vocal_mode_inherited: bool,
    /// Name of the applied vocal-mode preset.
    pub vocal_mode_preset: String,
    /// Per-mode weight overrides (`std::map`, hence ordered).
    pub vocal_mode_params: BTreeMap<String, f64>,
    /// Track/reference-level default pitch attributes (`voicePitch`).
    pub voice_pitch: PitchAttributes,
    /// Engine-generated pitch deviation curve in cents, sampled at the
    /// group-local position like `pitch_delta` (`systemPitchDelta` in
    /// `src/core/ProjectFile.cpp`, consumed in `ProjectRenderer::buildPitch`).
    /// Empty when the editor never generated one.
    pub system_pitch_delta: Curve,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
}

impl Default for GroupReference {
    fn default() -> Self {
        Self {
            group_id: String::new(),
            time_offset: 0,
            pitch_offset: 0,
            begin: 0,
            end: None,
            instrumental: false,
            audio_file: None,
            audio_duration_seconds: 0.0,
            vocal_mode_inherited: true,
            vocal_mode_preset: String::new(),
            vocal_mode_params: BTreeMap::new(),
            voice_pitch: PitchAttributes::default(),
            system_pitch_delta: Curve::default(),
            extra: Extra::default(),
        }
    }
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
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
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
            extra: Extra::default(),
        }
    }
}

/// A note resolved onto the track timeline: offset, transposed and cropped.
///
/// `window_start`/`window_end` are the placing reference's crop boundaries:
/// the renderer keeps the full model-input context but crops the mixed
/// output to this window, exactly as OpenSV's `cropStartSeconds`/`stopSeconds`
/// in `src/audio/ProjectRenderer.cpp`. `group`/`reference` borrow the placing
/// group and reference so pitch builders can sample their curves, attributes
/// and vocal-mode settings without another lookup.
#[derive(Debug, Clone, Copy)]
pub struct PlacedNote<'a> {
    pub start: Blick,
    pub end: Blick,
    /// Pitch after the reference's transposition, `0..=127`.
    pub pitch: u8,
    pub note: &'a Note,
    pub group: &'a NoteGroup,
    pub reference: &'a GroupReference,
    pub window_start: Blick,
    pub window_end: Option<Blick>,
}

/// A whole Synthesizer V project.
#[derive(Debug, Clone, Default)]
pub struct Project {
    pub name: String,
    pub tracks: Vec<Track>,
    /// Shared note groups that tracks may reference.
    pub library: Vec<NoteGroup>,
    pub tempo_map: TempoMap,
    /// Unknown SVP JSON fields kept for a lossless round trip.
    pub extra: Extra,
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
    reference: &'a GroupReference,
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
            group,
            reference,
            window_start: reference.begin,
            window_end: reference.end,
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
            instant_mode: true,
            musical_type: "singing".into(),
            accent: String::new(),
            attributes: PitchAttributes::default(),
            system_attributes: PitchAttributes::default(),
            extra: Extra::default(),
        }
    }

    fn group(id: &str, notes: Vec<Note>) -> NoteGroup {
        NoteGroup {
            id: id.into(),
            name: id.into(),
            notes,
            ..NoteGroup::default()
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
            ..GroupReference::default()
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
    fn references_carry_their_crop_window() {
        let mut track = Track::new("t", group("main", Vec::new()));
        track.groups.push(GroupReference {
            group_id: "lib".into(),
            time_offset: 4 * Q,
            pitch_offset: 0,
            begin: 4 * Q + Q / 2,
            end: Some(6 * Q),
            instrumental: false,
            ..GroupReference::default()
        });
        let library = vec![group("lib", vec![note(0, Q, 60)])];
        let project = Project {
            tracks: vec![track.clone()],
            library,
            ..Project::default()
        };

        let placed = project.placed_notes(&track).unwrap();
        assert_eq!(placed.len(), 1);
        assert_eq!((placed[0].start, placed[0].end), (4 * Q + Q / 2, 5 * Q));
        assert_eq!(placed[0].window_start, 4 * Q + Q / 2);
        assert_eq!(placed[0].window_end, Some(6 * Q));

        // The main reference is unbounded and starts at zero.
        let main = Track::new("m", group("m", vec![note(0, Q, 60)]));
        let project = Project {
            tracks: vec![main.clone()],
            ..Project::default()
        };
        let placed = project.placed_notes(&main).unwrap();
        assert_eq!(placed[0].window_start, 0);
        assert_eq!(placed[0].window_end, None);
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

    #[test]
    fn pitch_attributes_default_to_none_and_round_trip_by_key() {
        let empty = PitchAttributes::default();
        for field in PitchAttribute::ALL {
            assert_eq!(empty.get(field), None, "{}", field.name());
        }
        // JSON names are unique and in the C++ table order.
        let names: Vec<_> = PitchAttribute::ALL
            .iter()
            .map(|field| field.name())
            .collect();
        assert_eq!(
            names,
            [
                "tF0Offset",
                "tF0Left",
                "tF0Right",
                "dF0Left",
                "dF0Right",
                "tF0VbrStart",
                "tF0VbrLeft",
                "tF0VbrRight",
                "dF0Vbr",
                "pF0Vbr",
                "fF0Vbr",
                "dF0VbrMod",
                "rTone",
                "rIntonation",
            ]
        );
        let mut filled = PitchAttributes::default();
        for (index, field) in PitchAttribute::ALL.iter().enumerate() {
            filled.set(*field, index as f64 + 0.5);
        }
        for (index, field) in PitchAttribute::ALL.iter().enumerate() {
            assert_eq!(filled.get(*field), Some(index as f64 + 0.5));
        }
    }

    #[test]
    fn fresh_domain_values_carry_no_preserved_fields() {
        // Every struct with a C++ `preservedFieldsJson` counterpart starts
        // with an empty `extra`; nothing populates it yet.
        assert!(PitchAttributes::default().extra.is_empty());
        assert!(note(0, Q, 60).extra.is_empty());
        assert!(VocalModeSetting::default().extra.is_empty());
        assert!(group("g", Vec::new()).extra.is_empty());
        assert!(GroupReference::default().extra.is_empty());
        assert!(Track::new("t", group("g", Vec::new())).extra.is_empty());
        assert!(Project::default().extra.is_empty());
    }
}
