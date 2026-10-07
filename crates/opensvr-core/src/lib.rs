//! Project model for opensvr: the data structures shared by every other crate.
//!
//! This is the Rust counterpart of OpenSV's `src/core/Project.{h,cpp}`. It holds
//! only what the command-line tools need: tracks, note groups, group references,
//! mixer state, voice settings and the tempo map.

mod project;
mod tempo;

pub use project::{
    GroupReference, Language, Mixer, Note, NoteGroup, ParseLanguageError, PlacedNote, Project,
    ProjectError, Track, VoiceSettings,
};
pub use tempo::{TempoChange, TempoMap, TimeSignature};

/// Musical time unit of Synthesizer V.
///
/// One quarter note is [`BLICKS_PER_QUARTER`] blicks. The constant is
/// `480 * 1_470_000`, so it divides evenly into MIDI ticks, common audio sample
/// rates and arbitrary tuplets.
pub type Blick = i64;

/// Number of [`Blick`]s in one quarter note.
pub const BLICKS_PER_QUARTER: Blick = 705_600_000;

/// Largest blick magnitude accepted from a file, leaving headroom for arithmetic.
pub const MAX_FILE_BLICK: Blick = Blick::MAX / 4;
