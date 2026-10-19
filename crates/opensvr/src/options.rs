//! Render settings shared by every entry point.

use std::path::{Path, PathBuf};

use opensvr_core::{Language, Project};

/// The sample rate [`Options::default`] renders at, matching `opensvr render`.
pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;

/// What a render should use, and where to find it.
///
/// The three path overrides have the same meaning as the command line's
/// `--voice`, `--dictionary` and `--models-dir`: `None` keeps whatever each
/// track already carries, `Some` replaces it on every track. Resolving
/// environment variables is the caller's job, so a library embed with no
/// flags behaves the same no matter what the process environment says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Output sample rate in Hz, between [`MIN_SAMPLE_RATE`][crate::MIN_SAMPLE_RATE]
    /// and [`MAX_SAMPLE_RATE`][crate::MAX_SAMPLE_RATE].
    pub sample_rate: u32,
    /// Directory that a project string's relative paths resolve against.
    /// `None` leaves them relative to the process directory. Ignored when
    /// loading from a file, which has its own directory.
    pub base_dir: Option<PathBuf>,
    /// Voice database for every track, replacing the project's.
    pub voice: Option<PathBuf>,
    /// Pronunciation dictionary directory for every track.
    pub dictionary: Option<PathBuf>,
    /// Directory holding Flat voice models, away from the voice document's
    /// sibling `models` directory. Container voices ignore it.
    pub models_dir: Option<PathBuf>,
    /// Singing language for every track.
    pub language: Option<Language>,
    /// Which singer to use.
    pub backend: Backend,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            base_dir: None,
            voice: None,
            dictionary: None,
            models_dir: None,
            language: None,
            backend: Backend::default(),
        }
    }
}

/// Which singer a render uses.
///
/// [`Auto`][Self::Auto] is the command line's rule and the recommended
/// default: sing through the neural voice when a track has a database that
/// exists on disk, and fall back to placeholder tones otherwise. Missing
/// files count as voiceless, so a project whose voice was never installed
/// still renders. Pass [`Neural`][Self::Neural] to turn that case into an
/// error instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    #[default]
    Auto,
    Neural,
    Tones,
}

impl Backend {
    /// Resolves [`Auto`][Self::Auto] against `project`, returning the backend
    /// a render would actually use: [`Neural`][Self::Neural] or
    /// [`Tones`][Self::Tones]. Other choices resolve to themselves.
    #[must_use]
    pub fn resolve(self, project: &Project) -> Self {
        match self {
            Self::Auto => {
                let voiced = project
                    .tracks
                    .iter()
                    .any(|track| track.voice.database.as_deref().is_some_and(Path::is_file));
                if voiced { Self::Neural } else { Self::Tones }
            }
            explicit => explicit,
        }
    }
}

#[cfg(test)]
mod tests {
    use opensvr_core::{NoteGroup, Track};

    use super::*;

    #[test]
    fn defaults_match_the_command_line() {
        let options = Options::default();
        assert_eq!(options.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(options.sample_rate, 48_000);
        assert_eq!(options.backend, Backend::Auto);
        assert_eq!(
            (
                options.base_dir,
                options.voice,
                options.dictionary,
                options.models_dir,
                options.language
            ),
            (None, None, None, None, None)
        );
    }

    #[test]
    fn explicit_backends_ignore_the_project() {
        let project = Project {
            tracks: vec![Track::new("Lead", NoteGroup::default())],
            ..Project::default()
        };
        assert_eq!(Backend::Neural.resolve(&project), Backend::Neural);
        assert_eq!(Backend::Tones.resolve(&project), Backend::Tones);
    }
}
