//! One-call rendering for OpenSVR: project in, audio out.
//!
//! The engine is split across crates so each stage stays small, but a caller
//! that just wants audio should not have to assemble them. This crate is that
//! assembly: [`render_svp`] takes a project as a string, [`render_project`]
//! takes an already-parsed [`Project`], and [`Options`] carries the voice,
//! dictionary, language and backend settings the command line exposes as
//! flags. The per-crate APIs remain public for callers that need a stage.
//!
//! Voice databases, pronunciation dictionaries and Flat voice models are not
//! bundled. With no database the render falls back to placeholder tones,
//! exactly like `opensvr render`; pass [`Backend::Neural`] to make a missing
//! voice an error instead.
//!
//! For diagnostics, [`PhonemeResolver`] answers "what will this note sing?"
//! without rendering, which is what `opensvr info --phonemes` reports.
//!
//! ```
//! use opensvr::{CancelToken, Options, encode_wav, render_svp};
//!
//! // A project read from disk, stdin, or a database column: it is just JSON.
//! let svp = std::fs::read_to_string(concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/../../examples/two-tracks.svp",
//! ))
//! .unwrap();
//!
//! let options = Options {
//!     sample_rate: 8_000,
//!     ..Options::default()
//! };
//! let cancel = CancelToken::new();
//! let audio = render_svp(&svp, &options, &cancel)?;
//! let wav = encode_wav(&audio, options.sample_rate, &cancel)?;
//! assert!(wav.starts_with(b"RIFF"));
//! # Ok::<(), opensvr::Error>(())
//! ```

mod options;
mod phonemes;

use std::io;
use std::path::{Path, PathBuf, absolute};

pub use opensvr_audio::{
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, NeuralBackend, NoProgress, ProgressSink, RenderError,
    Renderer, StereoBuffer, ToneBackend, VoiceBackend, WavError,
};
pub use opensvr_core::{CancelToken, Language, Note, NoteGroup, Project, Track, VoiceSettings};
pub use opensvr_g2p::{G2pError, Resolved};
pub use opensvr_svp::SvpError;
pub use options::{Backend, DEFAULT_SAMPLE_RATE, Options};
pub use phonemes::{NotePhonemes, PhonemeError, PhonemeResolver};

use thiserror::Error;

/// A failure from any stage of a render.
///
/// The variants mirror the crate the failure came from, so a caller can still
/// recover the concrete error: loading is [`SvpError`], rendering is
/// [`RenderError`] and WAV output is [`WavError`].
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Load(#[from] SvpError),
    #[error(transparent)]
    Render(#[from] RenderError),
    #[error(transparent)]
    Wav(#[from] WavError),
    #[error("cannot resolve {}: {source}", path.display())]
    Path {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// Parses a project from an SVP string.
///
/// `name` only labels the project (the file name `load` would use), and
/// `base_dir` anchors the relative voice, dictionary and backing-track paths
/// the file carries. Pass `None` for a string with no location: those paths
/// stay relative and resolve against the process directory.
pub fn parse(svp: &str, name: &str, base_dir: Option<&Path>) -> Result<Project, Error> {
    let base_dir = base_dir.unwrap_or_else(|| Path::new(""));
    Ok(opensvr_svp::parse(svp.as_bytes(), name, base_dir)?)
}

/// Loads a project from `path` and applies `options` to every track.
///
/// [`Options::base_dir`] is ignored: a file already has a directory, and the
/// project's own relative paths resolve against it.
pub fn load(path: &Path, options: &Options) -> Result<Project, Error> {
    let mut project = opensvr_svp::load(path)?;
    prepare(&mut project, options)?;
    Ok(project)
}

/// Applies the per-track overrides in `options` in place.
///
/// A set override replaces every track's value, and relative paths are made
/// absolute so a later directory change cannot move them. Unset overrides
/// leave the project's own values untouched, which makes this idempotent.
/// [`render_project`] calls it on a clone, so a render never mutates its
/// argument.
pub fn prepare(project: &mut Project, options: &Options) -> Result<(), Error> {
    let absolute = |path: &Path| {
        absolute(path).map_err(|source| Error::Path {
            path: path.to_owned(),
            source,
        })
    };
    let voice = options.voice.as_deref().map(absolute).transpose()?;
    let dictionary = options.dictionary.as_deref().map(absolute).transpose()?;
    for track in &mut project.tracks {
        if let Some(voice) = &voice {
            track.voice.database = Some(voice.clone());
        }
        if let Some(dictionary) = &dictionary {
            track.voice.dictionary = Some(dictionary.clone());
        }
        if let Some(language) = options.language {
            track.voice.language = language;
        }
    }
    Ok(())
}

/// Parses `svp` and renders it, as one call.
///
/// The project name is empty; use [`parse`] and [`render_project`] when it
/// matters. Relative paths anchor on [`Options::base_dir`].
pub fn render_svp(
    svp: &str,
    options: &Options,
    cancel: &CancelToken,
) -> Result<StereoBuffer, Error> {
    let project = parse(svp, "", options.base_dir.as_deref())?;
    render_project(&project, options, cancel)
}

/// Renders `project` at [`Options::sample_rate`], discarding progress.
///
/// This is [`render_project_with_progress`] with a [`NoProgress`] sink, so the
/// samples are bitwise identical either way.
pub fn render_project(
    project: &Project,
    options: &Options,
    cancel: &CancelToken,
) -> Result<StereoBuffer, Error> {
    render_project_with_progress(project, options, cancel, &NoProgress)
}

/// Renders `project` at [`Options::sample_rate`], reporting each phrase.
///
/// `options` is applied to a clone of the project, so the argument is never
/// mutated. The backend is picked by [`Options::backend`], with [`Backend::Auto`]
/// reproducing the command line's rule: a track whose database is an existing
/// file sings through the neural voice, and anything else falls back to tones.
pub fn render_project_with_progress(
    project: &Project,
    options: &Options,
    cancel: &CancelToken,
    progress: &dyn ProgressSink,
) -> Result<StereoBuffer, Error> {
    let mut project = project.clone();
    prepare(&mut project, options)?;
    let audio = if options.backend.resolve(&project) == Backend::Neural {
        Renderer::new(NeuralBackend::with_models_dir(options.models_dir.clone()))
            .render_with_progress(&project, options.sample_rate, cancel, progress)?
    } else {
        Renderer::new(ToneBackend::default()).render_with_progress(
            &project,
            options.sample_rate,
            cancel,
            progress,
        )?
    };
    Ok(audio)
}

/// Writes a 24-bit stereo WAV, replacing `path` only on success.
pub fn write_wav(
    path: &Path,
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<(), Error> {
    Ok(opensvr_audio::write_wav(path, audio, sample_rate, cancel)?)
}

/// Encodes a 24-bit stereo WAV in memory, for callers with no path to replace.
pub fn encode_wav(
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<Vec<u8>, Error> {
    Ok(opensvr_audio::encode_wav(audio, sample_rate, cancel)?)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use opensvr_core::{Language, NoteGroup, Track};

    use super::*;

    /// The checked-in example: two tracks, no voice file on disk, so it
    /// renders as tones and needs no proprietary data.
    fn example() -> &'static str {
        include_str!("../../../examples/two-tracks.svp")
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("opensvr-facade-{}-{name}", std::process::id()))
    }

    fn project_with_two_tracks() -> Project {
        let first = Track::new("Lead", NoteGroup::default());
        let mut second = Track::new("Harmony", NoteGroup::default());
        second.voice.database = Some(PathBuf::from("voices/original.nofs"));
        Project {
            name: "song".into(),
            tracks: vec![first, second],
            ..Project::default()
        }
    }

    #[test]
    fn parse_anchors_relative_paths_on_the_base_directory() {
        let anchored = parse(example(), "two-tracks", Some(Path::new("/proj"))).unwrap();
        assert_eq!(
            anchored.tracks[0].voice.database.as_deref(),
            Some(Path::new("/proj/voices/example/voice.nofs"))
        );
        assert_eq!(
            anchored.tracks[0].voice.dictionary.as_deref(),
            Some(Path::new("/proj/clf-data"))
        );

        // With no location the file's relative paths stay relative: resolving
        // them against the process directory is the caller's choice, not a
        // silent `"."` here.
        let floating = parse(example(), "two-tracks", None).unwrap();
        assert_eq!(
            floating.tracks[0].voice.database.as_deref(),
            Some(Path::new("voices/example/voice.nofs"))
        );
    }

    #[test]
    fn prepare_applies_every_override_and_absolutizes_paths() {
        let mut project = project_with_two_tracks();
        let voice = std::env::temp_dir().join("chosen-voice.nofs");
        let dictionary = std::env::temp_dir().join("chosen-dictionary");
        let options = Options {
            voice: Some(voice.clone()),
            dictionary: Some(dictionary.clone()),
            language: Some(Language::English),
            ..Options::default()
        };
        prepare(&mut project, &options).unwrap();

        let voice = absolute(voice).unwrap();
        let dictionary = absolute(dictionary).unwrap();
        for track in &project.tracks {
            assert_eq!(track.voice.database.as_deref(), Some(voice.as_path()));
            assert_eq!(
                track.voice.dictionary.as_deref(),
                Some(dictionary.as_path())
            );
            assert_eq!(track.voice.language, Language::English);
        }
        // Running it again changes nothing: overrides replace, they do not
        // accumulate or re-anchor what is already absolute.
        let once = project.clone();
        prepare(&mut project, &options).unwrap();
        assert_eq!(project.tracks, once.tracks);
    }

    #[test]
    fn prepare_leaves_the_project_alone_without_overrides() {
        let mut project = project_with_two_tracks();
        let before = project.clone();
        prepare(&mut project, &Options::default()).unwrap();
        assert_eq!(project.tracks, before.tracks);
    }

    #[test]
    fn language_override_wins_over_a_project_that_names_a_language() {
        // The project itself says English through its main reference's
        // `database` block (Studio's form: no `svVoice`).
        let svp = r#"{
            "version": 153,
            "time": {
                "tempo": [{ "position": 0, "bpm": 120.0 }],
                "meter": [{ "index": 0, "numerator": 4, "denominator": 4 }]
            },
            "library": [],
            "tracks": [{
                "name": "Lead",
                "mainGroup": { "uuid": "main", "name": "main", "notes": [] },
                "mainRef": {
                    "groupID": "main",
                    "blickOffset": 0,
                    "pitchOffset": 0,
                    "database": {
                        "name": "voice",
                        "language": "english",
                        "phoneset": "arpabet",
                        "languageOverride": "",
                        "phonesetOverride": "",
                        "backendType": "",
                        "version": "-2"
                    }
                },
                "groups": [],
                "mixer": { "gainDecibel": 0.0, "pan": 0.0 }
            }]
        }"#;
        let mut project = parse(svp, "song", None).unwrap();
        assert_eq!(project.tracks[0].voice.language, Language::English);

        let options = Options {
            language: Some(Language::Japanese),
            ..Options::default()
        };
        prepare(&mut project, &options).unwrap();
        assert_eq!(project.tracks[0].voice.language, Language::Japanese);
    }

    #[test]
    fn auto_backend_follows_the_command_lines_rule() {
        let mut project = project_with_two_tracks();
        assert_eq!(Backend::Auto.resolve(&project), Backend::Tones);

        // A track whose database is a real file sings through the voice, even
        // when its neighbours have none.
        let voice = scratch("present.nofs");
        fs::write(&voice, b"not a real database, only a path").unwrap();
        project.tracks[0].voice.database = Some(voice.clone());
        assert_eq!(Backend::Auto.resolve(&project), Backend::Neural);

        // A configured-but-missing database counts as voiceless, as in the CLI.
        project.tracks[1].voice.database = Some(scratch("missing.nofs"));
        fs::remove_file(&voice).unwrap();
        assert_eq!(Backend::Auto.resolve(&project), Backend::Tones);

        // Explicit choices are never second-guessed.
        assert_eq!(Backend::Neural.resolve(&project), Backend::Neural);
        assert_eq!(Backend::Tones.resolve(&project), Backend::Tones);
    }

    #[test]
    fn render_svp_matches_parse_then_render() {
        let options = Options {
            sample_rate: 22_050,
            ..Options::default()
        };
        let cancel = CancelToken::new();
        let from_string = render_svp(example(), &options, &cancel).unwrap();

        let project = parse(example(), "two-tracks", None).unwrap();
        let from_project = render_project(&project, &options, &cancel).unwrap();
        assert_eq!(from_string, from_project);
        // 4 quarters of Lead and a Harmony motif ending at beat 7: 7 * 0.5 s
        // plus 0.1 s of release, as the CLI test pins.
        assert_eq!(from_string.frames(), 79_380);
    }

    #[test]
    fn auto_falls_back_when_the_voice_override_is_missing() {
        let options = Options {
            sample_rate: 8_000,
            voice: Some(scratch("definitely-missing.nofs")),
            ..Options::default()
        };
        let audio = render_svp(example(), &options, &CancelToken::new()).unwrap();
        assert!(audio.frames() > 0);
    }

    #[test]
    fn explicit_neural_reports_a_missing_voice() {
        let options = Options {
            sample_rate: 8_000,
            backend: Backend::Neural,
            ..Options::default()
        };
        let error = render_svp(example(), &options, &CancelToken::new()).unwrap_err();
        assert!(
            matches!(error, Error::Render(RenderError::Backend { .. })),
            "{error}"
        );
    }

    #[test]
    fn rejects_a_sample_rate_outside_the_renderer_range() {
        let options = Options {
            sample_rate: MIN_SAMPLE_RATE - 1,
            ..Options::default()
        };
        let error = render_svp(example(), &options, &CancelToken::new()).unwrap_err();
        assert!(
            matches!(
                error,
                Error::Render(RenderError::InvalidSampleRate(rate))
                    if rate == MIN_SAMPLE_RATE - 1
            ),
            "{error}"
        );
    }

    #[test]
    fn rejects_an_unsupported_project_version() {
        let svp = example().replace("\"version\": 153", "\"version\": 152");
        let error = parse(&svp, "song", None).unwrap_err();
        assert!(matches!(
            error,
            Error::Load(SvpError::UnsupportedVersion(152))
        ));
    }

    #[test]
    fn wav_helpers_agree_in_memory_and_on_disk() {
        let cancel = CancelToken::new();
        let audio = StereoBuffer {
            left: vec![0.5, -1.0],
            right: vec![0.0, 1.0],
        };
        let encoded = encode_wav(&audio, 44_100, &cancel).unwrap();

        let path = scratch("round-trip.wav");
        write_wav(&path, &audio, 44_100, &cancel).unwrap();
        assert_eq!(fs::read(&path).unwrap(), encoded);

        let reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (2, 44_100, 24)
        );
        assert_eq!(reader.duration(), 2);
        fs::remove_file(path).unwrap();
    }
}
