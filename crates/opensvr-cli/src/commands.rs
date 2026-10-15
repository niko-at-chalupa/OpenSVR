//! The three subcommands.

use std::io::Write as _;
use std::{
    fs, io,
    path::{self, Path, PathBuf},
};

use opensvr_audio::{
    CancelToken, NeuralBackend, NoProgress, ProgressSink, RenderError, Renderer, ToneBackend,
    WavError, write_wav,
};
use opensvr_core::Project;
use opensvr_midi::MidiError;
use opensvr_svp::SvpError;
use thiserror::Error;

use crate::cli::{Command, CompleteArgs, ConvertArgs, InfoArgs, MidiArgs, Overrides, RenderArgs};

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    Load(#[from] SvpError),
    #[error(transparent)]
    Render(#[from] RenderError),
    #[error(transparent)]
    Wav(#[from] WavError),
    #[error("MIDI export failed: {0}")]
    Midi(#[from] MidiError),
    #[error("MIDI import failed: {0}")]
    MidiLoad(MidiError),
    #[error("cannot write {}: {0}", .path.display())]
    ConvertWrite {
        path: PathBuf,
        #[source]
        source: SvpError,
    },
    #[error("unsupported extension for {} (expected .svp or .mid)", .path.display())]
    UnsupportedExtension { path: PathBuf },
    #[error("cannot write {}: {source}", .path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl CliError {
    /// Process exit status: 2 for load errors, 130 for cancellation, 3 for everything else.
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Load(_) | Self::MidiLoad(_) => 2,
            Self::Render(RenderError::Cancelled) | Self::Wav(WavError::Cancelled) => 130,
            _ => 3,
        }
    }
}

pub fn run(command: Command, cancel: &CancelToken) -> Result<(), CliError> {
    match command {
        Command::Info(args) => info(&args),
        Command::Render(args) => render(&args, cancel),
        Command::Midi(args) => midi(&args),
        Command::Convert(args) => convert(&args),
        Command::Complete(args) => complete(args),
        Command::Man => man(),
    }
}

fn info(args: &InfoArgs) -> Result<(), CliError> {
    let project = load(&args.project, &args.overrides)?;
    print_info(&project, &mut io::stdout().lock(), args.phonemes)?;
    Ok(())
}

/// A [`ProgressSink`] that drives an `indicatif` bar on stderr.
///
/// The bar starts without a length (the phrase total is only known inside the
/// renderer) and learns it from the first report. `indicatif` hides the bar
/// when stderr is piped, so scripts never see it; `--quiet` never creates one.
#[derive(Debug, Clone)]
struct BarProgress {
    bar: indicatif::ProgressBar,
}

impl ProgressSink for BarProgress {
    fn report(&self, done: usize, total: usize) {
        if self.bar.length().is_none() {
            self.bar.set_length(total as u64);
        }
        self.bar.set_position(done as u64);
    }
}

fn render(args: &RenderArgs, cancel: &CancelToken) -> Result<(), CliError> {
    let project = load(&args.project, &args.overrides)?;
    if !args.quiet {
        eprintln!(
            "Rendering {} ({} tracks) at {} Hz...",
            args.project.display(),
            project.tracks.len(),
            args.rate
        );
    }
    // Backend selection: any voice database that resolves to a real file
    // renders the whole project through the neural voice, matching C++
    // `ProjectRenderer`, which fails audible tracks without one ("Select a
    // voice.nofs database before playing singing notes."). Tracks without
    // notes render silence under either backend, so mixed projects only fail
    // when a voiceless track actually has notes to sing. A configured but
    // missing database counts as voiceless: the placeholder keeps those
    // renders (and CI without proprietary data) working, warning included,
    // while `info` flags the path as [MISSING] for diagnosis.
    let voiced = project
        .tracks
        .iter()
        .any(|track| track.voice.database.as_deref().is_some_and(Path::is_file));
    // Phrase progress goes to an `indicatif` bar on stderr unless `--quiet`.
    // The total is only known inside the renderer, so the bar starts without
    // a length and learns it from the first report; quiet renders pass the
    // no-op sink, keeping their output byte-identical.
    let bar = if args.quiet {
        None
    } else {
        Some(indicatif::ProgressBar::new(0))
    };
    let sink = bar.clone().map(|bar| BarProgress { bar });
    let progress: &dyn ProgressSink = sink.as_ref().map_or(&NoProgress, |sink| sink);
    let result = if voiced {
        Renderer::new(NeuralBackend::new())
            .render_with_progress(&project, args.rate, cancel, progress)
    } else {
        // Not progress output, so `--quiet` does not hide it: the result is not singing.
        eprintln!(
            "opensvr: note: no voice database found; notes are rendered as placeholder tones"
        );
        Renderer::new(ToneBackend::default())
            .render_with_progress(&project, args.rate, cancel, progress)
    };
    let audio = match result {
        Ok(audio) => {
            if let Some(bar) = &bar {
                bar.finish_and_clear();
            }
            audio
        }
        Err(error) => {
            if let Some(bar) = &bar {
                bar.abandon();
            }
            return Err(error.into());
        }
    };
    write_wav(&args.output, &audio, args.rate, cancel)?;
    if !args.quiet {
        let seconds = audio.frames() as f64 / f64::from(args.rate);
        eprintln!("Wrote {} ({seconds:.2} s)", args.output.display());
    }
    Ok(())
}

/// Prints a completion script for the requested shell to stdout.
///
/// The script is rendered into memory first so a closed stream surfaces as
/// a write error (exit 3) instead of a panic.
fn complete(args: CompleteArgs) -> Result<(), CliError> {
    io::stdout()
        .lock()
        .write_all(&crate::cli::completion_script(args.shell))?;
    Ok(())
}

/// Prints the troff man page (`opensvr.1`) to stdout.
fn man() -> Result<(), CliError> {
    io::stdout().lock().write_all(&crate::cli::man_page())?;
    Ok(())
}

fn midi(args: &MidiArgs) -> Result<(), CliError> {
    let project = opensvr_svp::load(&args.project)?;
    let bytes = opensvr_midi::export(&project)?;
    fs::write(&args.output, bytes).map_err(|source| CliError::Write {
        path: args.output.clone(),
        source,
    })
}

/// Converts between `.svp` and `.mid` in any direction: same-extension pairs
/// are load-then-save round trips, not byte copies.
fn convert(args: &ConvertArgs) -> Result<(), CliError> {
    let input = format_of(&args.input).ok_or_else(|| CliError::UnsupportedExtension {
        path: args.input.clone(),
    })?;
    let output = format_of(&args.output).ok_or_else(|| CliError::UnsupportedExtension {
        path: args.output.clone(),
    })?;
    let project = match input {
        Format::Svp => opensvr_svp::load(&args.input)?,
        Format::Midi => opensvr_midi::load(&args.input).map_err(CliError::MidiLoad)?,
    };
    match output {
        Format::Svp => {
            opensvr_svp::save(&args.output, &project).map_err(|source| CliError::ConvertWrite {
                path: args.output.clone(),
                source,
            })?;
        }
        Format::Midi => {
            let bytes = opensvr_midi::export(&project)?;
            fs::write(&args.output, bytes).map_err(|source| CliError::Write {
                path: args.output.clone(),
                source,
            })?;
        }
    }
    Ok(())
}

/// A project file format, picked by file extension (case-insensitively, so
/// `song.MID` works on case-preserving filesystems).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Svp,
    Midi,
}

fn format_of(path: &Path) -> Option<Format> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("svp") => Some(Format::Svp),
        Some(extension) if extension.eq_ignore_ascii_case("mid") => Some(Format::Midi),
        _ => None,
    }
}

/// Loads a project and applies the command-line overrides to every track.
///
/// Unset `--voice`/`--dict` flags fall back to `OPENSVR_VOICE`/`OPENSVR_DICT`
/// (non-empty only), so both the subcommand and flat forms share one
/// flag-beats-env-beats-file precedence.
fn load(path: &Path, overrides: &Overrides) -> Result<Project, CliError> {
    let mut project = opensvr_svp::load(path)?;
    let voice = overrides
        .voice
        .clone()
        .or_else(|| env_path("OPENSVR_VOICE"));
    let dictionary = overrides.dict.clone().or_else(|| env_path("OPENSVR_DICT"));
    let absolute = |path: &Path| path::absolute(path);
    for track in &mut project.tracks {
        if let Some(voice) = &voice {
            track.voice.database = Some(absolute(voice)?);
        }
        if let Some(dictionary) = &dictionary {
            track.voice.dictionary = Some(absolute(dictionary)?);
        }
        if let Some(language) = overrides.language {
            track.voice.language = language;
        }
    }
    Ok(project)
}

/// Reads an optional path override from the environment: unset or empty
/// counts as absent, so `VAR=""` never becomes a relative path.
fn env_path(name: &str) -> Option<PathBuf> {
    non_empty(std::env::var_os(name))
}

/// Drops empty values, keeping them from turning into bogus relative paths.
fn non_empty(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    let value = value?;
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn print_info(project: &Project, out: &mut impl io::Write, show_phonemes: bool) -> io::Result<()> {
    writeln!(out, "Project: {}", project.name)?;
    writeln!(out, "Tracks: {}", project.tracks.len())?;
    // Voice databases are opened lazily and cached per path: several tracks
    // often share one singer, and a 40 MiB voice should not be re-read.
    let mut voices: Vec<(PathBuf, Option<String>)> = Vec::new();
    // Dictionaries are likewise cached per directory and language.
    let mut dictionaries: Vec<(PathBuf, opensvr_core::Language, opensvr_g2p::DictionarySet)> =
        Vec::new();
    for (index, track) in project.tracks.iter().enumerate() {
        let voice = &track.voice;
        writeln!(
            out,
            "  [{index}] {}  notes={}  language={}",
            track.name,
            track.main_group.notes.len(),
            voice.language
        )?;
        writeln!(
            out,
            "      voice: {}",
            describe(voice.database.as_deref(), Path::is_file)
        )?;
        writeln!(
            out,
            "      dict:  {}",
            describe(voice.dictionary.as_deref(), Path::is_dir)
        )?;
        if let Some(path) = voice.database.as_deref()
            && path.is_file()
        {
            let cached = voices
                .iter()
                .find(|(known, _)| known == path)
                .map(|(_, summary)| summary.clone());
            let summary = cached.unwrap_or_else(|| {
                let summary = voice_summary(path);
                voices.push((path.to_owned(), summary.clone()));
                summary
            });
            match summary {
                Some(summary) => writeln!(out, "      singer: {summary}")?,
                None => writeln!(out, "      singer: [UNREADABLE]")?,
            }
        }
        if show_phonemes {
            print_phonemes(track, out, &mut dictionaries)?;
        }
    }
    Ok(())
}

/// Resolves and prints one line per note of the track's main group.
///
/// Dictionary load failures print once per track; per-note resolution errors
/// print on their note's line. Neither is fatal: `info` stays diagnostic.
fn print_phonemes(
    track: &opensvr_core::Track,
    out: &mut impl io::Write,
    dictionaries: &mut Vec<(PathBuf, opensvr_core::Language, opensvr_g2p::DictionarySet)>,
) -> io::Result<()> {
    let notes = &track.main_group.notes;
    if notes.is_empty() {
        return Ok(());
    }
    let Some(dir) = track.voice.dictionary.clone() else {
        writeln!(out, "      phonemes: [no dictionary directory]")?;
        return Ok(());
    };
    let language = track.voice.language;
    let cached = dictionaries
        .iter()
        .find(|(known_dir, known_language, _)| known_dir == &dir && *known_language == language)
        .map(|(_, _, set)| set.clone());
    let set = match cached {
        Some(set) => set,
        None => match opensvr_g2p::load_dictionaries(&dir, language) {
            Ok(set) => {
                dictionaries.push((dir.clone(), language, set.clone()));
                set
            }
            Err(error) => {
                writeln!(out, "      phonemes: [UNREADABLE: {error}]")?;
                return Ok(());
            }
        },
    };
    let mut resolver = opensvr_g2p::NoteResolver::new(language);
    writeln!(out, "      phonemes:")?;
    for (index, note) in notes.iter().enumerate() {
        let next_is_legato = notes.get(index + 1).is_some_and(|next| next.lyrics == "-");
        match resolver.push(&note.lyrics, &note.phonemes, &set, next_is_legato) {
            Ok(resolved) => writeln!(
                out,
                "        {:?} -> {} ({})",
                note.lyrics,
                resolved.phonemes.join(" "),
                resolved.language
            )?,
            Err(error) => writeln!(out, "        {:?} -> [ERROR] {error}", note.lyrics)?,
        }
    }
    Ok(())
}

/// One-line singer summary from a voice database, or `None` when it cannot be read.
fn voice_summary(path: &Path) -> Option<String> {
    use std::fmt::Write as _;
    let database = opensvr_nofs::VoiceDatabase::open(path).ok()?;
    let metadata = database.metadata();
    if metadata.name.is_empty() {
        return Some("(unknown voice)".to_owned());
    }
    let mut summary = metadata.name.clone();
    if !metadata.vendor.is_empty() {
        let _ = write!(summary, " ({})", metadata.vendor);
    }
    if !metadata.timbre_styles.is_empty() {
        let _ = write!(summary, "  modes={}", metadata.timbre_styles.join(" "));
    }
    Some(summary)
}

/// Formats an optional path, flagging it when `exists` says it is not there.
fn describe(path: Option<&Path>, exists: fn(&Path) -> bool) -> String {
    match path {
        None => "(none)".to_owned(),
        Some(path) if exists(path) => path.display().to_string(),
        Some(path) => format!("{}  [MISSING]", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use opensvr_core::{Language, NoteGroup, Track};

    use super::*;

    #[test]
    fn empty_env_values_count_as_unset() {
        assert_eq!(non_empty(None), None);
        assert_eq!(non_empty(Some(std::ffi::OsString::from(""))), None);
        assert_eq!(
            non_empty(Some(std::ffi::OsString::from("/voices/gumi.nofs"))),
            Some(PathBuf::from("/voices/gumi.nofs"))
        );
        assert_eq!(
            env_path("OPENSVR_DEFINITELY_UNSET_VARIABLE"),
            None,
            "an unset variable resolves to no override"
        );
    }

    #[test]
    fn info_flags_missing_files() {
        let mut track = Track::new("Lead", NoteGroup::default());
        track.voice.database = Some(PathBuf::from("/definitely/not/here.nofs"));
        track.voice.dictionary = Some(std::env::temp_dir());
        track.voice.language = Language::English;
        let project = Project {
            name: "song".into(),
            tracks: vec![track],
            ..Project::default()
        };

        let mut out = Vec::new();
        print_info(&project, &mut out, false).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("Project: song\nTracks: 1\n  [0] Lead  notes=0  language=english\n")
        );
        assert!(text.contains("voice: /definitely/not/here.nofs  [MISSING]"));
        assert!(
            !text.contains("dict:  (none)")
                && !text.contains(&format!("{}  [MISSING]", std::env::temp_dir().display()))
        );
    }

    #[test]
    fn info_with_phonemes_resolves_each_note() {
        let dir =
            std::env::temp_dir().join(format!("opensvr-info-phonemes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("japanese-romaji-phones.txt"),
            "a vowel\nk consonant\nsil silence\n",
        )
        .unwrap();
        std::fs::write(dir.join("japanese-romaji-dict.txt"), "ka k a\n").unwrap();
        std::fs::write(dir.join("japanese-hira2romaji-dict.txt"), "か ka\n").unwrap();
        std::fs::write(dir.join("japanese-kata2romaji-dict.txt"), "カ ka\n").unwrap();
        std::fs::write(dir.join("japanese-sute2romaji-dict.txt"), "ゃ ya\n").unwrap();

        let mut track = Track::new(
            "Lead",
            NoteGroup {
                id: "main".into(),
                name: "main".into(),
                notes: vec![
                    opensvr_core::Note {
                        onset: 0,
                        duration: 1,
                        pitch: 60,
                        lyrics: "か".into(),
                        phonemes: String::new(),
                        detune: 0.0,
                        instant_mode: true,
                        musical_type: "singing".into(),
                        accent: String::new(),
                        attributes: opensvr_core::PitchAttributes::default(),
                        system_attributes: opensvr_core::PitchAttributes::default(),
                        extra: opensvr_core::Extra::default(),
                    },
                    opensvr_core::Note {
                        onset: 1,
                        duration: 1,
                        pitch: 62,
                        lyrics: "-".into(),
                        phonemes: String::new(),
                        detune: 0.0,
                        instant_mode: true,
                        musical_type: "singing".into(),
                        accent: String::new(),
                        attributes: opensvr_core::PitchAttributes::default(),
                        system_attributes: opensvr_core::PitchAttributes::default(),
                        extra: opensvr_core::Extra::default(),
                    },
                    opensvr_core::Note {
                        onset: 2,
                        duration: 1,
                        pitch: 64,
                        lyrics: "zzz".into(),
                        phonemes: String::new(),
                        detune: 0.0,
                        instant_mode: true,
                        musical_type: "singing".into(),
                        accent: String::new(),
                        attributes: opensvr_core::PitchAttributes::default(),
                        system_attributes: opensvr_core::PitchAttributes::default(),
                        extra: opensvr_core::Extra::default(),
                    },
                ],
                ..NoteGroup::default()
            },
        );
        track.voice.dictionary = Some(dir);
        let project = Project {
            name: "song".into(),
            tracks: vec![track],
            ..Project::default()
        };

        let mut out = Vec::new();
        print_info(&project, &mut out, true).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"か\" -> k a (japanese)"), "{text}");
        assert!(text.contains("\"-\" -> a (japanese)"), "{text}");
        assert!(
            text.contains("\"zzz\" -> [ERROR] No pronunciation for dictionary key 'zzz'."),
            "{text}"
        );
    }

    #[test]
    fn info_with_phonemes_flags_a_missing_dictionary() {
        let mut track = Track::new(
            "Lead",
            NoteGroup {
                id: "main".into(),
                name: "main".into(),
                notes: vec![opensvr_core::Note {
                    onset: 0,
                    duration: 1,
                    pitch: 60,
                    lyrics: "ka".into(),
                    phonemes: String::new(),
                    detune: 0.0,
                    instant_mode: true,
                    musical_type: "singing".into(),
                    accent: String::new(),
                    attributes: opensvr_core::PitchAttributes::default(),
                    system_attributes: opensvr_core::PitchAttributes::default(),
                    extra: opensvr_core::Extra::default(),
                }],
                ..NoteGroup::default()
            },
        );
        track.voice.dictionary = Some(PathBuf::from("/definitely/not/here"));
        let project = Project {
            name: "song".into(),
            tracks: vec![track],
            ..Project::default()
        };

        let mut out = Vec::new();
        print_info(&project, &mut out, true).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("phonemes: [UNREADABLE: Cannot read synthesis resource:"),
            "{text}"
        );
    }

    #[test]
    fn exit_codes_follow_the_documented_contract() {
        assert_eq!(CliError::Render(RenderError::Cancelled).exit_code(), 130);
        assert_eq!(CliError::Wav(WavError::Cancelled).exit_code(), 130);
        assert_eq!(CliError::Render(RenderError::TooLong).exit_code(), 3);
        assert_eq!(
            CliError::Load(SvpError::UnsupportedVersion(1)).exit_code(),
            2
        );
        assert_eq!(
            CliError::MidiLoad(MidiError::NoNotes).exit_code(),
            2,
            "MIDI load errors are load errors"
        );
        assert_eq!(
            CliError::UnsupportedExtension {
                path: PathBuf::from("song.txt"),
            }
            .exit_code(),
            3,
            "unknown extensions are refused, not misread"
        );
        assert_eq!(
            CliError::ConvertWrite {
                path: PathBuf::from("song.svp"),
                source: SvpError::Invalid("nope".into()),
            }
            .exit_code(),
            3,
            "conversion write errors are write errors"
        );
    }

    #[test]
    fn extensions_match_case_insensitively() {
        assert_eq!(
            format_of(Path::new("song.SVP")),
            Some(Format::Svp),
            "format_of"
        );
        assert_eq!(format_of(Path::new("song.MID")), Some(Format::Midi));
        assert_eq!(format_of(Path::new("song.txt")), None);
        assert_eq!(format_of(Path::new("song")), None);
    }

    #[test]
    fn the_bar_sink_learns_its_total_from_the_first_report() {
        let sink = BarProgress {
            bar: indicatif::ProgressBar::hidden(),
        };
        assert_eq!(sink.bar.length(), None);
        sink.report(1, 3);
        assert_eq!(sink.bar.length(), Some(3));
        assert_eq!(sink.bar.position(), 1);
        sink.report(3, 3);
        assert_eq!(sink.bar.position(), 3);
    }
}
