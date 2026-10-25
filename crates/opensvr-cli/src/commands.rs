//! The subcommands.

use std::io::Write as _;
use std::{
    ffi::OsStr,
    fmt, fs, io,
    path::{Path, PathBuf},
};

use opensvr::{
    Backend, CancelToken, Error as EngineError, NoProgress, Options, ProgressSink, RenderError,
    StereoBuffer, WavError,
};
use opensvr_core::Project;
use opensvr_midi::MidiError;
use opensvr_svp::SvpError;
use thiserror::Error;

use crate::cli::{
    Command, CompleteArgs, ConvertArgs, GlobalArgs, InfoArgs, MidiArgs, Overrides, RenderArgs,
};

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Load(#[from] SvpError),
    #[error("MIDI export failed: {0}")]
    Midi(#[from] MidiError),
    #[error("MIDI import failed: {0}")]
    MidiLoad(MidiError),
    #[error("cannot serialize the project: {0}")]
    Serialize(SvpError),
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
            Self::Load(_) | Self::MidiLoad(_) | Self::Engine(EngineError::Load(_)) => 2,
            Self::Engine(
                EngineError::Render(RenderError::Cancelled) | EngineError::Wav(WavError::Cancelled),
            ) => 130,
            _ => 3,
        }
    }
}

/// Where a command writes its output: a file path, or standard output when
/// the caller passes `-`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Destination {
    File(PathBuf),
    Stdout,
}

impl fmt::Display for Destination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => write!(f, "{}", path.display()),
            Self::Stdout => f.write_str("standard output"),
        }
    }
}

/// Resolves `-o` against the default path derived from the input.
fn destination(output: Option<&Path>, default: PathBuf) -> Destination {
    match output {
        Some(path) if is_stdout(path) => Destination::Stdout,
        Some(path) => Destination::File(path.to_owned()),
        None => Destination::File(default),
    }
}

/// Whether `-o` asked for stdout rather than a file named `-`.
fn is_stdout(path: &Path) -> bool {
    path.as_os_str() == OsStr::new("-")
}

/// Writes `bytes` to a file, or streams them to stdout for [`Destination::Stdout`].
fn write_bytes(destination: &Destination, bytes: &[u8]) -> Result<(), CliError> {
    match destination {
        Destination::File(path) => fs::write(path, bytes).map_err(|source| CliError::Write {
            path: path.clone(),
            source,
        }),
        Destination::Stdout => {
            io::stdout().lock().write_all(bytes)?;
            Ok(())
        }
    }
}

pub fn run(global: GlobalArgs, command: Command, cancel: &CancelToken) -> Result<(), CliError> {
    match command {
        Command::Info(args) => info(&args),
        Command::Render(args) => render(global, &args, cancel),
        Command::Midi(args) => midi(&args),
        Command::Convert(args) => convert(&args),
        Command::Complete(args) => complete(args),
        Command::Man => man(),
    }
}

fn info(args: &InfoArgs) -> Result<(), CliError> {
    let settings = options(&args.overrides);
    let project = opensvr::load(&args.project, &settings)?;
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

fn render(global: GlobalArgs, args: &RenderArgs, cancel: &CancelToken) -> Result<(), CliError> {
    let mut options = options(&args.overrides);
    options.sample_rate = args.rate;
    let project = opensvr::load(&args.project, &options)?;
    let output = destination(args.output.as_deref(), args.project.with_extension("wav"));
    if !global.quiet {
        eprintln!(
            "Rendering {} ({} tracks) at {} Hz...",
            args.project.display(),
            project.tracks.len(),
            args.rate
        );
    }
    // Backend selection lives in the library (`Backend::Auto`), so an embed
    // and the CLI cannot disagree: any voice database that resolves to a real
    // file renders the whole project through the neural voice, matching C++
    // `ProjectRenderer`, which fails audible tracks without one ("Select a
    // voice.nofs database before playing singing notes."). A configured but
    // missing database counts as voiceless: the placeholder keeps those
    // renders (and CI without proprietary data) working, warning included,
    // while `info` flags the path as [MISSING] for diagnosis.
    let backend = options.backend.resolve(&project);
    if global.verbose {
        eprintln!(
            "opensvr: backend: {}",
            if backend == Backend::Neural {
                "neural voice"
            } else {
                "placeholder tones"
            }
        );
        eprintln!("opensvr: output: {output}");
    }
    // Phrase progress goes to an `indicatif` bar on stderr unless `--quiet`.
    // The total is only known inside the renderer, so the bar starts without
    // a length and learns it from the first report; quiet renders pass the
    // no-op sink, keeping their output byte-identical.
    let bar = if global.quiet {
        None
    } else {
        Some(indicatif::ProgressBar::new(0))
    };
    let sink = bar.clone().map(|bar| BarProgress { bar });
    let progress: &dyn ProgressSink = sink.as_ref().map_or(&NoProgress, |sink| sink);
    if backend == Backend::Tones {
        // Not progress output, so `--quiet` does not hide it: the result is not singing.
        eprintln!(
            "opensvr: note: no voice database found; notes are rendered as placeholder tones"
        );
    }
    let result = opensvr::render_project_with_progress(&project, &options, cancel, progress);
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
    write_output(&output, &audio, args.rate, cancel)?;
    if !global.quiet {
        let seconds = audio.frames() as f64 / f64::from(args.rate);
        eprintln!("Wrote {output} ({seconds:.2} s)");
    }
    Ok(())
}

/// Writes the rendered audio to a file (atomically) or to stdout.
fn write_output(
    destination: &Destination,
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<(), CliError> {
    match destination {
        Destination::File(path) => {
            opensvr::write_wav(path, audio, sample_rate, cancel)?;
        }
        Destination::Stdout => {
            io::stdout()
                .lock()
                .write_all(&opensvr::encode_wav(audio, sample_rate, cancel)?)?;
        }
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
    write_bytes(
        &destination(args.output.as_deref(), args.project.with_extension("mid")),
        &bytes,
    )
}

/// Converts between `.svp` and `.mid` in any direction: same-extension pairs
/// are load-then-save round trips, not byte copies. Omitting `-o` writes the
/// other format beside the input; `-o -` streams to stdout.
fn convert(args: &ConvertArgs) -> Result<(), CliError> {
    let input = format_of(&args.input).ok_or_else(|| CliError::UnsupportedExtension {
        path: args.input.clone(),
    })?;
    let destination = destination(
        args.output.as_deref(),
        args.input.with_extension(input.other().extension()),
    );
    let output = match &args.output {
        // `-` has no extension to read a format from, so target the other one.
        Some(path) if is_stdout(path) => input.other(),
        Some(path) => {
            format_of(path).ok_or_else(|| CliError::UnsupportedExtension { path: path.clone() })?
        }
        None => input.other(),
    };
    let project = match input {
        Format::Svp => opensvr_svp::load(&args.input)?,
        Format::Midi => opensvr_midi::load(&args.input).map_err(CliError::MidiLoad)?,
    };
    match output {
        Format::Midi => write_bytes(&destination, &opensvr_midi::export(&project)?),
        Format::Svp => match &destination {
            Destination::File(path) => {
                opensvr_svp::save(path, &project).map_err(|source| CliError::ConvertWrite {
                    path: path.clone(),
                    source,
                })
            }
            Destination::Stdout => {
                // No output directory exists on stdout: keep relative paths
                // anchored where the input project's own were.
                let base_dir = args.input.parent().unwrap_or_else(|| Path::new(""));
                let bytes = opensvr_svp::encode(&project, base_dir).map_err(CliError::Serialize)?;
                write_bytes(&destination, &bytes)
            }
        },
    }
}

/// A project file format, picked by file extension (case-insensitively, so
/// `song.MID` works on case-preserving filesystems).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Svp,
    Midi,
}

impl Format {
    /// The other format, picked when `-o` is omitted.
    const fn other(self) -> Self {
        match self {
            Self::Svp => Self::Midi,
            Self::Midi => Self::Svp,
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Svp => "svp",
            Self::Midi => "mid",
        }
    }
}

fn format_of(path: &Path) -> Option<Format> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if extension.eq_ignore_ascii_case("svp") => Some(Format::Svp),
        Some(extension) if extension.eq_ignore_ascii_case("mid") => Some(Format::Midi),
        _ => None,
    }
}

/// Turns flags and environment into the facade's render settings.
///
/// Unset `--voice`/`--dictionary` flags fall back to their environment
/// variables (non-empty only), so flag-beats-env-beats-file precedence holds
/// everywhere. `OPENSVR_DICT` is kept as a legacy alias. Everything after
/// this point — absolutizing paths, applying them to every track, choosing a
/// backend — is the library's job.
fn options(overrides: &Overrides) -> Options {
    Options {
        voice: overrides
            .voice
            .clone()
            .or_else(|| env_path("OPENSVR_VOICE")),
        dictionary: overrides
            .dictionary
            .clone()
            .or_else(|| env_path("OPENSVR_DICTIONARY").or_else(|| env_path("OPENSVR_DICT"))),
        models_dir: overrides
            .models_dir
            .clone()
            .or_else(|| env_path("OPENSVR_MODELS_DIR")),
        language: overrides.language.map(Into::into),
        ..Options::default()
    }
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
    // Dictionaries are likewise cached per directory and language: one
    // resolver per `info` run shares them across tracks.
    let mut resolver = opensvr::PhonemeResolver::new();
    for (index, track) in project.tracks.iter().enumerate() {
        let voice = &track.voice;
        let note_count = project
            .placed_notes(track)
            .map_or_else(|_| "?".to_owned(), |placed| placed.len().to_string());
        writeln!(
            out,
            "  [{index}] {}  notes={note_count}  language={}",
            track.name, voice.language
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
            print_phonemes(project, track, out, &mut resolver)?;
        }
    }
    Ok(())
}

/// Resolves and prints one line per placed note of the track.
///
/// Dictionary load failures print once per track; per-note resolution errors
/// print on their note's line. Neither is fatal: `info` stays diagnostic.
fn print_phonemes(
    project: &Project,
    track: &opensvr_core::Track,
    out: &mut impl io::Write,
    resolver: &mut opensvr::PhonemeResolver,
) -> io::Result<()> {
    match resolver.resolve_track(project, track) {
        Ok(entries) => {
            // An empty track prints no header.
            if entries.is_empty() {
                return Ok(());
            }
            writeln!(out, "      phonemes:")?;
            for entry in entries {
                match &entry.result {
                    Ok(outcome) => writeln!(
                        out,
                        "        {:?} -> {} ({})",
                        entry.note.lyrics,
                        outcome.phonemes.join(" "),
                        outcome.language
                    )?,
                    Err(error) => {
                        writeln!(out, "        {:?} -> [ERROR] {error}", entry.note.lyrics)?;
                    }
                }
            }
        }
        Err(opensvr::PhonemeError::NoDictionary) => {
            writeln!(out, "      phonemes: [no dictionary directory]")?;
        }
        Err(error) => writeln!(out, "      phonemes: [UNREADABLE: {error}]")?,
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
        track.voice.language = Language::Japanese;
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
        assert_eq!(
            CliError::Engine(EngineError::Render(RenderError::Cancelled)).exit_code(),
            130
        );
        assert_eq!(
            CliError::Engine(EngineError::Wav(WavError::Cancelled)).exit_code(),
            130
        );
        assert_eq!(
            CliError::Engine(EngineError::Render(RenderError::TooLong)).exit_code(),
            3
        );
        assert_eq!(
            CliError::Engine(EngineError::Load(SvpError::UnsupportedVersion(1))).exit_code(),
            2
        );
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
    fn output_paths_default_from_the_input() {
        assert_eq!(
            destination(None, PathBuf::from("song.wav")),
            Destination::File(PathBuf::from("song.wav"))
        );
        assert_eq!(
            destination(Some(Path::new("out.wav")), PathBuf::from("song.wav")),
            Destination::File(PathBuf::from("out.wav")),
            "an explicit -o wins over the default"
        );
        assert_eq!(
            destination(Some(Path::new("-")), PathBuf::from("song.wav")),
            Destination::Stdout
        );
        assert!(is_stdout(Path::new("-")));
        assert!(
            !is_stdout(Path::new("./-")),
            "only a bare dash means stdout"
        );
    }

    #[test]
    fn formats_swap_and_name_their_extension() {
        assert_eq!(Format::Svp.other(), Format::Midi);
        assert_eq!(Format::Midi.other(), Format::Svp);
        assert_eq!(Format::Svp.extension(), "svp");
        assert_eq!(Format::Midi.extension(), "mid");
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
