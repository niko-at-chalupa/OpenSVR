//! The three subcommands.

use std::{
    fs, io,
    path::{self, Path, PathBuf},
};

use opensvr_audio::{CancelToken, RenderError, Renderer, ToneBackend, WavError, write_wav};
use opensvr_core::Project;
use opensvr_midi::MidiError;
use opensvr_svp::SvpError;
use thiserror::Error;

use crate::cli::{Command, InfoArgs, MidiArgs, Overrides, RenderArgs};

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
            Self::Load(_) => 2,
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
    }
}

fn info(args: &InfoArgs) -> Result<(), CliError> {
    let project = load(&args.project, &args.overrides)?;
    print_info(&project, &mut io::stdout().lock(), args.phonemes)?;
    Ok(())
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
    // Not progress output, so `--quiet` does not hide it: the result is not singing.
    eprintln!(
        "opensvr: note: the neural voice engine is not ported yet; notes are rendered as placeholder tones"
    );

    let audio = Renderer::new(ToneBackend::default()).render(&project, args.rate, cancel)?;
    write_wav(&args.output, &audio, args.rate, cancel)?;
    if !args.quiet {
        let seconds = audio.frames() as f64 / f64::from(args.rate);
        eprintln!("Wrote {} ({seconds:.2} s)", args.output.display());
    }
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

/// Loads a project and applies the command-line overrides to every track.
fn load(path: &Path, overrides: &Overrides) -> Result<Project, CliError> {
    let mut project = opensvr_svp::load(path)?;
    let absolute = |path: &Path| path::absolute(path);
    for track in &mut project.tracks {
        if let Some(voice) = &overrides.voice {
            track.voice.database = Some(absolute(voice)?);
        }
        if let Some(dictionary) = &overrides.dict {
            track.voice.dictionary = Some(absolute(dictionary)?);
        }
        if let Some(language) = overrides.language {
            track.voice.language = language;
        }
    }
    Ok(project)
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
        let next_is_legato = notes
            .get(index + 1)
            .is_some_and(|next| next.lyrics == "-");
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
        let dir = std::env::temp_dir().join(format!(
            "opensvr-info-phonemes-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("japanese-romaji-phones.txt"), "a vowel\nk consonant\nsil silence\n").unwrap();
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
                    },
                    opensvr_core::Note {
                        onset: 1,
                        duration: 1,
                        pitch: 62,
                        lyrics: "-".into(),
                        phonemes: String::new(),
                        detune: 0.0,
                    },
                    opensvr_core::Note {
                        onset: 2,
                        duration: 1,
                        pitch: 64,
                        lyrics: "zzz".into(),
                        phonemes: String::new(),
                        detune: 0.0,
                    },
                ],
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
        assert!(text.contains("\"zzz\" -> [ERROR] No pronunciation for dictionary key 'zzz'."), "{text}");
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
                }],
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
        assert!(text.contains("phonemes: [UNREADABLE: Cannot read synthesis resource:"), "{text}");
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
    }
}
