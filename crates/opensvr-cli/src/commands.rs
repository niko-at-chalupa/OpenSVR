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
    print_info(&project, &mut io::stdout().lock())?;
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

fn print_info(project: &Project, out: &mut impl io::Write) -> io::Result<()> {
    writeln!(out, "Project: {}", project.name)?;
    writeln!(out, "Tracks: {}", project.tracks.len())?;
    // Voice databases are opened lazily and cached per path: several tracks
    // often share one singer, and a 40 MiB voice should not be re-read.
    let mut voices: Vec<(PathBuf, Option<String>)> = Vec::new();
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
        print_info(&project, &mut out).unwrap();
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
