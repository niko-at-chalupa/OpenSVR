//! Command-line definition.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use opensvr_audio::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
use opensvr_core::Language;

const AFTER_HELP: &str = "\
Voice databases and dictionaries are not bundled; point to files you own.
Exit codes: 0 ok, 1 usage error, 2 load error, 3 render/write error, 130 cancelled.";

#[derive(Debug, Parser)]
#[command(name = "opensvr", version, about, after_help = AFTER_HELP)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List tracks, voices and whether their files exist
    Info(InfoArgs),
    /// Render a project to a 24-bit stereo WAV file
    Render(RenderArgs),
    /// Export notes, lyrics, tempo and meter as a Standard MIDI File
    Midi(MidiArgs),
}

/// Settings that replace what the project file says, on every track.
#[derive(Debug, Args)]
pub struct Overrides {
    /// Use this voice database on every track
    #[arg(long, value_name = "VOICE.NOFS")]
    pub voice: Option<PathBuf>,
    /// Use this pronunciation dictionary directory on every track
    #[arg(long, value_name = "CLF-DATA")]
    pub dict: Option<PathBuf>,
    /// Override the singing language on every track (japanese, mandarin, english, cantonese, spanish)
    #[arg(long, value_name = "NAME")]
    pub language: Option<Language>,
}

#[derive(Debug, Args)]
pub struct InfoArgs {
    /// The .svp project to read
    pub project: PathBuf,
    #[command(flatten)]
    pub overrides: Overrides,
}

#[derive(Debug, Args)]
pub struct RenderArgs {
    /// The .svp project to render
    pub project: PathBuf,
    /// Output WAV path
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,
    #[command(flatten)]
    pub overrides: Overrides,
    /// Output sample rate in Hz
    #[arg(
        long,
        value_name = "HZ",
        default_value_t = 48_000,
        value_parser = clap::value_parser!(u32).range(i64::from(MIN_SAMPLE_RATE)..=192_000),
    )]
    pub rate: u32,
    /// Suppress progress messages
    #[arg(short, long)]
    pub quiet: bool,
}

#[derive(Debug, Args)]
pub struct MidiArgs {
    /// The .svp project to read
    pub project: PathBuf,
    /// Output MIDI path
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,
}

const _: () = assert!(
    MAX_SAMPLE_RATE >= 192_000,
    "the CLI cap must not exceed the renderer's"
);

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_render_options() {
        let cli = Cli::try_parse_from([
            "opensvr",
            "render",
            "song.svp",
            "-o",
            "out.wav",
            "--rate",
            "44100",
            "--language",
            "english",
            "-q",
        ])
        .unwrap();
        let Command::Render(args) = cli.command else {
            panic!("expected render")
        };
        assert_eq!((args.rate, args.quiet), (44_100, true));
        assert_eq!(args.overrides.language, Some(Language::English));
    }

    #[test]
    fn rejects_bad_usage() {
        for arguments in [
            &["opensvr"][..],
            &["opensvr", "render", "song.svp"],
            &[
                "opensvr", "render", "song.svp", "-o", "out.wav", "--rate", "100",
            ],
            &["opensvr", "info", "song.svp", "--language", "klingon"],
        ] {
            assert!(Cli::try_parse_from(arguments).is_err(), "{arguments:?}");
        }
    }
}
