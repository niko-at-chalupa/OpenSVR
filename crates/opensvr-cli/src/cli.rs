//! Command-line definition.

use std::path::PathBuf;

use clap::{Args, CommandFactory, Parser, Subcommand};
use opensvr_audio::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
use opensvr_core::Language;

const AFTER_HELP: &str = "\
Voice databases and dictionaries are not bundled; point to files you own.
OPENSVR_VOICE and OPENSVR_DICT default --voice and --dict when those flags are absent; explicit flags win.
Exit codes: 0 ok, 1 usage error, 2 load error, 3 render/write error, 130 cancelled.";

#[derive(Debug, Parser)]
#[command(name = "opensvr", version, about, after_help = AFTER_HELP)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum Command {
    /// List tracks, voices and whether their files exist
    Info(InfoArgs),
    /// Render a project to a 24-bit stereo WAV file
    Render(RenderArgs),
    /// Export notes, lyrics, tempo and meter as a Standard MIDI File
    Midi(MidiArgs),
    /// Convert between project formats (.svp and .mid, matched case-insensitively)
    Convert(ConvertArgs),
    /// Print a shell completion script to stdout
    Complete(CompleteArgs),
    /// Print a troff man page to stdout (see `opensvr.1`)
    Man,
}

/// Settings that replace what the project file says, on every track.
///
/// Each path falls back to its environment variable (`OPENSVR_VOICE`,
/// `OPENSVR_DICT`) when the flag is absent; explicit flags win, and empty
/// env values count as unset.
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct Overrides {
    /// Use this voice database on every track (default: OPENSVR_VOICE)
    #[arg(long, value_name = "VOICE.NOFS")]
    pub voice: Option<PathBuf>,
    /// Use this pronunciation dictionary directory on every track (default: OPENSVR_DICT)
    #[arg(long, value_name = "CLF-DATA")]
    pub dict: Option<PathBuf>,
    /// Override the singing language on every track (japanese, mandarin, english, cantonese, spanish)
    #[arg(long, value_name = "NAME")]
    pub language: Option<Language>,
}

#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct InfoArgs {
    /// The .svp project to read
    pub project: PathBuf,
    #[command(flatten)]
    pub overrides: Overrides,
    /// Resolve and print phonemes for each note (needs --dict or voices with dictionary paths)
    #[arg(long)]
    pub phonemes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Args)]
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

#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct MidiArgs {
    /// The .svp project to read
    pub project: PathBuf,
    /// Output MIDI path
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct ConvertArgs {
    /// Input project (.svp or .mid)
    pub input: PathBuf,
    /// Output project (.svp or .mid)
    pub output: PathBuf,
}

/// Target shell for `complete` (one of bash, elvish, fish, powershell, zsh).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Args)]
pub struct CompleteArgs {
    /// The shell to print completions for
    pub shell: clap_complete::Shell,
}

/// Renders the checked-in man page (`opensvr.1`) from the CLI definition.
///
/// `clap_mangen` leaves the date empty unless set, so this output is fully
/// deterministic and the sync test can compare it byte for byte. The
/// exit-code table and env-var lines ride along via `AFTER_HELP`, which the
/// renderer keeps as an extra section.
pub fn man_page() -> Vec<u8> {
    let mut page = Vec::new();
    clap_mangen::Man::new(Cli::command())
        .title("OPENSVR")
        .section("1")
        .render(&mut page)
        .expect("the man page renders into memory");
    page
}

/// Renders a completion script for `shell` into memory; the `complete`
/// subcommand streams these same bytes to stdout.
pub fn completion_script(shell: clap_complete::Shell) -> Vec<u8> {
    let mut script = Vec::new();
    let mut command = Cli::command();
    clap_complete::generate(shell, &mut command, "opensvr", &mut script);
    script
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
            &["opensvr", "convert", "song.svp"],
        ] {
            assert!(Cli::try_parse_from(arguments).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn parses_convert_options() {
        let cli = Cli::try_parse_from(["opensvr", "convert", "song.svp", "song.mid"]).unwrap();
        let Command::Convert(args) = cli.command else {
            panic!("expected convert")
        };
        assert_eq!(args.input, PathBuf::from("song.svp"));
        assert_eq!(args.output, PathBuf::from("song.mid"));
    }

    #[test]
    fn generated_artifacts_cover_the_cli_surface() {
        let page = String::from_utf8(man_page()).unwrap();
        for needle in [
            ".TH OPENSVR 1",
            "complete",
            "convert",
            "OPENSVR_VOICE",
            "OPENSVR_DICT",
            "Exit codes",
        ] {
            assert!(page.contains(needle), "the man page misses {needle}");
        }
        assert_eq!(man_page(), man_page(), "man output must be deterministic");
        for shell in [clap_complete::Shell::Bash, clap_complete::Shell::Zsh] {
            let script = String::from_utf8(completion_script(shell)).unwrap();
            for subcommand in ["info", "render", "midi", "convert", "complete", "man"] {
                assert!(
                    script.contains(subcommand),
                    "{shell:?} completions miss {subcommand}"
                );
            }
        }
    }
}
