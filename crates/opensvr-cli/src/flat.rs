//! Flat-form CLI compatibility shim.
//!
//! OpenSV's CLI takes a flat form — `opensv-cli <project.svp> -o <out.wav>
//! [options]` and `opensv-cli <project.svp> --info` — while `opensvr`
//! requires a subcommand. This module parses the flat grammar (a port of
//! `parseArgs` in `src/cli/Main.cpp`) and routes it to the existing
//! subcommand arguments, so old scripts keep working.

use std::path::PathBuf;

use opensvr_audio::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};

use crate::cli::{Command, InfoArgs, Overrides, RenderArgs};

/// Flat usage text, kept parallel to the C++ `printUsage` contract.
pub const FLAT_USAGE: &str = "\
Usage: opensvr <project.svp> -o <out.wav> [options]
       opensvr <project.svp> --info

Options:
  -o, --output <file>     Output WAV path (24-bit stereo)
      --voice <voice.nofs>  Use this voice database on every track
      --dict <clf-data>   Use this pronunciation dictionary directory on every track
      --language <name>   Override the singing language on every track (e.g. japanese)
      --rate <hz>         Output sample rate (default 48000)
      --info              List tracks, voices and whether their files exist, then exit
  -q, --quiet             Suppress progress messages
  -h, --help              Show this help

OPENSVR_VOICE and OPENSVR_DICT default --voice and --dict when those flags are absent; explicit flags win.
Voice databases and dictionaries are not bundled; point to files you own.
Exit codes: 0 ok, 1 usage error, 2 load error, 3 render/write error, 130 cancelled.";

/// Whether `argv[1]` names a subcommand of the new CLI.
pub fn is_subcommand(name: &str) -> bool {
    matches!(
        name,
        "info" | "render" | "midi" | "convert" | "complete" | "man"
    )
}

/// Whether `argv[1]` is a clap help/version request that must keep the
/// subcommand documentation.
pub fn is_help_version(name: &str) -> bool {
    matches!(name, "--help" | "-h" | "--version" | "-V")
}

/// Outcome of [`parse_flat`]: either a subcommand to run or a help request.
#[derive(Debug, PartialEq, Eq)]
pub enum FlatOutcome {
    Command(Command),
    Help,
}

/// Parses the flat grammar (arguments after the program name) into the same
/// [`Command`] values the subcommands build.
///
/// Error strings reuse the C++ texts for unknown options, missing values,
/// extra positionals, the missing-input case and the missing-`-o` case; the
/// `--rate` text also matches C++ (`--rate must be between 8000 and 192000`),
/// which coincides with the subcommand bounds.
pub fn parse_flat(args: &[String]) -> Result<FlatOutcome, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut voice: Option<PathBuf> = None;
    let mut dict: Option<PathBuf> = None;
    let mut language: Option<opensvr_core::Language> = None;
    let mut rate: u32 = 48_000;
    let mut info_only = false;
    let mut quiet = false;

    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        // Options that take a value: the C++ reader reports the flag as
        // written (`-o` vs `--output`) when the value is missing.
        if arg == "-o"
            || arg == "--output"
            || arg == "--voice"
            || arg == "--dict"
            || arg == "--language"
            || arg == "--rate"
        {
            index += 1;
            let Some(value) = args.get(index) else {
                return Err(format!("missing value for {arg}"));
            };
            match arg {
                "-o" | "--output" => output = Some(value.clone()),
                "--voice" => voice = Some(PathBuf::from(value)),
                "--dict" => dict = Some(PathBuf::from(value)),
                "--language" => match value.parse() {
                    Ok(parsed) => language = Some(parsed),
                    Err(error) => return Err(error.to_string()),
                },
                "--rate" => match value.parse::<u32>() {
                    Ok(parsed)
                        if (MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&parsed)
                            && MAX_SAMPLE_RATE >= 192_000 =>
                    {
                        rate = parsed;
                    }
                    _ => return Err("--rate must be between 8000 and 192000".to_owned()),
                },
                _ => unreachable!("option dispatch covers every valued flag"),
            }
        } else if arg == "-h" || arg == "--help" {
            return Ok(FlatOutcome::Help);
        } else if arg == "--info" {
            info_only = true;
        } else if arg == "-q" || arg == "--quiet" {
            quiet = true;
        } else if arg.starts_with('-') && arg.len() > 1 {
            return Err(format!("unknown option {arg}"));
        } else if input.is_none() {
            input = Some(arg.to_owned());
        } else {
            return Err(format!("unexpected extra argument {arg}"));
        }
        index += 1;
    }

    let Some(input) = input else {
        return Err("no input project given".to_owned());
    };
    let overrides = Overrides {
        voice,
        dict,
        language,
    };
    if info_only {
        Ok(FlatOutcome::Command(Command::Info(InfoArgs {
            project: PathBuf::from(input),
            overrides,
            phonemes: false,
        })))
    } else {
        let Some(output) = output else {
            return Err("no output file given (use -o)".to_owned());
        };
        Ok(FlatOutcome::Command(Command::Render(RenderArgs {
            project: PathBuf::from(input),
            output: PathBuf::from(output),
            overrides,
            rate,
            quiet,
        })))
    }
}

#[cfg(test)]
mod tests {
    use opensvr_core::Language;

    use super::*;

    fn argv(words: &[&str]) -> Vec<String> {
        words.iter().map(ToString::to_string).collect()
    }

    fn render_of(outcome: FlatOutcome) -> RenderArgs {
        let FlatOutcome::Command(Command::Render(args)) = outcome else {
            panic!("expected a render command, got {outcome:?}");
        };
        args
    }

    fn info_of(outcome: FlatOutcome) -> InfoArgs {
        let FlatOutcome::Command(Command::Info(args)) = outcome else {
            panic!("expected an info command, got {outcome:?}");
        };
        args
    }

    #[test]
    fn routes_a_minimal_render() {
        let args = render_of(parse_flat(&argv(&["song.svp", "-o", "out.wav"])).unwrap());
        assert_eq!(args.project, PathBuf::from("song.svp"));
        assert_eq!(args.output, PathBuf::from("out.wav"));
        assert_eq!((args.rate, args.quiet), (48_000, false));
        assert_eq!(args.overrides.voice, None);
    }

    #[test]
    fn routes_every_render_option() {
        let args = render_of(
            parse_flat(&argv(&[
                "song.svp",
                "--output",
                "out.wav",
                "--voice",
                "gumi.nofs",
                "--dict",
                "clf-data",
                "--language",
                "english",
                "--rate",
                "44100",
                "--quiet",
            ]))
            .unwrap(),
        );
        assert_eq!(args.output, PathBuf::from("out.wav"));
        assert_eq!(args.overrides.voice, Some(PathBuf::from("gumi.nofs")));
        assert_eq!(args.overrides.dict, Some(PathBuf::from("clf-data")));
        assert_eq!(args.overrides.language, Some(Language::English));
        assert_eq!((args.rate, args.quiet), (44_100, true));
    }

    #[test]
    fn info_drops_the_output_requirement() {
        let with_output =
            info_of(parse_flat(&argv(&["song.svp", "--info", "-o", "ignored.wav"])).unwrap());
        assert_eq!(with_output.project, PathBuf::from("song.svp"));
        assert!(!with_output.phonemes);

        let plain = info_of(parse_flat(&argv(&["song.svp", "--info"])).unwrap());
        assert_eq!(plain.project, PathBuf::from("song.svp"));
    }

    #[test]
    fn reports_the_cpp_failure_cases() {
        for (words, message) in [
            (&["--info"][..], "no input project given"),
            (&["song.svp"][..], "no output file given (use -o)"),
            (
                &["song.svp", "extra.svp", "-o", "out.wav"][..],
                "unexpected extra argument extra.svp",
            ),
            (
                &["song.svp", "--frobnicate"][..],
                "unknown option --frobnicate",
            ),
            (&["song.svp", "-o"][..], "missing value for -o"),
            (&["song.svp", "--output"][..], "missing value for --output"),
            (&["song.svp", "--voice"][..], "missing value for --voice"),
            (
                &["song.svp", "-o", "out.wav", "--rate", "100"][..],
                "--rate must be between 8000 and 192000",
            ),
            (
                &["song.svp", "-o", "out.wav", "--rate", "999999"][..],
                "--rate must be between 8000 and 192000",
            ),
            (
                &["song.svp", "-o", "out.wav", "--rate", "fast"][..],
                "--rate must be between 8000 and 192000",
            ),
        ] {
            assert_eq!(
                parse_flat(&argv(words)),
                Err(message.to_owned()),
                "{words:?}"
            );
        }
    }

    #[test]
    fn rejects_unknown_languages() {
        assert!(parse_flat(&argv(&["song.svp", "--info", "--language", "klingon"])).is_err());
    }

    #[test]
    fn help_wins_over_missing_arguments() {
        assert_eq!(parse_flat(&argv(&["--help"])), Ok(FlatOutcome::Help));
        assert_eq!(parse_flat(&argv(&["-h"])), Ok(FlatOutcome::Help));
    }
}
