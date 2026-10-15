//! `opensvr`: headless tools for Synthesizer V projects, a minimal Rust port of OpenSV's CLI.

mod cli;
mod commands;
mod flat;

use std::process::ExitCode;

use clap::Parser;
use opensvr_audio::CancelToken;

use crate::cli::Cli;

fn main() -> ExitCode {
    // Flat-form compatibility shim: when the first argument is not a
    // subcommand name (and is not a clap `--help`/`--version` request),
    // parse OpenSV's flat grammar and route it to the same subcommands.
    let argv: Vec<String> = std::env::args().collect();
    let first = argv.get(1).map(String::as_str);
    let flat = match first {
        None => true,
        Some(name) => !flat::is_subcommand(name) && !flat::is_help_version(name),
    };
    if flat {
        return match flat::parse_flat(&argv[1..]) {
            Ok(flat::FlatOutcome::Help) => {
                println!("{}", flat::FLAT_USAGE);
                ExitCode::SUCCESS
            }
            Ok(flat::FlatOutcome::Command(command)) => run(command),
            Err(message) => {
                eprintln!("opensvr: {message}\n");
                eprintln!("{}", flat::FLAT_USAGE);
                ExitCode::from(1)
            }
        };
    }

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            // Printing can only fail if the stream is closed; there is nothing useful left to do then.
            let _ = error.print();
            // `--help` and `--version` are not errors; real usage mistakes exit with 1, not clap's 2.
            return if error.use_stderr() {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            };
        }
    };

    run(cli.command)
}

/// Runs one routed subcommand with Ctrl-C cancellation.
fn run(command: crate::cli::Command) -> ExitCode {
    let cancel = CancelToken::new();
    let handler_token = cancel.clone();
    if let Err(error) = ctrlc::set_handler(move || handler_token.cancel()) {
        eprintln!("opensvr: warning: Ctrl-C will not cancel cleanly: {error}");
    }

    match commands::run(command, &cancel) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("opensvr: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
