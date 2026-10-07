//! `opensvr`: headless tools for Synthesizer V projects, a minimal Rust port of OpenSV's CLI.

mod cli;
mod commands;

use std::process::ExitCode;

use clap::Parser;
use opensvr_audio::CancelToken;

use crate::cli::Cli;

fn main() -> ExitCode {
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

    let cancel = CancelToken::new();
    let handler_token = cancel.clone();
    if let Err(error) = ctrlc::set_handler(move || handler_token.cancel()) {
        eprintln!("opensvr: warning: Ctrl-C will not cancel cleanly: {error}");
    }

    match commands::run(cli.command, &cancel) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("opensvr: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
