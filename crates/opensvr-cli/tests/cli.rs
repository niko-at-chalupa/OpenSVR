//! End-to-end tests that run the compiled `opensvr` binary.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/two-tracks.svp");

fn opensvr(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_opensvr"))
        .args(arguments)
        .output()
        .expect("the binary runs")
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("opensvr-cli-{}-{name}", std::process::id()))
}

#[test]
fn info_lists_tracks_and_flags_missing_voices() {
    let output = opensvr(&["info", EXAMPLE]);
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Project: two-tracks\nTracks: 2\n"));
    assert!(text.contains("[0] Lead  notes=4  language=japanese"));
    assert!(text.contains("voices/example/voice.nofs  [MISSING]"));
    assert!(text.contains("[1] Harmony  notes=0"));
    assert!(text.contains("voice: (none)"));
}

#[test]
fn overrides_apply_to_every_track() {
    let output = opensvr(&["info", EXAMPLE, "--language", "english"]);
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.matches("language=english").count(), 2);
}

#[test]
fn render_writes_a_wav_file() {
    let path = scratch("render.wav");
    let output = opensvr(&[
        "render",
        EXAMPLE,
        "-o",
        path.to_str().unwrap(),
        "--rate",
        "22050",
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    // 4 quarters of Lead and a Harmony motif ending at beat 7: 7 * 0.5 s + 0.1 s of release at 22,050 Hz.
    let reader = hound::WavReader::open(&path).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 24);
    assert_eq!(reader.duration(), 79_380);
    fs::remove_file(path).unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Rendering") && stderr.contains("placeholder tones"));
}

#[test]
fn quiet_hides_progress_but_not_the_placeholder_warning() {
    let path = scratch("quiet.wav");
    let output = opensvr(&["render", EXAMPLE, "-o", path.to_str().unwrap(), "-q"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Rendering") && !stderr.contains("Wrote"));
    assert!(stderr.contains("placeholder tones"));
    fs::remove_file(path).unwrap();
}

#[test]
fn midi_export_writes_a_standard_midi_file() {
    let path = scratch("export.mid");
    let output = opensvr(&["midi", EXAMPLE, "-o", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(0));
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"MThd");
    assert_eq!(
        &bytes[10..12],
        [0, 3],
        "conductor track plus two project tracks"
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn failures_use_the_documented_exit_codes() {
    assert_eq!(opensvr(&[]).status.code(), Some(1), "usage error");
    assert_eq!(
        opensvr(&["render", EXAMPLE]).status.code(),
        Some(1),
        "missing --output"
    );
    assert_eq!(
        opensvr(&["info", "/no/such/project.svp"]).status.code(),
        Some(2),
        "load error"
    );
    assert_eq!(opensvr(&["--help"]).status.code(), Some(0));

    let unwritable = scratch("missing-dir/out.wav");
    let output = opensvr(&["render", EXAMPLE, "-o", unwritable.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(3), "write error");
}
