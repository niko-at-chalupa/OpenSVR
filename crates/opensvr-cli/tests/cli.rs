//! End-to-end tests that run the compiled `opensvr` binary.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/two-tracks.svp");
const NEURAL_BASIC: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/audio/neural-basic.svp");

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
fn render_sings_when_a_voice_is_configured() {
    let Some(voice) = std::env::var("OPENSVR_TEST_VOICE").ok() else {
        return;
    };
    let path = scratch("neural.wav");
    let output = opensvr(&[
        "render",
        NEURAL_BASIC,
        "-o",
        path.to_str().unwrap(),
        "--rate",
        "8000",
        "--voice",
        &voice,
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("placeholder tones"),
        "a voiced render must not print the placeholder warning: {stderr}"
    );

    // Every note carries explicit phonemes, so no --dict is needed. Windows
    // mirror the audio-crate gated test at 8 kHz: the voiced "k a" region at
    // 0.1..0.3 s must dominate the "sil" note at 1.05..1.15 s.
    let mut reader = hound::WavReader::open(&path).unwrap();
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .map(|sample| sample.unwrap() as f32 / 8_388_607.0)
        .collect();
    let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
    assert!(
        (left.len() as f64 - 12_800.0).abs() <= 2.0,
        "frames: got {}, want ~12800",
        left.len()
    );
    let mean_abs = |window: &[f32]| -> f64 {
        window.iter().map(|sample| f64::from(sample.abs())).sum::<f64>()
            / window.len().max(1) as f64
    };
    let voiced = mean_abs(&left[800..2400]);
    let silence = mean_abs(&left[8400..9200]);
    let peak = left.iter().fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(peak > 0.05, "peak {peak}: the phrase must sing audibly");
    assert!(
        voiced > 10.0 * silence,
        "voiced {voiced} vs silence {silence}: vowels must dominate the rest note"
    );
    fs::remove_file(path).unwrap();
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
