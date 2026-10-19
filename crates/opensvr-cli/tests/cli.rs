//! End-to-end tests that run the compiled `opensvr` binary.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/two-tracks.svp");
const NEURAL_BASIC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/golden/audio/neural-basic.svp"
);

fn opensvr(arguments: &[&str]) -> Output {
    // Hermetic by default: ambient OPENSVR_VOICE/OPENSVR_DICTIONARY must not leak
    // into assertions about project-file paths (use `opensvr_env` to set them).
    opensvr_env(arguments, &[])
}

fn opensvr_env(arguments: &[&str], vars: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_opensvr"));
    command
        .args(arguments)
        .env_remove("OPENSVR_VOICE")
        .env_remove("OPENSVR_DICTIONARY")
        .env_remove("OPENSVR_DICT");
    for (key, value) in vars {
        command.env(key, value);
    }
    command.output().expect("the binary runs")
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
fn the_cli_and_the_library_render_the_same_samples() {
    // The CLI is now a shell around `opensvr`, so the two entry points must
    // agree byte for byte. The example has no voice on disk, which keeps this
    // runnable without proprietary data.
    let path = scratch("facade.wav");
    let output = opensvr(&[
        "render",
        EXAMPLE,
        "-o",
        path.to_str().unwrap(),
        "--rate",
        "22050",
        "-q",
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let from_cli = fs::read(&path).unwrap();
    fs::remove_file(path).unwrap();

    let svp = fs::read_to_string(EXAMPLE).unwrap();
    let options = opensvr::Options {
        sample_rate: 22_050,
        ..opensvr::Options::default()
    };
    let cancel = opensvr::CancelToken::new();
    let audio = opensvr::render_svp(&svp, &options, &cancel).unwrap();
    let from_library = opensvr::encode_wav(&audio, options.sample_rate, &cancel).unwrap();
    assert_eq!(from_cli, from_library);
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

    // Every note carries explicit phonemes, so no --dictionary is needed. Windows
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
        window
            .iter()
            .map(|sample| f64::from(sample.abs()))
            .sum::<f64>()
            / window.len().max(1) as f64
    };
    let voiced = mean_abs(&left[800..2400]);
    let silence = mean_abs(&left[8400..9200]);
    let peak = left
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
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
fn the_progress_bar_changes_no_sample() {
    // The bar draws to a terminal only (hidden when piped), but the render
    // behind it must still be bitwise identical to the quiet one.
    let loud = scratch("loud.wav");
    let quiet = scratch("progress-quiet.wav");
    let loud_output = opensvr(&["render", EXAMPLE, "-o", loud.to_str().unwrap()]);
    assert_eq!(loud_output.status.code(), Some(0));
    let quiet_output = opensvr(&["render", EXAMPLE, "-o", quiet.to_str().unwrap(), "-q"]);
    assert_eq!(quiet_output.status.code(), Some(0));
    assert_eq!(fs::read(&loud).unwrap(), fs::read(&quiet).unwrap());
    let stderr = String::from_utf8_lossy(&loud_output.stderr);
    assert!(stderr.contains("Rendering") && stderr.contains("Wrote"));
    fs::remove_file(loud).unwrap();
    fs::remove_file(quiet).unwrap();
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
fn convert_svp_to_mid_writes_a_standard_midi_file() {
    let path = scratch("convert.mid");
    let output = opensvr(&["convert", EXAMPLE, "-o", path.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(0));
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..4], b"MThd");
    assert_eq!(
        &bytes[10..12],
        [0, 3],
        "conductor track plus two project tracks"
    );
    // The dedicated exporter and convert agree byte for byte.
    let direct = scratch("convert-direct.mid");
    let direct_output = opensvr(&["midi", EXAMPLE, "-o", direct.to_str().unwrap()]);
    assert_eq!(direct_output.status.code(), Some(0));
    assert_eq!(fs::read(&direct).unwrap(), bytes);
    fs::remove_file(path).unwrap();
    fs::remove_file(direct).unwrap();
}

#[test]
fn convert_mid_to_svp_keeps_the_notes() {
    // Same file stems throughout, so the `.mid` project names coincide.
    let here = scratch("keep-a");
    let there = scratch("keep-b");
    std::fs::create_dir_all(&here).unwrap();
    std::fs::create_dir_all(&there).unwrap();
    let (mid_a, svp_a) = (here.join("song.mid"), here.join("song.svp"));
    let (mid_b, svp_b) = (there.join("song.mid"), there.join("song.svp"));

    assert_eq!(
        opensvr(&["convert", EXAMPLE, "-o", mid_a.to_str().unwrap()])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            mid_a.to_str().unwrap(),
            "-o",
            svp_a.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    // The sung notes survive the MIDI leg: Lead keeps its 4 lyric-bearing
    // notes. (Harmony flattens from 0 to 2 notes because MIDI has no library
    // groups: the exporter writes its placed library notes into the track,
    // and the importer cannot tell them apart. The timeline is unchanged.)
    let original = opensvr(&["info", EXAMPLE]);
    let after = opensvr(&["info", svp_a.to_str().unwrap()]);
    assert_eq!(original.status.code(), Some(0));
    assert_eq!(after.status.code(), Some(0));
    let after = String::from_utf8(after.stdout).unwrap();
    assert!(
        after.contains("[0] Lead  notes=4  language=japanese"),
        "{after}"
    );
    assert_eq!(after.matches("Tracks: 2").count(), 1);
    // And `.mid` -> `.svp` is a fixed point from here on.
    assert_eq!(
        opensvr(&[
            "convert",
            svp_a.to_str().unwrap(),
            "-o",
            mid_b.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            mid_b.to_str().unwrap(),
            "-o",
            svp_b.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    let again = opensvr(&["info", svp_b.to_str().unwrap()]);
    assert_eq!(again.status.code(), Some(0));
    assert_eq!(String::from_utf8(again.stdout).unwrap(), after);
    std::fs::remove_dir_all(here).unwrap();
    std::fs::remove_dir_all(there).unwrap();
}

#[test]
fn convert_svp_to_svp_converges_to_a_fixed_point() {
    // The first save appends default keys the input lacks (e.g.
    // `renderConfig`), which the second save then keeps in place: like the
    // C++ retained-object writer, `save` reaches its fixed point on the
    // second pass, not the first.
    let once = scratch("fixed-once.svp");
    let twice = scratch("fixed-twice.svp");
    let thrice = scratch("fixed-thrice.svp");
    assert_eq!(
        opensvr(&["convert", EXAMPLE, "-o", once.to_str().unwrap()])
            .status
            .code(),
        Some(0)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            once.to_str().unwrap(),
            "-o",
            twice.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            twice.to_str().unwrap(),
            "-o",
            thrice.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    assert_eq!(fs::read(&twice).unwrap(), fs::read(&thrice).unwrap());
    let first = opensvr(&["info", once.to_str().unwrap()]);
    let second = opensvr(&["info", twice.to_str().unwrap()]);
    assert_eq!(first.status.code(), Some(0));
    assert_eq!(second.status.code(), Some(0));
    // The `Project:` line follows the file stem through conversions; the
    // tracks below it must be identical.
    let without_name = |output: std::process::Output| {
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .skip(1)
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(without_name(second), without_name(first));
    fs::remove_file(once).unwrap();
    fs::remove_file(twice).unwrap();
    fs::remove_file(thrice).unwrap();
}

#[test]
fn convert_uses_the_documented_exit_codes() {
    let mid = scratch("exit-codes.mid");
    assert_eq!(
        opensvr(&["convert", EXAMPLE, "-o", mid.to_str().unwrap()])
            .status
            .code(),
        Some(0)
    );
    // Unknown extensions are refused, not misread.
    assert_eq!(
        opensvr(&[
            "convert",
            EXAMPLE,
            "-o",
            scratch("out.txt").to_str().unwrap()
        ])
        .status
        .code(),
        Some(3)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            scratch("in.txt").to_str().unwrap(),
            "-o",
            mid.to_str().unwrap()
        ])
        .status
        .code(),
        Some(3)
    );
    // Missing inputs are load errors.
    assert_eq!(
        opensvr(&[
            "convert",
            "/no/such/project.svp",
            "-o",
            mid.to_str().unwrap()
        ])
        .status
        .code(),
        Some(2)
    );
    assert_eq!(
        opensvr(&[
            "convert",
            "/no/such/project.mid",
            "-o",
            mid.to_str().unwrap()
        ])
        .status
        .code(),
        Some(2)
    );
    fs::remove_file(mid).unwrap();
}

#[test]
fn missing_outputs_pair_with_the_input() {
    // A copied project keeps its default output out of the repository.
    let dir = scratch("defaults");
    fs::create_dir_all(&dir).unwrap();
    let project = dir.join("two-tracks.svp");
    fs::copy(EXAMPLE, &project).unwrap();

    let wav = project.with_extension("wav");
    let render = opensvr(&["render", project.to_str().unwrap(), "-q"]);
    assert_eq!(render.status.code(), Some(0));
    assert_eq!(&fs::read(&wav).unwrap()[..4], b"RIFF");

    let mid = project.with_extension("mid");
    let midi = opensvr(&["midi", project.to_str().unwrap()]);
    assert_eq!(midi.status.code(), Some(0));
    assert_eq!(&fs::read(&mid).unwrap()[..4], b"MThd");

    // `convert` picks the other format beside the input.
    let converted = dir.join("converted.mid");
    let convert = opensvr(&[
        "convert",
        project.to_str().unwrap(),
        "-o",
        converted.to_str().unwrap(),
    ]);
    assert_eq!(convert.status.code(), Some(0));
    assert_eq!(fs::read(&mid).unwrap(), fs::read(&converted).unwrap());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn output_dash_streams_to_stdout() {
    let rendered = opensvr(&["render", EXAMPLE, "-o", "-", "--rate", "8000", "-q"]);
    assert_eq!(rendered.status.code(), Some(0));
    assert_eq!(&rendered.stdout[..4], b"RIFF");
    assert_eq!(&rendered.stdout[8..12], b"WAVE");

    let midi = opensvr(&["midi", EXAMPLE, "-o", "-"]);
    assert_eq!(midi.status.code(), Some(0));
    assert_eq!(&midi.stdout[..4], b"MThd");

    // `-` has no extension, so it streams the other format: `.svp` -> MIDI
    // bytes. That matches omitting `-o` entirely.
    let svp_to_midi = opensvr(&["convert", EXAMPLE, "-o", "-"]);
    assert_eq!(svp_to_midi.status.code(), Some(0));
    assert_eq!(&svp_to_midi.stdout[..4], b"MThd");

    // A `.mid` input streams `.svp`, byte-identical to what `save` writes:
    // relative paths anchor on the input's directory in both cases.
    let dir = scratch("stdout-svp");
    fs::create_dir_all(&dir).unwrap();
    let project = dir.join("two-tracks.svp");
    fs::copy(EXAMPLE, &project).unwrap();
    let mid = dir.join("song.mid");
    assert_eq!(
        opensvr(&[
            "convert",
            project.to_str().unwrap(),
            "-o",
            mid.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    let file = dir.join("out.svp");
    assert_eq!(
        opensvr(&[
            "convert",
            mid.to_str().unwrap(),
            "-o",
            file.to_str().unwrap()
        ])
        .status
        .code(),
        Some(0)
    );
    let svp = opensvr(&["convert", mid.to_str().unwrap(), "-o", "-"]);
    assert_eq!(svp.status.code(), Some(0));
    assert_eq!(svp.stdout, fs::read(&file).unwrap());
    assert_eq!(svp.stdout.last(), Some(&0));
    fs::remove_dir_all(dir).unwrap();
}

/// Renders an override path exactly as the binary prints it: overrides go
/// through `std::path::absolute()`, which on Windows yields `C:\...` with
/// backslashes, so tests must not hardcode Unix separators.
fn abs_display(path: &str) -> String {
    std::path::absolute(path).unwrap().display().to_string()
}

#[test]
fn env_vars_default_the_overrides_on_every_track() {
    let vars = &[
        ("OPENSVR_VOICE", "/env/voice.nofs"),
        ("OPENSVR_DICTIONARY", "/env/dicts"),
    ];
    let output = opensvr_env(&["info", EXAMPLE], vars);
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    // Both tracks pick the env paths up; neither exists, so both are flagged.
    // Overrides are absolutized before display (`C:\...` on Windows).
    let missing_voice = format!("{}  [MISSING]", abs_display("/env/voice.nofs"));
    let missing_dict = format!("{}  [MISSING]", abs_display("/env/dicts"));
    assert_eq!(text.matches(&missing_voice).count(), 2, "{text}");
    assert_eq!(text.matches(&missing_dict).count(), 2, "{text}");
    // The project file's own voice path is gone, overridden on every track.
    assert!(!text.contains("voices/example/voice.nofs"), "{text}");
}

#[test]
fn the_legacy_dict_env_var_still_works() {
    let output = opensvr_env(&["info", EXAMPLE], &[("OPENSVR_DICT", "/legacy/dicts")]);
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    let missing = format!("{}  [MISSING]", abs_display("/legacy/dicts"));
    assert_eq!(text.matches(&missing).count(), 2, "{text}");
}

#[test]
fn explicit_flags_beat_env_vars() {
    let output = opensvr_env(
        &["info", EXAMPLE, "--voice", "/flag/voice.nofs"],
        &[("OPENSVR_VOICE", "/env/voice.nofs")],
    );
    assert_eq!(output.status.code(), Some(0));
    let text = String::from_utf8(output.stdout).unwrap();
    let missing_flag = format!("{}  [MISSING]", abs_display("/flag/voice.nofs"));
    assert_eq!(text.matches(&missing_flag).count(), 2, "{text}");
    assert!(!text.contains(&abs_display("/env/voice.nofs")), "{text}");
}

#[test]
fn empty_env_values_count_as_unset() {
    let output = opensvr_env(
        &["info", EXAMPLE],
        &[("OPENSVR_VOICE", ""), ("OPENSVR_DICT", "")],
    );
    assert_eq!(output.status.code(), Some(0));
    let plain = opensvr(&["info", EXAMPLE]);
    assert_eq!(output.stdout, plain.stdout);
}

#[test]
fn man_matches_the_checked_in_page() {
    // `opensvr.1` lives beside the crate that owns the CLI definition; the
    // binary must print exactly those bytes so distro packaging can redirect
    // them (`opensvr man > opensvr.1` reproduces the file).
    let output = opensvr(&["man"]);
    assert_eq!(output.status.code(), Some(0));
    let checked_in = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/opensvr.1")).unwrap();
    assert_eq!(output.stdout, checked_in);
}

#[test]
fn complete_prints_scripts_naming_every_subcommand() {
    for shell in ["bash", "elvish", "fish", "powershell", "zsh"] {
        let output = opensvr(&["complete", shell]);
        assert_eq!(output.status.code(), Some(0), "{shell}");
        let text = String::from_utf8(output.stdout).unwrap();
        for subcommand in ["info", "render", "midi", "convert", "complete", "man"] {
            assert!(
                text.contains(subcommand),
                "{shell} script misses {subcommand}"
            );
        }
    }
}

#[test]
fn render_help_documents_the_models_dir_override() {
    let help = opensvr(&["render", "--help"]);
    assert_eq!(help.status.code(), Some(0));
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("--models-dir"), "{text}");
    assert!(text.contains("OPENSVR_MODELS_DIR"), "{text}");
}

#[test]
fn failures_use_the_documented_exit_codes() {
    assert_eq!(opensvr(&[]).status.code(), Some(1), "usage error");
    assert_eq!(
        opensvr(&["render"]).status.code(),
        Some(1),
        "missing project"
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
