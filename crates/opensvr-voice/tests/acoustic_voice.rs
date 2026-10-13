//! Gated tests for the real acoustic model in a local `.nofs` voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice in the
//! OpenSV references) to run; the tests return early when unset so CI stays
//! green without proprietary data. Voice files are never committed: only the
//! small text goldens under `tests/golden/voice/` are, dumped from the C++
//! engine's `AcousticModel` over the timing/pitch stages' real outputs (see
//! the throwaway dumper kept out of tree per `AGENTS.md`).
//!
//! Each golden chains all three stages: the test re-predicts durations with
//! the Rust timing model and the F0 curve with the Rust pitch model (proving
//! the inputs are genuine previous-stage outputs), then predicts acoustic
//! frames from the golden (exact C++) durations and pitch and compares all 72
//! channels per frame at `1e-5` relative tolerance. Driving the acoustic
//! comparison from the golden stage outputs keeps each stage's tolerance
//! from stacking into the next: the Rust timing and pitch predictions are
//! asserted against the same goldens in the same test. The chosen phrases
//! quantize to exactly the pitch frame count (the pitch grid rounds
//! boundaries while the acoustic grid truncates; the renderer resamples in
//! general, but these goldens need no resampling).

use std::path::{Path, PathBuf};

use opensvr_core::CancelToken;
use opensvr_dnni::DnniReader;
use opensvr_nofs::{VoiceConfiguration, VoiceDatabase};
use opensvr_voice::{
    AcousticModel, AcousticModelState, PhonemeDuration, PhonemeTiming, PitchModel, PitchNote,
    TimedPhoneme, midi_to_log_f0,
};

/// Relative tolerance for durations, pitch and acoustic frames (Phase 5 notes).
const RELATIVE_TOLERANCE: f64 = 1e-5;
/// Acoustic frame grid in seconds (both grids run at 5 ms here).
const FRAME_INTERVAL_SECONDS: f32 = 0.005;
/// Deterministic noise seed used by the dumper (the C++ default).
const NOISE_SEED: u32 = 5489;

struct Golden {
    syllables: Vec<opensvr_voice::TimingSyllable>,
    durations: Vec<PhonemeDuration>,
    pitch: Vec<f32>,
    acoustic_frames: usize,
    acoustic_channels: usize,
    acoustic: Vec<f32>,
}

fn parse_golden(path: &Path) -> Golden {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("golden {} must read: {error}", path.display()));
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .peekable();
    let mut golden = Golden {
        syllables: Vec::new(),
        durations: Vec::new(),
        pitch: Vec::new(),
        acoustic_frames: 0,
        acoustic_channels: 0,
        acoustic: Vec::new(),
    };
    while let Some(line) = lines.next() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("in") => {
                let language = words.next().expect("syllable language").to_owned();
                let midi_pitch: i32 = words.next().expect("midi").parse().expect("midi int");
                let duration_seconds: f64 =
                    words.next().expect("seconds").parse().expect("seconds float");
                let phonemes: Vec<String> = words.map(str::to_owned).collect();
                assert!(!phonemes.is_empty(), "golden syllable needs phonemes");
                golden.syllables.push(opensvr_voice::TimingSyllable {
                    language,
                    phonemes,
                    duration_seconds,
                    midi_pitch,
                    is_continuation: false,
                });
            }
            Some("durations") => {
                let count: usize = words.next().expect("count").parse().expect("count int");
                for _ in 0..count {
                    let entry = lines.next().expect("duration entry");
                    let mut fields = entry.split_whitespace();
                    golden.durations.push(PhonemeDuration {
                        language: fields.next().expect("language").to_owned(),
                        symbol: fields.next().expect("symbol").to_owned(),
                        syllable_index: fields
                            .next()
                            .expect("syllable index")
                            .parse()
                            .expect("syllable int"),
                        timing_interval_index: fields
                            .next()
                            .expect("interval index")
                            .parse()
                            .expect("interval int"),
                        duration_seconds: fields
                            .next()
                            .expect("seconds")
                            .parse()
                            .expect("seconds float"),
                    });
                }
            }
            Some("pitch") => {
                let count: usize = words.next().expect("count").parse().expect("count int");
                for _ in 0..count {
                    golden
                        .pitch
                        .push(lines.next().expect("pitch entry").parse().expect("pitch float"));
                }
            }
            Some("acoustic") => {
                golden.acoustic_frames =
                    words.next().expect("frames").parse().expect("frames int");
                golden.acoustic_channels =
                    words.next().expect("channels").parse().expect("channels int");
                for _ in 0..golden.acoustic_frames {
                    let entry = lines.next().expect("acoustic frame");
                    for value in entry.split_whitespace() {
                        golden
                            .acoustic
                            .push(value.parse().expect("acoustic float"));
                    }
                }
            }
            Some(other) => panic!("golden {} has unknown section {other}", path.display()),
            None => {}
        }
    }
    assert!(!golden.syllables.is_empty(), "golden needs syllables");
    assert!(!golden.durations.is_empty(), "golden needs durations");
    assert!(!golden.pitch.is_empty(), "golden needs pitch");
    assert_eq!(
        golden.acoustic.len(),
        golden.acoustic_frames * golden.acoustic_channels,
        "acoustic values must fill the frame grid"
    );
    golden
}

fn voice_path() -> Option<PathBuf> {
    std::env::var("OPENSVR_TEST_VOICE").ok().map(PathBuf::from)
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/voice")
}

fn golden_cases() -> Vec<PathBuf> {
    let mut cases: Vec<PathBuf> = std::fs::read_dir(golden_dir())
        .expect("golden dir must list")
        .map(|entry| entry.expect("golden entry").path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("acoustic-"))
        })
        .collect();
    cases.sort();
    cases
}

fn check_close(got: f64, want: f64, what: &str) {
    let allowed = RELATIVE_TOLERANCE * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= allowed,
        "{what}: got {got}, want {want}"
    );
}

#[test]
fn real_acoustic_model_matches_cpp_goldens() {
    let Some(voice) = voice_path() else {
        return;
    };
    let db = VoiceDatabase::open(voice.as_path()).expect("voice must open");
    let config = VoiceConfiguration::load(&db).expect("config must load");

    let duration_key = config.duration_model().key.clone();
    let duration_entry = db
        .entries()
        .iter()
        .find(|entry| entry.key == duration_key)
        .expect("voice must contain the duration entry");
    let duration_bytes = db
        .read_entry(duration_entry)
        .expect("entry must read")
        .to_vec();
    let duration_reader =
        DnniReader::from_bytes(duration_bytes).expect("duration entry must parse");
    let mut timing = PhonemeTiming::new();
    timing
        .load(&duration_reader, 0)
        .expect("duration model must load");

    let pitch_entry = db
        .find_entry("f0model-dds")
        .expect("voice must contain the pitch entry");
    let pitch_bytes = db.read_entry(pitch_entry).expect("entry must read").to_vec();
    let pitch_reader = DnniReader::from_bytes(pitch_bytes).expect("pitch entry must parse");
    let mut pitch = PitchModel::new();
    pitch
        .load(&pitch_reader, 0)
        .expect("pitch model must load");

    let acoustic_key = config.acoustic_model().key.clone();
    assert_eq!(config.acoustic_model().architecture, "dds");
    let acoustic_entry = db
        .entries()
        .iter()
        .find(|entry| entry.key == acoustic_key)
        .expect("voice must contain the acoustic entry");
    let acoustic_bytes = db
        .read_entry(acoustic_entry)
        .expect("entry must read")
        .to_vec();
    let acoustic_reader =
        DnniReader::from_bytes(acoustic_bytes).expect("acoustic entry must parse");
    let styles: Vec<String> = db.metadata().timbre_styles.clone();
    let mut acoustic = AcousticModel::new();
    acoustic
        .load_with_vocal_modes(&acoustic_reader, &styles, 0)
        .expect("acoustic model must load");
    assert_eq!(acoustic.frame_interval_seconds(), FRAME_INTERVAL_SECONDS);
    assert_eq!(acoustic.vocal_mode_count(), styles.len());

    let cases = golden_cases();
    assert!(!cases.is_empty(), "expected acoustic goldens");
    for case in &cases {
        check_case(&timing, &pitch, &acoustic, case);
    }
}

fn check_case(
    timing: &PhonemeTiming,
    pitch: &PitchModel,
    acoustic: &AcousticModel,
    case: &Path,
) {
    let golden = parse_golden(case);

    let mut durations = Vec::new();
    timing
        .predict(&golden.syllables, &mut durations)
        .unwrap_or_else(|error| panic!("{} timing must predict: {error}", case.display()));
    assert_eq!(
        durations.len(),
        golden.durations.len(),
        "{} phoneme count",
        case.display()
    );
    for (index, (got, want)) in durations.iter().zip(&golden.durations).enumerate() {
        assert_eq!(got.language, want.language, "{} phoneme {index} language", case.display());
        assert_eq!(got.symbol, want.symbol, "{} phoneme {index} symbol", case.display());
        check_close(
            got.duration_seconds,
            want.duration_seconds,
            &format!("{} phoneme {index} duration", case.display()),
        );
    }

    let notes: Vec<PitchNote> = golden
        .syllables
        .iter()
        .map(|syllable| PitchNote {
            syllable: syllable.clone(),
            ..PitchNote::default()
        })
        .collect();
    let cancel = CancelToken::new();
    let mut midi_pitch = Vec::new();
    pitch
        .run(&notes, &durations, &mut midi_pitch, 0, None, &cancel, None, &[])
        .unwrap_or_else(|error| panic!("{} pitch must predict: {error}", case.display()));
    assert_eq!(
        midi_pitch.len(),
        golden.pitch.len(),
        "{} pitch frame count",
        case.display()
    );
    for (index, (&got, &want)) in midi_pitch.iter().zip(&golden.pitch).enumerate() {
        check_close(
            f64::from(got),
            f64::from(want),
            &format!("{} pitch frame {index}", case.display()),
        );
    }

    let mut timed: Vec<TimedPhoneme> = Vec::new();
    opensvr_voice::quantize_phoneme_durations(
        &golden.durations,
        FRAME_INTERVAL_SECONDS,
        &mut timed,
    )
    .unwrap_or_else(|error| panic!("{} must quantize: {error}", case.display()));
    // The chosen phrases quantize to exactly the pitch frame count, so no
    // renderer-style resampling stands between the stages here.
    assert_eq!(
        timed.iter().map(|phoneme| phoneme.frame_count).sum::<usize>(),
        golden.pitch.len(),
        "{} quantized frames match pitch frames",
        case.display()
    );
    // The acoustic comparison runs from the golden (exact C++) stage outputs
    // so stage tolerances cannot stack; the Rust predictions above are
    // asserted against the same goldens.
    let log_f0: Vec<f32> = golden.pitch.iter().map(|pitch| midi_to_log_f0(*pitch)).collect();

    let mut frames = opensvr_dnni::DnniTensor::default();
    acoustic
        .run(&timed, &log_f0, &mut frames, NOISE_SEED, None, &cancel, None, &[])
        .unwrap_or_else(|error| panic!("{} acoustic must predict: {error}", case.display()));
    assert_eq!(frames.frames, golden.acoustic_frames, "{} frame count", case.display());
    assert_eq!(
        frames.channels, golden.acoustic_channels,
        "{} channel count",
        case.display()
    );
    assert_eq!(frames.channels, 72, "{} vocoder-ready width", case.display());
    for (index, (&got, &want)) in frames.values.iter().zip(&golden.acoustic).enumerate() {
        check_close(
            f64::from(got),
            f64::from(want),
            &format!("{} value {index}", case.display()),
        );
    }

    // A cached rerun through the phrase state must reproduce the frames.
    let mut state = AcousticModelState::default();
    let mut rerun = opensvr_dnni::DnniTensor::default();
    acoustic
        .run(&timed, &log_f0, &mut rerun, NOISE_SEED, Some(&mut state), &cancel, None, &[])
        .expect("stateful run must predict");
    assert!(state.get_bytes() > 0, "phrase caches are retained");
    let mut cached = opensvr_dnni::DnniTensor::default();
    acoustic
        .run(&timed, &log_f0, &mut cached, NOISE_SEED, Some(&mut state), &cancel, None, &[])
        .expect("cached run must predict");
    assert_eq!(cached.values, rerun.values, "cached frames are verbatim");
    assert_eq!(rerun.values, frames.values, "stateful run matches");
    println!(
        "{}: {} frames x {} channels ok",
        case.file_name().expect("name").to_string_lossy(),
        frames.frames,
        frames.channels
    );
}
