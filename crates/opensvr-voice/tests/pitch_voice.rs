//! Gated tests for the real pitch model in a local `.nofs` voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice in the
//! OpenSV references) to run; the tests return early when unset so CI stays
//! green without proprietary data. Voice files are never committed: only the
//! small text goldens under `tests/golden/voice/` are, dumped from the C++
//! engine's `PitchModel` over the timing stage's real predicted durations
//! (see the throwaway dumper kept out of tree per `AGENTS.md`).
//!
//! Each golden chains both stages: the test re-predicts durations with the
//! Rust timing model (proving the inputs are genuine previous-stage outputs),
//! then predicts the pitch curve with the Rust pitch model and compares
//! per-frame absolute MIDI pitch at `1e-5` relative tolerance.

use std::path::{Path, PathBuf};

use opensvr_core::CancelToken;
use opensvr_dnni::DnniReader;
use opensvr_nofs::{VoiceConfiguration, VoiceDatabase};
use opensvr_voice::{PhonemeDuration, PhonemeTiming, PitchModel, PitchModelState, PitchNote};

/// Relative tolerance for predicted durations and pitch (ROADMAP Phase 5 notes).
const RELATIVE_TOLERANCE: f64 = 1e-5;

struct Golden {
    syllables: Vec<opensvr_voice::TimingSyllable>,
    durations: Vec<PhonemeDuration>,
    pitch: Vec<f32>,
}

fn parse_syllable(words: &mut std::str::SplitWhitespace<'_>) -> opensvr_voice::TimingSyllable {
    let language = words.next().expect("syllable language").to_owned();
    let midi_pitch: i32 = words.next().expect("midi").parse().expect("midi int");
    let duration_seconds: f64 = words
        .next()
        .expect("seconds")
        .parse()
        .expect("seconds float");
    let phonemes: Vec<String> = words.map(str::to_owned).collect();
    assert!(!phonemes.is_empty(), "golden syllable needs phonemes");
    opensvr_voice::TimingSyllable {
        language,
        phonemes,
        duration_seconds,
        midi_pitch,
        is_continuation: false,
    }
}

fn parse_duration(entry: &str) -> PhonemeDuration {
    let mut fields = entry.split_whitespace();
    PhonemeDuration {
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
    }
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
    };
    while let Some(line) = lines.next() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("in") => {
                golden.syllables.push(parse_syllable(&mut words));
            }
            Some("durations") => {
                let count: usize = words.next().expect("count").parse().expect("count int");
                for _ in 0..count {
                    golden
                        .durations
                        .push(parse_duration(lines.next().expect("duration entry")));
                }
            }
            Some("pitch") => {
                let count: usize = words.next().expect("count").parse().expect("count int");
                for _ in 0..count {
                    golden.pitch.push(
                        lines
                            .next()
                            .expect("pitch entry")
                            .parse()
                            .expect("pitch float"),
                    );
                }
            }
            Some(other) => panic!("golden {} has unknown section {other}", path.display()),
            None => {}
        }
    }
    assert!(!golden.syllables.is_empty(), "golden needs syllables");
    assert!(!golden.durations.is_empty(), "golden needs durations");
    assert!(!golden.pitch.is_empty(), "golden needs pitch");
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
                .is_some_and(|name| name.to_string_lossy().starts_with("pitch-"))
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
fn real_pitch_model_matches_cpp_goldens() {
    let Some(voice) = voice_path() else {
        return;
    };
    let db = VoiceDatabase::open(voice.as_path()).expect("voice must open");

    // Timing stage first: the golden durations must be genuine
    // previous-stage outputs, reproduced here with the Rust timing model.
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

    // Pitch stage: the `f0model-dds` entry loads outside the voice config.
    let pitch_entry = db
        .find_entry("f0model-dds")
        .expect("voice must contain the pitch entry");
    let pitch_bytes = db
        .read_entry(pitch_entry)
        .expect("entry must read")
        .to_vec();
    let pitch_reader = DnniReader::from_bytes(pitch_bytes).expect("pitch entry must parse");
    let mut pitch = PitchModel::new();
    pitch.load(&pitch_reader, 0).expect("pitch model must load");
    assert_eq!(pitch.frame_interval_seconds(), 0.005);

    let cases = golden_cases();
    assert!(!cases.is_empty(), "expected pitch goldens");
    for case in &cases {
        check_case(&timing, &pitch, case);
    }
}

fn check_case(timing: &PhonemeTiming, pitch: &PitchModel, case: &Path) {
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
        assert_eq!(
            got.language,
            want.language,
            "{} phoneme {index} language",
            case.display()
        );
        assert_eq!(
            got.symbol,
            want.symbol,
            "{} phoneme {index} symbol",
            case.display()
        );
        assert_eq!(
            got.syllable_index,
            want.syllable_index,
            "{} phoneme {index} syllable",
            case.display()
        );
        assert_eq!(
            got.timing_interval_index,
            want.timing_interval_index,
            "{} phoneme {index} interval",
            case.display()
        );
        check_close(
            got.duration_seconds,
            want.duration_seconds,
            &format!("{} phoneme {index} duration", case.display()),
        );
    }

    // The dumper drives PitchModel with default note flags.
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
        .run(
            &notes,
            &durations,
            &mut midi_pitch,
            0,
            None,
            &cancel,
            None,
            &[],
        )
        .unwrap_or_else(|error| panic!("{} pitch must predict: {error}", case.display()));
    assert_eq!(
        midi_pitch.len(),
        golden.pitch.len(),
        "{} frame count",
        case.display()
    );
    for (index, (&got, &want)) in midi_pitch.iter().zip(&golden.pitch).enumerate() {
        check_close(
            f64::from(got),
            f64::from(want),
            &format!("{} frame {index}", case.display()),
        );
    }

    // A second run through the phrase state must reuse the curve verbatim.
    let mut state = PitchModelState::default();
    let mut rerun = Vec::new();
    pitch
        .run(
            &notes,
            &durations,
            &mut rerun,
            0,
            Some(&mut state),
            &cancel,
            None,
            &[],
        )
        .expect("first stateful run must predict");
    assert!(!state.reused_prediction, "first run computes");
    let mut cached = Vec::new();
    pitch
        .run(
            &notes,
            &durations,
            &mut cached,
            0,
            Some(&mut state),
            &cancel,
            None,
            &[],
        )
        .expect("second stateful run must predict");
    assert!(state.reused_prediction, "second run reuses");
    assert_eq!(cached, rerun, "cached curve is verbatim");
    assert_eq!(rerun, midi_pitch, "stateful run matches");
    println!(
        "{}: {} frames ok",
        case.file_name().expect("name").to_string_lossy(),
        midi_pitch.len()
    );
}
