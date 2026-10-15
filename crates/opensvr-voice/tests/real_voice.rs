//! Gated tests against the real duration model in a local `.nofs` voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice in the
//! OpenSV references) to run; the tests return early when unset so CI stays
//! green without proprietary data. Voice files are never committed: only the
//! small text goldens under `tests/golden/voice/` are, dumped from the C++
//! engine's voice-compatible timing loader (see the throwaway dumper kept out
//! of tree per `AGENTS.md`).

use std::path::{Path, PathBuf};

use opensvr_dnni::DnniReader;
use opensvr_nofs::{VoiceConfiguration, VoiceDatabase};
use opensvr_voice::{PhonemeDuration, PhonemeTiming, TimedPhoneme, TimingSyllable};

/// Relative tolerance for predicted durations (ROADMAP Phase 5 notes).
const RELATIVE_TOLERANCE: f64 = 1e-5;

struct Golden {
    syllables: Vec<TimingSyllable>,
    frame_interval: f32,
    durations: Vec<PhonemeDuration>,
    frames: Vec<TimedPhoneme>,
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
        frame_interval: 0.0,
        durations: Vec::new(),
        frames: Vec::new(),
    };
    while let Some(line) = lines.next() {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("in") => {
                let language = words.next().expect("syllable language").to_owned();
                let midi_pitch: i32 = words.next().expect("midi").parse().expect("midi int");
                let duration_seconds: f64 = words
                    .next()
                    .expect("seconds")
                    .parse()
                    .expect("seconds float");
                let phonemes: Vec<String> = words.map(str::to_owned).collect();
                assert!(!phonemes.is_empty(), "golden syllable needs phonemes");
                golden.syllables.push(TimingSyllable {
                    language,
                    phonemes,
                    duration_seconds,
                    midi_pitch,
                    is_continuation: false,
                });
            }
            Some("interval") => {
                golden.frame_interval = words
                    .next()
                    .expect("interval")
                    .parse()
                    .expect("interval float");
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
            Some("frames") => {
                let count: usize = words.next().expect("count").parse().expect("count int");
                for _ in 0..count {
                    let entry = lines.next().expect("frame entry");
                    let mut fields = entry.split_whitespace();
                    golden.frames.push(TimedPhoneme {
                        language: fields.next().expect("language").to_owned(),
                        symbol: fields.next().expect("symbol").to_owned(),
                        frame_count: fields.next().expect("frames").parse().expect("frames int"),
                    });
                }
            }
            Some(other) => panic!("golden {} has unknown section {other}", path.display()),
            None => {}
        }
    }
    assert!(!golden.syllables.is_empty(), "golden needs syllables");
    assert!(!golden.durations.is_empty(), "golden needs durations");
    assert_eq!(
        golden.durations.len(),
        golden.frames.len(),
        "durations and frames must pair up"
    );
    golden
}

fn load_real_model() -> Option<PhonemeTiming> {
    let Ok(voice) = std::env::var("OPENSVR_TEST_VOICE") else {
        return None;
    };
    let db = VoiceDatabase::open(PathBuf::from(&voice).as_path()).expect("voice must open");
    let config = VoiceConfiguration::load(&db).expect("config must load");
    let key = config.duration_model().key.clone();
    let entry = db
        .entries()
        .iter()
        .find(|entry| entry.key == key)
        .expect("voice must contain the duration entry");
    let bytes = db.read_entry(entry).expect("entry must read").to_vec();
    let reader = DnniReader::from_bytes(bytes).expect("duration entry must parse");
    let mut model = PhonemeTiming::new();
    model.load(&reader, 0).expect("duration model must load");
    Some(model)
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/voice")
}

#[test]
fn real_duration_model_matches_cpp_goldens() {
    let Some(model) = load_real_model() else {
        return;
    };
    let mut cases: Vec<PathBuf> = std::fs::read_dir(golden_dir())
        .expect("golden dir must list")
        .map(|entry| entry.expect("golden entry").path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("timing-"))
        })
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "expected timing goldens");
    for case in cases {
        let golden = parse_golden(&case);
        let mut predicted = Vec::new();
        model
            .predict(&golden.syllables, &mut predicted)
            .unwrap_or_else(|error| panic!("{} must predict: {error}", case.display()));
        assert_eq!(
            predicted.len(),
            golden.durations.len(),
            "{} phoneme count",
            case.display()
        );
        for (index, (got, want)) in predicted.iter().zip(&golden.durations).enumerate() {
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
            let allowed = RELATIVE_TOLERANCE * want.duration_seconds.abs().max(1.0);
            assert!(
                (got.duration_seconds - want.duration_seconds).abs() <= allowed,
                "{} phoneme {index}: got {}, want {}",
                case.display(),
                got.duration_seconds,
                want.duration_seconds
            );
        }
        let mut timed = Vec::new();
        opensvr_voice::quantize_phoneme_durations(&predicted, golden.frame_interval, &mut timed)
            .expect("must quantize");
        assert_eq!(
            timed,
            golden.frames,
            "{} quantized frames must match exactly",
            case.display()
        );
        println!(
            "{}: {} phonemes ok",
            case.file_name().expect("name").to_string_lossy(),
            predicted.len()
        );
    }
}
