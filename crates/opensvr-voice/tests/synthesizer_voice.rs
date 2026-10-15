//! Gated tests for the full [`VoiceSynthesizer`](opensvr_voice::VoiceSynthesizer)
//! chain in a local `.nofs` voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice in the
//! OpenSV references) to run; the tests return early when unset so CI stays
//! green without proprietary data. Voice files are never committed: only the
//! small text goldens under `tests/golden/voice/` are, dumped from the C++
//! engine (see the throwaway dumper kept out of tree per `AGENTS.md`).
//!
//! No new goldens are needed: the vocoder goldens' `in`/`durations`/`pitch`/
//! `acoustic`/`waveform` sections already pin the full chain. `predict`
//! re-predicts durations against the golden durations; `predict_pitch` runs
//! through the half-second silence padding (so its output is longer than the
//! unpadded golden pitch and is checked structurally, not against the
//! golden); `render` runs acoustic plus vocoder from the golden timed
//! phonemes and log-F0 and compares every waveform sample at `1e-5`
//! relative tolerance.

use std::path::{Path, PathBuf};

use opensvr_core::CancelToken;
use opensvr_nofs::VoiceDatabase;
use opensvr_voice::{
    DEFAULT_RENDER_SEED, NeuralVocoderOutput, PhonemeDuration, PitchNote, SynthesisStatistics,
    TimedPhoneme, VoiceSynthesizer, VoiceSynthesizerState, midi_to_log_f0,
};

/// Relative tolerance for durations and samples.
const RELATIVE_TOLERANCE: f64 = 1e-5;
/// Acoustic frame grid in seconds.
const FRAME_INTERVAL_SECONDS: f32 = 0.005;
/// Pitch context padded on each side by `predict_pitch` (0.5 s at 5 ms).
const PITCH_CONTEXT_FRAMES: usize = 100;
/// Guard band against the golden's truncation boundary (see `check_case`).
const TRUNCATION_GUARD_FRAMES: usize = 8;

struct Golden {
    syllables: Vec<opensvr_voice::TimingSyllable>,
    durations: Vec<PhonemeDuration>,
    pitch: Vec<f32>,
    acoustic: Vec<f32>,
    waveform_frames: usize,
    hop_samples: usize,
    sample_rate: f64,
    waveform: Vec<f32>,
}

#[allow(clippy::too_many_lines)]
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
        acoustic: Vec::new(),
        waveform_frames: 0,
        hop_samples: 0,
        sample_rate: 0.0,
        waveform: Vec::new(),
    };
    let mut acoustic_frames = 0_usize;
    let mut acoustic_channels = 0_usize;
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
                    golden.pitch.push(
                        lines
                            .next()
                            .expect("pitch entry")
                            .parse()
                            .expect("pitch float"),
                    );
                }
            }
            Some("acoustic") => {
                acoustic_frames = words.next().expect("frames").parse().expect("frames int");
                acoustic_channels = words
                    .next()
                    .expect("channels")
                    .parse()
                    .expect("channels int");
                for _ in 0..acoustic_frames {
                    let entry = lines.next().expect("acoustic frame");
                    for value in entry.split_whitespace() {
                        golden.acoustic.push(value.parse().expect("acoustic float"));
                    }
                }
            }
            Some("waveform") => {
                golden.waveform_frames = words.next().expect("frames").parse().expect("frames int");
                golden.hop_samples = words.next().expect("hop").parse().expect("hop int");
                golden.sample_rate = words.next().expect("rate").parse().expect("rate float");
                for _ in 0..golden.waveform_frames * golden.hop_samples {
                    golden.waveform.push(
                        lines
                            .next()
                            .expect("waveform sample")
                            .parse()
                            .expect("sample float"),
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
    assert_eq!(
        golden.acoustic.len(),
        acoustic_frames * acoustic_channels,
        "acoustic values must fill the frame grid"
    );
    assert_eq!(
        acoustic_channels, 72,
        "acoustic width must be vocoder-ready"
    );
    assert_eq!(
        golden.waveform.len(),
        golden.waveform_frames * golden.hop_samples,
        "waveform samples must fill the sample grid"
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
                .is_some_and(|name| name.to_string_lossy().starts_with("vocoder-"))
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
fn synthesizer_load_reports_intervals_and_vocal_modes() {
    let Some(voice) = voice_path() else {
        return;
    };
    let mut synthesizer = VoiceSynthesizer::new();
    synthesizer.load(voice.as_path()).expect("voice must load");
    // Loader validation: the acoustic entry's `_ftmf` scan must report 5 ms.
    assert_eq!(synthesizer.frame_interval_seconds(), 0.005);
    assert_eq!(synthesizer.pitch_frame_interval_seconds(), 0.005);
    let db = VoiceDatabase::open(voice.as_path()).expect("voice must open");
    assert_eq!(
        synthesizer.vocal_mode_names(),
        db.metadata().timbre_styles.as_slice()
    );
    assert!(
        !synthesizer.vocal_mode_names().is_empty(),
        "test voice carries timbre styles"
    );

    // A failed load preserves the previously loaded voice.
    assert!(
        synthesizer
            .load(Path::new("/nonexistent-voice-12345.nofs"))
            .is_err()
    );
    assert_eq!(synthesizer.frame_interval_seconds(), 0.005);
    let mut durations = Vec::new();
    synthesizer
        .predict(
            &[opensvr_voice::TimingSyllable {
                language: "japanese".to_owned(),
                phonemes: vec!["k".to_owned(), "a".to_owned()],
                duration_seconds: 0.4,
                midi_pitch: 60,
                is_continuation: false,
            }],
            &mut durations,
        )
        .expect("loaded voice survives the failed load");
    assert_eq!(durations.len(), 2);
}

#[test]
fn synthesizer_full_chain_matches_cpp_goldens() {
    let Some(voice) = voice_path() else {
        return;
    };
    let mut synthesizer = VoiceSynthesizer::new();
    synthesizer.load(voice.as_path()).expect("voice must load");

    let cases = golden_cases();
    assert!(!cases.is_empty(), "expected vocoder goldens");
    for case in &cases {
        check_case(&synthesizer, case);
    }
}

#[allow(clippy::too_many_lines)]
fn check_case(synthesizer: &VoiceSynthesizer, case: &Path) {
    let golden = parse_golden(case);
    let cancel = CancelToken::new();

    // 1. `predict` delegates to timing: durations match the golden exactly.
    let mut durations = Vec::new();
    synthesizer
        .predict(&golden.syllables, &mut durations)
        .unwrap_or_else(|error| panic!("{} predict must run: {error}", case.display()));
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
        check_close(
            got.duration_seconds,
            want.duration_seconds,
            &format!("{} phoneme {index} duration", case.display()),
        );
    }

    // 2. `predict_pitch` pads one silence note plus 0.5 s on each side, so its
    // output is longer than the unpadded golden pitch; check structure and
    // statistics instead of golden values.
    let notes: Vec<PitchNote> = golden
        .syllables
        .iter()
        .map(|syllable| PitchNote {
            syllable: syllable.clone(),
            ..PitchNote::default()
        })
        .collect();
    let mut statistics = SynthesisStatistics::default();
    let mut midi_pitch = Vec::new();
    synthesizer
        .predict_pitch(
            &notes,
            &mut midi_pitch,
            &cancel,
            Some(&mut statistics),
            None,
            &[],
        )
        .unwrap_or_else(|error| panic!("{} predict_pitch must run: {error}", case.display()));
    assert_eq!(
        midi_pitch.len(),
        golden.pitch.len() + 2 * PITCH_CONTEXT_FRAMES,
        "{} padded pitch frame count",
        case.display()
    );
    assert!(
        midi_pitch.iter().all(|pitch| pitch.is_finite()),
        "{} pitch is finite",
        case.display()
    );
    assert_eq!(
        statistics.pitch_frames,
        midi_pitch.len(),
        "{} pitch statistics",
        case.display()
    );
    assert!(
        statistics.dnni_computed_frames > 0,
        "{} pitch computes network frames",
        case.display()
    );
    assert!(
        statistics.pitch_milliseconds >= 0.0,
        "{} pitch timing accumulates",
        case.display()
    );

    // A stateful rerun must reuse the curve verbatim and add no pitch frames.
    let mut state = VoiceSynthesizerState::default();
    let mut rerun = Vec::new();
    let mut first_statistics = SynthesisStatistics::default();
    synthesizer
        .predict_pitch(
            &notes,
            &mut rerun,
            &cancel,
            Some(&mut first_statistics),
            Some(&mut state),
            &[],
        )
        .expect("first stateful pitch run must predict");
    assert!(!state.pitch.reused_prediction, "first run computes");
    let mut cached = Vec::new();
    let mut second_statistics = SynthesisStatistics::default();
    synthesizer
        .predict_pitch(
            &notes,
            &mut cached,
            &cancel,
            Some(&mut second_statistics),
            Some(&mut state),
            &[],
        )
        .expect("second stateful pitch run must predict");
    assert!(state.pitch.reused_prediction, "second run reuses");
    assert_eq!(cached, rerun, "cached curve is verbatim");
    assert_eq!(rerun, midi_pitch, "stateful run matches");
    assert_eq!(
        second_statistics.pitch_frames, 0,
        "reused prediction adds no pitch frames"
    );

    // 3. `render` runs acoustic plus vocoder from the golden timed phonemes
    // and log-F0 over the full phrase. The golden waveform is a 20-frame
    // prefix run (the C++ dumper fed the truncated acoustic prefix straight
    // into the vocoder), so samples within the vocoder's right-context field
    // of the truncation boundary legitimately differ: measured per-frame
    // worst error stays below 0.01% of tolerance through frame 11, grows
    // toward the boundary (frame 15 reaches ~93%), and exceeds tolerance
    // from frame 16 on. The comparison therefore covers the interior
    // `waveform_frames - TRUNCATION_GUARD_FRAMES` frames at full stage
    // tolerance; counts and the mix invariant below cover the whole output.
    let mut timed: Vec<TimedPhoneme> = Vec::new();
    opensvr_voice::quantize_phoneme_durations(
        &golden.durations,
        FRAME_INTERVAL_SECONDS,
        &mut timed,
    )
    .unwrap_or_else(|error| panic!("{} must quantize: {error}", case.display()));
    let log_f0: Vec<f32> = golden
        .pitch
        .iter()
        .map(|pitch| midi_to_log_f0(*pitch))
        .collect();
    let mut render_statistics = SynthesisStatistics::default();
    let mut output = NeuralVocoderOutput::default();
    synthesizer
        .render(
            &timed,
            &log_f0,
            &mut output,
            DEFAULT_RENDER_SEED,
            &cancel,
            Some(&mut render_statistics),
            None,
            &[],
        )
        .unwrap_or_else(|error| panic!("{} render must run: {error}", case.display()));
    let frame_count: usize = timed.iter().map(|phoneme| phoneme.frame_count).sum();
    assert_eq!(
        frame_count,
        golden.pitch.len(),
        "{} quantized frames match pitch frames",
        case.display()
    );
    assert_eq!(
        output.samples.len(),
        frame_count * golden.hop_samples,
        "{} full-phrase sample count",
        case.display()
    );
    assert_eq!(
        render_statistics.acoustic_frames,
        frame_count,
        "{} acoustic statistics",
        case.display()
    );
    assert_eq!(
        render_statistics.vocoder_frames,
        frame_count,
        "{} vocoder statistics",
        case.display()
    );
    assert!(
        render_statistics.dnni_computed_frames > 0,
        "{} render computes network frames",
        case.display()
    );
    assert_eq!(
        output.hop_samples,
        golden.hop_samples,
        "{} hop",
        case.display()
    );
    assert!(
        (output.sample_rate - golden.sample_rate).abs() < 1e-9,
        "{} sample rate",
        case.display()
    );
    // Structural invariant over the whole output: the mix is the
    // periodic/aperiodic sum.
    assert_eq!(output.periodic.len(), output.samples.len());
    assert_eq!(output.aperiodic.len(), output.samples.len());
    for (index, ((sample, periodic), aperiodic)) in output
        .samples
        .iter()
        .zip(&output.periodic)
        .zip(&output.aperiodic)
        .enumerate()
    {
        assert!(
            (f64::from(*sample) - (f64::from(*periodic) + f64::from(*aperiodic))).abs() < 1e-6,
            "{} mix {index}",
            case.display()
        );
    }
    let compared_frames = golden.waveform_frames - TRUNCATION_GUARD_FRAMES;
    let mut worst = 0.0f64;
    for frame in 0..compared_frames {
        for (index, (&got, &want)) in output.samples
            [frame * golden.hop_samples..(frame + 1) * golden.hop_samples]
            .iter()
            .zip(&golden.waveform[frame * golden.hop_samples..(frame + 1) * golden.hop_samples])
            .enumerate()
        {
            let allowed = RELATIVE_TOLERANCE * f64::from(want).abs().max(1.0);
            let error = (f64::from(got) - f64::from(want)).abs();
            worst = worst.max(error / allowed);
            assert!(
                error <= allowed,
                "{} sample {}: got {got}, want {want}",
                case.display(),
                frame * golden.hop_samples + index
            );
        }
    }
    println!(
        "{}: {} interior frames ok (worst {:.3} of tolerance, {} tail frames excluded)",
        case.file_name().expect("name").to_string_lossy(),
        compared_frames,
        worst,
        TRUNCATION_GUARD_FRAMES
    );

    // A stateful render through the phrase state must reproduce the samples.
    let mut render_state = VoiceSynthesizerState::default();
    let mut rerun_output = NeuralVocoderOutput::default();
    synthesizer
        .render(
            &timed,
            &log_f0,
            &mut rerun_output,
            DEFAULT_RENDER_SEED,
            &cancel,
            None,
            Some(&mut render_state),
            &[],
        )
        .expect("stateful render must run");
    assert!(render_state.get_bytes() > 0, "phrase caches are retained");
    let mut cached_output = NeuralVocoderOutput::default();
    synthesizer
        .render(
            &timed,
            &log_f0,
            &mut cached_output,
            DEFAULT_RENDER_SEED,
            &cancel,
            None,
            Some(&mut render_state),
            &[],
        )
        .expect("cached render must run");
    assert_eq!(
        cached_output.samples, rerun_output.samples,
        "cached samples are verbatim"
    );
    assert_eq!(
        rerun_output.samples, output.samples,
        "stateful render matches"
    );
}
