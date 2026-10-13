//! Gated end-to-end test for [`NeuralBackend`](opensvr_audio::NeuralBackend):
//! first sung output through the real phrase path on a real voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice) to
//! run; the test returns early when unset so CI stays green without
//! proprietary data. The SVP input is synthetic and committed under
//! `tests/golden/audio/`; the voice blob itself never is.
//!
//! Status per the active next-steps generation: every chain stage is
//! golden-pinned, but the C++ CLI reference binary is disqualified (built
//! from a dirty tree), so this test pins self-consistency — success,
//! determinism, and sung-vs-silence structure — instead of a C++ waveform
//! comparison. The sample-for-sample comparison lands once a clean CLI
//! rebuild provides trusted reference audio.

use std::path::PathBuf;

use opensvr_audio::{CancelToken, NeuralBackend, Renderer};

/// Project rate for the test: exercises the native-to-project resampler
/// (the voice renders natively at 48 kHz).
const SAMPLE_RATE: u32 = 8000;
/// Score length of `neural-basic.svp` (1.5 s) plus the 0.1 s rest context.
const EXPECTED_FRAMES: f64 = 12_800.0;

fn voice_path() -> Option<PathBuf> {
    std::env::var("OPENSVR_TEST_VOICE").ok().map(PathBuf::from)
}

fn mean_abs(samples: &[f32]) -> f64 {
    samples.iter().map(|sample| f64::from(sample.abs())).sum::<f64>()
        / samples.len().max(1) as f64
}

#[test]
fn neural_backend_sings_the_basic_phrase_deterministically() {
    let Some(voice) = voice_path() else {
        return;
    };
    let svp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/audio/neural-basic.svp");
    let mut project = opensvr_svp::load(&svp).expect("basic SVP must parse");
    assert_eq!(project.tracks.len(), 1);
    for track in &mut project.tracks {
        // Every note carries explicit phonemes, so no dictionary is needed.
        track.voice.database = Some(voice.clone());
    }

    let cancel = CancelToken::new();
    let mut renderer = Renderer::new(NeuralBackend::new());
    let first = renderer
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("neural render must succeed");

    // Timeline: 1.5 s of score plus 0.1 s of rest context, up to float dust.
    assert!(
        (first.frames() as f64 - EXPECTED_FRAMES).abs() <= 2.0,
        "frames: got {}, want ~{EXPECTED_FRAMES}",
        first.frames()
    );
    assert!(first.left.iter().all(|sample| sample.is_finite()));
    assert!(first.right.iter().all(|sample| sample.is_finite()));

    // Sung, not silent: the voiced "k a" region carries far more energy
    // than the "sil" note at 1.0..1.2 s (measured on GUMI AI).
    let voiced = mean_abs(&first.left[800..2400]);
    let silence = mean_abs(&first.left[8400..9200]);
    let peak: f32 = first
        .left
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(peak > 0.05, "peak {peak}: the phrase must sing audibly");
    assert!(
        voiced > 10.0 * silence,
        "voiced {voiced} vs silence {silence}: vowels must dominate the rest note"
    );

    // Deterministic: same seed, same summation order, same output bits.
    let second = renderer
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("neural re-render must succeed");
    assert_eq!(first, second, "re-renders must be bitwise identical");
}
