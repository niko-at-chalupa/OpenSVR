//! Gated end-to-end test: a Flat voice document loads and sings a short
//! phrase, with and without a named vocal mode.
//!
//! Set `OPENSVR_TEST_FLAT` to an extracted Flat voice document (see
//! `do-not-distribute/flat/`) to run; the test returns early when unset so
//! CI stays green without proprietary data. No goldens are compared: the
//! document vectors already have decode unit tests, so this test pins the
//! run path — every stage runs, vocal-mode weights are accepted, and the
//! mode audibly changes the output.

use std::path::Path;

use opensvr_core::CancelToken;
use opensvr_voice::{
    DEFAULT_RENDER_SEED, NeuralVocoderOutput, PitchNote, TimingSyllable, VoiceSynthesizer,
    midi_to_log_f0,
};

/// Renders one 0.3 s `a` at A4 through every stage; `mode` selects the
/// vocal-mode weight row (`None` sings the database default style).
fn render_with_mode(synthesizer: &VoiceSynthesizer, mode: Option<usize>) -> Vec<f32> {
    let syllable = TimingSyllable {
        language: "japanese".to_owned(),
        phonemes: vec!["a".to_owned()],
        duration_seconds: 0.3,
        midi_pitch: 69,
        is_continuation: false,
    };
    let mut durations = Vec::new();
    synthesizer
        .predict(std::slice::from_ref(&syllable), &mut durations)
        .expect("timing must predict");
    assert!(!durations.is_empty());

    let note = PitchNote {
        syllable,
        ..PitchNote::default()
    };
    let mut midi_pitch = Vec::new();
    synthesizer
        .predict_pitch(
            std::slice::from_ref(&note),
            &mut midi_pitch,
            &CancelToken::new(),
            None,
            None,
            &[],
        )
        .expect("pitch must predict");
    assert!(midi_pitch.iter().all(|value| value.is_finite()));

    let mut timed = Vec::new();
    opensvr_voice::quantize_phoneme_durations(&durations, 0.005, &mut timed)
        .expect("durations must quantize");
    let frames: usize = timed.iter().map(|phoneme| phoneme.frame_count).sum();
    assert!(frames > 0);
    // A flat log-F0 keeps acoustic/vocoder coverage independent of the
    // pitch run's silence padding (which re-times the same syllables).
    let log_f0 = vec![midi_to_log_f0(69.0); frames];

    let modes = synthesizer.vocal_mode_names().len();
    let mut weights = vec![0.0_f32; frames * modes];
    if let Some(mode) = mode {
        for frame in 0..frames {
            weights[frame * modes + mode] = 1.0;
        }
    }
    let mut output = NeuralVocoderOutput::default();
    synthesizer
        .render(
            &timed,
            &log_f0,
            &mut output,
            DEFAULT_RENDER_SEED,
            &CancelToken::new(),
            None,
            None,
            &weights,
        )
        .expect("render must run");
    assert!(!output.samples.is_empty());
    assert!(output.samples.iter().all(|value| value.is_finite()));
    output.samples
}

#[test]
fn flat_voice_sings_with_and_without_vocal_modes() {
    let Some(path) = std::env::var_os("OPENSVR_TEST_FLAT") else {
        return;
    };
    let mut synthesizer = VoiceSynthesizer::new();
    synthesizer
        .load(Path::new(&path))
        .expect("Flat voice must load");
    assert!(!synthesizer.vocal_mode_names().is_empty());
    assert_eq!(synthesizer.default_style().len(), 32);

    let plain = render_with_mode(&synthesizer, None);
    let styled = render_with_mode(&synthesizer, Some(0));
    assert_eq!(plain.len(), styled.len());
    assert!(
        plain != styled,
        "a full vocal mode must change the waveform"
    );
    let energy = |samples: &[f32]| samples.iter().map(|value| value * value).sum::<f32>();
    assert!(energy(&plain) > 0.0, "default render must not be silent");
    assert!(energy(&styled) > 0.0, "styled render must not be silent");
}
