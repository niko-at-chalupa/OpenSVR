use std::f64::consts::TAU;

use opensvr_core::{CancelToken, VoiceSettings};
use thiserror::Error;

use crate::{Phrase, REST_CONTEXT_SECONDS};

/// A note on the absolute timeline, in seconds, ready to be sung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimedNote<'a> {
    pub start: f64,
    pub end: f64,
    /// MIDI note number after transposition.
    pub pitch: u8,
    pub detune_cents: f64,
    /// Length of one quarter note at the note start, for pitch-curve timing.
    pub seconds_per_quarter: f64,
    pub lyrics: &'a str,
    /// Explicit phoneme override; empty when the lyrics should be looked up.
    pub phonemes: &'a str,
    /// Start of the placing group reference's crop window, in seconds.
    ///
    /// Backends keep the full model-input context (onset consonants live in
    /// the preceding rest); the renderer crops the mixed output here, exactly
    /// as OpenSV's `cropStartSeconds` in `src/audio/ProjectRenderer.cpp`.
    pub window_start: f64,
    /// End of the crop window, in seconds; `None` means unbounded.
    ///
    /// The mixed output stops here (`stopSeconds` in C++): the release tail
    /// never rings past the group edge.
    pub window_end: Option<f64>,
}

/// Mono audio rendered for one phrase.
#[derive(Debug, Clone, PartialEq)]
pub struct PhraseAudio {
    /// Absolute start time of `samples[0]` in seconds. Neural backends need
    /// leading context for onset consonants, so this precedes the first note;
    /// the renderer drops whatever falls before zero.
    pub start: f64,
    /// Mono samples at [`sample_rate`][Self::sample_rate].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// A voice backend failed to produce audio.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct BackendError(pub String);

/// Something that can sing: the seam where OpenSV's neural synthesis plugs in.
pub trait VoiceBackend {
    /// Synthesizes one `phrase` of non-overlapping, time-ordered notes as mono audio.
    ///
    /// The returned samples run [`REST_CONTEXT_SECONDS`] past the end of the last
    /// note so the release is not cut off; the renderer crops anything beyond the
    /// phrase's group window (and the project end).
    fn render_phrase(
        &mut self,
        voice: &VoiceSettings,
        phrase: &Phrase<'_>,
        sample_rate: u32,
        cancel: &CancelToken,
    ) -> Result<PhraseAudio, BackendError>;
}

/// Placeholder backend that plays each note as a sine tone and ignores lyrics and voice.
#[derive(Debug, Clone, Copy)]
pub struct ToneBackend {
    amplitude: f32,
}

impl Default for ToneBackend {
    fn default() -> Self {
        Self { amplitude: 0.25 }
    }
}

impl ToneBackend {
    /// Linear fade at both ends of each tone, avoiding clicks.
    const FADE_SECONDS: f64 = 0.01;
}

impl VoiceBackend for ToneBackend {
    fn render_phrase(
        &mut self,
        _voice: &VoiceSettings,
        phrase: &Phrase<'_>,
        sample_rate: u32,
        _cancel: &CancelToken,
    ) -> Result<PhraseAudio, BackendError> {
        let (Some(first), Some(last)) = (phrase.notes.first(), phrase.notes.last()) else {
            return Ok(PhraseAudio {
                start: 0.0,
                samples: Vec::new(),
                sample_rate,
            });
        };
        let rate = f64::from(sample_rate);
        let length = ((last.end + REST_CONTEXT_SECONDS - first.start) * rate).round() as usize;
        let mut samples = vec![0.0_f32; length];

        for note in phrase.notes {
            let offset = ((note.start - first.start) * rate).round() as usize;
            let count = ((note.end - note.start) * rate).round() as usize;
            let fade = ((Self::FADE_SECONDS * rate) as usize).clamp(1, (count / 2).max(1));
            let semitones = f64::from(note.pitch) - 69.0 + note.detune_cents / 100.0;
            let step = TAU * 440.0 * (semitones / 12.0).exp2() / rate;

            let window = samples.iter_mut().skip(offset).take(count);
            for (index, sample) in window.enumerate() {
                let edge = (index + 1).min(count - index);
                let envelope = (edge as f32 / fade as f32).min(1.0);
                *sample += self.amplitude * envelope * (step * index as f64).sin() as f32;
            }
        }
        Ok(PhraseAudio {
            start: first.start,
            samples,
            sample_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start: f64, end: f64, pitch: u8) -> TimedNote<'static> {
        TimedNote {
            start,
            end,
            pitch,
            detune_cents: 0.0,
            seconds_per_quarter: 0.5,
            lyrics: "la",
            phonemes: "",
            window_start: 0.0,
            window_end: None,
        }
    }

    fn render(notes: &[TimedNote<'_>], rate: u32) -> PhraseAudio {
        ToneBackend::default()
            .render_phrase(
                &VoiceSettings::default(),
                &Phrase { notes },
                rate,
                &CancelToken::new(),
            )
            .unwrap()
    }

    #[test]
    fn a4_completes_440_cycles_per_second() {
        let audio = render(&[note(0.0, 1.0, 69)], 44_100);
        assert_eq!(audio.start, 0.0);
        assert_eq!(audio.samples.len(), 48_510, "one second plus the release context");
        let rising = audio.samples[..44_100]
            .windows(2)
            .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
            .count();
        assert!(
            (439..=441).contains(&rising),
            "{rising} rising zero crossings"
        );
        assert!(audio.samples.iter().all(|s| s.abs() <= 0.25));
    }

    #[test]
    fn notes_are_placed_relative_to_the_phrase_start() {
        let notes = [note(2.0, 2.1, 60), note(2.5, 2.6, 62)];
        let audio = render(&notes, 1000);
        assert_eq!(audio.start, 2.0);
        assert_eq!(audio.samples.len(), 700);
        assert!(
            audio.samples[200..400].iter().all(|&s| s == 0.0),
            "the rest between notes is silent"
        );
        assert!(audio.samples[500..600].iter().any(|&s| s != 0.0));
        assert!(
            render(&[], 1000).samples.is_empty(),
            "an empty phrase renders no audio"
        );
    }
}
