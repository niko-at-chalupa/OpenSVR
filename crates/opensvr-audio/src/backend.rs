use std::f64::consts::TAU;

use opensvr_core::VoiceSettings;
use thiserror::Error;

use crate::REST_CONTEXT_SECONDS;

/// A note on the absolute timeline, in seconds, ready to be sung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimedNote<'a> {
    pub start: f64,
    pub end: f64,
    /// MIDI note number after transposition.
    pub pitch: u8,
    pub detune_cents: f64,
    pub lyrics: &'a str,
    /// Explicit phoneme override; empty when the lyrics should be looked up.
    pub phonemes: &'a str,
}

/// A voice backend failed to produce audio.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct BackendError(pub String);

/// Something that can sing: the seam where OpenSV's neural synthesis plugs in.
pub trait VoiceBackend {
    /// Synthesizes one phrase of non-overlapping, time-ordered `notes` as mono audio.
    ///
    /// The returned samples are at `sample_rate` and begin at `notes[0].start`.
    /// They should run [`REST_CONTEXT_SECONDS`] past the end of the last note so
    /// the release is not cut off; the renderer crops anything beyond the project end.
    fn render_phrase(
        &mut self,
        voice: &VoiceSettings,
        notes: &[TimedNote<'_>],
        sample_rate: u32,
    ) -> Result<Vec<f32>, BackendError>;
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
        notes: &[TimedNote<'_>],
        sample_rate: u32,
    ) -> Result<Vec<f32>, BackendError> {
        let (Some(first), Some(last)) = (notes.first(), notes.last()) else {
            return Ok(Vec::new());
        };
        let rate = f64::from(sample_rate);
        let length = ((last.end + REST_CONTEXT_SECONDS - first.start) * rate).round() as usize;
        let mut samples = vec![0.0_f32; length];

        for note in notes {
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
        Ok(samples)
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
            lyrics: "la",
            phonemes: "",
        }
    }

    #[test]
    fn a4_completes_440_cycles_per_second() {
        let samples = ToneBackend::default()
            .render_phrase(&VoiceSettings::default(), &[note(0.0, 1.0, 69)], 44_100)
            .unwrap();
        assert_eq!(samples.len(), 48_510, "one second plus the release context");
        let rising = samples[..44_100]
            .windows(2)
            .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
            .count();
        assert!(
            (439..=441).contains(&rising),
            "{rising} rising zero crossings"
        );
        assert!(samples.iter().all(|s| s.abs() <= 0.25));
    }

    #[test]
    fn notes_are_placed_relative_to_the_phrase_start() {
        let notes = [note(2.0, 2.1, 60), note(2.5, 2.6, 62)];
        let samples = ToneBackend::default()
            .render_phrase(&VoiceSettings::default(), &notes, 1000)
            .unwrap();
        assert_eq!(samples.len(), 700);
        assert!(
            samples[200..400].iter().all(|&s| s == 0.0),
            "the rest between notes is silent"
        );
        assert!(samples[500..600].iter().any(|&s| s != 0.0));
        assert!(
            ToneBackend::default()
                .render_phrase(&VoiceSettings::default(), &[], 1000)
                .unwrap()
                .is_empty()
        );
    }
}
