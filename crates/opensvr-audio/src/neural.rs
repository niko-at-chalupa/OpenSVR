//! Neural singing backend: whole phrases through `VoiceSynthesizer`.
//!
//! Ports the phrase body of OpenSV's `src/audio/ProjectRenderer.cpp` (the loop
//! over phrases cut at [`PHRASE_BREAK_SECONDS`][crate::PHRASE_BREAK_SECONDS]):
//! lyrics resolve to phonemes ([`NoteResolver`], with the `+`/`-` sequencing and
//! the next-legato lookahead), inter-note gaps become `sil` syllables, timing
//! predicts durations, [`quantize_phoneme_durations`][opensvr_voice::quantize_phoneme_durations]
//! snaps them to the acoustic grid, [`merge_continuation_phones`][crate::merge_continuation_phones]
//! folds legato repeats, pitch predicts the automatic curve, [`build_log_f0`][crate::build_log_f0]
//! assembles log-F0, and `render` (seed 5489, default vocal-mode weights) sings
//! native-rate audio that is resampled to the project rate.
//!
//! Voice databases load per phrase from `VoiceSettings::database`; dictionaries
//! load per phrase from `VoiceSettings::dictionary` (a missing directory only
//! fails phrases that actually need a lookup — explicit phonemes never touch
//! it, exactly as in C++). Per-phrase loading is the simple correct baseline;
//! stamp-keyed voice/dictionary caches are a later Phase 6 PR per `ROADMAP.md`.

use opensvr_core::{CancelToken, Language, VoiceSettings};
use opensvr_g2p::{DictionarySet, NoteResolver, TimingSyllable, load_dictionaries};
use opensvr_voice::{
    DEFAULT_RENDER_SEED, NeuralVocoderOutput, PITCH_CONTEXT_SECONDS, PhonemeDuration, PitchNote,
    VoiceSynthesizer, quantize_phoneme_durations,
};

use crate::pitch::{build_log_f0, merge_continuation_phones};
use crate::resample::resample_mono;
use crate::{BackendError, Phrase, PhraseAudio, REST_CONTEXT_SECONDS, VoiceBackend};

/// Largest synthesis unit: matches `maximumPhraseSeconds` in `ProjectRenderer.cpp`.
const MAXIMUM_PHRASE_SECONDS: f64 = 30.0;
/// Largest phoneme count per phrase: `maximumPhrasePhonemes`.
const MAXIMUM_PHRASE_PHONEMES: usize = 4096;
/// Legato lookahead tolerance in seconds, exactly as in C++.
const LEGATO_EPSILON_SECONDS: f64 = 0.000_001;

/// Timing and pitch-model inputs for one phrase.
#[derive(Debug, Clone)]
struct PhraseInputs {
    syllables: Vec<TimingSyllable>,
    pitch_notes: Vec<PitchNote>,
}

fn backend_error(message: impl Into<String>) -> BackendError {
    BackendError(message.into())
}

/// Builds timing syllables and pitch notes for a phrase's score notes.
///
/// Mirrors the C++ phrase loop: a `restContextSeconds` leading `sil`, one `sil`
/// per positive inter-note gap (also as a pitch silence past the first note),
/// then one syllable plus one pitch note per score note, and a trailing `sil`
/// syllable. `is_continuation` marks `-` legatos. Rejects phrases at the
/// 4096-phoneme inference limit.
fn build_phrase_inputs(
    notes: &[crate::TimedNote<'_>],
    language: Language,
    dictionaries: &DictionarySet,
) -> Result<PhraseInputs, BackendError> {
    let track_language = language.as_str().to_owned();
    let mut sequencer = NoteResolver::new(language);
    let mut syllables = Vec::with_capacity(notes.len() + 2);
    let mut pitch_notes = Vec::with_capacity(notes.len() + 1);
    let mut phoneme_count = 0_usize;
    let mut cursor_seconds = notes[0].start - REST_CONTEXT_SECONDS;

    for (index, note) in notes.iter().enumerate() {
        let gap_seconds = if index == 0 {
            REST_CONTEXT_SECONDS
        } else {
            note.start - cursor_seconds
        };
        if gap_seconds > 0.0 {
            syllables.push(TimingSyllable {
                language: track_language.clone(),
                phonemes: vec!["sil".to_owned()],
                duration_seconds: gap_seconds,
                midi_pitch: i32::from(note.pitch),
                is_continuation: false,
            });
            if index != 0 {
                let silence = PitchNote {
                    syllable: syllables.last().expect("just pushed").clone(),
                    is_silence: true,
                    ..PitchNote::default()
                };
                pitch_notes.push(silence);
            }
            phoneme_count += 1;
        }
        let is_legato = note.lyrics == "-";
        let next_is_legato = index + 1 < notes.len()
            && notes[index + 1].lyrics == "-"
            && (notes[index + 1].start - note.end).abs() <= LEGATO_EPSILON_SECONDS;
        let sung = sequencer
            .push(note.lyrics, note.phonemes, dictionaries, next_is_legato)
            .map_err(|error| backend_error(error.to_string()))?;
        let pitch = i32::from(note.pitch);
        let syllable = TimingSyllable {
            language: sung.language.as_str().to_owned(),
            phonemes: sung.phonemes.clone(),
            duration_seconds: note.end - note.start,
            midi_pitch: pitch,
            is_continuation: is_legato,
        };
        let pitch_note = PitchNote {
            syllable: syllable.clone(),
            is_silence: sung.phonemes == ["sil"],
            is_breath: sung.phonemes == ["br"],
            is_continuation: is_legato,
            ..PitchNote::default()
        };
        syllables.push(syllable);
        pitch_notes.push(pitch_note);
        phoneme_count += sung.phonemes.len();
        if phoneme_count >= MAXIMUM_PHRASE_PHONEMES {
            return Err(backend_error("The phrase exceeds the 4096-phoneme inference limit."));
        }
        cursor_seconds = note.end;
    }
    syllables.push(TimingSyllable {
        language: track_language,
        phonemes: vec!["sil".to_owned()],
        duration_seconds: REST_CONTEXT_SECONDS,
        midi_pitch: i32::from(notes[notes.len() - 1].pitch),
        is_continuation: false,
    });
    Ok(PhraseInputs {
        syllables,
        pitch_notes,
    })
}

/// Whether a note reaches the pronunciation dictionary (and therefore needs one).
///
/// Explicit overrides and dotted raw phonemes resolve without files; `+`/`-`
/// sequencing without overrides either reuses state or fails before any lookup.
fn needs_dictionary(lyrics: &str, explicit: &str) -> bool {
    lyrics != "+" && lyrics != "-" && explicit.is_empty() && !lyrics.starts_with('.')
}

/// Loads the track dictionaries, tolerating a missing directory for phrases
/// whose notes all carry explicit phonemes.
fn dictionaries_for(
    voice: &VoiceSettings,
    notes: &[crate::TimedNote<'_>],
) -> Result<DictionarySet, BackendError> {
    if voice.dictionary.is_none()
        && notes
            .iter()
            .any(|note| needs_dictionary(note.lyrics, note.phonemes))
    {
        return Err(backend_error(
            "Select the clf-data pronunciation dictionary directory or enter explicit phonemes.",
        ));
    }
    let dictionaries = match voice.dictionary.as_deref() {
        Some(directory) => load_dictionaries(directory, voice.language)
            .map_err(|error| backend_error(error.to_string()))?,
        // Every remaining resolution path is explicit or state-carried, and the
        // tolerated English-inference lookup degrades gracefully on empty tables.
        None => DictionarySet::default(),
    };
    Ok(dictionaries)
}

/// Sings whole phrases through the neural voice: the first sung output.
#[derive(Debug, Default)]
pub struct NeuralBackend;

impl NeuralBackend {
    pub fn new() -> Self {
        Self
    }
}

impl VoiceBackend for NeuralBackend {
    fn render_phrase(
        &mut self,
        voice: &VoiceSettings,
        phrase: &Phrase<'_>,
        sample_rate: u32,
        cancel: &CancelToken,
    ) -> Result<PhraseAudio, BackendError> {
        let (Some(first), Some(last)) = (phrase.notes.first(), phrase.notes.last()) else {
            return Ok(PhraseAudio {
                start: 0.0,
                samples: Vec::new(),
                sample_rate,
            });
        };
        // The duration model places onset consonants in the preceding rest;
        // only the mixed output is cropped, never the model input.
        let start_seconds = first.start - REST_CONTEXT_SECONDS;
        let release_end = last.end + REST_CONTEXT_SECONDS;
        if release_end - start_seconds > MAXIMUM_PHRASE_SECONDS {
            return Err(backend_error(
                "the connected phrase exceeds 30 seconds; add a rest of at least 0.2 seconds to separate phrases.",
            ));
        }
        let dictionaries = dictionaries_for(voice, phrase.notes)?;
        let inputs = build_phrase_inputs(phrase.notes, voice.language, &dictionaries)?;

        let database = voice
            .database
            .as_deref()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| {
                backend_error("Select a voice.nofs database before playing singing notes.")
            })?;
        let mut synthesizer = VoiceSynthesizer::new();
        synthesizer
            .load(database)
            .map_err(|error| backend_error(error.to_string()))?;

        let mut durations: Vec<PhonemeDuration> = Vec::new();
        synthesizer
            .predict(&inputs.syllables, &mut durations)
            .map_err(|error| backend_error(error.to_string()))?;
        let frame_interval = synthesizer.frame_interval_seconds();
        let mut timed = Vec::new();
        quantize_phoneme_durations(&durations, frame_interval, &mut timed)
            .map_err(|error| backend_error(error.to_string()))?;
        let timed = merge_continuation_phones(&durations, &inputs.syllables, &timed);
        let frame_count: usize = timed.iter().map(|phoneme| phoneme.frame_count).sum();
        if frame_count as f64 * f64::from(frame_interval)
            > MAXIMUM_PHRASE_SECONDS + f64::from(frame_interval)
        {
            return Err(backend_error(
                "The predicted phrase exceeds the 30-second inference limit.",
            ));
        }

        // Empty vibrato envelope means unity; empty vocal-mode weights mean defaults.
        let mut automatic_pitch = Vec::new();
        synthesizer
            .predict_pitch(&inputs.pitch_notes, &mut automatic_pitch, cancel, None, None, &[])
            .map_err(|error| backend_error(error.to_string()))?;
        let log_f0 = build_log_f0(
            phrase.notes,
            start_seconds,
            frame_interval,
            frame_count,
            &automatic_pitch,
            first.start - PITCH_CONTEXT_SECONDS,
            synthesizer.pitch_frame_interval_seconds(),
        )?;

        let mut rendered = NeuralVocoderOutput::default();
        synthesizer
            .render(
                &timed,
                &log_f0,
                &mut rendered,
                DEFAULT_RENDER_SEED,
                cancel,
                None,
                None,
                &[],
            )
            .map_err(|error| backend_error(error.to_string()))?;
        if !rendered.sample_rate.is_finite() || rendered.sample_rate <= 0.0 {
            return Err(backend_error("The voice renderer returned invalid audio."));
        }
        let native_rate = rendered.sample_rate.round() as u32;
        let samples = resample_mono(&rendered.samples, native_rate, sample_rate)?;
        Ok(PhraseAudio {
            start: start_seconds,
            samples,
            sample_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use opensvr_g2p::PhonemeDictionary;

    use super::*;

    fn parse_generic(phones: &str, dict: &str) -> PhonemeDictionary {
        PhonemeDictionary::parse(
            phones.as_bytes(),
            dict.as_bytes(),
            "phones.txt",
            "dict.txt",
            "x-phones.txt".to_owned(),
            "x-dict.txt".to_owned(),
        )
        .expect("synthetic dictionary parses")
    }

    fn japanese_set() -> DictionarySet {
        DictionarySet {
            local: parse_generic(
                "a vowel\nk consonant\ns consonant\nsil silence\n",
                "ka k a\nsa s a\n",
            ),
            english: None,
        }
    }

    fn note(start: f64, end: f64, pitch: u8, lyrics: &'static str) -> crate::TimedNote<'static> {
        crate::TimedNote {
            start,
            end,
            pitch,
            detune_cents: 0.0,
            seconds_per_quarter: 0.5,
            lyrics,
            phonemes: "",
            window_start: 0.0,
            window_end: None,
        }
    }

    fn languages(inputs: &PhraseInputs) -> Vec<&str> {
        inputs.syllables.iter().map(|s| s.language.as_str()).collect::<Vec<_>>()
    }

    #[test]
    fn adjacent_notes_share_leading_and_trailing_silence() {
        let notes = [note(0.0, 0.4, 60, "ka"), note(0.4, 0.8, 62, "sa")];
        let inputs = build_phrase_inputs(&notes, Language::Japanese, &japanese_set()).unwrap();
        assert_eq!(
            inputs
                .syllables
                .iter()
                .map(|s| s.phonemes.join(" "))
                .collect::<Vec<_>>(),
            ["sil", "k a", "s a", "sil"],
        );
        assert_eq!(languages(&inputs), ["japanese"; 4]);
        // Leading/trailing rests take the score pitch of their neighbour note.
        assert_eq!(inputs.syllables[0].midi_pitch, 60);
        assert_eq!(inputs.syllables[3].midi_pitch, 62);
        // No inter-note gap, so no pitch silence: one pitch note per score note.
        assert_eq!(inputs.pitch_notes.len(), 2);
        assert!(inputs.pitch_notes.iter().all(|n| !n.is_silence));
    }

    #[test]
    fn gaps_become_silence_in_both_streams() {
        let notes = [note(1.0, 1.4, 60, "ka"), note(1.6, 2.0, 62, "sa")];
        let inputs = build_phrase_inputs(&notes, Language::Japanese, &japanese_set()).unwrap();
        assert_eq!(inputs.syllables.len(), 5);
        assert_eq!(inputs.syllables[2].phonemes, ["sil"]);
        assert!(
            (inputs.syllables[2].duration_seconds - 0.2).abs() < 1e-9,
            "got {}",
            inputs.syllables[2].duration_seconds
        );
        assert_eq!(inputs.pitch_notes.len(), 3);
        assert!(inputs.pitch_notes[1].is_silence);
        assert_eq!(inputs.pitch_notes[1].syllable.midi_pitch, 62);
    }

    #[test]
    fn legato_marks_continuation_and_reuses_the_vowel() {
        let notes = [note(0.0, 0.4, 60, "ka"), note(0.4, 0.8, 62, "-")];
        let inputs = build_phrase_inputs(&notes, Language::Japanese, &japanese_set()).unwrap();
        assert_eq!(inputs.syllables[2].phonemes, ["a"]);
        assert!(inputs.syllables[2].is_continuation);
        assert!(inputs.pitch_notes[1].is_continuation);
    }

    #[test]
    fn missing_dictionaries_fail_only_lookup_phrases() {
        let voice = VoiceSettings::default();
        // Explicit phonemes never touch the dictionary.
        let explicit = [crate::TimedNote {
            phonemes: "k a",
            window_start: 0.0,
            window_end: None,
            ..note(0.0, 0.4, 60, "ka")
        }];
        assert!(dictionaries_for(&voice, &explicit).is_ok());
        // A plain lyric without overrides needs the files.
        let plain = [note(0.0, 0.4, 60, "ka")];
        assert_eq!(
            dictionaries_for(&voice, &plain).unwrap_err().to_string(),
            "Select the clf-data pronunciation dictionary directory or enter explicit phonemes."
        );
    }

    #[test]
    fn thirty_second_phrases_are_rejected() {
        let notes = [note(0.0, 31.0, 60, "ka")];
        let mut backend = NeuralBackend::new();
        let error = backend
            .render_phrase(
                &VoiceSettings::default(),
                &Phrase { notes: &notes },
                48_000,
                &CancelToken::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("exceeds 30 seconds"), "{error}");
    }

    #[test]
    fn empty_phrases_render_no_audio() {
        let mut backend = NeuralBackend::new();
        let audio = backend
            .render_phrase(
                &VoiceSettings::default(),
                &Phrase { notes: &[] },
                48_000,
                &CancelToken::new(),
            )
            .unwrap();
        assert!(audio.samples.is_empty());
    }
}
