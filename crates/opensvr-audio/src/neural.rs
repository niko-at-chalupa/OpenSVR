//! Neural singing backend: whole phrases through `VoiceSynthesizer`.
//!
//! Ports the phrase body of OpenSV's `src/audio/ProjectRenderer.cpp` (the loop
//! over phrases cut at [`PHRASE_BREAK_SECONDS`][crate::PHRASE_BREAK_SECONDS]):
//! lyrics resolve to phonemes ([`NoteResolver`], with the `+`/`-` sequencing and
//! the next-legato lookahead), inter-note gaps become `sil` syllables, timing
//! predicts durations, [`quantize_phoneme_durations`][opensvr_voice::quantize_phoneme_durations]
//! snaps them to the acoustic grid, [`merge_continuation_phones`][crate::merge_continuation_phones]
//! folds legato repeats, pitch predicts the automatic curve (shaped by the
//! phrase's `vibratoEnv` envelope), [`build_log_f0`][crate::build_log_f0]
//! assembles log-F0 (attributes, `pitchDelta` cents, envelope-scaled vibrato),
//! and `render` (seed 5489, per-phrase vocal-mode weights) sings
//! native-rate audio that is resampled to the project rate.
//!
//! Voice databases load once per file from `VoiceSettings::database` into an
//! MRU voice cache (cap 2, like C++); dictionaries load once per stamp set
//! from `VoiceSettings::dictionary` (a missing directory only fails phrases
//! that actually need a lookup — explicit phonemes never touch it, exactly as
//! in C++). Phrase renders keep their native-rate mono output keyed on the
//! synthesis inputs, so repeated identical phrases resample the cached audio
//! instead of running the vocoder again.

use std::sync::{Mutex, MutexGuard};

use opensvr_core::{CancelToken, Language, VoiceSettings};
use opensvr_g2p::{
    DictionarySet, NoteResolver, TimingSyllable, dictionary_stem, load_dictionaries,
};
use opensvr_voice::{
    DEFAULT_RENDER_SEED, NeuralVocoderOutput, PITCH_CONTEXT_SECONDS, PhonemeDuration, PitchNote,
    TimedPhoneme, VoiceSynthesizer, quantize_phoneme_durations,
};

use crate::cache::{DictionaryCache, DictionaryKey, FileStamp, PhraseCache, VoiceCache};
use crate::pitch::{
    build_log_f0, build_vibrato_envelope, build_vocal_mode_weights, merge_continuation_phones,
};
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

fn cannot_read(path: &std::path::Path) -> BackendError {
    backend_error(format!(
        "Cannot read synthesis resource: {}",
        path.display()
    ))
}

fn stamp_file(path: &std::path::Path) -> Result<FileStamp, BackendError> {
    FileStamp::read(path).map_err(|_| cannot_read(path))
}

/// Stamps every dictionary file `load_dictionaries` reads for one language.
///
/// File selection mirrors `load_dictionaries` (including the `cmudict-07b.txt`
/// preference and the best-effort Japanese English fallback), so the first
/// missing file reports the same path that path would. Errors use the C++
/// `readFileStamp` text verbatim, exactly as the loader does.
fn dictionary_stamps(
    directory: &std::path::Path,
    language: Language,
) -> Result<Vec<FileStamp>, BackendError> {
    let stem = dictionary_stem(language);
    let phones = directory.join(format!("{stem}-phones.txt"));
    let mut dictionary = directory.join(format!("{stem}-dict.txt"));
    if language == Language::English && directory.join("cmudict-07b.txt").is_file() {
        dictionary = directory.join("cmudict-07b.txt");
    }
    let mut stamps = vec![stamp_file(&phones)?, stamp_file(&dictionary)?];
    match language {
        Language::Mandarin => {
            stamps.push(stamp_file(&directory.join("cedict.txt"))?);
        }
        Language::Japanese => {
            for name in [
                "japanese-hira2romaji-dict.txt",
                "japanese-kata2romaji-dict.txt",
                "japanese-sute2romaji-dict.txt",
            ] {
                stamps.push(stamp_file(&directory.join(name))?);
            }
            // Best-effort fallback, exactly as in `load_dictionaries`: only
            // stamp the English files when they are all present.
            let english_phones = directory.join("english-arpabet-phones.txt");
            let cmu = directory.join("cmudict-07b.txt");
            let plain = directory.join("english-arpabet-dict.txt");
            if english_phones.is_file() && (cmu.is_file() || plain.is_file()) {
                stamps.push(stamp_file(&english_phones)?);
                stamps.push(stamp_file(if cmu.is_file() { &cmu } else { &plain })?);
            }
        }
        Language::English | Language::Cantonese | Language::Spanish => {}
    }
    Ok(stamps)
}

/// Stamps a voice database, reporting a missing file exactly as the
/// uncached `VoiceSynthesizer::load` path does today (`could not read voice
/// database ...`, via `VoiceDatabase::open`).
fn stamp_voice(database: &std::path::Path) -> Result<FileStamp, BackendError> {
    FileStamp::read(database).map_err(|source| {
        backend_error(format!(
            "could not read voice database {}: {source}",
            database.display()
        ))
    })
}

/// Sings whole phrases through the neural voice: the first sung output.
///
/// Voice databases, dictionaries and identical phrase renders are cached by
/// file stamp across the phrases (and re-renders) this backend sings, so one
/// render loads each file once. The counters below expose that sharing for
/// tests; cached audio is always resampled and cropped exactly like a fresh
/// render, so hits stay sample-identical.
///
/// The caches sit behind locks so tracks can sing on scoped threads through
/// one shared backend: lookups clone an `Arc` (voices), a set (dictionaries)
/// or the samples (phrases) under a brief lock and sing without holding it,
/// so concurrent phrases synthesize in parallel while each file still loads
/// once.
#[derive(Debug, Default)]
pub struct NeuralBackend {
    voices: Mutex<VoiceCache>,
    dictionaries: Mutex<DictionaryCache>,
    phrases: Mutex<PhraseCache>,
}

/// Recovers a cache lock after a panicking worker: cache entries stay valid.
fn lock_cache<T>(cache: &Mutex<T>) -> MutexGuard<'_, T> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl NeuralBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Successful voice-database loads so far (one per distinct file stamp).
    pub fn voice_loads(&self) -> usize {
        lock_cache(&self.voices).loads()
    }

    /// Successful dictionary loads so far (one per distinct stamp set).
    pub fn dictionary_loads(&self) -> usize {
        lock_cache(&self.dictionaries).loads()
    }

    /// Phrase renders served from the cache so far.
    pub fn phrase_hits(&self) -> usize {
        lock_cache(&self.phrases).hits()
    }

    /// Phrase renders synthesized (and retained) so far.
    pub fn phrase_misses(&self) -> usize {
        lock_cache(&self.phrases).misses()
    }

    /// Returns the track dictionaries, loading them once per stamp set.
    ///
    /// Phrases whose notes all carry explicit phonemes never touch the
    /// filesystem and bypass the cache, exactly as in C++ (where
    /// `resolvePhonemes` returns before the directory check for them).
    fn dictionaries_for(
        &self,
        voice: &VoiceSettings,
        notes: &[crate::TimedNote<'_>],
    ) -> Result<DictionarySet, BackendError> {
        let Some(directory) = voice.dictionary.as_deref() else {
            if notes
                .iter()
                .any(|note| needs_dictionary(note.lyrics, note.phonemes))
            {
                return Err(backend_error(
                    "Select the clf-data pronunciation dictionary directory or enter explicit phonemes.",
                ));
            }
            // Every remaining resolution path is explicit or state-carried, and the
            // tolerated English-inference lookup degrades gracefully on empty tables.
            return Ok(DictionarySet::default());
        };
        if !notes
            .iter()
            .any(|note| needs_dictionary(note.lyrics, note.phonemes))
        {
            return Ok(DictionarySet::default());
        }
        let stamps = dictionary_stamps(directory, voice.language)?;
        let key = DictionaryKey::new(directory.to_owned(), voice.language, stamps);
        lock_cache(&self.dictionaries)
            .get_or_load(key, || {
                load_dictionaries(directory, voice.language)
                    .map_err(|error| backend_error(error.to_string()))
            })
            .cloned()
    }
}

impl VoiceBackend for NeuralBackend {
    fn render_phrase(
        &self,
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
        let dictionaries = self.dictionaries_for(voice, phrase.notes)?;
        let inputs = build_phrase_inputs(phrase.notes, voice.language, &dictionaries)?;

        let database = voice
            .database
            .as_deref()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| {
                backend_error("Select a voice.nofs database before playing singing notes.")
            })?;
        let voice_stamp = stamp_voice(database)?;
        // Clone the shared synthesizer out of the cache before singing: the
        // lock is only held for the stamp lookup (or the single load), never
        // for prediction or vocoding, so tracks synthesize in parallel.
        let synthesizer = lock_cache(&self.voices).get_or_load(voice_stamp.clone(), || {
            let mut loaded = VoiceSynthesizer::new();
            loaded
                .load(database)
                .map_err(|error| backend_error(error.to_string()))?;
            Ok(loaded)
        })?;
        let synthesis = predict_phrase(&synthesizer, &inputs, phrase.notes, start_seconds, cancel)?;

        // Empty vocal-mode weights mean the database default style; a phrase
        // with modes produces per-frame weights that join the phrase cache
        // key, so a mode change misses the modeless entry.
        let vocal_mode_weights = build_vocal_mode_weights(
            phrase.notes,
            start_seconds,
            synthesizer.frame_interval_seconds(),
            synthesis.log_f0.len(),
            synthesizer.vocal_mode_names(),
        )?;
        if let Some((native_rate, native)) = lock_cache(&self.phrases).lookup(
            &voice_stamp,
            &inputs.syllables,
            &synthesis.log_f0,
            &vocal_mode_weights,
        ) {
            let samples = resample_mono(&native, native_rate, sample_rate)?;
            return Ok(PhraseAudio {
                start: start_seconds,
                samples,
                sample_rate,
            });
        }

        let (native_rate, native) = render_native(
            &synthesizer,
            &synthesis.timed,
            &synthesis.log_f0,
            &vocal_mode_weights,
            cancel,
        )?;
        // Cache the native-rate mono output before resampling: placement,
        // project rate and mixer never participate in synthesis, so a hit
        // resamples and crops exactly like this fresh render did.
        lock_cache(&self.phrases).retain(
            voice_stamp,
            inputs.syllables.clone(),
            synthesis.log_f0.clone(),
            vocal_mode_weights,
            native_rate,
            native.clone(),
        );
        let samples = resample_mono(&native, native_rate, sample_rate)?;
        Ok(PhraseAudio {
            start: start_seconds,
            samples,
            sample_rate,
        })
    }
}

/// Timed phonemes plus their log-F0 curve: everything `render` needs.
struct PhraseSynthesis {
    timed: Vec<TimedPhoneme>,
    log_f0: Vec<f32>,
}

/// Runs timing and pitch prediction for one phrase's inputs.
fn predict_phrase(
    synthesizer: &VoiceSynthesizer,
    inputs: &PhraseInputs,
    notes: &[crate::TimedNote<'_>],
    start_seconds: f64,
    cancel: &CancelToken,
) -> Result<PhraseSynthesis, BackendError> {
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

    // The pitch model hears the same `vibratoEnv` envelope the F0 assembly
    // uses, sampled onto its own grid (all ones when the phrase has no
    // curves, which the model treats exactly like the previous empty input).
    let first = notes.first().expect("phrases are never empty");
    let pitch_start = first.start - PITCH_CONTEXT_SECONDS;
    let pitch_interval = synthesizer.pitch_frame_interval_seconds();
    let pitch_duration =
        notes.last().expect("phrases are never empty").end - first.start + 2.0 * PITCH_CONTEXT_SECONDS;
    let pitch_envelope_count =
        (pitch_duration / f64::from(pitch_interval)).ceil() as usize + 2;
    let pitch_envelope =
        build_vibrato_envelope(notes, pitch_start, pitch_interval, pitch_envelope_count);
    let mut automatic_pitch = Vec::new();
    synthesizer
        .predict_pitch(
            &inputs.pitch_notes,
            &mut automatic_pitch,
            cancel,
            None,
            None,
            &pitch_envelope,
        )
        .map_err(|error| backend_error(error.to_string()))?;
    let log_f0 = build_log_f0(
        notes,
        start_seconds,
        frame_interval,
        frame_count,
        &automatic_pitch,
        pitch_start,
        synthesizer.pitch_frame_interval_seconds(),
    )?;
    Ok(PhraseSynthesis { timed, log_f0 })
}

/// Runs the acoustic model and vocoder, returning native-rate mono audio.
fn render_native(
    synthesizer: &VoiceSynthesizer,
    timed: &[TimedPhoneme],
    log_f0: &[f32],
    vocal_mode_weights: &[f32],
    cancel: &CancelToken,
) -> Result<(u32, Vec<f32>), BackendError> {
    let mut rendered = NeuralVocoderOutput::default();
    synthesizer
        .render(
            timed,
            log_f0,
            &mut rendered,
            DEFAULT_RENDER_SEED,
            cancel,
            None,
            None,
            vocal_mode_weights,
        )
        .map_err(|error| backend_error(error.to_string()))?;
    if !rendered.sample_rate.is_finite() || rendered.sample_rate <= 0.0 {
        return Err(backend_error("The voice renderer returned invalid audio."));
    }
    Ok((rendered.sample_rate.round() as u32, rendered.samples))
}

#[cfg(test)]
mod tests {
    use opensvr_core::{Extra, GroupReference, Note, NoteGroup, PitchAttributes, TempoMap};
    use opensvr_g2p::PhonemeDictionary;

    use super::*;

    static SCORE: std::sync::LazyLock<Note> = std::sync::LazyLock::new(|| Note {
        onset: 0,
        duration: 1,
        pitch: 60,
        lyrics: "la".into(),
        phonemes: String::new(),
        detune: 0.0,
        instant_mode: true,
        musical_type: "singing".into(),
        accent: String::new(),
        attributes: PitchAttributes::default(),
        system_attributes: PitchAttributes::default(),
        extra: Extra::default(),
    });
    static GROUP: std::sync::LazyLock<NoteGroup> = std::sync::LazyLock::new(NoteGroup::default);
    static REFERENCE: std::sync::LazyLock<GroupReference> =
        std::sync::LazyLock::new(GroupReference::default);
    static TEMPO: std::sync::LazyLock<TempoMap> = std::sync::LazyLock::new(TempoMap::default);

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
            note: &SCORE,
            group: &GROUP,
            reference: &REFERENCE,
            main_reference: &REFERENCE,
            tempo_map: &TEMPO,
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
        assert!(
            NeuralBackend::new()
                .dictionaries_for(&voice, &explicit)
                .is_ok()
        );
        // A plain lyric without overrides needs the files.
        let plain = [note(0.0, 0.4, 60, "ka")];
        assert_eq!(
            NeuralBackend::new()
                .dictionaries_for(&voice, &plain)
                .unwrap_err()
                .to_string(),
            "Select the clf-data pronunciation dictionary directory or enter explicit phonemes."
        );
    }

    #[test]
    fn explicit_phrases_skip_dictionary_files_entirely() {
        // A configured-but-missing directory must not fail a phrase whose
        // notes all carry explicit phonemes (C++ returns before its
        // directory check for those notes).
        let dir = std::env::temp_dir().join(format!(
            "opensvr-audio-neural-{}-missing-clf",
            std::process::id()
        ));
        let voice = VoiceSettings {
            dictionary: Some(dir),
            ..VoiceSettings::default()
        };
        let explicit = [crate::TimedNote {
            phonemes: "k a",
            window_start: 0.0,
            window_end: None,
            ..note(0.0, 0.4, 60, "ka")
        }];
        assert!(
            NeuralBackend::new()
                .dictionaries_for(&voice, &explicit)
                .is_ok()
        );
    }

    #[test]
    fn voice_stamps_fail_like_uncached_loads() {
        let missing = std::env::temp_dir().join("opensvr-audio-neural-missing-voice.nofs");
        assert!(!missing.is_file());
        let stamped = stamp_voice(&missing).unwrap_err().to_string();
        let mut synthesizer = VoiceSynthesizer::new();
        let loaded = synthesizer.load(&missing).unwrap_err().to_string();
        assert_eq!(stamped, loaded);
    }

    #[test]
    fn dictionary_stamps_report_missing_files_like_the_loader() {
        let dir = std::env::temp_dir().join(format!(
            "opensvr-audio-neural-{}-stamps",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let stamped = dictionary_stamps(&dir, Language::English)
            .unwrap_err()
            .to_string();
        let loaded = load_dictionaries(&dir, Language::English)
            .unwrap_err()
            .to_string();
        assert_eq!(stamped, loaded);
        assert!(stamped.starts_with("Cannot read synthesis resource: "));
    }

    /// Counting stub around the phrase cache plus the project resampler, in
    /// the same shape as `render_phrase`'s tail: a hit resamples the cached
    /// native audio instead of synthesizing again.
    struct CachedStub {
        cache: PhraseCache,
        renders: usize,
    }

    impl CachedStub {
        fn sing(
            &mut self,
            voice: &FileStamp,
            syllables: &[TimingSyllable],
            log_f0: &[f32],
            native: &[f32],
            project_rate: u32,
        ) -> Vec<f32> {
            let weights: Vec<f32> = Vec::new();
            if let Some((native_rate, cached)) =
                self.cache.lookup(voice, syllables, log_f0, &weights)
            {
                return resample_mono(&cached, native_rate, project_rate).unwrap();
            }
            self.renders += 1;
            self.cache.retain(
                voice.clone(),
                syllables.to_vec(),
                log_f0.to_vec(),
                weights,
                48_000,
                native.to_vec(),
            );
            resample_mono(native, 48_000, project_rate).unwrap()
        }
    }

    #[test]
    fn repeated_phrases_synthesize_once_and_match_fresh_renders() {
        let dir =
            std::env::temp_dir().join(format!("opensvr-audio-neural-{}-stub", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let voice_file = dir.join("voice.nofs");
        std::fs::write(&voice_file, b"voice").unwrap();
        let stamp = FileStamp::read(&voice_file).unwrap();
        let syllables = vec![TimingSyllable {
            language: "japanese".to_owned(),
            phonemes: vec!["k".to_owned(), "a".to_owned()],
            duration_seconds: 0.4,
            midi_pitch: 60,
            is_continuation: false,
        }];
        let log_f0 = vec![4.2_f32; 64];
        let native: Vec<f32> = (0..4800).map(|index| (index as f32 * 0.01).sin()).collect();

        let mut stub = CachedStub {
            cache: PhraseCache::default(),
            renders: 0,
        };
        let first = stub.sing(&stamp, &syllables, &log_f0, &native, 8000);
        let second = stub.sing(&stamp, &syllables, &log_f0, &native, 8000);
        assert_eq!(stub.renders, 1, "the identical phrase must synthesize once");
        assert_eq!(first, second, "cache hits stay sample-identical");
        // Cache off (a fresh backend) resamples the same native audio: equal bits.
        let fresh = resample_mono(&native, 48_000, 8000).unwrap();
        assert_eq!(first, fresh);
        assert_eq!(stub.cache.hits(), 1);
        assert_eq!(stub.cache.misses(), 1);
    }

    #[test]
    fn thirty_second_phrases_are_rejected() {
        let notes = [note(0.0, 31.0, 60, "ka")];
        let backend = NeuralBackend::new();
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
        let backend = NeuralBackend::new();
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
