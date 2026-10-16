//! Phrase-level voice orchestration (`VoiceSynthesizer`).
//!
//! Ports OpenSV's `src/synthesis/VoiceSynthesizer.{h,cpp}` (namespace
//! `sv::synthesis`): owns the four decoded stage models plus the voice
//! metadata (timbre styles as vocal-mode names, `.feature_rap_languages` as
//! rap gates) and exposes the public phrase API — [`VoiceSynthesizer::predict`],
//! [`VoiceSynthesizer::predict_pitch`] and [`VoiceSynthesizer::render`] — over
//! the ported stages. Orchestration is wiring only; every stage keeps its own
//! numerics and `1e-5` tolerance.
//!
//! Error strings reproduce the C++ `juce::Result` failure text verbatim so
//! golden tests can compare messages. A failed [`VoiceSynthesizer::load`]
//! preserves the previously loaded voice, exactly as in C++.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniReader, DnniRunStatistics, DnniTensor};
use opensvr_nofs::{ModelReference, VoiceConfiguration, VoiceDatabase};

use crate::pitch_features::PitchNote;
use crate::statistics::SynthesisStatistics;
use crate::timing::TimedPhoneme;
use crate::{
    AcousticFeatures, AcousticModel, AcousticModelState, NeuralVocoder, NeuralVocoderOutput,
    NeuralVocoderState, PhonemeDuration, PhonemeTiming, PitchModel, PitchModelState, Result,
    TimingSyllable, VoiceEmbeddings, invalid,
};

/// Silence context padded on each side of the pitch prediction interval.
pub const PITCH_CONTEXT_SECONDS: f64 = 0.5;

/// Default deterministic noise seed for [`VoiceSynthesizer::render`].
pub const DEFAULT_RENDER_SEED: u32 = 5489;

/// Retained inference snapshots for one phrase on the synthesis worker.
#[derive(Debug, Default)]
pub struct VoiceSynthesizerState {
    /// Pitch-model phrase cache.
    pub pitch: PitchModelState,
    /// Acoustic-model phrase caches.
    pub acoustic: AcousticModelState,
    /// Vocoder phrase caches.
    pub vocoder: NeuralVocoderState,
}

impl VoiceSynthesizerState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.pitch.get_bytes() + self.acoustic.get_bytes() + self.vocoder.get_bytes()
    }

    /// Resets all phrase snapshots and branch caches.
    pub fn clear(&mut self) {
        self.pitch.clear();
        self.acoustic.clear();
        self.vocoder.clear();
    }
}

/// Phrase-level singing voice: timing, pitch, acoustic and vocoder stages.
#[derive(Debug)]
pub struct VoiceSynthesizer {
    timing: PhonemeTiming,
    pitch: PitchModel,
    acoustic: AcousticModel,
    vocoder: NeuralVocoder,
    rap_languages: Vec<String>,
    vocal_mode_names: Vec<String>,
    /// Flat document default style (`(default)` embedding): the base the
    /// named vocal modes offset from at render. Empty for container voices,
    /// whose styles live in the acoustic model.
    default_style: Vec<f32>,
    /// Flat document vocal modes (named style vectors). Empty for container
    /// voices. Render-time mixing lands in the final Flat wiring generation.
    vocal_modes: BTreeMap<String, Vec<f32>>,
    frame_interval_seconds: f32,
}

impl VoiceSynthesizer {
    /// Creates an unloaded synthesizer; every method fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self {
            timing: PhonemeTiming::new(),
            pitch: PitchModel::new(),
            acoustic: AcousticModel::new(),
            vocoder: NeuralVocoder::new(),
            rap_languages: Vec::new(),
            vocal_mode_names: Vec::new(),
            default_style: Vec::new(),
            vocal_modes: BTreeMap::new(),
            frame_interval_seconds: 0.0,
        }
    }

    /// Opens the NOFS voice at `path` and decodes all four stage models.
    ///
    /// The pitch model loads from the `f0model-dds` entry; duration, acoustic
    /// and vocoder models load through the voice configuration's model
    /// references (`gen2a`, `dds`, `nhv`). The acoustic frame interval is
    /// scanned from the acoustic entry's `_ftmfv2`/`_ftmfv3` configuration.
    /// Flat document voices additionally decode their speaker/style
    /// embeddings, which override the model-embedded vectors. Everything is
    /// staged into a candidate first: a failed load preserves the previously
    /// loaded voice.
    pub fn load(&mut self, voice: &Path) -> Result<()> {
        self.load_with_models(voice, None)
    }

    /// Opens `voice`, resolving Flat document models from `models` instead of
    /// the document's sibling `models` directory. Container voices ignore it.
    pub fn load_with_models(&mut self, voice: &Path, models: Option<&Path>) -> Result<()> {
        let database = VoiceDatabase::open_with_models(voice, models)?;
        let configuration = VoiceConfiguration::load(&database)?;
        let mut candidate = Self::new();
        let embeddings = match database.document() {
            Some(document) => VoiceEmbeddings::load(document)?,
            None => VoiceEmbeddings::default(),
        };
        candidate.rap_languages = rap_languages(&database);
        candidate
            .default_style
            .clone_from(&embeddings.acoustic_style);
        candidate.vocal_modes = embeddings.vocal_modes.clone();
        if database.has_document() {
            // Document styles live in the embeddings table, not the model.
            candidate.vocal_mode_names = embeddings.vocal_modes.keys().cloned().collect();
        } else {
            candidate
                .vocal_mode_names
                .clone_from(&database.metadata().timbre_styles);
        }
        {
            let Some(entry) = database.find_entry("f0model-dds") else {
                return Err(invalid(
                    "Voice synthesizer: the voice has no supported f0model-dds pitch model",
                ));
            };
            let bytes = database.read_entry(entry)?.to_vec();
            let reader = DnniReader::from_bytes(bytes)?;
            candidate
                .pitch
                .load_with_speaker_override(&reader, 0, &embeddings.pitch_speaker)
                .map_err(|error| invalid(format!("Voice synthesizer pitch model: {error}")))?;
        }
        {
            let reader = read_model(
                &database,
                configuration.duration_model(),
                "gen2a",
                "duration",
            )?;
            candidate
                .timing
                .load_with_voice_override(&reader, 0, &embeddings.duration_voice)?;
        }
        {
            let reader = read_model(&database, configuration.acoustic_model(), "dds", "acoustic")?;
            if database.has_document() {
                candidate.acoustic.load_with_overrides(
                    &reader,
                    0,
                    &embeddings.acoustic_speaker,
                    &embeddings.acoustic_style,
                    &embeddings.vocal_modes,
                )?;
            } else {
                candidate.acoustic.load_with_vocal_modes(
                    &reader,
                    &candidate.vocal_mode_names,
                    0,
                )?;
            }
            candidate.frame_interval_seconds = read_frame_interval(&reader)?;
        }
        {
            let reader = read_model(&database, configuration.vocoder_model(), "nhv", "vocoder")?;
            candidate.vocoder.load(&reader, 0)?;
        }
        *self = candidate;
        Ok(())
    }

    /// Acoustic frame interval in seconds (0.0 until a voice is loaded).
    pub fn frame_interval_seconds(&self) -> f32 {
        self.frame_interval_seconds
    }

    /// Pitch-model frame interval in seconds (0.0 until a voice is loaded).
    pub fn pitch_frame_interval_seconds(&self) -> f32 {
        self.pitch.frame_interval_seconds()
    }

    /// Vocal-mode names from the voice metadata timbre styles (container
    /// voices) or the document embeddings table (Flat voices).
    pub fn vocal_mode_names(&self) -> &[String] {
        &self.vocal_mode_names
    }

    /// Flat document default style vector (`(default)` embedding). Empty for
    /// container voices, whose styles live in the acoustic model.
    pub fn default_style(&self) -> &[f32] {
        &self.default_style
    }

    /// Flat document vocal modes (named style vectors). Empty for container
    /// voices.
    pub fn vocal_modes(&self) -> &BTreeMap<String, Vec<f32>> {
        &self.vocal_modes
    }

    /// Predicts per-phoneme durations for a timing syllable phrase.
    ///
    /// `output` is only assigned on success.
    pub fn predict(
        &self,
        syllables: &[TimingSyllable],
        output: &mut Vec<PhonemeDuration>,
    ) -> Result<()> {
        if self.frame_interval_seconds == 0.0 {
            return Err(invalid("Voice synthesizer: no voice has been loaded"));
        }
        self.timing.predict(syllables, output)
    }

    /// Predicts absolute MIDI pitch per frame from score notes.
    ///
    /// Takes score notes and internal rests, without acoustic preroll or
    /// release: phoneme timing is re-predicted independently in the pitch
    /// model's context. One silence note plus [`PITCH_CONTEXT_SECONDS`] pads
    /// each side; `output` and the optional envelope-style statistics start
    /// that far before the first score note. `output` is only assigned on
    /// success.
    pub fn predict_pitch(
        &self,
        notes: &[PitchNote],
        output: &mut Vec<f32>,
        cancel: &CancelToken,
        mut statistics: Option<&mut SynthesisStatistics>,
        mut state: Option<&mut VoiceSynthesizerState>,
        vibrato_envelope: &[f32],
    ) -> Result<()> {
        if self.frame_interval_seconds == 0.0 || notes.is_empty() {
            return Err(invalid(
                "Pitch prediction requires a loaded voice and score notes.",
            ));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Pitch prediction was cancelled."));
        }
        let started = Instant::now();
        let mut network_statistics = DnniRunStatistics::default();
        let supported_notes = padded_notes(notes, &self.rap_languages);
        let syllables: Vec<TimingSyllable> = supported_notes
            .iter()
            .map(|note| note.syllable.clone())
            .collect();
        // The original F0 task predicts durations in its own half-second
        // context; acoustic timing has different boundary intervals and
        // cannot be reused here.
        let mut phonemes = Vec::new();
        self.timing.predict(&syllables, &mut phonemes)?;
        let pitch_state = state.as_mut().map(|state| &mut state.pitch);
        let result = self.pitch.run(
            &supported_notes,
            &phonemes,
            output,
            0,
            pitch_state,
            cancel,
            Some(&mut network_statistics),
            vibrato_envelope,
        );
        let succeeded = result.is_ok();
        let reused = state
            .as_ref()
            .is_some_and(|state| state.pitch.reused_prediction);
        if let Some(statistics) = statistics.as_mut() {
            statistics.pitch_milliseconds += started.elapsed().as_secs_f64() * 1000.0;
            if succeeded && (state.is_none() || !reused) {
                statistics.pitch_frames += output.len();
            }
            statistics.dnni_computed_frames += network_statistics.computed_frames;
            statistics.dnni_reused_frames += network_statistics.reused_frames;
            statistics.dnni_context_frames += network_statistics.context_frames;
        }
        result
    }

    /// Renders waveform samples from timed phonemes plus the log-F0 curve.
    ///
    /// Runs the acoustic model (diffusion noise from `seed`) then the vocoder
    /// with empty shape control. The cancellation callback runs synchronously
    /// between inference stages and blocks. `output` is only assigned on
    /// success.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        phonemes: &[TimedPhoneme],
        log_f0: &[f32],
        output: &mut NeuralVocoderOutput,
        seed: u32,
        cancel: &CancelToken,
        mut statistics: Option<&mut SynthesisStatistics>,
        mut state: Option<&mut VoiceSynthesizerState>,
        vocal_mode_weights: &[f32],
    ) -> Result<()> {
        let mut network_statistics = DnniRunStatistics::default();
        if self.frame_interval_seconds == 0.0 {
            return Err(invalid("Voice synthesizer: no voice has been loaded"));
        }
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        let mut frames = DnniTensor::default();
        let acoustic_started = Instant::now();
        let acoustic_state = state.as_mut().map(|state| &mut state.acoustic);
        let acoustic_result = self.acoustic.run(
            phonemes,
            log_f0,
            &mut frames,
            seed,
            acoustic_state,
            cancel,
            Some(&mut network_statistics),
            vocal_mode_weights,
        );
        let acoustic_ok = acoustic_result.is_ok();
        let acoustic_frames = if acoustic_ok { frames.frames } else { 0 };
        if let Some(statistics) = statistics.as_mut() {
            statistics.acoustic_milliseconds += acoustic_started.elapsed().as_secs_f64() * 1000.0;
            if acoustic_ok {
                statistics.acoustic_frames += acoustic_frames;
            }
        }
        if let Err(error) = acoustic_result {
            if let Some(statistics) = statistics.as_mut() {
                statistics.dnni_computed_frames += network_statistics.computed_frames;
                statistics.dnni_reused_frames += network_statistics.reused_frames;
                statistics.dnni_context_frames += network_statistics.context_frames;
            }
            return Err(error);
        }
        if cancel.is_cancelled() {
            if let Some(statistics) = statistics.as_mut() {
                statistics.dnni_computed_frames += network_statistics.computed_frames;
                statistics.dnni_reused_frames += network_statistics.reused_frames;
                statistics.dnni_context_frames += network_statistics.context_frames;
            }
            return Err(invalid("Cancelled"));
        }
        let mut candidate = NeuralVocoderOutput::default();
        let vocoder_started = Instant::now();
        let vocoder_state = state.as_mut().map(|state| &mut state.vocoder);
        let vocoder_result = self.vocoder.run(
            &frames,
            &mut candidate,
            &[],
            vocoder_state,
            cancel,
            Some(&mut network_statistics),
        );
        let vocoder_ok = vocoder_result.is_ok();
        if let Some(statistics) = statistics.as_mut() {
            statistics.vocoder_milliseconds += vocoder_started.elapsed().as_secs_f64() * 1000.0;
            if vocoder_ok {
                statistics.vocoder_frames += frames.frames;
            }
        }
        if let Err(error) = vocoder_result {
            if let Some(statistics) = statistics.as_mut() {
                statistics.dnni_computed_frames += network_statistics.computed_frames;
                statistics.dnni_reused_frames += network_statistics.reused_frames;
                statistics.dnni_context_frames += network_statistics.context_frames;
            }
            return Err(error);
        }
        if cancel.is_cancelled() {
            if let Some(statistics) = statistics.as_mut() {
                statistics.dnni_computed_frames += network_statistics.computed_frames;
                statistics.dnni_reused_frames += network_statistics.reused_frames;
                statistics.dnni_context_frames += network_statistics.context_frames;
            }
            return Err(invalid("Cancelled"));
        }
        if let Some(statistics) = statistics.as_mut() {
            statistics.dnni_computed_frames += network_statistics.computed_frames;
            statistics.dnni_reused_frames += network_statistics.reused_frames;
            statistics.dnni_context_frames += network_statistics.context_frames;
        }
        *output = candidate;
        Ok(())
    }
}

impl Default for VoiceSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Splits the `.feature_rap_languages` property like the C++ loader.
///
/// Mirrors `juce::StringArray::fromTokens(text, false)`: tokens split on
/// whitespace; empty tokens never match a real language, so they are dropped.
fn split_rap_languages(text: &str) -> Vec<String> {
    text.split([' ', '\n', '\r', '\t'])
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Gathers the rap languages for a voice: the container
/// `.feature_rap_languages` tokens plus, for Flat document voices, the
/// document's `rap_languages` array when present, else `support_languages`.
/// Mirrors the reference loader, which appends the document list after the
/// container tokens.
fn rap_languages(database: &VoiceDatabase) -> Vec<String> {
    let mut languages = split_rap_languages(database.metadata().property(".feature_rap_languages"));
    let Some(object) = database
        .document()
        .and_then(|document| document.as_object())
    else {
        return languages;
    };
    let key = if object.contains_key("rap_languages") {
        "rap_languages"
    } else {
        "support_languages"
    };
    if let Some(array) = object.get(key).and_then(|value| value.as_array()) {
        languages.extend(
            array
                .iter()
                .filter_map(|value| value.as_str())
                .map(str::to_owned),
        );
    }
    languages
}

/// Loads one stage model through its voice-configuration reference.
///
/// Checks the architecture tag, finds the NOFS entry by raw key bytes, reads
/// it and parses the DNNI blob. Error strings carry the
/// `Voice synthesizer <role> model: ` context, exactly as in C++.
fn read_model(
    database: &VoiceDatabase,
    reference: &ModelReference,
    architecture: &str,
    role: &str,
) -> Result<DnniReader> {
    let context = format!("Voice synthesizer {role} model: ");
    if reference.architecture != architecture {
        return Err(invalid(format!(
            "{context}unsupported architecture '{}', expected '{architecture}'",
            reference.architecture
        )));
    }
    let Some(entry) = database
        .entries()
        .iter()
        .find(|entry| entry.key == reference.key)
    else {
        return Err(invalid(format!(
            "{context}the configured NOFS entry is missing"
        )));
    };
    let bytes = database
        .read_entry(entry)
        .map_err(|error| invalid(format!("{context}{error}")))?
        .to_vec();
    DnniReader::from_bytes(bytes).map_err(|error| invalid(format!("{context}{error}")))
}

/// Scans the acoustic entry for its `_ftmfv2`/`_ftmfv3` feature configuration.
///
/// Exactly one such node must exist; its frame interval becomes the
/// synthesizer's acoustic grid.
fn read_frame_interval(reader: &DnniReader) -> Result<f32> {
    let mut feature_node = None;
    for (index, node) in reader.nodes().iter().enumerate() {
        if node.name != "_ftmfv2" && node.name != "_ftmfv3" {
            continue;
        }
        if feature_node.is_some() {
            return Err(invalid(
                "Voice synthesizer: acoustic model contains multiple feature configurations",
            ));
        }
        feature_node = Some(index);
    }
    let Some(feature_node) = feature_node else {
        return Err(invalid(
            "Voice synthesizer: acoustic model has no supported _ftmfv2/v3 feature configuration",
        ));
    };
    let mut features = AcousticFeatures::new();
    features.load(reader, feature_node)?;
    Ok(features.frame_interval_seconds())
}

/// One silence note padding the pitch prediction interval.
fn silence_note(language: &str, midi_pitch: i32) -> PitchNote {
    PitchNote {
        syllable: TimingSyllable {
            language: language.to_owned(),
            phonemes: vec!["sil".to_owned()],
            duration_seconds: PITCH_CONTEXT_SECONDS,
            midi_pitch,
            is_continuation: false,
        },
        is_breath: false,
        is_silence: true,
        is_continuation: false,
        is_rap: false,
        tone: 0,
        vibrato_modulation: 1.0,
    }
}

/// Pads score notes with one silence note plus context on each side.
///
/// Rap flags are gated by the voice metadata rap languages: a note only keeps
/// `is_rap` when its language is listed. Leading and trailing silence notes
/// never rap.
fn padded_notes(notes: &[PitchNote], rap_languages: &[String]) -> Vec<PitchNote> {
    let mut supported = Vec::with_capacity(notes.len() + 2);
    supported.push(silence_note(
        &notes[0].syllable.language,
        notes[0].syllable.midi_pitch,
    ));
    supported.extend(notes.iter().cloned());
    supported.push(silence_note(
        &notes[notes.len() - 1].syllable.language,
        notes[notes.len() - 1].syllable.midi_pitch,
    ));
    for note in &mut supported {
        note.is_rap = note.is_rap
            && rap_languages
                .iter()
                .any(|language| language == &note.syllable.language);
    }
    supported
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn note(language: &str, midi_pitch: i32, is_rap: bool) -> PitchNote {
        PitchNote {
            syllable: TimingSyllable {
                language: language.to_owned(),
                phonemes: vec!["a".to_owned()],
                duration_seconds: 0.4,
                midi_pitch,
                is_continuation: false,
            },
            is_rap,
            ..PitchNote::default()
        }
    }

    #[test]
    fn fresh_synthesizer_reports_zero_intervals() {
        let synthesizer = VoiceSynthesizer::new();
        assert_eq!(synthesizer.frame_interval_seconds(), 0.0);
        assert_eq!(synthesizer.pitch_frame_interval_seconds(), 0.0);
        assert_eq!(synthesizer.vocal_mode_names().len(), 0);
        assert_eq!(synthesizer.default_style().len(), 0);
        assert!(synthesizer.vocal_modes().is_empty());
        assert_eq!(VoiceSynthesizerState::default().get_bytes(), 0);
        let mut state = VoiceSynthesizerState::default();
        state.clear();
        assert_eq!(state.get_bytes(), 0);
    }

    #[test]
    fn unloaded_synthesizer_rejects_everything() {
        let synthesizer = VoiceSynthesizer::new();
        let mut durations = Vec::new();
        assert_eq!(
            synthesizer
                .predict(&[], &mut durations)
                .unwrap_err()
                .to_string(),
            "Voice synthesizer: no voice has been loaded"
        );
        let mut pitch = Vec::new();
        assert_eq!(
            synthesizer
                .predict_pitch(
                    &[note("japanese", 60, false)],
                    &mut pitch,
                    &CancelToken::new(),
                    None,
                    None,
                    &[]
                )
                .unwrap_err()
                .to_string(),
            "Pitch prediction requires a loaded voice and score notes."
        );
        let mut output = NeuralVocoderOutput::default();
        assert_eq!(
            synthesizer
                .render(
                    &[],
                    &[],
                    &mut output,
                    DEFAULT_RENDER_SEED,
                    &CancelToken::new(),
                    None,
                    None,
                    &[]
                )
                .unwrap_err()
                .to_string(),
            "Voice synthesizer: no voice has been loaded"
        );
    }

    #[test]
    fn predict_pitch_rejects_empty_notes_before_cancel() {
        let synthesizer = VoiceSynthesizer::new();
        let cancel = CancelToken::new();
        cancel.cancel();
        let mut pitch = Vec::new();
        // The loaded-voice/empty-notes check runs before the cancel check.
        assert_eq!(
            synthesizer
                .predict_pitch(&[], &mut pitch, &cancel, None, None, &[])
                .unwrap_err()
                .to_string(),
            "Pitch prediction requires a loaded voice and score notes."
        );
    }

    #[test]
    fn padding_adds_one_silence_note_plus_context_on_each_side() {
        let notes = vec![note("japanese", 60, false), note("japanese", 62, false)];
        let padded = padded_notes(&notes, &[]);
        assert_eq!(padded.len(), 4);
        for (index, expected) in [(0, 60), (1, 60), (2, 62), (3, 62)] {
            assert_eq!(padded[index].syllable.midi_pitch, expected, "note {index}");
        }
        for side in [0, 3] {
            assert!(padded[side].is_silence, "side {side} is silence");
            assert!(!padded[side].is_breath);
            assert!(!padded[side].is_rap);
            assert_eq!(padded[side].syllable.phonemes, vec!["sil".to_owned()]);
            assert_eq!(
                padded[side].syllable.duration_seconds,
                PITCH_CONTEXT_SECONDS
            );
            assert_eq!(padded[side].tone, 0);
            assert_eq!(padded[side].vibrato_modulation, 1.0);
        }
        assert_eq!(padded[0].syllable.language, "japanese");
        assert_eq!(padded[3].syllable.language, "japanese");
        // Score notes pass through untouched.
        assert!(!padded[1].is_silence);
        assert!(!padded[2].is_silence);
        assert_eq!(padded[1].syllable.duration_seconds, 0.4);
    }

    #[test]
    fn rap_gating_keeps_only_listed_languages() {
        let notes = vec![
            note("japanese", 60, true),
            note("english", 62, true),
            note("japanese", 64, false),
        ];
        let gated = padded_notes(&notes, &["japanese".to_owned()]);
        // Indices 1..=3 are the score notes; 0 and 4 are leading/trailing silence.
        assert!(!gated[0].is_rap, "leading silence never raps");
        assert!(gated[1].is_rap, "listed language keeps rap");
        assert!(!gated[2].is_rap, "unlisted language loses rap");
        assert!(!gated[3].is_rap, "non-rap stays non-rap");
        assert!(!gated[4].is_rap, "trailing silence never raps");

        let empty = padded_notes(&notes, &[]);
        assert!(empty[1..=3].iter().all(|note| !note.is_rap));
    }

    #[test]
    fn rap_language_splitting_drops_empty_tokens() {
        assert_eq!(
            split_rap_languages("japanese english"),
            vec!["japanese".to_owned(), "english".to_owned()]
        );
        assert_eq!(split_rap_languages("").len(), 0);
        assert_eq!(
            split_rap_languages("japanese  english"),
            vec!["japanese".to_owned(), "english".to_owned()]
        );
    }

    fn flat_database(document: &serde_json::Value) -> (PathBuf, VoiceDatabase) {
        let dir = std::env::temp_dir().join(format!(
            "opensvr-synth-flat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models_dir = dir.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        for role in ["acoustic", "vocoder", "duration", "f0"] {
            let id = document[role].as_str().unwrap();
            std::fs::write(models_dir.join(format!("{id}.dnni")), b"model").unwrap();
        }
        let doc = dir.join("info.nofs");
        std::fs::write(&doc, serde_json::to_string(&document).unwrap()).unwrap();
        let database = VoiceDatabase::open(&doc).unwrap();
        (dir, database)
    }

    fn flat_document() -> serde_json::Value {
        serde_json::json!({
            "acoustic": "aco",
            "vocoder": "voc",
            "duration": "dur",
            "f0": "f0",
            "support_languages": ["japanese"],
        })
    }

    #[test]
    fn rap_languages_prefer_the_document_list() {
        // `rap_languages` wins over `support_languages` when present.
        let mut document = flat_document();
        document["rap_languages"] = serde_json::json!(["english"]);
        let (dir, database) = flat_database(&document);
        assert_eq!(rap_languages(&database), vec!["english".to_owned()]);
        std::fs::remove_dir_all(&dir).unwrap();

        // Otherwise the document falls back to `support_languages`.
        let (dir, database) = flat_database(&flat_document());
        assert_eq!(rap_languages(&database), vec!["japanese".to_owned()]);
        std::fs::remove_dir_all(&dir).unwrap();

        // Container voices keep the `.feature_rap_languages` tokens (empty here).
        assert_eq!(rap_languages(&VoiceDatabase::default()).len(), 0);
    }

    #[test]
    fn models_override_dir_wins_over_the_sibling_directory() {
        let dir = std::env::temp_dir().join(format!(
            "opensvr-synth-models-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let document = flat_document();
        let doc = dir.join("info.nofs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&doc, serde_json::to_string(&document).unwrap()).unwrap();
        // No sibling `models` directory: loading without an override names it.
        let missing = VoiceSynthesizer::new().load(&doc).unwrap_err().to_string();
        assert!(
            missing.starts_with("NOFS: model file is missing: "),
            "{missing}"
        );

        // With an override holding (invalid) model bytes, opening consults
        // the override: the failure moves past lookup to DNNI parsing.
        let elsewhere = dir.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        for role in ["acoustic", "vocoder", "duration", "f0"] {
            let id = document[role].as_str().unwrap();
            std::fs::write(elsewhere.join(format!("{id}.dnni")), b"model").unwrap();
        }
        let mut synthesizer = VoiceSynthesizer::new();
        let error = synthesizer
            .load_with_models(&doc, Some(&elsewhere))
            .unwrap_err()
            .to_string();
        // `DnniReader::from_bytes` fails unprefixed (as on the container
        // path); the point is the failure moved past model lookup.
        assert!(!error.contains("model file is missing"), "{error}");
        assert!(error.contains("DNNI"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn real_flat_embeddings_decode_when_provided() {
        let Some(path) = std::env::var_os("OPENSVR_TEST_FLAT") else {
            return;
        };
        let path = std::path::Path::new(&path);
        let database = VoiceDatabase::open(path).unwrap();
        assert!(database.has_document());
        let document = database.document().unwrap();
        let embeddings = VoiceEmbeddings::load(document).unwrap();
        // The document overrides every stage vector; all decode finite.
        for vector in [
            &embeddings.acoustic_speaker,
            &embeddings.acoustic_style,
            &embeddings.pitch_speaker,
            &embeddings.duration_voice,
        ] {
            assert!(!vector.is_empty());
            assert!(vector.iter().all(|value| value.is_finite()));
        }
        assert_eq!(embeddings.acoustic_speaker.len(), 32);
        assert_eq!(embeddings.acoustic_style.len(), 32);
        assert_eq!(embeddings.pitch_speaker.len(), 32);
        assert_eq!(embeddings.duration_voice.len(), 128);
        assert!(!embeddings.vocal_modes.is_empty());
        // No `rap_languages` field: the support list gates rap notes.
        assert_eq!(rap_languages(&database), database.metadata().languages,);
        assert!(!rap_languages(&database).is_empty());
    }

    // --- Minimal v2 blob builders for the interval-scan error paths ---

    const FNV_PRIME_64: u64 = 0x0100_0000_01b3;
    const CHILD_XOR_MASK: u32 = 0xac5e_7bd5;

    fn fnv(seed: u64, name: &str) -> u64 {
        let mut hash = seed;
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME_64);
        }
        hash
    }

    fn named_id(name: &str) -> u64 {
        fnv(0x0dcd_5918_9d5a_0f24, name)
    }

    fn v2_node(type_id: u64, payload: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
        let low = type_id as u32;
        let high = (type_id >> 32) as u32;
        let encoded = (children.len() as u32).wrapping_mul(3) ^ low ^ high ^ CHILD_XOR_MASK;
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_41ff_u32.to_le_bytes());
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&encoded.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        for child in children {
            out.extend_from_slice(child);
        }
        out
    }

    fn v2_file(root: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        out.extend_from_slice(&2_u32.to_le_bytes());
        out.extend_from_slice(root);
        out
    }

    #[test]
    fn interval_scan_rejects_a_missing_configuration() {
        let root = v2_node(named_id("modl0"), &[], &[]);
        let reader = DnniReader::from_bytes(v2_file(&root)).expect("synthetic blob must parse");
        assert_eq!(
            read_frame_interval(&reader).unwrap_err().to_string(),
            "Voice synthesizer: acoustic model has no supported _ftmfv2/v3 feature configuration"
        );
    }

    #[test]
    fn interval_scan_rejects_multiple_configurations() {
        let first = v2_node(named_id("_ftmfv2"), &[], &[]);
        let second = v2_node(named_id("_ftmfv3"), &[], &[]);
        let root = v2_node(named_id("modm0"), &[], &[first, second]);
        let reader = DnniReader::from_bytes(v2_file(&root)).expect("synthetic blob must parse");
        assert_eq!(
            reader
                .nodes()
                .iter()
                .filter(|node| node.name == "_ftmfv2" || node.name == "_ftmfv3")
                .count(),
            2
        );
        assert_eq!(
            read_frame_interval(&reader).unwrap_err().to_string(),
            "Voice synthesizer: acoustic model contains multiple feature configurations"
        );
    }
}
