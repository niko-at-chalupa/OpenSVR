//! Timing-model input/output types (no model yet).
//!
//! Ports the `TimingSyllable` and `PhonemeDuration` structs from OpenSV's
//! `src/synthesis/PhonemeTiming.h`. Duration prediction depends on DNNI
//! inference, so it lands in Phase 5 with golden tensors from the C++ build;
//! until then these types let the renderer-facing code name syllables and
//! per-phoneme durations.

/// One musical syllable offered to the timing prediction model.
#[derive(Debug, Clone, PartialEq)]
pub struct TimingSyllable {
    /// Language identifier (e.g. `"japanese"`, `"english"`, `"mandarin"`).
    pub language: String,
    /// Phoneme sequence belonging to this syllable.
    pub phonemes: Vec<String>,
    /// Musical duration of the note in seconds (from the tempo map).
    pub duration_seconds: f64,
    /// Musical pitch (MIDI note number).
    pub midi_pitch: i32,
    /// True for a slurred continuation of a previous lyric.
    pub is_continuation: bool,
}

/// Duration assigned to one phoneme by the neural timing model.
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeDuration {
    /// Language of this phoneme.
    pub language: String,
    /// Phoneme token (e.g. `"k"`, `"aa"`, `"sil"`).
    pub symbol: String,
    /// Zero-based index of the parent syllable.
    pub syllable_index: usize,
    /// Index of the timing alignment interval.
    pub timing_interval_index: usize,
    /// Predicted duration in seconds (before frame quantization).
    pub duration_seconds: f64,
}
