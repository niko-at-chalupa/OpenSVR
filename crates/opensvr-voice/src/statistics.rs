//! Performance and execution profiling statistics accumulated during synthesis.
//!
//! Ports OpenSV's `src/synthesis/SynthesisStatistics.h` (namespace
//! `sv::synthesis`): frame counts across the pitch, acoustic and vocoder
//! domains, DNNI cache/inference counters shared by every stage, and
//! wall-clock timings per stage.
//!
//! The struct is owned and reset by the caller on the synthesis worker thread
//! and is thread-confined (not thread-safe by itself).
//!
//! Porting note: the Rust [`crate::NeuralVocoder`] does not expose the
//! vocoder sub-stage timings (`vocoder_network/source/residual/filter`), so
//! [`VoiceSynthesizer`](crate::VoiceSynthesizer) only accumulates the total
//! `vocoder_milliseconds` (plus `vocoder_frames`); the four sub-stage fields
//! stay zero until the vocoder port reports them.

/// Profiling counters accumulated across one synthesis pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynthesisStatistics {
    /// Frames evaluated by the pitch prediction model.
    pub pitch_frames: usize,
    /// Acoustic frames evaluated (e.g. at the 5 ms frame rate).
    pub acoustic_frames: usize,
    /// Frames rendered by the vocoder spectral processing.
    pub vocoder_frames: usize,
    /// Neural-network output frames computed from scratch.
    pub dnni_computed_frames: usize,
    /// Frames retrieved from the phrase cache without recomputation.
    pub dnni_reused_frames: usize,
    /// Overlapping receptive-field / context frames required.
    pub dnni_context_frames: usize,
    /// Wall-clock time spent in pitch inference and decoding (ms).
    pub pitch_milliseconds: f64,
    /// Wall-clock time spent in acoustic model inference (ms).
    pub acoustic_milliseconds: f64,
    /// Total wall-clock time spent in neural vocoder synthesis (ms).
    pub vocoder_milliseconds: f64,
    /// Time spent running vocoder neural network layers (ms).
    ///
    /// Not yet reported by the Rust vocoder; always zero (see module docs).
    pub vocoder_network_milliseconds: f64,
    /// Time spent generating periodic (pulse) glottal excitation (ms).
    ///
    /// Not yet reported by the Rust vocoder; always zero (see module docs).
    pub vocoder_source_milliseconds: f64,
    /// Time spent generating aperiodic / noise residual (ms).
    ///
    /// Not yet reported by the Rust vocoder; always zero (see module docs).
    pub vocoder_residual_milliseconds: f64,
    /// Time spent in filter-bank convolution and synthesis (ms).
    ///
    /// Not yet reported by the Rust vocoder; always zero (see module docs).
    pub vocoder_filter_milliseconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_statistics_are_zero() {
        let statistics = SynthesisStatistics::default();
        assert_eq!(statistics.pitch_frames, 0);
        assert_eq!(statistics.acoustic_frames, 0);
        assert_eq!(statistics.vocoder_frames, 0);
        assert_eq!(statistics.dnni_computed_frames, 0);
        assert_eq!(statistics.dnni_reused_frames, 0);
        assert_eq!(statistics.dnni_context_frames, 0);
        assert_eq!(statistics.pitch_milliseconds, 0.0);
        assert_eq!(statistics.acoustic_milliseconds, 0.0);
        assert_eq!(statistics.vocoder_milliseconds, 0.0);
        assert_eq!(statistics.vocoder_network_milliseconds, 0.0);
        assert_eq!(statistics.vocoder_source_milliseconds, 0.0);
        assert_eq!(statistics.vocoder_residual_milliseconds, 0.0);
        assert_eq!(statistics.vocoder_filter_milliseconds, 0.0);
    }
}
