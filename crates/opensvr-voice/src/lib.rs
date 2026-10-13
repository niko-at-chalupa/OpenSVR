//! Neural singing-voice stages: the DNNI-backed half of Phase 5.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/PhonemeTiming.*`,
//! `src/synthesis/FeatureNormalizer.*`, `src/synthesis/Pitch*.{h,cpp}`,
//! `src/synthesis/Acoustic*.{h,cpp}` and `src/synthesis/{NeuralVocoder,
//! VocoderFilterBank,VocoderResidual,VocoderSpectralHeads,
//! PeriodicExcitation,LipRadiation}.{h,cpp}` (namespace `sv::synthesis`),
//! plus the `VoiceSynthesizer` orchestrator and `SynthesisStatistics` counters.
//! Timing turns [`TimingSyllable`]s
//! (re-exported from [`opensvr_g2p`], where the
//! types already live) into per-phoneme durations in seconds, then quantizes
//! them into acoustic [`TimedPhoneme`] frames; the pitch stage predicts the
//! automatic per-frame F0 curve from score [`PitchNote`]s plus phoneme
//! durations; the acoustic stage predicts vocoder-ready frames from timed
//! phonemes plus the log-F0 curve; the vocoder stage renders waveform
//! samples from the acoustic frames; the synthesizer loads all four stages
//! from a NOFS voice and wires `predict` / `predict_pitch` / `render`.
//!
//! Error strings reproduce the C++ `juce::Result` failure text verbatim so
//! golden tests can compare messages. Dependency direction follows the
//! workspace rule (`core` <- `nofs`/`dnni`/`g2p` <- `voice`): this crate never
//! touches `opensvr-audio` or the CLI.

mod acoustic_features;
mod acoustic_model;
mod excitation;
mod fft;
mod filter_bank;
mod lip_radiation;
mod noise;
mod normalizer;
mod pitch_context;
mod pitch_curve;
mod pitch_decoder;
mod pitch_features;
mod pitch_model;
mod pitch_postprocess;
mod residual;
mod spectral_heads;
mod statistics;
mod synthesizer;
mod timing;
mod vocoder;

pub use acoustic_features::{AcousticFeatureConfig, AcousticFeatures};
pub use acoustic_model::{AcousticFrameInterval, AcousticModel, AcousticModelState};
pub use excitation::{
    PeriodicExcitationFrame, render_periodic_excitation, render_periodic_modulation,
};
pub use filter_bank::VocoderFilterBank;
pub use lip_radiation::{
    LipRadiationResponse, apply_lip_radiation_to_harmonics, lip_radiation_response,
};
pub use normalizer::FeatureNormalizer;
pub use opensvr_g2p::{PhonemeDuration, TimingSyllable};
pub use pitch_context::{PitchContext, PitchContextState};
pub use pitch_curve::{
    PitchCurveNote, midi_to_log_f0, render_automatic_pitch_mask, render_pitch_prediction_base,
    render_pitch_transitions, render_pitch_vibrato,
};
pub use pitch_decoder::{PitchDecoder, PitchDecoderNotes};
pub use pitch_features::{PitchFeatureOutput, PitchFeatures, PitchNote};
pub use pitch_model::{PitchModel, PitchModelState};
pub use pitch_postprocess::{PitchFrameInterval, apply_pitch_postprocess};
pub use residual::{VocoderResidual, VocoderResidualState};
pub use spectral_heads::{VocoderSpectralHeads, VocoderSpectralHeadsState};
pub use statistics::SynthesisStatistics;
pub use synthesizer::{
    DEFAULT_RENDER_SEED, PITCH_CONTEXT_SECONDS, VoiceSynthesizer, VoiceSynthesizerState,
};
pub use timing::{PhonemeTiming, TimedPhoneme, quantize_phoneme_durations};
pub use vocoder::{NeuralVocoder, NeuralVocoderOutput, NeuralVocoderState};

use opensvr_dnni::DnniError;
use opensvr_g2p::G2pError;
use opensvr_nofs::{NofsError, VoiceConfigError};
use thiserror::Error;

/// Errors raised while loading or running the voice stages.
///
/// The `Invalid` payloads reproduce the C++ `juce::Result` failure strings
/// verbatim (including the `Phoneme timing: ` prefix of the timing stage and
/// the unprefixed `Feature normalization ...` / `Phoneme ... _psv2 ...` /
/// `DNNI ...` strings propagated from the lower crates).
#[derive(Debug, Error)]
pub enum VoiceError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Dnni(#[from] DnniError),
    #[error(transparent)]
    G2p(#[from] G2pError),
    #[error(transparent)]
    Nofs(#[from] NofsError),
    #[error(transparent)]
    VoiceConfig(#[from] VoiceConfigError),
}

pub(crate) type Result<T> = std::result::Result<T, VoiceError>;

pub(crate) fn invalid(message: impl Into<String>) -> VoiceError {
    VoiceError::Invalid(message.into())
}

/// Reborrows an optional DNNI statistics accumulator for sequential network
/// calls inside one `run` (each call needs its own `Option<&mut _>`).
pub(crate) fn reborrow_stats<'a>(
    statistics: &'a mut Option<&mut opensvr_dnni::DnniRunStatistics>,
) -> Option<&'a mut opensvr_dnni::DnniRunStatistics> {
    match statistics {
        Some(inner) => Some(&mut **inner),
        None => None,
    }
}
