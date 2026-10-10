//! Neural singing-voice stages: the DNNI-backed half of Phase 5.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/PhonemeTiming.*`
//! and `src/synthesis/FeatureNormalizer.*` (namespace `sv::synthesis`). It
//! turns [`TimingSyllable`]s (re-exported from [`opensvr_g2p`], where the
//! types already live) into per-phoneme durations in seconds, then quantizes
//! them into acoustic [`TimedPhoneme`] frames.
//!
//! Error strings reproduce the C++ `juce::Result` failure text verbatim so
//! golden tests can compare messages. Dependency direction follows the
//! workspace rule (`core` <- `dnni`/`g2p` <- `voice`): this crate never
//! touches `opensvr-audio` or the CLI.

mod normalizer;
mod timing;

pub use normalizer::FeatureNormalizer;
pub use opensvr_g2p::{PhonemeDuration, TimingSyllable};
pub use timing::{PhonemeTiming, TimedPhoneme, quantize_phoneme_durations};

use opensvr_dnni::DnniError;
use opensvr_g2p::G2pError;
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
}

pub(crate) type Result<T> = std::result::Result<T, VoiceError>;

pub(crate) fn invalid(message: impl Into<String>) -> VoiceError {
    VoiceError::Invalid(message.into())
}
