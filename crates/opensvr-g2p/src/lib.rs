//! Grapheme-to-phoneme pronunciation for opensvr: the DNNI-free half of Phase 3.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/PhoneSet.*`,
//! `src/synthesis/PhonemeDictionary.*` and
//! `src/audio/ProjectRenderer.cpp::resolvePhonemes` (plus the `+`/`-`
//! lyric sequencing around it). It turns note lyrics into phoneme symbols:
//! explicit overrides win, otherwise text dictionaries in a `clf-data`
//! directory are consulted per track language.
//!
//! File I/O lives at the crate edge (`load` functions taking paths); parsing
//! and resolution are pure functions over bytes and `&str` so tests can use
//! synthetic tables. The neural duration model (`PhonemeTiming`) is not part
//! of this crate: [`timing`] exposes only its input/output types until
//! Phase 5, which needs golden tensors from the C++ build first.

mod dictionary;
mod phoneset;
mod resolve;
mod timing;

pub use dictionary::{PhonemeDefinition, PhonemeDictionary};
pub use phoneset::{PhoneSet, read_phone_set};
pub use resolve::{
    DictionarySet, NoteResolver, Resolved, is_english_vowel_phoneme, load_dictionaries, resolve,
    split_english_syllables,
};
pub use timing::{PhonemeDuration, TimingSyllable};

use std::path::PathBuf;

use thiserror::Error;

/// Errors raised while loading dictionaries or resolving phonemes.
///
/// Variants that originate in OpenSV reproduce the C++ `juce::Result` failure
/// strings verbatim so golden tests can compare messages. Renderer-level
/// context (`Track 'x': note 'y' at ...`) is added by the caller, not here.
#[derive(Debug, Error)]
pub enum G2pError {
    #[error("{0}")]
    Invalid(String),
    #[error("could not read {path}: {source}", path = .path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub(crate) type Result<T> = std::result::Result<T, G2pError>;

pub(crate) fn invalid(message: impl Into<String>) -> G2pError {
    G2pError::Invalid(message.into())
}

pub(crate) fn line_error(path: &std::path::Path, line: usize, message: &str) -> G2pError {
    invalid(format!("{}:{line}: {message}", path.display()))
}

#[cfg(test)]
pub(crate) fn temp_dir_for_tests(name: &str) -> std::path::PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "opensvr-g2p-{name}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).expect("test temp dir creates");
    dir
}
