//! NOFS voice database and voice configuration for opensvr.
//!
//! Ports OpenSV's `src/synthesis/VoiceDatabase.*` and `VoiceConfiguration.*`:
//! the archive container (magic `0xf580`, version 10) and the obfuscated
//! configuration table (magic `0xFEFF`) that points at the neural models.

mod config;
mod database;

pub use config::{
    ModelReference, VoiceConfigError, VoiceConfiguration, VoiceConfigurationEntry,
    VoiceConfigurationType,
};
pub use database::{NofsError, VoiceDatabase, VoiceEntry, VoiceMetadata};
