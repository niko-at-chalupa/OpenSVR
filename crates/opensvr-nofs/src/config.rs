//! Obfuscated voice configuration table inside a NOFS archive.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/VoiceConfiguration.{h,cpp}`.
//! The table lives in the NOFS entry with the 1-byte key `0x91` and points at the
//! three neural models (duration, acoustic, vocoder). Entry names are obfuscated
//! with an LCG XOR cipher; error strings match the C++ `juce::Result` text.

use thiserror::Error;

use crate::{NofsError, VoiceDatabase};

/// Errors raised while parsing the voice configuration table.
#[derive(Debug, Error)]
pub enum VoiceConfigError {
    #[error(transparent)]
    Nofs(#[from] NofsError),
    #[error("{0}")]
    Invalid(String),
}

type Result<T> = std::result::Result<T, VoiceConfigError>;

fn invalid(message: impl Into<String>) -> VoiceConfigError {
    VoiceConfigError::Invalid(message.into())
}

const MAXIMUM_CONFIGURATION_BYTES: usize = 1024 * 1024;
const MAXIMUM_ENTRIES: u32 = 16384;
const CONFIG_MAGIC: u32 = 0xfeff;
const CONFIG_ENTRY_KEY: u8 = 0x91;

/// Value types in the configuration block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceConfigurationType {
    Strings = 0,
    Numbers = 1,
    Integers = 2,
}

/// One decoded configuration property.
#[derive(Debug, Clone)]
pub struct VoiceConfigurationEntry {
    pub entry_type: VoiceConfigurationType,
    pub ordinal: u32,
    /// Raw obfuscated key bytes from the file.
    pub key: Vec<u8>,
    /// De-obfuscated UTF-8 property name.
    pub name: String,
    pub strings: Vec<Vec<u8>>,
    pub numbers: Vec<f64>,
    pub integers: Vec<u32>,
}

/// Locates a DNNI model inside the NOFS archive.
#[derive(Debug, Clone, Default)]
pub struct ModelReference {
    /// NOFS entry key of the model file (binary compare).
    pub key: Vec<u8>,
    /// Architecture tag, e.g. `gen2a`, `dds`, `nhv`.
    pub architecture: String,
    /// Decoded key when printable, else empty.
    pub name: String,
}

/// Parsed configuration table with the three model references.
#[derive(Debug, Clone, Default)]
pub struct VoiceConfiguration {
    entries: Vec<VoiceConfigurationEntry>,
    duration: ModelReference,
    acoustic: ModelReference,
    vocoder: ModelReference,
}

impl VoiceConfiguration {
    /// Loads and validates the table from an open [`VoiceDatabase`].
    pub fn load(database: &VoiceDatabase) -> Result<Self> {
        let entry = database
            .entries()
            .iter()
            .find(|entry| entry.key.len() == 1 && entry.key[0] == CONFIG_ENTRY_KEY)
            .ok_or_else(|| {
                invalid("The NOFS database has no supported voice configuration entry.")
            })?;
        if entry.value_size as usize > MAXIMUM_CONFIGURATION_BYTES {
            return Err(invalid("Voice configuration exceeds the 1 MiB limit."));
        }
        let bytes = database.read_entry(entry)?.to_vec();
        let mut replacement = Self::default();
        parse_configuration(&bytes, &mut replacement.entries)?;
        replacement.duration = read_model_reference(
            &replacement.entries,
            database,
            "model_duration",
            "model_duration_arch",
        )?;
        replacement.acoustic = read_model_reference(
            &replacement.entries,
            database,
            "model_timbre_pred",
            "model_timbre_arch",
        )?;
        replacement.vocoder = read_model_reference(
            &replacement.entries,
            database,
            "model_vocoder",
            "model_vocoder_arch",
        )?;
        Ok(replacement)
    }

    pub fn entries(&self) -> &[VoiceConfigurationEntry] {
        &self.entries
    }

    pub fn duration_model(&self) -> &ModelReference {
        &self.duration
    }

    pub fn acoustic_model(&self) -> &ModelReference {
        &self.acoustic
    }

    pub fn vocoder_model(&self) -> &ModelReference {
        &self.vocoder
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn error(&self, detail: impl Into<String>) -> VoiceConfigError {
        invalid(format!(
            "Voice configuration at 0x{:x}: {}",
            self.position,
            detail.into()
        ))
    }

    fn read_u16(&mut self) -> Option<u16> {
        let bytes: [u8; 2] = self.bytes.get(self.position..self.position + 2)?.try_into().ok()?;
        self.position += 2;
        Some(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let bytes: [u8; 4] = self.bytes.get(self.position..self.position + 4)?.try_into().ok()?;
        self.position += 4;
        Some(u32::from_le_bytes(bytes))
    }

    fn read_double(&mut self) -> Option<f64> {
        let bytes: [u8; 8] = self.bytes.get(self.position..self.position + 8)?.try_into().ok()?;
        self.position += 8;
        let value = f64::from_le_bytes(bytes);
        value.is_finite().then_some(value)
    }

    fn read_string(&mut self) -> Option<Vec<u8>> {
        let size = usize::from(self.read_u16()?);
        let value = self.bytes.get(self.position..self.position + size)?.to_vec();
        self.position += size;
        Some(value)
    }
}

/// De-obfuscates an entry name with the LCG XOR cipher from the C++ version.
fn decode_name(bytes: &[u8]) -> Vec<u8> {
    let mut state: u32 = 0xbefc_b5e0;
    bytes
        .iter()
        .map(|&byte| {
            let decoded = byte ^ (state as u8);
            state = state.wrapping_mul(0x41c6_4e6d).wrapping_add(0x3039);
            decoded
        })
        .collect()
}

fn is_text(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && i32::try_from(bytes.len()).is_ok()
        && bytes.iter().all(|&b| b >= 0x20 && b != 0x7f)
        && std::str::from_utf8(bytes).is_ok()
}

fn parse_configuration(bytes: &[u8], entries: &mut Vec<VoiceConfigurationEntry>) -> Result<()> {
    if bytes.len() < 8 || bytes.len() > MAXIMUM_CONFIGURATION_BYTES {
        return Err(invalid(
            "Voice configuration is truncated or exceeds the 1 MiB limit.",
        ));
    }
    let mut reader = Reader { bytes, position: 0 };
    let signature = reader
        .read_u32()
        .ok_or_else(|| reader.error("invalid configuration header."))?;
    let count = reader
        .read_u32()
        .ok_or_else(|| reader.error("invalid configuration header."))?;
    if signature != CONFIG_MAGIC {
        return Err(reader.error("invalid configuration header."));
    }
    if count > MAXIMUM_ENTRIES || count as usize > reader.remaining() / 14 {
        return Err(reader.error("entry count exceeds the remaining data or resource limit."));
    }

    let mut ordinals = std::collections::HashSet::new();
    entries.reserve(count as usize);
    for _ in 0..count {
        let raw_type = reader
            .read_u32()
            .ok_or_else(|| reader.error("truncated entry header."))?;
        let ordinal = reader
            .read_u32()
            .ok_or_else(|| reader.error("truncated entry header."))?;
        let key = reader
            .read_string()
            .ok_or_else(|| reader.error("truncated entry header."))?;
        let value_count = reader
            .read_u32()
            .ok_or_else(|| reader.error("truncated entry header."))? as usize;
        if !ordinals.insert(ordinal) {
            return Err(reader.error("duplicate entry ordinal."));
        }
        let decoded = decode_name(&key);
        let name = String::from_utf8(decoded).ok().filter(|text| is_text(text.as_bytes()));
        let Some(name) = name else {
            // `decode_name` collects bytes as chars; re-check the raw UTF-8 validity
            // the same way the C++ `isText` does (printable ASCII range + valid UTF-8).
            return Err(reader.error("entry name is not valid printable UTF-8."));
        };

        let mut entry = VoiceConfigurationEntry {
            entry_type: VoiceConfigurationType::Strings,
            ordinal,
            key,
            name,
            strings: Vec::new(),
            numbers: Vec::new(),
            integers: Vec::new(),
        };
        match raw_type {
            0 => {
                if value_count > reader.remaining() / 2 {
                    return Err(reader.error("string count exceeds the remaining data."));
                }
                entry.entry_type = VoiceConfigurationType::Strings;
                entry.strings.reserve(value_count);
                for _ in 0..value_count {
                    entry
                        .strings
                        .push(reader.read_string().ok_or_else(|| reader.error("truncated string value."))?);
                }
            }
            1 => {
                if value_count > reader.remaining() / size_of::<f64>() {
                    return Err(reader.error("number count exceeds the remaining data."));
                }
                entry.entry_type = VoiceConfigurationType::Numbers;
                entry.numbers.reserve(value_count);
                for _ in 0..value_count {
                    entry.numbers.push(
                        reader
                            .read_double()
                            .ok_or_else(|| reader.error("truncated or non-finite number value."))?,
                    );
                }
            }
            2 => {
                if value_count > reader.remaining() / size_of::<u32>() {
                    return Err(reader.error("integer count exceeds the remaining data."));
                }
                entry.entry_type = VoiceConfigurationType::Integers;
                entry.integers.reserve(value_count);
                for _ in 0..value_count {
                    entry.integers.push(
                        reader
                            .read_u32()
                            .ok_or_else(|| reader.error("truncated integer value."))?,
                    );
                }
            }
            other => return Err(reader.error(format!("unsupported value type {other}."))),
        }
        entries.push(entry);
    }
    if reader.remaining() != 0 {
        return Err(reader.error("unexpected bytes after the final entry."));
    }
    Ok(())
}

fn find_single_string(entries: &[VoiceConfigurationEntry], name: &str) -> Result<Vec<u8>> {
    let mut found: Option<&VoiceConfigurationEntry> = None;
    for entry in entries {
        if entry.name != name {
            continue;
        }
        if found.is_some() {
            return Err(invalid(format!(
                "Voice configuration has duplicate model field: {name}"
            )));
        }
        found = Some(entry);
    }
    match found {
        Some(entry)
            if entry.entry_type == VoiceConfigurationType::Strings
                && entry.strings.len() == 1
                && !entry.strings[0].is_empty() =>
        {
            Ok(entry.strings[0].clone())
        }
        _ => Err(invalid(format!(
            "Voice configuration requires one non-empty string for: {name}"
        ))),
    }
}

fn read_model_reference(
    entries: &[VoiceConfigurationEntry],
    database: &VoiceDatabase,
    key_field: &str,
    architecture_field: &str,
) -> Result<ModelReference> {
    let key = find_single_string(entries, key_field)?;
    let architecture_bytes = find_single_string(entries, architecture_field)?;
    let architecture = String::from_utf8(architecture_bytes)
        .ok()
        .filter(|text| is_text(text.as_bytes()))
        .ok_or_else(|| invalid("Voice configuration model architecture is not valid UTF-8 text."))?;
    if !database.entries().iter().any(|entry| entry.key == key) {
        return Err(invalid(format!(
            "Voice configuration references a missing model: {key_field}"
        )));
    }
    let name = String::from_utf8(decode_name(&key))
        .ok()
        .filter(|text| is_text(text.as_bytes()))
        .unwrap_or_default();
    Ok(ModelReference {
        key,
        architecture,
        name,
    })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::VoiceDatabase;

    fn encode_name(name: &str) -> Vec<u8> {
        let mut state: u32 = 0xbefc_b5e0;
        name.bytes()
            .map(|byte| {
                let encoded = byte ^ (state as u8);
                state = state.wrapping_mul(0x41c6_4e6d).wrapping_add(0x3039);
                encoded
            })
            .collect()
    }

    fn config_blob(records: &[(u32, u32, &str, ConfigValue)]) -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend(CONFIG_MAGIC.to_le_bytes());
        blob.extend((records.len() as u32).to_le_bytes());
        for (entry_type, ordinal, name, value) in records {
            blob.extend(entry_type.to_le_bytes());
            blob.extend(ordinal.to_le_bytes());
            let key = encode_name(name);
            blob.extend((key.len() as u16).to_le_bytes());
            blob.extend(key);
            match value {
                ConfigValue::Strings(values) => {
                    blob.extend((values.len() as u32).to_le_bytes());
                    for value in *values {
                        blob.extend((value.len() as u16).to_le_bytes());
                        blob.extend(*value);
                    }
                }
                ConfigValue::Numbers(values) => {
                    blob.extend((values.len() as u32).to_le_bytes());
                    for value in *values {
                        blob.extend(value.to_le_bytes());
                    }
                }
                ConfigValue::Integers(values) => {
                    blob.extend((values.len() as u32).to_le_bytes());
                    for value in *values {
                        blob.extend(value.to_le_bytes());
                    }
                }
            }
        }
        blob
    }

    enum ConfigValue<'a> {
        Strings(&'a [&'a [u8]]),
        Numbers(&'a [f64]),
        Integers(&'a [u32]),
    }

    fn write_record(key: &[u8], value: &[u8]) -> Vec<u8> {
        let record_size = (16 + key.len() + value.len()) as u32;
        let mut record = Vec::new();
        record.extend(record_size.to_le_bytes());
        record.extend(1u16.to_le_bytes());
        record.extend((key.len() as u16).to_le_bytes());
        record.extend(key);
        record.extend((value.len() as u32).to_le_bytes());
        record.extend(value);
        record.extend(record_size.to_le_bytes());
        record
    }

    fn database_with_config(blob: &[u8], extra_models: &[(&[u8], &[u8])]) -> VoiceDatabase {
        let mut records: Vec<(Vec<u8>, Vec<u8>)> = vec![(vec![CONFIG_ENTRY_KEY], blob.to_vec())];
        for (key, value) in extra_models {
            records.push((key.to_vec(), value.to_vec()));
        }
        let mut data = Vec::new();
        data.extend(0xf580u32.to_le_bytes());
        data.extend(10u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.extend(10u32.to_le_bytes());
        data.extend(0x1000u16.to_le_bytes());
        data.extend(vec![0u8; 0]);
        data.extend(10u32.to_le_bytes());
        for (key, value) in &records {
            data.extend(write_record(key, value));
        }
        let total = data.len() as u64;
        data[8..16].copy_from_slice(&total.to_le_bytes());
        VoiceDatabase::from_bytes(PathBuf::from("voice.nofs"), data).unwrap()
    }

    fn model_config() -> Vec<u8> {
        config_blob(&[
            (0, 1, "model_duration", ConfigValue::Strings(&[b"dur-key"])),
            (
                0,
                2,
                "model_duration_arch",
                ConfigValue::Strings(&[b"gen2a"]),
            ),
            (
                0,
                3,
                "model_timbre_pred",
                ConfigValue::Strings(&[b"aco-key"]),
            ),
            (
                0,
                4,
                "model_timbre_arch",
                ConfigValue::Strings(&[b"dds"]),
            ),
            (0, 5, "model_vocoder", ConfigValue::Strings(&[b"voc-key"])),
            (
                0,
                6,
                "model_vocoder_arch",
                ConfigValue::Strings(&[b"nhv"]),
            ),
        ])
    }

    #[test]
    fn decodes_names_and_model_references() {
        assert_eq!(decode_name(&encode_name("model_duration")), b"model_duration");
        let blob = model_config();
        let db = database_with_config(
            &blob,
            &[(b"dur-key", b"model-a"), (b"aco-key", b"model-b"), (b"voc-key", b"model-c")],
        );
        let config = VoiceConfiguration::load(&db).unwrap();
        assert_eq!(config.entries().len(), 6);
        assert_eq!(config.duration_model().architecture, "gen2a");
        assert_eq!(config.acoustic_model().architecture, "dds");
        assert_eq!(config.vocoder_model().architecture, "nhv");
        assert_eq!(config.duration_model().key, b"dur-key");
    }

    #[test]
    fn rejects_missing_config_and_models() {
        let mut data = Vec::new();
        data.extend(0xf580u32.to_le_bytes());
        data.extend(10u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.extend(10u32.to_le_bytes());
        data.extend(0x1000u16.to_le_bytes());
        data.extend(10u32.to_le_bytes());
        data.extend(write_record(b".name", b"Voice"));
        let total = data.len() as u64;
        data[8..16].copy_from_slice(&total.to_le_bytes());
        let db = VoiceDatabase::from_bytes(PathBuf::new(), data).unwrap();
        assert_eq!(
            VoiceConfiguration::load(&db).unwrap_err().to_string(),
            "The NOFS database has no supported voice configuration entry."
        );

        let db = database_with_config(&model_config(), &[]);
        assert_eq!(
            VoiceConfiguration::load(&db).unwrap_err().to_string(),
            "Voice configuration references a missing model: model_duration"
        );
    }

    #[test]
    fn rejects_truncated_and_trailing_blobs() {
        for blob in [vec![0u8; 4], {
            let mut blob = model_config();
            blob.push(0);
            blob
        }] {
            let db = database_with_config(&blob, &[(b"dur-key", b"a"), (b"aco-key", b"b"), (b"voc-key", b"c")]);
            assert!(VoiceConfiguration::load(&db).is_err());
        }
        let error = {
            let db = database_with_config(&[0u8; 4], &[]);
            VoiceConfiguration::load(&db).unwrap_err().to_string()
        };
        assert!(error.starts_with("Voice configuration"), "{error}");
    }

    #[test]
    fn numbers_and_integers_round_trip() {
        let blob = config_blob(&[
            (0, 1, "model_duration", ConfigValue::Strings(&[b"dur-key"])),
            (
                0,
                2,
                "model_duration_arch",
                ConfigValue::Strings(&[b"gen2a"]),
            ),
            (
                0,
                3,
                "model_timbre_pred",
                ConfigValue::Strings(&[b"aco-key"]),
            ),
            (
                0,
                4,
                "model_timbre_arch",
                ConfigValue::Strings(&[b"dds"]),
            ),
            (0, 5, "model_vocoder", ConfigValue::Strings(&[b"voc-key"])),
            (
                0,
                6,
                "model_vocoder_arch",
                ConfigValue::Strings(&[b"nhv"]),
            ),
            (1, 7, "threshold", ConfigValue::Numbers(&[0.5, -1.25])),
            (2, 8, "flags", ConfigValue::Integers(&[1, 2, 3])),
        ]);
        let db = database_with_config(
            &blob,
            &[(b"dur-key", b"a"), (b"aco-key", b"b"), (b"voc-key", b"c")],
        );
        let config = VoiceConfiguration::load(&db).unwrap();
        let numbers = config.entries().iter().find(|e| e.name == "threshold").unwrap();
        assert_eq!(numbers.numbers, [0.5, -1.25]);
        let integers = config.entries().iter().find(|e| e.name == "flags").unwrap();
        assert_eq!(integers.integers, [1, 2, 3]);
    }

    #[test]
    fn real_voice_config_loads_when_provided() {
        let Some(path) = std::env::var_os("OPENSVR_TEST_VOICE") else {
            return;
        };
        let db = VoiceDatabase::open(std::path::Path::new(&path)).unwrap();
        let config = VoiceConfiguration::load(&db).unwrap();
        assert!(!config.entries().is_empty());
        assert!(!config.duration_model().architecture.is_empty());
    }
}
