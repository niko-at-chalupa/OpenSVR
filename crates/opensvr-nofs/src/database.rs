//! Reader for Synthesizer V's NOFS voice database format.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/VoiceDatabase.{h,cpp}`
//! (namespace `sv::synthesis`). Layout, limits and error strings match the C++
//! version so golden tests can compare messages.
//!
//! File layout (all integers little-endian):
//! - 16-byte header: magic `0xf580` (u32), version `10` (u32),
//!   total file length (u64) matching the physical size.
//! - Index block at offset 16: total size (u32, including the 4-byte trailer),
//!   type `0x1000` (u16); the last 4 bytes repeat the size. Contents are ignored.
//! - Contiguous value records: size (u32), type `1` (u16), key length (u16),
//!   key bytes, value length (u32), payload, 4-byte trailer repeating the size.

use std::{
    collections::HashSet,
    fs,
    io,
    path::{Path, PathBuf},
};

use thiserror::Error;

/// Errors raised while opening or reading a NOFS archive.
#[derive(Debug, Error)]
pub enum NofsError {
    #[error("could not read voice database {}: {source}", .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("NOFS: {0}")]
    Invalid(String),
}

type Result<T> = std::result::Result<T, NofsError>;

fn invalid(message: impl Into<String>) -> NofsError {
    NofsError::Invalid(message.into())
}

const NOFS_MAGIC: u32 = 0xf580;
const SUPPORTED_VERSION: u32 = 10;
const INDEX_BLOCK_TYPE: u16 = 0x1000;
const VALUE_BLOCK_TYPE: u16 = 1;
const FILE_HEADER_SIZE: u64 = 16;
const MAXIMUM_ENTRY_SIZE: u32 = 512 * 1024 * 1024;
const MAXIMUM_METADATA_SIZE: u32 = 1024 * 1024;
const MAXIMUM_TOTAL_KEY_SIZE: u64 = 16 * 1024 * 1024;
const MAXIMUM_TOTAL_METADATA_SIZE: u64 = 16 * 1024 * 1024;
const MAXIMUM_ENTRY_COUNT: usize = 100_000;

/// One file or resource entry stored in the archive.
#[derive(Debug, Clone)]
pub struct VoiceEntry {
    /// Raw key bytes; may be printable text or arbitrary binary.
    pub key: Vec<u8>,
    /// UTF-8 decoding of [`VoiceEntry::key`] when it is printable text, else empty.
    pub name: String,
    /// Absolute byte offset of the payload in the file.
    pub value_offset: u64,
    /// Payload size in bytes.
    pub value_size: u32,
}

/// Singer and voice metadata extracted from dot-prefixed entries.
#[derive(Debug, Clone, Default)]
pub struct VoiceMetadata {
    pub name: String,
    pub vendor: String,
    pub version: i32,
    pub language: String,
    pub phoneset: String,
    pub voice_type: String,
    pub languages: Vec<String>,
    pub timbre_styles: Vec<String>,
    /// All dot-prefixed entries in file order, as `(key, text)` pairs.
    pub properties: Vec<(String, String)>,
}

impl Default for VoiceDatabase {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            data: Vec::new(),
            entries: Vec::new(),
            metadata: VoiceMetadata {
                version: -1,
                ..VoiceMetadata::default()
            },
        }
    }
}

impl VoiceMetadata {
    fn empty() -> Self {
        Self {
            version: -1,
            ..Self::default()
        }
    }

    /// Value of a dot-prefixed property, or `""` when absent.
    /// Mirrors `juce::StringPairArray::operator[]`.
    pub fn property(&self, name: &str) -> &str {
        self.properties
            .iter()
            .find(|(key, _)| key == name)
            .map_or("", |(_, value)| value.as_str())
    }
}

/// An open NOFS archive: the raw bytes plus the parsed entry table.
///
/// Entry payloads are borrowed from the in-memory file image, so large voices
/// are read once and never copied per entry.
#[derive(Debug)]
pub struct VoiceDatabase {
    path: PathBuf,
    data: Vec<u8>,
    entries: Vec<VoiceEntry>,
    metadata: VoiceMetadata,
}

impl VoiceDatabase {
    /// Opens and validates `path`, parsing the entry table and metadata.
    ///
    /// On failure no partial state is kept; the error message matches the C++
    /// `juce::Result` text (without re-reading: the caller gets `Err` only).
    pub fn open(path: &Path) -> Result<Self> {
        let data = fs::read(path).map_err(|source| NofsError::Io {
            path: path.to_owned(),
            source,
        })?;
        let (entries, metadata) = parse(&data)?;
        Ok(Self {
            path: path.to_owned(),
            data,
            entries,
            metadata,
        })
    }

    /// Parses an in-memory NOFS image. The database owns a copy of `data`.
    pub fn from_bytes(path: PathBuf, data: Vec<u8>) -> Result<Self> {
        let (entries, metadata) = parse(&data)?;
        Ok(Self {
            path,
            data,
            entries,
            metadata,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn entries(&self) -> &[VoiceEntry] {
        &self.entries
    }

    pub fn metadata(&self) -> &VoiceMetadata {
        &self.metadata
    }

    /// Finds the first entry whose printable [`VoiceEntry::name`] equals `name`.
    pub fn find_entry(&self, name: &str) -> Option<&VoiceEntry> {
        self.entries
            .iter()
            .find(|entry| !entry.name.is_empty() && entry.name == name)
    }

    /// Borrows an entry's payload. The entry must belong to this database
    /// (same key bytes, offset and size), mirroring the C++ membership check.
    pub fn read_entry<'a>(&'a self, entry: &VoiceEntry) -> Result<&'a [u8]> {
        let owned = self.entries.iter().any(|existing| {
            existing.key == entry.key
                && existing.value_offset == entry.value_offset
                && existing.value_size == entry.value_size
        });
        if !owned {
            return Err(invalid("entry does not belong to the open database"));
        }
        if entry.value_size > MAXIMUM_ENTRY_SIZE {
            return Err(invalid("entry exceeds the 512 MiB read limit"));
        }
        let start = entry.value_offset as usize;
        let len = entry.value_size as usize;
        self.data
            .get(start..start.saturating_add(len))
            .filter(|slice| slice.len() == len)
            .ok_or_else(|| invalid("cannot read entry value"))
    }
}

#[allow(clippy::too_many_lines)]
fn parse(data: &[u8]) -> Result<(Vec<VoiceEntry>, VoiceMetadata)> {
    let file_size = data.len() as u64;
    if data.len() < (FILE_HEADER_SIZE + 10) as usize {
        return Err(invalid("truncated file header"));
    }
    if read_u32(data, 0) != NOFS_MAGIC {
        return Err(invalid("unrecognised file signature"));
    }
    if read_u32(data, 4) != SUPPORTED_VERSION {
        return Err(invalid("unsupported container version"));
    }
    if read_u64(data, 8) != file_size {
        return Err(invalid("declared file length does not match the file"));
    }

    let Some(index_probe) = data.get(16..24) else {
        return Err(invalid("cannot read index block"));
    };
    let _ = index_probe;
    let index_size = u64::from(read_u32(data, 16));
    let index_type = read_u16(data, 20);
    if index_size < 10 || index_size > file_size - FILE_HEADER_SIZE || index_type != INDEX_BLOCK_TYPE
    {
        return Err(invalid("invalid or unsupported index block"));
    }
    let trailer_at = (FILE_HEADER_SIZE + index_size - 4) as usize;
    let Some(trailer) = data.get(trailer_at..trailer_at.saturating_add(4)) else {
        return Err(invalid("index block trailer mismatch"));
    };
    if u64::from(u32::from_le_bytes(trailer[..4].try_into().unwrap())) != index_size {
        return Err(invalid("index block trailer mismatch"));
    }

    let mut entries = Vec::new();
    let mut keys = HashSet::new();
    let mut total_key_size: u64 = 0;
    let mut offset = FILE_HEADER_SIZE + index_size;
    while offset < file_size {
        if entries.len() >= MAXIMUM_ENTRY_COUNT
            || file_size - offset < 16
            || (offset + 8) as usize > data.len()
        {
            return Err(invalid("too many entries or truncated value block header"));
        }
        let at = offset as usize;
        let record_size = u64::from(read_u32(data, at));
        let record_type = read_u16(data, at + 4);
        let key_size = u64::from(read_u16(data, at + 6));
        if record_type != VALUE_BLOCK_TYPE {
            return Err(invalid(format!(
                "unsupported value block type at {offset}"
            )));
        }
        if key_size == 0 || record_size < 16 + key_size || record_size > file_size - offset {
            return Err(invalid("invalid value block size"));
        }
        total_key_size += key_size;
        if total_key_size > MAXIMUM_TOTAL_KEY_SIZE {
            return Err(invalid("entry keys exceed the 16 MiB read limit"));
        }
        let key_start = at + 8;
        let key_end = key_start + key_size as usize;
        let Some(key) = data.get(key_start..key_end) else {
            return Err(invalid("cannot read entry key"));
        };
        if !keys.insert(key.to_vec()) {
            return Err(invalid("duplicate entry key"));
        }
        let name = if is_text(key, false) {
            String::from_utf8(key.to_vec()).unwrap_or_default()
        } else {
            String::new()
        };
        let value_len_at = key_end;
        let Some(len_bytes) = data.get(value_len_at..value_len_at.saturating_add(4)) else {
            return Err(invalid("cannot read entry value length"));
        };
        if len_bytes.len() != 4 {
            return Err(invalid("cannot read entry value length"));
        }
        let value_size = u32::from_le_bytes(len_bytes.try_into().unwrap());
        let value_offset = offset + 12 + key_size;
        if u64::from(value_size) + key_size + 16 != record_size {
            return Err(invalid("value length does not match its block"));
        }
        let trailer_at = (offset + record_size - 4) as usize;
        let Some(trailer) = data.get(trailer_at..trailer_at.saturating_add(4)) else {
            return Err(invalid("value block trailer mismatch"));
        };
        if u64::from(u32::from_le_bytes(trailer[..4].try_into().unwrap())) != record_size {
            return Err(invalid("value block trailer mismatch"));
        }
        entries.push(VoiceEntry {
            key: key.to_vec(),
            name,
            value_offset,
            value_size,
        });
        offset += record_size;
    }
    if entries.is_empty() {
        return Err(invalid("container has no entries"));
    }

    // Metadata must validate too; a failure invalidates the whole open.
    let metadata = read_metadata(data, &entries)?;
    Ok((entries, metadata))
}

fn read_metadata(data: &[u8], entries: &[VoiceEntry]) -> Result<VoiceMetadata> {
    let mut metadata = VoiceMetadata::empty();
    let mut total: u64 = 0;
    for entry in entries {
        if !entry.name.starts_with('.') {
            continue;
        }
        if entry.value_size > MAXIMUM_METADATA_SIZE {
            return Err(invalid(format!(
                "metadata entry is too large: {}",
                entry.name
            )));
        }
        total += u64::from(entry.value_size);
        if total > MAXIMUM_TOTAL_METADATA_SIZE {
            return Err(invalid("metadata exceeds the 16 MiB read limit"));
        }
        let start = entry.value_offset as usize;
        let len = entry.value_size as usize;
        let Some(value) = data.get(start..start.saturating_add(len)) else {
            return Err(invalid("cannot read entry value"));
        };
        if value.len() != len || !is_text(value, true) {
            return Err(invalid(format!(
                "metadata is not valid text: {}",
                entry.name
            )));
        }
        let text = String::from_utf8(value.to_vec()).unwrap_or_default();
        metadata.properties.push((entry.name.clone(), text));
    }

    metadata.name = metadata.property(".name").to_owned();
    metadata.vendor = metadata.property(".vendor").to_owned();
    metadata.language = metadata.property(".language").to_owned();
    metadata.phoneset = metadata.property(".phoneset").to_owned();
    metadata.voice_type = metadata.property(".type").to_owned();
    metadata.languages = split_tokens(metadata.property(".multi"));
    metadata.timbre_styles = split_tokens(metadata.property(".timbre_styles"));

    let version = metadata.property(".version");
    if !version.is_empty() {
        let digits = !version.is_empty()
            && version.len() <= 9
            && version.bytes().all(|b| b.is_ascii_digit());
        if !digits {
            return Err(invalid("invalid database version metadata"));
        }
        metadata.version = version.parse().unwrap_or(-1);
    }
    Ok(metadata)
}

/// Splits like `juce::StringArray::addTokens(text, false)`: on `space \n \r \t`,
/// keeping empty tokens between consecutive separators.
fn split_tokens(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' {
            tokens.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    tokens.push(current);
    tokens
}

/// Whether raw bytes are printable UTF-8 text.
/// Empty counts as text; NUL, DEL and other C0 controls (except `\n \r \t`
/// when `allow_whitespace`) disqualify, mirroring the C++ `isText`.
fn is_text(bytes: &[u8], allow_whitespace: bool) -> bool {
    if bytes.is_empty() {
        return true;
    }
    if bytes.len() > i32::MAX as usize || std::str::from_utf8(bytes).is_err() {
        return false;
    }
    bytes.iter().all(|&byte| {
        byte != 0
            && byte != 0x7f
            && (byte >= 0x20
                || (allow_whitespace && (byte == b'\n' || byte == b'\r' || byte == b'\t')))
    })
}

fn read_u16(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(data[at..at + 2].try_into().unwrap())
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

fn read_u64(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(data[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn write_record(key: &[u8], value: &[u8]) -> Vec<u8> {
        let record_size = (16 + key.len() + value.len()) as u32;
        let mut record = Vec::new();
        record.extend(record_size.to_le_bytes());
        record.extend(VALUE_BLOCK_TYPE.to_le_bytes());
        record.extend((key.len() as u16).to_le_bytes());
        record.extend(key);
        record.extend((value.len() as u32).to_le_bytes());
        record.extend(value);
        record.extend(record_size.to_le_bytes());
        record
    }

    pub(crate) fn archive(records: &[(&[u8], &[u8])], index_size: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(NOFS_MAGIC.to_le_bytes());
        data.extend(SUPPORTED_VERSION.to_le_bytes());
        data.extend(0u64.to_le_bytes()); // patched below
        data.extend(index_size.to_le_bytes());
        data.extend(INDEX_BLOCK_TYPE.to_le_bytes());
        data.extend(vec![0u8; (index_size - 10) as usize]);
        data.extend(index_size.to_le_bytes());
        for (key, value) in records {
            data.extend(write_record(key, value));
        }
        let total = data.len() as u64;
        data[8..16].copy_from_slice(&total.to_le_bytes());
        data
    }

    fn open_bytes(records: &[(&[u8], &[u8])]) -> VoiceDatabase {
        let data = archive(records, 10);
        VoiceDatabase::from_bytes(PathBuf::from("test.nofs"), data).unwrap()
    }

    #[test]
    fn parses_entries_and_metadata() {
        let db = open_bytes(&[
            (b".name", b"Kasane Teto"),
            (b".version", b"104"),
            (b".multi", b"japanese english"),
            (b".timbre_styles", b"Power Soft"),
            (b"samples", b"\x00\x01\x02"),
        ]);
        assert_eq!(db.entries().len(), 5);
        assert_eq!(db.metadata().name, "Kasane Teto");
        assert_eq!(db.metadata().version, 104);
        assert_eq!(db.metadata().languages, ["japanese", "english"]);
        assert_eq!(db.metadata().timbre_styles, ["Power", "Soft"]);
        assert_eq!(db.find_entry(".name").unwrap().value_size, 11);

        // Binary keys get no printable name and are skipped by find_entry.
        assert!(db.find_entry("samples").is_some());
        let binary = open_bytes(&[(b"\xff\x00binary", b"payload")]);
        assert!(binary.entries()[0].name.is_empty());
        assert!(binary.find_entry("").is_none());

        let payload = {
            let entry = db.find_entry("samples").unwrap().clone();
            db.read_entry(&entry).unwrap().to_vec()
        };
        assert_eq!(payload, [0, 1, 2]);
    }

    #[test]
    fn version_defaults_to_minus_one_and_rejects_bad_versions() {
        let db = open_bytes(&[(b".name", b"Voice")]);
        assert_eq!(db.metadata().version, -1);
        for version in ["v1", "1234567890"] {
            let data = archive(&[(b".name", b"V"), (b".version", version.as_bytes())], 10);
            let error = VoiceDatabase::from_bytes(PathBuf::new(), data).unwrap_err();
            assert_eq!(error.to_string(), "NOFS: invalid database version metadata");
        }
    }

    #[test]
    fn rejects_foreign_entries_and_bad_headers() {
        let db = open_bytes(&[(b".name", b"V")]);
        let foreign = VoiceEntry {
            key: b".name".to_vec(),
            name: ".name".into(),
            value_offset: 999,
            value_size: 1,
        };
        assert_eq!(
            db.read_entry(&foreign).unwrap_err().to_string(),
            "NOFS: entry does not belong to the open database"
        );

        let mut data = archive(&[(b".name", b"V")], 10);
        data[0] = 0x00;
        assert_eq!(
            VoiceDatabase::from_bytes(PathBuf::new(), data)
                .unwrap_err()
                .to_string(),
            "NOFS: unrecognised file signature"
        );

        let data = archive(&[], 10);
        assert_eq!(
            VoiceDatabase::from_bytes(PathBuf::new(), data)
                .unwrap_err()
                .to_string(),
            "NOFS: container has no entries"
        );

        // Declared length must match the physical size.
        let mut data = archive(&[(b".name", b"V")], 10);
        data[8] ^= 0xff;
        assert_eq!(
            VoiceDatabase::from_bytes(PathBuf::new(), data)
                .unwrap_err()
                .to_string(),
            "NOFS: declared file length does not match the file"
        );
    }

    #[test]
    fn rejects_duplicate_keys_and_bad_trailers() {
        let data = archive(&[(b".name", b"A"), (b".name", b"B")], 10);
        assert_eq!(
            VoiceDatabase::from_bytes(PathBuf::new(), data)
                .unwrap_err()
                .to_string(),
            "NOFS: duplicate entry key"
        );

        let mut data = archive(&[(b".name", b"A")], 10);
        let last = data.len() - 1;
        data[last] ^= 0xff;
        assert_eq!(
            VoiceDatabase::from_bytes(PathBuf::new(), data)
                .unwrap_err()
                .to_string(),
            "NOFS: value block trailer mismatch"
        );
    }

    #[test]
    fn real_voice_file_parses_when_provided() {
        let Some(path) = std::env::var_os("OPENSVR_TEST_VOICE") else {
            return;
        };
        let db = VoiceDatabase::open(Path::new(&path)).unwrap();
        assert!(!db.entries().is_empty());
        assert!(!db.metadata().name.is_empty());
    }
}
