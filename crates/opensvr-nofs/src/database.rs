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
//!
//! Flat voices skip the container: the voice file is a JSON document with
//! singer metadata plus `acoustic`/`vocoder`/`duration`/`f0` model IDs, and
//! the models are sibling `.dnni` files. A leading `{` (past any byte-order
//! mark and JSON whitespace) selects document mode in
//! [`VoiceDatabase::open`]; model bytes are read eagerly into an owned blob
//! and served through the same entry interface.

use std::{
    collections::HashSet,
    fs, io,
    ops::Deref,
    path::{Path, PathBuf},
};

use serde_json::Value;
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
/// Longest accepted Flat model ID, mirroring the reference filename guard.
const MAXIMUM_MODEL_ID_LENGTH: usize = 240;
/// Model roles named by a Flat voice document, in entry order.
const MODEL_FIELDS: [&str; 4] = ["duration", "acoustic", "vocoder", "f0"];

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
            data: Backing::Owned(Vec::new()),
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
/// `open` serves entries as borrowed slices over a read-only file mapping,
/// like the C++ `VoiceDatabase` serving reads from its held-open stream:
/// the file is never copied into an owned buffer. `from_bytes` keeps an
/// owned copy for tests and non-file sources.
#[derive(Debug)]
pub struct VoiceDatabase {
    path: PathBuf,
    data: Backing,
    entries: Vec<VoiceEntry>,
    metadata: VoiceMetadata,
}

/// Backing bytes for an open archive: a file mapping or an owned copy.
#[derive(Debug)]
enum Backing {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Owned(data) => data,
            Self::Mapped(map) => map,
        }
    }
}

/// Maps `file` read-only. The single `unsafe` in the workspace lives here
/// (see the crate lint scope): `memmap2` marks mapping `unsafe` because the
/// OS may raise `SIGBUS` if the file is truncated under a live mapping.
///
/// Why this call is sound: the file is opened read-only and the mapping is
/// private and read-only, so this process never mutates it; the `Mmap` is
/// owned by the returned `VoiceDatabase` and borrowed slices never outlive
/// it. Mutating a voice file while a render holds it open is a caller
/// contract violation — the same class of misuse as rewriting the file
/// under the C++ `VoiceDatabase`'s held-open stream — and normal flows
/// never do it: the stamp-keyed voice cache re-opens on any size/mtime
/// change, closing the old database with its cache entry.
#[allow(unsafe_code)]
fn map_file(file: &fs::File) -> io::Result<memmap2::Mmap> {
    // SAFETY: upheld as documented above.
    unsafe { memmap2::Mmap::map(file) }
}

impl VoiceDatabase {
    /// Opens and validates `path`, parsing the entry table and metadata.
    ///
    /// The file is mapped read-only and entries borrow from the mapping;
    /// no owned copy of the image is kept.
    ///
    /// A leading `{` (past any byte-order mark and JSON whitespace) selects
    /// Flat document mode instead: the sibling `.dnni` model files are read
    /// eagerly into an owned blob. See [`VoiceDatabase::open_with_models`].
    ///
    /// On failure no partial state is kept; the error message matches the C++
    /// `juce::Result` text (without re-reading: the caller gets `Err` only).
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_models(path, None)
    }

    /// Opens `path`, resolving Flat document models from `models` instead of
    /// the document's sibling `models` directory. Container voices ignore it.
    pub fn open_with_models(path: &Path, models: Option<&Path>) -> Result<Self> {
        let io_error = |source: io::Error| NofsError::Io {
            path: path.to_owned(),
            source,
        };
        let mut file = fs::File::open(path).map_err(io_error)?;
        let len = file.metadata().map_err(io_error)?.len();
        if len < FILE_HEADER_SIZE + 10 {
            // Too small for a container header; it can only be a (tiny)
            // document. This also keeps empty files portable (zero-length
            // mappings fail on Windows).
            let bytes = fs::read(path).map_err(io_error)?;
            if !bytes.is_empty() && is_document_probe(&bytes) {
                let text = String::from_utf8_lossy(&bytes);
                return open_document(&text, path, models);
            }
            return Err(invalid("truncated file header"));
        }
        let probe_len = len.min(64) as usize;
        let mut probe = vec![0u8; probe_len];
        std::io::Read::read_exact(&mut file, &mut probe).map_err(io_error)?;
        if !is_document_probe(&probe) {
            let map = map_file(&file).map_err(io_error)?;
            let (entries, metadata) = parse(&map)?;
            return Ok(Self {
                path: path.to_owned(),
                data: Backing::Mapped(map),
                entries,
                metadata,
            });
        }
        drop(file);
        let bytes = fs::read(path).map_err(io_error)?;
        // The probe already saw text; lossy decoding only decides which
        // document error an undecodable file reports.
        let text = String::from_utf8_lossy(&bytes);
        open_document(&text, path, models)
    }

    /// Parses an in-memory NOFS image. The database owns a copy of `data`.
    ///
    /// A leading `{` selects Flat document mode: `data` is parsed as a JSON
    /// voice document and sibling models resolve under `path`'s parent.
    pub fn from_bytes(path: PathBuf, data: Vec<u8>) -> Result<Self> {
        if is_document_probe(&data) {
            let text = String::from_utf8_lossy(&data);
            return open_document(&text, &path, None);
        }
        let (entries, metadata) = parse(&data)?;
        Ok(Self {
            path,
            data: Backing::Owned(data),
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
    if index_size < 10
        || index_size > file_size - FILE_HEADER_SIZE
        || index_type != INDEX_BLOCK_TYPE
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
            return Err(invalid(format!("unsupported value block type at {offset}")));
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

/// Whether `prefix` starts a Flat voice document: first byte past any
/// byte-order mark and JSON whitespace is `{`, mirroring the reference
/// document probe (byte-order mark, `space \t \r \n \f`).
fn is_document_probe(prefix: &[u8]) -> bool {
    let mut bytes = prefix;
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        bytes = &bytes[3..];
    }
    let first = bytes
        .iter()
        .copied()
        .find(|byte| !matches!(byte, b' ' | b'\t' | b'\r' | b'\n' | 0x0C));
    matches!(first, Some(b'{'))
}

/// Whether a Flat model ID is a safe file name, mirroring the reference
/// installer-side guard: non-blank, bounded, no separators or reserved
/// device names, no control characters.
fn is_safe_model_id(id: &str) -> bool {
    if id.trim().is_empty()
        || id.len() > MAXIMUM_MODEL_ID_LENGTH
        || id == "."
        || id == ".."
        || id.ends_with(' ')
        || id.ends_with('.')
    {
        return false;
    }
    for character in id.chars() {
        if character < '\u{20}'
            || ('\u{7f}'..='\u{9f}').contains(&character)
            || matches!(
                character,
                '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
            )
        {
            return false;
        }
    }
    let device = id
        .split('.')
        .next()
        .unwrap_or(id)
        .trim_end_matches(' ')
        .to_uppercase();
    if matches!(device.as_str(), "CON" | "PRN" | "AUX" | "NUL") {
        return false;
    }
    if device.len() == 4
        && (device.starts_with("COM") || device.starts_with("LPT"))
        && matches!(device.as_bytes()[3], b'1'..=b'9')
    {
        return false;
    }
    true
}

/// Locates `<directory>/<id>.dnni`, falling back to a case-insensitive
/// filename match for filesystems that disagree about case. Returns the
/// expected path even when nothing matches, so missing-file errors name it.
fn find_model_file(directory: &Path, id: &str) -> PathBuf {
    let leaf = format!("{id}.dnni");
    let expected = directory.join(&leaf);
    if expected.is_file() {
        return expected;
    }
    if let Ok(scan) = fs::read_dir(directory) {
        for entry in scan.flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(&leaf)
            {
                return entry.path();
            }
        }
    }
    expected
}

/// Resolves a Flat model ID to a file: the explicit override directory
/// first, then the document's sibling `models` directory, then the
/// document's own directory.
fn resolve_model(
    doc_dir: Option<&Path>,
    models_override: Option<&Path>,
    id: &str,
) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(directory) = models_override {
        candidates.push(find_model_file(directory, id));
    }
    if let Some(directory) = doc_dir {
        candidates.push(find_model_file(&directory.join("models"), id));
        candidates.push(find_model_file(directory, id));
    }
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    let expected = candidates
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(format!("{id}.dnni")));
    Err(invalid(format!(
        "model file is missing: {}",
        expected.display()
    )))
}

/// Opens a Flat voice document: validates the four model references,
/// reads the sibling models eagerly into one owned blob, aliases the `f0`
/// entry as `f0model-dds`, and maps document metadata onto the fields
/// `info` prints. All four roles are required; the synthesizer needs each.
fn open_document(
    text: &str,
    doc_path: &Path,
    models_override: Option<&Path>,
) -> Result<VoiceDatabase> {
    if text.is_empty() {
        return Err(invalid("voice document is empty"));
    }
    let document: Value =
        serde_json::from_str(text).map_err(|_| invalid("voice document is not a JSON object"))?;
    let Value::Object(object) = &document else {
        return Err(invalid("voice document is not a JSON object"));
    };
    let doc_dir = doc_path.parent();
    let mut blob = Vec::new();
    let mut entries = Vec::new();
    let mut f0_id = String::new();
    let mut f0_offset = 0u64;
    let mut f0_size = 0u32;
    for field in MODEL_FIELDS {
        let id = match object.get(field) {
            Some(Value::String(id)) if is_safe_model_id(id) => id.clone(),
            _ => return Err(invalid(format!("unsafe model identifier for {field}"))),
        };
        let model_path = resolve_model(doc_dir, models_override, &id)?;
        let bytes = fs::read(&model_path).map_err(|source| NofsError::Io {
            path: model_path.clone(),
            source,
        })?;
        if bytes.len() > MAXIMUM_ENTRY_SIZE as usize {
            return Err(invalid(format!(
                "model file exceeds the 512 MiB read limit: {}",
                model_path.display()
            )));
        }
        let offset = blob.len() as u64;
        blob.extend_from_slice(&bytes);
        entries.push(VoiceEntry {
            key: id.as_bytes().to_vec(),
            name: id.clone(),
            value_offset: offset,
            value_size: bytes.len() as u32,
        });
        if field == "f0" {
            f0_id = id;
            f0_offset = offset;
            f0_size = bytes.len() as u32;
        }
    }
    // The pitch model resolves by fixed name rather than through the
    // document, which names the `f0` role only.
    entries.push(VoiceEntry {
        key: f0_id.as_bytes().to_vec(),
        name: "f0model-dds".to_owned(),
        value_offset: f0_offset,
        value_size: f0_size,
    });
    let metadata = document_metadata(object);
    Ok(VoiceDatabase {
        path: doc_path.to_owned(),
        data: Backing::Owned(blob),
        entries,
        metadata,
    })
}

/// Maps Flat document metadata onto the container's dot-prefixed property
/// fields, so `info` prints both formats identically.
fn document_metadata(object: &serde_json::Map<String, Value>) -> VoiceMetadata {
    let string = |key: &str| -> String {
        object
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let strings = |key: &str| -> Vec<String> {
        object
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    };
    let styles: Vec<String> = object
        .get("styles")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name"))
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let mut metadata = VoiceMetadata::empty();
    metadata.name = string("name");
    metadata.vendor = string("vendor");
    metadata.language = string("language");
    metadata.phoneset = string("phoneset");
    metadata.languages = strings("support_languages");
    metadata.timbre_styles = styles;
    metadata
        .properties
        .push((".name".to_owned(), metadata.name.clone()));
    metadata
        .properties
        .push((".vendor".to_owned(), metadata.vendor.clone()));
    metadata
        .properties
        .push((".language".to_owned(), metadata.language.clone()));
    metadata
        .properties
        .push((".phoneset".to_owned(), metadata.phoneset.clone()));
    let version = string("version");
    if !version.is_empty()
        && version.len() <= 9
        && version.bytes().all(|byte| byte.is_ascii_digit())
    {
        metadata.version = version.parse().unwrap_or(-1);
        metadata.properties.push((".version".to_owned(), version));
    }
    if !metadata.languages.is_empty() {
        metadata
            .properties
            .push((".multi".to_owned(), metadata.languages.join(",")));
    }
    if !metadata.timbre_styles.is_empty() {
        metadata.properties.push((
            ".timbre_styles".to_owned(),
            metadata.timbre_styles.join(","),
        ));
    }
    metadata
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
        assert_eq!(binary.entries()[0].name.len(), 0);
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
    fn open_serves_entries_from_a_mapping() {
        let dir = std::env::temp_dir().join(format!(
            "opensvr-nofs-map-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("voice.nofs");
        let records = &[
            (b".name".as_slice(), b"Kasane Teto".as_slice()),
            (b".version".as_slice(), b"104".as_slice()),
            (b"samples".as_slice(), b"\x00\x01\x02".as_slice()),
        ];
        std::fs::write(&path, archive(records, 10)).unwrap();

        let mapped = VoiceDatabase::open(&path).unwrap();
        let owned = VoiceDatabase::from_bytes(path.clone(), std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(mapped.entries().len(), owned.entries().len());
        assert_eq!(mapped.metadata().name, "Kasane Teto");
        for (left, right) in mapped.entries().iter().zip(owned.entries().iter()) {
            assert_eq!(left.key, right.key);
            assert_eq!(left.value_offset, right.value_offset);
            assert_eq!(left.value_size, right.value_size);
            assert_eq!(
                mapped.read_entry(left).unwrap(),
                owned.read_entry(right).unwrap()
            );
        }
        // Payloads read through the map, with the membership check intact.
        let entry = mapped.find_entry("samples").unwrap().clone();
        assert_eq!(mapped.read_entry(&entry).unwrap(), [0, 1, 2]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_keeps_file_errors_unchanged() {
        let missing = std::env::temp_dir().join(format!(
            "opensvr-nofs-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let error = VoiceDatabase::open(&missing).unwrap_err().to_string();
        assert!(
            error.starts_with("could not read voice database "),
            "unexpected: {error}"
        );

        let dir = std::env::temp_dir().join(format!(
            "opensvr-nofs-empty-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty.nofs");
        std::fs::write(&empty, []).unwrap();
        assert_eq!(
            VoiceDatabase::open(&empty).unwrap_err().to_string(),
            "NOFS: truncated file header"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn real_voice_file_parses_when_provided() {
        let Some(path) = std::env::var_os("OPENSVR_TEST_VOICE") else {
            return;
        };
        let db = VoiceDatabase::open(Path::new(&path)).unwrap();
        assert!(!db.entries().is_empty());
        assert_ne!(db.metadata().name.len(), 0);
    }

    fn flat_document(models: &[(&str, &[u8])]) -> serde_json::Value {
        serde_json::json!({
            "name": "Flat Singer",
            "version": "100",
            "vendor": "Flat Vendor",
            "language": "japanese",
            "phoneset": "romaji",
            "support_languages": ["english", "japanese"],
            "acoustic": models[0].0,
            "vocoder": models[1].0,
            "duration": models[2].0,
            "f0": models[3].0,
            "styles": [{"name": "(base)"}, {"name": "Joyful"}],
        })
    }

    fn write_flat_voice(models: &[(&str, &[u8])]) -> (TempDirGuard, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "opensvr-nofs-flat-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models_dir = dir.join("models");
        std::fs::create_dir_all(&models_dir).unwrap();
        for (id, bytes) in models {
            std::fs::write(models_dir.join(format!("{id}.dnni")), bytes).unwrap();
        }
        let doc = dir.join("info.nofs");
        std::fs::write(
            &doc,
            serde_json::to_string_pretty(&flat_document(models)).unwrap(),
        )
        .unwrap();
        (TempDirGuard(dir), doc)
    }

    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const FLAT_MODELS: [(&str, &[u8]); 4] = [
        ("acoustic-hash", b"acoustic-bytes"),
        ("vocoder-hash", b"vocoder-bytes"),
        ("duration-hash", b"duration-bytes"),
        ("f0-hash", b"f0-bytes"),
    ];

    #[test]
    fn document_probe_skips_bom_and_whitespace() {
        assert!(is_document_probe(b"{\"a\":1}"));
        assert!(is_document_probe(b"\xef\xbb\xbf{\"a\":1}"));
        assert!(is_document_probe(b" \t\r\n\x0c {\"a\":1}"));
        assert!(is_document_probe(b"{}"));
        assert!(!is_document_probe(b""));
        assert!(!is_document_probe(b" \t\r\n\x0c"));
        assert!(!is_document_probe(b"\xef\xbb\xbf"));
        // Binary container magic (`0xf580` little-endian) is not a document.
        assert!(!is_document_probe(&[0x80, 0xF5, 0x00, 0x00]));
        assert!(!is_document_probe(b"[1,2]"));
    }

    #[test]
    fn safe_model_id_table() {
        for id in ["abc", "04fde26b27b186201780739f06eb7e54", "a.b", "UPPER-1"] {
            assert!(is_safe_model_id(id), "unexpected rejection: {id}");
        }
        for id in [
            "",
            "   ",
            ".",
            "..",
            "a/b",
            "a\\b",
            "a:b",
            "a?b",
            "trailing ",
            "trailing.",
            "CON",
            "con.json",
            "COM1",
            "lpt9",
            "nul",
            "a\x01b",
            "a\x7fb",
        ] {
            assert!(!is_safe_model_id(id), "unexpected acceptance: {id}");
        }
        assert!(!is_safe_model_id(&"a".repeat(241)));
        assert!(is_safe_model_id(&"a".repeat(240)));
    }

    #[test]
    fn opens_flat_document_with_alias_and_metadata() {
        let (_guard, doc) = write_flat_voice(&FLAT_MODELS);
        let db = VoiceDatabase::open(&doc).unwrap();
        // Four models plus the `f0model-dds` alias.
        assert_eq!(db.entries().len(), 5);
        for (id, bytes) in FLAT_MODELS {
            let entry = db.find_entry(id).unwrap();
            assert_eq!(db.read_entry(entry).unwrap(), bytes);
        }
        let alias = db.find_entry("f0model-dds").unwrap();
        assert_eq!(db.read_entry(alias).unwrap(), b"f0-bytes");
        assert_eq!(
            alias.value_offset,
            db.find_entry("f0-hash").unwrap().value_offset
        );

        let metadata = db.metadata();
        assert_eq!(metadata.name, "Flat Singer");
        assert_eq!(metadata.vendor, "Flat Vendor");
        assert_eq!(metadata.language, "japanese");
        assert_eq!(metadata.phoneset, "romaji");
        assert_eq!(metadata.version, 100);
        assert_eq!(metadata.languages, ["english", "japanese"]);
        assert_eq!(metadata.timbre_styles, ["(base)", "Joyful"]);
        assert_eq!(metadata.property(".multi"), "english,japanese");
        assert_eq!(metadata.property(".timbre_styles"), "(base),Joyful");
    }

    #[test]
    fn document_open_rejects_bad_documents() {
        let (_guard, doc) = write_flat_voice(&FLAT_MODELS);
        let parent = doc.parent().unwrap();

        // Not an object.
        std::fs::write(&doc, b"[1,2]").unwrap();
        assert_eq!(
            VoiceDatabase::open(&doc).unwrap_err().to_string(),
            "NOFS: truncated file header"
        );
        // Broken JSON reports the document verdict.
        std::fs::write(&doc, b"{\"name\":").unwrap();
        assert_eq!(
            VoiceDatabase::open(&doc).unwrap_err().to_string(),
            "NOFS: voice document is not a JSON object"
        );
        // Unsafe model ID names its field.
        let mut bad = flat_document(&FLAT_MODELS);
        bad["acoustic"] = serde_json::Value::String("../evil".into());
        std::fs::write(&doc, serde_json::to_string(&bad).unwrap()).unwrap();
        assert_eq!(
            VoiceDatabase::open(&doc).unwrap_err().to_string(),
            "NOFS: unsafe model identifier for acoustic"
        );
        // Missing role reports the same verdict.
        let mut missing = flat_document(&FLAT_MODELS);
        missing.as_object_mut().unwrap().remove("f0");
        std::fs::write(&doc, serde_json::to_string(&missing).unwrap()).unwrap();
        assert_eq!(
            VoiceDatabase::open(&doc).unwrap_err().to_string(),
            "NOFS: unsafe model identifier for f0"
        );
        // A valid document with a missing model file names the file.
        std::fs::write(
            &doc,
            serde_json::to_string_pretty(&flat_document(&FLAT_MODELS)).unwrap(),
        )
        .unwrap();
        std::fs::remove_file(parent.join("models").join("f0-hash.dnni")).unwrap();
        let error = VoiceDatabase::open(&doc).unwrap_err().to_string();
        assert!(
            error.starts_with("NOFS: model file is missing: "),
            "unexpected: {error}"
        );
    }

    #[test]
    fn document_models_resolve_case_insensitively_and_by_override() {
        let (_guard, doc) = write_flat_voice(&FLAT_MODELS);
        let parent = doc.parent().unwrap();
        // Uppercase the acoustic file: exact lookup misses, fallback hits.
        std::fs::rename(
            parent.join("models").join("acoustic-hash.dnni"),
            parent.join("models").join("ACOUSTIC-HASH.DNNI"),
        )
        .unwrap();
        let db = VoiceDatabase::open(&doc).unwrap();
        assert_eq!(
            db.read_entry(db.find_entry("acoustic-hash").unwrap())
                .unwrap(),
            b"acoustic-bytes"
        );

        // An explicit override directory wins over the sibling directory.
        let elsewhere = parent.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("acoustic-hash.dnni"), b"override").unwrap();
        let db = VoiceDatabase::open_with_models(&doc, Some(&elsewhere)).unwrap();
        assert_eq!(
            db.read_entry(db.find_entry("acoustic-hash").unwrap())
                .unwrap(),
            b"override"
        );
    }

    #[test]
    fn from_bytes_opens_documents_beside_their_models() {
        let (_guard, doc) = write_flat_voice(&FLAT_MODELS);
        let bytes = std::fs::read(&doc).unwrap();
        let db = VoiceDatabase::from_bytes(doc.clone(), bytes).unwrap();
        assert_eq!(db.entries().len(), 5);
        assert_eq!(db.metadata().name, "Flat Singer");
    }

    #[test]
    fn real_flat_voice_parses_when_provided() {
        let Some(path) = std::env::var_os("OPENSVR_TEST_FLAT") else {
            return;
        };
        let db = VoiceDatabase::open(Path::new(&path)).unwrap();
        // Four models plus the `f0model-dds` alias.
        assert_eq!(db.entries().len(), 5);
        let pitch = db.find_entry("f0model-dds").unwrap();
        let bytes = db.read_entry(pitch).unwrap();
        // Every DNNI model starts with the `0x7fca00ff` magic.
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[..4], &[0xFF, 0x00, 0xCA, 0x7F]);
        assert_ne!(db.metadata().name.len(), 0);
    }
}
