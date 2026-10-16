//! Stamp-keyed voice, dictionary and phrase caches for neural rendering.
//!
//! Ports the caching half of OpenSV's `src/audio/ProjectRenderer.h/.cpp`
//! (`FileStamp`, `CachedVoice`, `CachedDictionary`, `CachedPhrase` with
//! `readFileStamp`, `findVoice`, `matchesTiming` and `retainPhrase`): one
//! render loads each voice database and dictionary file once no matter how
//! many phrases it sings, and repeated identical phrases reuse their
//! native-rate mono output. Cached entries are placement-independent — the
//! mixer crops them per phrase exactly like fresh renders — and keyed on
//! synthesis inputs (voice stamp, syllables, log-F0, vocal-mode weights), so
//! sharing them across tracks and re-renders is safe.
//!
//! `CachedInference` (DNNI-state reuse) is intentionally not ported: it is an
//! editing optimization and one-shot renders do not need it.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use opensvr_core::Language;
use opensvr_g2p::{DictionarySet, TimingSyllable};
use opensvr_voice::VoiceSynthesizer;

use crate::BackendError;

/// Largest voice count, exactly as in C++ (`voices.size() == 2` evicts oldest).
pub const MAX_CACHED_VOICES: usize = 2;
/// Largest dictionary count, exactly as in C++ (`dictionaries.size() == 4`).
pub const MAX_CACHED_DICTIONARIES: usize = 4;
/// Largest phrase count, exactly as in C++ (`maximumCachedPhrases`).
pub const MAX_CACHED_PHRASES: usize = 1024;
/// Largest phrase-cache size in bytes (`maximumPhraseCacheBytes`, 128 MiB).
pub const MAX_PHRASE_CACHE_BYTES: usize = 128 << 20;

/// File identity for cache keys: path plus size plus modification time.
///
/// Mirrors C++ `FileStamp { file, size, modifiedMilliseconds }`. Equality
/// covers all three fields, so an in-place rewrite with identical size still
/// misses when its mtime moves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileStamp {
    path: PathBuf,
    size: u64,
    modified: std::time::SystemTime,
}

impl FileStamp {
    /// Reads the stamp for `path`.
    ///
    /// Returns the underlying I/O error when the file is missing or
    /// unreadable; callers map it to the same message their uncached load
    /// path reports today, so a missing file fails a phrase exactly as
    /// before (never as a silent cache miss).
    pub fn read(path: &Path) -> Result<Self, std::io::Error> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is not a file", path.display()),
            ));
        }
        Ok(Self {
            path: path.to_owned(),
            size: metadata.len(),
            modified: metadata.modified()?,
        })
    }

    /// The stamped path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Voice-synthesizer cache key: the database stamp plus the Flat
/// models-directory override. The override participates so the same document
/// with different model sources never shares a synthesizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceKey {
    voice: FileStamp,
    models: Option<PathBuf>,
}

impl VoiceKey {
    /// Builds a lookup key from the stamped database and the models override
    /// the synthesizer was loaded with (`None` means sibling resolution).
    pub fn new(voice: FileStamp, models: Option<PathBuf>) -> Self {
        Self { voice, models }
    }
}

/// MRU voice-synthesizer cache keyed by [`VoiceKey`], cap 2 like C++.
///
/// A hit rotates the entry to the back; a miss loads through the caller's
/// closure (which reports load failures exactly as the uncached path does)
/// and evicts the least-recently-used entry once full. Only successful loads
/// are retained.
#[derive(Debug, Default)]
pub struct VoiceCache {
    entries: Vec<VoiceEntry>,
    loads: usize,
}

#[derive(Debug)]
struct VoiceEntry {
    key: VoiceKey,
    synthesizer: Arc<VoiceSynthesizer>,
}

impl VoiceCache {
    /// Returns the synthesizer for `key`, loading it on a miss.
    ///
    /// `load` runs only on a miss; its error propagates without caching. The
    /// synthesizer is shared by reference count so worker threads can drop
    /// the cache lock before singing: synthesis itself only needs `&self`.
    pub fn get_or_load(
        &mut self,
        key: VoiceKey,
        load: impl FnOnce() -> Result<VoiceSynthesizer, BackendError>,
    ) -> Result<Arc<VoiceSynthesizer>, BackendError> {
        if let Some(position) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.remove(position);
            let synthesizer = entry.synthesizer.clone();
            self.entries.push(entry);
            return Ok(synthesizer);
        }
        let synthesizer = Arc::new(load()?);
        if self.entries.len() == MAX_CACHED_VOICES {
            self.entries.remove(0);
        }
        self.entries.push(VoiceEntry {
            key,
            synthesizer: synthesizer.clone(),
        });
        self.loads += 1;
        Ok(synthesizer)
    }

    /// Successful voice loads so far (misses that loaded).
    pub fn loads(&self) -> usize {
        self.loads
    }

    /// Cached voice count.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no voice is cached.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Dictionary-set cache key: directory, language and every file stamp the
/// loader reads (phones, dictionary plus the Mandarin/Japanese extras and the
/// best-effort Japanese English fallback), mirroring C++'s six-stamp lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryKey {
    directory: PathBuf,
    language: Language,
    stamps: Vec<FileStamp>,
}

impl DictionaryKey {
    /// Builds a lookup key from the stamped files `load_dictionaries` reads.
    pub fn new(directory: PathBuf, language: Language, stamps: Vec<FileStamp>) -> Self {
        Self {
            directory,
            language,
            stamps,
        }
    }
}

/// MRU dictionary cache keyed by [`DictionaryKey`], cap 4 like C++.
#[derive(Debug, Default)]
pub struct DictionaryCache {
    entries: Vec<DictionaryEntry>,
    loads: usize,
}

#[derive(Debug, Clone)]
struct DictionaryEntry {
    key: DictionaryKey,
    set: DictionarySet,
}

impl DictionaryCache {
    /// Returns the dictionaries for `key`, loading them on a miss.
    pub fn get_or_load(
        &mut self,
        key: DictionaryKey,
        load: impl FnOnce() -> Result<DictionarySet, BackendError>,
    ) -> Result<&DictionarySet, BackendError> {
        if let Some(position) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.remove(position);
            self.entries.push(entry);
        } else {
            let set = load()?;
            if self.entries.len() == MAX_CACHED_DICTIONARIES {
                self.entries.remove(0);
            }
            self.entries.push(DictionaryEntry { key, set });
            self.loads += 1;
        }
        let index = self.entries.len() - 1;
        Ok(&self.entries[index].set)
    }

    /// Successful dictionary loads so far.
    pub fn loads(&self) -> usize {
        self.loads
    }

    /// Cached dictionary count.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no dictionary is cached.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Cached native-rate mono phrase output with its full synthesis key.
///
/// The key mirrors the C++ render-track closure: `matchesTiming` (voice
/// stamp plus full `TimingSyllable` equality) then exact `logF0` plus
/// vocal-mode-weights equality. Placement, output rate and mixer never
/// participate: a hit resamples and crops exactly like a fresh render.
#[derive(Debug, Clone)]
struct PhraseEntry {
    voice: VoiceKey,
    syllables: Vec<TimingSyllable>,
    log_f0: Vec<f32>,
    weights: Vec<f32>,
    native_rate: u32,
    samples: Vec<f32>,
    bytes: usize,
}

impl PhraseEntry {
    fn matches(
        &self,
        voice: &VoiceKey,
        syllables: &[TimingSyllable],
        log_f0: &[f32],
        weights: &[f32],
    ) -> bool {
        self.voice == *voice
            && self.syllables == syllables
            && self.log_f0 == *log_f0
            && self.weights == *weights
    }
}

/// Approximates the entry's heap footprint for the byte-cap LRU.
///
/// Counts the stored vectors plus the variable-length synthesis strings, on
/// top of one entry's inline size — the same shape as C++ `phraseBytes`
/// (which additionally counts the path text and visualization peaks, neither
/// of which this port keeps).
fn phrase_entry_bytes(
    voice: &VoiceKey,
    syllables: &[TimingSyllable],
    log_f0: &[f32],
    weights: &[f32],
    samples: &[f32],
) -> usize {
    let mut bytes = size_of::<PhraseEntry>()
        + voice.voice.path().as_os_str().len()
        + voice
            .models
            .as_ref()
            .map_or(0, |models| models.as_os_str().len())
        + (samples.len() + log_f0.len() + weights.len()) * size_of::<f32>();
    for syllable in syllables {
        bytes += size_of::<TimingSyllable>()
            + syllable.language.len()
            + syllable.phonemes.len() * size_of::<String>();
        for phoneme in &syllable.phonemes {
            bytes += phoneme.len();
        }
    }
    bytes
}

/// LRU native-audio cache: at most 1024 phrases under a 128 MiB byte cap.
///
/// A hit rotates the entry to the back; `retain` drops entries from the
/// front while the new entry would overflow either limit, and drops the new
/// entry outright when it alone exceeds the byte cap — all exactly as in C++
/// `retainPhrase`.
#[derive(Debug, Default)]
pub struct PhraseCache {
    entries: VecDeque<PhraseEntry>,
    bytes: usize,
    hits: usize,
    misses: usize,
}

impl PhraseCache {
    /// Returns the cached native output for an identical synthesis, if any.
    ///
    /// The returned samples are native-rate mono; the caller resamples them
    /// to the project rate, exactly as with a fresh render.
    pub fn lookup(
        &mut self,
        voice: &VoiceKey,
        syllables: &[TimingSyllable],
        log_f0: &[f32],
        weights: &[f32],
    ) -> Option<(u32, Vec<f32>)> {
        let position = self
            .entries
            .iter()
            .position(|entry| entry.matches(voice, syllables, log_f0, weights))?;
        let entry = self
            .entries
            .remove(position)
            .expect("position came from this deque");
        let output = (entry.native_rate, entry.samples.clone());
        self.entries.push_back(entry);
        self.hits += 1;
        Some(output)
    }

    /// Retains newly rendered native output under its synthesis key.
    pub fn retain(
        &mut self,
        voice: VoiceKey,
        syllables: Vec<TimingSyllable>,
        log_f0: Vec<f32>,
        weights: Vec<f32>,
        native_rate: u32,
        samples: Vec<f32>,
    ) {
        let bytes = phrase_entry_bytes(&voice, &syllables, &log_f0, &weights, &samples);
        if bytes > MAX_PHRASE_CACHE_BYTES {
            return;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= MAX_CACHED_PHRASES
                || self.bytes > MAX_PHRASE_CACHE_BYTES - bytes)
        {
            if let Some(evicted) = self.entries.pop_front() {
                self.bytes -= evicted.bytes;
            }
        }
        self.entries.push_back(PhraseEntry {
            voice,
            syllables,
            log_f0,
            weights,
            native_rate,
            samples,
            bytes,
        });
        self.bytes += bytes;
        self.misses += 1;
    }

    /// Cache hits so far.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Cache misses that retained a render so far.
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Cached phrase count.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no phrase is cached.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Accounted cache bytes.
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "opensvr-cache-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        fs::create_dir_all(&dir).expect("test temp dir creates");
        dir
    }

    fn syllable(phonemes: &[&str]) -> TimingSyllable {
        TimingSyllable {
            language: "japanese".to_owned(),
            phonemes: phonemes.iter().map(|phone| (*phone).to_owned()).collect(),
            duration_seconds: 0.4,
            midi_pitch: 60,
            is_continuation: false,
        }
    }

    #[test]
    fn stamps_reject_missing_files() {
        let missing = temp_dir("missing").join("nope.nofs");
        assert!(FileStamp::read(&missing).is_err());
    }

    #[test]
    fn stamps_cover_path_size_and_mtime() {
        let dir = temp_dir("stamp");
        let first = dir.join("a.nofs");
        let second = dir.join("b.nofs");
        fs::write(&first, b"voice").unwrap();
        fs::write(&second, b"voice").unwrap();
        let stamp = FileStamp::read(&first).unwrap();
        // Same bytes at a different path are a different file.
        assert_ne!(stamp, FileStamp::read(&second).unwrap());
        // Same path with different bytes is a different stamp.
        fs::write(&first, b"voice!").unwrap();
        assert_ne!(stamp, FileStamp::read(&first).unwrap());
        // Same path and size with a fresh mtime is a different stamp.
        fs::write(&first, b"other!").unwrap();
        assert_ne!(stamp, FileStamp::read(&first).unwrap());
        assert_eq!(stamp.path(), first.as_path());
    }

    #[test]
    fn voices_load_once_and_evict_the_oldest_of_two() {
        let dir = temp_dir("voices");
        let files: Vec<PathBuf> = (0..3)
            .map(|index| {
                let path = dir.join(format!("{index}.nofs"));
                fs::write(&path, b"voice").unwrap();
                path
            })
            .collect();
        let stamps: Vec<FileStamp> = files
            .iter()
            .map(|path| FileStamp::read(path).unwrap())
            .collect();
        let mut loads = 0_usize;
        let mut cache = VoiceCache::default();
        let load = |loads: &mut usize| {
            *loads += 1;
            Ok(VoiceSynthesizer::new())
        };
        let key = |stamp: &FileStamp| VoiceKey::new(stamp.clone(), None);
        cache
            .get_or_load(key(&stamps[0]), || load(&mut loads))
            .unwrap();
        cache
            .get_or_load(key(&stamps[0]), || load(&mut loads))
            .unwrap();
        assert_eq!(loads, 1, "a repeated stamp must not reload");
        assert_eq!(cache.loads(), 1);
        cache
            .get_or_load(key(&stamps[1]), || load(&mut loads))
            .unwrap();
        // Touch the first entry so the second becomes least-recently-used.
        cache
            .get_or_load(key(&stamps[0]), || load(&mut loads))
            .unwrap();
        cache
            .get_or_load(key(&stamps[2]), || load(&mut loads))
            .unwrap();
        assert_eq!(cache.len(), MAX_CACHED_VOICES);
        assert_eq!(loads, 3);
        // The rotated first entry survived; the evicted second reloads.
        cache
            .get_or_load(key(&stamps[0]), || load(&mut loads))
            .unwrap();
        assert_eq!(loads, 3);
        cache
            .get_or_load(key(&stamps[1]), || load(&mut loads))
            .unwrap();
        assert_eq!(loads, 4);
    }

    #[test]
    fn voice_keys_separate_models_dir_overrides() {
        let dir = temp_dir("voice-models-key");
        let path = dir.join("voice.nofs");
        fs::write(&path, b"voice").unwrap();
        let stamp = FileStamp::read(&path).unwrap();
        let mut loads = 0_usize;
        let mut cache = VoiceCache::default();
        let load = |loads: &mut usize| {
            *loads += 1;
            Ok(VoiceSynthesizer::new())
        };
        cache
            .get_or_load(VoiceKey::new(stamp.clone(), None), || load(&mut loads))
            .unwrap();
        // The same database with a models override misses the sibling entry.
        cache
            .get_or_load(
                VoiceKey::new(stamp.clone(), Some(PathBuf::from("models"))),
                || load(&mut loads),
            )
            .unwrap();
        assert_eq!(loads, 2);
        cache
            .get_or_load(VoiceKey::new(stamp, None), || load(&mut loads))
            .unwrap();
        assert_eq!(loads, 2, "the sibling entry must still hit");
    }

    #[test]
    fn voices_propagate_load_failures_without_caching() {
        let dir = temp_dir("voice-fail");
        let path = dir.join("broken.nofs");
        fs::write(&path, b"voice").unwrap();
        let stamp = FileStamp::read(&path).unwrap();
        let mut cache = VoiceCache::default();
        let error = cache
            .get_or_load(VoiceKey::new(stamp, None), || {
                Err(BackendError("Cannot read synthesis resource.".into()))
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "Cannot read synthesis resource.");
        assert!(cache.is_empty());
        assert_eq!(cache.loads(), 0);
    }

    fn dictionary_key(dir: &Path, language: Language, names: &[&str]) -> DictionaryKey {
        DictionaryKey {
            directory: dir.to_owned(),
            language,
            stamps: names
                .iter()
                .map(|name| FileStamp::read(&dir.join(name)).unwrap())
                .collect(),
        }
    }

    #[test]
    fn dictionaries_key_on_directory_language_and_stamps() {
        let dir = temp_dir("dicts");
        for name in ["phones.txt", "dict.txt"] {
            fs::write(dir.join(name), b"table").unwrap();
        }
        let mut loads = 0_usize;
        let mut cache = DictionaryCache::default();
        assert!(cache.is_empty());
        let key = dictionary_key(&dir, Language::Japanese, &["phones.txt", "dict.txt"]);
        cache
            .get_or_load(key.clone(), || {
                loads += 1;
                Ok(DictionarySet::default())
            })
            .unwrap();
        assert_eq!(cache.len(), 1);
        cache
            .get_or_load(key, || {
                loads += 1;
                Ok(DictionarySet::default())
            })
            .unwrap();
        assert_eq!(loads, 1);
        // Another language with identical stamps is a different key.
        cache
            .get_or_load(
                dictionary_key(&dir, Language::English, &["phones.txt", "dict.txt"]),
                || {
                    loads += 1;
                    Ok(DictionarySet::default())
                },
            )
            .unwrap();
        assert_eq!(loads, 2);
        // Rewriting one file misses again.
        fs::write(dir.join("dict.txt"), b"table!").unwrap();
        cache
            .get_or_load(
                dictionary_key(&dir, Language::Japanese, &["phones.txt", "dict.txt"]),
                || {
                    loads += 1;
                    Ok(DictionarySet::default())
                },
            )
            .unwrap();
        assert_eq!(loads, 3);
    }

    #[test]
    fn phrases_hit_on_inputs_and_miss_on_any_change() {
        let dir = temp_dir("phrases");
        let voice = dir.join("voice.nofs");
        fs::write(&voice, b"voice").unwrap();
        let stamp = FileStamp::read(&voice).unwrap();
        let syllables = vec![syllable(&["k", "a"])];
        let log_f0 = vec![4.0_f32, 4.1];
        let weights: Vec<f32> = Vec::new();
        let samples = vec![0.1_f32, 0.2, 0.3];
        let mut cache = PhraseCache::default();
        cache.retain(
            VoiceKey::new(stamp.clone(), None),
            syllables.clone(),
            log_f0.clone(),
            weights.clone(),
            48_000,
            samples.clone(),
        );
        assert_eq!(cache.misses(), 1);
        assert!(!cache.is_empty());
        let hit = cache
            .lookup(
                &VoiceKey::new(stamp.clone(), None),
                &syllables,
                &log_f0,
                &weights,
            )
            .expect("identical inputs must hit");
        assert_eq!(hit, (48_000, samples));
        assert_eq!(cache.hits(), 1);
        // Every key component participates: voice, syllables, F0, weights.
        let other_voice = dir.join("other.nofs");
        fs::write(&other_voice, b"other").unwrap();
        let other_stamp = FileStamp::read(&other_voice).unwrap();
        assert!(
            cache
                .lookup(
                    &VoiceKey::new(other_stamp, None),
                    &syllables,
                    &log_f0,
                    &weights,
                )
                .is_none()
        );
        let other_syllables = vec![syllable(&["s", "a"])];
        assert!(
            cache
                .lookup(
                    &VoiceKey::new(stamp.clone(), None),
                    &other_syllables,
                    &log_f0,
                    &weights,
                )
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &VoiceKey::new(stamp.clone(), None),
                    &syllables,
                    &[5.0, 4.1],
                    &weights,
                )
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &VoiceKey::new(stamp.clone(), None),
                    &syllables,
                    &log_f0,
                    &[0.5]
                )
                .is_none()
        );
        // The models override joins the key: same inputs, other source, miss.
        assert!(
            cache
                .lookup(
                    &VoiceKey::new(stamp, Some(PathBuf::from("models"))),
                    &syllables,
                    &log_f0,
                    &weights
                )
                .is_none()
        );
        assert_eq!(cache.hits(), 1, "misses must not count as hits");
    }

    #[test]
    fn phrases_evict_the_oldest_first_under_both_caps() {
        let dir = temp_dir("phrase-lru");
        let voice = dir.join("voice.nofs");
        fs::write(&voice, b"voice").unwrap();
        let stamp = FileStamp::read(&voice).unwrap();
        let mut cache = PhraseCache::default();
        for index in 0..=MAX_CACHED_PHRASES {
            // Vary the pitch so every entry has a distinct key.
            let mut syllable = syllable(&["a"]);
            syllable.midi_pitch = i32::try_from(index).expect("loop bound fits");
            cache.retain(
                VoiceKey::new(stamp.clone(), None),
                vec![syllable],
                vec![4.0],
                Vec::new(),
                48_000,
                vec![0.1],
            );
        }
        assert_eq!(cache.len(), MAX_CACHED_PHRASES);
        assert!(cache.bytes() <= MAX_PHRASE_CACHE_BYTES);
        // Oversized entries never enter the cache.
        let huge = vec![0.0_f32; MAX_PHRASE_CACHE_BYTES];
        let before = cache.len();
        cache.retain(
            VoiceKey::new(stamp.clone(), None),
            vec![syllable(&["a"])],
            vec![4.0; 8],
            Vec::new(),
            48_000,
            huge,
        );
        assert_eq!(cache.len(), before);
    }
}
