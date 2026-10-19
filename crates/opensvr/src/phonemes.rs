//! Pronunciation diagnostics: resolve a track's lyrics to phonemes.
//!
//! This is the resolution `opensvr info --phonemes` reports, exposed so an
//! embed can answer "what will this note sing?" without rendering. It is a
//! diagnostic path, not part of rendering: the renderer resolves through
//! [`opensvr_audio::NeuralBackend`], which caches by file stamp.
//!
//! Dictionaries are loaded once per directory and language and kept for the
//! life of the resolver, so a project whose tracks share a singer pays for it
//! once.

use std::path::PathBuf;

use opensvr_core::{Language, Note, Track};
use opensvr_g2p::{DictionarySet, G2pError, NoteResolver, Resolved, load_dictionaries};
use thiserror::Error;

/// Why a track's phonemes could not be resolved at all.
///
/// A note that fails on its own does not raise this: it is reported in place
/// by [`NotePhonemes::result`], matching the per-note errors `info` prints.
#[derive(Debug, Error)]
pub enum PhonemeError {
    /// The track has notes but no dictionary directory configured.
    #[error("no dictionary directory")]
    NoDictionary,
    /// The dictionary files exist but could not be read.
    #[error(transparent)]
    Dictionaries(#[from] G2pError),
}

/// One note's resolved phonemes, or the reason that note failed.
///
/// `note` is borrowed so a caller can print the original lyrics and position
/// alongside the result; the note order is the track's order, which the
/// resolver's `+`/`-` state depends on.
#[derive(Debug)]
pub struct NotePhonemes<'a> {
    pub note: &'a Note,
    pub result: Result<Resolved, G2pError>,
}

/// Resolves track lyrics to phonemes, caching dictionaries across tracks.
///
/// The cache key is the dictionary directory plus the track language, exactly
/// like the CLI's, so sharing one resolver across a project's tracks loads each
/// dictionary once.
#[derive(Debug, Default)]
pub struct PhonemeResolver {
    dictionaries: Vec<(PathBuf, Language, DictionarySet)>,
    loads: usize,
}

impl PhonemeResolver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Successful dictionary loads so far (one per directory and language).
    #[must_use]
    pub fn dictionary_loads(&self) -> usize {
        self.loads
    }

    /// Resolves every note of `track.main_group` in order.
    ///
    /// An empty track resolves to no notes without needing a dictionary. A
    /// track with notes but no dictionary directory returns
    /// [`PhonemeError::NoDictionary`], even when every note carries explicit
    /// phonemes, matching what `info --phonemes` reports today.
    pub fn resolve_track<'a>(
        &mut self,
        track: &'a Track,
    ) -> Result<Vec<NotePhonemes<'a>>, PhonemeError> {
        let notes = &track.main_group.notes;
        if notes.is_empty() {
            return Ok(Vec::new());
        }
        let Some(directory) = track.voice.dictionary.clone() else {
            return Err(PhonemeError::NoDictionary);
        };
        let language = track.voice.language;
        let dictionaries = self.dictionaries_for(directory, language)?;

        let mut resolver = NoteResolver::new(language);
        let mut entries = Vec::with_capacity(notes.len());
        for (index, note) in notes.iter().enumerate() {
            let next_is_legato = notes.get(index + 1).is_some_and(|next| next.lyrics == "-");
            entries.push(NotePhonemes {
                note,
                result: resolver.push(&note.lyrics, &note.phonemes, &dictionaries, next_is_legato),
            });
        }
        Ok(entries)
    }

    /// Returns `directory`'s dictionaries for `language`, loading them once.
    fn dictionaries_for(
        &mut self,
        directory: PathBuf,
        language: Language,
    ) -> Result<DictionarySet, PhonemeError> {
        let cached = self
            .dictionaries
            .iter()
            .find(|(known, known_language, _)| known == &directory && *known_language == language)
            .map(|(_, _, set)| set.clone());
        if let Some(set) = cached {
            return Ok(set);
        }
        let set = load_dictionaries(&directory, language)?;
        self.dictionaries.push((directory, language, set.clone()));
        self.loads += 1;
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use opensvr_core::{Extra, NoteGroup, PitchAttributes};

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("opensvr-phonemes-{}-{name}", std::process::id()))
    }

    /// The smallest Japanese dictionary the resolver accepts, copied from the
    /// CLI's phoneme test.
    fn write_dictionary(directory: &Path) {
        fs::create_dir_all(directory).unwrap();
        fs::write(
            directory.join("japanese-romaji-phones.txt"),
            "a vowel\nk consonant\nsil silence\n",
        )
        .unwrap();
        fs::write(directory.join("japanese-romaji-dict.txt"), "ka k a\n").unwrap();
        fs::write(directory.join("japanese-hira2romaji-dict.txt"), "か ka\n").unwrap();
        fs::write(directory.join("japanese-kata2romaji-dict.txt"), "カ ka\n").unwrap();
        fs::write(directory.join("japanese-sute2romaji-dict.txt"), "ゃ ya\n").unwrap();
    }

    fn note(lyrics: &str) -> Note {
        Note {
            onset: 0,
            duration: 1,
            pitch: 60,
            lyrics: lyrics.into(),
            phonemes: String::new(),
            detune: 0.0,
            instant_mode: true,
            musical_type: "singing".into(),
            accent: String::new(),
            attributes: PitchAttributes::default(),
            system_attributes: PitchAttributes::default(),
            extra: Extra::default(),
        }
    }

    fn track(notes: Vec<Note>, directory: Option<PathBuf>) -> Track {
        let mut track = Track::new(
            "Lead",
            NoteGroup {
                id: "main".into(),
                name: "main".into(),
                notes,
                ..NoteGroup::default()
            },
        );
        track.voice.language = Language::Japanese;
        track.voice.dictionary = directory;
        track
    }

    #[test]
    fn resolves_notes_in_order_with_legato() {
        let directory = scratch("basic");
        write_dictionary(&directory);
        let track = track(
            vec![note("か"), note("-"), note("zzz")],
            Some(directory.clone()),
        );
        let mut resolver = PhonemeResolver::new();
        let entries = resolver.resolve_track(&track).unwrap();

        let first = entries[0].result.as_ref().unwrap();
        assert_eq!(first.phonemes, ["k", "a"]);
        assert_eq!(first.language, Language::Japanese);
        // `-` re-sings the previous vowel without a dictionary lookup.
        let legato = entries[1].result.as_ref().unwrap();
        assert_eq!(legato.phonemes, ["a"]);
        // A miss is reported on its note, not returned from the track call.
        let error = entries[2].result.as_ref().unwrap_err().to_string();
        assert!(
            error.contains("No pronunciation for dictionary key 'zzz'."),
            "{error}"
        );
        assert_eq!(resolver.dictionary_loads(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn caches_one_load_per_directory_and_language() {
        let first = scratch("cache-a");
        let second = scratch("cache-b");
        write_dictionary(&first);
        write_dictionary(&second);
        let mut resolver = PhonemeResolver::new();

        let a = track(vec![note("か")], Some(first.clone()));
        let b = track(vec![note("か")], Some(first));
        let c = track(vec![note("か")], Some(second.clone()));
        for track in [&a, &b, &c] {
            resolver.resolve_track(track).unwrap();
        }
        assert_eq!(
            resolver.dictionary_loads(),
            2,
            "a shared directory loads once; a new one loads again"
        );
        fs::remove_dir_all(second).unwrap();
    }

    #[test]
    fn an_empty_track_needs_no_dictionary() {
        let mut resolver = PhonemeResolver::new();
        let empty = track(Vec::new(), None);
        let entries = resolver.resolve_track(&empty).unwrap();
        assert!(entries.is_empty());
        assert_eq!(resolver.dictionary_loads(), 0);
    }

    #[test]
    fn reports_a_missing_dictionary_directory() {
        let mut resolver = PhonemeResolver::new();
        let error = resolver
            .resolve_track(&track(vec![note("か")], None))
            .unwrap_err();
        assert!(matches!(error, PhonemeError::NoDictionary), "{error}");
    }

    #[test]
    fn reports_an_unreadable_dictionary() {
        let directory = scratch("empty");
        fs::create_dir_all(&directory).unwrap();
        let mut resolver = PhonemeResolver::new();
        let error = resolver
            .resolve_track(&track(vec![note("か")], Some(directory.clone())))
            .unwrap_err();
        assert!(matches!(error, PhonemeError::Dictionaries(_)), "{error}");
        fs::remove_dir_all(directory).unwrap();
    }
}
