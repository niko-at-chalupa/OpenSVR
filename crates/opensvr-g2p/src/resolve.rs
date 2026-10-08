//! Note lyrics to phonemes (`ProjectRenderer::resolvePhonemes` and friends).
//!
//! Ports `ProjectRenderer::resolvePhonemes` (`src/audio/ProjectRenderer.cpp`)
//! with its helpers (`dictionaryStem`, `isEnglishVowelPhoneme`,
//! `splitEnglishSyllables`, `deferTrailingEnglishCoda`). Precedence, in order:
//!
//! 1. Explicit `phonemes` win; a leading `.` in lyrics means raw phonemes
//!    (whitespace-split, NUL-rejected, non-empty).
//! 2. Otherwise dictionary lookup in the track language, except Latin words
//!    (`[A-Za-z'-]+`) in Japanese tracks, which try the English dictionary
//!    first (uppercase-start and `prefer_english` rules).
//!
//! File selection mirrors the C++ version: `<stem>-phones.txt` plus
//! `<stem>-dict.txt` (`cmudict-07b.txt` preferred for English), kana tables
//! for Japanese, `cedict.txt` for Mandarin. Unlike the C++ renderer there is
//! no file-stamp cache here — dictionaries are loaded once per render at the
//! caller, which is all the one-shot CLI needs (the `CachedDictionary` layer
//! is Phase 6). [`NoteResolver`] additionally ports the `+`/`-` lyric
//! sequencing from the phrase loop so per-note callers stay faithful.

use std::path::Path;

use opensvr_core::Language;

use crate::{G2pError, PhonemeDictionary, Result, invalid};

/// Phonemes resolved for one note: the symbols, the continuation carrier for
/// a following `-` legato, and the language actually used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub phonemes: Vec<String>,
    pub continuation: String,
    pub language: Language,
}

/// Loaded dictionaries for one track language: the local dictionary plus the
/// English one (Japanese Latin-word fallback; best-effort, `None` when its
/// files are absent).
#[derive(Debug, Clone, Default)]
pub struct DictionarySet {
    pub local: PhonemeDictionary,
    pub english: Option<PhonemeDictionary>,
}

/// Returns the `clf-data` file stem for a language.
///
/// Mirrors C++ `dictionaryStem`: japanese maps to `japanese-romaji`,
/// mandarin to `mandarin-xsampa`, english to `english-arpabet`, cantonese to
/// `cantonese-xsampa` and spanish to `spanish-xsampa`.
pub fn dictionary_stem(language: Language) -> &'static str {
    match language {
        Language::Japanese => "japanese-romaji",
        Language::Mandarin => "mandarin-xsampa",
        Language::English => "english-arpabet",
        Language::Cantonese => "cantonese-xsampa",
        Language::Spanish => "spanish-xsampa",
    }
}

/// Returns true for the 16 ARPABET vowels that carry English continuations.
pub fn is_english_vowel_phoneme(symbol: &str) -> bool {
    matches!(
        symbol,
        "aa" | "ae"
            | "ah"
            | "ao"
            | "aw"
            | "ax"
            | "ay"
            | "eh"
            | "er"
            | "ey"
            | "ih"
            | "iy"
            | "ow"
            | "oy"
            | "uh"
            | "uw"
    )
}

/// Splits an English word pronunciation at syllable boundaries.
///
/// Each syllable keeps its onset consonants except that a consonant cluster
/// before a nucleus gives its last consonant to the next syllable
/// (maximal-onset with one consonant). A single nucleus stays whole.
pub fn split_english_syllables(phonemes: &[String]) -> Vec<Vec<String>> {
    let mut nuclei = Vec::new();
    for (index, symbol) in phonemes.iter().enumerate() {
        if is_english_vowel_phoneme(symbol) {
            nuclei.push(index);
        }
    }
    if nuclei.len() < 2 {
        return vec![phonemes.to_vec()];
    }
    let mut syllables = Vec::with_capacity(nuclei.len());
    let mut start = 0;
    for window in nuclei.windows(2) {
        let consonants = window[1] - window[0] - 1;
        let next_start = if consonants == 0 {
            window[1]
        } else {
            window[1] - 1
        };
        syllables.push(phonemes[start..next_start].to_vec());
        start = next_start;
    }
    syllables.push(phonemes[start..].to_vec());
    syllables
}

/// Moves an English trailing coda off `phonemes` into `deferred`.
///
/// Only applies past the last vowel; vowel-less words keep every phone.
fn defer_trailing_english_coda(phonemes: &mut Vec<String>, deferred: &mut Vec<String>) {
    let Some(vowel) = phonemes.iter().rposition(|symbol| is_english_vowel_phoneme(symbol)) else {
        return;
    };
    if vowel + 1 < phonemes.len() {
        deferred.extend_from_slice(&phonemes[vowel + 1..]);
        phonemes.truncate(vowel + 1);
    }
}

fn cannot_read(path: &Path) -> G2pError {
    invalid(format!(
        "Cannot read synthesis resource: {}",
        path.display()
    ))
}

fn last_vowel_or_tail(phonemes: &[String], dictionary: &PhonemeDictionary) -> String {
    if phonemes.is_empty() {
        return String::new();
    }
    let definitions = dictionary.phonemes();
    for phone in phonemes.iter().rev() {
        if definitions.iter().any(|definition| {
            &definition.symbol == phone
                && (definition.category == "vowel" || definition.category == "diphthong")
        }) {
            return phone.clone();
        }
    }
    phonemes.last().cloned().unwrap_or_default()
}

fn english_fallback_available(dir: &Path) -> bool {
    dir.join("english-arpabet-phones.txt").is_file()
        && (dir.join("cmudict-07b.txt").is_file()
            || dir.join("english-arpabet-dict.txt").is_file())
}

fn load_english(dir: &Path) -> Option<PhonemeDictionary> {
    let phones = dir.join("english-arpabet-phones.txt");
    let cmu = dir.join("cmudict-07b.txt");
    let dictionary = if cmu.is_file() {
        cmu
    } else {
        dir.join("english-arpabet-dict.txt")
    };
    PhonemeDictionary::load(&phones, &dictionary).ok()
}

/// Loads the dictionaries for one track language from a `clf-data` directory.
///
/// The English dictionary loads best-effort (as in the C++ renderer, where a
/// failed English load only disables the Japanese Latin-word fallback).
/// An empty directory errors; missing local files error with the C++
/// `readFileStamp` text.
pub fn load_dictionaries(dir: &Path, language: Language) -> Result<DictionarySet> {
    if dir.as_os_str().is_empty() {
        return Err(invalid(
            "Select the clf-data pronunciation dictionary directory or enter explicit phonemes.",
        ));
    }
    let stem = dictionary_stem(language);
    let phones = dir.join(format!("{stem}-phones.txt"));
    let mut dictionary = dir.join(format!("{stem}-dict.txt"));
    if language == Language::English && dir.join("cmudict-07b.txt").is_file() {
        dictionary = dir.join("cmudict-07b.txt");
    }
    if !phones.is_file() {
        return Err(cannot_read(&phones));
    }
    if !dictionary.is_file() {
        return Err(cannot_read(&dictionary));
    }
    let local = match language {
        Language::Mandarin => {
            let cedict = dir.join("cedict.txt");
            if !cedict.is_file() {
                return Err(cannot_read(&cedict));
            }
            PhonemeDictionary::load_mandarin(&phones, &dictionary, &cedict)?
        }
        Language::Japanese => {
            let hiragana = dir.join("japanese-hira2romaji-dict.txt");
            let katakana = dir.join("japanese-kata2romaji-dict.txt");
            let small_kana = dir.join("japanese-sute2romaji-dict.txt");
            for file in [&hiragana, &katakana, &small_kana] {
                if !file.is_file() {
                    return Err(cannot_read(file));
                }
            }
            PhonemeDictionary::load_japanese(&phones, &dictionary, &hiragana, &katakana, &small_kana)?
        }
        _ => PhonemeDictionary::load(&phones, &dictionary)?,
    };
    let english = if language == Language::Japanese && english_fallback_available(dir) {
        load_english(dir)
    } else {
        None
    };
    Ok(DictionarySet { local, english })
}

/// Resolves one note's lyrics (or explicit override) to phonemes.
///
/// `prefer_english` is the renderer's continuation language check: the
/// previous note resolved to English. Error strings match the C++
///
/// `resolvePhonemes` text.
pub fn resolve(
    lyrics: &str,
    explicit: &str,
    language: Language,
    prefer_english: bool,
    dictionaries: &DictionarySet,
) -> Result<Resolved> {
    let mut explicit_text = explicit;
    let mut dotted = false;
    if explicit_text.is_empty() && lyrics.starts_with('.') {
        explicit_text = &lyrics[1..];
        dotted = true;
    }
    if !explicit_text.is_empty() || dotted {
        if explicit_text.contains('\0') {
            return Err(invalid("Explicit phonemes must be valid UTF-8 text."));
        }
        let phonemes: Vec<String> = explicit_text
            .split([' ', '\t', '\r', '\n'])
            .filter(|field| !field.is_empty())
            .map(str::to_owned)
            .collect();
        if phonemes.is_empty() {
            return Err(invalid("The explicit phoneme sequence is empty."));
        }
        let mut continuation = phonemes.last().cloned().unwrap_or_default();
        if language == Language::English
            && let Some(vowel) = phonemes.iter().rev().find(|symbol| is_english_vowel_phoneme(symbol))
        {
            continuation = vowel.clone();
        }
        return Ok(Resolved {
            phonemes,
            continuation,
            language,
        });
    }

    let latin_word = !lyrics.is_empty()
        && lyrics
            .bytes()
            .all(|b| b.is_ascii_alphabetic() || b == b'\'' || b == b'-');
    if language == Language::Japanese
        && latin_word
        && let Some(english) = &dictionaries.english
    {
        let local_has = dictionaries.local.has_entry(lyrics);
        let starts_uppercase = lyrics.as_bytes().first().is_some_and(u8::is_ascii_uppercase);
        if english.has_entry(lyrics) && (prefer_english || starts_uppercase || !local_has) {
            let phonemes = english.lookup(lyrics)?;
            let continuation = last_vowel_or_tail(&phonemes, english);
            return Ok(Resolved {
                phonemes,
                continuation,
                language: Language::English,
            });
        }
    }

    match dictionaries.local.lookup(lyrics) {
        Ok(phonemes) => {
            let continuation = last_vowel_or_tail(&phonemes, &dictionaries.local);
            Ok(Resolved {
                phonemes,
                continuation,
                language,
            })
        }
        Err(local_error) => {
            if language == Language::Japanese
                && latin_word
                && let Some(english) = &dictionaries.english
                && let Ok(phonemes) = english.lookup(lyrics)
            {
                let continuation = last_vowel_or_tail(&phonemes, english);
                return Ok(Resolved {
                    phonemes,
                    continuation,
                    language: Language::English,
                });
            }
            Err(local_error)
        }
    }
}

/// Per-note lyric sequencing: `+` syllable breaks and `-` legatos.
///
/// Ports the phrase-loop lyric handling around `resolvePhonemes` so callers
/// without timing information (such as `info --phonemes`) resolve notes the
/// same way the renderer does. Multi-syllable English words split across
/// following `+` notes; `-` repeats the continuation carrier; a trailing
/// English coda is deferred onto the final legato when the caller reports it
/// via `next_is_legato`.
#[derive(Debug, Clone)]
pub struct NoteResolver {
    language: Language,
    continuation: String,
    continuation_language: Language,
    word_language: Language,
    pending_syllables: Vec<Vec<String>>,
    next_syllable: usize,
    pending_coda: Vec<String>,
}

impl NoteResolver {
    pub fn new(language: Language) -> Self {
        Self {
            language,
            continuation: String::new(),
            continuation_language: language,
            word_language: language,
            pending_syllables: Vec::new(),
            next_syllable: 0,
            pending_coda: Vec::new(),
        }
    }

    /// Resolves one note in sequence; state carries to the following note.
    pub fn push(
        &mut self,
        lyrics: &str,
        explicit: &str,
        dictionaries: &DictionarySet,
        next_is_legato: bool,
    ) -> Result<Resolved> {
        let prefer_english = self.continuation_language == Language::English;
        let is_break = lyrics == "+";
        let is_legato = lyrics == "-";
        let mut resolved = if is_legato {
            self.legato(explicit, dictionaries, prefer_english)?
        } else if is_break {
            self.break_note(explicit, dictionaries, prefer_english)?
        } else {
            resolve(lyrics, explicit, self.language, prefer_english, dictionaries)?
        };

        let automatic_english = resolved.language == Language::English
            && explicit.is_empty()
            && !lyrics.is_empty()
            && !lyrics.starts_with('.');
        if !is_break && !is_legato {
            self.word_language = resolved.language;
            self.pending_syllables.clear();
            self.next_syllable = 0;
            if resolved.language == Language::English && !lyrics.is_empty() && !lyrics.starts_with('.') {
                let mut word_phonemes = resolved.phonemes.clone();
                if !explicit.is_empty()
                    && let Ok(inferred) = resolve(lyrics, "", self.language, prefer_english, dictionaries)
                    && inferred.language == Language::English
                {
                    word_phonemes = inferred.phonemes;
                }
                let syllables = split_english_syllables(&word_phonemes);
                if syllables.len() > 1 {
                    self.pending_syllables = syllables[1..].to_vec();
                    if explicit.is_empty() {
                        resolved.phonemes = syllables.into_iter().next().expect("split is nonempty");
                    }
                }
            }
        }
        if !is_legato {
            self.pending_coda.clear();
            if automatic_english && next_is_legato {
                defer_trailing_english_coda(&mut resolved.phonemes, &mut self.pending_coda);
            }
        } else if !next_is_legato && !self.pending_coda.is_empty() {
            resolved.phonemes.extend(self.pending_coda.iter().cloned());
            self.pending_coda.clear();
        }

        self.continuation_language = resolved.language;
        // resolve() selects the continuation (last vowel/diphthong, else the
        // tail); English prefers its own vowel table on top of that.
        self.continuation = resolved.continuation.clone();
        if resolved.language == Language::English
            && !resolved.phonemes.is_empty()
            && let Some(vowel) = resolved
                .phonemes
                .iter()
                .rev()
                .find(|symbol| is_english_vowel_phoneme(symbol))
        {
            self.continuation = vowel.clone();
            resolved.continuation = vowel.clone();
        }
        if resolved.phonemes.len() == 1
            && (resolved.phonemes[0] == "sil" || resolved.phonemes[0] == "br")
        {
            self.continuation.clear();
            self.continuation_language = self.language;
            self.word_language = self.language;
            self.pending_syllables.clear();
            self.next_syllable = 0;
        }
        Ok(resolved)
    }

    fn legato(
        &mut self,
        explicit: &str,
        dictionaries: &DictionarySet,
        prefer_english: bool,
    ) -> Result<Resolved> {
        if self.continuation.is_empty() && explicit.is_empty() {
            return Err(invalid(
                "A legato lyric (-) must immediately follow a note with a resolved syllable.",
            ));
        }
        if !explicit.is_empty() {
            let mut resolved = resolve("-", explicit, self.language, prefer_english, dictionaries)?;
            resolved.language = self.continuation_language;
            if resolved.language == Language::English {
                if let Some(vowel) = resolved
                    .phonemes
                    .iter()
                    .rev()
                    .find(|symbol| is_english_vowel_phoneme(symbol))
                {
                    self.continuation = vowel.clone();
                    resolved.continuation = vowel.clone();
                }
            } else if let Some(last) = resolved.phonemes.last() {
                self.continuation = last.clone();
                resolved.continuation = last.clone();
            }
            return Ok(resolved);
        }
        Ok(Resolved {
            phonemes: vec![self.continuation.clone()],
            continuation: self.continuation.clone(),
            language: self.continuation_language,
        })
    }

    fn break_note(
        &mut self,
        explicit: &str,
        dictionaries: &DictionarySet,
        prefer_english: bool,
    ) -> Result<Resolved> {
        if !explicit.is_empty() {
            let mut resolved = resolve("+", explicit, self.language, prefer_english, dictionaries)?;
            resolved.language = self.word_language;
            if self.next_syllable < self.pending_syllables.len() {
                self.next_syllable += 1;
            }
            return Ok(resolved);
        }
        if self.next_syllable >= self.pending_syllables.len() {
            return Err(invalid(
                "'+' advances to the next syllable, but no remaining syllable is available; enter its phonemes explicitly if needed.",
            ));
        }
        let phonemes = self.pending_syllables[self.next_syllable].clone();
        self.next_syllable += 1;
        Ok(Resolved {
            phonemes,
            continuation: self.continuation.clone(),
            language: self.word_language,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temp_dir_for_tests;

    const PHONES: &str = "\
aa vowel\nae vowel\nah vowel\nao vowel\naw diphthong\nax vowel\nay diphthong\n\
eh vowel\ner vowel\ney diphthong\nih vowel\niy vowel\now diphthong\noy diphthong\n\
uh vowel\nuw vowel\nb consonant\nch consonant\nd consonant\ndh consonant\nf consonant\n\
g consonant\nhh consonant\njh consonant\nk consonant\nl consonant\nm consonant\n\
n consonant\nng consonant\np consonant\nr consonant\ns consonant\nsh consonant\n\
t consonant\nth consonant\nv consonant\nw consonant\ny consonant\nz consonant\n\
sil silence\nbr breath\n";

    const ROMAJI_PHONES: &str = "\
a vowel\ni vowel\nu vowel\ne vowel\no vowel\nk consonant\ns consonant\n\
sh consonant\nt consonant\nsil silence\nbr breath\n";

    fn parse_generic(phones: &str, dict: &str) -> PhonemeDictionary {
        PhonemeDictionary::parse(
            phones.as_bytes(),
            dict.as_bytes(),
            "phones.txt",
            "dict.txt",
            "x-phones.txt".to_owned(),
            "x-dict.txt".to_owned(),
        )
        .expect("synthetic dictionary parses")
    }

    fn english_dict() -> PhonemeDictionary {
        PhonemeDictionary::parse(
            PHONES.as_bytes(),
            "hello hh ah l ow\ncats k ae t s\n".as_bytes(),
            "english-arpabet-phones.txt",
            "english-arpabet-dict.txt",
            "english-arpabet-phones.txt".to_owned(),
            "english-arpabet-dict.txt".to_owned(),
        )
        .expect("synthetic english dictionary parses")
    }

    fn japanese_set(local_dict: &str) -> DictionarySet {
        // Local entries must use romaji inventory symbols (the arpabet-only
        // dictionary stands in for the English fallback).
        DictionarySet {
            local: parse_generic(ROMAJI_PHONES, local_dict),
            english: Some(english_dict()),
        }
    }

    #[test]
    fn stems_cover_all_languages() {
        assert_eq!(dictionary_stem(Language::Japanese), "japanese-romaji");
        assert_eq!(dictionary_stem(Language::Mandarin), "mandarin-xsampa");
        assert_eq!(dictionary_stem(Language::English), "english-arpabet");
        assert_eq!(dictionary_stem(Language::Cantonese), "cantonese-xsampa");
        assert_eq!(dictionary_stem(Language::Spanish), "spanish-xsampa");
    }

    #[test]
    fn english_vowel_table() {
        for vowel in [
            "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy",
            "ow", "oy", "uh", "uw",
        ] {
            assert!(is_english_vowel_phoneme(vowel), "{vowel}");
        }
        for other in ["", "k", "s", "sh", "sil", "AA", "ah0"] {
            assert!(!is_english_vowel_phoneme(other), "{other}");
        }
    }

    #[test]
    fn syllable_splitting_cases() {
        let words = |phones: &[&str]| {
            split_english_syllables(&phones.iter().map(|phone| (*phone).to_owned()).collect::<Vec<_>>())
        };
        // Single nucleus stays whole.
        assert_eq!(words(&["hh", "ah", "l"]), vec![vec!["hh", "ah", "l"]]);
        // One intervening consonant moves to the next syllable.
        assert_eq!(
            words(&["hh", "ah", "l", "ow"]),
            vec![vec!["hh", "ah"], vec!["l", "ow"]]
        );
        // Adjacent nuclei split between them.
        assert_eq!(
            words(&["ae", "ow"]),
            vec![vec!["ae"], vec!["ow"]]
        );
        // A cluster gives its last consonant onward.
        assert_eq!(
            words(&["s", "t", "r", "iy", "k", "ey"]),
            vec![vec!["s", "t", "r", "iy"], vec!["k", "ey"]]
        );
        // Vowel-less words stay whole.
        assert_eq!(words(&["s", "t"]), vec![vec!["s", "t"]]);
    }

    #[test]
    fn explicit_phonemes_win_over_the_dictionary() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let resolved = resolve("ka", "sh i", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.phonemes, ["sh", "i"]);
        assert_eq!(resolved.continuation, "i");
        assert_eq!(resolved.language, Language::Japanese);
    }

    #[test]
    fn dotted_lyrics_are_raw_phonemes() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let resolved = resolve(".k a", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.phonemes, ["k", "a"]);
    }

    #[test]
    fn empty_explicit_sequences_are_rejected() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        for (lyrics, explicit) in [("ka", "   "), (".", ""), (".  \t", "")] {
            assert_eq!(
                resolve(lyrics, explicit, Language::Japanese, false, &dictionaries)
                    .unwrap_err()
                    .to_string(),
                "The explicit phoneme sequence is empty.",
                "{lyrics:?}/{explicit:?}"
            );
        }
    }

    #[test]
    fn explicit_nul_is_rejected() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        assert_eq!(
            resolve("ka", "k\0a", Language::Japanese, false, &dictionaries)
                .unwrap_err()
                .to_string(),
            "Explicit phonemes must be valid UTF-8 text."
        );
    }

    #[test]
    fn explicit_english_continuation_is_the_last_vowel() {
        let dictionaries = DictionarySet {
            local: english_dict(),
            english: None,
        };
        let resolved = resolve("hello", "k ae t s", Language::English, false, &dictionaries).unwrap();
        assert_eq!(resolved.continuation, "ae");
        // No vowel at all: the tail carries the legato.
        let resolved = resolve("x", "k s", Language::English, false, &dictionaries).unwrap();
        assert_eq!(resolved.continuation, "s");
    }

    #[test]
    fn lowercase_latin_prefers_the_local_japanese_entry() {
        // Local dictionary knows "hashi" too, word starts lowercase, no
        // English context: the Japanese reading wins.
        let dictionaries = japanese_set("hashi sh a sh i\nka k a\n");
        let resolved = resolve("hashi", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.language, Language::Japanese);
        assert_eq!(resolved.phonemes, ["sh", "a", "sh", "i"]);
        // Continuation is the last vowel/diphthong, not the tail consonant.
        assert_eq!(resolved.continuation, "i");
    }

    #[test]
    fn uppercase_latin_start_takes_english() {
        let dictionaries = japanese_set("hello sh i\nka k a\n");
        let resolved = resolve("Hello", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.language, Language::English);
        assert_eq!(resolved.phonemes, ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn prefer_english_context_takes_english() {
        let dictionaries = japanese_set("hello sh i\nka k a\n");
        let resolved = resolve("hello", "", Language::Japanese, true, &dictionaries).unwrap();
        assert_eq!(resolved.language, Language::English);
    }

    #[test]
    fn english_fallback_applies_when_local_misses() {
        // "hello" is not in the local dictionary at all: even lowercase
        // without context falls through to English on the second attempt.
        let dictionaries = japanese_set("ka k a\n");
        let resolved = resolve("hello", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.language, Language::English);
        assert_eq!(resolved.phonemes, ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn local_miss_without_english_reports_the_local_error() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        assert_eq!(
            resolve("zzz", "", Language::Japanese, false, &dictionaries)
                .unwrap_err()
                .to_string(),
            "No pronunciation for dictionary key 'zzz'."
        );
    }

    #[test]
    fn continuation_prefers_vowels_over_the_tail() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ask a s k\n"),
            english: None,
        };
        let resolved = resolve("ask", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(resolved.continuation, "a");
    }

    #[test]
    fn loader_rejects_an_empty_directory() {
        assert_eq!(
            load_dictionaries(Path::new(""), Language::Japanese)
                .unwrap_err()
                .to_string(),
            "Select the clf-data pronunciation dictionary directory or enter explicit phonemes."
        );
    }

    #[test]
    fn loader_reports_missing_files_like_read_file_stamp() {
        let dir = temp_dir_for_tests("resolve-missing");
        let error = load_dictionaries(&dir, Language::English).unwrap_err().to_string();
        assert!(error.starts_with("Cannot read synthesis resource: "), "{error}");
        assert!(error.ends_with("english-arpabet-phones.txt"), "{error}");
    }

    #[test]
    fn loader_prefers_cmudict_for_english() {
        let dir = temp_dir_for_tests("resolve-cmu");
        std::fs::write(dir.join("english-arpabet-phones.txt"), PHONES).unwrap();
        std::fs::write(dir.join("english-arpabet-dict.txt"), "fallbackonly f ao l ow\n").unwrap();
        std::fs::write(dir.join("cmudict-07b.txt"), "cmuonly k ey m y uw\n").unwrap();
        let dictionaries = load_dictionaries(&dir, Language::English).unwrap();
        assert!(dictionaries.local.has_entry("cmuonly"));
        assert!(!dictionaries.local.has_entry("fallbackonly"));
    }

    #[test]
    fn loader_falls_back_to_the_plain_english_dict() {
        let dir = temp_dir_for_tests("resolve-plain");
        std::fs::write(dir.join("english-arpabet-phones.txt"), PHONES).unwrap();
        std::fs::write(dir.join("english-arpabet-dict.txt"), "fallbackonly f ao l ow\n").unwrap();
        let dictionaries = load_dictionaries(&dir, Language::English).unwrap();
        assert!(dictionaries.local.has_entry("fallbackonly"));
    }

    #[test]
    fn loader_requires_kana_tables_for_japanese() {
        let dir = temp_dir_for_tests("resolve-jp-partial");
        std::fs::write(dir.join("japanese-romaji-phones.txt"), ROMAJI_PHONES).unwrap();
        std::fs::write(dir.join("japanese-romaji-dict.txt"), "ka k a\n").unwrap();
        let error = load_dictionaries(&dir, Language::Japanese).unwrap_err().to_string();
        assert!(error.contains("japanese-hira2romaji-dict.txt"), "{error}");
    }

    #[test]
    fn loader_wires_japanese_and_english_together() {
        let dir = temp_dir_for_tests("resolve-jp-full");
        std::fs::write(dir.join("japanese-romaji-phones.txt"), ROMAJI_PHONES).unwrap();
        std::fs::write(dir.join("japanese-romaji-dict.txt"), "ka k a\n").unwrap();
        std::fs::write(dir.join("japanese-hira2romaji-dict.txt"), "か ka\n").unwrap();
        std::fs::write(dir.join("japanese-kata2romaji-dict.txt"), "カ ka\n").unwrap();
        std::fs::write(dir.join("japanese-sute2romaji-dict.txt"), "ゃ ya\n").unwrap();
        std::fs::write(dir.join("english-arpabet-phones.txt"), PHONES).unwrap();
        std::fs::write(dir.join("cmudict-07b.txt"), "hello hh ah l ow\n").unwrap();
        let dictionaries = load_dictionaries(&dir, Language::Japanese).unwrap();
        assert!(dictionaries.english.is_some());
        let kana = resolve("か", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(kana.phonemes, ["k", "a"]);
        let latin = resolve("Hello", "", Language::Japanese, false, &dictionaries).unwrap();
        assert_eq!(latin.language, Language::English);
    }

    #[test]
    fn resolver_repeats_continuations_on_legato() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::Japanese);
        let first = resolver.push("ka", "", &dictionaries, false).unwrap();
        assert_eq!(first.phonemes, ["k", "a"]);
        let legato = resolver.push("-", "", &dictionaries, false).unwrap();
        assert_eq!(legato.phonemes, ["a"]);
        assert_eq!(legato.language, Language::Japanese);
    }

    #[test]
    fn legato_without_a_syllable_is_rejected() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::Japanese);
        assert_eq!(
            resolver
                .push("-", "", &dictionaries, false)
                .unwrap_err()
                .to_string(),
            "A legato lyric (-) must immediately follow a note with a resolved syllable."
        );
    }

    #[test]
    fn legato_with_an_explicit_override_resolves_it() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::Japanese);
        resolver.push("ka", "", &dictionaries, false).unwrap();
        let legato = resolver.push("-", "s u", &dictionaries, false).unwrap();
        assert_eq!(legato.phonemes, ["s", "u"]);
        // Continuation follows the override for the next legato.
        let again = resolver.push("-", "", &dictionaries, false).unwrap();
        assert_eq!(again.phonemes, ["u"]);
    }

    #[test]
    fn plus_consumes_pending_english_syllables() {
        let dictionaries = DictionarySet {
            local: english_dict(),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::English);
        let first = resolver.push("hello", "", &dictionaries, false).unwrap();
        assert_eq!(first.phonemes, ["hh", "ah"]);
        let second = resolver.push("+", "", &dictionaries, false).unwrap();
        assert_eq!(second.phonemes, ["l", "ow"]);
        assert_eq!(second.language, Language::English);
    }

    #[test]
    fn plus_without_pending_syllables_is_rejected() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\n"),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::Japanese);
        resolver.push("ka", "", &dictionaries, false).unwrap();
        assert_eq!(
            resolver
                .push("+", "", &dictionaries, false)
                .unwrap_err()
                .to_string(),
            "'+' advances to the next syllable, but no remaining syllable is available; enter its phonemes explicitly if needed."
        );
    }

    #[test]
    fn english_coda_defers_onto_the_final_legato() {
        let dictionaries = DictionarySet {
            local: english_dict(),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::English);
        let first = resolver.push("cats", "", &dictionaries, true).unwrap();
        // Everything past the last vowel ([t, s]) defers to the legato.
        assert_eq!(first.phonemes, ["k", "ae"]);
        let legato = resolver.push("-", "", &dictionaries, false).unwrap();
        assert_eq!(legato.phonemes, ["ae", "t", "s"]);
    }

    #[test]
    fn silence_resets_the_continuation() {
        let dictionaries = DictionarySet {
            local: parse_generic(ROMAJI_PHONES, "ka k a\nrest sil\n"),
            english: None,
        };
        let mut resolver = NoteResolver::new(Language::Japanese);
        resolver.push("ka", "", &dictionaries, false).unwrap();
        let rest = resolver.push("rest", "", &dictionaries, false).unwrap();
        assert_eq!(rest.phonemes, ["sil"]);
        assert!(
            resolver
                .push("-", "", &dictionaries, false)
                .unwrap_err()
                .to_string()
                .contains("legato")
        );
    }
}
