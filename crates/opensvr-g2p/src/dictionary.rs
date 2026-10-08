//! Multi-language pronouncing dictionaries (`PhonemeDictionary`).
//!
//! Ports OpenSV's `src/synthesis/PhonemeDictionary.{h,cpp}`: parsing of the
//! two-column `clf-data` text files (phones inventory plus word map), the
//! Japanese kana tables, Mandarin pinyin normalization with CC-CEDICT hanzi
//! mappings, English CMUDict handling (stress stripping, alias normalization,
//! variant suffixes), and dictionary `lookup` with its English fallback chain
//! (common words, then rule-based grapheme heuristics).

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use crate::{Result, G2pError, invalid, line_error};

/// One phoneme of the active language: its token and phonetic category.
///
/// The `"vowel"` / `"diphthong"` categories drive continuation selection in
/// [`crate::resolve`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeDefinition {
    pub symbol: String,
    pub category: String,
}

/// Grapheme-to-phoneme dictionary for one language pipeline.
///
/// Loaded once per `(phones, dictionary)` file pair (plus kana or CEDICT
/// tables for Japanese and Mandarin); `lookup` is then a read-only query.
/// Files are read with [`PhonemeDictionary::load`] and friends; parsing is
/// available over bytes via [`PhonemeDictionary::parse`] for tests.
#[derive(Debug, Clone, Default)]
pub struct PhonemeDictionary {
    phonemes: Vec<PhonemeDefinition>,
    symbols: HashSet<String>,
    entries: HashMap<String, Vec<String>>,
    kana_to_romaji: HashMap<String, String>,
    small_kana_to_romaji: HashMap<String, String>,
    is_mandarin: bool,
    is_japanese: bool,
    is_english_arpabet: bool,
}

/// Spelled-out ARPABET for single English capital letters (`A` -> `EY`, ...).
const ENGLISH_LETTER_NAMES: [&str; 26] = [
    "ey", "b iy", "s iy", "d iy", "iy", "eh f", "jh iy", "ey ch", "ay", "jh ey", "k ey",
    "eh l", "eh m", "eh n", "ow", "p iy", "k y uw", "aa r", "eh s", "t iy", "y uw",
    "v iy", "d ah b ah l y uw", "eh k s", "w ay", "z iy",
];

/// Hardcoded pronunciations for common English words missing from the files.
const ENGLISH_FALLBACKS: [(&str, &[&str]); 14] = [
    ("a", &["ae"]),
    ("i", &["ay"]),
    ("hello", &["hh", "ah", "l", "ow"]),
    ("chat", &["ch", "ae", "t"]),
    ("hate", &["hh", "ey", "t"]),
    ("now", &["n", "aw"]),
    ("hua", &["hh", "w", "ah"]),
    ("kanru", &["k", "ae", "n", "r", "uw"]),
    ("hue", &["hh", "y", "uw"]),
    ("you", &["y", "uw"]),
    ("the", &["dh", "ah"]),
    ("she", &["sh", "iy"]),
    ("we", &["w", "iy"]),
    ("me", &["m", "iy"]),
];

/// Digraph/long-vowel patterns of the heuristic English G2P, in match order.
const ENGLISH_PATTERNS: [(&str, &str); 11] = [
    ("sh", "sh"),
    ("ch", "ch"),
    ("th", "th"),
    ("ph", "f"),
    ("ng", "ng"),
    ("qu", "kw"),
    ("ee", "iy"),
    ("ea", "iy"),
    ("oo", "uw"),
    ("ow", "aw"),
    ("ai", "ey"),
];

/// Historical CMU phone variants mapped to standard SV ARPABET symbols.
fn normalize_english_phone(symbol: &mut String) {
    if let Some(last @ ('0'..='2')) = symbol.chars().last() {
        let _ = last;
        symbol.pop();
    }
    let alias = match symbol.as_str() {
        "ax" | "ax-h" => Some("ah"),
        "axr" => Some("er"),
        "el" => Some("l"),
        "em" => Some("m"),
        "en" => Some("n"),
        "eng" => Some("ng"),
        "hv" => Some("hh"),
        "ix" => Some("ih"),
        "nx" => Some("n"),
        "ux" => Some("uw"),
        "wh" => Some("w"),
        _ => None,
    };
    if let Some(target) = alias {
        symbol.clone_from(&target.to_owned());
    }
}

/// Lowercases an English key and strips CMUDict variant suffixes (`w(2)`).
fn normalize_english_key(mut key: String) -> String {
    key = key.to_lowercase();
    if key.ends_with(')')
        && let Some(open) = key.rfind('(')
        && open + 2 < key.len()
        && key[open + 1..key.len() - 1].bytes().all(|b| b.is_ascii_digit())
    {
        key.truncate(open);
    }
    key
}

/// Lowercases pinyin, maps `u:`/`ü` to `v`, strips a trailing tone 1-5.
///
/// Returns `None` when the result is empty or holds non-`a-z` characters.
fn normalize_pinyin(text: &str) -> Option<String> {
    let mut normalized = text.to_lowercase().replace("u:", "v").replace('ü', "v");
    if let Some(last @ ('1'..='5')) = normalized.chars().last() {
        let _ = last;
        normalized.pop();
    }
    if normalized.is_empty() || !normalized.bytes().all(|b| b.is_ascii_lowercase()) {
        return None;
    }
    Some(normalized)
}

/// Validates raw file bytes: non-empty, UTF-8, no NUL, no control characters.
///
/// Mirrors the C++ `decodeUtf8` checks (the error strings name the file).
fn decode_text(bytes: &[u8], label: &str) -> Result<String> {
    if bytes.is_empty() {
        return Err(invalid(format!("Dictionary file is empty: {label}")));
    }
    let text = if let Some(stripped) = bytes.strip_prefix(b"\xef\xbb\xbf") {
        stripped
    } else {
        bytes
    };
    if text.contains(&0) {
        return Err(invalid(format!("{label}: Text contains an embedded NUL.")));
    }
    let text = std::str::from_utf8(text)
        .map_err(|_| invalid(format!("{label}: Text is not valid UTF-8.")))?;
    if text
        .bytes()
        .any(|b| (b < 0x20 && ![b'\t', b'\r', b'\n'].contains(&b)) || b == 0x7f)
    {
        return Err(invalid(format!(
            "{label}: Text contains an unsupported control character."
        )));
    }
    Ok(text.to_owned())
}

fn split_lines(text: &str) -> Vec<&str> {
    // juce::StringArray::fromLines splits on \r\n, \n and \r.
    let mut lines = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\n' || bytes[index] == b'\r' {
            lines.push(&text[start..index]);
            if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
                index += 1;
            }
            start = index + 1;
        }
        index += 1;
    }
    lines.push(&text[start..]);
    lines
}

fn split_fields(line: &str) -> Vec<&str> {
    line.split([' ', '\t']).filter(|field| !field.is_empty()).collect()
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| G2pError::Io {
        path: path.to_owned(),
        source,
    })
}

impl PhonemeDictionary {
    /// Loads a generic two-column dictionary: phones inventory plus word map.
    pub fn load(phones_path: &Path, dictionary_path: &Path) -> Result<Self> {
        let phones_bytes = read_file(phones_path)?;
        let dictionary_bytes = read_file(dictionary_path)?;
        Self::parse(
            &phones_bytes,
            &dictionary_bytes,
            &phones_path.display().to_string(),
            &dictionary_path.display().to_string(),
            phones_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            dictionary_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
        )
    }

    /// Loads Japanese tables: phones, romaji dictionary, hiragana, katakana
    /// and small-kana mappings.
    pub fn load_japanese(
        phones_path: &Path,
        dictionary_path: &Path,
        hiragana_path: &Path,
        katakana_path: &Path,
        small_kana_path: &Path,
    ) -> Result<Self> {
        let mut loaded = Self::load(phones_path, dictionary_path)?;
        for path in [hiragana_path, katakana_path] {
            let bytes = read_file(path)?;
            let label = path.display().to_string();
            load_kana_mappings(&bytes, &label, &mut loaded.kana_to_romaji)?;
        }
        let bytes = read_file(small_kana_path)?;
        let label = small_kana_path.display().to_string();
        load_kana_mappings(&bytes, &label, &mut loaded.small_kana_to_romaji)?;
        if loaded.kana_to_romaji.is_empty() || loaded.small_kana_to_romaji.is_empty() {
            return Err(invalid(
                "Japanese kana conversion dictionaries contain no entries.",
            ));
        }
        loaded.is_japanese = true;
        Ok(loaded)
    }

    /// Loads Mandarin tables: phones, pinyin dictionary and CEDICT hanzi file.
    pub fn load_mandarin(
        phones_path: &Path,
        dictionary_path: &Path,
        cedict_path: &Path,
    ) -> Result<Self> {
        let mut loaded = Self::load(phones_path, dictionary_path)?;
        let label = dictionary_path.display().to_string();
        let mut normalized: HashMap<String, Vec<String>> =
            HashMap::with_capacity(loaded.entries.len());
        for (key, pronunciation) in std::mem::take(&mut loaded.entries) {
            let normalized_key = normalize_pinyin(&key).unwrap_or(key);
            match normalized.get(&normalized_key) {
                Some(existing) if *existing != pronunciation => {
                    return Err(invalid(format!(
                        "{label}: Conflicting pronunciations for normalized pinyin '{normalized_key}'."
                    )));
                }
                Some(_) => {}
                None => {
                    normalized.insert(normalized_key, pronunciation);
                }
            }
        }
        loaded.entries = normalized;

        let bytes = read_file(cedict_path)?;
        let cedict_label = cedict_path.display().to_string();
        let cedict_label_path = Path::new(&cedict_label).to_path_buf();
        let text = decode_text(&bytes, &cedict_label)?;
        let mut supported = 0_usize;
        for (index, line) in split_lines(&text).iter().enumerate() {
            let content = line.trim();
            if content.is_empty() || content.starts_with('#') {
                continue;
            }
            let mut headwords = Vec::new();
            let mut reading = Vec::new();
            parse_cedict_entry(content, &mut headwords, &mut reading)
                .map_err(|message| line_error(&cedict_label_path, index + 1, &message))?;
            if reading.len() != 1 {
                continue;
            }
            let Some(pinyin) = normalize_pinyin(&reading[0]) else {
                continue;
            };
            let Some(pronunciation) = loaded.entries.get(&pinyin).cloned() else {
                continue;
            };
            for headword in &headwords {
                loaded.entries.entry(headword.clone()).or_insert_with(|| pronunciation.clone());
            }
            supported += 1;
        }
        if supported == 0 {
            return Err(invalid(format!(
                "CEDICT contains no supported single-syllable readings: {cedict_label}"
            )));
        }
        loaded.is_mandarin = true;
        Ok(loaded)
    }

    /// Parses `phones` (inventory) and `dictionary` (word map) file bytes.
    ///
    /// `phones_name` / `dictionary_name` are the file names, which select the
    /// English ARPABET and CMUDict handling; the `*_label` strings name the
    /// files in error messages.
    pub fn parse(
        phones_bytes: &[u8],
        dictionary_bytes: &[u8],
        phones_label: &str,
        dictionary_label: &str,
        phones_name: String,
        dictionary_name: String,
    ) -> Result<Self> {
        let phones_text = decode_text(phones_bytes, phones_label)?;
        let dictionary_text = decode_text(dictionary_bytes, dictionary_label)?;
        let phones_path = Path::new(phones_label).to_path_buf();
        let dictionary_path = Path::new(dictionary_label).to_path_buf();

        let mut loaded = PhonemeDictionary::default();
        loaded.is_english_arpabet = phones_name == "english-arpabet-phones.txt";
        let is_cmu = loaded.is_english_arpabet && dictionary_name == "cmudict-07b.txt";

        let mut symbols = HashSet::new();
        for (index, line) in split_lines(&phones_text).iter().enumerate() {
            let fields = split_fields(line);
            if fields.is_empty() {
                continue;
            }
            if fields.len() != 2 {
                return Err(line_error(
                    &phones_path,
                    index + 1,
                    "Expected exactly two fields: phoneme category.",
                ));
            }
            if !symbols.insert(fields[0].to_owned()) {
                return Err(line_error(
                    &phones_path,
                    index + 1,
                    &format!("Duplicate phoneme '{}'.", fields[0]),
                ));
            }
            loaded.phonemes.push(PhonemeDefinition {
                symbol: fields[0].to_owned(),
                category: fields[1].to_owned(),
            });
        }
        if loaded.phonemes.is_empty() {
            return Err(invalid(format!(
                "Phoneme inventory contains no definitions: {phones_label}"
            )));
        }
        loaded.symbols = symbols;

        for (index, line) in split_lines(&dictionary_text).iter().enumerate() {
            if is_cmu && line.trim_start().starts_with(";;;") {
                continue;
            }
            let fields = split_fields(line);
            if fields.is_empty() {
                continue;
            }
            if fields.len() < 2 {
                return Err(line_error(
                    &dictionary_path,
                    index + 1,
                    "Expected a dictionary key followed by one or more phonemes.",
                ));
            }
            let source_key = fields[0].to_owned();
            let key = if loaded.is_english_arpabet {
                normalize_english_key(source_key.clone())
            } else {
                source_key.clone()
            };
            if loaded.entries.contains_key(&key) {
                if is_cmu || source_key != key {
                    continue;
                }
                return Err(line_error(
                    &dictionary_path,
                    index + 1,
                    &format!("Duplicate dictionary key '{}'.", fields[0]),
                ));
            }
            let mut pronunciation = Vec::with_capacity(fields.len() - 1);
            let mut supported = true;
            for field in &fields[1..] {
                let mut symbol = (*field).to_owned();
                if is_cmu {
                    normalize_english_phone(&mut symbol);
                }
                if !loaded.symbols.contains(&symbol) {
                    if is_cmu {
                        supported = false;
                        break;
                    }
                    return Err(line_error(
                        &dictionary_path,
                        index + 1,
                        &format!("Undefined phoneme '{field}' in entry '{}'.", fields[0]),
                    ));
                }
                pronunciation.push(symbol);
            }
            if !supported {
                continue;
            }
            loaded.entries.insert(key, pronunciation);
        }
        if loaded.entries.is_empty() {
            return Err(invalid(format!(
                "Pronunciation dictionary contains no entries: {dictionary_label}"
            )));
        }
        Ok(loaded)
    }

    /// Translates one lyric syllable into phoneme symbols.
    pub fn lookup(&self, lyrics: &str) -> Result<Vec<String>> {
        if !self.is_loaded() {
            return Err(invalid("Phoneme dictionary is not loaded."));
        }
        if lyrics.contains('\0') {
            return Err(invalid("Text contains an embedded NUL."));
        }
        if lyrics
            .bytes()
            .any(|b| (b < 0x20 && ![b'\t', b'\r', b'\n'].contains(&b)) || b == 0x7f)
        {
            return Err(invalid("Text contains an unsupported control character."));
        }
        if lyrics.is_empty() {
            return Err(invalid("Dictionary lookup key is empty."));
        }
        if lyrics.contains([' ', '\t', '\r', '\n']) {
            return Err(invalid(
                "Dictionary lookup requires exactly one key without surrounding whitespace.",
            ));
        }

        if self.is_english_arpabet
            && lyrics.len() == 1
            && lyrics.as_bytes()[0].is_ascii_uppercase()
        {
            let spelling = ENGLISH_LETTER_NAMES[(lyrics.as_bytes()[0] - b'A') as usize];
            let mut pronunciation = Vec::new();
            for field in spelling.split(' ') {
                if !self.symbols.contains(field) {
                    return Err(invalid(format!(
                        "The phoneme inventory cannot spell the English letter '{lyrics}'."
                    )));
                }
                pronunciation.push(field.to_owned());
            }
            return Ok(pronunciation);
        }

        let mut key = lyrics.to_owned();
        if self.is_japanese {
            let mut romanized = String::new();
            let mut all_kana = true;
            for character in lyrics.chars() {
                let kana = character.to_string();
                if let Some(romaji) = self.kana_to_romaji.get(&kana) {
                    romanized.push_str(romaji);
                    continue;
                }
                let Some(small) = self.small_kana_to_romaji.get(&kana) else {
                    all_kana = false;
                    break;
                };
                // Small-kana digraph contraction: ki + ya -> kya,
                // shi + ya -> sha, chi + ya -> cha.
                if small.len() == 2
                    && small.starts_with('y')
                    && !romanized.is_empty()
                    && romanized.ends_with('i')
                {
                    romanized.pop();
                    let digraph = romanized.ends_with("sh")
                        || romanized.ends_with("ch")
                        || romanized.ends_with('j');
                    if !digraph {
                        romanized.push('y');
                    }
                    romanized.push(small.chars().last().expect("two chars"));
                } else {
                    romanized.push_str(small);
                }
            }
            if all_kana {
                key = romanized;
            }
        }

        if self.is_mandarin && let Some(normalized) = normalize_pinyin(lyrics) {
            key = normalized;
        }

        if let Some(pronunciation) = self.entries.get(&key) {
            return Ok(pronunciation.clone());
        }
        if self.is_mandarin {
            return Err(invalid(format!(
                "No Mandarin pronunciation for '{lyrics}'. Use one dictionary-listed syllable per note: a Chinese character or pinyin (optional tone 1-5). Split multi-character lyrics across notes, or enter phonemes explicitly."
            )));
        }

        let candidate = key.to_lowercase();
        if let Some(pronunciation) = self.entries.get(&candidate) {
            return Ok(pronunciation.clone());
        }
        let stripped: String = candidate
            .bytes()
            .filter(|b| b.is_ascii_alphanumeric())
            .map(char::from)
            .collect();
        if stripped.is_empty() {
            return Err(invalid(format!(
                "No pronunciation for dictionary key '{lyrics}'."
            )));
        }

        if let Some((_, phones)) = ENGLISH_FALLBACKS.iter().find(|(word, _)| *word == stripped) {
            return Ok(phones.iter().map(|phone| (*phone).to_owned()).collect());
        }

        let mut guessed = Vec::new();
        let mut index = 0;
        while index < stripped.len() {
            let next = &stripped[index..];
            if let Some((start, phone)) = ENGLISH_PATTERNS
                .iter()
                .find(|(start, _)| next.starts_with(start))
            {
                guessed.push((*phone).to_owned());
                index += start.len();
                continue;
            }
            let letter = stripped.as_bytes()[index] as char;
            guessed.push(
                match letter {
                    'a' => "ae",
                    'e' => "eh",
                    'i' => "ih",
                    'o' => "aa",
                    'u' => "uh",
                    'y' => "y",
                    'h' => "hh",
                    't' => "t",
                    'd' => "d",
                    'k' => "k",
                    'p' => "p",
                    'b' => "b",
                    'm' => "m",
                    'n' => "n",
                    'l' => "l",
                    'r' => "r",
                    's' => "s",
                    'f' => "f",
                    'v' => "v",
                    'z' => "z",
                    'j' => "jh",
                    'g' => "g",
                    'w' => "w",
                    _ => "ah",
                }
                .to_owned(),
            );
            index += 1;
        }
        if guessed.is_empty() || guessed.iter().any(|symbol| !self.symbols.contains(symbol)) {
            return Err(invalid(format!(
                "No pronunciation for dictionary key '{lyrics}'."
            )));
        }
        Ok(guessed)
    }

    /// Parses and validates an explicit space-separated phoneme override.
    pub fn parse_explicit_phonemes(&self, text: &str) -> Result<Vec<String>> {
        if !self.is_loaded() {
            return Err(invalid("Phoneme dictionary is not loaded."));
        }
        let fields: Vec<&str> = text.split([' ', '\t', '\r', '\n']).filter(|field| !field.is_empty()).collect();
        if fields.is_empty() {
            return Err(invalid("Explicit phoneme sequence is empty."));
        }
        let mut parsed = Vec::with_capacity(fields.len());
        for field in fields {
            if !self.symbols.contains(field) {
                return Err(invalid(format!("Unknown explicit phoneme '{field}'.")));
            }
            parsed.push(field.to_owned());
        }
        Ok(parsed)
    }

    /// Returns true when the dictionary has an entry for `key`.
    pub fn has_entry(&self, key: &str) -> bool {
        if key.is_empty() {
            return false;
        }
        let candidate = if self.is_english_arpabet {
            normalize_english_key(key.to_owned())
        } else {
            key.to_lowercase()
        };
        if self.entries.contains_key(&candidate) {
            return true;
        }
        let stripped: String = candidate
            .bytes()
            .filter(|b| b.is_ascii_alphanumeric())
            .map(char::from)
            .collect();
        !stripped.is_empty() && self.entries.contains_key(&stripped)
    }

    /// Returns true when phones and entries both loaded.
    pub fn is_loaded(&self) -> bool {
        !self.phonemes.is_empty() && !self.entries.is_empty()
    }

    /// Returns all phoneme definitions with their categories.
    pub fn phonemes(&self) -> &[PhonemeDefinition] {
        &self.phonemes
    }

    /// Returns the number of vocabulary words in the dictionary.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

fn load_kana_mappings(
    bytes: &[u8],
    label: &str,
    mappings: &mut HashMap<String, String>,
) -> Result<()> {
    let text = decode_text(bytes, label)?;
    let path = Path::new(label).to_path_buf();
    for (index, line) in split_lines(&text).iter().enumerate() {
        let fields = split_fields(line);
        if fields.is_empty() {
            continue;
        }
        if fields.len() != 2 {
            return Err(line_error(
                &path,
                index + 1,
                "Expected exactly two fields: kana and romaji.",
            ));
        }
        match mappings.get(fields[0]) {
            Some(existing) if *existing != fields[1] => {
                return Err(line_error(
                    &path,
                    index + 1,
                    &format!("Conflicting romaji for kana '{}'.", fields[0]),
                ));
            }
            _ => {
                mappings.insert(fields[0].to_owned(), fields[1].to_owned());
            }
        }
    }
    Ok(())
}

/// Parses one CC-CEDICT line: `trad simp [pin1 yin1] /defs/`.
fn parse_cedict_entry(line: &str, headwords: &mut Vec<String>, reading: &mut Vec<String>) -> std::result::Result<(), String> {
    let opening = line.find('[').ok_or_else(|| {
        "Expected a CEDICT entry: traditional simplified [pinyin].".to_owned()
    })?;
    let closing = line.find(']').ok_or_else(|| {
        "Expected a CEDICT entry: traditional simplified [pinyin].".to_owned()
    })?;
    if opening == 0
        || closing <= opening
        || !line[..opening].ends_with(|c: char| c.is_whitespace())
    {
        return Err("Expected a CEDICT entry: traditional simplified [pinyin].".to_owned());
    }
    let heads = split_fields(&line[..opening]);
    if heads.len() != 2 || heads.iter().any(|head| head.contains(['[', ']'])) {
        return Err(
            "Expected exactly two CEDICT headwords before the pinyin reading.".to_owned(),
        );
    }
    let pinyin = &line[opening + 1..closing];
    if pinyin.contains(['[', ']']) {
        return Err("CEDICT pinyin reading contains an unexpected bracket.".to_owned());
    }
    let syllables = split_fields(pinyin);
    if syllables.is_empty() {
        return Err("CEDICT pinyin reading is empty.".to_owned());
    }
    let suffix = &line[closing + 1..];
    let definitions = suffix.trim();
    if !definitions.is_empty()
        && (!suffix.starts_with(|c: char| c.is_whitespace())
            || !definitions.starts_with('/')
            || !definitions.ends_with('/'))
    {
        return Err(
            "Unexpected text after the CEDICT pinyin reading; definitions must be slash-delimited."
                .to_owned(),
        );
    }
    *headwords = heads.iter().map(|head| (*head).to_owned()).collect();
    *reading = syllables.iter().map(|syllable| (*syllable).to_owned()).collect();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::temp_dir_for_tests;

    const MINI_ARPABET_PHONES: &str = "\
aa vowel\nae vowel\nah vowel\nao vowel\naw diphthong\nax vowel\nay diphthong\n\
eh vowel\ner vowel\ney diphthong\nih vowel\niy vowel\now diphthong\noy diphthong\n\
uh vowel\nuw vowel\nb consonant\nch consonant\nd consonant\ndh consonant\nf consonant\n\
g consonant\nhh consonant\njh consonant\nk consonant\nl consonant\nm consonant\n\
n consonant\nng consonant\np consonant\nr consonant\ns consonant\nsh consonant\n\
t consonant\nth consonant\nv consonant\nw consonant\ny consonant\nz consonant\n\
sil silence\nbr breath\n";

    const MINI_ROMAJI_PHONES: &str = "\
a vowel\ni vowel\nu vowel\ne vowel\no vowel\nk consonant\ns consonant\n\
sh consonant\nch consonant\nt consonant\ny consonant\nn consonant\n\
sil silence\nbr breath\n";

    const MINI_ROMAJI_DICT: &str = "\
ka k a\nki k i\nkya k y a\nshi sh i\nsha sh a\nchi ch i\nchu ch u\nebr-check sil\n";

    fn arpabet(
        phones: &str,
        dict: &str,
        phones_name: &str,
        dict_name: &str,
    ) -> Result<PhonemeDictionary> {
        PhonemeDictionary::parse(
            phones.as_bytes(),
            dict.as_bytes(),
            "phones.txt",
            "dict.txt",
            phones_name.to_owned(),
            dict_name.to_owned(),
        )
    }

    fn english(dict: &str) -> PhonemeDictionary {
        arpabet(
            MINI_ARPABET_PHONES,
            dict,
            "english-arpabet-phones.txt",
            "english-arpabet-dict.txt",
        )
        .expect("synthetic english dictionary parses")
    }

    fn cmu(dict: &str) -> PhonemeDictionary {
        arpabet(
            MINI_ARPABET_PHONES,
            dict,
            "english-arpabet-phones.txt",
            "cmudict-07b.txt",
        )
        .expect("synthetic cmu dictionary parses")
    }

    fn romaji() -> PhonemeDictionary {
        PhonemeDictionary::parse(
            MINI_ROMAJI_PHONES.as_bytes(),
            MINI_ROMAJI_DICT.as_bytes(),
            "phones.txt",
            "dict.txt",
            "japanese-romaji-phones.txt".to_owned(),
            "japanese-romaji-dict.txt".to_owned(),
        )
        .expect("synthetic romaji dictionary parses")
    }

    #[test]
    fn basic_lookup_returns_entry() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(dictionary.lookup("hello").unwrap(), ["hh", "ah", "l", "ow"]);
        assert_eq!(dictionary.entry_count(), 1);
        assert!(dictionary.is_loaded());
        assert_eq!(dictionary.phonemes().len(), 41);
    }

    #[test]
    fn empty_phones_inventory_is_rejected() {
        // An empty file is rejected up front, like C++ readText does.
        let error = arpabet("", "hello hh ah l ow\n", "p", "d").unwrap_err();
        assert_eq!(error.to_string(), "Dictionary file is empty: phones.txt");
        // A whitespace-only file parses to zero definitions instead.
        let error = arpabet("  \n", "hello hh ah l ow\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Phoneme inventory contains no definitions: phones.txt"
        );
    }

    #[test]
    fn empty_dictionary_is_rejected() {
        let error = arpabet(MINI_ARPABET_PHONES, "\n  \n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Pronunciation dictionary contains no entries: dict.txt"
        );
    }

    #[test]
    fn malformed_phones_line_reports_line_number() {
        let error = arpabet("aa vowel extra\n", "hello hh ah l ow\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "phones.txt:1: Expected exactly two fields: phoneme category."
        );
    }

    #[test]
    fn duplicate_phoneme_is_rejected() {
        let error = arpabet("aa vowel\naa vowel\n", "x aa\n", "p", "d").unwrap_err();
        assert_eq!(error.to_string(), "phones.txt:2: Duplicate phoneme 'aa'.");
    }

    #[test]
    fn short_dictionary_line_reports_line_number() {
        let error = arpabet(MINI_ARPABET_PHONES, "hello\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "dict.txt:1: Expected a dictionary key followed by one or more phonemes."
        );
    }

    #[test]
    fn duplicate_key_is_rejected_for_generic_dictionaries() {
        let error = arpabet(MINI_ARPABET_PHONES, "hello hh\nhello ah\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "dict.txt:2: Duplicate dictionary key 'hello'."
        );
    }

    #[test]
    fn undefined_phoneme_names_entry_and_key() {
        let error = arpabet(MINI_ARPABET_PHONES, "hello hh xx\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "dict.txt:1: Undefined phoneme 'xx' in entry 'hello'."
        );
    }

    #[test]
    fn utf8_bom_is_stripped() {
        let mut phones = b"\xef\xbb\xbf".to_vec();
        phones.extend_from_slice(MINI_ARPABET_PHONES.as_bytes());
        let dictionary = PhonemeDictionary::parse(
            &phones,
            b"hello hh ah l ow\n",
            "phones.txt",
            "dict.txt",
            "english-arpabet-phones.txt".to_owned(),
            "english-arpabet-dict.txt".to_owned(),
        )
        .unwrap();
        assert_eq!(dictionary.lookup("hello").unwrap(), ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn empty_file_is_rejected() {
        let error = arpabet("", "", "p", "d").unwrap_err();
        assert_eq!(error.to_string(), "Dictionary file is empty: phones.txt");
    }

    #[test]
    fn embedded_nul_is_rejected() {
        let error = arpabet("aa vo\0wel\n", "x aa\n", "p", "d").unwrap_err();
        assert_eq!(error.to_string(), "phones.txt: Text contains an embedded NUL.");
    }

    #[test]
    fn control_characters_are_rejected() {
        let error = arpabet("aa vo\x07wel\n", "x aa\n", "p", "d").unwrap_err();
        assert_eq!(
            error.to_string(),
            "phones.txt: Text contains an unsupported control character."
        );
    }

    #[test]
    fn cmu_comments_variants_stress_and_aliases() {
        // Keys normalize to lowercase; phones are inventory symbols, so the
        // synthetic file uses the lowercase forms directly.
        let dictionary = cmu(";;; comment\nHELLO hh ah l ow\nHELLO(2) hh eh l ow\nBAT b ae1 t\nOLD ax\n");
        // First pronunciation variant wins; key lookup is case-insensitive.
        assert_eq!(dictionary.lookup("hello").unwrap(), ["hh", "ah", "l", "ow"]);
        // Stress digit stripped: AE1 -> AE.
        assert_eq!(dictionary.lookup("bat").unwrap(), ["b", "ae", "t"]);
        // Dialect alias: AX -> AH.
        assert_eq!(dictionary.lookup("old").unwrap(), ["ah"]);
    }

    #[test]
    fn cmu_variant_duplicate_is_kept_silently() {
        let dictionary = cmu("HELLO hh ah l ow\nhello(2) hh eh l ow\n");
        assert_eq!(dictionary.lookup("hello").unwrap(), ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn cmu_entry_with_unknown_phone_is_skipped() {
        let dictionary = cmu("WEIRD hh xx\nhello hh ah l ow\n");
        assert!(!dictionary.has_entry("weird"));
        // Falls through to the letter heuristics: w eh ih r d.
        assert_eq!(dictionary.lookup("weird").unwrap(), ["w", "eh", "ih", "r", "d"]);
        assert_eq!(dictionary.lookup("hello").unwrap(), ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn lookup_validates_its_key() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(
            dictionary.lookup("").unwrap_err().to_string(),
            "Dictionary lookup key is empty."
        );
        assert_eq!(
            dictionary.lookup("a b").unwrap_err().to_string(),
            "Dictionary lookup requires exactly one key without surrounding whitespace."
        );
        assert_eq!(
            dictionary.lookup("a\0b").unwrap_err().to_string(),
            "Text contains an embedded NUL."
        );
        assert_eq!(
            PhonemeDictionary::default().lookup("hello").unwrap_err().to_string(),
            "Phoneme dictionary is not loaded."
        );
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(dictionary.lookup("HELLO").unwrap(), ["hh", "ah", "l", "ow"]);
    }

    #[test]
    fn punctuation_is_stripped_before_lookup() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(
            dictionary.lookup("hello!").unwrap(),
            ["hh", "ah", "l", "ow"]
        );
        assert_eq!(
            dictionary.lookup("!!!").unwrap_err().to_string(),
            "No pronunciation for dictionary key '!!!'."
        );
    }

    #[test]
    fn common_english_words_fall_back() {
        let dictionary = english("hello hh ah l ow\n");
        for (word, expected) in [
            ("the", vec!["dh", "ah"]),
            ("you", vec!["y", "uw"]),
            ("a", vec!["ae"]),
            ("i", vec!["ay"]),
        ] {
            assert_eq!(dictionary.lookup(word).unwrap(), expected, "{word}");
        }
    }

    #[test]
    fn unknown_english_words_use_letter_heuristics() {
        let dictionary = english("hello hh ah l ow\n");
        // bat: no patterns, per-letter mapping.
        assert_eq!(dictionary.lookup("bat").unwrap(), ["b", "ae", "t"]);
        // Patterns win over letters: sh, ee.
        assert_eq!(dictionary.lookup("sheep").unwrap(), ["sh", "iy", "p"]);
    }

    #[test]
    fn heuristic_fails_when_inventory_lacks_the_guess() {
        let phones = MINI_ARPABET_PHONES.replace("ae vowel\n", "");
        let dictionary = arpabet(
            &phones,
            "hello hh ah l ow\n",
            "english-arpabet-phones.txt",
            "english-arpabet-dict.txt",
        )
        .unwrap();
        assert_eq!(
            dictionary.lookup("bat").unwrap_err().to_string(),
            "No pronunciation for dictionary key 'bat'."
        );
    }

    #[test]
    fn single_capital_letters_are_spelled_out() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(dictionary.lookup("A").unwrap(), ["ey"]);
        assert_eq!(dictionary.lookup("B").unwrap(), ["b", "iy"]);
    }

    #[test]
    fn unspellable_letters_report_the_inventory() {
        let phones = "ey vowel\niy vowel\n";
        let dictionary = arpabet(
            phones,
            "x ey\n",
            "english-arpabet-phones.txt",
            "english-arpabet-dict.txt",
        )
        .unwrap();
        assert_eq!(
            dictionary.lookup("Z").unwrap_err().to_string(),
            "The phoneme inventory cannot spell the English letter 'Z'."
        );
    }

    #[test]
    fn explicit_phonemes_are_validated_against_the_inventory() {
        let dictionary = english("hello hh ah l ow\n");
        assert_eq!(
            dictionary.parse_explicit_phonemes("hh ah").unwrap(),
            ["hh", "ah"]
        );
        assert_eq!(
            dictionary.parse_explicit_phonemes("hh  xx").unwrap_err().to_string(),
            "Unknown explicit phoneme 'xx'."
        );
        assert_eq!(
            dictionary
                .parse_explicit_phonemes("  \t ")
                .unwrap_err()
                .to_string(),
            "Explicit phoneme sequence is empty."
        );
        assert_eq!(
            PhonemeDictionary::default()
                .parse_explicit_phonemes("hh")
                .unwrap_err()
                .to_string(),
            "Phoneme dictionary is not loaded."
        );
    }

    #[test]
    fn has_entry_normalizes_and_strips() {
        let dictionary = cmu("HELLO hh ah l ow\n");
        assert!(dictionary.has_entry("hello"));
        assert!(dictionary.has_entry("HELLO"));
        assert!(dictionary.has_entry("hello!"));
        assert!(!dictionary.has_entry("goodbye"));
        assert!(!dictionary.has_entry(""));
        assert!(!dictionary.has_entry("!!!"));
    }

    #[test]
    fn japanese_kana_romanizes_with_contraction() {
        let dir = temp_dir_for_tests("dict-jp");
        write_test_file(&dir.join("japanese-romaji-phones.txt"), MINI_ROMAJI_PHONES);
        write_test_file(&dir.join("japanese-romaji-dict.txt"), MINI_ROMAJI_DICT);
        write_test_file(&dir.join("hira.txt"), "か ka\nき ki\nし shi\nち chi\n");
        write_test_file(&dir.join("kata.txt"), "カ ka\nキ ki\n");
        write_test_file(&dir.join("sute.txt"), "ゃ ya\nゅ yu\nょ yo\n");
        let dictionary = PhonemeDictionary::load_japanese(
            &dir.join("japanese-romaji-phones.txt"),
            &dir.join("japanese-romaji-dict.txt"),
            &dir.join("hira.txt"),
            &dir.join("kata.txt"),
            &dir.join("sute.txt"),
        )
        .unwrap();
        // Direct romaji still works.
        assert_eq!(dictionary.lookup("ka").unwrap(), ["k", "a"]);
        // Hiragana with small-kana contraction: き + ゃ -> kya.
        assert_eq!(dictionary.lookup("きゃ").unwrap(), ["k", "y", "a"]);
        // Digraph onset: し + ゃ -> sha (no extra y).
        assert_eq!(dictionary.lookup("しゃ").unwrap(), ["sh", "a"]);
        assert_eq!(dictionary.lookup("ちゅ").unwrap(), ["ch", "u"]);
        // Katakana shares the table.
        assert_eq!(dictionary.lookup("カ").unwrap(), ["k", "a"]);
    }

    #[test]
    fn japanese_conflicting_kana_is_rejected() {
        let dir = temp_dir_for_tests("dict-jp-conflict");
        write_test_file(&dir.join("phones.txt"), MINI_ROMAJI_PHONES);
        write_test_file(&dir.join("dict.txt"), MINI_ROMAJI_DICT);
        write_test_file(&dir.join("hira.txt"), "か ka\n");
        write_test_file(&dir.join("kata.txt"), "か ka\n");
        write_test_file(&dir.join("sute.txt"), "ゃ ya\n");
        // Same kana with a different romaji conflicts.
        write_test_file(&dir.join("kata2.txt"), "か ki\n");
        let error = PhonemeDictionary::load_japanese(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("hira.txt"),
            &dir.join("kata2.txt"),
            &dir.join("sute.txt"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("Conflicting romaji for kana 'か'."), "{error}");
    }

    #[test]
    fn japanese_malformed_kana_line_is_rejected() {
        let dir = temp_dir_for_tests("dict-jp-badline");
        write_test_file(&dir.join("phones.txt"), MINI_ROMAJI_PHONES);
        write_test_file(&dir.join("dict.txt"), MINI_ROMAJI_DICT);
        write_test_file(&dir.join("hira.txt"), "か ka extra\n");
        write_test_file(&dir.join("kata.txt"), "カ ka\n");
        write_test_file(&dir.join("sute.txt"), "ゃ ya\n");
        let error = PhonemeDictionary::load_japanese(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("hira.txt"),
            &dir.join("kata.txt"),
            &dir.join("sute.txt"),
        )
        .unwrap_err();
        assert!(
            error.to_string().ends_with(":1: Expected exactly two fields: kana and romaji."),
            "{error}"
        );
    }

    #[test]
    fn mandarin_normalizes_tones_and_hanzi() {
        let dir = temp_dir_for_tests("dict-zh");
        write_test_file(&dir.join("phones.txt"), "a vowel\ni vowel\nn consonant\nl consonant\nv vowel\nsil silence\n");
        write_test_file(&dir.join("dict.txt"), "ni n i\nlü l v\n");
        write_test_file(
            &dir.join("cedict.txt"),
            "# comment\n你 你 [ni3] /you/\n中国 中国 [zhong1 guo2] /China/\n",
        );
        let dictionary = PhonemeDictionary::load_mandarin(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("cedict.txt"),
        )
        .unwrap();
        // Tone number stripped.
        assert_eq!(dictionary.lookup("ni3").unwrap(), ["n", "i"]);
        // ü normalized to v (both spellings).
        assert_eq!(dictionary.lookup("lü").unwrap(), ["l", "v"]);
        assert_eq!(dictionary.lookup("lu:").unwrap(), ["l", "v"]);
        // Hanzi mapped through CEDICT to the pinyin pronunciation.
        assert_eq!(dictionary.lookup("你").unwrap(), ["n", "i"]);
        // Multi-syllable readings are skipped, not mapped.
        assert!(dictionary.lookup("中国").unwrap_err().to_string().starts_with("No Mandarin pronunciation"));
    }

    #[test]
    fn mandarin_miss_names_the_syllable_rule() {
        let dir = temp_dir_for_tests("dict-zh-miss");
        write_test_file(&dir.join("phones.txt"), "a vowel\nn consonant\n");
        write_test_file(&dir.join("dict.txt"), "na n a\n");
        write_test_file(&dir.join("cedict.txt"), "那 那 [na4] /that/\n");
        let dictionary = PhonemeDictionary::load_mandarin(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("cedict.txt"),
        )
        .unwrap();
        let error = dictionary.lookup("ma").unwrap_err().to_string();
        assert!(error.starts_with("No Mandarin pronunciation for 'ma'. Use one dictionary-listed syllable per note"), "{error}");
    }

    #[test]
    fn mandarin_conflicting_normalized_pinyin_is_rejected() {
        let dir = temp_dir_for_tests("dict-zh-conflict");
        write_test_file(&dir.join("phones.txt"), "a vowel\ni vowel\nn consonant\n");
        write_test_file(&dir.join("dict.txt"), "ni n i\nNI1 n a\n");
        write_test_file(&dir.join("cedict.txt"), "你 你 [ni3] /you/\n");
        let error = PhonemeDictionary::load_mandarin(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("cedict.txt"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("Conflicting pronunciations for normalized pinyin 'ni'."), "{error}");
    }

    #[test]
    fn mandarin_without_supported_readings_is_rejected() {
        let dir = temp_dir_for_tests("dict-zh-empty");
        write_test_file(&dir.join("phones.txt"), "a vowel\nn consonant\n");
        write_test_file(&dir.join("dict.txt"), "na n a\n");
        write_test_file(&dir.join("cedict.txt"), "中国 中国 [zhong1 guo2] /China/\n");
        let error = PhonemeDictionary::load_mandarin(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("cedict.txt"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("CEDICT contains no supported single-syllable readings"), "{error}");
    }

    #[test]
    fn cedict_malformed_lines_report_numbers() {
        let dir = temp_dir_for_tests("dict-cedict-bad");
        write_test_file(&dir.join("phones.txt"), "a vowel\nn consonant\n");
        write_test_file(&dir.join("dict.txt"), "na n a\n");
        write_test_file(&dir.join("cedict.txt"), "this line has no brackets\n");
        let error = PhonemeDictionary::load_mandarin(
            &dir.join("phones.txt"),
            &dir.join("dict.txt"),
            &dir.join("cedict.txt"),
        )
        .unwrap_err();
        assert!(error.to_string().ends_with(":1: Expected a CEDICT entry: traditional simplified [pinyin]."), "{error}");
    }

    #[test]
    fn missing_files_are_io_errors() {
        let error = PhonemeDictionary::load(
            Path::new("/definitely/not/here-phones.txt"),
            Path::new("/definitely/not/here-dict.txt"),
        )
        .unwrap_err();
        assert!(error.to_string().starts_with("could not read /definitely/not/here-phones.txt"), "{error}");
    }

    #[test]
    fn romaji_dictionary_spot_checks() {
        let dictionary = romaji();
        assert_eq!(dictionary.lookup("shi").unwrap(), ["sh", "i"]);
        assert!(dictionary.has_entry("ka"));
        assert!(!dictionary.has_entry("zzz"));
        // Unknown romaji fails: the heuristic ARPABET guesses are not in the
        // small romaji inventory, so there is no pronunciation.
        assert_eq!(
            dictionary.lookup("zzz").unwrap_err().to_string(),
            "No pronunciation for dictionary key 'zzz'."
        );
    }

    #[test]
    fn real_dictionaries_parse_when_provided() {
        let Some(dir) = std::env::var_os("OPENSVR_DICT").map(PathBuf::from) else {
            return;
        };
        for (language, stem) in [
            ("english", "english-arpabet"),
            ("japanese", "japanese-romaji"),
            ("mandarin", "mandarin-xsampa"),
            ("cantonese", "cantonese-xsampa"),
            ("spanish", "spanish-xsampa"),
        ] {
            let phones = dir.join(format!("{stem}-phones.txt"));
            let dict = dir.join(format!("{stem}-dict.txt"));
            if !phones.is_file() || !dict.is_file() {
                continue;
            }
            let dictionary = PhonemeDictionary::load(&phones, &dict).expect(language);
            assert!(!dictionary.phonemes().is_empty(), "{language}");
            assert!(dictionary.entry_count() > 0, "{language}");
        }
    }

    fn write_test_file(path: &Path, contents: &str) {
        std::fs::write(path, contents).expect("test fixture writes");
    }
}
