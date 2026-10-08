//! Phonetic inventory (`PhoneSet`) read from a DNNI `_psv2` node.
//!
//! Ports OpenSV's `src/synthesis/PhoneSet.{h,cpp}`. The binary payload layout
//! is: a length-prefixed name, then four length-prefixed string lists
//! (`symbols`, `categories`, `unifiedSymbols`, `classes`). All integers are
//! 32-bit little-endian; every string length is bounded (4096 bytes) and must
//! be valid UTF-8 without embedded NULs.

use opensvr_dnni::DnniReader;

use crate::{Result, invalid};

/// Phonetic table and inventory of a voice model or language.
///
/// `symbols` holds the phoneme tokens (e.g. `"a"`, `"k"`, `"sil"`) in index
/// order; `categories` is parallel to it (e.g. `"vowel"`, `"stop"`).
/// `unified_symbols` is the cross-lingual mapping per symbol and `classes`
/// the distinct category names.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhoneSet {
    pub name: String,
    pub symbols: Vec<String>,
    pub categories: Vec<String>,
    pub unified_symbols: Vec<String>,
    pub classes: Vec<String>,
}

impl PhoneSet {
    /// Returns the category of `symbol`, or `None` when it is not in the set.
    pub fn category_of(&self, symbol: &str) -> Option<&str> {
        self.symbols
            .iter()
            .position(|candidate| candidate == symbol)
            .map(|index| self.categories[index].as_str())
    }

    /// Returns true for the continuation carriers: vowels and diphthongs.
    pub fn is_continuation_carrier(&self, symbol: &str) -> bool {
        matches!(self.category_of(symbol), Some("vowel" | "diphthong"))
    }
}

const MAX_STRING_BYTES: usize = 4096;
const MAX_STRING_COUNT: usize = 4096;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_string(bytes: &[u8], position: &mut usize) -> Option<String> {
    if bytes.len() - *position < 4 {
        return None;
    }
    let length = read_u32(bytes, *position) as usize;
    *position += 4;
    if length > bytes.len() - *position || length > MAX_STRING_BYTES {
        return None;
    }
    let text = bytes[*position..*position + length].to_vec();
    if text.contains(&0) || std::str::from_utf8(&text).is_err() {
        return None;
    }
    *position += length;
    // SAFETY: just validated as UTF-8 without NULs.
    Some(String::from_utf8(text).expect("validated UTF-8"))
}

fn read_string_list(bytes: &[u8], position: &mut usize) -> Option<Vec<String>> {
    if bytes.len() - *position < 4 {
        return None;
    }
    let count = read_u32(bytes, *position) as usize;
    *position += 4;
    if count > MAX_STRING_COUNT || count > (bytes.len() - *position) / 4 {
        return None;
    }
    let mut strings = Vec::with_capacity(count);
    for _ in 0..count {
        strings.push(read_string(bytes, position)?);
    }
    Some(strings)
}

/// Reads and validates the `_psv2` node at `node_index`.
///
/// The node must be a leaf `_psv2` node; on any validation failure nothing
/// is returned and the error text matches the C++ `readPhoneSet` message.
pub fn read_phone_set(reader: &DnniReader, node_index: usize) -> Result<PhoneSet> {
    let node = reader.nodes().get(node_index).ok_or_else(|| {
        invalid("Phoneme tables require a leaf _psv2 node.")
    })?;
    if node.name != "_psv2" || !node.children.is_empty() {
        return Err(invalid("Phoneme tables require a leaf _psv2 node."));
    }
    parse_phone_set(reader.payload(node_index))
}

fn parse_phone_set(bytes: &[u8]) -> Result<PhoneSet> {
    let mut position = 0_usize;
    let mut set = PhoneSet::default();
    let parsed = read_string(bytes, &mut position)
        .map(|name| set.name = name)
        .is_some()
        && read_string_list(bytes, &mut position)
            .map(|symbols| set.symbols = symbols)
            .is_some()
        && read_string_list(bytes, &mut position)
            .map(|categories| set.categories = categories)
            .is_some()
        && read_string_list(bytes, &mut position)
            .map(|unified| set.unified_symbols = unified)
            .is_some()
        && read_string_list(bytes, &mut position)
            .map(|classes| set.classes = classes)
            .is_some()
        && position == bytes.len();
    if !parsed {
        return Err(invalid("Invalid _psv2 string table."));
    }
    if set.name.is_empty()
        || set.symbols.is_empty()
        || set.categories.len() != set.symbols.len()
        || set.unified_symbols.len() < set.symbols.len()
    {
        return Err(invalid(
            "Phoneme _psv2 table has inconsistent symbol metadata.",
        ));
    }
    let mut seen_classes = std::collections::HashSet::new();
    for class in &set.classes {
        if class.is_empty() || !seen_classes.insert(class.clone()) {
            return Err(invalid(
                "Phoneme _psv2 class names must be nonempty and unique.",
            ));
        }
    }
    let mut seen_symbols = std::collections::HashSet::new();
    for (index, symbol) in set.symbols.iter().enumerate() {
        if symbol.is_empty()
            || !seen_symbols.insert(symbol.clone())
            || !seen_classes.contains(&set.categories[index])
            || set.unified_symbols[index].is_empty()
        {
            return Err(invalid(
                "Phoneme _psv2 contains an invalid symbol, category, or mapping.",
            ));
        }
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    fn encode_list(out: &mut Vec<u8>, items: &[&str]) {
        out.extend_from_slice(&(items.len() as u32).to_le_bytes());
        for item in items {
            encode_string(out, item);
        }
    }

    fn payload() -> Vec<u8> {
        let mut out = Vec::new();
        encode_string(&mut out, "japanese-romaji");
        encode_list(&mut out, &["a", "k", "sil"]);
        encode_list(&mut out, &["vowel", "stop", "silence"]);
        encode_list(&mut out, &["a", "k", "sil"]);
        encode_list(&mut out, &["vowel", "stop", "silence"]);
        out
    }

    fn v1_file(node_name: &str, payload: &[u8], children: u32) -> Vec<u8> {
        let mut tag = [0_u8; 8];
        tag[..node_name.len()].copy_from_slice(node_name.as_bytes());
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        out.extend_from_slice(&tag);
        out.extend_from_slice(&children.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn load(payload: &[u8]) -> DnniReader {
        DnniReader::from_bytes(v1_file("_psv2", payload, 0)).expect("synthetic blob parses")
    }

    #[test]
    fn valid_table_reads() {
        let set = read_phone_set(&load(&payload()), 0).unwrap();
        assert_eq!(set.name, "japanese-romaji");
        assert_eq!(set.symbols, ["a", "k", "sil"]);
        assert_eq!(set.category_of("a"), Some("vowel"));
        assert_eq!(set.category_of("k"), Some("stop"));
        assert_eq!(set.category_of("nope"), None);
        assert!(set.is_continuation_carrier("a"));
        assert!(!set.is_continuation_carrier("k"));
    }

    #[test]
    fn wrong_type_is_rejected() {
        let reader =
            DnniReader::from_bytes(v1_file("prim0", &payload(), 0)).expect("blob parses");
        assert_eq!(
            read_phone_set(&reader, 0).unwrap_err().to_string(),
            "Phoneme tables require a leaf _psv2 node."
        );
    }

    #[test]
    fn node_with_children_is_rejected() {
        let mut bytes = v1_file("_psv2", &payload(), 1);
        // A child must follow the parent header for the blob to parse.
        let mut tag = [0_u8; 8];
        tag[.."prim0".len()].copy_from_slice(b"prim0");
        bytes.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        bytes.extend_from_slice(&tag);
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        let reader = DnniReader::from_bytes(bytes).expect("blob parses");
        assert!(
            read_phone_set(&reader, 0)
                .unwrap_err()
                .to_string()
                .contains("leaf _psv2")
        );
        // The leaf child itself is not a phone set either.
        assert!(
            read_phone_set(&reader, 1)
                .unwrap_err()
                .to_string()
                .contains("leaf _psv2")
        );
    }

    #[test]
    fn missing_node_is_rejected() {
        let reader = load(&payload());
        assert!(
            read_phone_set(&reader, 7)
                .unwrap_err()
                .to_string()
                .contains("leaf _psv2")
        );
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let mut bytes = payload();
        bytes.truncate(bytes.len() - 3);
        assert_eq!(
            read_phone_set(&load(&bytes), 0).unwrap_err().to_string(),
            "Invalid _psv2 string table."
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = payload();
        bytes.push(0);
        assert_eq!(
            read_phone_set(&load(&bytes), 0).unwrap_err().to_string(),
            "Invalid _psv2 string table."
        );
    }

    #[test]
    fn mismatched_symbol_category_lengths_are_rejected() {
        let mut out = Vec::new();
        encode_string(&mut out, "x");
        encode_list(&mut out, &["a", "k"]);
        encode_list(&mut out, &["vowel"]);
        encode_list(&mut out, &["a", "k"]);
        encode_list(&mut out, &["vowel"]);
        assert_eq!(
            read_phone_set(&load(&out), 0).unwrap_err().to_string(),
            "Phoneme _psv2 table has inconsistent symbol metadata."
        );
    }

    #[test]
    fn empty_name_is_rejected() {
        let mut out = Vec::new();
        encode_string(&mut out, "");
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["vowel"]);
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["vowel"]);
        assert_eq!(
            read_phone_set(&load(&out), 0).unwrap_err().to_string(),
            "Phoneme _psv2 table has inconsistent symbol metadata."
        );
    }

    #[test]
    fn duplicate_class_names_are_rejected() {
        let mut out = Vec::new();
        encode_string(&mut out, "x");
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["vowel"]);
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["vowel", "vowel"]);
        assert_eq!(
            read_phone_set(&load(&out), 0).unwrap_err().to_string(),
            "Phoneme _psv2 class names must be nonempty and unique."
        );
    }

    #[test]
    fn unknown_category_is_rejected() {
        let mut out = Vec::new();
        encode_string(&mut out, "x");
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["pluck"]);
        encode_list(&mut out, &["a"]);
        encode_list(&mut out, &["vowel"]);
        assert_eq!(
            read_phone_set(&load(&out), 0).unwrap_err().to_string(),
            "Phoneme _psv2 contains an invalid symbol, category, or mapping."
        );
    }

    #[test]
    fn duplicate_symbols_are_rejected() {
        let mut out = Vec::new();
        encode_string(&mut out, "x");
        encode_list(&mut out, &["a", "a"]);
        encode_list(&mut out, &["vowel", "vowel"]);
        encode_list(&mut out, &["a", "a"]);
        encode_list(&mut out, &["vowel"]);
        assert!(
            read_phone_set(&load(&out), 0)
                .unwrap_err()
                .to_string()
                .contains("invalid symbol")
        );
    }

    #[test]
    fn oversized_string_length_is_rejected() {
        let mut out = Vec::new();
        out.extend_from_slice(&5000_u32.to_le_bytes());
        out.extend_from_slice(&[b'a'; 8]);
        assert_eq!(
            read_phone_set(&load(&out), 0).unwrap_err().to_string(),
            "Invalid _psv2 string table."
        );
    }
}
