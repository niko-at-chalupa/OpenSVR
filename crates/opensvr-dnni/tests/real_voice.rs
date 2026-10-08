//! Gated tests against the real voice models in a local `.nofs` file.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice in the
//! OpenSV references) to run; the test returns early when unset so CI stays
//! green without proprietary data. Voice files are never committed.

use std::path::PathBuf;

use opensvr_dnni::DnniReader;
use opensvr_nofs::VoiceDatabase;

/// Minimum payload size treated as a neural model entry.
///
/// The four known model entries are 4–22 MiB; the `0x91` config blob and
/// metadata entries are a few KiB at most.
const MODEL_SIZE_THRESHOLD: u32 = 1_000_000;

#[test]
fn real_model_entries_parse() {
    let Ok(voice) = std::env::var("OPENSVR_TEST_VOICE") else {
        return;
    };
    let db = VoiceDatabase::open(PathBuf::from(&voice).as_path()).expect("voice must open");

    // The pitch model has a printable key and must resolve to a known type.
    let pitch = db
        .find_entry("f0model-dds")
        .expect("voice must contain f0model-dds");
    let pitch_bytes = db.read_entry(pitch).expect("pitch model must read");
    let pitch_model = DnniReader::from_bytes(pitch_bytes.to_vec()).expect("pitch model parses");
    assert!(matches!(pitch_model.version(), 1 | 2));
    assert!(!pitch_model.nodes().is_empty());
    // The pitch root itself is not in the known-type table (OpenSV renders it
    // as hex too), but the tree must contain resolved primitive/model nodes.
    assert!(
        pitch_model
            .nodes()
            .iter()
            .any(|node| node.name == "prim0" || node.name.starts_with("mod")),
        "pitch model must contain resolved layer names"
    );
    println!(
        "f0model-dds: version {} nodes {} root {}",
        pitch_model.version(),
        pitch_model.nodes().len(),
        pitch_model.nodes()[0].name
    );

    // Every large entry in the archive must be a parseable DNNI model.
    let mut models = 0;
    for entry in db.entries() {
        if entry.value_size < MODEL_SIZE_THRESHOLD {
            continue;
        }
        let bytes = db.read_entry(entry).expect("model entry must read");
        let model = DnniReader::from_bytes(bytes.to_vec())
            .expect("model entry {entry:?} must parse");
        assert!(!model.nodes().is_empty());
        println!(
            "model size {} version {} nodes {} root {}",
            entry.value_size,
            model.version(),
            model.nodes().len(),
            model.nodes()[0].name
        );
        models += 1;
    }
    assert!(models >= 4, "expected at least 4 model entries, saw {models}");
}

#[test]
fn real_prim0_matrix_matches_independent_decode() {
    let Ok(voice) = std::env::var("OPENSVR_TEST_VOICE") else {
        return;
    };
    let db = VoiceDatabase::open(PathBuf::from(&voice).as_path()).expect("voice must open");
    let pitch = db
        .find_entry("f0model-dds")
        .expect("voice must contain f0model-dds");
    let pitch_bytes = db.read_entry(pitch).expect("pitch model must read");
    let model = DnniReader::from_bytes(pitch_bytes.to_vec()).expect("pitch model parses");

    let index = model
        .nodes()
        .iter()
        .position(|node| node.name == "prim0")
        .expect("pitch model must contain a prim0 matrix");
    let node = &model.nodes()[index];

    // Independent decode straight from the payload bytes, without going
    // through `read_float_matrix`: fixed dense layout, manual LE conversion.
    let payload = model.payload(index);
    assert!(payload.len() >= 8);
    let rows = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let columns = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    let rest = &payload[8..];
    assert_eq!(rest.len() % 4, 0);
    let mut expected = Vec::with_capacity(rest.len() / 4);
    for chunk in rest.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        assert!(value.is_finite());
        expected.push(value);
    }

    let matrix = model.read_float_matrix(index).expect("prim0 must decode");
    assert_eq!(matrix.rows, rows);
    assert_eq!(matrix.columns, columns);
    assert_eq!(matrix.values, expected);

    let head: Vec<f32> = matrix.values.iter().copied().take(8).collect();
    let sum: f64 = matrix.values.iter().map(|v| f64::from(*v)).sum();
    println!(
        "prim0 node {index} at file offset {:#x}: {rows}x{columns} head {head:?} sum {sum}",
        node.offset
    );
}
