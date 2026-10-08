//! Binary reader and parser for Synthesizer V DNNI neural network files.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/DnniReader.{h,cpp}`
//! (namespace `sv::synthesis`). Error strings match the C++ `juce::Result` text
//! so golden tests can compare messages.

use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

use thiserror::Error;

/// Errors raised while loading or decoding a DNNI model.
///
/// The `Invalid` payloads reproduce the C++ `juce::Result` failure strings
/// verbatim (including the `DNNI ...` prefix and `0x...` hex offsets).
#[derive(Debug, Error)]
pub enum DnniError {
    #[error("{0}")]
    Invalid(String),
    #[error("could not read DNNI file {}: {source}", .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

type Result<T> = std::result::Result<T, DnniError>;

fn invalid(message: impl Into<String>) -> DnniError {
    DnniError::Invalid(message.into())
}

fn malformed(offset: usize, reason: &str) -> DnniError {
    invalid(format!("DNNI offset 0x{offset:x}: {reason}"))
}

const MAXIMUM_MODEL_BYTES: usize = 512 * 1024 * 1024;
const MAXIMUM_NODES: usize = 1_000_000;
const MAXIMUM_DEPTH: u32 = 256;
const NODE_HEADER_BYTES: usize = 20;
const FILE_MAGIC: u32 = 0x7fca_00ff;
const NODE_MARKERS: [u32; 2] = [0x7fca_40ff, 0x7fca_41ff];
const CHILD_COUNT_XOR_MASK: u32 = 0xac5e_7bd5;
const FNV_PRIME_64: u64 = 0x0100_0000_01b3;

/// One node in the DNNI neural network tree.
///
/// Module nodes represent compound layers (e.g. GRU, dilated conv, ResNet
/// blocks); leaf primitive nodes store weights, biases or metadata (e.g. a
/// `prim0` matrix or `prim1` vector).
#[derive(Debug, Clone)]
pub struct DnniNode {
    /// Node header marker word (`0x7fca40ff` or `0x7fca41ff`).
    pub marker: u32,
    /// 64-bit type hash / identifier.
    pub type_id: u64,
    /// Human-readable layer or primitive name (e.g. `prim0`, `modl0`).
    /// Unknown v2 hashes render as `0x` plus 16 lowercase hex digits.
    pub name: String,
    /// Byte offset of the 20-byte node header in the file.
    pub offset: usize,
    /// Byte offset of the node's raw payload data.
    pub payload_offset: usize,
    /// Length in bytes of the node's payload.
    pub payload_size: usize,
    /// Indices of child nodes in the flat node array.
    pub children: Vec<usize>,
}

/// Decoded 2D weight matrix in row-major layout.
///
/// `values[row * columns + column]`; `rows` is the output dimension and
/// `columns` the input dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct DnniMatrix {
    pub rows: u32,
    pub columns: u32,
    pub values: Vec<f32>,
}

/// A parsed DNNI model: the raw bytes plus the flat node table.
///
/// Payloads are borrowed from the in-memory file image, so models are read
/// once and decoded lazily per node.
#[derive(Debug, Default)]
pub struct DnniReader {
    data: Vec<u8>,
    version: u32,
    nodes: Vec<DnniNode>,
}

impl DnniReader {
    /// Loads and parses a DNNI model from disk.
    ///
    /// On failure no partial state is kept.
    pub fn load(path: &Path) -> Result<Self> {
        let size = fs::metadata(path).map_or(0, |meta| meta.len());
        if size < 8 || size > MAXIMUM_MODEL_BYTES as u64 {
            return Err(invalid(format!(
                "DNNI file is missing or outside the supported size range: {}",
                path.display()
            )));
        }
        let data = fs::read(path).map_err(|source| DnniError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::from_bytes(data)
    }

    /// Loads and parses a DNNI model from an in-memory byte block.
    ///
    /// On failure no partial state is kept; the error message matches the C++
    /// `juce::Result` text.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let mut reader = Self {
            data,
            version: 0,
            nodes: Vec::new(),
        };
        reader.parse()?;
        Ok(reader)
    }

    /// Returns the format version (`1` or `2`).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns the flat list of parsed nodes.
    pub fn nodes(&self) -> &[DnniNode] {
        &self.nodes
    }

    /// Returns a zero-copy byte view of the payload for the given node.
    ///
    /// Mirrors the C++ behavior of returning an empty span for an
    /// out-of-range node index.
    pub fn payload(&self, node_index: usize) -> &[u8] {
        let Some(node) = self.nodes.get(node_index) else {
            return &[];
        };
        let end = node.payload_offset.saturating_add(node.payload_size);
        self.data
            .get(node.payload_offset..end)
            .filter(|slice| slice.len() == node.payload_size)
            .unwrap_or(&[])
    }

    /// Decodes a `prim1` 1D float vector node (e.g. bias weights).
    pub fn read_float_vector(&self, node_index: usize) -> Result<Vec<f32>> {
        let Some(node) = self.nodes.get(node_index) else {
            return Err(invalid(
                "The requested DNNI node is not a prim1 float vector.",
            ));
        };
        if node.name != "prim1" {
            return Err(invalid(
                "The requested DNNI node is not a prim1 float vector.",
            ));
        }
        let payload = self.payload(node_index);
        if payload.len() < 4 {
            return Err(malformed(node.payload_offset, "truncated vector size."));
        }
        let count = read_u32(payload, 0) as usize;
        decode_floats(&payload[4..], count)
    }

    /// Decodes a matrix node (`prim0`, `prim2`, `prim3`, `prim4`, `prim5`)
    /// into floating-point weights.
    pub fn read_float_matrix(&self, node_index: usize) -> Result<DnniMatrix> {
        let Some(node) = self.nodes.get(node_index) else {
            return Err(invalid(
                "The requested DNNI node is not a supported float matrix.",
            ));
        };
        if !matches!(
            node.name.as_str(),
            "prim0" | "prim2" | "prim3" | "prim4" | "prim5"
        ) {
            return Err(invalid(
                "The requested DNNI node is not a supported float matrix.",
            ));
        }
        decode_matrix(self.payload(node_index), &node.name)
    }

    fn parse(&mut self) -> Result<()> {
        if self.data.len() < 8 || self.data.len() > MAXIMUM_MODEL_BYTES {
            return Err(malformed(
                0,
                "model must contain a header and fit within 512 MiB.",
            ));
        }
        if read_u32(&self.data, 0) != FILE_MAGIC {
            return Err(malformed(0, "unrecognised file signature."));
        }
        let version = read_u32(&self.data, 4);
        if version != 1 && version != 2 {
            return Err(malformed(
                4,
                &format!("unsupported format version {version}."),
            ));
        }
        self.version = version;

        let mut position = 8_usize;
        let mut root = 0_usize;
        self.parse_node(&mut position, 0, &mut root)?;
        if position != self.data.len() {
            return Err(malformed(position, "trailing bytes after the root node."));
        }
        Ok(())
    }

    fn parse_node(
        &mut self,
        position: &mut usize,
        depth: u32,
        node_index: &mut usize,
    ) -> Result<()> {
        if depth > MAXIMUM_DEPTH || self.nodes.len() >= MAXIMUM_NODES {
            return Err(malformed(
                *position,
                "node depth or count limit exceeded.",
            ));
        }
        if self.data.len().saturating_sub(*position) < NODE_HEADER_BYTES {
            return Err(malformed(*position, "truncated node header."));
        }
        let header_at = *position;
        let marker = read_u32(&self.data, header_at);
        if !NODE_MARKERS.contains(&marker) {
            return Err(malformed(header_at, "unrecognised node marker."));
        }
        let type_low = read_u32(&self.data, header_at + 4);
        let type_high = read_u32(&self.data, header_at + 8);
        let type_id = u64::from(type_low) | (u64::from(type_high) << 32);

        let mut name = String::new();
        if self.version == 1 {
            let mut terminated = false;
            for index in 4..12_usize {
                let character = self.data[header_at + index];
                if character == 0 {
                    terminated = true;
                } else if terminated || !(0x20..=0x7e).contains(&character) {
                    return Err(malformed(header_at + index, "invalid node type tag."));
                } else {
                    name.push(character as char);
                }
            }
        } else {
            name = resolve_type(type_id);
            if name.is_empty() {
                name = format!("0x{type_id:016x}");
            }
        }
        if name.is_empty() {
            return Err(malformed(header_at + 4, "empty node type tag."));
        }

        let mut child_count = read_u32(&self.data, header_at + 12);
        if self.version == 2 {
            child_count ^= type_low ^ type_high ^ CHILD_COUNT_XOR_MASK;
            if child_count % 3 != 0 {
                return Err(malformed(
                    header_at + 12,
                    "invalid version 2 child count encoding.",
                ));
            }
            child_count /= 3;
        }

        let payload_size = read_u32(&self.data, header_at + 16) as usize;
        if payload_size > self.data.len().saturating_sub(header_at + NODE_HEADER_BYTES) {
            return Err(malformed(header_at + 16, "payload extends past the file."));
        }
        let payload_offset = header_at + NODE_HEADER_BYTES;
        *position = payload_offset + payload_size;
        let child_count = child_count as usize;
        if child_count > self.data.len().saturating_sub(*position) / NODE_HEADER_BYTES
            || child_count > MAXIMUM_NODES.saturating_sub(self.nodes.len())
        {
            return Err(malformed(
                payload_offset - 8,
                "child count exceeds the remaining file.",
            ));
        }

        *node_index = self.nodes.len();
        self.nodes.push(DnniNode {
            marker,
            type_id,
            name,
            offset: header_at,
            payload_offset,
            payload_size,
            children: Vec::new(),
        });
        self.nodes[*node_index].children.reserve(child_count);

        for _ in 0..child_count {
            let mut child_index = 0_usize;
            self.parse_node(position, depth + 1, &mut child_index)?;
            self.nodes[*node_index].children.push(child_index);
        }
        Ok(())
    }
}

/// Reads a 32-bit unsigned little-endian integer.
///
/// Callers guarantee `bytes.len() - offset >= 4` via the bounds checks in
/// [`DnniReader::parse_node`] and the payload decoders below.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from(bytes[offset])
        | (u32::from(bytes[offset + 1]) << 8)
        | (u32::from(bytes[offset + 2]) << 16)
        | (u32::from(bytes[offset + 3]) << 24)
}

/// Resolves an obfuscated 64-bit DNNI version 2 type id to its ASCII name.
///
/// In format version 2, layer type names are hashed with 64-bit FNV-1a using
/// one of 12 known seed values. Returns an empty string for unknown hashes.
fn resolve_type(type_id: u64) -> String {
    const SEEDS: [u64; 12] = [
        0x0dcd_5918_9d5a_0f24,
        0x5934_6d79_970c_a21e,
        0xcf35_19b7_73b7_67bf,
        0x562c_8e41_fb7f_bee2,
        0xbd6d_2545_7e1e_d24e,
        0x0123_4567_89ab_cdef,
        0x7654_3210_fedc_ba98,
        0x0246_8ace_eca8_6420,
        0xeca8_6420_2468_acee,
        0x7055_6f59_6576_6947,
        0x5a4d_5a4d_5a4d_5a4d,
        0x0000_0062_6d6f_6379,
    ];
    const NAMES: [&str; 37] = [
        "prim0", "prim1", "prim2", "prim3", "prim4", "prim5", "modm0", "modl0",
        "modl1", "modl3", "modl4", "modl6", "moda0", "moda1", "moda2", "moda3",
        "moda4", "moda5", "moda7", "_gnc1v0", "_ncwnv0", "cmpg1", "cmpu0",
        "cmpu1", "_vocfv1", "_vocfv2", "_ppusv0", "_ppdsv0", "_psv2",
        "_rldtg0", "_rldms0", "_stbkv1", "_vqctx1", "_didsv0", "_ftmfv2",
        "_ftmfv3", "_dctov0",
    ];

    for name in NAMES {
        for seed in SEEDS {
            let mut hash = seed;
            for byte in name.bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME_64);
            }
            if hash == type_id {
                return name.to_owned();
            }
        }
    }
    String::new()
}

/// Decodes an array of 32-bit IEEE-754 floats, rejecting non-finite values.
fn decode_floats(bytes: &[u8], count: usize) -> Result<Vec<f32>> {
    let count_u64 = count as u64;
    if count_u64 > bytes.len() as u64 / 4 || count_u64 * 4 != bytes.len() as u64 {
        return Err(invalid(
            "DNNI float tensor dimensions do not match its payload.",
        ));
    }
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let value = f32::from_le_bytes([
            bytes[index * 4],
            bytes[index * 4 + 1],
            bytes[index * 4 + 2],
            bytes[index * 4 + 3],
        ]);
        if !value.is_finite() {
            return Err(invalid(
                "DNNI float tensor contains a non-finite value.",
            ));
        }
        values.push(value);
    }
    Ok(values)
}

/// Reads a signed quantized integer (8-bit or 16-bit) from byte memory.
fn read_quantized(bytes: &[u8], offset: usize, bits: u32) -> i32 {
    if bits == 8 {
        i32::from(i8::from_ne_bytes([bytes[offset]]))
    } else {
        let raw = u16::from(bytes[offset]) | (u16::from(bytes[offset + 1]) << 8);
        i32::from(i16::from_le_bytes(raw.to_le_bytes()))
    }
}

/// Decompresses and dequantizes weight matrices from DNNI payloads.
///
/// Supported representations: unquantized dense (`prim0`, `prim2`), quantized
/// dense (`prim4`, 8/16-bit with per-row scaling), unquantized sparse
/// (`prim3`, Block Compressed Sparse Row) and quantized sparse (`prim5`).
#[allow(clippy::too_many_lines)]
fn decode_matrix(payload: &[u8], kind: &str) -> Result<DnniMatrix> {
    let quantized = kind == "prim4" || kind == "prim5";
    let sparse = kind == "prim3" || kind == "prim5";
    let mut bits = 32_u32;
    let mut scales = Vec::new();
    let mut payload = payload;

    // 1. Quantization header for quantized matrices.
    if quantized {
        if payload.len() < 16 {
            return Err(invalid("DNNI quantization header is truncated."));
        }
        bits = read_u32(payload, 0);
        if bits != 8 && bits != 16 {
            return Err(invalid(
                "DNNI quantized matrix supports only verified signed 8/16-bit weights.",
            ));
        }
        // The second field controls original-engine input quantization, not
        // weight scaling.
        if read_u32(payload, 8) != 0 {
            return Err(invalid(
                "DNNI residual quantization is not yet implemented.",
            ));
        }
        let scale_count = read_u32(payload, 12) as usize;
        if scale_count > payload.len().saturating_sub(16) / 4 {
            return Err(invalid(
                "DNNI quantization scale array is truncated.",
            ));
        }
        scales = decode_floats(&payload[16..16 + scale_count * 4], scale_count)?;
        payload = &payload[16 + scale_count * 4..];
    }

    // 2. Dimensions (rows, columns).
    let header_size = if sparse { 20 } else { 8 };
    if payload.len() < header_size {
        return Err(invalid("DNNI matrix header is truncated."));
    }
    let rows = read_u32(payload, 0);
    let columns = read_u32(payload, 4);
    let element_count = u64::from(rows) * u64::from(columns);
    if element_count > MAXIMUM_MODEL_BYTES as u64 / 4 {
        return Err(invalid("Decoded DNNI matrix exceeds the 512 MiB limit."));
    }
    if quantized && scales.len() as u64 != u64::from(rows) {
        return Err(invalid(
            "DNNI quantized matrix must have one scale per output row.",
        ));
    }
    let element_count = element_count as usize;

    // 3. Dense unquantized float matrix.
    if !sparse && !quantized {
        let values = decode_floats(&payload[8..], element_count)?;
        return Ok(DnniMatrix {
            rows,
            columns,
            values,
        });
    }

    let bytes_per_element = (bits / 8) as usize;
    // 128.0 for 8-bit, 32768.0 for 16-bit (mirrors `std::ldexp(1.0f, bits - 1)`).
    // The 32-bit branch only feeds the sparse-unquantized path, where the
    // divisor is unused; the constant keeps the arithmetic total.
    let divisor = match bits {
        8 => 128.0_f32,
        16 => 32_768.0_f32,
        _ => 2_147_483_648.0_f32,
    };

    // 4. Dense quantized matrix (row-scaled, column-major integer storage
    //    mapped to row-major float storage).
    if !sparse {
        payload = &payload[8..];
        if element_count as u64 * bytes_per_element as u64 != payload.len() as u64 {
            return Err(invalid(
                "DNNI quantized matrix dimensions do not match its payload.",
            ));
        }
        let mut values = vec![0.0_f32; element_count];
        for row in 0..rows as usize {
            for column in 0..columns as usize {
                let offset = (column * rows as usize + row) * bytes_per_element;
                let integer = read_quantized(payload, offset, bits);
                values[row * columns as usize + column] =
                    (integer as f32 / divisor) * scales[row];
            }
        }
        return Ok(DnniMatrix {
            rows,
            columns,
            values,
        });
    }

    // 5. Block Compressed Sparse Row (BCSR) sparse matrix.
    let block_rows = read_u32(payload, 8) as usize;
    let block_columns = read_u32(payload, 12) as usize;
    let block_count = read_u32(payload, 16) as usize;
    if block_rows == 0
        || block_columns == 0
        || block_rows > rows as usize
        || block_columns > columns as usize
        || rows as usize % block_rows != 0
        || columns as usize % block_columns != 0
    {
        return Err(invalid(
            "DNNI sparse matrix has unsupported partial or empty block dimensions.",
        ));
    }
    let block_row_count = rows as usize / block_rows;
    let index_bytes = block_count as u64 * 2;
    let pointer_bytes = (block_row_count as u64 + 1) * 4;
    let block_elements = block_rows * block_columns;
    if block_count as u64 > MAXIMUM_MODEL_BYTES as u64 / bytes_per_element as u64 / block_elements as u64
    {
        return Err(invalid(
            "DNNI sparse coefficient array exceeds the memory limit.",
        ));
    }
    let coefficient_count = block_count * block_elements;
    if coefficient_count as u64 > MAXIMUM_MODEL_BYTES as u64 / bytes_per_element as u64
        || 20 + index_bytes + pointer_bytes
            + coefficient_count as u64 * bytes_per_element as u64
            != payload.len() as u64
    {
        return Err(invalid(
            "DNNI sparse matrix arrays do not match its payload.",
        ));
    }
    let base = 20 + index_bytes as usize;
    let pointers_at = base + pointer_bytes as usize;
    let indices = &payload[20..base];
    let pointers = &payload[base..pointers_at];
    let coefficients = &payload[pointers_at..];
    if read_u32(pointers, 0) != 0
        || read_u32(pointers, block_row_count * 4) != block_count as u32
    {
        return Err(invalid(
            "DNNI sparse row pointers do not span all blocks.",
        ));
    }
    let mut values = vec![0.0_f32; element_count];
    for block_row in 0..block_row_count {
        let begin = read_u32(pointers, block_row * 4) as usize;
        let end = read_u32(pointers, (block_row + 1) * 4) as usize;
        if begin > end || end > block_count {
            return Err(invalid(
                "DNNI sparse row pointers are out of order or range.",
            ));
        }
        for block in begin..end {
            let column_block =
                indices[block * 2] as usize | ((indices[block * 2 + 1] as usize) << 8);
            if column_block >= columns as usize / block_columns {
                return Err(invalid(
                    "DNNI sparse block column is outside the matrix.",
                ));
            }
            for column in 0..block_columns {
                for row in 0..block_rows {
                    let at = (block * block_rows * block_columns
                        + column * block_rows
                        + row)
                        * bytes_per_element;
                    let output_row = block_row * block_rows + row;
                    let output_column = column_block * block_columns + column;
                    let value = if quantized {
                        (read_quantized(coefficients, at, bits) as f32 / divisor)
                            * scales[output_row]
                    } else {
                        f32::from_le_bytes([
                            coefficients[at],
                            coefficients[at + 1],
                            coefficients[at + 2],
                            coefficients[at + 3],
                        ])
                    };
                    let destination =
                        &mut values[output_row * columns as usize + output_column];
                    *destination += value;
                    if !destination.is_finite() {
                        return Err(invalid(
                            "DNNI sparse matrix has a non-finite accumulated weight.",
                        ));
                    }
                }
            }
        }
    }
    Ok(DnniMatrix {
        rows,
        columns,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v1_tag(name: &str) -> [u8; 8] {
        let mut tag = [0_u8; 8];
        tag[..name.len()].copy_from_slice(name.as_bytes());
        tag
    }

    fn v1_leaf(name: &str, payload: &[u8]) -> Vec<u8> {
        v1_node_raw(v1_tag(name), 0, payload, &[])
    }

    fn v1_node(name: &str, payload: &[u8], children: u32, child_bytes: &[u8]) -> Vec<u8> {
        v1_node_raw(v1_tag(name), children, payload, child_bytes)
    }

    fn v1_node_raw(tag: [u8; 8], children: u32, payload: &[u8], child_bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        out.extend_from_slice(&tag);
        out.extend_from_slice(&children.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(child_bytes);
        out
    }

    fn file_bytes(version: u32, root: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&FILE_MAGIC.to_le_bytes());
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(root);
        out
    }

    fn fnv(seed: u64, name: &str) -> u64 {
        let mut hash = seed;
        for byte in name.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME_64);
        }
        hash
    }

    fn v2_node(type_id: u64, children: u32, payload: &[u8], child_bytes: &[u8]) -> Vec<u8> {
        let low = type_id as u32;
        let high = (type_id >> 32) as u32;
        let encoded = children
            .wrapping_mul(3)
            ^ low
            ^ high
            ^ CHILD_COUNT_XOR_MASK;
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_41ff_u32.to_le_bytes());
        out.extend_from_slice(&type_id.to_le_bytes());
        out.extend_from_slice(&encoded.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(child_bytes);
        out
    }

    fn vector_payload(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn dense_payload(rows: u32, columns: u32, values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&columns.to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn quant_header(bits: u32, residual: u32, scales: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&residual.to_le_bytes());
        out.extend_from_slice(&(scales.len() as u32).to_le_bytes());
        for scale in scales {
            out.extend_from_slice(&scale.to_le_bytes());
        }
        out
    }

    fn load_v1(name: &str, payload: &[u8]) -> DnniReader {
        let child = v1_node_raw(v1_tag(name), 0, payload, &[]);
        DnniReader::from_bytes(file_bytes(1, &child)).expect("synthetic v1 must parse")
    }

    #[test]
    fn v1_tree_with_child_parses() {
        let child = v1_leaf("prim1", b"");
        let root = v1_node("modm0", b"hi", 1, &child);
        let reader = DnniReader::from_bytes(file_bytes(1, &root)).unwrap();
        assert_eq!(reader.version(), 1);
        assert_eq!(reader.nodes().len(), 2);
        let root_node = &reader.nodes()[0];
        assert_eq!(root_node.marker, 0x7fca_40ff);
        assert_eq!(root_node.name, "modm0");
        assert_eq!(root_node.offset, 8);
        assert_eq!(root_node.payload_offset, 28);
        assert_eq!(root_node.payload_size, 2);
        assert_eq!(root_node.children, vec![1]);
        assert_eq!(reader.payload(0), b"hi");
        let leaf = &reader.nodes()[1];
        assert_eq!(leaf.name, "prim1");
        assert!(leaf.children.is_empty());
        assert_eq!(leaf.type_id, u64::from_le_bytes({
            let mut id = [0_u8; 8];
            id.copy_from_slice(&v1_tag("prim1"));
            id
        }));
    }

    #[test]
    fn v1_empty_type_tag_is_rejected() {
        let node = v1_node_raw([0; 8], 0, &[], &[]);
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0xc: empty node type tag.");
    }

    #[test]
    fn v1_non_printable_type_tag_is_rejected() {
        let tag = [b'a', 0x01, 0, 0, 0, 0, 0, 0];
        let node = v1_node_raw(tag, 0, &[], &[]);
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0xd: invalid node type tag.");
    }

    #[test]
    fn v1_tag_bytes_after_nul_are_rejected() {
        let tag = [b'a', 0, b'b', 0, 0, 0, 0, 0];
        let node = v1_node_raw(tag, 0, &[], &[]);
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0xe: invalid node type tag.");
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = file_bytes(1, &v1_leaf("prim1", b""));
        bytes[0] = 0x00;
        let error = DnniReader::from_bytes(bytes).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0x0: unrecognised file signature.");
    }

    #[test]
    fn bad_version_is_rejected() {
        let error = DnniReader::from_bytes(file_bytes(3, &[])).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI offset 0x4: unsupported format version 3."
        );
    }

    #[test]
    fn short_file_is_rejected() {
        let error = DnniReader::from_bytes(vec![0x7f, 0xca]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI offset 0x0: model must contain a header and fit within 512 MiB."
        );
    }

    #[test]
    fn bad_marker_is_rejected() {
        let mut node = v1_leaf("prim1", b"");
        node[0] = 0x00;
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0x8: unrecognised node marker.");
    }

    #[test]
    fn truncated_header_is_rejected() {
        let mut bytes = file_bytes(1, &[]);
        bytes.truncate(12);
        let error = DnniReader::from_bytes(bytes).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0x8: truncated node header.");
    }

    #[test]
    fn payload_past_file_is_rejected() {
        let mut node = v1_leaf("prim1", b"abc");
        node.truncate(node.len() - 1);
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0x18: payload extends past the file.");
    }

    #[test]
    fn child_count_past_file_is_rejected() {
        let node = v1_node_raw(v1_tag("modm0"), 4, &[], &[]);
        let error = DnniReader::from_bytes(file_bytes(1, &node)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI offset 0x14: child count exceeds the remaining file."
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = file_bytes(1, &v1_leaf("prim1", b""));
        bytes.push(0x00);
        let error = DnniReader::from_bytes(bytes).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI offset 0x1c: trailing bytes after the root node."
        );
    }

    #[test]
    fn depth_limit_is_enforced() {
        let mut nested = v1_leaf("prim1", b"");
        for _ in 0..258 {
            nested = v1_node("modm0", &[], 1, &nested);
        }
        let error = DnniReader::from_bytes(file_bytes(1, &nested)).unwrap_err();
        assert!(
            error
                .to_string()
                .ends_with("node depth or count limit exceeded."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn v2_known_names_resolve() {
        for name in ["prim0", "prim1", "modl0", "_psv2", "_dctov0", "cmpu1"] {
            let id = fnv(0x0dcd_5918_9d5a_0f24, name);
            let node = v2_node(id, 0, &[], &[]);
            let reader = DnniReader::from_bytes(file_bytes(2, &node)).unwrap();
            assert_eq!(reader.version(), 2);
            assert_eq!(reader.nodes()[0].name, name, "seed 0 must resolve {name}");
            assert_eq!(reader.nodes()[0].type_id, id);
        }
    }

    #[test]
    fn v2_unknown_id_renders_as_hex() {
        let node = v2_node(0x1234_5678_9abc_def0, 0, &[], &[]);
        let reader = DnniReader::from_bytes(file_bytes(2, &node)).unwrap();
        assert_eq!(reader.nodes()[0].name, "0x123456789abcdef0");
    }

    #[test]
    fn v2_obfuscated_child_count_round_trips() {
        let leaf_a = v2_node(fnv(0x0123_4567_89ab_cdef, "prim0"), 0, &[1, 2], &[]);
        let leaf_b = v2_node(fnv(0x0123_4567_89ab_cdef, "prim1"), 0, &[3], &[]);
        let mut children = Vec::new();
        children.extend_from_slice(&leaf_a);
        children.extend_from_slice(&leaf_b);
        let root = v2_node(fnv(0x0123_4567_89ab_cdef, "modm0"), 2, &[], &children);
        let reader = DnniReader::from_bytes(file_bytes(2, &root)).unwrap();
        assert_eq!(reader.nodes().len(), 3);
        assert_eq!(reader.nodes()[0].children, vec![1, 2]);
        assert_eq!(reader.payload(1), &[1, 2]);
        assert_eq!(reader.payload(2), &[3]);
    }

    #[test]
    fn v2_bad_child_encoding_is_rejected() {
        let id = fnv(0x0dcd_5918_9d5a_0f24, "prim0");
        let low = id as u32;
        let high = (id >> 32) as u32;
        // Encoded value 1 is not divisible by 3 after the XOR mask.
        let encoded = 1_u32 ^ low ^ high ^ CHILD_COUNT_XOR_MASK;
        let mut node = Vec::new();
        node.extend_from_slice(&0x7fca_41ff_u32.to_le_bytes());
        node.extend_from_slice(&id.to_le_bytes());
        node.extend_from_slice(&encoded.to_le_bytes());
        node.extend_from_slice(&0_u32.to_le_bytes());
        let error = DnniReader::from_bytes(file_bytes(2, &node)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI offset 0x14: invalid version 2 child count encoding."
        );
    }

    #[test]
    fn missing_file_reports_path() {
        let error =
            DnniReader::load(Path::new("/nonexistent-model-12345.dnni")).unwrap_err();
        assert_eq!(
            error.to_string(),
            "DNNI file is missing or outside the supported size range: \
             /nonexistent-model-12345.dnni"
        );
    }

    #[test]
    fn payload_out_of_range_is_empty() {
        let reader = load_v1("prim1", &vector_payload(&[1.0]));
        assert_eq!(reader.payload(99), &[]);
    }

    #[test]
    fn prim1_vector_round_trips() {
        let reader = load_v1("prim1", &vector_payload(&[1.0, -2.5, 0.0]));
        assert_eq!(reader.read_float_vector(0).unwrap(), vec![1.0, -2.5, 0.0]);
    }

    #[test]
    fn prim1_rejects_wrong_type_and_index() {
        let reader = load_v1("prim0", &dense_payload(1, 1, &[1.0]));
        assert_eq!(
            reader.read_float_vector(0).unwrap_err().to_string(),
            "The requested DNNI node is not a prim1 float vector."
        );
        assert_eq!(
            reader.read_float_vector(7).unwrap_err().to_string(),
            "The requested DNNI node is not a prim1 float vector."
        );
    }

    #[test]
    fn prim1_truncated_size_is_rejected() {
        let reader = load_v1("prim1", &[0x01, 0x02]);
        let error = reader.read_float_vector(0).unwrap_err();
        assert_eq!(error.to_string(), "DNNI offset 0x1c: truncated vector size.");
    }

    #[test]
    fn prim1_count_mismatch_is_rejected() {
        let mut payload = vec![5, 0, 0, 0];
        payload.extend_from_slice(&1.0_f32.to_le_bytes());
        let reader = load_v1("prim1", &payload);
        assert_eq!(
            reader.read_float_vector(0).unwrap_err().to_string(),
            "DNNI float tensor dimensions do not match its payload."
        );
    }

    #[test]
    fn prim1_non_finite_is_rejected() {
        let reader = load_v1("prim1", &vector_payload(&[f32::INFINITY]));
        assert_eq!(
            reader.read_float_vector(0).unwrap_err().to_string(),
            "DNNI float tensor contains a non-finite value."
        );
    }

    #[test]
    fn prim0_dense_round_trips() {
        let values = vec![1.0, -2.0, 3.5, 4.25, 0.0, -0.5];
        let reader = load_v1("prim0", &dense_payload(2, 3, &values));
        let matrix = reader.read_float_matrix(0).unwrap();
        assert_eq!(
            matrix,
            DnniMatrix {
                rows: 2,
                columns: 3,
                values,
            }
        );
    }

    #[test]
    fn prim2_uses_the_dense_path() {
        let reader = load_v1("prim2", &dense_payload(1, 2, &[1.0, 2.0]));
        let matrix = reader.read_float_matrix(0).unwrap();
        assert_eq!(matrix.values, vec![1.0, 2.0]);
    }

    #[test]
    fn matrix_rejects_wrong_type_and_index() {
        let reader = load_v1("prim1", &vector_payload(&[1.0]));
        for index in [0, 9] {
            assert_eq!(
                reader.read_float_matrix(index).unwrap_err().to_string(),
                "The requested DNNI node is not a supported float matrix."
            );
        }
    }

    #[test]
    fn dense_non_finite_is_rejected() {
        let reader = load_v1("prim0", &dense_payload(1, 1, &[f32::NAN]));
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI float tensor contains a non-finite value."
        );
    }

    #[test]
    fn dense_payload_mismatch_is_rejected() {
        let mut payload = dense_payload(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        payload.push(0x00);
        let reader = load_v1("prim0", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI float tensor dimensions do not match its payload."
        );
    }

    #[test]
    fn matrix_header_truncated_is_rejected() {
        let reader = load_v1("prim0", &[1, 2, 3, 4]);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI matrix header is truncated."
        );
    }

    #[test]
    fn huge_matrix_is_rejected_before_allocation() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        payload.extend_from_slice(&2_u32.to_le_bytes());
        let reader = load_v1("prim0", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "Decoded DNNI matrix exceeds the 512 MiB limit."
        );
    }

    #[test]
    fn quant8_dense_decodes_column_major() {
        // 2 rows x 3 columns, scales [2.0, 0.5]; integers stored column-major.
        let mut payload = quant_header(8, 0, &[2.0, 0.5]);
        payload.extend_from_slice(&2_u32.to_le_bytes());
        payload.extend_from_slice(&3_u32.to_le_bytes());
        payload.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
        let reader = load_v1("prim4", &payload);
        let matrix = reader.read_float_matrix(0).unwrap();
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.columns, 3);
        let expected = vec![
            10.0 / 128.0 * 2.0,
            30.0 / 128.0 * 2.0,
            50.0 / 128.0 * 2.0,
            20.0 / 128.0 * 0.5,
            40.0 / 128.0 * 0.5,
            60.0 / 128.0 * 0.5,
        ];
        assert_eq!(matrix.values, expected);
    }

    #[test]
    fn quant16_dense_applies_row_scales() {
        let mut payload = quant_header(16, 0, &[1.0]);
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&2_u32.to_le_bytes());
        payload.extend_from_slice(&(-16_384_i16).to_le_bytes());
        payload.extend_from_slice(&16_384_i16.to_le_bytes());
        let reader = load_v1("prim4", &payload);
        let matrix = reader.read_float_matrix(0).unwrap();
        assert_eq!(matrix.values, vec![-0.5, 0.5]);
    }

    #[test]
    fn quant_header_truncated_is_rejected() {
        let reader = load_v1("prim4", &[8, 0, 0]);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI quantization header is truncated."
        );
    }

    #[test]
    fn quant_bad_bits_are_rejected() {
        let reader = load_v1("prim4", &quant_header(7, 0, &[1.0]));
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI quantized matrix supports only verified signed 8/16-bit weights."
        );
    }

    #[test]
    fn quant_residual_is_rejected() {
        let reader = load_v1("prim4", &quant_header(8, 1, &[1.0]));
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI residual quantization is not yet implemented."
        );
    }

    #[test]
    fn quant_scale_array_truncated_is_rejected() {
        let mut header = quant_header(8, 0, &[1.0]);
        header.truncate(header.len() - 2);
        let reader = load_v1("prim4", &header);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI quantization scale array is truncated."
        );
    }

    #[test]
    fn quant_non_finite_scale_is_rejected() {
        let reader = load_v1("prim4", &quant_header(8, 0, &[f32::NAN]));
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI float tensor contains a non-finite value."
        );
    }

    #[test]
    fn quant_scale_count_mismatch_is_rejected() {
        let mut payload = quant_header(8, 0, &[1.0]);
        payload.extend_from_slice(&2_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&[1, 2]);
        let reader = load_v1("prim4", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI quantized matrix must have one scale per output row."
        );
    }

    #[test]
    fn quant_dims_mismatch_is_rejected() {
        let mut payload = quant_header(8, 0, &[1.0, 1.0]);
        payload.extend_from_slice(&2_u32.to_le_bytes());
        payload.extend_from_slice(&2_u32.to_le_bytes());
        payload.extend_from_slice(&[1, 2, 3]);
        let reader = load_v1("prim4", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI quantized matrix dimensions do not match its payload."
        );
    }

    fn sparse_payload(
        rows: u32,
        columns: u32,
        block_rows: u32,
        block_columns: u32,
        indices: &[u8],
        pointers: &[u32],
        coefficients: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&columns.to_le_bytes());
        out.extend_from_slice(&block_rows.to_le_bytes());
        out.extend_from_slice(&block_columns.to_le_bytes());
        let block_count = (indices.len() / 2) as u32;
        out.extend_from_slice(&block_count.to_le_bytes());
        out.extend_from_slice(indices);
        for pointer in pointers {
            out.extend_from_slice(&pointer.to_le_bytes());
        }
        out.extend_from_slice(coefficients);
        out
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn sparse_unquantized_places_blocks() {
        // 4x4 with 2x2 blocks: block 0 covers rows 0-1 / cols 2-3,
        // block 1 covers rows 2-3 / cols 0-1. Coefficients within a block
        // run column-major (column * block_rows + row).
        let payload = sparse_payload(
            4,
            4,
            2,
            2,
            &[1, 0, 0, 0],
            &[0, 1, 2],
            &floats(&[1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]),
        );
        let reader = load_v1("prim3", &payload);
        let matrix = reader.read_float_matrix(0).unwrap();
        assert_eq!(
            matrix.values,
            vec![
                0.0, 0.0, 1.0, 2.0, //
                0.0, 0.0, 3.0, 4.0, //
                5.0, 6.0, 0.0, 0.0, //
                7.0, 8.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn sparse_quant_applies_row_scales() {
        // 2x2 with 1x1 blocks on the diagonal, 8-bit ints, scales [2.0, 4.0].
        let mut payload = quant_header(8, 0, &[2.0, 4.0]);
        payload.extend(sparse_payload(2, 2, 1, 1, &[0, 0, 1, 0], &[0, 1, 2], &[64, 192]));
        let reader = load_v1("prim5", &payload);
        let matrix = reader.read_float_matrix(0).unwrap();
        // 192 as i8 is -64: -64 / 128 * 4.0 = -2.0.
        assert_eq!(matrix.values, vec![1.0, 0.0, 0.0, -2.0]);
    }

    #[test]
    fn sparse_partial_blocks_are_rejected() {
        let payload = sparse_payload(4, 4, 3, 2, &[], &[0], &[]);
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse matrix has unsupported partial or empty block dimensions."
        );
    }

    #[test]
    fn sparse_empty_blocks_are_rejected() {
        let payload = sparse_payload(4, 4, 0, 2, &[], &[0], &[]);
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse matrix has unsupported partial or empty block dimensions."
        );
    }

    #[test]
    fn sparse_huge_block_count_is_rejected() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&1000_u32.to_le_bytes());
        payload.extend_from_slice(&1000_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&1_u32.to_le_bytes());
        payload.extend_from_slice(&200_000_000_u32.to_le_bytes());
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse coefficient array exceeds the memory limit."
        );
    }

    #[test]
    fn sparse_size_mismatch_is_rejected() {
        let mut payload = sparse_payload(2, 2, 1, 1, &[0, 0], &[0, 1, 1], &floats(&[1.0]));
        payload.push(0x00);
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse matrix arrays do not match its payload."
        );
    }

    #[test]
    fn sparse_pointers_must_span_all_blocks() {
        let payload = sparse_payload(2, 2, 1, 1, &[0, 0], &[1, 1, 1], &floats(&[1.0]));
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse row pointers do not span all blocks."
        );
    }

    #[test]
    fn sparse_pointers_out_of_order_are_rejected() {
        // Pointers [0, 2, 1]: first row claims blocks 0..2 of 1 total, and the
        // span check (last == block count) passes, so the range check fires.
        let payload = sparse_payload(2, 2, 1, 1, &[0, 0], &[0, 2, 1], &floats(&[1.0]));
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse row pointers are out of order or range."
        );
    }

    #[test]
    fn sparse_column_out_of_range_is_rejected() {
        let payload = sparse_payload(2, 2, 1, 1, &[5, 0], &[0, 1, 1], &floats(&[1.0]));
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse block column is outside the matrix."
        );
    }

    #[test]
    fn sparse_non_finite_accumulation_is_rejected() {
        let payload = sparse_payload(
            2,
            2,
            1,
            1,
            &[0, 0],
            &[0, 1, 1],
            &floats(&[f32::INFINITY]),
        );
        let reader = load_v1("prim3", &payload);
        assert_eq!(
            reader.read_float_matrix(0).unwrap_err().to_string(),
            "DNNI sparse matrix has a non-finite accumulated weight."
        );
    }
}
