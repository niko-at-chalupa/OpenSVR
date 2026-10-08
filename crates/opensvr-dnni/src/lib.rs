//! Reader for Synthesizer V DNNI neural network files.
//!
//! Ports OpenSV's `src/synthesis/DnniReader.{h,cpp}`
//! (namespace `sv::synthesis`). Layout, limits and error strings match the C++
//! version so golden tests can compare messages.
//!
//! File layout (all integers little-endian):
//! - 8-byte header: magic `0x7fca00ff` (u32), format version `1` or `2` (u32).
//! - Node tree: each node is a 20-byte header (marker `0x7fca40ff`/`0x7fca41ff`,
//!   8-byte type id, child count, payload size), then payload bytes, then
//!   children sequentially.
//! - v1 type ids are 8-byte ASCII tags; v2 type ids are 64-bit FNV-1a hashes
//!   (child count obfuscated with an XOR mask). Payloads decode to `prim0`..`prim5`
//!   matrices (dense, row-scaled 8/16-bit quantized, BCSR sparse) and `prim1`
//!   float vectors.

mod inference;
mod reader;

pub use inference::{
    DenseMatrix, DnniInference, DnniTensor, Layer, Operation, Tensor,
};
pub use reader::{DnniError, DnniMatrix, DnniNode, DnniReader};
