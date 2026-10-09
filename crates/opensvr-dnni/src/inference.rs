//! Blocked-SIMD neural network inference for Synthesizer V DNNI models.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/DnniInference.{h,cpp}`
//! (namespace `sv::synthesis`): model loading plus every kernel over blocked
//! `[f32; BLOCK]` weights (LLVM auto-vectorized, no nightly SIMD), and the
//! incremental dirty-range [`DnniCache`]. The per-element summation orders
//! match the C++ engine exactly (block-outer, column-inner; kernel-outer for
//! convolutions), so numerics agree with the C++ goldens to `1e-5` relative.
//!
//! Error strings reproduce the C++ `juce::Result` failure text verbatim
//! (including the `DNNI inference at 0x<offset>: <reason>` prefix) so golden
//! tests can compare messages. Two C++ failure modes have no Rust equivalent
//! and are therefore not reproduced: `std::bad_alloc` paths (`Insufficient
//! memory ...`), since Rust aborts on allocation failure instead of throwing.

use opensvr_core::CancelToken;

use crate::reader::{DnniError, DnniReader};

/// Maximum number of elements in one tensor or the parameter store (64 Mi).
const MAXIMUM_ELEMENTS: usize = 64 * 1024 * 1024;

/// How often cancellable loops poll the [`CancelToken`] (every 32 steps).
const CANCEL_POLL_STEPS: usize = 32;

fn invalid(message: impl Into<String>) -> DnniError {
    DnniError::Invalid(message.into())
}

fn node_error(offset: usize, reason: &str) -> DnniError {
    invalid(format!("DNNI inference at 0x{offset:x}: {reason}"))
}

fn cancelled() -> DnniError {
    invalid("DNNI inference cancelled.")
}

/// 2D tensor for neural network inputs, activations and outputs.
///
/// Frame-major layout: `values[frame * channels + channel]`, matching
/// `DnniTensor` in `DnniInference.h`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tensor {
    /// Time steps / sequence length (rows).
    pub frames: usize,
    /// Feature dimension / channels per time step (columns).
    pub channels: usize,
    /// Flattened frame-major array of floats.
    pub values: Vec<f32>,
}

/// Alias keeping the C++ `DnniTensor` name usable next to [`Tensor`].
pub type DnniTensor = Tensor;

/// Operational statistics for a single inference pass, mirroring the C++
/// `DnniRunStatistics` (`DnniInference.h:35-40`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DnniRunStatistics {
    /// Output frames evaluated from scratch this run.
    pub computed_frames: usize,
    /// Frames reused verbatim from the cache.
    pub reused_frames: usize,
    /// Receptive-field halo frames processed but not emitted.
    pub context_frames: usize,
}

/// Maximum elements storable in one phrase cache (64 Mi floats / 64 MiB).
const MAXIMUM_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Per-phrase cache for incremental dirty-range re-synthesis, mirroring the C++
/// `DnniInference::Cache` (`DnniInference.h:72-85`).
///
/// Invariants (kept from the C++ port):
/// - One instance belongs to one phrase on the synthesis worker thread.
/// - Never shared across threads.
#[derive(Debug, Clone, Default)]
pub struct DnniCache {
    /// Last seen input tensor (full sequence).
    pub input: Tensor,
    /// Optional conditioning tensor snapshotted alongside the input.
    pub condition: Tensor,
    /// Cached output for the most recent successful run.
    pub output: Tensor,
    /// Monotonic tag that distinguishes loaded model generations; a `load`
    /// bump invalidates cached tensors from a prior model.
    pub model_identity: u64,
    /// Whether a condition tensor was present for the cached run.
    pub has_condition: bool,
}

impl DnniCache {
    /// Resets the cache to its default state, matching `Cache::clear()`.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Byte size of retained float capacities, matching `Cache::getBytes()`.
    ///
    /// Returns `capacity * 4` summed across input, condition and output (using
    /// `Vec::capacity`, not `len`), exactly as the C++ port does.
    pub fn get_bytes(&self) -> usize {
        (self.input.values.capacity()
            + self.condition.values.capacity()
            + self.output.values.capacity())
            * size_of::<f32>()
    }
}

/// True when two tensors describe the same shape (frames, channels, count).
///
/// Mirrors the C++ `sameShape` (`DnniInference.cpp:54-57`).
fn same_shape(first: &Tensor, second: &Tensor) -> bool {
    first.frames == second.frames
        && first.channels == second.channels
        && first.values.len() == second.values.len()
}

/// Inclusive `[first, end)` frame range over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameRange {
    first: usize,
    end: usize,
}

/// Expands a frame range by a receptive-field radius, clamped to `[0, frames]`.
///
/// Mirrors the C++ `expandRange` (`DnniInference.cpp:72-75`).
fn expand_range(range: FrameRange, radius: usize, frames: usize) -> FrameRange {
    FrameRange {
        first: range.first.saturating_sub(radius),
        end: (range.end + radius).min(frames),
    }
}

/// Bitwise float-slice equality, the safe-Rust equivalent of `memcmp`.
///
/// Compares `to_bits()` lane by lane so sign-of-zero and NaN payload
/// differences count as changed, exactly as the C++ byte comparison does.
/// (`f32: PartialEq` would treat `-0.0 == 0.0` and `NaN != NaN`.)
fn frames_bit_equal(first: &[f32], second: &[f32]) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Finds output-frame ranges that differ between two tensors' cached state.
///
/// Input is compared first (per frame via byte equality); condition is only
/// compared when input is unchanged, matching the C++
/// `findChangedOutputRanges` (`DnniInference.cpp:84-110`). Adjacent ranges are
/// pre-merged when their radius-expanded spans touch.
fn find_changed_output_ranges(
    previous: &Tensor,
    current: &Tensor,
    previous_condition: Option<&Tensor>,
    current_condition: Option<&Tensor>,
    radius: usize,
) -> Vec<FrameRange> {
    let mut ranges: Vec<FrameRange> = Vec::new();
    let condition_channels = current_condition.map_or(0, |condition| condition.channels);
    for frame in 0..current.frames {
        let input_offset = frame * current.channels;
        // Byte equality like the C++ `memcmp`: compare bit patterns so that
        // e.g. `-0.0` vs `0.0` counts as changed, exactly as the C++ does.
        // (`f32: PartialEq` would treat them as equal.)
        let input_changed = !frames_bit_equal(
            &previous.values[input_offset..input_offset + current.channels],
            &current.values[input_offset..input_offset + current.channels],
        );
        let condition_changed = !input_changed
            && current_condition.is_some()
            && previous_condition.is_some_and(|previous| {
                let offset = frame * condition_channels;
                !frames_bit_equal(
                    &previous.values[offset..offset + condition_channels],
                    &current_condition.unwrap().values[offset..offset + condition_channels],
                )
            });
        if !input_changed && !condition_changed {
            continue;
        }
        let range = expand_range(
            FrameRange { first: frame, end: frame + 1 },
            radius,
            current.frames,
        );
        if ranges.is_empty() || ranges.last().unwrap().end < range.first {
            ranges.push(range);
        } else {
            ranges.last_mut().unwrap().end = ranges.last().unwrap().end.max(range.end);
        }
    }
    ranges
}

/// Copies frames `[first, end)` out of a tensor, matching `sliceFrames`
/// (`DnniInference.cpp:112-117`).
fn slice_frames(input: &Tensor, first: usize, end: usize) -> Tensor {
    let begin = first * input.channels;
    let finish = end * input.channels;
    Tensor {
        frames: end - first,
        channels: input.channels,
        values: input.values[begin..finish].to_vec(),
    }
}

/// Maximum elements per SIMD weight block (4 vectors × 4 lanes).
///
/// Matches the C++ `channelsPerBlock` at the SSE width; on AVX2 the C++ engine
/// doubles it to 32, but the Rust port fixes it at 16 per `AGENTS.md` and leans
/// on LLVM auto-vectorization, so numerics stay portable and bit-stable.
pub const BLOCK: usize = 16;

/// Number of `[f32; BLOCK]` vectors in one blocked weight block.
///
/// Mirrors the C++ `vectorsPerBlock = 4`; on SSE each vector holds 4 floats,
/// giving `4 * 4 = 16` output channels per block.
const VECTORS_PER_BLOCK: usize = 4;

/// Blocked SIMD weight matrix, transposing row-major storage into per-column
/// 16-lane blocks.
///
/// This is the Rust counterpart of the C++ `Matrix` with `WeightBlock =
/// std::array<Vector, vectorsPerBlock>`. Weights for each row-block are laid out
/// so that `values[block * columns + column]` contains the 16 consecutive
/// output-channel weights for that column, zero-padded on the tail block. This
/// lets `multiply_matrix` load one `[f32; BLOCK]` per column and feed it to a
/// 16-way unrolled accumulator that LLVM can vectorize without nightly SIMD.
///
/// `rows` is the output dimension and `columns` the input dimension; the packed
/// block count is `ceil(rows / BLOCK)`, with `values.len() == block_count *
/// columns`. The original logical rows (`rows`) are retained because the
/// convolution and GRU runners address outputs by channel index and must stop at
/// the real row count, not the padded block boundary.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DenseMatrix {
    /// Output dimension (number of rows before SIMD padding).
    pub rows: usize,
    /// Input dimension (number of columns).
    pub columns: usize,
    /// One `[f32; BLOCK]` per (block, column), row-major within each block with
    /// zero-padding on the tail. Length is `block_count * columns`.
    pub values: Vec<[f32; BLOCK]>,
}

/// Layer operation, mirroring `DnniInference::Operation` in `DnniInference.h`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Operation {
    /// Sequential container of layers (`modm0`).
    #[default]
    Sequence,
    /// Fully-connected affine transformation (`modl0`).
    Dense,
    /// 1D convolution with kernel size, stride, padding, dilation (`modl1`).
    Convolution,
    /// Gated activation unit (`_gnc1v0`).
    GatedConvolution,
    /// WaveNet residual stack with skip connections (`_ncwnv0`).
    ResidualConvolution,
    /// Unidirectional gated recurrent unit (`modl3`).
    Gru,
    /// Bidirectional GRU, forward + backward concatenated (`modl6`).
    BidirectionalGru,
    /// Rectified linear unit (`moda0`).
    Relu,
    /// Hyperbolic tangent (`moda1`).
    Tanh,
    /// Logistic sigmoid (`moda2`).
    Sigmoid,
    /// Leaky ReLU (`moda3`).
    LeakyRelu,
    /// Exponential linear unit (`moda4`).
    Elu,
    /// Pass-through identity (`moda5`).
    Identity,
    /// Sigmoid linear unit / Swish (`moda7`).
    Silu,
}

/// One loaded network layer.
///
/// Plain-data mirror of `DnniInference::Layer`: `matrices` are blocked SIMD
/// weights (one entry per convolution kernel tap, six entries for GRU
/// projections), `bias` holds the dense/conv bias or the six concatenated GRU
/// biases `[b_ir, b_iz, b_in, b_hr, b_hz, b_hn]`.
///
/// The default convolution geometry (`stride = 1`, `dilation = 1`) matches the
/// C++ `Layer` member initializers; dense layers rely on it to address input
/// frame `frame`.
#[derive(Debug, Clone)]
pub struct Layer {
    pub operation: Operation,
    /// Byte offset of the source DNNI node, used for error strings.
    pub source_offset: usize,
    pub children: Vec<Layer>,
    pub matrices: Vec<DenseMatrix>,
    pub bias: Vec<f32>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub input_channels: usize,
    pub gate_channels: usize,
    pub condition_channels: usize,
    pub stage_count: usize,
    pub alpha: f32,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            operation: Operation::Sequence,
            source_offset: 0,
            children: Vec::new(),
            matrices: Vec::new(),
            bias: Vec::new(),
            stride: 1,
            padding: 0,
            dilation: 1,
            input_channels: 0,
            gate_channels: 0,
            condition_channels: 0,
            stage_count: 0,
            alpha: 0.0,
        }
    }
}

/// Scalar DNNI inference engine: load once, run many times.
#[derive(Debug, Default)]
pub struct DnniInference {
    root: Option<Layer>,
    context_radius: Option<usize>,
    /// Monotonic tag identifying the loaded model generation; a `load` bumps
    /// it so stale [`DnniCache`] entries from a prior model are rejected.
    model_identity: u64,
    loaded: bool,
}

impl DnniInference {
    /// Creates an unloaded engine; [`run`](Self::run) fails until [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true after a successful [`load`](Self::load).
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Finite receptive-field context radius, if the loaded network has one.
    ///
    /// `None` means either nothing is loaded or the network contains a layer
    /// (GRU, strided/non-symmetric convolution, ...) whose output depends on
    /// more than a fixed frame window. Mirrors `findContextRadius`.
    pub fn context_radius(&self) -> Option<usize> {
        if self.loaded {
            self.context_radius
        } else {
            None
        }
    }

    /// Loads and initializes weights and structure from a parsed DNNI model.
    ///
    /// `root_node` is the node index of the network root (usually 0). On
    /// failure the engine keeps its previous state.
    pub fn load(&mut self, reader: &DnniReader, root_node: usize) -> Result<(), DnniError> {
        let mut replacement = Layer::default();
        let mut parameter_count = 0_usize;
        load_layer(reader, root_node, &mut replacement, &mut parameter_count)?;
        self.context_radius = find_context_radius(&replacement);
        self.root = Some(replacement);
        self.model_identity = next_model_identity();
        self.loaded = true;
        Ok(())
    }

    /// Executes neural network forward inference.
    ///
    /// `condition` is the optional conditioning tensor (e.g. speaker
    /// embeddings). `output` is only assigned on success. Cancellation is
    /// polled at entry, at every nested layer entry, and every 32 steps of
    /// the GRU and dense/conv loops.
    pub fn run(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        condition: Option<&Tensor>,
        cancel: &CancelToken,
    ) -> Result<(), DnniError> {
        self.run_with_cache(input, output, condition, None, cancel, None)
    }

    /// Executes inference with optional incremental-cache reuse and statistics.
    ///
    /// Mirrors the C++ `run(..., Cache*, ..., DnniRunStatistics*)` overload
    /// (`DnniInference.cpp:698-853`): when `cache` is armed and the network has
    /// a finite receptive field, changed output-frame ranges are detected by
    /// byte-comparing snapshots, expanded by the context radius, and only the
    /// affected windows are re-synthesized — the rest are copied verbatim.
    ///
    /// The plain [`run`](Self::run) delegates here with no cache, so the two
    /// paths share validation and cancel checks.
    pub fn run_with_cache(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        condition: Option<&Tensor>,
        cache: Option<&mut DnniCache>,
        cancel: &CancelToken,
        statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<(), DnniError> {
        let Some(root) = self.root.as_ref().filter(|_| self.loaded) else {
            return Err(invalid("No DNNI inference model has been loaded."));
        };
        if !valid_tensor(input) {
            return Err(invalid(
                "DNNI input must contain finite frame-major values with valid dimensions within 64 Mi elements.",
            ));
        }
        if condition.is_some_and(|tensor| !valid_tensor(tensor)) {
            return Err(invalid(
                "DNNI condition must contain finite frame-major values with valid dimensions within 64 Mi elements.",
            ));
        }
        if cancel.is_cancelled() {
            return Err(cancelled());
        }

        // Cache-gating mirrors C++ lines 718–719:
        // canCache ⟺ cache armed ∧ finite radius ∧ (no condition OR
        // condition frames match input frames).
        let radius = match self.context_radius {
            Some(radius) => radius,
            None => 0,
        };
        let can_cache = cache.is_some()
            && self.context_radius.is_some()
            && condition.is_none_or(|tensor| tensor.frames == input.frames);
        let has_cached_output = can_cache
            && cache.as_deref().is_some_and(|cache| {
                cache.model_identity == self.model_identity
                    && same_shape(&cache.input, input)
                    && cache.has_condition == condition.is_some()
                    && condition.is_none_or(|tensor| same_shape(&cache.condition, tensor))
            });

        let mut output_ranges: Vec<FrameRange> = Vec::new();
        if has_cached_output {
            let cache_ref = cache.as_deref().expect("has_cached_output implies armed cache");
            output_ranges = find_changed_output_ranges(
                &cache_ref.input,
                input,
                if cache_ref.has_condition { Some(&cache_ref.condition) } else { None },
                condition,
                radius,
            );
            if output_ranges.is_empty() {
                // Identical input/condition: reuse the entire cached output.
                *output = cache_ref.output.clone();
                if let Some(statistics) = statistics {
                    statistics.reused_frames += output.frames;
                }
                return Ok(());
            }
        }

        let mut replacement = Tensor::default();
        let mut computed_frames = 0_usize;
        let mut reused_frames = 0_usize;
        let mut context_frames = 0_usize;

        if has_cached_output {
            let cache_ref = cache.as_deref().expect("has_cached_output implies armed cache");
            replacement = cache_ref.output.clone();
            let mut updated_frames = 0_usize;
            let mut first_range = 0_usize;
            while first_range < output_ranges.len() {
                if cancel.is_cancelled() {
                    return Err(cancelled());
                }
                // Expand then re-merge overlapping radius windows, matching
                // the C++ window-merge loop (lines 752–761).
                let mut window = expand_range(output_ranges[first_range], radius, input.frames);
                let mut end_range = first_range + 1;
                while end_range < output_ranges.len() {
                    let next_window = expand_range(output_ranges[end_range], radius, input.frames);
                    if next_window.first > window.end {
                        break;
                    }
                    window.end = window.end.max(next_window.end);
                    end_range += 1;
                }
                let local_output = if window.first == 0 && window.end == input.frames {
                    let mut local = Tensor::default();
                    run_layer(root, input, &mut local, condition, cancel)?;
                    local
                } else {
                    let local_input = slice_frames(input, window.first, window.end);
                    let local_condition = match condition {
                        Some(condition) => Some(slice_frames(condition, window.first, window.end)),
                        None => None,
                    };
                    let mut local = Tensor::default();
                    run_layer(
                        root,
                        &local_input,
                        &mut local,
                        local_condition.as_ref(),
                        cancel,
                    )?;
                    local
                };
                if local_output.frames != window.end - window.first
                    || local_output.channels != replacement.channels
                {
                    return Err(invalid(
                        "DNNI cached inference produced an unexpected output shape.",
                    ));
                }
                // Paste only the changed centre ranges; halos are computed
                // but not written back.
                for index in first_range..end_range {
                    let range = output_ranges[index];
                    let source_start = (range.first - window.first) * local_output.channels;
                    let destination_start = range.first * replacement.channels;
                    let span = (range.end - range.first) * replacement.channels;
                    replacement.values[destination_start..destination_start + span]
                        .copy_from_slice(&local_output.values[source_start..source_start + span]);
                    updated_frames += range.end - range.first;
                }
                computed_frames += local_output.frames;
                first_range = end_range;
            }
            reused_frames = input.frames - updated_frames;
            context_frames = computed_frames - updated_frames;
        } else {
            run_layer(root, input, &mut replacement, condition, cancel)?;
            computed_frames = replacement.frames;
        }
        if cancel.is_cancelled() {
            return Err(cancelled());
        }

        // Snapshot the run iff it is cacheable and within the byte cap.
        if let Some(cache) = cache {
            let condition_elements = condition.map_or(0, |tensor| tensor.values.len());
            let snapshot_elements = input.values.len() + condition_elements + replacement.values.len();
            if can_cache
                && replacement.frames == input.frames
                && snapshot_elements <= MAXIMUM_CACHE_BYTES / size_of::<f32>()
            {
                let updated = DnniCache {
                    input: input.clone(),
                    condition: condition.map_or(Tensor::default(), |t| t.clone()),
                    output: replacement.clone(),
                    model_identity: self.model_identity,
                    has_condition: condition.is_some(),
                };
                if updated.get_bytes() <= MAXIMUM_CACHE_BYTES {
                    *cache = updated;
                } else {
                    cache.clear();
                }
            } else {
                cache.clear();
            }
        }

        *output = replacement;
        if let Some(statistics) = statistics {
            statistics.computed_frames += computed_frames;
            statistics.reused_frames += reused_frames;
            statistics.context_frames += context_frames;
        }
        Ok(())
    }
}

/// Monotonic model-identity source, mirroring the C++ `nextModelIdentity`
/// atomic (`DnniInference.cpp:22`).
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_model_identity() -> u64 {
    // Relaxed ordering matches the C++ `memory_order_relaxed`; identities only
    // need to be unique, not sequenced relative to other operations. The
    // counter starts at 1 like the C++ `nextModelIdentity{1}` so that no live
    // model ever carries the default cache's `0` tag.
    NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

fn valid_shape(frames: usize, channels: usize) -> bool {
    channels > 0 && channels <= MAXIMUM_ELEMENTS && frames <= MAXIMUM_ELEMENTS / channels
}

fn finite_values(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn valid_tensor(tensor: &Tensor) -> bool {
    valid_shape(tensor.frames, tensor.channels)
        && tensor.values.len() == tensor.frames * tensor.channels
        && finite_values(&tensor.values)
}

/// Numerically stable logistic sigmoid: `1 / (1 + exp(-x))`.
///
/// Matches `DnniInference.cpp:48-52`: `exp(-|x|)` reflected for negatives to
/// avoid overflow. The C++ code calls `std::exp` on a `float`, which promotes
/// to the `double` overload, so the exponential is evaluated in `f64` here
/// too and narrowed only at the end.
fn sigmoid(value: f32) -> f32 {
    let exponential = f64::from(-value.abs()).exp();
    if value >= 0.0 {
        (1.0 / (1.0 + exponential)) as f32
    } else {
        (exponential / (1.0 + exponential)) as f32
    }
}

fn read_i32(payload: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ])
}

/// Blocked SIMD matrix-vector product: `output[block, lane] = sum over columns in order`.
///
/// This is the SIMD equivalent of `multiply_matrix`: the C++ `multiplyMatrix`
/// zeroes four per-block accumulators and sums column-inner, so each output
/// element is the columns summed in increasing order, overwriting (not
/// accumulating into) the target. The packed path keeps that per-element order
/// (ROADMAP principle 4) — only the vector width of the accumulation changes.
///
/// For the test suite there is a `#[cfg(test)]` scalar fallback that reads
/// the same blocked layout but accumulates each lane through a plain `f32`
/// sum in column order, proving the packed path does not reorder
/// floating-point additions relative to the scalar reference.
///
/// Deliberately outlined (`#[inline(never)]`): in the final linked release
/// binary the standalone body lowers the 16-lane accumulator to four packed
/// `mulps`/`addps` pairs with a broadcasted input lane (verified with
/// `objdump -d` on the release test harness; note `cargo rustc -- --emit=asm`
/// is misleading here because workspace ThinLTO defers codegen to link time).
/// A body inlined into the GRU/dense caller control flow scalarizes instead.
/// Call overhead is negligible next to `columns × 16` fused multiplies per
/// invocation.
#[inline(never)]
pub fn multiply_matrix(matrix: &DenseMatrix, input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= matrix.rows);
    let block_count = matrix.values.len() / matrix.columns.max(1);
    for (block_index, first_channel) in (0..block_count).map(|b| (b, b * BLOCK)) {
        let row_count = (matrix.rows - first_channel).min(BLOCK);
        // One flat 16-lane accumulator per block, column-inner: LLVM lowers
        // this to packed `mulps`/`addps` with a broadcasted input lane (16-wide
        // on AVX2, paired SSE ops otherwise) without changing the per-element
        // column order. `VECTORS_PER_BLOCK` documents the C++ `WeightBlock`
        // correspondence: lanes `[4k, 4k+4)` are the k-th SIMD register.
        debug_assert!(BLOCK == 4 * VECTORS_PER_BLOCK);
        let mut acc = [0.0_f32; BLOCK];
        // Bounds checks are hoisted to one per block; the column loop runs
        // over iterators with no per-lane checks, which is what lets LLVM
        // SLP-vectorize the 16-lane body into packed arithmetic.
        let block_weights = &matrix.values[block_index * matrix.columns..][..matrix.columns];
        for (value, weights) in input.iter().zip(block_weights.iter()) {
            let value = *value;
            for lane in 0..BLOCK {
                acc[lane] += weights[lane] * value;
            }
        }
        output[first_channel..first_channel + row_count].copy_from_slice(&acc[..row_count]);
    }
}

/// Scalar reference for [`multiply_matrix`], gated to test builds.
///
/// Reads the same blocked layout but accumulates each lane through a plain
/// `f32` sum in column order, proving the packed path does not reorder
/// floating-point additions relative to a scalar reference. Unit tests assert
/// exact (bitwise) agreement between the two.
#[cfg(test)]
fn multiply_matrix_scalar(matrix: &DenseMatrix, input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= matrix.rows);
    let block_count = matrix.values.len() / matrix.columns.max(1);
    for block in 0..block_count {
        let first_channel = block * BLOCK;
        let row_count = (matrix.rows - first_channel).min(BLOCK);
        for row in 0..row_count {
            let mut sum = 0.0_f32;
            for column in 0..matrix.columns {
                sum += matrix.values[block * matrix.columns + column][row] * input[column];
            }
            output[first_channel + row] = sum;
        }
    }
}

fn load_matrix(
    reader: &DnniReader,
    node_index: usize,
    parameter_count: &mut usize,
) -> Result<DenseMatrix, DnniError> {
    let offset = reader
        .nodes()
        .get(node_index)
        .map_or(0, |node| node.offset);
    let decoded = match reader.read_float_matrix(node_index) {
        Ok(matrix) => matrix,
        Err(error) => return Err(error),
    };
    let rows = decoded.rows as usize;
    let columns = decoded.columns as usize;
    if rows == 0
        || !valid_shape(rows, columns)
        || decoded.values.len() != rows * columns
    {
        return Err(node_error(offset, "invalid matrix shape."));
    }
    // C++ `loadMatrix`: transpose row-major → per-column [f32; BLOCK] blocks,
    // zero-padded on the tail. The padded element count (blockCount * columns)
    // is what the C++ loader accounts against `maximumElements`.
    let block_count = (rows + BLOCK - 1) / BLOCK;
    let padded_rows = block_count * BLOCK;
    if columns > (MAXIMUM_ELEMENTS - *parameter_count) / padded_rows {
        return Err(node_error(
            offset,
            "packed model exceeds the parameter memory limit.",
        ));
    }
    *parameter_count += padded_rows * columns;
    let mut values = vec![[0.0; BLOCK]; block_count * columns];
    for (block, first_row) in (0..block_count).map(|b| (b, b * BLOCK)) {
        let row_count = (rows - first_row).min(BLOCK);
        for column in 0..columns {
            let slot = &mut values[block * columns + column];
            for row in 0..row_count {
                slot[row] = decoded.values[(first_row + row) * columns + column];
            }
        }
    }
    Ok(DenseMatrix {
        rows,
        columns,
        values,
    })
}

#[allow(clippy::too_many_lines)]
fn load_layer(
    reader: &DnniReader,
    node_index: usize,
    layer: &mut Layer,
    parameter_count: &mut usize,
) -> Result<(), DnniError> {
    let nodes = reader.nodes();
    let Some(node) = nodes.get(node_index) else {
        return Err(invalid(
            "DNNI inference root or child node is out of range.",
        ));
    };
    let payload = reader.payload(node_index);
    layer.source_offset = node.offset;

    if node.name == "modm0" {
        if !payload.is_empty() {
            return Err(node_error(node.offset, "sequence payload must be empty."));
        }
        layer.operation = Operation::Sequence;
        for child_index in node.children.clone() {
            let mut child = Layer::default();
            load_layer(reader, child_index, &mut child, parameter_count)?;
            // Nested sequences have no state; flattening prevents tensor
            // copies accumulating with depth.
            if child.operation == Operation::Sequence {
                layer.children.extend(child.children);
            } else {
                layer.children.push(child);
            }
        }
        return Ok(());
    }

    if node.name == "_ncwnv0" {
        if payload.len() != 12 {
            return Err(node_error(
                node.offset,
                "residual convolution v0 requires three int32 channel dimensions.",
            ));
        }
        let input_channels = read_i32(payload, 0);
        let hidden_channels = read_i32(payload, 4);
        let condition_channels = read_i32(payload, 8);
        if input_channels <= 0 || hidden_channels <= 0 || condition_channels < 0 {
            return Err(node_error(
                node.offset,
                "residual convolution channel dimensions are invalid.",
            ));
        }
        layer.operation = Operation::ResidualConvolution;
        layer.input_channels = input_channels as usize;
        layer.gate_channels = hidden_channels as usize;
        layer.condition_channels = condition_channels as usize;
        let child_count = if condition_channels > 0 { 4 } else { 3 };
        if node.children.len() != child_count {
            return Err(node_error(
                node.offset,
                "residual convolution requires three parameter groups and an optional condition projection.",
            ));
        }
        for group in 0..3 {
            let group_index = node.children[group];
            let valid_group = nodes.get(group_index).is_some_and(|group_node| {
                group_node.name == "cmpg1" && reader.payload(group_index).is_empty()
            });
            if !valid_group {
                return Err(node_error(
                    node.offset,
                    "residual convolution requires empty-payload cmpg1 parameter groups.",
                ));
            }
            let count = nodes[group_index].children.len();
            if group == 0 {
                layer.stage_count = count;
            }
            if count == 0 || count != layer.stage_count {
                return Err(node_error(
                    nodes[group_index].offset,
                    "residual convolution parameter groups must have matching non-zero lengths.",
                ));
            }
        }
        // Each stage stores the gate, its skip projection, and the input's
        // residual projection.
        layer
            .children
            .resize_with(layer.stage_count * 3 + usize::from(condition_channels > 0), Layer::default);
        for stage in 0..layer.stage_count {
            let stage_input = if stage == 0 {
                layer.input_channels
            } else {
                layer.gate_channels
            };
            for group in 0..3 {
                let child_index = nodes[node.children[group]].children[stage];
                let expected = if group == 0 { "_gnc1v0" } else { "modl1" };
                if nodes.get(child_index).is_none_or(|child| child.name != expected) {
                    return Err(node_error(
                        node.offset,
                        "residual convolution contains an unsupported stage operator.",
                    ));
                }
                let child = &mut layer.children[stage * 3 + group];
                load_layer(reader, child_index, child, parameter_count)?;
                if group == 0 {
                    if child.input_channels != stage_input
                        || child.gate_channels != layer.gate_channels
                        || child.condition_channels != layer.condition_channels
                    {
                        return Err(node_error(
                            nodes[child_index].offset,
                            "residual gate dimensions do not match its parent network.",
                        ));
                    }
                    for convolution in &child.children {
                        if !preserves_frames(convolution) {
                            return Err(node_error(
                                convolution.source_offset,
                                "whole-sequence residual gates require stride-one convolutions that preserve frame count.",
                            ));
                        }
                    }
                } else {
                    let dimensions_match = child
                        .matrices
                        .first()
                        .is_some_and(|matrix| {
                            let expected_input = if group == 1 {
                                layer.gate_channels
                            } else {
                                stage_input
                            };
                            matrix.columns == expected_input
                                && matrix.rows == layer.gate_channels
                        });
                    if !dimensions_match || !preserves_frames(child) {
                        return Err(node_error(
                            nodes[child_index].offset,
                            "residual or skip projection dimensions and timing do not match the network.",
                        ));
                    }
                }
            }
        }
        if condition_channels > 0 {
            let child_index = node.children[3];
            if nodes
                .get(child_index)
                .is_none_or(|child| child.name != "modl1")
            {
                return Err(node_error(
                    node.offset,
                    "residual network condition projection must be Conv1D.",
                ));
            }
            let projection = layer.children.last_mut().expect("condition slot");
            load_layer(reader, child_index, projection, parameter_count)?;
            let dimensions_match = projection.matrices.first().is_some_and(|matrix| {
                matrix.columns == layer.condition_channels
                    && matrix.rows == layer.gate_channels
            });
            if !dimensions_match
                || projection.matrices.len() != 1
                || !preserves_frames(projection)
            {
                return Err(node_error(
                    nodes[child_index].offset,
                    "residual network condition projection must preserve frames and map condition channels to hidden channels.",
                ));
            }
        }
        return Ok(());
    }

    if node.name == "_gnc1v0" {
        if payload.len() != 12 {
            return Err(node_error(
                node.offset,
                "gated Conv1D v0 requires three int32 channel dimensions.",
            ));
        }
        let input_channels = read_i32(payload, 0);
        let gate_channels = read_i32(payload, 4);
        let condition_channels = read_i32(payload, 8);
        if input_channels <= 0
            || gate_channels <= 0
            || condition_channels < 0
            || gate_channels as usize > MAXIMUM_ELEMENTS / 2
        {
            return Err(node_error(
                node.offset,
                "gated Conv1D channel dimensions are invalid.",
            ));
        }
        layer.operation = Operation::GatedConvolution;
        layer.input_channels = input_channels as usize;
        layer.gate_channels = gate_channels as usize;
        layer.condition_channels = condition_channels as usize;
        let child_count = if condition_channels > 0 { 2 } else { 1 };
        if node.children.len() != child_count {
            return Err(node_error(
                node.offset,
                "gated Conv1D v0 requires an input convolution and an optional condition convolution.",
            ));
        }
        layer.children.resize_with(child_count, Layer::default);
        for (index, child_index) in node.children.clone().into_iter().enumerate() {
            if nodes
                .get(child_index)
                .is_none_or(|child| child.name != "modl1")
            {
                return Err(node_error(
                    node.offset,
                    "gated Conv1D v0 children must be Conv1D operators.",
                ));
            }
            let convolution = &mut layer.children[index];
            load_layer(reader, child_index, convolution, parameter_count)?;
            let expected_input = if index == 0 {
                layer.input_channels
            } else {
                layer.condition_channels
            };
            let dimensions_match = convolution.matrices.first().is_some_and(|matrix| {
                matrix.columns == expected_input && matrix.rows == 2 * layer.gate_channels
            });
            if !dimensions_match {
                return Err(node_error(
                    nodes[child_index].offset,
                    "gated convolution matrices do not match the declared channel dimensions.",
                ));
            }
            if index == 1 && convolution.matrices.len() != 1 {
                return Err(node_error(
                    nodes[child_index].offset,
                    "gated Conv1D v0 condition convolution must have a one-frame kernel.",
                ));
            }
        }
        return Ok(());
    }

    if node.name == "modl6" {
        if !payload.is_empty() || node.children.len() != 2 {
            return Err(node_error(
                node.offset,
                "bidirectional GRU requires two GRU children and an empty payload.",
            ));
        }
        layer.operation = Operation::BidirectionalGru;
        layer.children.resize_with(2, Layer::default);
        for (direction, child_index) in node.children.clone().into_iter().enumerate() {
            if nodes
                .get(child_index)
                .is_none_or(|child| child.name != "modl3")
            {
                return Err(node_error(
                    node.offset,
                    "bidirectional GRU children must be modl3 operators.",
                ));
            }
            load_layer(
                reader,
                child_index,
                &mut layer.children[direction],
                parameter_count,
            )?;
        }
        layer.input_channels = layer.children[0].input_channels;
        if layer.input_channels != layer.children[1].input_channels
            || layer.children[0].gate_channels > MAXIMUM_ELEMENTS - layer.children[1].gate_channels
        {
            return Err(node_error(
                node.offset,
                "bidirectional GRU directions have incompatible input channels or excessive output channels.",
            ));
        }
        layer.gate_channels =
            layer.children[0].gate_channels + layer.children[1].gate_channels;
        return Ok(());
    }

    if node.name == "modl3" {
        if !payload.is_empty() || node.children.len() != 12 {
            return Err(node_error(
                node.offset,
                "GRU requires six matrix/bias pairs and an empty payload.",
            ));
        }
        layer.operation = Operation::Gru;
        layer.matrices.reserve(6);
        for projection in 0..6 {
            let matrix_index = node.children[2 * projection];
            let bias_index = node.children[2 * projection + 1];
            let valid_pair = nodes.get(matrix_index).is_some_and(|matrix_node| {
                matrix_node.children.is_empty()
            }) && nodes.get(bias_index).is_some_and(|bias_node| {
                bias_node.children.is_empty() && bias_node.name == "prim1"
            });
            if !valid_pair {
                return Err(node_error(
                    node.offset,
                    "GRU parameters must alternate leaf matrices and float bias vectors.",
                ));
            }
            let matrix = load_matrix(reader, matrix_index, parameter_count)?;
            if projection == 0 {
                layer.input_channels = matrix.columns;
                layer.gate_channels = matrix.rows;
            }
            let expected_columns = if projection < 3 {
                layer.input_channels
            } else {
                layer.gate_channels
            };
            if matrix.rows != layer.gate_channels || matrix.columns != expected_columns {
                return Err(node_error(
                    nodes[matrix_index].offset,
                    "GRU input and recurrent matrices have incompatible dimensions.",
                ));
            }
            layer.matrices.push(matrix);
            let bias = match reader.read_float_vector(bias_index) {
                Ok(bias) => bias,
                Err(error) => return Err(error),
            };
            if bias.len() != layer.gate_channels
                || bias.len() > MAXIMUM_ELEMENTS - *parameter_count
            {
                return Err(node_error(
                    nodes[bias_index].offset,
                    "GRU bias dimensions are invalid or exceed the parameter memory limit.",
                ));
            }
            *parameter_count += bias.len();
            layer.bias.extend_from_slice(&bias);
        }
        return Ok(());
    }

    if node.name == "modl0" || node.name == "modl1" {
        let mut kernel_count = 1_usize;
        if node.name == "modl0" {
            layer.operation = Operation::Dense;
            if !payload.is_empty() || node.children.is_empty() || node.children.len() > 2 {
                return Err(node_error(
                    node.offset,
                    "dense requires an empty payload, one matrix and an optional bias.",
                ));
            }
        } else {
            layer.operation = Operation::Convolution;
            if payload.len() != 20 {
                return Err(node_error(
                    node.offset,
                    "Conv1D requires five int32 parameters.",
                ));
            }
            let kernel = read_i32(payload, 0);
            let stride = read_i32(payload, 4);
            let padding = read_i32(payload, 8);
            let dilation = read_i32(payload, 12);
            let groups = read_i32(payload, 16);
            if kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0 || groups != 1 {
                return Err(node_error(
                    node.offset,
                    "Conv1D requires positive kernel, stride and dilation, non-negative padding, and groups=1.",
                ));
            }
            kernel_count = kernel as usize;
            layer.stride = stride as usize;
            layer.padding = padding as usize;
            layer.dilation = dilation as usize;
            if node.children.len() < kernel_count
                || node.children.len() > kernel_count + 1
            {
                return Err(node_error(
                    node.offset,
                    "Conv1D child count does not match its kernel count and optional bias.",
                ));
            }
        }

        let mut has_bias = false;
        for (index, child_index) in node.children.clone().into_iter().enumerate() {
            let Some(child) = nodes.get(child_index) else {
                return Err(node_error(node.offset, "parameter node is out of range."));
            };
            if !child.children.is_empty() {
                return Err(node_error(
                    child.offset,
                    "tensor parameters cannot have children.",
                ));
            }
            if child.name == "prim1" {
                if has_bias || (layer.operation == Operation::Dense && index != 1) {
                    return Err(node_error(
                        child.offset,
                        "unexpected or duplicate bias vector.",
                    ));
                }
                match reader.read_float_vector(child_index) {
                    Ok(bias) => layer.bias = bias,
                    Err(error) => return Err(error),
                }
                has_bias = true;
                if layer.bias.len() > MAXIMUM_ELEMENTS - *parameter_count {
                    return Err(node_error(
                        child.offset,
                        "decoded model exceeds the parameter memory limit.",
                    ));
                }
                *parameter_count += layer.bias.len();
            } else {
                let packed = load_matrix(reader, child_index, parameter_count)?;
                layer.matrices.push(packed);
            }
        }
        if layer.matrices.len() != kernel_count {
            return Err(node_error(
                node.offset,
                "matrix count does not match the operator.",
            ));
        }
        let (first_rows, first_columns) = {
            let first = &layer.matrices[0];
            (first.rows, first.columns)
        };
        for matrix in &layer.matrices {
            if matrix.rows != first_rows || matrix.columns != first_columns {
                return Err(node_error(
                    node.offset,
                    "Conv1D kernel matrices must have equal shapes.",
                ));
            }
        }
        if has_bias && layer.bias.len() != first_rows {
            return Err(node_error(
                node.offset,
                "bias length does not match output channels.",
            ));
        }
        return Ok(());
    }

    layer.operation = match node.name.as_str() {
        "moda0" => Operation::Relu,
        "moda1" => Operation::Tanh,
        "moda2" => Operation::Sigmoid,
        "moda3" => Operation::LeakyRelu,
        "moda4" => Operation::Elu,
        "moda5" => Operation::Identity,
        "moda7" => Operation::Silu,
        _ => {
            // The reader renders unknown v2 hashes as `0x...` names already;
            // an empty name falls back to the raw id, as in the C++ loader.
            let rendered = if node.name.is_empty() {
                format!("0x{:x}", node.type_id)
            } else {
                node.name.clone()
            };
            return Err(node_error(
                node.offset,
                &format!("unsupported operator {rendered}."),
            ));
        }
    };
    if !node.children.is_empty() {
        return Err(node_error(
            node.offset,
            "activation cannot have child parameters.",
        ));
    }
    if layer.operation == Operation::LeakyRelu || layer.operation == Operation::Elu {
        if payload.len() != size_of::<f32>() {
            return Err(node_error(
                node.offset,
                "activation requires one float32 alpha parameter.",
            ));
        }
        layer.alpha = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        if !layer.alpha.is_finite() {
            return Err(node_error(
                node.offset,
                "activation alpha must be finite.",
            ));
        }
    } else if !payload.is_empty() {
        return Err(node_error(
            node.offset,
            "activation payload must be empty.",
        ));
    }
    Ok(())
}

/// True when a convolution maps frame count to itself.
fn preserves_frames(layer: &Layer) -> bool {
    layer.operation == Operation::Convolution
        && layer.stride == 1
        && 2 * layer.padding as u64
            == (layer.matrices.len().saturating_sub(1)) as u64 * layer.dilation as u64
}

fn find_context_radius(layer: &Layer) -> Option<usize> {
    if layer.operation == Operation::Gru || layer.operation == Operation::BidirectionalGru {
        // Recurrent state depends on the whole preceding sequence (both
        // directions for modl6), so fixed-radius cache windows are not exact.
        return None;
    }
    if layer.operation == Operation::Convolution {
        let extent = (layer.matrices.len().saturating_sub(1)) as u64 * layer.dilation as u64;
        if layer.stride != 1
            || extent != 2 * layer.padding as u64
            || layer.padding > MAXIMUM_ELEMENTS
        {
            return None;
        }
        return Some(layer.padding);
    }
    if layer.operation == Operation::Sequence {
        let mut radius = 0_usize;
        for child in &layer.children {
            let child_radius = find_context_radius(child)?;
            radius = radius.checked_add(child_radius)?;
            if radius > MAXIMUM_ELEMENTS {
                return None;
            }
        }
        return Some(radius);
    }
    if layer.operation == Operation::GatedConvolution {
        let mut radius = 0_usize;
        for child in &layer.children {
            radius = radius.max(find_context_radius(child)?);
        }
        return Some(radius);
    }
    if layer.operation == Operation::ResidualConvolution {
        let mut radius = 0_usize;
        for stage in 0..layer.stage_count {
            let gate = find_context_radius(&layer.children[stage * 3])?;
            let skip = find_context_radius(&layer.children[stage * 3 + 1])?;
            let residual = find_context_radius(&layer.children[stage * 3 + 2])?;
            if skip > MAXIMUM_ELEMENTS - gate {
                return None;
            }
            // The next stage receives gate + residual. Every skip also
            // contributes to the final output, so include the longest branch
            // at each stage.
            let stage_radius = (gate + skip).max(residual);
            radius = radius.checked_add(stage_radius)?;
            if radius > MAXIMUM_ELEMENTS {
                return None;
            }
        }
        if layer.condition_channels > 0 {
            let condition_radius =
                find_context_radius(layer.children.last().expect("condition slot"))?;
            radius = radius.max(condition_radius);
        }
        return Some(radius);
    }
    Some(0)
}

fn run_gru(
    layer: &Layer,
    input: &Tensor,
    output: &mut Tensor,
    reverse: bool,
    cancel: &CancelToken,
) -> Result<(), DnniError> {
    if input.channels != layer.input_channels
        || !valid_shape(input.frames, layer.gate_channels)
    {
        return Err(node_error(
            layer.source_offset,
            "GRU input channels do not match the model or output exceeds the memory limit.",
        ));
    }
    let hidden_channels = layer.gate_channels;
    output.frames = input.frames;
    output.channels = hidden_channels;
    output.values.resize(output.frames * output.channels, 0.0);
    let mut hidden = vec![0.0_f32; hidden_channels];
    let mut projections = vec![0.0_f32; 6 * hidden_channels];
    for step in 0..input.frames {
        if step % CANCEL_POLL_STEPS == 0 && cancel.is_cancelled() {
            return Err(cancelled());
        }
        let frame = if reverse {
            input.frames - 1 - step
        } else {
            step
        };
        let input_frame = &input.values[frame * input.channels..(frame + 1) * input.channels];
        for projection in 0..6 {
            let source = if projection < 3 { input_frame } else { &hidden };
            let target =
                &mut projections[projection * hidden_channels..(projection + 1) * hidden_channels];
            multiply_matrix(&layer.matrices[projection], source, target);
        }
        if !finite_values(&projections) {
            return Err(node_error(
                layer.source_offset,
                "GRU projection produced a non-finite value.",
            ));
        }
        for channel in 0..hidden_channels {
            let reset_input = layer.bias[channel]
                + layer.bias[3 * hidden_channels + channel]
                + projections[channel]
                + projections[3 * hidden_channels + channel];
            let update_input = layer.bias[hidden_channels + channel]
                + layer.bias[4 * hidden_channels + channel]
                + projections[hidden_channels + channel]
                + projections[4 * hidden_channels + channel];
            let candidate_recurrent =
                layer.bias[5 * hidden_channels + channel] + projections[5 * hidden_channels + channel];
            let candidate_input = sigmoid(reset_input) * candidate_recurrent
                + projections[2 * hidden_channels + channel]
                + layer.bias[2 * hidden_channels + channel];
            if !reset_input.is_finite() || !update_input.is_finite() || !candidate_input.is_finite()
            {
                return Err(node_error(
                    layer.source_offset,
                    "GRU gate produced a non-finite value.",
                ));
            }
            let update = sigmoid(update_input);
            hidden[channel] =
                update * hidden[channel] + (1.0 - update) * candidate_input.tanh();
        }
        output.values[frame * hidden_channels..(frame + 1) * hidden_channels]
            .copy_from_slice(&hidden);
    }
    Ok(())
}

fn run_layer(
    layer: &Layer,
    input: &Tensor,
    output: &mut Tensor,
    condition: Option<&Tensor>,
    cancel: &CancelToken,
) -> Result<(), DnniError> {
    if cancel.is_cancelled() {
        return Err(cancelled());
    }
    if layer.operation == Operation::Sequence {
        let mut current = input.clone();
        for child in &layer.children {
            let mut next = Tensor::default();
            run_layer(child, &current, &mut next, condition, cancel)?;
            current = next;
        }
        *output = current;
        return Ok(());
    }

    if layer.operation == Operation::Gru {
        return run_gru(layer, input, output, false, cancel);
    }

    if layer.operation == Operation::BidirectionalGru {
        if !valid_shape(input.frames, layer.gate_channels) {
            return Err(node_error(
                layer.source_offset,
                "bidirectional GRU output exceeds the memory limit.",
            ));
        }
        let mut forward = Tensor::default();
        let mut backward = Tensor::default();
        run_gru(&layer.children[0], input, &mut forward, false, cancel)?;
        run_gru(&layer.children[1], input, &mut backward, true, cancel)?;
        output.frames = input.frames;
        output.channels = layer.gate_channels;
        output.values.resize(output.frames * output.channels, 0.0);
        // modl6 buffers through end-of-sequence and concatenates forward and
        // backward channels at their original frame positions, in that order.
        let forward_channels = layer.children[0].gate_channels;
        for frame in 0..output.frames {
            let destination =
                &mut output.values[frame * output.channels..(frame + 1) * output.channels];
            destination[..forward_channels].copy_from_slice(
                &forward.values[frame * forward.channels..(frame + 1) * forward.channels],
            );
            destination[forward_channels..].copy_from_slice(
                &backward.values[frame * backward.channels..(frame + 1) * backward.channels],
            );
        }
        return Ok(());
    }

    if layer.operation == Operation::ResidualConvolution {
        if input.channels != layer.input_channels {
            return Err(node_error(
                layer.source_offset,
                "residual convolution input channel count does not match the network.",
            ));
        }
        if layer.condition_channels > 0 {
            let valid_condition = condition.is_some_and(|tensor| {
                tensor.channels == layer.condition_channels && tensor.frames == input.frames
            });
            if !valid_condition {
                return Err(node_error(
                    layer.source_offset,
                    "residual convolution requires a condition tensor with matching frames and declared channels.",
                ));
            }
            let projection = layer.children.last().expect("condition slot");
            run_layer(
                projection,
                condition.expect("checked"),
                output,
                None,
                cancel,
            )?;
        } else {
            output.frames = input.frames;
            output.channels = layer.gate_channels;
            if !valid_shape(output.frames, output.channels) {
                return Err(node_error(
                    layer.source_offset,
                    "residual convolution output exceeds the memory limit.",
                ));
            }
            assign_zero(&mut output.values, output.frames * output.channels);
        }
        let mut current = input.clone();
        for stage in 0..layer.stage_count {
            let mut residual = Tensor::default();
            run_layer(&layer.children[stage * 3 + 2], &current, &mut residual, None, cancel)?;
            let mut gated = Tensor::default();
            run_layer(&layer.children[stage * 3], &current, &mut gated, condition, cancel)?;
            let mut skip = Tensor::default();
            run_layer(&layer.children[stage * 3 + 1], &gated, &mut skip, None, cancel)?;
            if residual.frames != output.frames
                || gated.frames != output.frames
                || skip.frames != output.frames
                || residual.channels != output.channels
                || gated.channels != output.channels
                || skip.channels != output.channels
            {
                return Err(node_error(
                    layer.source_offset,
                    "residual convolution branches produced incompatible shapes.",
                ));
            }
            // Skip connection contributes to overall network output.
            for (slot, delta) in output.values.iter_mut().zip(&skip.values) {
                *slot += *delta;
            }
            // Residual connection feeds into next dilated stage.
            for (slot, delta) in gated.values.iter_mut().zip(&residual.values) {
                *slot += *delta;
            }
            if !finite_values(&output.values) || !finite_values(&gated.values) {
                return Err(node_error(
                    layer.source_offset,
                    "residual convolution accumulation produced a non-finite value.",
                ));
            }
            current = gated;
        }
        return Ok(());
    }

    if layer.operation == Operation::GatedConvolution {
        if input.channels != layer.input_channels {
            return Err(node_error(
                layer.source_offset,
                "gated Conv1D input channels do not match the declared input.",
            ));
        }
        if layer.condition_channels > 0
            && condition.is_none_or(|tensor| tensor.channels != layer.condition_channels)
        {
            return Err(node_error(
                layer.source_offset,
                "gated Conv1D requires a condition tensor with the declared channel count.",
            ));
        }
        let mut gate_input = Tensor::default();
        run_layer(&layer.children[0], input, &mut gate_input, None, cancel)?;
        if gate_input.channels != 2 * layer.gate_channels {
            return Err(node_error(
                layer.source_offset,
                "gated Conv1D input projection must produce two gate-channel groups.",
            ));
        }
        if layer.condition_channels > 0 {
            let mut projected = Tensor::default();
            run_layer(
                &layer.children[1],
                condition.expect("checked"),
                &mut projected,
                None,
                cancel,
            )?;
            if projected.frames != gate_input.frames || projected.channels != gate_input.channels
            {
                return Err(node_error(
                    layer.source_offset,
                    "input and condition convolutions must produce matching frames and channels.",
                ));
            }
            for (slot, delta) in gate_input.values.iter_mut().zip(&projected.values) {
                *slot += *delta;
            }
            if !finite_values(&gate_input.values) {
                return Err(node_error(
                    layer.source_offset,
                    "gated Conv1D conditioning produced a non-finite value.",
                ));
            }
        }
        output.frames = gate_input.frames;
        output.channels = layer.gate_channels;
        output.values.resize(output.frames * output.channels, 0.0);
        for frame in 0..output.frames {
            let gate_offset = frame * gate_input.channels;
            for channel in 0..output.channels {
                output.values[frame * output.channels + channel] =
                    gate_input.values[gate_offset + channel].tanh()
                        * sigmoid(gate_input.values[gate_offset + layer.gate_channels + channel]);
            }
        }
        return Ok(());
    }

    if layer.operation == Operation::Dense || layer.operation == Operation::Convolution {
        let first = &layer.matrices[0];
        if input.channels != first.columns {
            return Err(node_error(
                layer.source_offset,
                "input channels do not match the matrix columns.",
            ));
        }
        output.channels = first.rows;
        output.frames = input.frames;
        if layer.operation == Operation::Convolution {
            // Parameters are int32 and tensor dimensions are bounded, so these
            // fit u64.
            let receptive_field =
                (layer.matrices.len() as u64 - 1) * layer.dilation as u64 + 1;
            let padded_frames = input.frames as u64 + 2 * layer.padding as u64;
            output.frames = if padded_frames < receptive_field {
                0
            } else {
                ((padded_frames - receptive_field) / layer.stride as u64 + 1) as usize
            };
        }
        if !valid_shape(output.frames, output.channels) {
            return Err(node_error(
                layer.source_offset,
                "output tensor exceeds the memory limit.",
            ));
        }
        assign_zero(&mut output.values, output.frames * output.channels);
        for frame in 0..output.frames {
            if frame % CANCEL_POLL_STEPS == 0 && cancel.is_cancelled() {
                return Err(cancelled());
            }
            // Per-output summation order is kernel-outer, column-inner: each
            // output element accumulates tap 0..K, and within a tap the input
            // channels in increasing order. Each kernel writes a disjoint
            // [first, first+rows) slice of the output, so a temporary buffer
            // per tap is accumulated and added back in order, mirroring the C++
            // 2-frame unrolling (which keeps independent accumulators per frame).
            for (kernel, matrix) in layer.matrices.iter().enumerate() {
                let tap = frame as i64 * layer.stride as i64 + kernel as i64 * layer.dilation as i64
                    - layer.padding as i64;
                if tap < 0 || tap as usize >= input.frames {
                    // Out-of-range taps contribute zero (padding), never
                    // clamping.
                    continue;
                }
                let source = &input.values[tap as usize * input.channels..];
                let mut product = vec![0.0_f32; matrix.rows];
                multiply_matrix(matrix, source, &mut product);
                let destination = &mut output.values[frame * output.channels..];
                for channel_out in 0..matrix.rows {
                    destination[channel_out] += product[channel_out];
                }
            }
            if !layer.bias.is_empty() {
                for channel_out in 0..output.channels {
                    output.values[frame * output.channels + channel_out] +=
                        layer.bias[channel_out];
                }
            }
        }
    } else {
        *output = input.clone();
        for value in &mut output.values {
            match layer.operation {
                Operation::Relu => *value = value.max(0.0),
                Operation::Tanh => *value = value.tanh(),
                Operation::Sigmoid | Operation::Silu => {
                    // Equivalent to 1/(1+exp(-x)), avoiding overflow for
                    // negative inputs.
                    let activation = sigmoid(*value);
                    if layer.operation == Operation::Silu {
                        *value *= activation;
                    } else {
                        *value = activation;
                    }
                }
                Operation::LeakyRelu => {
                    *value *= if *value > 0.0 { 1.0 } else { layer.alpha };
                }
                Operation::Elu => {
                    // Preserve the reference formula for every finite alpha,
                    // including negative values. A non-negative alpha leaves
                    // positive inputs unchanged without evaluating exp(x).
                    if layer.alpha < 0.0 || *value < 0.0 {
                        *value = value.max(0.0) + (layer.alpha * value.exp_m1()).min(0.0);
                    }
                }
                _ => {}
            }
        }
    }
    if !finite_values(&output.values) {
        return Err(node_error(
            layer.source_offset,
            "operator produced a non-finite output.",
        ));
    }
    Ok(())
}

/// Mirrors the C++ `output.values.assign(n, 0.0f)` idiom used above.
fn assign_zero(values: &mut Vec<f32>, count: usize) {
    values.clear();
    values.resize(count, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_token() -> CancelToken {
        CancelToken::new()
    }

    fn cancelled_token() -> CancelToken {
        let token = CancelToken::new();
        token.cancel();
        token
    }

    fn tag(name: &str) -> [u8; 8] {
        let mut tag = [0_u8; 8];
        tag[..name.len()].copy_from_slice(name.as_bytes());
        tag
    }

    fn node_bytes(name: &str, payload: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        out.extend_from_slice(&tag(name));
        out.extend_from_slice(&(children.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        for child in children {
            out.extend_from_slice(child);
        }
        out
    }

    fn leaf(name: &str, payload: &[u8]) -> Vec<u8> {
        node_bytes(name, payload, &[])
    }

    fn file_bytes(root: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        out.extend_from_slice(&1_u32.to_le_bytes());
        out.extend_from_slice(root);
        out
    }

    fn load_reader(root: &[u8]) -> DnniReader {
        DnniReader::from_bytes(file_bytes(root)).expect("synthetic blob must parse")
    }

    fn load_engine(root: &[u8]) -> DnniInference {
        let reader = load_reader(root);
        let mut engine = DnniInference::new();
        engine.load(&reader, 0).expect("synthetic model must load");
        engine
    }

    fn load_error(root: &[u8]) -> String {
        let reader = load_reader(root);
        let mut engine = DnniInference::new();
        engine.load(&reader, 0).unwrap_err().to_string()
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

    fn vector_payload(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn conv_payload(kernel: i32, stride: i32, padding: i32, dilation: i32) -> Vec<u8> {
        let mut out = Vec::new();
        for parameter in [kernel, stride, padding, dilation, 1] {
            out.extend_from_slice(&parameter.to_le_bytes());
        }
        out
    }

    fn dims3(first: i32, second: i32, third: i32) -> Vec<u8> {
        let mut out = Vec::new();
        for parameter in [first, second, third] {
            out.extend_from_slice(&parameter.to_le_bytes());
        }
        out
    }

    fn alpha_payload(alpha: f32) -> Vec<u8> {
        alpha.to_le_bytes().to_vec()
    }

    fn dense_model(rows: u32, columns: u32, values: &[f32], bias: Option<&[f32]>) -> Vec<u8> {
        let matrix = leaf("prim0", &dense_payload(rows, columns, values));
        let mut children = vec![matrix];
        if let Some(bias) = bias {
            children.push(leaf("prim1", &vector_payload(bias)));
        }
        node_bytes("modl0", &[], &children)
    }

    fn conv_model(
        kernel: i32,
        stride: i32,
        padding: i32,
        dilation: i32,
        taps: &[Vec<f32>],
        out_channels: u32,
        in_channels: u32,
        bias: Option<&[f32]>,
    ) -> Vec<u8> {
        let mut children = Vec::new();
        for tap in taps {
            children.push(leaf(
                "prim0",
                &dense_payload(out_channels, in_channels, tap),
            ));
        }
        if let Some(bias) = bias {
            children.push(leaf("prim1", &vector_payload(bias)));
        }
        node_bytes(
            "modl1",
            &conv_payload(kernel, stride, padding, dilation),
            &children,
        )
    }

    /// Minimal single-stage residual network (no condition): gate with
    /// zero weights, skip gain `skip`, residual gain `residual`.
    fn residual_model(skip: f32, residual: f32) -> Vec<u8> {
        let gate_conv = node_bytes(
            "modl1",
            &conv_payload(1, 1, 0, 1),
            &[leaf("prim0", &dense_payload(2, 1, &[0.0, 0.0]))],
        );
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[gate_conv]);
        let skip_node = node_bytes(
            "modl1",
            &conv_payload(1, 1, 0, 1),
            &[leaf("prim0", &dense_payload(1, 1, &[skip]))],
        );
        let residual_node = node_bytes(
            "modl1",
            &conv_payload(1, 1, 0, 1),
            &[leaf("prim0", &dense_payload(1, 1, &[residual]))],
        );
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip_node]),
            node_bytes("cmpg1", &[], &[residual_node]),
        ];
        node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups)
    }

    /// Six 1x1 GRU projections with the given weights and zero biases.
    fn gru_model(weights: &[f32; 6]) -> Vec<u8> {
        let mut children = Vec::new();
        for weight in weights {
            children.push(leaf("prim0", &dense_payload(1, 1, &[*weight])));
            children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        node_bytes("modl3", &[], &children)
    }

    fn run_ok(engine: &DnniInference, frames: usize, channels: usize, values: &[f32]) -> Tensor {
        run_cond_ok(engine, frames, channels, values, None)
    }

    fn run_cond_ok(
        engine: &DnniInference,
        frames: usize,
        channels: usize,
        values: &[f32],
        condition: Option<Tensor>,
    ) -> Tensor {
        let input = Tensor {
            frames,
            channels,
            values: values.to_vec(),
        };
        let mut output = Tensor::default();
        engine
            .run(&input, &mut output, condition.as_ref(), &test_token())
            .expect("run must succeed");
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?} vs {expected:?}");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            let allowed = tolerance * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= allowed,
                "index {index}: got {got}, want {want}"
            );
        }
    }

    // --- `run`-level errors ---

    #[test]
    fn unloaded_engine_fails() {
        let engine = DnniInference::new();
        assert!(!engine.is_loaded());
        assert_eq!(engine.context_radius(), None);
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert_eq!(
            error.to_string(),
            "No DNNI inference model has been loaded."
        );
        assert!(output.values.is_empty());
    }

    #[test]
    fn bad_root_index_fails() {
        let reader = load_reader(&leaf("moda5", &[]));
        let mut engine = DnniInference::new();
        let error = engine.load(&reader, 7).unwrap_err().to_string();
        assert_eq!(
            error.to_string(),
            "DNNI inference root or child node is out of range."
        );
        assert!(!engine.is_loaded());
    }

    #[test]
    fn non_finite_input_fails() {
        let engine = load_engine(&leaf("moda5", &[]));
        for values in [vec![f32::INFINITY], vec![f32::NAN]] {
            let input = Tensor {
                frames: 1,
                channels: 1,
                values,
            };
            let mut output = Tensor {
                frames: 9,
                channels: 9,
                values: vec![7.0; 81],
            };
            let error = engine
                .run(&input, &mut output, None, &test_token())
                .unwrap_err().to_string();
            assert_eq!(
                error.to_string(),
                "DNNI input must contain finite frame-major values with valid dimensions within 64 Mi elements."
            );
            // Output is left untouched on failure.
            assert_eq!(output.values, vec![7.0; 81]);
        }
    }

    #[test]
    fn misshaped_input_fails() {
        let engine = load_engine(&leaf("moda5", &[]));
        for input in [
            Tensor {
                frames: 2,
                channels: 1,
                values: vec![1.0],
            },
            Tensor {
                frames: 1,
                channels: 0,
                values: vec![],
            },
        ] {
            let mut output = Tensor::default();
            let error = engine
                .run(&input, &mut output, None, &test_token())
                .unwrap_err().to_string();
            assert_eq!(
                error.to_string(),
                "DNNI input must contain finite frame-major values with valid dimensions within 64 Mi elements."
            );
        }
    }

    #[test]
    fn bad_condition_fails() {
        let engine = load_engine(&leaf("moda5", &[]));
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let condition = Tensor {
            frames: 1,
            channels: 1,
            values: vec![f32::NAN],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, Some(&condition), &test_token())
            .unwrap_err().to_string();
        assert_eq!(
            error.to_string(),
            "DNNI condition must contain finite frame-major values with valid dimensions within 64 Mi elements."
        );
    }

    #[test]
    fn pre_cancelled_token_aborts_run() {
        let engine = load_engine(&leaf("moda5", &[]));
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &cancelled_token())
            .unwrap_err().to_string();
        assert_eq!(error, "DNNI inference cancelled.");
        assert!(output.values.is_empty());
    }

    #[test]
    fn failed_load_keeps_previous_model() {
        let reader = load_reader(&leaf("moda5", &[]));
        let mut engine = DnniInference::new();
        engine.load(&reader, 0).unwrap();
        let bad = load_reader(&leaf("modl4", &[]));
        let error = engine.load(&bad, 0).unwrap_err().to_string();
        assert!(error.ends_with("unsupported operator modl4."));
        assert!(engine.is_loaded());
        let output = run_ok(&engine, 1, 1, &[2.0]);
        assert_eq!(output.values, vec![2.0]);
    }

    // --- Sequences ---

    #[test]
    fn sequence_payload_must_be_empty() {
        let root = node_bytes("modm0", &[1], &[]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: sequence payload must be empty."
        );
    }

    #[test]
    fn nested_sequences_flatten() {
        let dense = dense_model(1, 1, &[2.0], None);
        let inner = node_bytes("modm0", &[], &[dense]);
        let root = node_bytes("modm0", &[], &[inner, leaf("moda5", &[])]);
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), Some(0));
        let output = run_ok(&engine, 1, 1, &[3.0]);
        assert_eq!(output.values, vec![6.0]);
    }

    // --- Dense ---

    #[test]
    fn dense_computes_affine_map() {
        let root = dense_model(2, 3, &[1.0, 2.0, 3.0, 0.5, -1.0, 0.25], Some(&[0.5, -0.5]));
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), Some(0));
        let output = run_ok(&engine, 2, 3, &[1.0, 2.0, 3.0, 0.0, 0.0, 1.0]);
        assert_close(&output.values, &[14.5, -1.25, 3.5, -0.25], 1e-6);
    }

    #[test]
    fn dense_without_bias_sums_plain_products() {
        let root = dense_model(1, 2, &[3.0, 4.0], None);
        let output = run_ok(&load_engine(&root), 1, 2, &[2.0, 1.0]);
        assert_eq!(output.values, vec![10.0]);
    }

    #[test]
    fn dense_requires_matrix_and_optional_bias() {
        let root = node_bytes("modl0", &[], &[]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: dense requires an empty payload, one matrix and an optional bias."
        );
        let root = node_bytes("modl0", &[0], &[leaf("prim0", &dense_payload(1, 1, &[1.0]))]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: dense requires an empty payload, one matrix and an optional bias."
        );
    }

    #[test]
    fn dense_bias_must_be_second_child() {
        // prim1 first, matrix second: the bias is in the wrong slot.
        let root = node_bytes(
            "modl0",
            &[],
            &[
                leaf("prim1", &vector_payload(&[1.0])),
                leaf("prim0", &dense_payload(1, 1, &[1.0])),
            ],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x1c: unexpected or duplicate bias vector."
        );
    }

    #[test]
    fn dense_input_channels_must_match() {
        let engine = load_engine(&dense_model(2, 2, &[1.0, 0.0, 0.0, 1.0], None));
        let input = Tensor {
            frames: 1,
            channels: 3,
            values: vec![1.0, 2.0, 3.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("input channels do not match the matrix columns."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn dense_overflow_reports_non_finite_output() {
        let root = dense_model(1, 1, &[3.0e38], None);
        let engine = load_engine(&root);
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![10.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("operator produced a non-finite output."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn dense_huge_output_reports_memory_limit() {
        // 2M output channels x 40 frames exceeds 64 Mi elements; the weight
        // blob itself is only 8 MiB.
        let rows = 2_000_000_u32;
        let weights = vec![0.0_f32; rows as usize];
        let root = dense_model(rows, 1, &weights, None);
        let engine = load_engine(&root);
        let input = Tensor {
            frames: 40,
            channels: 1,
            values: vec![0.0; 40],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("output tensor exceeds the memory limit."),
            "unexpected: {error}"
        );
    }

    // --- Convolution ---

    #[test]
    fn conv_applies_kernel_taps_with_padding_zeros() {
        let root = conv_model(3, 1, 1, 1, &[vec![1.0], vec![2.0], vec![4.0]], 1, 1, None);
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), Some(1));
        let output = run_ok(&engine, 4, 1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.values, vec![10.0, 17.0, 24.0, 11.0]);
    }

    #[test]
    fn conv_stride_skips_frames() {
        let root = conv_model(2, 2, 0, 1, &[vec![1.0], vec![1.0]], 1, 1, Some(&[1.0]));
        let engine = load_engine(&root);
        // Strided convolutions have no finite context radius.
        assert_eq!(engine.context_radius(), None);
        let output = run_ok(&engine, 5, 1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(output.values, vec![4.0, 8.0]);
    }

    #[test]
    fn conv_dilation_spreads_taps() {
        let root = conv_model(2, 1, 0, 2, &[vec![1.0], vec![1.0]], 1, 1, None);
        let output = run_ok(&load_engine(&root), 4, 1, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output.values, vec![4.0, 6.0]);
    }

    #[test]
    fn conv_receptive_field_larger_than_input_yields_no_frames() {
        let root = conv_model(
            3,
            1,
            0,
            1,
            &[vec![1.0], vec![1.0], vec![1.0]],
            1,
            1,
            None,
        );
        let engine = load_engine(&root);
        let output = run_ok(&engine, 2, 1, &[5.0, 6.0]);
        assert_eq!(output.frames, 0);
        assert_eq!(output.channels, 1);
        assert!(output.values.is_empty());
    }

    #[test]
    fn conv_bad_payload_size_fails() {
        let root = node_bytes(
            "modl1",
            &[0; 8],
            &[leaf("prim0", &dense_payload(1, 1, &[1.0]))],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: Conv1D requires five int32 parameters."
        );
    }

    #[test]
    fn conv_bad_parameters_fail() {
        for parameters in [
            [0_i32, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, -1, 1],
            [1, 1, 0, 0],
        ] {
            let mut payload = Vec::new();
            for parameter in parameters {
                payload.extend_from_slice(&parameter.to_le_bytes());
            }
            payload.extend_from_slice(&1_i32.to_le_bytes());
            let root = node_bytes(
                "modl1",
                &payload,
                &[leaf("prim0", &dense_payload(1, 1, &[1.0]))],
            );
            assert!(
                load_error(&root).ends_with(
                    "Conv1D requires positive kernel, stride and dilation, non-negative padding, and groups=1."
                ),
                "parameters {parameters:?}"
            );
        }
        let mut payload = conv_payload(1, 1, 0, 1);
        payload.truncate(16);
        payload.extend_from_slice(&2_i32.to_le_bytes());
        let root = node_bytes(
            "modl1",
            &payload,
            &[leaf("prim0", &dense_payload(1, 1, &[1.0]))],
        );
        assert!(
            load_error(&root).ends_with(
                "Conv1D requires positive kernel, stride and dilation, non-negative padding, and groups=1."
            ),
            "groups=2"
        );
    }

    #[test]
    fn conv_child_count_must_match_kernel() {
        let root = node_bytes(
            "modl1",
            &conv_payload(2, 1, 0, 1),
            &[leaf("prim0", &dense_payload(1, 1, &[1.0]))],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: Conv1D child count does not match its kernel count and optional bias."
        );
    }

    #[test]
    fn conv_kernel_shapes_must_match() {
        let root = node_bytes(
            "modl1",
            &conv_payload(2, 1, 0, 1),
            &[
                leaf("prim0", &dense_payload(1, 1, &[1.0])),
                leaf("prim0", &dense_payload(1, 2, &[1.0, 2.0])),
            ],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: Conv1D kernel matrices must have equal shapes."
        );
    }

    #[test]
    fn conv_bias_length_must_match_outputs() {
        let root = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, Some(&[1.0, 2.0]));
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: bias length does not match output channels."
        );
    }

    #[test]
    fn matrix_count_must_match_operator() {
        // Two biases and no matrix: no matrix survives loading.
        let root = node_bytes(
            "modl1",
            &conv_payload(1, 1, 0, 1),
            &[
                leaf("prim1", &vector_payload(&[1.0])),
                leaf("prim1", &vector_payload(&[2.0])),
            ],
        );
        let error = load_error(&root);
        assert!(
            error.ends_with("unexpected or duplicate bias vector.")
                || error.ends_with("matrix count does not match the operator."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn parameter_node_out_of_range_fails() {
        // A hand-built tree cannot reference an out-of-range child, so this
        // path is exercised through `load` with a bad root index instead.
        let reader = load_reader(&leaf("moda5", &[]));
        let mut engine = DnniInference::new();
        let error = engine.load(&reader, 99).unwrap_err().to_string();
        assert_eq!(
            error.to_string(),
            "DNNI inference root or child node is out of range."
        );
    }

    #[test]
    fn tensor_parameters_cannot_have_children() {
        let nested = node_bytes(
            "prim0",
            &dense_payload(1, 1, &[1.0]),
            &[leaf("prim1", &vector_payload(&[1.0]))],
        );
        let root = node_bytes("modl0", &[], &[nested]);
        let error = load_error(&root);
        assert!(
            error.ends_with("tensor parameters cannot have children."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn duplicate_bias_fails() {
        // Kernel count 2 with one matrix and two biases: the second bias is
        // a duplicate (child count 3 is within [kernel, kernel + 1]).
        let children = vec![
            leaf("prim0", &dense_payload(1, 1, &[1.0])),
            leaf("prim1", &vector_payload(&[1.0])),
            leaf("prim1", &vector_payload(&[2.0])),
        ];
        let root = node_bytes("modl1", &conv_payload(2, 1, 0, 1), &children);
        let error = load_error(&root);
        assert!(
            error.ends_with("unexpected or duplicate bias vector."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn matrix_loader_rejects_empty_shape() {
        let root = dense_model(0, 3, &[], None);
        let error = load_error(&root);
        assert!(
            error.ends_with("invalid matrix shape."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn matrix_loader_propagates_reader_errors() {
        // A lone prim1 child of a kernel-1 convolution leaves zero matrices.
        let root = node_bytes(
            "modl1",
            &conv_payload(1, 1, 0, 1),
            &[leaf("prim1", &vector_payload(&[1.0]))],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: matrix count does not match the operator."
        );
        // A non-finite weight surfaces the reader message verbatim.
        let root = dense_model(1, 1, &[f32::NAN], None);
        let error = load_error(&root);
        assert!(
            error.ends_with("DNNI float tensor contains a non-finite value."),
            "unexpected: {error}"
        );
    }

    // --- Activations ---

    #[test]
    fn activations_apply_elementwise() {
        let cases: &[(&str, Option<f32>, &[f32], &[f32])] = &[
            ("moda0", None, &[-2.0, 0.0, 3.0], &[0.0, 0.0, 3.0]),
            ("moda1", None, &[0.0, 10.0], &[0.0, 1.0]),
            ("moda2", None, &[0.0], &[0.5]),
            ("moda5", None, &[-1.5, 2.5], &[-1.5, 2.5]),
            ("moda3", Some(0.1), &[-2.0, 3.0], &[-0.2, 3.0]),
            ("moda4", Some(1.0), &[-1.0, 2.0], &[-0.63212055, 2.0]),
            ("moda7", None, &[0.0, 2.0], &[0.0, 1.7615942]),
        ];
        for (name, alpha, input, expected) in cases {
            let payload = alpha.map_or(Vec::new(), alpha_payload);
            let engine = load_engine(&leaf(name, &payload));
            assert_eq!(engine.context_radius(), Some(0), "{name}");
            let output = run_ok(&engine, input.len(), 1, input);
            assert_close(&output.values, expected, 1e-5);
        }
    }

    #[test]
    fn sigmoid_is_stable_for_large_inputs() {
        let engine = load_engine(&leaf("moda2", &[]));
        let output = run_ok(&engine, 2, 1, &[-100.0, 100.0]);
        assert!(output.values[0] < 1e-30, "got {}", output.values[0]);
        assert!((output.values[1] - 1.0).abs() < 1e-6, "got {}", output.values[1]);
    }

    #[test]
    fn elu_keeps_negative_alpha_formula() {
        // alpha < 0 with positive input still evaluates the formula branch.
        let engine = load_engine(&leaf("moda4", &alpha_payload(-0.5)));
        let output = run_ok(&engine, 1, 1, &[1.0]);
        let expected = 1.0_f32.max(0.0) + (-0.5 * 1.0_f32.exp_m1()).min(0.0);
        assert_close(&output.values, &[expected], 1e-6);
    }

    #[test]
    fn unsupported_operators_fail() {
        for name in ["modl4", "moda6", "cmpg1", "prim0"] {
            let error = load_error(&leaf(name, &[]));
            assert!(
                error.ends_with(&format!("unsupported operator {name}.")),
                "unexpected: {error}"
            );
        }
    }

    #[test]
    fn unknown_v2_operator_renders_hex() {
        // Version 2 blob with an unknown type hash: the reader names it
        // `0x...` and the loader reports it verbatim.
        let type_id = 0x1234_5678_9abc_def0_u64;
        let low = type_id as u32;
        let high = (type_id >> 32) as u32;
        let encoded = 0_u32.wrapping_mul(3) ^ low ^ high ^ 0xac5e_7bd5;
        let mut node = Vec::new();
        node.extend_from_slice(&0x7fca_41ff_u32.to_le_bytes());
        node.extend_from_slice(&type_id.to_le_bytes());
        node.extend_from_slice(&encoded.to_le_bytes());
        node.extend_from_slice(&0_u32.to_le_bytes());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&node);
        let reader = DnniReader::from_bytes(bytes).unwrap();
        let mut engine = DnniInference::new();
        let error = engine.load(&reader, 0).unwrap_err().to_string();
        assert_eq!(
            error.to_string(),
            "DNNI inference at 0x8: unsupported operator 0x123456789abcdef0."
        );
    }

    #[test]
    fn activation_with_children_fails() {
        let root = node_bytes("moda0", &[], &[leaf("prim1", &vector_payload(&[1.0]))]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: activation cannot have child parameters."
        );
    }

    #[test]
    fn leaky_relu_needs_alpha() {
        assert_eq!(
            load_error(&leaf("moda3", &[])),
            "DNNI inference at 0x8: activation requires one float32 alpha parameter."
        );
        assert_eq!(
            load_error(&leaf("moda4", &[1, 2])),
            "DNNI inference at 0x8: activation requires one float32 alpha parameter."
        );
    }

    #[test]
    fn non_finite_alpha_fails() {
        assert!(
            load_error(&leaf("moda3", &alpha_payload(f32::INFINITY)))
                .ends_with("activation alpha must be finite.")
        );
    }

    #[test]
    fn plain_activation_with_payload_fails() {
        assert_eq!(
            load_error(&leaf("moda0", &[0])),
            "DNNI inference at 0x8: activation payload must be empty."
        );
    }

    // --- Gated convolution ---

    #[test]
    fn gated_conv_computes_tanh_times_sigmoid() {
        // 2 input channels, 1 gate channel, identity filter/gate weights.
        let conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0, 0.0, 1.0]], 2, 2, None);
        let root = node_bytes("_gnc1v0", &dims3(2, 1, 0), &[conv]);
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), Some(0));
        let output = run_ok(&engine, 1, 2, &[2.0, 3.0]);
        assert_close(&output.values, &[0.91830772], 1e-5);
    }

    #[test]
    fn gated_conv_adds_condition_projection() {
        let input_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 2.0]], 2, 1, None);
        let cond_conv = conv_model(1, 1, 0, 1, &[vec![10.0, 20.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[input_conv, cond_conv]);
        let engine = load_engine(&root);
        let condition = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let output = run_cond_ok(&engine, 1, 1, &[1.0], Some(condition));
        assert_close(&output.values, &[1.0], 1e-5);
    }

    #[test]
    fn gated_conv_bad_payload_fails() {
        assert_eq!(
            load_error(&node_bytes("_gnc1v0", &[0; 8], &[])),
            "DNNI inference at 0x8: gated Conv1D v0 requires three int32 channel dimensions."
        );
    }

    #[test]
    fn gated_conv_bad_dimensions_fail() {
        for dims in [dims3(0, 1, 0), dims3(1, 0, 0), dims3(1, 1, -1)] {
            let error = load_error(&node_bytes("_gnc1v0", &dims, &[]));
            assert!(
                error.ends_with("gated Conv1D channel dimensions are invalid."),
                "unexpected: {error}"
            );
        }
    }

    #[test]
    fn gated_conv_needs_matching_children() {
        let conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: gated Conv1D v0 requires an input convolution and an optional condition convolution."
        );
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[leaf("moda5", &[]), conv.clone()]);
        let error = load_error(&root);
        assert!(
            error.ends_with("gated Conv1D v0 requires an input convolution and an optional condition convolution.")
                || error.ends_with("gated Conv1D v0 children must be Conv1D operators."),
            "unexpected: {error}"
        );
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[conv]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: gated Conv1D v0 requires an input convolution and an optional condition convolution."
        );
    }

    #[test]
    fn gated_conv_children_must_be_conv() {
        let root = node_bytes(
            "_gnc1v0",
            &dims3(1, 1, 0),
            &[leaf("moda0", &[])],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: gated Conv1D v0 children must be Conv1D operators."
        );
    }

    #[test]
    fn gated_conv_matrix_dimensions_must_match() {
        // Declares 1 input channel but the convolution reads 2.
        let conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0, 0.0, 1.0]], 2, 2, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[conv]);
        let error = load_error(&root);
        assert!(
            error.ends_with("gated convolution matrices do not match the declared channel dimensions."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gated_conv_condition_kernel_must_be_single() {
        let input_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let cond_conv = node_bytes(
            "modl1",
            &conv_payload(2, 1, 0, 1),
            &[
                leaf("prim0", &dense_payload(2, 1, &[1.0, 0.0])),
                leaf("prim0", &dense_payload(2, 1, &[0.0, 1.0])),
            ],
        );
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[input_conv, cond_conv]);
        let error = load_error(&root);
        assert!(
            error.ends_with("gated Conv1D v0 condition convolution must have a one-frame kernel."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gated_conv_run_checks_channels() {
        let conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[conv]);
        let engine = load_engine(&root);
        let input = Tensor {
            frames: 1,
            channels: 2,
            values: vec![1.0, 2.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("gated Conv1D input channels do not match the declared input."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gated_conv_run_needs_condition() {
        let input_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let cond_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[input_conv, cond_conv]);
        let engine = load_engine(&root);
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "gated Conv1D requires a condition tensor with the declared channel count."
            ),
            "unexpected: {error}"
        );
        let wrong = Tensor {
            frames: 1,
            channels: 2,
            values: vec![1.0, 2.0],
        };
        let error = engine
            .run(&input, &mut output, Some(&wrong), &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "gated Conv1D requires a condition tensor with the declared channel count."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gated_conv_mismatched_condition_frames_fail() {
        // Condition convolution strides by 2, so its output is shorter.
        let input_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let cond_conv = conv_model(1, 2, 0, 1, &[vec![1.0, 0.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[input_conv, cond_conv]);
        let engine = load_engine(&root);
        let condition = Tensor {
            frames: 2,
            channels: 1,
            values: vec![1.0, 1.0],
        };
        let input = Tensor {
            frames: 2,
            channels: 1,
            values: vec![1.0, 1.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, Some(&condition), &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "input and condition convolutions must produce matching frames and channels."
            ),
            "unexpected: {error}"
        );
    }

    // --- Residual convolution ---

    #[test]
    fn residual_single_stage_routes_skip_and_feedback() {
        // Zero gate, skip gain 2, residual gain 3, input [1, 2]:
        // output (skip sum) is [0, 0]; the residual feeds stage-local state.
        let engine = load_engine(&residual_model(2.0, 3.0));
        assert_eq!(engine.context_radius(), Some(0));
        let output = run_ok(&engine, 2, 1, &[1.0, 2.0]);
        assert_eq!(output.values, vec![0.0, 0.0]);
    }

    #[test]
    fn residual_bad_payload_fails() {
        assert_eq!(
            load_error(&node_bytes("_ncwnv0", &[0; 8], &[])),
            "DNNI inference at 0x8: residual convolution v0 requires three int32 channel dimensions."
        );
    }

    #[test]
    fn residual_bad_dimensions_fail() {
        for dims in [dims3(0, 1, 0), dims3(1, 0, 0), dims3(1, 1, -1)] {
            let error = load_error(&node_bytes("_ncwnv0", &dims, &[]));
            assert!(
                error.ends_with("residual convolution channel dimensions are invalid."),
                "unexpected: {error}"
            );
        }
    }

    #[test]
    fn residual_needs_parameter_groups() {
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &[]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: residual convolution requires three parameter groups and an optional condition projection."
        );
        let groups = [
            node_bytes("cmpg1", &[1], &[]),
            node_bytes("cmpg1", &[], &[]),
            node_bytes("cmpg1", &[], &[]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "residual convolution requires empty-payload cmpg1 parameter groups."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_groups_need_matching_lengths() {
        let gate = node_bytes(
            "_gnc1v0",
            &dims3(1, 1, 0),
            &[conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None)],
        );
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let empty: Vec<Vec<u8>> = vec![];
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &empty),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "residual convolution parameter groups must have matching non-zero lengths."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_stage_operators_are_checked() {
        // Group 0 holds a modl1 where a gate belongs.
        let bad_gate = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[bad_gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with("residual convolution contains an unsupported stage operator."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_gate_dimensions_are_checked() {
        // Gate declares 2 input channels; the parent supplies 1.
        let gate_conv = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0, 0.0, 0.0]], 2, 2, None);
        let gate = node_bytes("_gnc1v0", &dims3(2, 1, 0), &[gate_conv]);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with("residual gate dimensions do not match its parent network."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_gates_must_preserve_frames() {
        // Gate convolution strides by 2.
        let gate_conv = conv_model(1, 2, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[gate_conv]);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "whole-sequence residual gates require stride-one convolutions that preserve frame count."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_projection_dimensions_are_checked() {
        let gate_conv = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 0), &[gate_conv]);
        // Skip maps 1 channel to 2 instead of 1 to 1.
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0, 2.0]], 2, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 0), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "residual or skip projection dimensions and timing do not match the network."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_condition_projection_must_be_conv() {
        let gate_conv = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[gate_conv.clone(), gate_conv]);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
            leaf("moda5", &[]),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 1), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with("residual network condition projection must be Conv1D."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_condition_projection_dimensions_are_checked() {
        // Projection maps 2 channels instead of the declared 1.
        let gate = node_bytes(
            "_gnc1v0",
            &dims3(1, 1, 1),
            &[
                conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None),
                conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None),
            ],
        );
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        // Projection maps 2 channels instead of the declared 1.
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
            conv_model(1, 1, 0, 1, &[vec![1.0, 1.0]], 1, 2, None),
        ];
        let root = node_bytes("_ncwnv0", &dims3(1, 1, 1), &groups);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "residual network condition projection must preserve frames and map condition channels to hidden channels."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_run_checks_input_channels() {
        let engine = load_engine(&residual_model(1.0, 1.0));
        let input = Tensor {
            frames: 1,
            channels: 2,
            values: vec![1.0, 2.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "residual convolution input channel count does not match the network."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_run_needs_matching_condition() {
        let gate_conv = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate_cond = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[gate_conv, gate_cond]);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let projection = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
            projection,
        ];
        let engine = load_engine(&node_bytes("_ncwnv0", &dims3(1, 1, 1), &groups));
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let mut output = Tensor::default();
        // Missing condition.
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "residual convolution requires a condition tensor with matching frames and declared channels."
            ),
            "unexpected: {error}"
        );
        // Wrong channel count.
        let wrong = Tensor {
            frames: 1,
            channels: 2,
            values: vec![1.0, 2.0],
        };
        let error = engine
            .run(&input, &mut output, Some(&wrong), &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "residual convolution requires a condition tensor with matching frames and declared channels."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn residual_run_with_condition_accumulates_projection() {
        let gate_conv = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate_cond = conv_model(1, 1, 0, 1, &[vec![0.0, 0.0]], 2, 1, None);
        let gate = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[gate_conv, gate_cond]);
        let skip = conv_model(1, 1, 0, 1, &[vec![1.0]], 1, 1, None);
        let residual = conv_model(1, 1, 0, 1, &[vec![0.0]], 1, 1, None);
        // Condition projection gain 5: output starts at [5], gate is zero so
        // the skip adds nothing.
        let projection = conv_model(1, 1, 0, 1, &[vec![5.0]], 1, 1, None);
        let groups = [
            node_bytes("cmpg1", &[], &[gate]),
            node_bytes("cmpg1", &[], &[skip]),
            node_bytes("cmpg1", &[], &[residual]),
            projection,
        ];
        let engine = load_engine(&node_bytes("_ncwnv0", &dims3(1, 1, 1), &groups));
        let condition = Tensor {
            frames: 1,
            channels: 1,
            values: vec![1.0],
        };
        let output = run_cond_ok(&engine, 1, 1, &[7.0], Some(condition));
        assert_eq!(output.values, vec![5.0]);
    }

    // --- GRU ---

    #[test]
    fn gru_recurrence_matches_reference_values() {
        // Input projections 1.0, recurrent projections 0.5, zero biases.
        let engine = load_engine(&gru_model(&[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]));
        assert_eq!(engine.context_radius(), None);
        let output = run_ok(&engine, 2, 1, &[1.0, 0.0]);
        assert_close(&output.values, &[0.20482419, 0.13316301], 1e-5);
    }

    #[test]
    fn gru_requires_six_pairs() {
        let root = node_bytes("modl3", &[], &[]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: GRU requires six matrix/bias pairs and an empty payload."
        );
    }

    #[test]
    fn gru_parameters_must_alternate() {
        // Two matrices in a row instead of matrix/bias.
        let mut children = vec![
            leaf("prim0", &dense_payload(1, 1, &[1.0])),
            leaf("prim0", &dense_payload(1, 1, &[1.0])),
        ];
        for _ in 0..5 {
            children.push(leaf("prim0", &dense_payload(1, 1, &[1.0])));
            children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        let root = node_bytes("modl3", &[], &children);
        let error = load_error(&root);
        assert!(
            error.ends_with("GRU parameters must alternate leaf matrices and float bias vectors."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gru_matrix_dimensions_are_checked() {
        // Second input projection reads 2 channels instead of 1.
        let mut children = vec![
            leaf("prim0", &dense_payload(1, 1, &[1.0])),
            leaf("prim1", &vector_payload(&[0.0])),
            leaf("prim0", &dense_payload(1, 2, &[1.0, 1.0])),
            leaf("prim1", &vector_payload(&[0.0])),
        ];
        for _ in 0..4 {
            children.push(leaf("prim0", &dense_payload(1, 1, &[1.0])));
            children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        let root = node_bytes("modl3", &[], &children);
        let error = load_error(&root);
        assert!(
            error.ends_with("GRU input and recurrent matrices have incompatible dimensions."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gru_bias_length_is_checked() {
        let mut children = vec![
            leaf("prim0", &dense_payload(1, 1, &[1.0])),
            leaf("prim1", &vector_payload(&[0.0, 0.0])),
        ];
        for _ in 0..5 {
            children.push(leaf("prim0", &dense_payload(1, 1, &[1.0])));
            children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        let root = node_bytes("modl3", &[], &children);
        let error = load_error(&root);
        assert!(
            error.ends_with("GRU bias dimensions are invalid or exceed the parameter memory limit."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gru_run_checks_input_channels() {
        let engine = load_engine(&gru_model(&[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]));
        let input = Tensor {
            frames: 1,
            channels: 2,
            values: vec![1.0, 2.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with(
                "GRU input channels do not match the model or output exceeds the memory limit."
            ),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gru_huge_projection_reports_non_finite() {
        // 3e38 input projection overflows to infinity.
        let engine = load_engine(&gru_model(&[3.0e38, 1.0, 1.0, 0.5, 0.5, 0.5]));
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![10.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("GRU projection produced a non-finite value."),
            "unexpected: {error}"
        );
    }

    #[test]
    fn gru_overflowing_gate_reports_non_finite() {
        // Zero weights but enormous biases: the projections stay finite while
        // the gate sums overflow.
        let mut children = Vec::new();
        for _ in 0..6 {
            children.push(leaf("prim0", &dense_payload(1, 1, &[0.0])));
            children.push(leaf("prim1", &vector_payload(&[3.0e38])));
        }
        let engine = load_engine(&node_bytes("modl3", &[], &children));
        let input = Tensor {
            frames: 1,
            channels: 1,
            values: vec![0.0],
        };
        let mut output = Tensor::default();
        let error = engine
            .run(&input, &mut output, None, &test_token())
            .unwrap_err().to_string();
        assert!(
            error.ends_with("GRU gate produced a non-finite value."),
            "unexpected: {error}"
        );
    }

    // --- Bidirectional GRU ---

    #[test]
    fn bidirectional_gru_concatenates_directions() {
        let weights = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5];
        let forward = gru_model(&weights);
        let backward = gru_model(&weights);
        let engine = load_engine(&node_bytes("modl6", &[], &[forward, backward]));
        assert_eq!(engine.context_radius(), None);
        let output = run_ok(&engine, 2, 1, &[1.0, 0.0]);
        assert_eq!(output.frames, 2);
        assert_eq!(output.channels, 2);
        // Forward half equals a standalone forward GRU run.
        let solo = load_engine(&gru_model(&weights));
        let solo_out = run_ok(&solo, 2, 1, &[1.0, 0.0]);
        assert_close(&[output.values[0]], &solo_out.values[0..1], 0.0);
        assert_close(&[output.values[2]], &solo_out.values[1..2], 0.0);
        // Backward half equals a forward run on the reversed input, reversed.
        let solo_rev = run_ok(&solo, 2, 1, &[0.0, 1.0]);
        assert_close(
            &[output.values[1], output.values[3]],
            &[solo_rev.values[1], solo_rev.values[0]],
            0.0,
        );
    }

    #[test]
    fn bidirectional_gru_needs_two_gru_children() {
        let gru = gru_model(&[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]);
        let root = node_bytes("modl6", &[], &[gru]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: bidirectional GRU requires two GRU children and an empty payload."
        );
        let root = node_bytes("modl6", &[1], &[gru_model(&[1.0; 6]), gru_model(&[1.0; 6])]);
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: bidirectional GRU requires two GRU children and an empty payload."
        );
        let root = node_bytes(
            "modl6",
            &[],
            &[leaf("moda5", &[]), gru_model(&[1.0; 6])],
        );
        assert_eq!(
            load_error(&root),
            "DNNI inference at 0x8: bidirectional GRU children must be modl3 operators."
        );
    }

    #[test]
    fn bidirectional_gru_directions_must_agree() {
        // Second direction reads 2 input channels.
        let mut first_children = Vec::new();
        for _ in 0..6 {
            first_children.push(leaf("prim0", &dense_payload(1, 1, &[1.0])));
            first_children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        let first = node_bytes("modl3", &[], &first_children);
        let mut children = Vec::new();
        for (rows, columns) in [(1, 2), (1, 2), (1, 2), (1, 1), (1, 1), (1, 1)] {
            let values = vec![1.0; (rows * columns) as usize];
            children.push(leaf("prim0", &dense_payload(rows, columns, &values)));
            children.push(leaf("prim1", &vector_payload(&[0.0])));
        }
        let second = node_bytes("modl3", &[], &children);
        let root = node_bytes("modl6", &[], &[first, second]);
        let error = load_error(&root);
        assert!(
            error.ends_with(
                "bidirectional GRU directions have incompatible input channels or excessive output channels."
            ),
            "unexpected: {error}"
        );
    }

    // --- Context radius ---

    #[test]
    fn context_radius_sums_sequence_takes_max_of_gated() {
        // Sequence of pad-1 and pad-2 frame-preserving convolutions.
        let first = conv_model(3, 1, 1, 1, &[vec![1.0], vec![1.0], vec![1.0]], 1, 1, None);
        let second = conv_model(5, 1, 2, 1, &vec![vec![1.0]; 5], 1, 1, None);
        let engine = load_engine(&node_bytes("modm0", &[], &[first, second]));
        assert_eq!(engine.context_radius(), Some(3));
    }

    #[test]
    fn asymmetric_conv_has_no_radius() {
        // padding 1 with kernel 5 needs padding 2 for symmetry.
        let root = conv_model(
            5,
            1,
            1,
            1,
            &vec![vec![1.0]; 5],
            1,
            1,
            None,
        );
        assert_eq!(load_engine(&root).context_radius(), None);
    }

    // --- Blocked packing ---

    /// Loads a dense blob and returns its packed matrix through the real
    /// `load_matrix` path, so packing tests cover the transpose, not a
    /// test-local reimplementation.
    fn packed_matrix_via_blob(rows: usize, columns: usize, values: &[f32]) -> DenseMatrix {
        let engine = load_engine(&dense_model(
            rows as u32,
            columns as u32,
            values,
            None,
        ));
        engine
            .root
            .as_ref()
            .expect("dense engine must have a root")
            .matrices[0]
            .clone()
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364_1362_2384_6793_005)
            .wrapping_add(1_4426_9504_0889_6634_07);
        ((*state >> 33) as f32 / u32::MAX as f32 - 0.5) * 4.0
    }

    #[test]
    fn packing_tail_block_is_zero_padded() {
        // 17 rows x 3 columns: two blocks, second block holds row 16 only.
        let values: Vec<f32> = (0..51).map(|index| index as f32 * 0.25 + 1.0).collect();
        let matrix = packed_matrix_via_blob(17, 3, &values);
        assert_eq!(matrix.rows, 17);
        assert_eq!(matrix.columns, 3);
        assert_eq!(matrix.values.len(), 2 * 3);
        for column in 0..3 {
            // Full first block: lanes hold rows 0..16 of this column.
            let full = &matrix.values[column];
            for row in 0..16 {
                assert_eq!(full[row], values[row * 3 + column], "row {row} col {column}");
            }
            // Tail block: lane 0 holds row 16, the rest are zero pad.
            let tail = &matrix.values[3 + column];
            assert_eq!(tail[0], values[16 * 3 + column]);
            assert_eq!(&tail[1..], &[0.0; 15]);
        }
    }

    #[test]
    fn packing_exact_block_and_single_element() {
        // Exact multiple of BLOCK: one block, no padding anywhere.
        let values: Vec<f32> = (0..32).map(|index| index as f32 - 16.0).collect();
        let matrix = packed_matrix_via_blob(16, 2, &values);
        assert_eq!(matrix.values.len(), 2);
        for column in 0..2 {
            for row in 0..16 {
                assert_eq!(matrix.values[column][row], values[row * 2 + column]);
            }
        }
        // 1x1: a single block with lane 0 live and 15 zero lanes.
        let matrix = packed_matrix_via_blob(1, 1, &[2.5]);
        assert_eq!(matrix.values.len(), 1);
        assert_eq!(matrix.values[0][0], 2.5);
        assert_eq!(&matrix.values[0][1..], &[0.0; 15]);
    }

    #[test]
    fn packed_multiply_matches_scalar_exactly() {
        // Packed vs scalar-fallback agreement must be bitwise: both sum each
        // lane's columns in increasing order from a zero accumulator.
        for (rows, columns) in [(1, 1), (2, 3), (16, 16), (17, 3), (31, 5), (5, 33)] {
            let mut state = 0x1234_5678_9abc_def0_u64 ^ (rows as u64 * 31 + columns as u64);
            let values: Vec<f32> = (0..rows * columns).map(|_| lcg(&mut state)).collect();
            let matrix = packed_matrix_via_blob(rows, columns, &values);
            let input: Vec<f32> = (0..columns).map(|_| lcg(&mut state)).collect();
            let mut packed = vec![0.0_f32; rows];
            let mut scalar = vec![0.0_f32; rows];
            multiply_matrix(&matrix, &input, &mut packed);
            multiply_matrix_scalar(&matrix, &input, &mut scalar);
            assert_eq!(packed, scalar, "rows={rows} columns={columns}");
            // Independent row-major reference agrees within float tolerance.
            let mut reference = vec![0.0_f32; rows];
            for row in 0..rows {
                let mut sum = 0.0_f32;
                for column in 0..columns {
                    sum += values[row * columns + column] * input[column];
                }
                reference[row] = sum;
            }
            assert_close(&packed, &reference, 1e-6);
        }
    }

    // --- Cache helpers ---

    fn run_cache_ok(
        engine: &DnniInference,
        input: &Tensor,
        condition: Option<&Tensor>,
        cache: Option<&mut DnniCache>,
        statistics: Option<&mut DnniRunStatistics>,
    ) -> Tensor {
        let mut output = Tensor::default();
        engine
            .run_with_cache(input, &mut output, condition, cache, &test_token(), statistics)
            .expect("cached run must succeed");
        output
    }

    fn tensor(frames: usize, channels: usize, values: Vec<f32>) -> Tensor {
        Tensor {
            frames,
            channels,
            values,
        }
    }

    #[test]
    fn cache_cold_run_populates_and_rerun_reuses_all_frames() {
        let engine = load_engine(&dense_model(2, 2, &[1.0, 0.0, 0.0, 1.0], None));
        let input = tensor(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut cache = DnniCache::default();
        let mut first_stats = DnniRunStatistics::default();
        let first = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut first_stats));
        assert_eq!(
            first_stats,
            DnniRunStatistics {
                computed_frames: 4,
                reused_frames: 0,
                context_frames: 0,
            }
        );
        assert_ne!(cache.model_identity, 0);
        assert!(!cache.has_condition);

        // Identical rerun: every frame reused, output bytes equal.
        let mut second_stats = DnniRunStatistics::default();
        let second = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut second_stats));
        assert_eq!(second.values, first.values);
        assert_eq!(
            second_stats,
            DnniRunStatistics {
                computed_frames: 0,
                reused_frames: 4,
                context_frames: 0,
            }
        );
    }

    #[test]
    fn cache_empty_range_return_only_touches_reused() {
        let engine = load_engine(&dense_model(1, 1, &[2.0], None));
        let input = tensor(3, 1, vec![1.0, 2.0, 3.0]);
        let mut cache = DnniCache::default();
        run_cache_ok(&engine, &input, None, Some(&mut cache), None);
        // Pre-seeded counters must survive the fast return except `reused`.
        let mut statistics = DnniRunStatistics {
            computed_frames: 5,
            reused_frames: 7,
            context_frames: 9,
        };
        run_cache_ok(
            &engine,
            &input,
            None,
            Some(&mut cache),
            Some(&mut statistics),
        );
        assert_eq!(
            statistics,
            DnniRunStatistics {
                computed_frames: 5,
                reused_frames: 10,
                context_frames: 9,
            }
        );
    }

    #[test]
    fn cache_single_frame_edit_recomputes_window() {
        // Kernel-3, pad-1 convolution: context radius 1, weights sum taps.
        let root = conv_model(3, 1, 1, 1, &[vec![1.0], vec![1.0], vec![1.0]], 1, 1, None);
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), Some(1));
        let input = tensor(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut cache = DnniCache::default();
        run_cache_ok(&engine, &input, None, Some(&mut cache), None);

        // Edit frame 3 only: dirty range expands to {2, 5}, the compute
        // window expands again to {1, 6} (5 frames, 2 of them halo).
        let edited = tensor(6, 1, vec![1.0, 2.0, 3.0, 40.0, 5.0, 6.0]);
        let mut statistics = DnniRunStatistics::default();
        let output = run_cache_ok(
            &engine,
            &edited,
            None,
            Some(&mut cache),
            Some(&mut statistics),
        );
        assert_eq!(
            statistics,
            DnniRunStatistics {
                computed_frames: 5,
                reused_frames: 3,
                context_frames: 2,
            }
        );
        // Cached incremental output equals a fresh full run exactly.
        let fresh = run_ok(&engine, 6, 1, &[1.0, 2.0, 3.0, 40.0, 5.0, 6.0]);
        assert_eq!(output.values, fresh.values);
    }

    #[test]
    fn cache_two_far_edits_stay_split() {
        let root = conv_model(3, 1, 1, 1, &[vec![1.0], vec![1.0], vec![1.0]], 1, 1, None);
        let engine = load_engine(&root);
        let input = tensor(10, 1, vec![1.0; 10]);
        let mut cache = DnniCache::default();
        run_cache_ok(&engine, &input, None, Some(&mut cache), None);

        // Edits at frames 1 and 8: expanded ranges {0, 3} and {7, 10} stay
        // disjoint after the second window expansion ({0, 4}, {6, 10}).
        let mut edited_values = vec![1.0; 10];
        edited_values[1] = 9.0;
        edited_values[8] = 9.0;
        let edited = tensor(10, 1, edited_values);
        let mut statistics = DnniRunStatistics::default();
        let output = run_cache_ok(
            &engine,
            &edited,
            None,
            Some(&mut cache),
            Some(&mut statistics),
        );
        assert_eq!(
            statistics,
            DnniRunStatistics {
                computed_frames: 8,
                reused_frames: 4,
                context_frames: 2,
            }
        );
        let fresh = run_ok(&engine, 10, 1, &[1.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0]);
        assert_eq!(output.values, fresh.values);
    }

    #[test]
    fn cache_condition_mismatch_disables_caching() {
        // A dense layer ignores the condition tensor, so mismatched
        // condition frames let the run succeed while disabling the snapshot.
        let engine = load_engine(&dense_model(1, 1, &[2.0], None));
        let input = tensor(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let mismatched = tensor(3, 1, vec![1.0, 1.0, 1.0]);
        let mut cache = DnniCache::default();
        let output = run_cache_ok(&engine, &input, Some(&mismatched), Some(&mut cache), None);
        assert_eq!(output.values, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(cache.model_identity, 0);
        assert_eq!(cache.get_bytes(), 0);
    }

    #[test]
    fn cache_gru_has_no_radius_and_clears() {
        let engine = load_engine(&gru_model(&[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]));
        assert_eq!(engine.context_radius(), None);
        let input = tensor(2, 1, vec![1.0, 0.0]);
        let mut cache = DnniCache::default();
        let mut first_stats = DnniRunStatistics::default();
        let first = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut first_stats));
        assert_eq!(first_stats.computed_frames, 2);
        // Recurrent state cannot be windowed: the cache is cleared and the
        // second run recomputes everything.
        assert_eq!(cache.model_identity, 0);
        let mut second_stats = DnniRunStatistics::default();
        let second = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut second_stats));
        assert_eq!(second.values, first.values);
        assert_eq!(second_stats.computed_frames, 2);
        assert_eq!(second_stats.reused_frames, 0);
    }

    #[test]
    fn cache_model_reload_invalidates() {
        let mut engine = load_engine(&dense_model(1, 1, &[2.0], None));
        let input = tensor(2, 1, vec![3.0, 4.0]);
        let mut cache = DnniCache::default();
        let first = run_cache_ok(&engine, &input, None, Some(&mut cache), None);
        assert_eq!(first.values, vec![6.0, 8.0]);

        // Reload with different weights: the stale snapshot must not be
        // reused even though the input is identical.
        let reader = load_reader(&dense_model(1, 1, &[5.0], None));
        engine.load(&reader, 0).expect("reload must succeed");
        let mut statistics = DnniRunStatistics::default();
        let second = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut statistics));
        assert_eq!(second.values, vec![15.0, 20.0]);
        assert_eq!(statistics.computed_frames, 2);
        assert_eq!(statistics.reused_frames, 0);
    }

    #[test]
    fn cache_strided_conv_clears_on_frame_mismatch() {
        // Stride-2 convolution has no finite radius; the 5-frame input also
        // yields a 2-frame output, so no snapshot is taken either way.
        let root = conv_model(2, 2, 0, 1, &[vec![1.0], vec![1.0]], 1, 1, Some(&[0.0]));
        let engine = load_engine(&root);
        assert_eq!(engine.context_radius(), None);
        let input = tensor(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut cache = DnniCache::default();
        let output = run_cache_ok(&engine, &input, None, Some(&mut cache), None);
        assert_eq!(output.values, vec![3.0, 7.0]);
        assert_eq!(cache.model_identity, 0);
        assert_eq!(cache.get_bytes(), 0);
    }

    #[test]
    fn cache_oversize_snapshot_clears() {
        // Past the 64 MiB / 16 Mi-element snapshot cap the run still
        // succeeds, but the cache is cleared instead of snapshotted.
        let engine = load_engine(&dense_model(1, 1, &[1.0], None));
        let frames = 8_500_000_usize;
        let input = tensor(frames, 1, vec![1.0; frames]);
        let mut cache = DnniCache::default();
        let mut statistics = DnniRunStatistics::default();
        let output = run_cache_ok(&engine, &input, None, Some(&mut cache), Some(&mut statistics));
        assert_eq!(output.values.len(), frames);
        assert_eq!(statistics.computed_frames, frames);
        assert_eq!(cache.model_identity, 0);
        assert_eq!(cache.get_bytes(), 0);
    }

    #[test]
    fn cache_get_bytes_counts_capacity_and_clear_resets() {
        let engine = load_engine(&dense_model(1, 1, &[1.0], None));
        let input = tensor(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let mut cache = DnniCache::default();
        run_cache_ok(&engine, &input, None, Some(&mut cache), None);
        // Capacity-based accounting always covers at least the live lengths.
        assert!(cache.get_bytes() >= (4 + 0 + 4) * size_of::<f32>());
        assert_ne!(cache.model_identity, 0);
        cache.clear();
        assert_eq!(cache.model_identity, 0);
        assert!(!cache.has_condition);
        assert!(cache.input.values.is_empty());
        assert_eq!(cache.get_bytes(), 0);
    }

    #[test]
    fn expand_range_clamps_to_bounds() {
        assert_eq!(
            expand_range(FrameRange { first: 0, end: 1 }, 2, 10),
            FrameRange { first: 0, end: 3 }
        );
        assert_eq!(
            expand_range(FrameRange { first: 9, end: 10 }, 2, 10),
            FrameRange { first: 7, end: 10 }
        );
        assert_eq!(
            expand_range(FrameRange { first: 0, end: 10 }, 5, 10),
            FrameRange { first: 0, end: 10 }
        );
        assert_eq!(
            expand_range(FrameRange { first: 3, end: 4 }, 0, 10),
            FrameRange { first: 3, end: 4 }
        );
    }

    #[test]
    fn same_shape_compares_frames_channels_and_len() {
        let base = tensor(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(same_shape(&base, &tensor(2, 2, vec![0.0; 4])));
        assert!(!same_shape(&base, &tensor(4, 1, vec![1.0, 2.0, 3.0, 4.0])));
        assert!(!same_shape(&base, &tensor(2, 1, vec![1.0, 2.0])));
        assert!(!same_shape(&base, &tensor(2, 2, vec![1.0, 2.0])));
    }

    #[test]
    fn slice_frames_copies_window() {
        let input = tensor(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            slice_frames(&input, 1, 3),
            tensor(2, 2, vec![3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(slice_frames(&input, 0, 4), input);
    }

    #[test]
    fn find_changed_ranges_merge_touching_and_condition_only() {
        let previous = tensor(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Single edit at frame 2, radius 1 → [{1, 4}].
        let mut current_values = previous.values.clone();
        current_values[2] = 30.0;
        let current = tensor(6, 1, current_values);
        assert_eq!(
            find_changed_output_ranges(&previous, &current, None, None, 1),
            vec![FrameRange { first: 1, end: 4 }]
        );
        // Edits at frames 1 and 4, radius 1: expanded {0, 3} and {3, 6}
        // touch at 3, so they pre-merge into [{0, 6}].
        let mut merged_values = previous.values.clone();
        merged_values[1] = 20.0;
        merged_values[4] = 50.0;
        let merged = tensor(6, 1, merged_values);
        assert_eq!(
            find_changed_output_ranges(&previous, &merged, None, None, 1),
            vec![FrameRange { first: 0, end: 6 }]
        );
        // Far edits stay split with radius 0.
        let mut split_values = previous.values.clone();
        split_values[0] = 10.0;
        split_values[5] = 60.0;
        let split = tensor(6, 1, split_values);
        assert_eq!(
            find_changed_output_ranges(&previous, &split, None, None, 0),
            vec![
                FrameRange { first: 0, end: 1 },
                FrameRange { first: 5, end: 6 },
            ]
        );
        // Condition-only edit with identical input still flags the frame.
        let previous_condition = tensor(6, 1, vec![0.0; 6]);
        let mut condition_values = vec![0.0; 6];
        condition_values[2] = 1.0;
        let current_condition = tensor(6, 1, condition_values);
        assert_eq!(
            find_changed_output_ranges(
                &previous,
                &previous,
                Some(&previous_condition),
                Some(&current_condition),
                0,
            ),
            vec![FrameRange { first: 2, end: 3 }]
        );
        // `-0.0` vs `0.0` counts as changed, matching `memcmp` semantics.
        let zero_previous = tensor(6, 1, vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0]);
        let mut neg_zero_values = zero_previous.values.clone();
        neg_zero_values[2] = -0.0;
        let neg_zero = tensor(6, 1, neg_zero_values);
        assert_eq!(
            find_changed_output_ranges(&zero_previous, &neg_zero, None, None, 0),
            vec![FrameRange { first: 2, end: 3 }]
        );
    }

    #[test]
    fn cached_run_with_condition_matches_fresh() {
        let input_conv = conv_model(1, 1, 0, 1, &[vec![1.0, 2.0]], 2, 1, None);
        let cond_conv = conv_model(1, 1, 0, 1, &[vec![10.0, 20.0]], 2, 1, None);
        let root = node_bytes("_gnc1v0", &dims3(1, 1, 1), &[input_conv, cond_conv]);
        let engine = load_engine(&root);
        let input = tensor(3, 1, vec![1.0, 2.0, 3.0]);
        let condition = tensor(3, 1, vec![1.0, 1.0, 1.0]);
        let mut cache = DnniCache::default();
        let first = run_cache_ok(
            &engine,
            &input,
            Some(&condition),
            Some(&mut cache),
            None,
        );
        assert!(cache.has_condition);

        // Identical rerun reuses everything, condition included.
        let mut statistics = DnniRunStatistics::default();
        let second = run_cache_ok(
            &engine,
            &input,
            Some(&condition),
            Some(&mut cache),
            Some(&mut statistics),
        );
        assert_eq!(second.values, first.values);
        assert_eq!(statistics.reused_frames, 3);

        // Editing only the condition recomputes the affected window and
        // still matches a fresh run exactly.
        let mut edited_condition_values = vec![1.0, 1.0, 1.0];
        edited_condition_values[1] = 5.0;
        let edited_condition = tensor(3, 1, edited_condition_values);
        let third = run_cache_ok(
            &engine,
            &input,
            Some(&edited_condition),
            Some(&mut cache),
            None,
        );
        let fresh = run_cond_ok(&engine, 3, 1, &[1.0, 2.0, 3.0], Some(edited_condition));
        assert_eq!(third.values, fresh.values);
    }

    // --- Golden tensors dumped from the C++ engine ---
    //
    // `tests/golden/dnni/<case>.{dnni,in.txt,cond.txt,out.txt}` were produced
    // by the throwaway dumper in `do-not-distribute/dnni-dump/` (same networks
    // as the hand-computed unit tests above):
    //
    //   ./make_blobs.py /tmp/blobs   # writes <case>.dnni + <case>.in.txt
    //   ./dump <case>.dnni <case>.in.txt [<case>.cond.txt] > <case>.out.txt
    //
    // linked against OpenSV's `DnniReader`/`DnniInference` plus JUCE
    // (`juce_core`, `juce_events`, `juce_audio_basics`, `juce_dsp`). The full
    // `OpenSVEngine` static target does not build with the environment's
    // GCC 16 (`PhonemeTiming.cpp:187` rejects brace-init into `push_back`;
    // upstream issue, unrelated to DNNI), so only the DNNI translation units
    // plus the JUCE modules they need were compiled. Tolerance is 1e-5
    // relative: the scalar port keeps the C++ per-element summation order but
    // evaluates the sigmoid exponential in the same double precision as the
    // C++ code, so agreement is far tighter in practice.

    fn golden_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/dnni")
    }

    fn read_golden_tensor(path: &std::path::Path) -> Tensor {
        let text = std::fs::read_to_string(path).expect("golden file must exist");
        let mut lines = text.lines();
        let header: Vec<usize> = lines
            .next()
            .expect("golden header")
            .split_whitespace()
            .map(|part| part.parse().expect("header integers"))
            .collect();
        assert_eq!(header.len(), 2, "malformed golden header");
        let values: Vec<f32> = lines
            .map(|line| line.parse().expect("golden floats"))
            .collect();
        assert_eq!(values.len(), header[0] * header[1], "malformed golden body");
        Tensor {
            frames: header[0],
            channels: header[1],
            values,
        }
    }

    #[test]
    fn golden_tensors_match_cpp_engine() {
        for case in [
            "dense",
            "conv_pad",
            "conv_stride",
            "conv_dilation",
            "gated",
            "gated_cond",
            "residual",
            "gru",
            "bigru",
            "activations",
        ] {
            let dir = golden_dir();
            let blob = std::fs::read(dir.join(format!("{case}.dnni"))).expect("golden blob");
            let reader = DnniReader::from_bytes(blob).expect("golden blob must parse");
            let mut engine = DnniInference::new();
            engine.load(&reader, 0).expect("golden model must load");
            let input = read_golden_tensor(&dir.join(format!("{case}.in.txt")));
            let condition_path = dir.join(format!("{case}.cond.txt"));
            let condition = condition_path
                .exists()
                .then(|| read_golden_tensor(&condition_path));
            let mut output = Tensor::default();
            engine
                .run(&input, &mut output, condition.as_ref(), &test_token())
                .expect("golden run must succeed");
            let expected = read_golden_tensor(&dir.join(format!("{case}.out.txt")));
            assert_eq!(output.frames, expected.frames, "{case} frames");
            assert_eq!(output.channels, expected.channels, "{case} channels");
            assert_close(&output.values, &expected.values, 1e-5);
            // A cached second run over identical inputs must reproduce the
            // same bytes exactly (full reuse for finite-radius networks, full
            // recompute otherwise).
            let mut cache = DnniCache::default();
            let mut statistics = DnniRunStatistics::default();
            engine
                .run_with_cache(
                    &input,
                    &mut Tensor::default(),
                    condition.as_ref(),
                    Some(&mut cache),
                    &test_token(),
                    Some(&mut statistics),
                )
                .expect("golden cache prime must succeed");
            let mut rerun = Tensor::default();
            engine
                .run_with_cache(
                    &input,
                    &mut rerun,
                    condition.as_ref(),
                    Some(&mut cache),
                    &test_token(),
                    None,
                )
                .expect("golden cached rerun must succeed");
            assert_eq!(rerun.frames, output.frames, "{case} cached frames");
            assert_eq!(rerun.channels, output.channels, "{case} cached channels");
            assert_eq!(rerun.values, output.values, "{case} cached bytes");
        }
    }
}

