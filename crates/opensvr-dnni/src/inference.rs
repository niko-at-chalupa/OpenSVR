//! Scalar neural network inference for Synthesizer V DNNI models.
//!
//! This is the Rust counterpart of OpenSV's `src/synthesis/DnniInference.{h,cpp}`
//! (namespace `sv::synthesis`), first half: model loading plus every kernel in
//! plain scalar form over row-major weights. Blocked SIMD weight packing and
//! the incremental dirty-range `Cache` are deferred to the follow-up PR; the
//! scalar summation order is documented at each kernel so the packed port can
//! preserve it.
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

/// Plain row-major weight matrix.
///
/// `values[row * columns + column]`; `rows` is the output dimension and
/// `columns` the input dimension. The C++ engine stores these in blocked SIMD
/// layout (`WeightBlock`); the scalar port keeps them row-major and the
/// packed port must preserve the documented summation orders.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
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
/// Plain-data mirror of `DnniInference::Layer`: `matrices` are row-major
/// (one entry per convolution kernel tap, six entries for GRU projections),
/// `bias` holds the dense/conv bias or the six concatenated GRU biases
/// `[b_ir, b_iz, b_in, b_hr, b_hz, b_hn]`.
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
        let mut replacement = Tensor::default();
        run_layer(root, input, &mut replacement, condition, cancel)?;
        if cancel.is_cancelled() {
            return Err(cancelled());
        }
        *output = replacement;
        Ok(())
    }
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

/// Scalar matrix-vector product: `output[row] = sum over columns in order`.
///
/// This is the scalar equivalent of `multiplyMatrix`: the C++ version zeroes
/// per-row SIMD accumulators and sums column-inner, so each output element is
/// the columns summed in increasing order, overwriting (not accumulating
/// into) the target. The packed port must keep that per-element order
/// (ROADMAP principle 4).
fn multiply_matrix(matrix: &DenseMatrix, input: &[f32], output: &mut [f32]) {
    for (row, slot) in output.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for column in 0..matrix.columns {
            sum += matrix.values[row * matrix.columns + column] * input[column];
        }
        *slot = sum;
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
    // The C++ loader accounts zero-padded SIMD blocks here; the scalar port
    // stores plain row-major weights, so it accounts the true element count
    // under the same message and limit.
    if columns > (MAXIMUM_ELEMENTS - *parameter_count) / rows {
        return Err(node_error(
            offset,
            "packed model exceeds the parameter memory limit.",
        ));
    }
    *parameter_count += rows * columns;
    Ok(DenseMatrix {
        rows,
        columns,
        values: decoded.values,
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
            // channels in increasing order. The C++ 2-frame unrolling keeps
            // independent accumulator sets per frame, so the scalar port drops
            // it without changing values.
            for (kernel, matrix) in layer.matrices.iter().enumerate() {
                let tap = frame as i64 * layer.stride as i64 + kernel as i64 * layer.dilation as i64
                    - layer.padding as i64;
                if tap < 0 || tap as usize >= input.frames {
                    // Out-of-range taps contribute zero (padding), never
                    // clamping.
                    continue;
                }
                let source = &input.values[tap as usize * input.channels..];
                for channel_out in 0..output.channels {
                    let weights = &matrix.values[channel_out * matrix.columns..];
                    let mut sum = output.values[frame * output.channels + channel_out];
                    for channel_in in 0..input.channels {
                        sum += weights[channel_in] * source[channel_in];
                    }
                    output.values[frame * output.channels + channel_out] = sum;
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
        }
    }
}

