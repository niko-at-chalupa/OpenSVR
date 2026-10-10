//! Per-channel feature normalization backed by a `cmpu0` DNNI node.
//!
//! Ports OpenSV's `src/synthesis/FeatureNormalizer.{h,cpp}`. The `cmpu0`
//! node holds three `prim1` vector children: per-channel `lower` bounds,
//! per-channel `upper` bounds, and the two-float target `[lower, upper]`
//! range. Forward (`normalize`) maps raw features into the target range
//! with an optional clamp; `denormalize` maps back to physical units.

use opensvr_dnni::{DnniReader, DnniTensor};

use crate::{Result, invalid};

/// Min-max feature scaler loaded from a `cmpu0` node.
#[derive(Debug, Clone, Default)]
pub struct FeatureNormalizer {
    lower: Vec<f32>,
    upper: Vec<f32>,
    target_lower: f32,
    target_upper: f32,
}

impl FeatureNormalizer {
    /// Loads normalization bounds from the `cmpu0` node at `node_index`.
    ///
    /// The node must have an empty payload and exactly three `prim1` vector
    /// children (`lower`, `upper`, `[targetLower, targetUpper]`). Nothing is
    /// stored on failure.
    pub fn load(&mut self, reader: &DnniReader, node_index: usize) -> Result<()> {
        let node = reader.nodes().get(node_index).filter(|node| {
            node.name == "cmpu0"
                && reader.payload(node_index).is_empty()
                && node.children.len() == 3
        });
        let Some(node) = node else {
            return Err(invalid(
                "Feature normalization requires an empty cmpu0 node with three vector children.",
            ));
        };
        let children = node.children.clone();
        let lower = reader.read_float_vector(children[0])?;
        let upper = reader.read_float_vector(children[1])?;
        let range = reader.read_float_vector(children[2])?;

        if lower.is_empty()
            || lower.len() != upper.len()
            || range.len() != 2
            || range[0] >= range[1]
            || !(range[1] - range[0]).is_finite()
        {
            return Err(invalid("Invalid cmpu0 channel count or target range."));
        }
        for (channel, (&low, &high)) in lower.iter().zip(&upper).enumerate() {
            if low > high || !(high - low).is_finite() {
                return Err(invalid(format!(
                    "Invalid cmpu0 source range at channel {channel}."
                )));
            }
        }

        self.lower = lower;
        self.upper = upper;
        self.target_lower = range[0];
        self.target_upper = range[1];
        Ok(())
    }

    /// Returns the number of channels this normalizer was loaded with.
    pub fn channel_count(&self) -> usize {
        self.lower.len()
    }

    /// Maps raw features into the normalized target range.
    ///
    /// With `clamp` (the C++ default) outputs are clamped to
    /// `[targetLower, targetUpper]`. `output` is only assigned on success.
    pub fn normalize(
        &self,
        input: &DnniTensor,
        output: &mut DnniTensor,
        clamp: bool,
    ) -> Result<()> {
        self.transform(input, output, false, clamp)
    }

    /// Maps normalized features back to physical units (never clamped).
    ///
    /// `output` is only assigned on success.
    pub fn denormalize(&self, input: &DnniTensor, output: &mut DnniTensor) -> Result<()> {
        self.transform(input, output, true, false)
    }

    fn transform(
        &self,
        input: &DnniTensor,
        output: &mut DnniTensor,
        inverse: bool,
        clamp: bool,
    ) -> Result<()> {
        if self.lower.is_empty() {
            return Err(invalid("Feature normalization has not been loaded."));
        }
        if input.channels != self.lower.len()
            || input.channels == 0
            || input.frames > usize::MAX / input.channels
            || input.frames * input.channels != input.values.len()
        {
            return Err(invalid(
                "Feature normalization input tensor has an invalid shape.",
            ));
        }

        let mut values = Vec::with_capacity(input.values.len());
        let target_range = self.target_upper - self.target_lower;
        for (index, &value) in input.values.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(
                    "Feature normalization input contains a non-finite value.",
                ));
            }
            let channel = index % input.channels;
            let source_range = self.upper[channel] - self.lower[channel];
            let transformed = if inverse {
                self.lower[channel] + source_range * (value - self.target_lower) / target_range
            } else {
                // Floor of 1e-5 in the denominator prevents division by zero
                // for constant features.
                let scaled = self.target_lower
                    + target_range * (value - self.lower[channel])
                        / source_range.max(1.0e-5f32);
                if clamp {
                    scaled.clamp(self.target_lower, self.target_upper)
                } else {
                    scaled
                }
            };
            if !transformed.is_finite() {
                return Err(invalid(
                    "Feature normalization produced a non-finite value.",
                ));
            }
            values.push(transformed);
        }
        *output = DnniTensor {
            frames: input.frames,
            channels: input.channels,
            values,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(name: &str, payload: &[u8]) -> Vec<u8> {
        let mut tag = [0_u8; 8];
        tag[..name.len()].copy_from_slice(name.as_bytes());
        let mut out = Vec::new();
        out.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        out.extend_from_slice(&tag);
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
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

    /// Builds a v1 blob whose root is a `cmpu0` node over the three vectors.
    fn cmpu_reader(lower: &[f32], upper: &[f32], range: &[f32]) -> DnniReader {
        let children = [lower, upper, range]
            .iter()
            .map(|values| leaf("prim1", &vector_payload(values)))
            .collect::<Vec<_>>()
            .concat();
        let mut tag = [0_u8; 8];
        tag[.."cmpu0".len()].copy_from_slice(b"cmpu0");
        let mut root = Vec::new();
        root.extend_from_slice(&0x7fca_40ff_u32.to_le_bytes());
        root.extend_from_slice(&tag);
        root.extend_from_slice(&3_u32.to_le_bytes());
        root.extend_from_slice(&0_u32.to_le_bytes());
        root.extend_from_slice(&children);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x7fca_00ff_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&root);
        DnniReader::from_bytes(bytes).expect("synthetic cmpu0 must parse")
    }

    fn loaded() -> FeatureNormalizer {
        // lower [-5, 0], upper [5, 10], target [0, 1]:
        // channel 0 maps [-5, 5] -> [0, 1], channel 1 maps [0, 10] -> [0, 1].
        let reader = cmpu_reader(&[-5.0, 0.0], &[5.0, 10.0], &[0.0, 1.0]);
        let mut normalizer = FeatureNormalizer::default();
        normalizer.load(&reader, 0).expect("valid cmpu0 must load");
        assert_eq!(normalizer.channel_count(), 2);
        normalizer
    }

    #[test]
    fn normalize_applies_min_max_scaling() {
        let normalizer = loaded();
        let input = DnniTensor {
            frames: 2,
            channels: 2,
            values: vec![0.0, 5.0, -5.0, 10.0],
        };
        let mut output = DnniTensor::default();
        normalizer
            .normalize(&input, &mut output, false)
            .expect("scaling must succeed");
        assert_eq!(output.frames, 2);
        assert_eq!(output.channels, 2);
        // (0 - -5) / 10 = 0.5; (5 - 0) / 10 = 0.5; ends map to 0 and 1.
        assert_eq!(output.values, vec![0.5, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn normalize_clamps_to_target_range_by_default() {
        let normalizer = loaded();
        let input = DnniTensor {
            frames: 1,
            channels: 2,
            values: vec![-500.0, 500.0],
        };
        let mut clamped = DnniTensor::default();
        normalizer
            .normalize(&input, &mut clamped, true)
            .expect("clamped scaling must succeed");
        assert_eq!(clamped.values, vec![0.0, 1.0]);
        let mut raw = DnniTensor::default();
        normalizer
            .normalize(&input, &mut raw, false)
            .expect("unclamped scaling must succeed");
        assert_eq!(raw.values, vec![-49.5, 50.0]);
    }

    #[test]
    fn denormalize_inverts_the_forward_map() {
        let normalizer = loaded();
        let mut back = DnniTensor::default();
        normalizer
            .denormalize(
                &DnniTensor {
                    frames: 1,
                    channels: 2,
                    values: vec![0.5, 0.25],
                },
                &mut back,
            )
            .expect("denormalize must succeed");
        // channel 0: -5 + 10 * 0.5 = 0; channel 1: 0 + 10 * 0.25 = 2.5.
        assert_eq!(back.values, vec![0.0, 2.5]);
    }

    #[test]
    fn constant_channel_uses_the_epsilon_floor() {
        let reader = cmpu_reader(&[3.0], &[3.0], &[0.0, 1.0]);
        let mut normalizer = FeatureNormalizer::default();
        normalizer.load(&reader, 0).expect("constant range must load");
        let mut output = DnniTensor::default();
        normalizer
            .normalize(
                &DnniTensor {
                    frames: 1,
                    channels: 1,
                    values: vec![3.0],
                },
                &mut output,
                false,
            )
            .expect("constant channel must scale");
        assert_eq!(output.values, vec![0.0]);
    }

    #[test]
    fn wrong_node_type_is_rejected() {
        let reader = cmpu_reader(&[0.0], &[1.0], &[0.0, 1.0]);
        // Index 1 is a prim1 leaf, not a cmpu0 node.
        let mut normalizer = FeatureNormalizer::default();
        assert_eq!(
            normalizer.load(&reader, 1).unwrap_err().to_string(),
            "Feature normalization requires an empty cmpu0 node with three vector children."
        );
        assert_eq!(
            normalizer.load(&reader, 99).unwrap_err().to_string(),
            "Feature normalization requires an empty cmpu0 node with three vector children."
        );
        assert_eq!(normalizer.channel_count(), 0);
    }

    #[test]
    fn bad_channel_counts_are_rejected() {
        for (lower, upper, range) in [
            (vec![], vec![], vec![0.0, 1.0]),
            (vec![0.0], vec![0.0, 1.0], vec![0.0, 1.0]),
            (vec![0.0], vec![1.0], vec![0.0]),
            (vec![0.0], vec![1.0], vec![1.0, 0.0]),
            (vec![0.0], vec![1.0], vec![0.5, 0.5]),
        ] {
            let reader = cmpu_reader(&lower, &upper, &range);
            let mut normalizer = FeatureNormalizer::default();
            assert_eq!(
                normalizer.load(&reader, 0).unwrap_err().to_string(),
                "Invalid cmpu0 channel count or target range.",
                "lower {lower:?} upper {upper:?} range {range:?}"
            );
        }
    }

    #[test]
    fn inverted_source_range_reports_its_channel() {
        let reader = cmpu_reader(&[2.0, 0.0], &[1.0, 1.0], &[0.0, 1.0]);
        let mut normalizer = FeatureNormalizer::default();
        assert_eq!(
            normalizer.load(&reader, 0).unwrap_err().to_string(),
            "Invalid cmpu0 source range at channel 0."
        );
    }

    #[test]
    fn unloaded_normalizer_rejects_transforms() {
        let normalizer = FeatureNormalizer::default();
        let input = DnniTensor {
            frames: 1,
            channels: 1,
            values: vec![0.0],
        };
        let mut output = DnniTensor::default();
        assert_eq!(
            normalizer
                .normalize(&input, &mut output, true)
                .unwrap_err()
                .to_string(),
            "Feature normalization has not been loaded."
        );
        assert_eq!(
            normalizer
                .denormalize(&input, &mut output)
                .unwrap_err()
                .to_string(),
            "Feature normalization has not been loaded."
        );
    }

    #[test]
    fn misshaped_input_is_rejected() {
        let normalizer = loaded();
        let mut output = DnniTensor {
            frames: 7,
            channels: 7,
            values: vec![1.0; 49],
        };
        for input in [
            DnniTensor {
                frames: 1,
                channels: 3,
                values: vec![0.0; 3],
            },
            DnniTensor {
                frames: 1,
                channels: 2,
                values: vec![0.0; 3],
            },
            DnniTensor {
                frames: 1,
                channels: 0,
                values: vec![],
            },
        ] {
            assert_eq!(
                normalizer
                    .normalize(&input, &mut output, true)
                    .unwrap_err()
                    .to_string(),
                "Feature normalization input tensor has an invalid shape."
            );
            // The output is untouched on failure.
            assert_eq!(output.values, vec![1.0; 49]);
        }
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let normalizer = loaded();
        let mut output = DnniTensor::default();
        for value in [f32::INFINITY, f32::NAN] {
            let input = DnniTensor {
                frames: 1,
                channels: 2,
                values: vec![value, 0.0],
            };
            assert_eq!(
                normalizer
                    .normalize(&input, &mut output, true)
                    .unwrap_err()
                    .to_string(),
                "Feature normalization input contains a non-finite value."
            );
        }
    }

    #[test]
    fn non_finite_output_is_rejected() {
        // upper = +inf passes the load check (inf - 0 is finite... no:
        // inf is not finite, so craft overflow instead): a huge value whose
        // scaled output overflows to infinity.
        let reader = cmpu_reader(&[-1.0e30], &[1.0e30], &[-1.0e30, 1.0e30]);
        let mut normalizer = FeatureNormalizer::default();
        normalizer.load(&reader, 0).expect("wide range must load");
        let mut output = DnniTensor::default();
        // Denormalizing the largest finite f32 overflows: 1e30 + 2e30 * ~3.4e38.
        let input = DnniTensor {
            frames: 1,
            channels: 1,
            values: vec![f32::MAX],
        };
        assert_eq!(
            normalizer
                .denormalize(&input, &mut output)
                .unwrap_err()
                .to_string(),
            "Feature normalization produced a non-finite value."
        );
    }
}
