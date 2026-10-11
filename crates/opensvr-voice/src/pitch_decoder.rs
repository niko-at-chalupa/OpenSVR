//! Two-pass neural pitch decoder (`gen5`).
//!
//! Ports OpenSV's `src/synthesis/PitchDecoder.{h,cpp}` (namespace
//! `sv::synthesis`): nine sub-networks turning pooled context, sinusoidal
//! control embeddings and Gaussian noise into a normalized pitch contour.
//! Pass one (scalar projection, coarse residual, preliminary head) extracts
//! per-note tilt (median slope) and shift (80th percentile) statistics on
//! vowel frames; pass two fuses vibrato/tilt/shift control embeddings through
//! a correction network and a refined residual into the final pitch head.

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniInference, DnniReader, DnniRunStatistics, DnniTensor};

use crate::pitch_features::PitchFeatures;
use crate::{Result, invalid};

/// Memory ceiling for working tensors (64 M single-precision elements).
const MAXIMUM_ELEMENTS: usize = 64 * 1024 * 1024;

/// Verified `gen5` pitch decoder container type hashes.
const DECODER_TYPES: [u64; 12] = [
    0x6713_3569_b1e0_c9a7,
    0x0833_163a_9cfb_e985,
    0x9d1c_eeb0_19e4_ee82,
    0xc2d6_83aa_ffb9_c539,
    0x7878_dd76_1d20_8ab5,
    0x833c_88b7_3fda_ad72,
    0x8385_3dd7_b2ff_c31b,
    0xcd07_5093_5e49_d6e3,
    0x3727_7fa8_5c86_b715,
    0x0dcc_2370_1223_d7ca,
    0xe886_cc8e_9fe4_91d4,
    0xca7b_4a8a_0270_7df0,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Pitch decoder: {}", reason.into()))
}

/// Per-note metadata exchanged between the caller and [`PitchDecoder`].
///
/// Inputs are borrowed; the model-predicted tilt (semitones/sec) and shift
/// (semitones) are populated on success. `requested_tilt` / `requested_shift`
/// overrides are `None` (or NaN entries) to use the model prediction.
#[derive(Debug, Default)]
pub struct PitchDecoderNotes<'a> {
    /// 5 ms frames per note.
    pub frame_counts: &'a [usize],
    /// Score MIDI pitch per note.
    pub midi_pitch: &'a [f32],
    /// Vowel mask per frame (nonzero where statistics are evaluated).
    pub vowel_frames: &'a [u8],
    /// Optional per-note tilt overrides (NaN = model prediction).
    pub requested_tilt: Option<&'a [f32]>,
    /// Optional per-note shift overrides (NaN = model prediction).
    pub requested_shift: Option<&'a [f32]>,
    /// Model-predicted tilt per note (output).
    pub predicted_tilt: Vec<f32>,
    /// Model-predicted shift per note (output).
    pub predicted_shift: Vec<f32>,
}

/// Upsamples a grouped tensor by repeating each group `stride` times.
///
/// Frame `t` copies group `t / stride`; the output has `frames` rows.
fn repeat_frames(input: &DnniTensor, frames: usize, stride: usize) -> DnniTensor {
    let mut output = DnniTensor {
        frames,
        channels: input.channels,
        values: vec![0.0; frames * input.channels],
    };
    for frame in 0..frames {
        let source = (frame / stride) * input.channels;
        let destination = frame * input.channels;
        output.values[destination..destination + input.channels]
            .copy_from_slice(&input.values[source..source + input.channels]);
    }
    output
}

/// Statistical quantile via partial sort: the value at `(len - 1) * position`.
fn quantile(mut values: Vec<f32>, position: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f32 * position) as usize;
    let (_, selected, _) = values.select_nth_unstable_by(index, f32::total_cmp);
    *selected
}

/// Expands per-note control scalars into a continuous per-frame signal.
///
/// Each note's influence extends five frames past its boundaries with a
/// two-sided sigmoid taper:
/// `weight = 1 / ((1 + exp(-(duration - local))) * (1 + exp(-local)))`.
fn expand_note_values(counts: &[usize], values: &[f32], frames: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; frames];
    let mut onset = 0_i64;
    for (note, &duration) in counts.iter().enumerate() {
        let duration = i64::try_from(duration).unwrap_or(i64::MAX);
        if values[note] != 0.0 {
            let first = (-5_i64).max(-onset);
            let end = (duration + 5).min(i64::try_from(frames).unwrap_or(i64::MAX) - onset);
            for local in first..end {
                let weight = 1.0
                    / ((1.0 + (-(duration - local) as f64).exp()) * (1.0 + (-local as f64).exp()));
                output[(onset + local) as usize] += values[note] * weight as f32;
            }
        }
        onset += duration;
    }
    output
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn valid_tensor(tensor: &DnniTensor) -> bool {
    tensor.channels > 0
        && tensor.channels <= MAXIMUM_ELEMENTS
        && tensor.frames <= MAXIMUM_ELEMENTS / tensor.channels
        && tensor.values.len() == tensor.frames * tensor.channels
        && tensor.values.iter().all(|value| value.is_finite())
}

/// Nine-network pitch decoder; immutable after loading.
#[derive(Debug, Default)]
pub struct PitchDecoder {
    networks: [DnniInference; 9],
    embedding_channels: usize,
    frame_stride: usize,
    embedding_scale: f32,
    loaded: bool,
}

impl PitchDecoder {
    /// Creates an unloaded decoder; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the nine sub-networks and the decoder configuration.
    ///
    /// The container carries a 12-byte payload (`embeddingChannels: i32`,
    /// `embeddingScale: f32`, `frameStride: i32`) and exactly nine children.
    /// A failed load preserves the current decoder.
    pub fn load(&mut self, reader: &DnniReader, node_index: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes
            .get(node_index)
            .filter(|node| DECODER_TYPES.contains(&node.type_id))
        else {
            return Err(failure("expected the gen5 nine-network decoder."));
        };
        if node.children.len() != 9 {
            return Err(failure("expected the gen5 nine-network decoder."));
        }
        let payload = reader.payload(node_index);
        if payload.len() != 12 {
            return Err(failure("embedding and frame-group parameters are missing."));
        }
        let channels = read_i32(payload, 0);
        let scale = f32::from_bits(read_i32(payload, 4) as u32);
        let stride = read_i32(payload, 8);
        if channels < 2
            || channels % 2 != 0
            || channels > 4096
            || stride <= 0
            || stride > 4096
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(failure(
                "invalid sinusoidal embedding or frame-group dimensions.",
            ));
        }
        let children = node.children.clone();
        let mut networks: [DnniInference; 9] = Default::default();
        for (index, child) in children.into_iter().enumerate() {
            if let Err(error) = networks[index].load(reader, child) {
                return Err(failure(format!("network {index}: {error}")));
            }
        }
        self.networks = networks;
        self.embedding_channels = channels as usize;
        self.embedding_scale = scale;
        self.frame_stride = stride as usize;
        self.loaded = true;
        Ok(())
    }

    /// Executes two-pass neural pitch decoding into one scalar per frame.
    ///
    /// `context` is the frame conditioning tensor, `scalar_context` the
    /// feed-forward branch, `noise`/`controls`/`vibrato_control` the per-frame
    /// streams, and `notes` the note metadata (populated with predicted tilt
    /// and shift). `output` is only assigned on success.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn run(
        &self,
        context: &DnniTensor,
        scalar_context: &DnniTensor,
        noise: &[f32],
        controls: &DnniTensor,
        vibrato_control: &[f32],
        frontend: &PitchFeatures,
        notes: &mut PitchDecoderNotes<'_>,
        output: &mut DnniTensor,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        if !self.loaded {
            return Err(failure("model has not been loaded."));
        }
        if !valid_tensor(context)
            || context.frames == 0
            || !valid_tensor(scalar_context)
            || scalar_context.frames != context.frames
            || !valid_tensor(controls)
            || controls.frames != context.frames
            || controls.channels != 3
            || noise.len() != context.frames
            || vibrato_control.len() != context.frames
            || !noise.iter().all(|value| value.is_finite())
            || !vibrato_control.iter().all(|value| value.is_finite())
        {
            return Err(failure("context, noise, and control frames do not match."));
        }
        let note_count = notes.frame_counts.len();
        if notes.midi_pitch.len() != note_count
            || notes.vowel_frames.len() != context.frames
            || notes
                .requested_tilt
                .is_some_and(|tilt| tilt.len() != note_count)
            || notes
                .requested_shift
                .is_some_and(|shift| shift.len() != note_count)
            || !notes.midi_pitch.iter().all(|value| value.is_finite())
        {
            return Err(failure(
                "note durations, pitches, and vowel mask do not match.",
            ));
        }
        let mut counted_frames = 0_usize;
        for &count in notes.frame_counts {
            if count > context.frames - counted_frames {
                return Err(failure("note durations extend beyond the input frames."));
            }
            counted_frames += count;
        }
        let frames = context.frames;
        let groups = frames.div_ceil(self.frame_stride);
        let condition_channels = context.channels + self.embedding_channels * 3 + self.frame_stride;
        if condition_channels > MAXIMUM_ELEMENTS
            || groups > MAXIMUM_ELEMENTS / condition_channels
            || context.channels > MAXIMUM_ELEMENTS / (frames * 2)
        {
            return Err(failure("working tensors exceed the memory limit."));
        }

        let mut run_network = |index: usize,
                               input: &DnniTensor,
                               condition: Option<&DnniTensor>,
                               result: &mut DnniTensor| {
            let status = self.networks[index].run_with_cache(
                input,
                result,
                condition,
                None,
                cancel,
                crate::reborrow_stats(&mut statistics),
            );
            if let Err(error) = status {
                return Err(failure(format!("network {index}: {error}")));
            }
            Ok(())
        };

        // Pass 1a: scalar projection pooled per group with control embeddings.
        let mut scalar = DnniTensor::default();
        run_network(0, scalar_context, None, &mut scalar)?;
        if scalar.frames != frames || scalar.channels != 1 {
            return Err(failure(
                "initial projection must produce one scalar per frame.",
            ));
        }
        let mut grouped_scalar = DnniTensor {
            frames: groups,
            channels: self.frame_stride,
            values: vec![0.0; groups * self.frame_stride],
        };
        let mut pooled_context = DnniTensor {
            frames: groups,
            channels: context.channels,
            values: vec![0.0; groups * context.channels],
        };
        let mut condition = DnniTensor {
            frames: groups,
            channels: condition_channels,
            values: vec![0.0; groups * condition_channels],
        };
        let divisor = self.frame_stride as f32;
        let embedding_half = self.embedding_channels / 2;
        for group in 0..groups {
            if cancel.is_cancelled() {
                return Err(failure("cancelled."));
            }
            let mut control_sum = [0.0_f32; 3];
            for local in 0..self.frame_stride {
                // The last incomplete group repeats its final frame.
                let frame = (group * self.frame_stride + local).min(frames - 1);
                grouped_scalar.values[group * self.frame_stride + local] = scalar.values[frame];
                condition.values[group * condition_channels
                    + context.channels
                    + self.embedding_channels * 3
                    + local] = noise[frame];
                for channel in 0..context.channels {
                    pooled_context.values[group * context.channels + channel] +=
                        context.values[frame * context.channels + channel];
                }
                for (control, sum) in control_sum.iter_mut().enumerate() {
                    *sum += controls.values[frame * 3 + control];
                }
            }
            for channel in 0..context.channels {
                pooled_context.values[group * context.channels + channel] /= divisor;
                condition.values[group * condition_channels + channel] =
                    pooled_context.values[group * context.channels + channel];
            }
            for (control, sum) in control_sum.iter().enumerate() {
                let position = sum * self.embedding_scale / divisor;
                for channel in 0..embedding_half {
                    // -2 * log2(10000), as stored by the original encoder.
                    let phase = (channel as f32 * -26.575_424_f32 / self.embedding_channels as f32)
                        .exp2()
                        * position;
                    let base = group * condition_channels
                        + context.channels
                        + control * self.embedding_channels;
                    condition.values[base + channel] = phase.sin();
                    condition.values[base + channel + embedding_half] = phase.cos();
                }
            }
        }

        // Pass 1b: coarse residual plus preliminary pitch head.
        let mut coarse = DnniTensor::default();
        run_network(5, &grouped_scalar, Some(&condition), &mut coarse)?;
        if coarse.frames != groups || coarse.channels != context.channels {
            return Err(failure(
                "first residual pass changed the expected frame groups or context width.",
            ));
        }
        let mut preliminary = DnniTensor::default();
        let upsampled = repeat_frames(&coarse, frames, self.frame_stride);
        run_network(7, &upsampled, None, &mut preliminary)?;
        if preliminary.frames != frames || preliminary.channels != 1 {
            return Err(failure(
                "first pitch head must produce one scalar per frame.",
            ));
        }
        let mut preliminary_midi = Vec::new();
        frontend.denormalize_pitch(&preliminary.values, &mut preliminary_midi)?;

        // Analytical extraction: median slope (tilt) and 80th percentile
        // (shift) on vowel frames, with tanh-weighted user overrides.
        let mut predicted_tilt = vec![0.0_f32; note_count];
        let mut predicted_shift = vec![0.0_f32; note_count];
        let mut desired_tilt = vec![0.0_f32; note_count];
        let mut desired_shift = vec![0.0_f32; note_count];
        let mut tilt_weight = vec![0.0_f32; note_count];
        let mut shift_weight = vec![0.0_f32; note_count];
        let mut onset = 0_usize;
        for note in 0..note_count {
            let mut differences = Vec::new();
            let mut pitches = Vec::new();
            for frame in onset..onset + notes.frame_counts[note] {
                if notes.vowel_frames[frame] != 0 {
                    differences.push(if frame == 0 {
                        0.0
                    } else {
                        preliminary_midi[frame] - preliminary_midi[frame - 1]
                    });
                    pitches.push(preliminary_midi[frame]);
                }
            }
            predicted_tilt[note] =
                (differences.len() as f64 * 0.05 * f64::from(quantile(differences, 0.5))) as f32;
            predicted_shift[note] = if pitches.is_empty() {
                0.0
            } else {
                quantile(pitches, 0.8) - notes.midi_pitch[note]
            };
            if let Some(requested) = notes.requested_tilt {
                if !requested[note].is_nan() {
                    desired_tilt[note] = requested[note];
                    tilt_weight[note] =
                        1.5 * ((requested[note] - predicted_tilt[note]).abs() * 10.0).tanh();
                }
            }
            if let Some(requested) = notes.requested_shift {
                if !requested[note].is_nan() {
                    desired_shift[note] = requested[note];
                    shift_weight[note] =
                        1.5 * ((requested[note] - predicted_shift[note]).abs() * 5.0).tanh();
                }
            }
            onset += notes.frame_counts[note];
        }
        let expanded_tilt = expand_note_values(notes.frame_counts, &desired_tilt, frames);
        let expanded_shift = expand_note_values(notes.frame_counts, &desired_shift, frames);
        let expanded_tilt_weight = expand_note_values(notes.frame_counts, &tilt_weight, frames);
        let expanded_shift_weight = expand_note_values(notes.frame_counts, &shift_weight, frames);

        // Pass 2a: pooled vibrato/tilt/shift control embeddings.
        let mut pooled_vibrato = DnniTensor {
            frames: groups,
            channels: 1,
            values: vec![0.0; groups],
        };
        let mut pooled_tilt = DnniTensor {
            frames: groups,
            channels: 1,
            values: vec![0.0; groups],
        };
        let mut pooled_shift = DnniTensor {
            frames: groups,
            channels: 1,
            values: vec![0.0; groups],
        };
        let mut pooled_tilt_weight = vec![0.0_f32; groups];
        let mut pooled_shift_weight = vec![0.0_f32; groups];
        for group in 0..groups {
            for local in 0..self.frame_stride {
                let frame = (group * self.frame_stride + local).min(frames - 1);
                pooled_vibrato.values[group] += vibrato_control[frame];
                pooled_tilt.values[group] += expanded_tilt[frame];
                pooled_shift.values[group] += expanded_shift[frame];
                pooled_tilt_weight[group] += expanded_tilt_weight[frame];
                pooled_shift_weight[group] += expanded_shift_weight[frame];
            }
            pooled_vibrato.values[group] =
                (pooled_vibrato.values[group] / divisor).clamp(-0.999, 0.999);
            pooled_tilt.values[group] /= divisor;
            pooled_shift.values[group] /= divisor;
            pooled_tilt_weight[group] /= divisor;
            pooled_shift_weight[group] /= divisor;
        }
        let mut vibrato_embedding = DnniTensor::default();
        let mut tilt_embedding = DnniTensor::default();
        let mut shift_embedding = DnniTensor::default();
        run_network(1, &pooled_vibrato, None, &mut vibrato_embedding)?;
        run_network(2, &pooled_tilt, None, &mut tilt_embedding)?;
        run_network(3, &pooled_shift, None, &mut shift_embedding)?;
        if vibrato_embedding.frames != groups
            || vibrato_embedding.channels != context.channels
            || tilt_embedding.frames != groups
            || tilt_embedding.channels != context.channels
            || shift_embedding.frames != groups
            || shift_embedding.channels != context.channels
        {
            return Err(failure("control embeddings must match the context width."));
        }

        // Pass 2b: correction network fusing pooled context and controls.
        let mut correction_input = DnniTensor {
            frames: groups,
            channels: context.channels * 2,
            values: vec![0.0; groups * context.channels * 2],
        };
        for group in 0..groups {
            for channel in 0..context.channels {
                let index = group * context.channels + channel;
                correction_input.values[group * correction_input.channels + channel] =
                    pooled_context.values[index];
                correction_input.values
                    [group * correction_input.channels + context.channels + channel] =
                    vibrato_embedding.values[index]
                        + pooled_tilt_weight[group] * tilt_embedding.values[index]
                        + pooled_shift_weight[group] * shift_embedding.values[index];
            }
        }
        let mut correction = DnniTensor::default();
        run_network(4, &correction_input, None, &mut correction)?;
        if correction.frames != groups || correction.channels != context.channels {
            return Err(failure("control correction must match the pooled context."));
        }
        for group in 0..groups {
            for channel in 0..context.channels {
                let index = group * context.channels + channel;
                condition.values[group * condition.channels + channel] =
                    pooled_context.values[index] + correction.values[index];
            }
        }

        // Pass 2c: refined residual plus final pitch head.
        let mut refined = DnniTensor::default();
        run_network(6, &coarse, Some(&condition), &mut refined)?;
        if refined.frames != groups || refined.channels != context.channels {
            return Err(failure(
                "second residual pass changed the expected frame groups or context width.",
            ));
        }
        let mut result = DnniTensor::default();
        let upsampled = repeat_frames(&refined, frames, self.frame_stride);
        run_network(8, &upsampled, None, &mut result)?;
        if result.frames != frames || result.channels != 1 {
            return Err(failure(
                "final pitch head must produce one scalar per frame.",
            ));
        }
        *output = result;
        notes.predicted_tilt = predicted_tilt;
        notes.predicted_shift = predicted_shift;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_selects_sorted_positions() {
        assert_eq!(quantile(vec![5.0, 1.0, 4.0, 2.0, 3.0], 0.5), 3.0);
        assert_eq!(quantile(vec![5.0, 1.0, 4.0, 2.0, 3.0], 0.8), 4.0);
        assert_eq!(quantile(vec![7.0], 0.8), 7.0);
        assert_eq!(quantile(Vec::new(), 0.5), 0.0);
    }

    #[test]
    fn repeat_frames_tiles_each_group() {
        let input = DnniTensor {
            frames: 2,
            channels: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
        };
        let output = repeat_frames(&input, 5, 3);
        assert_eq!(output.frames, 5);
        assert_eq!(output.channels, 2);
        assert_eq!(
            output.values,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn expand_note_values_matches_hand_computed_taper() {
        // One 4-frame note with value 2: at local frame 2 the weight is
        // 1 / (1 + e^-2)^2 ~= 0.7758039, so the frame reads ~= 1.5516078.
        let expanded = expand_note_values(&[4], &[2.0], 4);
        assert!(
            (expanded[2] - 1.551_607_8).abs() < 1e-6,
            "got {}",
            expanded[2]
        );
        // Zero values contribute nothing.
        assert_eq!(expand_note_values(&[4], &[0.0], 4), vec![0.0; 4]);
        // Influence reaches five frames past a 1-frame note (local < 1 + 5).
        let extended = expand_note_values(&[1], &[1.0], 8);
        assert!(extended[1] > 0.0 && extended[5] > 0.0 && extended[6] == 0.0);
    }

    #[test]
    fn loader_rejects_a_non_decoder_node() {
        let reader = DnniReader::from_bytes(vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .expect("blob must parse");
        assert_eq!(
            PitchDecoder::new()
                .load(&reader, 0)
                .unwrap_err()
                .to_string(),
            "Pitch decoder: expected the gen5 nine-network decoder."
        );
    }

    #[test]
    fn unloaded_decoder_rejects_run() {
        let context = DnniTensor {
            frames: 1,
            channels: 1,
            values: vec![0.0],
        };
        let mut notes = PitchDecoderNotes::default();
        let mut output = DnniTensor::default();
        assert_eq!(
            PitchDecoder::new()
                .run(
                    &context,
                    &context,
                    &[0.0],
                    &DnniTensor {
                        frames: 1,
                        channels: 3,
                        values: vec![0.0; 3]
                    },
                    &[0.0],
                    &PitchFeatures::new(),
                    &mut notes,
                    &mut output,
                    &CancelToken::new(),
                    None,
                )
                .unwrap_err()
                .to_string(),
            "Pitch decoder: model has not been loaded."
        );
    }
}
