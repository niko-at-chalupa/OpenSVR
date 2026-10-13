//! Band-split spectral prediction heads (`VocoderSpectralHeads`).
//!
//! Ports OpenSV's `src/synthesis/VocoderSpectralHeads.{h,cpp}` (namespace
//! `sv::synthesis`): one sequential network per frequency band predicts three
//! consecutive spectra of its declared width, which are assembled into three
//! full-width outputs with inclusive crossfade endpoints shared with the
//! neighboring band.

use opensvr_core::CancelToken;
use opensvr_dnni::{DnniCache, DnniInference, DnniReader, DnniRunStatistics, DnniTensor};

use crate::{Result, invalid};

const MAXIMUM_ELEMENTS: usize = 64 * 1024 * 1024;

/// All twelve serialized aliases selecting the same factory.
const SPECTRAL_HEADS_TYPE_IDS: [u64; 12] = [
    0x0e61_00d4_51b6_d124,
    0xb3f0_948e_f978_07c2,
    0x8093_1725_4e8c_dddd,
    0xd1f5_c2de_fa67_db76,
    0xe1ae_8e9e_9aae_0c92,
    0xf6a2_2ab6_62b9_092d,
    0xdf01_8dfa_8da4_4218,
    0x7dc3_20b1_4fdb_c1c0,
    0x4f06_7df8_6e61_07f2,
    0xc3ac_ffc4_cbf2_7405,
    0xc666_b367_9006_cb9f,
    0x2b4a_1a67_01e5_15cb,
];

fn failure(reason: impl Into<String>) -> crate::VoiceError {
    invalid(format!("Vocoder spectral heads: {}", reason.into()))
}

fn read_word(payload: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ])
}

#[derive(Debug)]
struct Band {
    first_bin: usize,
    bin_count: usize,
    fade_in: usize,
    fade_out: usize,
    network: DnniInference,
}

/// Retained per-band inference caches for one phrase.
#[derive(Debug, Default)]
pub struct VocoderSpectralHeadsState {
    /// One cache per band network.
    pub bands: Vec<DnniCache>,
}

impl VocoderSpectralHeadsState {
    /// Byte size of the retained inference snapshots.
    pub fn get_bytes(&self) -> usize {
        self.bands.capacity() * size_of::<DnniCache>()
            + self.bands.iter().map(DnniCache::get_bytes).sum::<usize>()
    }

    /// Resets all band caches.
    pub fn clear(&mut self) {
        for band in &mut self.bands {
            band.clear();
        }
    }
}

/// Band-split spectral predictor producing three full-width spectra.
#[derive(Debug, Default)]
pub struct VocoderSpectralHeads {
    bands: Vec<Band>,
    output_bins: usize,
}

impl VocoderSpectralHeads {
    /// Creates an unloaded head set; [`run`](Self::run) fails until
    /// [`load`](Self::load) succeeds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads the band networks from the spectral-head operator node.
    ///
    /// A failed load preserves the current heads.
    pub fn load(&mut self, reader: &DnniReader, node_index: usize) -> Result<()> {
        let nodes = reader.nodes();
        let Some(node) = nodes
            .get(node_index)
            .filter(|node| SPECTRAL_HEADS_TYPE_IDS.contains(&node.type_id))
        else {
            return Err(failure(
                "the selected node is not the verified spectral-head operator.",
            ));
        };
        let children = node.children.clone();
        let payload = reader.payload(node_index);
        if payload.len() < 8 || read_word(payload, 0) != 3 {
            return Err(failure("the operator must declare three output spectra."));
        }
        let band_count = read_word(payload, 4) as usize;
        if band_count == 0
            || band_count > (payload.len() - 8) / 16
            || payload.len() != 8 + band_count * 16
        {
            return Err(failure(
                "the band count does not match the serialized band descriptors.",
            ));
        }
        if children.len() != 1 || children[0] >= nodes.len() {
            return Err(failure("the operator requires one group of band networks."));
        }
        let group = &nodes[children[0]];
        if group.name != "cmpg1"
            || !reader.payload(children[0]).is_empty()
            || group.children.len() != band_count
        {
            return Err(failure(
                "the network group does not match the declared bands.",
            ));
        }
        let group_children = group.children.clone();
        let mut replacement = Vec::with_capacity(band_count);
        let mut replacement_bins = 0_usize;
        for (index, &network_index) in group_children.iter().enumerate() {
            let offset = 8 + index * 16;
            let first_bin = read_word(payload, offset) as usize;
            let end_bin = read_word(payload, offset + 4) as usize;
            let fade_in = read_word(payload, offset + 8) as usize;
            let fade_out = read_word(payload, offset + 12) as usize;
            if first_bin >= end_bin || end_bin > MAXIMUM_ELEMENTS / 3 {
                return Err(failure("a band has an invalid bin range."));
            }
            let bin_count = end_bin - first_bin;
            if fade_in == 1
                || fade_out == 1
                || fade_in > bin_count
                || fade_out > bin_count - fade_in
            {
                return Err(failure(
                    "band fades must be empty or contain at least two bins, and must not overlap.",
                ));
            }
            if network_index >= nodes.len() || nodes[network_index].name != "modm0" {
                return Err(failure("each band requires a sequential network."));
            }
            let mut network = DnniInference::new();
            network.load(reader, network_index)?;
            replacement_bins = replacement_bins.max(end_bin);
            replacement.push(Band {
                first_bin,
                bin_count,
                fade_in,
                fade_out,
                network,
            });
        }
        self.bands = replacement;
        self.output_bins = replacement_bins;
        Ok(())
    }

    /// Whole-sequence inference: one assembled spectrum per frame in each of
    /// three outputs (these are spectra, not waveform samples).
    ///
    /// `outputs` is only assigned on success.
    pub fn run(
        &self,
        input: &DnniTensor,
        outputs: &mut [DnniTensor; 3],
        mut state: Option<&mut VocoderSpectralHeadsState>,
        cancel: &CancelToken,
        mut statistics: Option<&mut DnniRunStatistics>,
    ) -> Result<()> {
        if cancel.is_cancelled() {
            return Err(invalid("Cancelled"));
        }
        if self.bands.is_empty() {
            return Err(failure("no model has been loaded."));
        }
        if input.frames > MAXIMUM_ELEMENTS / self.output_bins.max(1) {
            return Err(failure(
                "assembled spectra exceed the 64 Mi element limit per output.",
            ));
        }
        if let Some(state) = state.as_mut() {
            state
                .bands
                .resize_with(self.bands.len(), DnniCache::default);
        }
        let mut replacement = [
            DnniTensor {
                frames: input.frames,
                channels: self.output_bins,
                values: vec![0.0; input.frames * self.output_bins],
            },
            DnniTensor {
                frames: input.frames,
                channels: self.output_bins,
                values: vec![0.0; input.frames * self.output_bins],
            },
            DnniTensor {
                frames: input.frames,
                channels: self.output_bins,
                values: vec![0.0; input.frames * self.output_bins],
            },
        ];
        for (band_index, band) in self.bands.iter().enumerate() {
            let mut predicted = DnniTensor::default();
            band.network.run_with_cache(
                input,
                &mut predicted,
                None,
                state.as_mut().map(|state| &mut state.bands[band_index]),
                cancel,
                crate::reborrow_stats(&mut statistics),
            )?;
            if predicted.frames != input.frames || predicted.channels != 3 * band.bin_count {
                return Err(failure(
                    "a band network must preserve frame count and produce three consecutive spectra of its declared width.",
                ));
            }
            // The serialized band network packs complete spectra
            // consecutively. Both crossfade endpoints are included, matching
            // the neighboring band.
            for frame in 0..input.frames {
                for (spectrum, output) in replacement.iter_mut().enumerate() {
                    let source_offset = frame * predicted.channels + spectrum * band.bin_count;
                    let destination_offset = frame * self.output_bins + band.first_bin;
                    for bin in 0..band.bin_count {
                        let mut value = predicted.values[source_offset + bin];
                        if bin < band.fade_in {
                            value *= bin as f32 / (band.fade_in - 1) as f32;
                        } else if bin >= band.bin_count - band.fade_out {
                            value *= (band.bin_count - 1 - bin) as f32 / (band.fade_out - 1) as f32;
                        }
                        let destination = &mut output.values[destination_offset + bin];
                        *destination += value;
                        if !destination.is_finite() {
                            return Err(failure("band accumulation produced a non-finite value."));
                        }
                    }
                }
            }
        }
        *outputs = replacement;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_heads_reject_run() {
        let mut outputs = [
            DnniTensor::default(),
            DnniTensor::default(),
            DnniTensor::default(),
        ];
        assert_eq!(
            VocoderSpectralHeads::new()
                .run(
                    &DnniTensor::default(),
                    &mut outputs,
                    None,
                    &CancelToken::new(),
                    None
                )
                .unwrap_err()
                .to_string(),
            "Vocoder spectral heads: no model has been loaded."
        );
    }

    #[test]
    fn loader_rejects_a_non_operator_node() {
        let bytes = vec![
            0xff, 0x00, 0xca, 0x7f, 0x01, 0x00, 0x00, 0x00, 0xff, 0x40, 0xca, 0x7f, b'm', b'o',
            b'd', b'l', b'0', 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let reader = DnniReader::from_bytes(bytes).expect("blob must parse");
        assert_eq!(
            VocoderSpectralHeads::new()
                .load(&reader, 0)
                .unwrap_err()
                .to_string(),
            "Vocoder spectral heads: the selected node is not the verified spectral-head operator."
        );
    }

    #[test]
    fn fresh_state_holds_no_bytes() {
        assert_eq!(VocoderSpectralHeadsState::default().get_bytes(), 0);
    }
}
