//! Native-to-project resampling for neural phrase audio.
//!
//! The vocoder renders at its native rate (48000 Hz); projects render at any
//! rate, so every phrase is resampled with a fixed-ratio FFT (sinc) resampler
//! from the `rubato` crate, per `ROADMAP.md`. This replaces C++
//! `juce::WindowedSincInterpolator` (whose 100-sample input latency the engine
//! primes away); here the chunk loop drops the resampler's reported output
//! delay from the concatenated output, so the result starts at the input start
//! with no placement correction. The impulse test below pins that alignment.
//!
//! The loop is written out instead of using
//! [`rubato::Resampler::process_all_into_buffer`] because that helper only
//! shifts `delay` frames to the front, leaving a stale duplicate behind
//! whenever one chunk's output exceeds the delay (measured: a repeated impulse
//! 21 frames later on 48000 -> 8000 Hz).

use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Indexing, Resampler};

use crate::BackendError;

/// Resamples mono `input` from `input_rate` to `output_rate`.
///
/// An equal-rate pair returns the input unchanged (no filter, no delay). The
/// output holds exactly `ceil(len * output_rate / input_rate)` frames, aligned
/// to the input start. Errors carry the underlying resampler failure text.
pub fn resample_mono(
    input: &[f32],
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f32>, BackendError> {
    if input_rate == output_rate {
        return Ok(input.to_vec());
    }
    if input_rate == 0 || output_rate == 0 {
        return Err(BackendError(
            "the voice renderer returned invalid audio.".into(),
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        output_rate as usize,
        1024,
        1,
        FixedSync::Input,
    )
    .map_err(|error| BackendError(error.to_string()))?;
    let delay = resampler.output_delay();
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_max();

    let mut raw = Vec::new();
    let mut offset = 0_usize;
    while offset < input.len() {
        let have = (input.len() - offset).min(chunk_in);
        let mut padded = vec![0.0_f32; chunk_in];
        padded[..have].copy_from_slice(&input[offset..offset + have]);
        let input_channels = [padded];
        let input_buffer = SequentialSliceOfVecs::new(&input_channels, 1, chunk_in)
            .map_err(|error| BackendError(error.to_string()))?;
        let mut output_channels = [vec![0.0_f32; chunk_out]];
        let mut output_buffer = SequentialSliceOfVecs::new_mut(&mut output_channels, 1, chunk_out)
            .map_err(|error| BackendError(error.to_string()))?;
        let indexing = Indexing::new().partial_len(have);
        let (_, produced) = resampler
            .process_into_buffer(&input_buffer, &mut output_buffer, Some(&indexing))
            .map_err(|error| BackendError(error.to_string()))?;
        raw.extend_from_slice(&output_channels[0][..produced]);
        offset += have;
    }
    // Drop the filter's startup transient; the tail ring-out past the last
    // input frame is cut, as with the C++ interpolator's finite impulse.
    let aligned = raw.get(delay..).unwrap_or(&[]);
    let expected =
        (input.len() as f64 * f64::from(output_rate) / f64::from(input_rate)).ceil() as usize;
    let mut output = aligned.to_vec();
    output.resize(expected, 0.0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders a sine at `frequency`/`rate` for `seconds`.
    fn sine(frequency: f64, rate: u32, seconds: f64) -> Vec<f32> {
        let count = (seconds * f64::from(rate)).round() as usize;
        (0..count)
            .map(|index| {
                (2.0 * std::f64::consts::PI * frequency * index as f64 / f64::from(rate)) as f32
            })
            .map(f32::sin)
            .collect()
    }

    #[test]
    fn equal_rates_pass_through_untouched() {
        let input = sine(440.0, 48_000, 0.1);
        assert_eq!(resample_mono(&input, 48_000, 48_000).unwrap(), input);
    }

    #[test]
    fn output_length_follows_the_ratio() {
        // 480 samples at 48 kHz down to 8 kHz: exactly 80 frames.
        let input = sine(440.0, 48_000, 0.01);
        let output = resample_mono(&input, 48_000, 8_000).unwrap();
        assert_eq!(output.len(), 80);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn impulse_response_peaks_at_the_first_frame() {
        // A unit impulse at frame 0 must come out at frame 0: no startup delay
        // may leak into the placement math, and no stale duplicate may repeat
        // it later. (`max_by` keeps the last maximum, so a repeat would fail.)
        // The 48000 -> 8000 pair has a half-sample filter delay, so its peak
        // splits evenly across frames 0 and 1; alignment is pinned to ±1 frame.
        let mut input = vec![0.0_f32; 4800];
        input[0] = 1.0;
        for (input_rate, output_rate) in [(48_000, 8_000), (48_000, 44_100), (48_000, 16_000)] {
            let output = resample_mono(&input, input_rate, output_rate).unwrap();
            let (peak, &peak_value) = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .unwrap();
            assert!(
                peak <= 1,
                "{input_rate} -> {output_rate}: impulse peaks at frame {peak}"
            );
            assert!(
                ((output[0].abs() - peak_value.abs()) / peak_value.abs()).abs() < 0.01,
                "{input_rate} -> {output_rate}: frame 0 holds the peak"
            );
        }
    }

    #[test]
    fn sine_keeps_its_frequency_through_the_resampler() {
        // 440 Hz at 48 kHz down to 8 kHz: zero crossings over the interior
        // (away from the window edges) still read 440 Hz.
        let input = sine(440.0, 48_000, 0.5);
        let output = resample_mono(&input, 48_000, 8_000).unwrap();
        let interior = &output[80..output.len() - 80];
        let crossings = interior
            .windows(2)
            .filter(|w| w[0] <= 0.0 && w[1] > 0.0)
            .count();
        let seconds = interior.len() as f64 / 8_000.0;
        let measured = crossings as f64 / seconds;
        assert!(
            (measured - 440.0).abs() < 5.0,
            "measured {measured} Hz over {seconds:.3} s"
        );
    }

    #[test]
    fn bad_rates_are_rejected() {
        assert!(resample_mono(&[0.0; 8], 0, 8_000).is_err());
    }
}
