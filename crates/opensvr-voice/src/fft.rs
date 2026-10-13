//! JUCE-compatible FFT wrappers over `rustfft`.
//!
//! Ports the exact scaling conventions of OpenSV's `juce::dsp::FFT`
//! (third-party JUCE `frequency/juce_FFT.cpp`, fallback engine — the build
//! used for golden dumps): the complex forward transform is unscaled
//! (negative exponent), the complex inverse applies `1/N`, the real forward
//! packs reals as complex and runs the unscaled forward, and the real
//! inverse reconstructs negative frequencies by conjugation before the
//! scaled inverse. `rustfft` itself leaves both directions unscaled, so the
//! `1/N` on every inverse path here is what keeps the port bit-compatible
//! with the C++ filter-bank and excitation stages.
//!
//! The real layouts mirror JUCE's interleaved `[re, im]` pairs: bin `k` of
//! size `N` lives at `[2k, 2k + 1]` for `k` in `0..N`.

use rustfft::{FftPlanner, num_complex::Complex32};

/// Cached forward/inverse plans for one power-of-two size.
pub(crate) struct JuceFft {
    size: usize,
    forward: std::sync::Arc<dyn rustfft::Fft<f32>>,
    inverse: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl JuceFft {
    /// Builds the plan pair for `size` (must be a positive power of two).
    pub(crate) fn new(size: usize) -> Self {
        assert!(
            size.is_power_of_two() && size > 0,
            "FFT size must be a power of two"
        );
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        Self {
            size,
            forward,
            inverse,
        }
    }

    /// Out-of-place complex forward transform (unscaled, like JUCE).
    pub(crate) fn forward_complex(&self, input: &[Complex32]) -> Vec<Complex32> {
        assert_eq!(input.len(), self.size, "complex forward input length");
        let mut buffer = input.to_vec();
        self.forward.process(&mut buffer);
        buffer
    }

    /// Out-of-place complex inverse transform (with JUCE's `1/N`, in `f32`).
    pub(crate) fn inverse_complex(&self, input: &[Complex32]) -> Vec<Complex32> {
        assert_eq!(input.len(), self.size, "complex inverse input length");
        let mut buffer = input.to_vec();
        self.inverse.process(&mut buffer);
        let scale = 1.0f32 / self.size as f32;
        for value in &mut buffer {
            *value *= scale;
        }
        buffer
    }

    /// Real forward transform: the complex DFT of the real sequence
    /// (unscaled). Matches JUCE's `performRealOnlyForwardTransform`, which
    /// packs `{sample, 0}` and runs the complex forward.
    pub(crate) fn real_forward(&self, input: &[f32]) -> Vec<Complex32> {
        assert_eq!(input.len(), self.size, "real forward input length");
        let packed: Vec<Complex32> = input
            .iter()
            .map(|sample| Complex32::new(*sample, 0.0))
            .collect();
        self.forward_complex(&packed)
    }

    /// Real inverse transform from the non-negative bins.
    ///
    /// Only the first `size / 2 + 1` entries are read (like JUCE's
    /// `performRealOnlyInverseTransform`); negative frequencies are rebuilt
    /// with the fallback loop `input[i] = conj(input[size - i])` for
    /// `i in size/2..size`, then the scaled complex inverse is applied and
    /// the real parts returned.
    pub(crate) fn real_inverse(&self, spectrum: &[Complex32]) -> Vec<f32> {
        assert_eq!(spectrum.len(), self.size, "real inverse spectrum length");
        let mut full = spectrum.to_vec();
        for index in (self.size / 2)..self.size {
            full[index] = full[self.size - index].conj();
        }
        let time = self.inverse_complex(&full);
        time.iter().map(|value| value.re).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_unscaled_and_inverse_applies_one_over_n() {
        // Impulse forward is all ones (no scaling); the round trip is exact.
        let fft = JuceFft::new(8);
        let mut impulse = vec![Complex32::new(0.0, 0.0); 8];
        impulse[0] = Complex32::new(1.0, 0.0);
        let spectrum = fft.forward_complex(&impulse);
        for bin in &spectrum {
            assert!((bin.re - 1.0).abs() < 1e-6 && bin.im.abs() < 1e-6);
        }
        let back = fft.inverse_complex(&spectrum);
        for (index, value) in back.iter().enumerate() {
            let want = if index == 0 { 1.0 } else { 0.0 };
            assert!(
                (value.re - want).abs() < 1e-6 && value.im.abs() < 1e-6,
                "bin {index}: {value}"
            );
        }
    }

    #[test]
    fn real_round_trip_recovers_sine_and_impulse() {
        for size in [16, 256, 1024] {
            let fft = JuceFft::new(size);
            // Sine at bin 3.
            let signal: Vec<f32> = (0..size)
                .map(|index| (2.0 * std::f32::consts::PI * 3.0 * index as f32 / size as f32).sin())
                .collect();
            let spectrum = fft.real_forward(&signal);
            // Conjugate symmetry of a real DFT.
            for index in 1..size {
                let mirror = spectrum[size - index].conj();
                assert!(
                    (spectrum[index].re - mirror.re).abs() < 1e-4
                        && (spectrum[index].im - mirror.im).abs() < 1e-4,
                    "size {size} bin {index}"
                );
            }
            // Sine energy lands on bins 3 and N-3 with magnitude N/2 (unscaled).
            assert!((spectrum[3].norm() - size as f32 / 2.0).abs() < 1e-2);
            let back = fft.real_inverse(&spectrum);
            for (index, (&got, &want)) in back.iter().zip(&signal).enumerate() {
                assert!((got - want).abs() < 1e-4, "size {size} sample {index}");
            }
            // Impulse round trip through the real path.
            let mut impulse = vec![0.0f32; size];
            impulse[0] = 1.0;
            let back = fft.real_inverse(&fft.real_forward(&impulse));
            for (index, &got) in back.iter().enumerate() {
                let want = if index == 0 { 1.0 } else { 0.0 };
                assert!((got - want).abs() < 1e-5, "size {size} impulse {index}");
            }
        }
    }

    #[test]
    fn negative_frequencies_are_rebuilt_by_conjugation() {
        // Cover the JUCE fallback tail loop explicitly: only the first
        // N/2+1 bins feed the inverse; the rest are conjugated mirrors.
        let fft = JuceFft::new(8);
        let mut spectrum = vec![Complex32::new(0.0, 0.0); 8];
        spectrum[0] = Complex32::new(8.0, 0.0);
        spectrum[1] = Complex32::new(1.0, 2.0);
        let time = fft.real_inverse(&spectrum);
        // DC 8/N = 1 plus the bin-1 cosine of amplitude 2/N.
        for (index, &got) in time.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * index as f32 / 8.0;
            let want = 1.0 + 0.25 * (angle.cos() - 2.0 * angle.sin());
            assert!((got - want).abs() < 1e-5, "sample {index}: {got} vs {want}");
        }
    }
}
