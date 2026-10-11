//! Deterministic Gaussian noise over an exact `std::mt19937` port.
//!
//! Both the pitch and acoustic stages feed a hand-written Box-Muller
//! rejection loop from a `std::mt19937` engine (never
//! `std::normal_distribution` equivalents), so the stream matches the C++
//! engine bit for bit. This module holds the single shared implementation;
//! the pitch stage reseeds at phoneme/note boundaries while the acoustic
//! stage draws one contiguous stream per run.

/// 32-bit Mersenne Twister matching `std::mt19937` bit for bit.
#[derive(Debug, Clone)]
pub(crate) struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub(crate) fn seed(seed: u32) -> Self {
        let mut state = [0_u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253_u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let merged =
                    (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                let mut value = self.state[(i + 397) % 624] ^ (merged >> 1);
                if merged & 1 != 0 {
                    value ^= 0x9908_b0df;
                }
                self.state[i] = value;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }
}

/// `std::ilogb` for `f32`: unbiased exponent of the normalized mantissa.
///
/// `cast_possible_wrap` is allowed: the masked exponent and the
/// subnormal leading-zero count provably fit in `i32`.
#[allow(clippy::cast_possible_wrap)]
fn ilogb(value: f32) -> i32 {
    let bits = value.to_bits();
    let biased = ((bits >> 23) & 0xff) as i32;
    if biased == 0 {
        let mantissa = bits & 0x7f_ffff;
        if mantissa == 0 {
            return i32::MIN;
        }
        -127 - (mantissa.leading_zeros() as i32 - 9)
    } else {
        biased - 127
    }
}

/// Exact `std::scalbn`: `value * 2^exp` with a single final rounding.
fn scalbn(value: f32, exp: i32) -> f32 {
    (f64::from(value) * f64::from(2).powi(exp)) as f32
}

/// Gaussian noise source with the C++ `GaussianNoise` spare semantics.
///
/// A boundary reseed resets only the MT19937 engine: a cached spare sample
/// from the pair generated before the boundary is still served next.
#[derive(Debug, Clone)]
pub(crate) struct GaussianNoise {
    engine: Mt19937,
    spare: f32,
    has_spare: bool,
}

impl GaussianNoise {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            engine: Mt19937::seed(seed),
            spare: 0.0,
            has_spare: false,
        }
    }

    pub(crate) fn reseed(&mut self, seed: u32) {
        self.engine = Mt19937::seed(seed);
    }

    pub(crate) fn next(&mut self) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            let first = self.engine.next_u32() as f32 * f32::from_bits(0x3000_0000) - 1.0;
            let second = self.engine.next_u32() as f32 * f32::from_bits(0x3000_0000) - 1.0;
            let mut squared = first * first + second * second;
            if squared >= 1.0 || first == 0.0 || second == 0.0 {
                continue;
            }
            let mut first = first;
            let mut second = second;
            let logarithm = if squared <= 0.0001 {
                let exponent = ilogb(first.abs().max(second.abs()));
                first = scalbn(first, -exponent);
                second = scalbn(second, -exponent);
                squared = first * first + second * second;
                squared.ln() + exponent as f32 * 1.386_294_4_f32
            } else {
                squared.ln()
            };
            let scale = (-2.0 * logarithm / squared).sqrt();
            self.spare = second * scale;
            self.has_spare = true;
            return first * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_reference_stream() {
        // First outputs of MT19937 seeded 5489 (the language default, also
        // used by the acoustic model): classic test vector.
        let mut engine = Mt19937::seed(5489);
        assert_eq!(engine.next_u32(), 3_499_211_612);
        assert_eq!(engine.next_u32(), 581_869_302);
        assert_eq!(engine.next_u32(), 3_890_346_734);
    }

    #[test]
    fn gaussian_noise_is_deterministic_per_seed() {
        let draw = |seed: u32| {
            let mut noise = GaussianNoise::new(seed);
            (0..16).map(|_| noise.next()).collect::<Vec<_>>()
        };
        assert_eq!(draw(123), draw(123));
        assert_eq!(draw(5489), draw(5489));
        assert_ne!(draw(123), draw(124));
        for value in draw(123) {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn reseed_keeps_a_cached_spare() {
        // Drain one pair so a spare is cached, reseed, and check the spare
        // is still served first: the post-reseed stream differs from a
        // freshly seeded engine at the first draw but rejoins it after.
        let mut noise = GaussianNoise::new(123);
        let first = noise.next();
        let spare = noise.next();
        let mut reseeded = GaussianNoise::new(123);
        let _ = reseeded.next();
        reseeded.reseed(999);
        assert_eq!(reseeded.next(), spare);
        assert_eq!(first, GaussianNoise::new(123).next());
    }
}
