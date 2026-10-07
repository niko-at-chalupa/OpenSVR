use crate::{BLICKS_PER_QUARTER, Blick};

/// A tempo marker: `bpm` applies from `position` until the next marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoChange {
    pub position: Blick,
    pub bpm: f64,
}

/// A meter marker. `bar` is zero-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignature {
    pub bar: u32,
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            bar: 0,
            numerator: 4,
            denominator: 4,
        }
    }
}

impl TimeSignature {
    /// Whether the numerator and denominator are in range and the denominator is a power of two.
    pub fn is_valid(self) -> bool {
        (1..=256).contains(&self.numerator)
            && (1..=256).contains(&self.denominator)
            && self.denominator.is_power_of_two()
    }

    /// Length of one bar in blicks.
    pub fn bar_length(self) -> Blick {
        Blick::from(self.numerator) * BLICKS_PER_QUARTER * 4 / Blick::from(self.denominator)
    }
}

/// Converts between musical time (blicks) and wall-clock seconds.
///
/// Invariant: both lists are non-empty, sorted, and start at position / bar 0.
/// [`TempoMap::new`] establishes it, so lookups never have to handle an empty map.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoMap {
    tempos: Vec<TempoChange>,
    time_signatures: Vec<TimeSignature>,
}

impl Default for TempoMap {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

impl TempoMap {
    const DEFAULT_BPM: f64 = 120.0;

    /// Builds a map, sorting the markers and inserting 120 BPM and 4/4 at the start if missing.
    pub fn new(mut tempos: Vec<TempoChange>, mut time_signatures: Vec<TimeSignature>) -> Self {
        tempos.sort_by_key(|tempo| tempo.position);
        if tempos.first().is_none_or(|tempo| tempo.position > 0) {
            tempos.insert(
                0,
                TempoChange {
                    position: 0,
                    bpm: Self::DEFAULT_BPM,
                },
            );
        }
        time_signatures.sort_by_key(|signature| signature.bar);
        if time_signatures
            .first()
            .is_none_or(|signature| signature.bar > 0)
        {
            time_signatures.insert(0, TimeSignature::default());
        }
        Self {
            tempos,
            time_signatures,
        }
    }

    pub fn tempos(&self) -> &[TempoChange] {
        &self.tempos
    }

    pub fn time_signatures(&self) -> &[TimeSignature] {
        &self.time_signatures
    }

    /// Tempo in effect at `position`.
    pub fn tempo_at(&self, position: Blick) -> f64 {
        let next = self
            .tempos
            .partition_point(|tempo| tempo.position <= position);
        self.tempos[next.saturating_sub(1)].bpm
    }

    /// Elapsed seconds at `position`, integrating across tempo segments.
    pub fn blick_to_seconds(&self, position: Blick) -> f64 {
        let mut seconds = 0.0;
        let mut cursor: Blick = 0;
        let mut bpm = self.tempos[0].bpm;
        for tempo in self
            .tempos
            .iter()
            .take_while(|tempo| tempo.position <= position)
        {
            seconds += (tempo.position - cursor) as f64 * seconds_per_blick(bpm);
            cursor = tempo.position;
            bpm = tempo.bpm;
        }
        seconds + (position - cursor) as f64 * seconds_per_blick(bpm)
    }

    /// Inverse of [`blick_to_seconds`](Self::blick_to_seconds), rounded to the nearest blick.
    ///
    /// Non-finite input maps to 0; out-of-range values saturate.
    pub fn seconds_to_blick(&self, seconds: f64) -> Blick {
        if !seconds.is_finite() {
            return 0;
        }
        let mut elapsed = 0.0;
        let mut cursor: Blick = 0;
        let mut bpm = self.tempos[0].bpm;
        for tempo in &self.tempos {
            let segment = (tempo.position - cursor) as f64 * seconds_per_blick(bpm);
            if elapsed + segment > seconds {
                break;
            }
            elapsed += segment;
            cursor = tempo.position;
            bpm = tempo.bpm;
        }
        let remaining = ((seconds - elapsed) / seconds_per_blick(bpm)).round() as Blick;
        cursor.saturating_add(remaining)
    }
}

fn seconds_per_blick(bpm: f64) -> f64 {
    60.0 / (bpm * BLICKS_PER_QUARTER as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(tempos: &[(Blick, f64)]) -> TempoMap {
        let tempos = tempos
            .iter()
            .map(|&(position, bpm)| TempoChange { position, bpm })
            .collect();
        TempoMap::new(tempos, Vec::new())
    }

    #[test]
    fn default_map_is_120_bpm_in_four_four() {
        let map = TempoMap::default();
        assert_eq!(map.tempo_at(0), 120.0);
        assert_eq!(map.time_signatures(), [TimeSignature::default()]);
        assert!((map.blick_to_seconds(BLICKS_PER_QUARTER) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn integrates_across_tempo_changes() {
        // One quarter at 120 BPM (0.5 s), then one quarter at 60 BPM (1 s).
        let map = map(&[(0, 120.0), (BLICKS_PER_QUARTER, 60.0)]);
        assert!((map.blick_to_seconds(2 * BLICKS_PER_QUARTER) - 1.5).abs() < 1e-9);
        assert_eq!(map.tempo_at(BLICKS_PER_QUARTER - 1), 120.0);
        assert_eq!(map.tempo_at(BLICKS_PER_QUARTER), 60.0);
    }

    #[test]
    fn seconds_round_trip() {
        let map = map(&[(0, 90.0), (3 * BLICKS_PER_QUARTER, 140.0)]);
        for position in [0, 1, BLICKS_PER_QUARTER, 5 * BLICKS_PER_QUARTER + 12_345] {
            let back = map.seconds_to_blick(map.blick_to_seconds(position));
            assert!((back - position).abs() <= 1, "{position} -> {back}");
        }
    }

    #[test]
    fn inserts_missing_start_markers() {
        let map = map(&[(BLICKS_PER_QUARTER, 80.0)]);
        assert_eq!(
            map.tempos()[0],
            TempoChange {
                position: 0,
                bpm: 120.0
            }
        );
        assert_eq!(map.tempos().len(), 2);
    }

    #[test]
    fn bar_length_follows_the_meter() {
        let waltz = TimeSignature {
            bar: 0,
            numerator: 3,
            denominator: 4,
        };
        let compound = TimeSignature {
            bar: 0,
            numerator: 6,
            denominator: 8,
        };
        assert_eq!(waltz.bar_length(), 3 * BLICKS_PER_QUARTER);
        assert_eq!(compound.bar_length(), 3 * BLICKS_PER_QUARTER);
        assert!(
            !TimeSignature {
                bar: 0,
                numerator: 4,
                denominator: 3
            }
            .is_valid()
        );
    }
}
