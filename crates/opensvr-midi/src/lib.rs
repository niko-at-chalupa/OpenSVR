//! Standard MIDI File export (format 1, 9600 ticks per quarter note).
//!
//! Track 0 is a conductor track with the project name, tempo changes and meter
//! changes; every project track follows with its notes and lyric events.

use opensvr_core::{BLICKS_PER_QUARTER, Blick, Project, ProjectError, TimeSignature};
use thiserror::Error;

/// High resolution, so fine microtiming survives the trip.
const TICKS_PER_QUARTER: u16 = 9600;
const BLICKS_PER_TICK: Blick = BLICKS_PER_QUARTER / TICKS_PER_QUARTER as Blick;
/// Largest tick that still fits a four-byte variable-length quantity.
const MAX_TICK: u32 = 0x0FFF_FFFF;
const VELOCITY: u8 = 100;

const META_TRACK_NAME: u8 = 0x03;
const META_LYRIC: u8 = 0x05;
const META_TEMPO: u8 = 0x51;
const META_TIME_SIGNATURE: u8 = 0x58;

#[derive(Debug, Error)]
pub enum MidiError {
    #[error(transparent)]
    Project(#[from] ProjectError),
    #[error("the project has too many tracks for a MIDI file")]
    TooManyTracks,
    #[error("a position of {0} blicks cannot be represented at 9600 ticks per quarter note")]
    Unrepresentable(Blick),
    #[error("tempo markers must be strictly increasing and between about 3.6 and 60,000,000 BPM")]
    InvalidTempo,
    #[error("a time signature is invalid or its position is out of range")]
    InvalidTimeSignature,
}

/// Serializes `project` to the bytes of a Standard MIDI File.
pub fn export(project: &Project) -> Result<Vec<u8>, MidiError> {
    let mut chunks = vec![conductor_track(project)?];
    for (index, track) in project.tracks.iter().enumerate() {
        let channel = channel(index);
        let mut events = TrackBuilder::default();
        events.meta(0, META_TRACK_NAME, track.name.as_bytes());
        for placed in project.placed_notes(track)? {
            let start = ticks(placed.start)?;
            let end = ticks(placed.end)?;
            if !placed.note.lyrics.is_empty() {
                events.meta(start, META_LYRIC, placed.note.lyrics.as_bytes());
            }
            events.push(
                start,
                Priority::NoteOn,
                vec![0x90 | channel, placed.pitch, VELOCITY],
            );
            events.push(
                end,
                Priority::NoteOff,
                vec![0x80 | channel, placed.pitch, 0],
            );
        }
        chunks.push(events.finish());
    }

    let track_count = u16::try_from(chunks.len()).map_err(|_| MidiError::TooManyTracks)?;
    let mut file = Vec::new();
    file.extend_from_slice(b"MThd");
    file.extend_from_slice(&6_u32.to_be_bytes());
    file.extend_from_slice(&1_u16.to_be_bytes());
    file.extend_from_slice(&track_count.to_be_bytes());
    file.extend_from_slice(&TICKS_PER_QUARTER.to_be_bytes());
    for chunk in chunks {
        file.extend(chunk);
    }
    Ok(file)
}

fn conductor_track(project: &Project) -> Result<Vec<u8>, MidiError> {
    let map = &project.tempo_map;
    let mut events = TrackBuilder::default();
    events.meta(0, META_TRACK_NAME, project.name.as_bytes());

    let mut previous_position = None;
    for tempo in map.tempos() {
        let microseconds = (60_000_000.0 / tempo.bpm).round();
        if previous_position.is_some_and(|previous| tempo.position <= previous)
            || !(1.0..=16_777_215.0).contains(&microseconds)
        {
            return Err(MidiError::InvalidTempo);
        }
        previous_position = Some(tempo.position);
        let bytes = (microseconds as u32).to_be_bytes();
        events.meta(ticks(tempo.position)?, META_TEMPO, &bytes[1..]);
    }

    let mut position: Blick = 0;
    let mut previous = TimeSignature::default();
    for (index, signature) in map.time_signatures().iter().enumerate() {
        if !signature.is_valid() || (index > 0 && signature.bar <= previous.bar) {
            return Err(MidiError::InvalidTimeSignature);
        }
        position = Blick::from(signature.bar - previous.bar)
            .checked_mul(previous.bar_length())
            .and_then(|length| position.checked_add(length))
            .ok_or(MidiError::InvalidTimeSignature)?;
        // Payload: numerator, denominator as a power of two, 24 clocks per click, 8 notated 32nds per quarter.
        let payload = [
            signature.numerator as u8,
            signature.denominator.ilog2() as u8,
            24,
            8,
        ];
        events.meta(ticks(position)?, META_TIME_SIGNATURE, &payload);
        previous = *signature;
    }
    Ok(events.finish())
}

/// Zero-based MIDI channel for the track at `index`: cycles through 15 channels, skipping drums (channel 10).
fn channel(index: usize) -> u8 {
    let channel = (index % 15) as u8;
    if channel >= 9 { channel + 1 } else { channel }
}

fn ticks(position: Blick) -> Result<u32, MidiError> {
    if position % BLICKS_PER_TICK != 0 {
        return Err(MidiError::Unrepresentable(position));
    }
    u32::try_from(position / BLICKS_PER_TICK)
        .ok()
        .filter(|&ticks| ticks <= MAX_TICK)
        .ok_or(MidiError::Unrepresentable(position))
}

/// Ordering of events that share a tick: metadata first, a note ends before the next begins,
/// and a lyric precedes its note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Priority {
    Meta,
    NoteOff,
    Lyric,
    NoteOn,
}

#[derive(Debug, Default)]
struct TrackBuilder {
    events: Vec<(u32, Priority, Vec<u8>)>,
}

impl TrackBuilder {
    fn push(&mut self, tick: u32, priority: Priority, data: Vec<u8>) {
        self.events.push((tick, priority, data));
    }

    fn meta(&mut self, tick: u32, kind: u8, payload: &[u8]) {
        let mut data = vec![0xFF, kind];
        write_vlq(&mut data, payload.len() as u32);
        data.extend_from_slice(payload);
        let priority = if kind == META_LYRIC {
            Priority::Lyric
        } else {
            Priority::Meta
        };
        self.push(tick, priority, data);
    }

    /// Encodes the events as an `MTrk` chunk with delta times and an end-of-track marker.
    fn finish(mut self) -> Vec<u8> {
        self.events
            .sort_by_key(|&(tick, priority, _)| (tick, priority));
        let mut body = Vec::new();
        let mut now = 0;
        for (tick, _, data) in &self.events {
            write_vlq(&mut body, tick - now);
            body.extend_from_slice(data);
            now = *tick;
        }
        body.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut chunk = b"MTrk".to_vec();
        chunk.extend_from_slice(&(body.len() as u32).to_be_bytes());
        chunk.extend(body);
        chunk
    }
}

/// Appends `value` (at most 28 bits) as a MIDI variable-length quantity.
fn write_vlq(out: &mut Vec<u8>, value: u32) {
    debug_assert!(value <= MAX_TICK);
    let mut groups = [0_u8; 4];
    let mut count = 0;
    let mut rest = value;
    loop {
        groups[count] = (rest & 0x7F) as u8;
        count += 1;
        rest >>= 7;
        if rest == 0 {
            break;
        }
    }
    for (position, &group) in groups[..count].iter().enumerate().rev() {
        out.push(if position == 0 { group } else { group | 0x80 });
    }
}

#[cfg(test)]
mod tests {
    use opensvr_core::{BLICKS_PER_QUARTER as Q, Note, NoteGroup, TempoChange, TempoMap, Track};

    use super::*;

    fn project(onset: Blick) -> Project {
        let note = Note {
            onset,
            duration: Q,
            pitch: 60,
            lyrics: "la".into(),
            phonemes: String::new(),
            detune: 0.0,
        };
        let group = NoteGroup {
            id: "main".into(),
            name: "main".into(),
            notes: vec![note],
        };
        Project {
            name: "song".into(),
            tracks: vec![Track::new("Lead", group)],
            ..Project::default()
        }
    }

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    #[test]
    fn variable_length_quantities() {
        for (value, expected) in [
            (0, &[0x00][..]),
            (0x7F, &[0x7F]),
            (0x80, &[0x81, 0x00]),
            (0x2000, &[0xC0, 0x00]),
            (0x3FFF, &[0xFF, 0x7F]),
            (MAX_TICK, &[0xFF, 0xFF, 0xFF, 0x7F]),
        ] {
            let mut out = Vec::new();
            write_vlq(&mut out, value);
            assert_eq!(out, expected, "value {value:#x}");
        }
    }

    #[test]
    fn writes_a_well_formed_file() {
        let bytes = export(&project(0)).unwrap();
        assert_eq!(&bytes[..4], b"MThd");
        assert_eq!(
            &bytes[8..14],
            [0, 1, 0, 2, 0x25, 0x80],
            "format 1, two tracks, 9600 PPQ"
        );
        assert!(
            contains(&bytes, &[0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]),
            "120 BPM tempo"
        );
        assert!(
            contains(&bytes, &[0xFF, 0x58, 0x04, 4, 2, 24, 8]),
            "4/4 meter"
        );
        assert!(
            contains(&bytes, &[0xFF, 0x05, 2, b'l', b'a', 0x00, 0x90, 60, 100]),
            "lyric precedes note-on"
        );
        // The note lasts one quarter: 9600 ticks, encoded as 0xCB 0x00 before the note-off.
        assert!(contains(&bytes, &[0xCB, 0x00, 0x80, 60, 0]));
        assert!(bytes.ends_with(&[0x00, 0xFF, 0x2F, 0x00]));
    }

    #[test]
    fn rejects_positions_between_ticks() {
        assert!(matches!(
            export(&project(1)),
            Err(MidiError::Unrepresentable(1))
        ));
    }

    #[test]
    fn rejects_duplicate_tempo_positions() {
        let mut project = project(0);
        let tempo = TempoChange {
            position: 0,
            bpm: 90.0,
        };
        project.tempo_map = TempoMap::new(vec![tempo, tempo], Vec::new());
        assert!(matches!(export(&project), Err(MidiError::InvalidTempo)));
    }

    #[test]
    fn channels_skip_the_drum_channel() {
        let channels: Vec<_> = (0..16).map(channel).collect();
        assert_eq!(channels[8], 8);
        assert_eq!(channels[9], 10);
        assert!(!channels.contains(&9));
        assert_eq!(channels[15], channels[0]);
    }
}
