//! Standard MIDI File import (formats 0 and 1, PPQ timing only).
//!
//! Rust counterpart of `importMidiFile` in OpenSV's `src/core/MidiFile.cpp`:
//! tempo and meter events rebuild the [`TempoMap`][opensvr_core::TempoMap],
//! note-on/off pairs become notes, and lyric meta events become lyrics.
//! Every [`MidiError`] variant below maps to one C++ `fail` string, and the
//! parser mirrors the JUCE behavior OpenSV builds on (unknown chunks are
//! skipped, a track stops at its first malformed event, same-tick note-offs
//! sort before note-ons of the same pitch).

use std::{collections::BTreeMap, collections::VecDeque, fmt::Write as _, fs, path::Path};

use opensvr_core::{
    BLICKS_PER_QUARTER, Blick, Extra, Note, NoteGroup, PitchAttributes, Project, TempoChange,
    TempoMap, TimeSignature, Track,
};

use crate::MidiError;

/// Reads the Standard MIDI File at `path` into a [`Project`].
///
/// The project name is the file stem, mirroring
/// `file.getFileNameWithoutExtension()` in `importMidiFile`.
pub fn load(path: &Path) -> Result<Project, MidiError> {
    let bytes = fs::read(path).map_err(MidiError::Io)?;
    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    parse(&bytes, name)
}

/// Parses Standard MIDI File bytes into a [`Project`] named `name`.
///
/// Mirrors `importMidiFile`: formats 0 and 1 with PPQ timing, default 120 BPM
/// and 4/4 meter, one track per MIDI channel-group, then the Rust equivalent
/// of `normaliseProject` (notes stable-sorted by onset; [`TempoMap::new`]
/// sorts the markers and restores the defaults).
pub fn parse(bytes: &[u8], name: impl Into<String>) -> Result<Project, MidiError> {
    let mut song = Song::parse(bytes)?;
    let tempo_map = import_tempo_map(&song.tracks, song.ppq)?;
    let mut tracks = Vec::new();
    for (index, events) in song.tracks.iter_mut().enumerate() {
        import_track(events, index, song.ppq, &mut tracks)?;
    }
    if tracks.is_empty() {
        return Err(MidiError::NoNotes);
    }
    Ok(Project {
        name: name.into(),
        tracks,
        library: Vec::new(),
        tempo_map,
        extra: Extra::default(),
    })
}

/// A parsed song: PPQ division plus one event list per track chunk, in file order.
#[derive(Debug)]
struct Song {
    ppq: u64,
    tracks: Vec<Vec<TrackEvent>>,
}

/// One timestamped event. Only the variants the import reads carry data;
/// anything else is skipped over so later events keep their positions.
#[derive(Debug, Clone)]
struct TrackEvent {
    tick: u64,
    kind: EventKind,
}

#[derive(Debug, Clone)]
enum EventKind {
    /// Channel voice note event: `on` is the 0x90 status nibble (velocity may
    /// be 0), `!on` is 0x80. A 0x90 with velocity 0 pairs as a note-off,
    /// exactly like OpenSV's `isNoteOn()` / `isNoteOnOrOff()` split.
    Note {
        channel: u8,
        pitch: u8,
        velocity: u8,
        on: bool,
    },
    TrackName(Vec<u8>),
    Lyric(Vec<u8>),
    /// Raw payload of a `0x51` tempo event; the length is validated later so a
    /// bad event reports the C++ "invalid tempo event" failure.
    Tempo(Vec<u8>),
    /// Raw payload of a `0x58` time-signature event, validated later.
    Meter(Vec<u8>),
    Other,
}

impl TrackEvent {
    /// A genuine note start for the same-tick reorder: 0x90 with velocity.
    /// (JUCE's `isNoteOn()` with its default argument.)
    fn reorder_on(&self) -> bool {
        matches!(
            self.kind,
            EventKind::Note {
                on: true,
                velocity: 1..,
                ..
            }
        )
    }

    /// The note-off that pairs with `key` in the same-tick reorder: a 0x80
    /// event on the same channel and pitch. (JUCE's `isNoteOff()` default.)
    fn reorder_off_for(&self, key: (u8, u8)) -> bool {
        matches!(
            self.kind,
            EventKind::Note {
                on: false,
                channel,
                pitch,
                ..
            } if (channel, pitch) == key
        )
    }

    fn note_key(&self) -> Option<((u8, u8), u8, bool)> {
        match self.kind {
            EventKind::Note {
                channel,
                pitch,
                velocity,
                on,
            } => Some(((channel, pitch), velocity, on)),
            _ => None,
        }
    }
}

impl Song {
    fn parse(bytes: &[u8]) -> Result<Self, MidiError> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take(4) != Some(b"MThd".as_slice()) {
            return Err(MidiError::Truncated);
        }
        // JUCE ignores the header length beyond a bounds check; every real
        // file writes 6, so anything else is treated as truncated.
        if cursor.u32() != Some(6) {
            return Err(MidiError::Truncated);
        }
        let Some(format) = cursor.u16() else {
            return Err(MidiError::Truncated);
        };
        let Some(count) = cursor.u16() else {
            return Err(MidiError::Truncated);
        };
        let Some(division) = cursor.u16() else {
            return Err(MidiError::Truncated);
        };
        // JUCE's header parser rejects formats above 2 and format 0 files
        // that do not hold exactly one track.
        if format > 2 || (format == 0 && count != 1) {
            return Err(MidiError::Truncated);
        }
        // `importMidiFile` checks the format before the timing, in this order.
        if format == 2 {
            return Err(MidiError::UnsupportedFormat);
        }
        if division == 0 || division & 0x8000 != 0 {
            return Err(MidiError::SmpteTiming);
        }
        let ppq = u64::from(division & 0x7FFF);
        let mut tracks = Vec::with_capacity(usize::from(count));
        for _ in 0..count {
            let chunk = cursor.take(4).ok_or(MidiError::Truncated)?;
            let Some(size) = cursor.u32() else {
                return Err(MidiError::Truncated);
            };
            let Some(body) = cursor.take(size as usize) else {
                return Err(MidiError::Truncated);
            };
            if chunk == b"MTrk" {
                tracks.push(read_track(body));
            }
        }
        // JUCE fails the whole file on trailing bytes (`size == 0` check).
        if !cursor.is_empty() {
            return Err(MidiError::Truncated);
        }
        Ok(Self { ppq, tracks })
    }
}

/// Reads one track chunk into timestamped events, stopping at the first
/// malformed delta or message like JUCE's `readTrack` does.
fn read_track(body: &[u8]) -> Vec<TrackEvent> {
    let mut events = Vec::new();
    let mut tick: u64 = 0;
    let mut cursor = Cursor::new(body);
    // Running status: the last channel-voice status byte, kept across
    // meta and sysex events exactly like JUCE's `lastStatusByte`.
    let mut running: Option<u8> = None;
    while !cursor.is_empty() {
        let Some(delta) = cursor.vlq() else {
            break;
        };
        let Some(next) = tick.checked_add(u64::from(delta)) else {
            break;
        };
        tick = next;
        let Some(byte) = cursor.u8() else {
            break;
        };
        let status = if byte >= 0x80 {
            byte
        } else if running.is_some() {
            cursor.back_up();
            running.expect("running status was just checked")
        } else {
            break;
        };
        let Some(kind) = read_message(&mut cursor, status) else {
            break;
        };
        if status < 0xF0 {
            running = Some(status);
        }
        events.push(TrackEvent { tick, kind });
    }
    events
}

/// Reads one message body after its status byte; `None` stops the track.
fn read_message(cursor: &mut Cursor<'_>, status: u8) -> Option<EventKind> {
    match status {
        0x80..=0xEF => {
            // 0xC0/0xD0 carry one data byte, the rest carry two.
            let length = if status & 0xF0 == 0xC0 || status & 0xF0 == 0xD0 {
                1
            } else {
                2
            };
            let data = cursor.take(length)?;
            let channel = status & 0x0F;
            match status & 0xF0 {
                0x80 => Some(EventKind::Note {
                    channel,
                    pitch: data[0],
                    velocity: data[1],
                    on: false,
                }),
                0x90 => Some(EventKind::Note {
                    channel,
                    pitch: data[0],
                    velocity: data[1],
                    on: true,
                }),
                _ => Some(EventKind::Other),
            }
        }
        0xF0 | 0xF7 => {
            let length = cursor.vlq()?;
            cursor.take(length as usize)?;
            Some(EventKind::Other)
        }
        0xFF => {
            let meta = cursor.u8()?;
            let length = cursor.vlq()?;
            let payload = cursor.take(length as usize)?;
            Some(match meta {
                0x03 => EventKind::TrackName(payload.to_vec()),
                0x05 => EventKind::Lyric(payload.to_vec()),
                0x51 => EventKind::Tempo(payload.to_vec()),
                0x58 => EventKind::Meter(payload.to_vec()),
                _ => EventKind::Other,
            })
        }
        // System-common and real-time bytes never appear in the files the
        // exporter writes; stop the track rather than misread them.
        _ => None,
    }
}

/// Byte cursor over the file or a single chunk.
#[derive(Debug)]
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    /// Steps back one byte so a running-status data byte is re-read as data.
    fn back_up(&mut self) {
        self.pos -= 1;
    }

    fn u16(&mut self) -> Option<u16> {
        let bytes = self.take(2)?;
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let bytes = self.take(4)?;
        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(length)?;
        if end > self.bytes.len() {
            return None;
        }
        let head = &self.bytes[self.pos..end];
        self.pos = end;
        Some(head)
    }

    /// Reads a variable-length quantity: at most 4 bytes, 7 bits each.
    fn vlq(&mut self) -> Option<u32> {
        let mut value: u32 = 0;
        for _ in 0..4 {
            let byte = self.u8()?;
            value = value << 7 | u32::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }
}

/// Converts MIDI ticks at `ppq` to blicks, mirroring `ticksToBlick`: the
/// scaled value rounds half away from zero (half up here, since ticks are
/// never negative) and positions past `Blick::MAX` are refused.
fn ticks_to_blick(ticks: u64, ppq: u64) -> Option<Blick> {
    let scaled = i128::from(ticks) * i128::from(BLICKS_PER_QUARTER);
    let rounded = (scaled + i128::from(ppq) / 2) / i128::from(ppq);
    if rounded >= i128::from(Blick::MAX) {
        return None;
    }
    Some(rounded as Blick)
}

/// Rebuilds the tempo map from every track's tempo and meter events, in file
/// order so later tracks win duplicate positions like the C++ `std::map`
/// assignment. Mirrors `importTempoMap`, including the default 120 BPM and
/// 4/4 meter and the bar-boundary alignment of meter changes.
fn import_tempo_map(tracks: &[Vec<TrackEvent>], ppq: u64) -> Result<TempoMap, MidiError> {
    let mut tempos: BTreeMap<Blick, f64> = BTreeMap::from([(0, 120.0)]);
    for events in tracks {
        for event in events {
            if let EventKind::Tempo(payload) = &event.kind {
                let position =
                    ticks_to_blick(event.tick, ppq).ok_or(MidiError::InvalidTempoEvent)?;
                if payload.len() != 3 {
                    return Err(MidiError::InvalidTempoEvent);
                }
                let micros = u32::from(payload[0]) << 16
                    | u32::from(payload[1]) << 8
                    | u32::from(payload[2]);
                if micros == 0 {
                    return Err(MidiError::InvalidTempoEvent);
                }
                tempos.insert(position, 60_000_000.0 / f64::from(micros));
            }
        }
    }

    let mut meters: BTreeMap<Blick, (u8, u8)> = BTreeMap::from([(0, (4, 4))]);
    for events in tracks {
        for event in events {
            if let EventKind::Meter(payload) = &event.kind {
                let position =
                    ticks_to_blick(event.tick, ppq).ok_or(MidiError::InvalidMeterEvent)?;
                if payload.len() != 4 || payload[1] > 7 {
                    return Err(MidiError::InvalidMeterEvent);
                }
                let signature = TimeSignature {
                    bar: 0,
                    numerator: u32::from(payload[0]),
                    denominator: 1 << payload[1],
                    extra: Extra::default(),
                };
                if !signature.is_valid() {
                    return Err(MidiError::InvalidTimeSignature);
                }
                meters.insert(position, (payload[0], payload[1]));
            }
        }
    }

    let mut signatures = Vec::with_capacity(meters.len());
    let mut previous_position: Blick = 0;
    let mut previous = TimeSignature::default();
    for (position, (numerator, exponent)) in meters {
        let mut signature = TimeSignature {
            bar: 0,
            numerator: u32::from(numerator),
            denominator: 1 << exponent,
            extra: Extra::default(),
        };
        let length = previous.bar_length();
        let distance = position - previous_position;
        if distance % length != 0 {
            return Err(MidiError::OffBarMeterChange);
        }
        let bars = distance / length;
        if bars > Blick::from(i32::MAX) - Blick::from(previous.bar) {
            return Err(MidiError::BarOverflow);
        }
        signature.bar = (Blick::from(previous.bar) + bars) as u32;
        signatures.push(signature.clone());
        previous_position = position;
        previous = signature;
    }

    let tempos = tempos
        .into_iter()
        .map(|(position, bpm)| TempoChange {
            position,
            bpm,
            extra: Extra::default(),
        })
        .collect();
    Ok(TempoMap::new(tempos, signatures))
}

/// Imports one MIDI track chunk into project tracks, splitting channels like
/// `importTrack`: the track name defaults to `"Track N"`, lyric events
/// concatenate per tick, note-ons pair with the earliest pending note-on of
/// the same channel and pitch, and multi-channel tracks gain
/// `" (channel M)"` suffixes.
fn import_track(
    events: &mut [TrackEvent],
    source_index: usize,
    ppq: u64,
    out: &mut Vec<Track>,
) -> Result<(), MidiError> {
    let mut name = format!("Track {}", source_index + 1);
    let mut lyrics: BTreeMap<Blick, String> = BTreeMap::new();
    for event in events.iter() {
        let position = ticks_to_blick(event.tick, ppq).ok_or(MidiError::EventOutOfRange)?;
        match &event.kind {
            EventKind::TrackName(bytes) => {
                name = String::from_utf8_lossy(bytes).into_owned();
            }
            EventKind::Lyric(bytes) => {
                lyrics
                    .entry(position)
                    .or_default()
                    .push_str(&String::from_utf8_lossy(bytes));
            }
            _ => {}
        }
    }

    reorder_same_tick(events);

    let mut channels: [Option<Track>; 16] = Default::default();
    let mut active: BTreeMap<(u8, u8), VecDeque<usize>> = BTreeMap::new();
    for event in events.iter() {
        let Some(((channel, pitch), velocity, on)) = event.note_key() else {
            continue;
        };
        let position = ticks_to_blick(event.tick, ppq).ok_or(MidiError::EventOutOfRange)?;
        let pending = active.entry((channel, pitch)).or_default();
        if on && velocity != 0 {
            let track = channels[usize::from(channel)].get_or_insert_with(|| {
                Track::new(
                    name.clone(),
                    NoteGroup {
                        id: format!("midi:{}:{channel}", source_index + 1),
                        name: name.clone(),
                        ..NoteGroup::default()
                    },
                )
            });
            let lyrics = lyrics
                .get(&position)
                .cloned()
                .unwrap_or_else(|| "la".into());
            pending.push_back(track.main_group.notes.len());
            track.main_group.notes.push(Note {
                onset: position,
                duration: 0,
                // `normaliseProject` clamps corrupt pitches into range.
                pitch: pitch.min(127),
                lyrics,
                phonemes: String::new(),
                detune: 0.0,
                instant_mode: true,
                musical_type: "singing".into(),
                accent: String::new(),
                attributes: PitchAttributes::default(),
                system_attributes: PitchAttributes::default(),
                extra: Extra::default(),
            });
        } else if let Some(&index) = pending.front() {
            pending.pop_front();
            let track = channels[usize::from(channel)]
                .as_mut()
                .expect("a pending note implies its channel track exists");
            let note = &mut track.main_group.notes[index];
            if position <= note.onset {
                return Err(MidiError::ZeroDurationNote);
            }
            note.duration = position - note.onset;
        }
        // A note-off with nothing pending is ignored, like the C++ `else if`.
    }

    if active.values().any(|pending| !pending.is_empty()) {
        return Err(MidiError::MissingNoteOff);
    }

    let channel_count = channels.iter().filter(|track| track.is_some()).count();
    for (channel, slot) in channels.iter_mut().enumerate() {
        if let Some(mut track) = slot.take() {
            if channel_count > 1 {
                let _ = write!(track.name, " (channel {})", channel + 1);
            }
            // The C++ `normaliseProject` stable-sorts notes by onset.
            track.main_group.notes.sort_by_key(|note| note.onset);
            out.push(track);
        }
    }
    Ok(())
}

/// Moves same-tick note-offs before note-ons of the same channel and pitch,
/// mirroring JUCE's `reorderNoteOnsAfterNoteOffs` (which runs at track-read
/// time): back-to-back notes sharing a boundary tick pair correctly instead
/// of collapsing to zero duration.
fn reorder_same_tick(events: &mut [TrackEvent]) {
    let mut start = 0;
    while start < events.len() {
        let mut end = start;
        while end < events.len() && events[end].tick == events[start].tick {
            end += 1;
        }
        let mut at = start;
        while let Some(offset) = events[at..end].iter().position(TrackEvent::reorder_on) {
            let on = at + offset;
            let Some((key, _, _)) = events[on].note_key() else {
                break;
            };
            if let Some(off) = (at..on)
                .rev()
                .find(|&index| events[index].reorder_off_for(key))
            {
                events.swap(off, on);
                at = on + 1;
            } else {
                break;
            }
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use opensvr_core::BLICKS_PER_QUARTER as Q;

    use super::*;
    use crate::export;

    /// Minimal event writer with absolute ticks.
    struct Events {
        bytes: Vec<u8>,
        now: u64,
    }

    impl Events {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                now: 0,
            }
        }

        fn at(&mut self, tick: u64, data: &[u8]) {
            write_vlq(&mut self.bytes, (tick - self.now) as u32);
            self.bytes.extend_from_slice(data);
            self.now = tick;
        }

        fn meta(&mut self, tick: u64, kind: u8, payload: &[u8]) {
            let mut data = vec![0xFF, kind];
            write_vlq(&mut data, payload.len() as u32);
            data.extend_from_slice(payload);
            self.at(tick, &data);
        }

        fn name(&mut self, tick: u64, text: &str) {
            self.meta(tick, 0x03, text.as_bytes());
        }

        fn lyric(&mut self, tick: u64, text: &str) {
            self.meta(tick, 0x05, text.as_bytes());
        }

        fn tempo(&mut self, tick: u64, micros: u32) {
            let bytes = micros.to_be_bytes();
            self.meta(tick, 0x51, &bytes[1..]);
        }

        fn meter(&mut self, tick: u64, numerator: u8, exponent: u8) {
            self.meta(tick, 0x58, &[numerator, exponent, 24, 8]);
        }

        fn on(&mut self, tick: u64, channel: u8, pitch: u8, velocity: u8) {
            self.at(tick, &[0x90 | channel, pitch, velocity]);
        }

        fn off(&mut self, tick: u64, channel: u8, pitch: u8) {
            self.at(tick, &[0x80 | channel, pitch, 0x40]);
        }

        fn filler(&mut self, tick: u64) {
            self.meta(tick, 0x01, b"x");
        }

        fn body(mut self) -> Vec<u8> {
            self.meta(self.now, 0x2F, &[]);
            self.bytes
        }

        fn finish(self) -> Vec<u8> {
            chunk(*b"MTrk", &self.body())
        }
    }

    fn write_vlq(out: &mut Vec<u8>, value: u32) {
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

    fn chunk(tag: [u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = tag.to_vec();
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    fn song(format: u16, division: u16, tracks: &[Vec<u8>]) -> Vec<u8> {
        let mut file = b"MThd".to_vec();
        file.extend_from_slice(&6_u32.to_be_bytes());
        file.extend_from_slice(&format.to_be_bytes());
        file.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
        file.extend_from_slice(&division.to_be_bytes());
        for track in tracks {
            file.extend_from_slice(track);
        }
        file
    }

    fn note_track(division: u16, build: impl FnOnce(&mut Events)) -> Vec<u8> {
        let mut events = Events::new();
        build(&mut events);
        song(1, division, &[events.finish()])
    }

    fn note(onset: Blick, duration: Blick, pitch: u8, lyrics: &str) -> Note {
        Note {
            onset,
            duration,
            pitch,
            lyrics: lyrics.into(),
            phonemes: String::new(),
            detune: 0.0,
            instant_mode: true,
            musical_type: "singing".into(),
            accent: String::new(),
            attributes: PitchAttributes::default(),
            system_attributes: PitchAttributes::default(),
            extra: Extra::default(),
        }
    }

    #[test]
    fn imports_a_hand_built_format_0_file() {
        let mut events = Events::new();
        events.name(0, "Lead");
        events.tempo(0, 500_000);
        events.meter(0, 4, 2);
        events.lyric(0, "la");
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        let bytes = song(0, 480, &[events.finish()]);

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.name, "song");
        assert_eq!(project.tracks.len(), 1);
        let track = &project.tracks[0];
        assert_eq!(track.name, "Lead");
        assert_eq!(track.main_group.notes.len(), 1);
        let imported = &track.main_group.notes[0];
        assert_eq!(
            (
                imported.onset,
                imported.duration,
                imported.pitch,
                imported.lyrics.as_str()
            ),
            (0, Q, 60, "la")
        );
        assert_eq!(
            project.tempo_map.tempos(),
            &[TempoChange {
                position: 0,
                bpm: 120.0,
                extra: Extra::default()
            }]
        );
        assert_eq!(
            project.tempo_map.time_signatures(),
            &[TimeSignature::default()]
        );
        assert!(!track.main_group.id.is_empty());
        assert_eq!(track.main_ref.group_id, track.main_group.id);
    }

    #[test]
    fn imports_format_1_with_a_conductor_track() {
        let mut conductor = Events::new();
        conductor.name(0, "song");
        conductor.tempo(0, 600_000);
        conductor.meter(0, 3, 2);
        let mut music = Events::new();
        music.name(0, "Lead");
        music.lyric(960, "lu");
        music.on(960, 1, 62, 100);
        music.off(1920, 1, 62);
        let bytes = song(1, 960, &[conductor.finish(), music.finish()]);

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks.len(), 1);
        // A single channel keeps the plain track name.
        assert_eq!(project.tracks[0].name, "Lead");
        let imported = &project.tracks[0].main_group.notes;
        assert_eq!(imported.len(), 1);
        assert_eq!(
            (
                imported[0].onset,
                imported[0].duration,
                imported[0].pitch,
                imported[0].lyrics.as_str()
            ),
            (Q, Q, 62, "lu")
        );
        assert_eq!(project.tempo_map.tempos()[0].bpm, 100.0);
        let meter = &project.tempo_map.time_signatures()[0];
        assert_eq!((meter.bar, meter.numerator, meter.denominator), (0, 3, 4));
    }

    #[test]
    fn export_import_round_trip_is_semantically_stable() {
        let lead = NoteGroup {
            id: "lead".into(),
            name: "Lead".into(),
            notes: vec![note(0, Q, 60, "la"), note(Q, Q, 62, "lu")],
            ..NoteGroup::default()
        };
        let bass = NoteGroup {
            id: "bass".into(),
            name: "Bass".into(),
            notes: vec![note(0, 2 * Q, 48, "lo")],
            ..NoteGroup::default()
        };
        let tempo_map = TempoMap::new(
            vec![
                TempoChange {
                    position: 0,
                    bpm: 120.0,
                    extra: Extra::default(),
                },
                TempoChange {
                    position: Q,
                    bpm: 100.0,
                    extra: Extra::default(),
                },
            ],
            vec![
                TimeSignature::default(),
                TimeSignature {
                    bar: 1,
                    numerator: 3,
                    denominator: 4,
                    extra: Extra::default(),
                },
            ],
        );
        let project = Project {
            name: "song".into(),
            tracks: vec![Track::new("Lead", lead), Track::new("Bass", bass)],
            tempo_map,
            ..Project::default()
        };

        let bytes = export(&project).unwrap();
        let back = parse(&bytes, "song").unwrap();

        assert_eq!(back.name, "song");
        assert_eq!(back.tracks.len(), 2);
        for (original, imported) in project.tracks.iter().zip(back.tracks.iter()) {
            assert_eq!(imported.name, original.name);
            let before: Vec<_> = original
                .main_group
                .notes
                .iter()
                .map(|note| (note.onset, note.duration, note.pitch, note.lyrics.as_str()))
                .collect();
            let after: Vec<_> = imported
                .main_group
                .notes
                .iter()
                .map(|note| (note.onset, note.duration, note.pitch, note.lyrics.as_str()))
                .collect();
            assert_eq!(after, before, "track {}", original.name);
            assert!(!imported.main_group.id.is_empty());
            assert_eq!(imported.main_ref.group_id, imported.main_group.id);
        }
        let tempos: Vec<_> = back
            .tempo_map
            .tempos()
            .iter()
            .map(|tempo| (tempo.position, tempo.bpm))
            .collect();
        assert_eq!(tempos, [(0, 120.0), (Q, 100.0)]);
        let meters: Vec<_> = back
            .tempo_map
            .time_signatures()
            .iter()
            .map(|signature| (signature.bar, signature.numerator, signature.denominator))
            .collect();
        assert_eq!(meters, [(0, 4, 4), (1, 3, 4)]);
    }

    #[test]
    fn splits_channels_with_suffixes() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
            events.on(0, 2, 64, 100);
            events.off(480, 0, 60);
            events.off(480, 2, 64);
        });

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks.len(), 2);
        assert_eq!(project.tracks[0].name, "Track 1 (channel 1)");
        assert_eq!(project.tracks[1].name, "Track 1 (channel 3)");
        assert_eq!(project.tracks[0].main_group.notes[0].pitch, 60);
        assert_eq!(project.tracks[1].main_group.notes[0].pitch, 64);
    }

    #[test]
    fn concatenates_lyrics_and_keeps_the_last_track_name() {
        let bytes = note_track(480, |events| {
            events.name(0, "First");
            events.name(0, "Second");
            events.lyric(0, "he");
            events.lyric(0, "llo");
            events.on(0, 0, 60, 100);
            events.off(480, 0, 60);
        });

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks[0].name, "Second");
        assert_eq!(project.tracks[0].main_group.notes[0].lyrics, "hello");
    }

    #[test]
    fn pairs_back_to_back_notes_sharing_a_tick() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
            events.off(480, 0, 60);
            events.on(480, 0, 60, 100);
            events.off(960, 0, 60);
        });

        let project = parse(&bytes, "song").unwrap();
        let notes = &project.tracks[0].main_group.notes;
        assert_eq!(notes.len(), 2);
        assert_eq!((notes[0].onset, notes[0].duration), (0, Q));
        assert_eq!((notes[1].onset, notes[1].duration), (Q, Q));
    }

    #[test]
    fn velocity_zero_note_on_closes_the_note() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
            events.on(480, 0, 60, 0);
        });

        let project = parse(&bytes, "song").unwrap();
        let notes = &project.tracks[0].main_group.notes;
        assert_eq!(notes.len(), 1);
        assert_eq!((notes[0].onset, notes[0].duration), (0, Q));
    }

    #[test]
    fn stray_note_off_is_ignored() {
        let bytes = note_track(480, |events| {
            events.off(0, 0, 60);
            events.on(0, 0, 62, 100);
            events.off(480, 0, 62);
        });

        let project = parse(&bytes, "song").unwrap();
        let notes = &project.tracks[0].main_group.notes;
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].pitch, 62);
    }

    #[test]
    fn notes_without_lyrics_default_to_la() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
            events.off(480, 0, 60);
        });

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks[0].main_group.notes[0].lyrics, "la");
    }

    #[test]
    fn skips_unknown_chunks() {
        let mut events = Events::new();
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        let bytes = song(1, 480, &[chunk(*b"XXXX", b"junk"), events.finish()]);

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks.len(), 1);
    }

    #[test]
    fn ignores_trailing_junk_inside_a_track() {
        let mut events = Events::new();
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        let mut body = events.body();
        body.push(0xFF);
        let bytes = song(1, 480, &[chunk(*b"MTrk", &body)]);

        let project = parse(&bytes, "song").unwrap();
        assert_eq!(project.tracks[0].main_group.notes.len(), 1);
    }

    #[test]
    fn ticks_to_blick_rounds_and_rejects_overflow() {
        assert_eq!(ticks_to_blick(480, 480), Some(Q));
        // 705_600_000 / 11 = 64_145_454.54... rounds up.
        assert_eq!(ticks_to_blick(1, 11), Some(64_145_455));
        assert_eq!(ticks_to_blick(u64::MAX, 1), None);
    }

    #[test]
    fn missing_note_off_is_an_error() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
        });
        assert!(matches!(
            parse(&bytes, "song"),
            Err(MidiError::MissingNoteOff)
        ));
    }

    #[test]
    fn zero_duration_note_is_an_error() {
        let bytes = note_track(480, |events| {
            events.on(0, 0, 60, 100);
            events.off(0, 0, 60);
        });
        assert!(matches!(
            parse(&bytes, "song"),
            Err(MidiError::ZeroDurationNote)
        ));
    }

    #[test]
    fn empty_files_have_no_notes() {
        let bytes = note_track(480, |_| {});
        assert!(matches!(parse(&bytes, "song"), Err(MidiError::NoNotes)));
    }

    #[test]
    fn rejects_truncated_files() {
        assert!(matches!(parse(b"nope", "song"), Err(MidiError::Truncated)));
        assert!(matches!(parse(b"MThd", "song"), Err(MidiError::Truncated)));
        // A header length other than 6 is not a real MIDI header.
        let mut bad_length = b"MThd".to_vec();
        bad_length.extend_from_slice(&7_u32.to_be_bytes());
        assert!(matches!(
            parse(&bad_length, "song"),
            Err(MidiError::Truncated)
        ));
        // Declared chunk data that never arrives.
        let mut overrun = song(1, 480, &[]);
        overrun.extend_from_slice(b"MTrk");
        overrun.extend_from_slice(&100_u32.to_be_bytes());
        overrun.extend_from_slice(&[0x00; 10]);
        overrun[10] = 0;
        overrun[11] = 1;
        assert!(matches!(parse(&overrun, "song"), Err(MidiError::Truncated)));
        // Trailing bytes after the last chunk.
        let mut events = Events::new();
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        let mut trailing = song(1, 480, &[events.finish()]);
        trailing.push(0x00);
        assert!(matches!(
            parse(&trailing, "song"),
            Err(MidiError::Truncated)
        ));
    }

    #[test]
    fn rejects_unsupported_format_and_smpte_timing() {
        let mut events = Events::new();
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        let track = events.finish();
        let tracks = [track];
        assert!(matches!(
            parse(&song(2, 480, &tracks), "song"),
            Err(MidiError::UnsupportedFormat)
        ));
        assert!(matches!(
            parse(&song(5, 480, &tracks), "song"),
            Err(MidiError::Truncated)
        ));
        assert!(matches!(
            parse(&song(1, 0xE728, &tracks), "song"),
            Err(MidiError::SmpteTiming)
        ));
        assert!(matches!(
            parse(&song(1, 0, &tracks), "song"),
            Err(MidiError::SmpteTiming)
        ));
    }

    #[test]
    fn rejects_format_0_with_more_than_one_track() {
        let mut first = Events::new();
        first.on(0, 0, 60, 100);
        first.off(480, 0, 60);
        let mut second = Events::new();
        second.on(0, 0, 62, 100);
        second.off(480, 0, 62);
        assert!(matches!(
            parse(&song(0, 480, &[first.finish(), second.finish()]), "song"),
            Err(MidiError::Truncated)
        ));
    }

    #[test]
    fn rejects_bad_tempo_events() {
        // Payload length other than 3.
        let short = note_track(480, |events| {
            events.at(0, &[0xFF, 0x51, 0x02, 0x07, 0xA1]);
        });
        assert!(matches!(
            parse(&short, "song"),
            Err(MidiError::InvalidTempoEvent)
        ));
        // Zero microseconds means non-positive seconds per quarter.
        let zero = note_track(480, |events| {
            events.tempo(0, 0);
        });
        assert!(matches!(
            parse(&zero, "song"),
            Err(MidiError::InvalidTempoEvent)
        ));
    }

    #[test]
    fn rejects_bad_meter_events() {
        // Payload length other than 4.
        let short = note_track(480, |events| {
            events.at(0, &[0xFF, 0x58, 0x03, 0x04, 0x02, 0x18]);
        });
        assert!(matches!(
            parse(&short, "song"),
            Err(MidiError::InvalidMeterEvent)
        ));
        // The denominator exponent must fit the 1..=128 range.
        let exponent = note_track(480, |events| {
            events.meter(0, 4, 8);
        });
        assert!(matches!(
            parse(&exponent, "song"),
            Err(MidiError::InvalidMeterEvent)
        ));
        // A zero numerator is not a valid signature.
        let numerator = note_track(480, |events| {
            events.meter(0, 0, 2);
        });
        assert!(matches!(
            parse(&numerator, "song"),
            Err(MidiError::InvalidTimeSignature)
        ));
    }

    #[test]
    fn rejects_meter_changes_inside_a_bar() {
        let bytes = note_track(480, |events| {
            events.meter(0, 4, 2);
            events.meter(480, 3, 2);
        });
        assert!(matches!(
            parse(&bytes, "song"),
            Err(MidiError::OffBarMeterChange)
        ));
    }

    #[test]
    fn rejects_bar_number_overflow() {
        // A 1/128 bar is 300 ticks at 9600 PPQ; 2420 maximum VLQ advances
        // land on a bar boundary past `i32::MAX`.
        let mut events = Events::new();
        events.meter(0, 1, 7);
        let mut tick = 0;
        for _ in 0..2420 {
            tick += 0x0FFF_FFFF;
            events.filler(tick);
        }
        events.meter(tick, 4, 2);
        let bytes = song(1, 9600, &[events.finish()]);
        assert!(matches!(parse(&bytes, "song"), Err(MidiError::BarOverflow)));
    }

    #[test]
    fn rejects_positions_past_blick_max() {
        // Sixty maximum VLQ advances at PPQ 1 scale far past `Blick::MAX`.
        let mut events = Events::new();
        let mut tick = 0;
        for _ in 0..60 {
            tick += 0x0FFF_FFFF;
            events.filler(tick);
        }
        let bytes = song(1, 1, &[events.finish()]);
        assert!(matches!(
            parse(&bytes, "song"),
            Err(MidiError::EventOutOfRange)
        ));
    }

    #[test]
    fn load_missing_file_is_an_io_error() {
        let missing = std::env::temp_dir().join("opensvr-midi-missing-test.mid");
        let _ = fs::remove_file(&missing);
        assert!(matches!(load(&missing), Err(MidiError::Io(_))));
    }

    #[test]
    fn load_names_the_project_after_the_file_stem() {
        let path = std::env::temp_dir().join("opensvr-import-test.mid");
        let mut events = Events::new();
        events.on(0, 0, 60, 100);
        events.off(480, 0, 60);
        fs::write(&path, song(1, 480, &[events.finish()])).unwrap();
        let project = load(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(project.name, "opensvr-import-test");
        assert_eq!(project.tracks.len(), 1);
    }
}
