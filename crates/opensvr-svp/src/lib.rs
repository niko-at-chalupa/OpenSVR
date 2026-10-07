//! Reader for Synthesizer V Studio project files (`.svp`, format version 153).
//!
//! The file is JSON. It is first deserialized into private "raw" structs that
//! mirror the file layout, then validated and converted into the domain types of
//! [`opensvr_core`]. Fields the command-line tools do not use (parameter curves,
//! vocal modes, take lists, ...) are ignored; see `ROADMAP.md` for lossless saving.

use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
};

use opensvr_core::{
    Blick, GroupReference, MAX_FILE_BLICK, Mixer, Note, NoteGroup, Project, TempoChange, TempoMap,
    TimeSignature, Track, VoiceSettings,
};
use serde::Deserialize;
use thiserror::Error;

/// The only project format version this reader understands (Synthesizer V Studio 1.11.2).
pub const FORMAT_VERSION: i64 = 153;

#[derive(Debug, Error)]
pub enum SvpError {
    #[error("could not read project {}: {source}", .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("could not parse project JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "only Synthesizer V Studio 1.11.2 projects (format version 153) are supported, found version {0}"
    )]
    UnsupportedVersion(i64),
    #[error("invalid Synthesizer V project: {0}")]
    Invalid(String),
}

type Result<T> = std::result::Result<T, SvpError>;

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(SvpError::Invalid(message.into()))
}

/// Loads a project from disk. Relative voice and dictionary paths resolve against the file's directory.
pub fn load(path: &Path) -> Result<Project> {
    let bytes = fs::read(path).map_err(|source| SvpError::Io {
        path: path.to_owned(),
        source,
    })?;
    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    parse(&bytes, name, path.parent().unwrap_or(Path::new("")))
}

/// Parses project JSON. Trailing NUL bytes, which Synthesizer V appends, are ignored.
pub fn parse(bytes: &[u8], name: impl Into<String>, base_dir: &Path) -> Result<Project> {
    let end = bytes
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(0, |last| last + 1);
    let bytes = &bytes[..end];
    if bytes.is_empty() {
        return invalid("the file is empty");
    }

    let header: Header = serde_json::from_slice(bytes)?;
    if header.version != FORMAT_VERSION {
        return Err(SvpError::UnsupportedVersion(header.version));
    }

    let raw: RawProject = serde_json::from_slice(bytes)?;
    let library = raw
        .library
        .into_iter()
        .map(RawGroup::into_group)
        .collect::<Result<Vec<_>>>()?;
    let tracks = raw
        .tracks
        .into_iter()
        .map(|track| track.into_track(base_dir))
        .collect::<Result<Vec<_>>>()?;
    let project = Project {
        name: name.into(),
        tracks,
        library,
        tempo_map: raw.time.into_tempo_map()?,
    };
    check_references(&project)?;
    Ok(project)
}

/// Group UUIDs must be unique and every non-instrumental reference must resolve.
fn check_references(project: &Project) -> Result<()> {
    let mut ids = HashSet::new();
    let groups = project
        .library
        .iter()
        .chain(project.tracks.iter().map(|track| &track.main_group));
    for group in groups {
        if !ids.insert(group.id.as_str()) {
            return invalid(format!("duplicate group UUID {}", group.id));
        }
    }
    for track in &project.tracks {
        let dangling = track
            .groups
            .iter()
            .any(|reference| !reference.instrumental && !ids.contains(reference.group_id.as_str()));
        if dangling {
            return invalid(format!("track {:?} references a missing group", track.name));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// File layout
// ---------------------------------------------------------------------------

/// Read first, so an unsupported version is reported as such rather than as a layout error.
#[derive(Deserialize)]
struct Header {
    version: i64,
}

#[derive(Deserialize)]
struct RawProject {
    tracks: Vec<RawTrack>,
    library: Vec<RawGroup>,
    time: RawTime,
}

#[derive(Deserialize)]
struct RawTime {
    tempo: Vec<RawTempo>,
    meter: Vec<RawMeter>,
}

#[derive(Deserialize)]
struct RawTempo {
    position: i64,
    bpm: f64,
}

#[derive(Deserialize)]
struct RawMeter {
    index: i64,
    numerator: i64,
    denominator: i64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrack {
    #[serde(default)]
    name: String,
    main_group: RawGroup,
    main_ref: RawReference,
    groups: Vec<RawReference>,
    sv_voice: Option<RawVoice>,
    mixer: RawMixer,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawVoice {
    database_path: String,
    language: String,
    dictionary_directory: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawMixer {
    gain_decibel: f64,
    pan: f64,
    #[serde(default)]
    mute: bool,
    #[serde(default)]
    solo: bool,
}

#[derive(Deserialize)]
struct RawGroup {
    uuid: String,
    #[serde(default)]
    name: String,
    notes: Vec<RawNote>,
}

#[derive(Deserialize)]
struct RawNote {
    onset: i64,
    duration: i64,
    pitch: i64,
    lyrics: String,
    #[serde(default)]
    phonemes: String,
    #[serde(default)]
    detune: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawReference {
    #[serde(rename = "groupID")]
    group_id: String,
    blick_offset: i64,
    pitch_offset: i64,
    #[serde(default)]
    blick_absolute_begin: i64,
    #[serde(default = "unbounded")]
    blick_absolute_end: i64,
    #[serde(default)]
    is_instrumental: bool,
}

/// Sentinel the format uses for "no right crop boundary".
fn unbounded() -> i64 {
    -1
}

// ---------------------------------------------------------------------------
// Validation and conversion
// ---------------------------------------------------------------------------

fn blick(value: i64, what: &str) -> Result<Blick> {
    if value.unsigned_abs() > MAX_FILE_BLICK.unsigned_abs() {
        return invalid(format!("{what} is out of range"));
    }
    Ok(value)
}

fn resolve(base_dir: &Path, text: &str) -> Option<PathBuf> {
    (!text.is_empty()).then(|| base_dir.join(text))
}

impl RawTime {
    fn into_tempo_map(self) -> Result<TempoMap> {
        let tempos = self
            .tempo
            .into_iter()
            .map(|tempo| {
                let position = blick(tempo.position, "a tempo position")?;
                if position < 0 || !tempo.bpm.is_finite() || tempo.bpm <= 0.0 {
                    return invalid("a tempo marker is invalid");
                }
                Ok(TempoChange {
                    position,
                    bpm: tempo.bpm,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let signatures = self
            .meter
            .into_iter()
            .map(|meter| {
                let signature = TimeSignature {
                    bar: u32::try_from(meter.index).unwrap_or(u32::MAX),
                    numerator: u32::try_from(meter.numerator).unwrap_or(0),
                    denominator: u32::try_from(meter.denominator).unwrap_or(0),
                };
                if meter.index < 0 || meter.index > i64::from(i32::MAX) || !signature.is_valid() {
                    return invalid("a meter marker is invalid");
                }
                Ok(signature)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TempoMap::new(tempos, signatures))
    }
}

impl RawNote {
    fn into_note(self) -> Result<Note> {
        let onset = blick(self.onset, "a note onset")?;
        let duration = blick(self.duration, "a note duration")?;
        let pitch = u8::try_from(self.pitch).ok().filter(|&pitch| pitch <= 127);
        match pitch {
            Some(pitch) if duration > 0 && self.detune.is_finite() => Ok(Note {
                onset,
                duration,
                pitch,
                lyrics: self.lyrics,
                phonemes: self.phonemes,
                detune: self.detune,
            }),
            _ => invalid("a note has invalid timing, pitch, or detune"),
        }
    }
}

impl RawGroup {
    fn into_group(self) -> Result<NoteGroup> {
        if self.uuid.is_empty() {
            return invalid("a note group needs a UUID");
        }
        let mut notes = self
            .notes
            .into_iter()
            .map(RawNote::into_note)
            .collect::<Result<Vec<_>>>()?;
        notes.sort_by_key(|note| note.onset);
        Ok(NoteGroup {
            id: self.uuid,
            name: self.name,
            notes,
        })
    }
}

impl RawReference {
    fn into_reference(self) -> Result<GroupReference> {
        let pitch_offset = i32::try_from(self.pitch_offset)
            .ok()
            .filter(|offset| (-127..=127).contains(offset));
        let Some(pitch_offset) = pitch_offset else {
            return invalid("a group reference has an invalid pitch offset");
        };
        let begin = blick(self.blick_absolute_begin, "a crop boundary")?;
        let end = match blick(self.blick_absolute_end, "a crop boundary")? {
            -1 => None,
            end => Some(end),
        };
        if begin < 0 || end.is_some_and(|end| end <= begin) {
            return invalid("a group reference has invalid crop boundaries");
        }
        Ok(GroupReference {
            group_id: self.group_id,
            time_offset: blick(self.blick_offset, "a group offset")?,
            pitch_offset,
            begin,
            end,
            instrumental: self.is_instrumental,
        })
    }
}

impl RawTrack {
    fn into_track(self, base_dir: &Path) -> Result<Track> {
        let voice = match self.sv_voice {
            None => VoiceSettings::default(),
            Some(voice) => VoiceSettings {
                database: resolve(base_dir, &voice.database_path),
                language: voice
                    .language
                    .parse()
                    .or_else(|error| invalid(format!("svVoice: {error}")))?,
                dictionary: resolve(base_dir, &voice.dictionary_directory),
            },
        };

        let main_group = self.main_group.into_group()?;
        let main_ref = self.main_ref.into_reference()?;
        if main_ref.group_id != main_group.id {
            return invalid("a track's mainRef does not refer to its mainGroup");
        }

        let mixer = self.mixer;
        let gain = 10.0_f64.powf(mixer.gain_decibel / 20.0);
        if !gain.is_finite() || !mixer.pan.is_finite() || !(-1.0..=1.0).contains(&mixer.pan) {
            return invalid("a track has invalid mixer settings");
        }

        Ok(Track {
            name: self.name,
            main_group,
            main_ref,
            groups: self
                .groups
                .into_iter()
                .map(RawReference::into_reference)
                .collect::<Result<_>>()?,
            mixer: Mixer {
                gain,
                pan: mixer.pan,
                mute: mixer.mute,
                solo: mixer.solo,
            },
            voice,
        })
    }
}

#[cfg(test)]
mod tests {
    use opensvr_core::{BLICKS_PER_QUARTER as Q, Language};
    use serde_json::{Value, json};

    use super::*;

    fn group(uuid: &str, notes: &Value) -> Value {
        json!({ "uuid": uuid, "name": uuid, "notes": notes })
    }

    fn reference(group_id: &str) -> Value {
        json!({ "groupID": group_id, "blickOffset": 0, "pitchOffset": 0 })
    }

    fn sample() -> Value {
        let notes = json!([
            { "onset": Q, "duration": Q, "pitch": 62, "lyrics": "re" },
            { "onset": 0, "duration": Q, "pitch": 60, "lyrics": "do", "detune": 12.5 },
        ]);
        json!({
            "version": 153,
            "time": {
                "tempo": [{ "position": 0, "bpm": 100.0 }],
                "meter": [{ "index": 0, "numerator": 3, "denominator": 4 }],
            },
            "library": [group("lib", &json!([]))],
            "tracks": [{
                "name": "Lead",
                "mainGroup": group("main", &notes),
                "mainRef": reference("main"),
                "groups": [reference("lib")],
                "mixer": { "gainDecibel": -6.0, "pan": 0.5, "mute": false, "solo": true },
                "svVoice": { "databasePath": "voices/a.nofs", "language": "english", "dictionaryDirectory": "" },
            }],
        })
    }

    fn parse_value(value: &Value) -> Result<Project> {
        parse(value.to_string().as_bytes(), "song", Path::new("/proj"))
    }

    #[test]
    fn loads_a_project() {
        let project = parse_value(&sample()).unwrap();
        assert_eq!(project.name, "song");
        assert_eq!(project.tempo_map.tempo_at(0), 100.0);
        assert_eq!(project.tempo_map.time_signatures()[0].numerator, 3);

        let track = &project.tracks[0];
        assert_eq!(track.name, "Lead");
        assert_eq!(
            track.main_group.notes[0].lyrics, "do",
            "notes are sorted by onset"
        );
        assert_eq!(track.main_group.notes[0].detune, 12.5);
        assert!(track.mixer.solo);
        assert!((track.mixer.gain - 0.501_187).abs() < 1e-5);
        assert_eq!(track.voice.language, Language::English);
        assert_eq!(
            track.voice.database.as_deref(),
            Some(Path::new("/proj/voices/a.nofs"))
        );
        assert_eq!(track.voice.dictionary, None);
    }

    #[test]
    fn ignores_trailing_nul_bytes() {
        let mut bytes = sample().to_string().into_bytes();
        bytes.extend([0, 0, 0]);
        assert!(parse(&bytes, "song", Path::new("")).is_ok());
        assert!(matches!(
            parse(&[0, 0], "song", Path::new("")),
            Err(SvpError::Invalid(_))
        ));
    }

    #[test]
    fn rejects_other_format_versions() {
        let mut value = sample();
        value["version"] = json!(152);
        assert!(matches!(
            parse_value(&value),
            Err(SvpError::UnsupportedVersion(152))
        ));
    }

    #[test]
    fn rejects_invalid_content() {
        let cases: [(&str, Value); 5] = [
            ("/tracks/0/mainGroup/notes/0/pitch", json!(128)),
            ("/tracks/0/mainGroup/notes/0/duration", json!(0)),
            ("/tracks/0/mixer/pan", json!(2.0)),
            ("/tracks/0/groups/0/groupID", json!("missing")),
            ("/tracks/0/svVoice/language", json!("klingon")),
        ];
        for (pointer, replacement) in cases {
            let mut value = sample();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                matches!(parse_value(&value), Err(SvpError::Invalid(_))),
                "{pointer} should be rejected"
            );
        }
    }

    #[test]
    fn crop_sentinel_means_unbounded() {
        let mut value = sample();
        value["tracks"][0]["groups"][0]["blickAbsoluteBegin"] = json!(Q);
        value["tracks"][0]["groups"][0]["blickAbsoluteEnd"] = json!(3 * Q);
        let project = parse_value(&value).unwrap();
        let track = &project.tracks[0];
        assert_eq!(track.main_ref.end, None);
        assert_eq!(
            (track.groups[0].begin, track.groups[0].end),
            (Q, Some(3 * Q))
        );
    }
}
