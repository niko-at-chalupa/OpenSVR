//! Reader and writer for Synthesizer V Studio project files (`.svp`, format version 153).
//!
//! The file is JSON. Reads go through private "raw" structs that mirror the
//! file layout, then validate into the domain types of [`opensvr_core`].
//! Group automation curves (`parameters.pitchDelta` and `parameters.vibratoEnv`)
//! and per-mode vocal-mode curves are read with the same validation as
//! OpenSV's `readCurve`, as is each reference's `systemPitchDelta` contour;
//! unknown vendor fields at every level are captured
//! into [`Extra`](opensvr_core::Extra) exactly like OpenSV's `preserve`.
//! Writes mirror OpenSV's `write*` functions: each level starts from
//! `retainedObject(extra)` and overwrites the handled keys from the domain
//! values, so `parse(serialize(project))` is a fixed point.

use std::{
    collections::{BTreeMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
};

use opensvr_core::{
    AutomationPoint, Blick, Curve, Extra, GroupReference, Interpolation, Language, MAX_FILE_BLICK,
    Mixer, Note, NoteGroup, PitchAttribute, PitchAttributes, Project, TempoChange, TempoMap,
    TimeSignature, Track, VocalModeSetting, VoiceSettings,
};
use serde::Deserialize;
use serde_json::{Map, Value};
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

/// Mirrors OpenSV's `preserve` in `src/core/ProjectFile.cpp`: strips the
/// handled keys from a clone of the original object and serializes the rest.
/// Key order is the document order (via `preserve_order`), as with JUCE's
/// insertion-ordered properties; the empty form is `"{}"` on both sides
/// (JUCE `writeAsJSON` emits `{` … `}` with no entries).
fn preserve(fields: &Map<String, Value>, strips: &[&str]) -> Result<Extra> {
    let mut kept = fields.clone();
    for strip in strips {
        kept.shift_remove(*strip);
    }
    Ok(Extra::new(serde_json::to_string(&kept)?))
}

/// Loads a project from disk. Relative voice, dictionary, and backing-track paths resolve against the file's directory.
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
        .map(|value| RawGroup::into_group(&value))
        .collect::<Result<Vec<_>>>()?;
    let tracks = raw
        .tracks
        .into_iter()
        .map(|track| RawTrack::into_track(&track, base_dir))
        .collect::<Result<Vec<_>>>()?;
    // The whole document parses (above), so the root object exists; its
    // unknown fields are preserved like every other level. C++ strips only
    // `tracks`/`library` here (`ProjectFile.cpp:814`), keeping `version`,
    // `time` and vendor keys.
    let doc: Value = serde_json::from_slice(bytes)?;
    let Some(root) = doc.as_object() else {
        return invalid("the project root must be an object");
    };
    let project = Project {
        name: name.into(),
        tracks,
        library,
        tempo_map: raw.time.into_tempo_map()?,
        extra: preserve(root, &["tracks", "library"])?,
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
    tracks: Vec<Value>,
    library: Vec<Value>,
    time: RawTime,
}

#[derive(Deserialize)]
struct RawTime {
    tempo: Vec<Value>,
    meter: Vec<Value>,
}

#[derive(Deserialize)]
struct RawTempo {
    position: Value,
    bpm: f64,
}

#[derive(Deserialize)]
struct RawMeter {
    index: Value,
    numerator: Value,
    denominator: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTrack {
    #[serde(default)]
    name: String,
    main_group: Value,
    main_ref: Value,
    groups: Vec<Value>,
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
    notes: Vec<Value>,
    /// `parameters` stays as [`Value`]: the two automation curves are read
    /// out of it, while every other key (and the key itself) is preserved
    /// verbatim, exactly like `readGroup`'s massaged clone in
    /// `src/core/ProjectFile.cpp`.
    #[serde(default)]
    parameters: Option<Value>,
    /// `vocalModes` stays as [`Value`] so a non-object fails with the same
    /// `Invalid` error OpenSV's `readGroup` reports, rather than a JSON error.
    /// (`None`/JSON null both mean absent, as JUCE parses null to void.)
    #[serde(default, rename = "vocalModes")]
    vocal_modes: Option<Value>,
}

/// A `{"mode": ..., "points": [pos0, val0, ...]}` automation curve.
/// Points stay as [`Value`] so non-numeric entries fail with the same
/// `Invalid` error OpenSV's `readCurve` reports, rather than a JSON error.
#[derive(Debug, Deserialize)]
struct RawCurve {
    mode: String,
    points: Vec<Value>,
}

#[derive(Deserialize)]
struct RawNote {
    onset: Value,
    duration: Value,
    pitch: Value,
    lyrics: String,
    #[serde(default)]
    phonemes: String,
    #[serde(default)]
    detune: f64,
    /// Note `attributes` stays as [`Value`] so a non-object fails with the
    /// same `Invalid` error OpenSV's `readPitchAttributes` reports, rather
    /// than a JSON error. (`None`/JSON null both mean absent, as JUCE parses
    /// null to void.)
    #[serde(default)]
    attributes: Option<Value>,
    /// Note `systemAttributes`, read with the same table.
    #[serde(default, rename = "systemAttributes")]
    system_attributes: Option<Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawReference {
    #[serde(rename = "groupID")]
    group_id: String,
    blick_offset: Value,
    pitch_offset: Value,
    #[serde(default)]
    blick_absolute_begin: Option<Value>,
    #[serde(default)]
    blick_absolute_end: Option<Value>,
    #[serde(default)]
    is_instrumental: bool,
    /// Reference `audio` backing-track metadata (`filename`, `duration`).
    /// Kept as [`Value`] so a malformed block is the single `Invalid` error
    /// OpenSV's `readReference` reports, rather than a JSON error;
    /// absent/null means defaults.
    #[serde(default)]
    audio: Option<Value>,
    /// Reference `voice` overrides (`vocalModeInherited`, `vocalModePreset`,
    /// `vocalModeParams`). Kept as [`Value`] so a non-object is an `Invalid`
    /// error like the C++ block; absent/null means defaults.
    #[serde(default)]
    voice: Option<Value>,
}

/// Mirrors OpenSV's `readInteger` in `src/core/ProjectFile.cpp`: integer or
/// double JSON that is finite, integral (`trunc`) and within
/// [`MAX_FILE_BLICK`]; anything else is `None` (the caller reports the
/// field's existing `Invalid` error, never a JSON error).
fn read_integer(value: &Value) -> Option<Blick> {
    let number = value.as_f64()?;
    if !number.is_finite() || number.trunc() != number || number.abs() > MAX_FILE_BLICK as f64 {
        return None;
    }
    if let Some(exact) = value.as_i64() {
        return Some(exact);
    }
    if let Some(unsigned) = value.as_u64() {
        return Blick::try_from(unsigned).ok();
    }
    Some(number as Blick)
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
            .map(|value| {
                let tempo: RawTempo = serde_json::from_value(value.clone())?;
                let Some(position) =
                    read_integer(&tempo.position).map(|raw| blick(raw, "a tempo position"))
                else {
                    return invalid("a tempo marker is invalid");
                };
                let position = position?;
                if position < 0 || !tempo.bpm.is_finite() || tempo.bpm <= 0.0 {
                    return invalid("a tempo marker is invalid");
                }
                let Some(entry) = value.as_object() else {
                    return invalid("a tempo marker is invalid");
                };
                Ok(TempoChange {
                    position,
                    bpm: tempo.bpm,
                    // C++ `ProjectFile.cpp:847` strips only position/bpm.
                    extra: preserve(entry, &["position", "bpm"])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let signatures = self
            .meter
            .into_iter()
            .map(|value| {
                let meter: RawMeter = serde_json::from_value(value.clone())?;
                let (Some(index), Some(numerator), Some(denominator)) = (
                    read_integer(&meter.index),
                    read_integer(&meter.numerator),
                    read_integer(&meter.denominator),
                ) else {
                    return invalid("a meter marker is invalid");
                };
                let signature = TimeSignature {
                    bar: u32::try_from(index).unwrap_or(u32::MAX),
                    numerator: u32::try_from(numerator).unwrap_or(0),
                    denominator: u32::try_from(denominator).unwrap_or(0),
                    extra: Extra::default(),
                };
                if index < 0 || index > i64::from(i32::MAX) || !signature.is_valid() {
                    return invalid("a meter marker is invalid");
                }
                let Some(entry) = value.as_object() else {
                    return invalid("a meter marker is invalid");
                };
                // C++ `ProjectFile.cpp:862` strips only index/numerator/denominator.
                let extra = preserve(entry, &["index", "numerator", "denominator"])?;
                Ok(TimeSignature { extra, ..signature })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TempoMap::new(tempos, signatures))
    }
}

impl RawNote {
    fn into_note(value: &Value) -> Result<Note> {
        let raw: RawNote = serde_json::from_value(value.clone())?;
        let (Some(onset), Some(duration), Some(pitch_raw)) = (
            read_integer(&raw.onset),
            read_integer(&raw.duration),
            read_integer(&raw.pitch),
        ) else {
            return invalid("a note has invalid timing, pitch, or detune");
        };
        let onset = blick(onset, "a note onset")?;
        let duration = blick(duration, "a note duration")?;
        let pitch = u8::try_from(pitch_raw).ok().filter(|&pitch| pitch <= 127);
        let Some(pitch) = pitch.filter(|_| duration > 0 && raw.detune.is_finite()) else {
            return invalid("a note has invalid timing, pitch, or detune");
        };
        let attributes = read_pitch_attributes(raw.attributes.as_ref(), "note attributes")?;
        let system_attributes =
            read_pitch_attributes(raw.system_attributes.as_ref(), "note systemAttributes")?;
        let Some(note) = value.as_object() else {
            return invalid("a note has invalid timing, pitch, or detune");
        };
        // Plain scalar style fields, exactly like `readGroup`
        // (`ProjectFile.cpp:298-320`): present values must have the right
        // JSON type, absent keys (or JSON null, which JUCE parses to void)
        // keep the C++ member defaults.
        let accent = match note.get("accent") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) => text.clone(),
            _ => return invalid("a note's accent must be a string."),
        };
        let instant_mode = match note.get("instantMode") {
            None | Some(Value::Null) => true,
            Some(Value::Bool(mode)) => *mode,
            _ => return invalid("a note's instantMode must be a boolean."),
        };
        let musical_type = match note.get("musicalType") {
            None | Some(Value::Null) => "singing".to_owned(),
            Some(Value::String(style)) => style.clone(),
            _ => return invalid("a note's musicalType must be a string."),
        };
        // C++ `ProjectFile.cpp:329` strips the handled keys, including the
        // three scalar style members above; only truly unknown keys survive
        // into `extra`, exactly as there.
        let extra = preserve(
            note,
            &[
                "onset",
                "duration",
                "pitch",
                "lyrics",
                "phonemes",
                "accent",
                "detune",
                "instantMode",
                "musicalType",
                "attributes",
                "systemAttributes",
            ],
        )?;
        Ok(Note {
            onset,
            duration,
            pitch,
            lyrics: raw.lyrics,
            phonemes: raw.phonemes,
            detune: raw.detune,
            instant_mode,
            musical_type,
            accent,
            attributes,
            system_attributes,
            extra,
        })
    }
}

impl RawGroup {
    fn into_group(value: &Value) -> Result<NoteGroup> {
        let raw: RawGroup = serde_json::from_value(value.clone())?;
        if raw.uuid.is_empty() {
            return invalid("a note group needs a UUID");
        }
        let mut notes = raw
            .notes
            .into_iter()
            .map(|note| RawNote::into_note(&note))
            .collect::<Result<Vec<_>>>()?;
        notes.sort_by_key(|note| note.onset);
        let parameters = raw.parameters.as_ref().and_then(Value::as_object);
        let pitch_delta = read_curve_inner(
            parameters.and_then(|map| map.get("pitchDelta")),
            "pitchDelta",
            true,
        )?;
        let vibrato_env = read_curve_inner(
            parameters.and_then(|map| map.get("vibratoEnv")),
            "vibratoEnv",
            true,
        )?;
        let vocal_modes = read_vocal_modes(raw.vocal_modes.as_ref())?;
        let Some(group) = value.as_object() else {
            return invalid("a note group needs a UUID");
        };
        // Mirror `readGroup` (`ProjectFile.cpp:277-281`): the preserved
        // `parameters` blob is the original minus the two curves Rust reads,
        // and is always present (empty when the file omits it).
        let mut massaged = parameters.cloned().unwrap_or_default();
        massaged.shift_remove("pitchDelta");
        massaged.shift_remove("vibratoEnv");
        let mut kept = group.clone();
        kept.insert("parameters".to_owned(), Value::Object(massaged));
        // C++ strips `uuid`/`name`/`notes`/`vocalModes`, keeping the massaged
        // `parameters` blob and unknown keys.
        let extra = preserve(&kept, &["uuid", "name", "notes", "vocalModes"])?;
        Ok(NoteGroup {
            id: raw.uuid,
            name: raw.name,
            notes,
            pitch_delta,
            vibrato_env,
            vocal_modes,
            extra,
        })
    }
}

/// Reads the `vocalModes` object of a group exactly like the vocal-mode block
/// of OpenSV's `readGroup` in `src/core/ProjectFile.cpp`: absent (or JSON
/// null, which JUCE parses to void) stays empty; a non-object is an error; a
/// per-mode number is a static amount; a per-mode object carries an optional
/// `enabled` boolean, an optional finite `amount`, and an optional automation
/// curve (either `mode` or `points` present) read with `readCurve` validation.
/// Anything else is an error.
fn read_vocal_modes(raw: Option<&Value>) -> Result<BTreeMap<String, VocalModeSetting>> {
    let Some(raw) = raw else {
        return Ok(BTreeMap::new());
    };
    if raw.is_null() {
        return Ok(BTreeMap::new());
    }
    let Value::Object(modes) = raw else {
        return invalid("vocalModes must be an object of mode curves.");
    };
    let mut out = BTreeMap::new();
    for (name, mode_value) in modes {
        if mode_value.is_null()
            || mode_value.is_boolean()
            || mode_value.is_string()
            || mode_value.is_array()
        {
            return invalid(format!(
                "vocalModes.{name} must be an amount or settings object."
            ));
        }
        if mode_value.is_number() {
            match mode_value.as_f64() {
                Some(amount) if amount.is_finite() => {
                    out.insert(
                        name.clone(),
                        VocalModeSetting {
                            enabled: true,
                            amount,
                            has_curve: false,
                            curve: Curve::default(),
                            // A plain amount carries no preserved JSON: the
                            // C++ `preserve` call sits in the object branch
                            // only, so this keeps the default empty string.
                            extra: Extra::default(),
                        },
                    );
                }
                _ => return invalid(format!("vocalModes.{name} must have a finite amount.")),
            }
            continue;
        }
        // Per-mode settings object.
        let Value::Object(fields) = mode_value else {
            return invalid(format!(
                "vocalModes.{name} must be an amount or settings object."
            ));
        };
        let mut setting = VocalModeSetting::default();
        if let Some(enabled) = fields.get("enabled") {
            match enabled.as_bool() {
                Some(value) => setting.enabled = value,
                None => {
                    return invalid(format!("vocalModes.{name}.enabled must be a boolean."));
                }
            }
        }
        if let Some(amount) = fields.get("amount") {
            match amount.as_f64() {
                Some(value) if value.is_finite() => setting.amount = value,
                _ => {
                    return invalid(format!("vocalModes.{name}.amount must be a finite number."));
                }
            }
        }
        if fields.contains_key("mode") || fields.contains_key("points") {
            let curve_name = format!("vocalModes.{name}");
            // Unlike `pitchDelta`/`vibratoEnv`, `normaliseProject` never
            // normalises per-mode curves, so the points stay in file order.
            setting.curve = read_curve_inner(Some(mode_value), &curve_name, false)?;
            setting.has_curve = true;
        }
        // C++ `ProjectFile.cpp:386` preserves the settings object minus the
        // four handled keys.
        setting.extra = preserve(fields, &["enabled", "amount", "mode", "points"])?;
        out.insert(name.clone(), setting);
    }
    Ok(out)
}

/// Reads the `voice` overrides of a group reference exactly like the `voice`
/// block of OpenSV's `readReference`: absent/null keeps the C++ defaults
/// (`vocalModeInherited = true`, empty preset/params); a non-object is an
/// error; `vocalModeParams` must be an object of finite numbers.
fn read_voice_overrides(raw: Option<&Value>) -> Result<(bool, String, BTreeMap<String, f64>)> {
    let Some(raw) = raw else {
        return Ok((true, String::new(), BTreeMap::new()));
    };
    if raw.is_null() {
        return Ok((true, String::new(), BTreeMap::new()));
    }
    let Value::Object(voice) = raw else {
        return invalid("a group reference's voice settings must be an object.");
    };
    let mut inherited = true;
    if let Some(value) = voice.get("vocalModeInherited") {
        match value.as_bool() {
            Some(value) => inherited = value,
            None => return invalid("vocalModeInherited must be a boolean."),
        }
    }
    let mut preset = String::new();
    if let Some(value) = voice.get("vocalModePreset") {
        match value.as_str() {
            Some(value) => value.clone_into(&mut preset),
            None => return invalid("vocalModePreset must be a string."),
        }
    }
    let mut params = BTreeMap::new();
    if let Some(mode_params) = voice.get("vocalModeParams") {
        if mode_params.is_null() {
            // JUCE parses JSON null to void, i.e. absent.
        } else {
            let Value::Object(entries) = mode_params else {
                return invalid("vocalModeParams must be an object of mode amounts.");
            };
            for (name, amount) in entries {
                match amount.as_f64() {
                    Some(value) if value.is_finite() => {
                        params.insert(name.clone(), value);
                    }
                    _ => {
                        return invalid(format!("vocalModeParams.{name} must be a finite number."));
                    }
                }
            }
        }
    }
    Ok((inherited, preset, params))
}

/// Reads the `audio` backing-track metadata of a group reference exactly like
/// the `audio` block of OpenSV's `readReference` (`src/core/ProjectFile.cpp`):
/// absent (or JSON null, which JUCE parses to void) keeps the defaults
/// (`None`, `0.0`); otherwise the value must be an object with a string
/// `filename` and a finite `duration >= 0.0`, else one shared `Invalid`
/// error. A non-empty filename resolves against the project directory, while
/// an empty one stays unset, as in the C++.
fn read_audio(raw: Option<&Value>, base_dir: &Path) -> Result<(Option<PathBuf>, f64)> {
    let Some(raw) = raw else {
        return Ok((None, 0.0));
    };
    if raw.is_null() {
        return Ok((None, 0.0));
    }
    let invalid_audio = || invalid("an instrumental reference has invalid audio metadata.");
    let Value::Object(audio) = raw else {
        return invalid_audio();
    };
    let Some(Value::String(filename)) = audio.get("filename") else {
        return invalid_audio();
    };
    let Some(duration) = audio.get("duration").and_then(Value::as_f64) else {
        return invalid_audio();
    };
    if !duration.is_finite() || duration < 0.0 {
        return invalid_audio();
    }
    Ok((resolve(base_dir, filename), duration))
}

/// Reads pitch/vibrato attributes exactly like OpenSV's `readPitchAttributes`
/// in `src/core/ProjectFile.cpp`: absent (or JSON null, which JUCE parses to
/// void) keeps the defaults; a non-object is an error; each present field
/// must be a finite number. The loop runs over the [`PitchAttribute::ALL`]
/// table, so the 14 keys live in `opensvr-core` rather than in 14 blocks
/// here. Unknown fields are preserved like everywhere else; the three
/// `vocalMode*` keys are stripped too because the reference `voice` object
/// doubles as the `voicePitch` container (`ProjectFile.cpp:208-210`).
fn read_pitch_attributes(raw: Option<&Value>, context: &str) -> Result<PitchAttributes> {
    let mut out = PitchAttributes::default();
    let Some(raw) = raw else {
        return Ok(out);
    };
    if raw.is_null() {
        return Ok(out);
    }
    let Value::Object(fields) = raw else {
        return invalid(format!("{context} must be an object."));
    };
    for field in PitchAttribute::ALL {
        let Some(value) = fields.get(field.name()) else {
            continue;
        };
        match value.as_f64() {
            Some(amount) if amount.is_finite() => out.set(field, amount),
            _ => {
                return invalid(format!(
                    "{context}.{} must be a finite number.",
                    field.name()
                ));
            }
        }
    }
    let mut kept = fields.clone();
    for field in PitchAttribute::ALL {
        kept.shift_remove(field.name());
    }
    for strip in ["vocalModeInherited", "vocalModePreset", "vocalModeParams"] {
        kept.shift_remove(strip);
    }
    out.extra = Extra::new(serde_json::to_string(&kept)?);
    Ok(out)
}

/// Reads one automation curve exactly like OpenSV's `readCurve` in
/// `src/core/ProjectFile.cpp`: an absent curve is OK and keeps the default
/// (cubic, empty); otherwise the mode must be known, the points must form
/// time/value pairs, positions must be finite integers within
/// [`MAX_FILE_BLICK`], and values must be finite. Sorting and duplicate
/// handling mirror `normaliseCurve` (via `normaliseProject` after load):
/// points are stable-sorted by position and the last point at each position
/// wins, matching the editor behavior where later edits overwrite.
///
/// Unknown curve keys are preserved (`ProjectFile.cpp:261` strips only
/// `mode`/`points`); absent curves keep the default empty string.
///
/// Per-mode vocal curves bypass the sort/dedup via `normalise = false`,
/// because the C++ `normaliseProject` never normalises them (only
/// `pitchDelta`/`vibratoEnv`/`systemPitchDelta`).
fn read_curve_inner(value: Option<&Value>, name: &str, normalise: bool) -> Result<Curve> {
    let Some(value) = value else {
        return Ok(Curve::default());
    };
    if value.is_null() {
        return Ok(Curve::default());
    }
    let Value::Object(curve) = value else {
        return invalid(format!("{name} must contain a mode and points array."));
    };
    let raw: RawCurve = serde_json::from_value(value.clone())
        .map_err(|_| SvpError::Invalid(format!("{name} must contain a mode and points array.")))?;
    let mode: Interpolation = raw
        .mode
        .parse()
        .map_err(|_| SvpError::Invalid(format!("{name} has an unsupported interpolation mode.")))?;
    if raw.points.len() % 2 != 0 {
        return invalid(format!("{name} points must contain time/value pairs."));
    }
    let mut points = Vec::with_capacity(raw.points.len() / 2);
    for pair in raw.points.chunks_exact(2) {
        if pair[0].as_i64().is_none() && pair[0].as_f64().is_none()
            || pair[1].as_i64().is_none() && pair[1].as_f64().is_none()
        {
            return invalid(format!("{name} contains non-numeric points."));
        }
        let position = match pair[0].as_i64() {
            // Integer JSON keeps its exact value, as in the C++
            // `static_cast<Blick>(position)` of the original `juce::var`.
            Some(exact) if exact.unsigned_abs() <= MAX_FILE_BLICK.unsigned_abs() => exact,
            Some(_) => return invalid(format!("{name} contains an invalid point.")),
            // Doubles must be finite integers in range, as in the C++
            // `trunc`/`maximumFileBlick` checks on `positionNumber`.
            None => match pair[0].as_f64() {
                Some(number)
                    if number.is_finite()
                        && number.trunc() == number
                        && number.abs() <= MAX_FILE_BLICK as f64 =>
                {
                    number as Blick
                }
                _ => return invalid(format!("{name} contains an invalid point.")),
            },
        };
        match pair[1].as_f64() {
            Some(value) if value.is_finite() => points.push(AutomationPoint { position, value }),
            _ => return invalid(format!("{name} contains an invalid point.")),
        }
    }
    if !normalise {
        return Ok(Curve {
            mode,
            points,
            extra: preserve(curve, &["mode", "points"])?,
        });
    }
    if points.len() > 1
        && points
            .windows(2)
            .any(|pair| pair[0].position > pair[1].position)
    {
        points.sort_by_key(|point| point.position);
    }
    // Deduplicate keeping the last point at each position: after the stable
    // sort above, file-later duplicates come later, so a reverse pass keeps
    // them, exactly like the reverse `std::unique` in `normaliseCurve`.
    let mut deduped = Vec::with_capacity(points.len());
    for point in points.into_iter().rev() {
        if deduped.last().map(|last: &AutomationPoint| last.position) != Some(point.position) {
            deduped.push(point);
        }
    }
    deduped.reverse();
    Ok(Curve {
        mode,
        points: deduped,
        extra: preserve(curve, &["mode", "points"])?,
    })
}

impl RawReference {
    fn into_reference(value: &Value, base_dir: &Path) -> Result<GroupReference> {
        let raw: RawReference = serde_json::from_value(value.clone())?;
        let (Some(offset_raw), Some(pitch_raw)) = (
            read_integer(&raw.blick_offset),
            read_integer(&raw.pitch_offset),
        ) else {
            return invalid("a group reference has an invalid pitch offset");
        };
        let pitch_offset = i32::try_from(pitch_raw)
            .ok()
            .filter(|offset| (-127..=127).contains(offset));
        let Some(pitch_offset) = pitch_offset else {
            return invalid("a group reference has an invalid pitch offset");
        };
        let begin = match raw.blick_absolute_begin.as_ref() {
            None => 0,
            Some(raw) => {
                let Some(bound) = read_integer(raw) else {
                    return invalid("a group reference has invalid crop boundaries");
                };
                blick(bound, "a crop boundary")?
            }
        };
        let end = match raw.blick_absolute_end.as_ref() {
            // The format's sentinel for "no right crop boundary".
            None => None,
            Some(raw) => match read_integer(raw) {
                Some(-1) => None,
                Some(bound) => Some(blick(bound, "a crop boundary")?),
                None => return invalid("a group reference has invalid crop boundaries"),
            },
        };
        if begin < 0 || end.is_some_and(|end| end <= begin) {
            return invalid("a group reference has invalid crop boundaries");
        }
        let (vocal_mode_inherited, vocal_mode_preset, vocal_mode_params) =
            read_voice_overrides(raw.voice.as_ref())?;
        // Reference pitch attributes come from the SAME `voice` object, which
        // doubles as the `voicePitch` container in the C++ `readReference`.
        let voice_pitch = read_pitch_attributes(raw.voice.as_ref(), "reference voice")?;
        // Backing-track metadata is read regardless of the instrumental flag,
        // as in the C++: the `audio` block sits outside any such check.
        let (audio_file, audio_duration_seconds) = read_audio(raw.audio.as_ref(), base_dir)?;
        let Some(reference) = value.as_object() else {
            return invalid("a group reference has an invalid pitch offset");
        };
        // C++ `ProjectFile.cpp:484` strips the handled keys but NOT `audio`:
        // the saver overwrites it from `audioFile`, so the round trip still
        // works. `systemPitchDelta` is a handled key like the group curves:
        // it reads into `system_pitch_delta` with the same `readCurve`
        // validation and `normaliseCurve` sort/dedup as its siblings.
        let system_pitch_delta =
            read_curve_inner(reference.get("systemPitchDelta"), "systemPitchDelta", true)?;
        let extra = preserve(
            reference,
            &[
                "groupID",
                "blickOffset",
                "pitchOffset",
                "blickAbsoluteBegin",
                "blickAbsoluteEnd",
                "isInstrumental",
                "voice",
                "systemPitchDelta",
            ],
        )?;
        Ok(GroupReference {
            group_id: raw.group_id,
            time_offset: blick(offset_raw, "a group offset")?,
            pitch_offset,
            begin,
            end,
            instrumental: raw.is_instrumental,
            audio_file,
            audio_duration_seconds,
            vocal_mode_inherited,
            vocal_mode_preset,
            vocal_mode_params,
            voice_pitch,
            system_pitch_delta,
            extra,
        })
    }
}

impl RawTrack {
    fn into_track(value: &Value, base_dir: &Path) -> Result<Track> {
        let raw: RawTrack = serde_json::from_value(value.clone())?;
        let voice = match raw.sv_voice {
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

        let main_group = RawGroup::into_group(&raw.main_group)?;
        let main_ref = RawReference::into_reference(&raw.main_ref, base_dir)?;
        if main_ref.group_id != main_group.id {
            return invalid("a track's mainRef does not refer to its mainGroup");
        }

        let mixer = raw.mixer;
        let gain = 10.0_f64.powf(mixer.gain_decibel / 20.0);
        if !gain.is_finite() || !mixer.pan.is_finite() || !(-1.0..=1.0).contains(&mixer.pan) {
            return invalid("a track has invalid mixer settings");
        }

        let groups = raw
            .groups
            .into_iter()
            .map(|reference| RawReference::into_reference(&reference, base_dir))
            .collect::<Result<_>>()?;
        let Some(track) = value.as_object() else {
            return invalid("a track needs a groups array");
        };
        // C++ `ProjectFile.cpp:498` strips the five handled keys, keeping the
        // `mixer` blob and unknown keys verbatim.
        let extra = preserve(
            track,
            &["name", "mainGroup", "mainRef", "groups", "svVoice"],
        )?;
        Ok(Track {
            name: raw.name,
            main_group,
            main_ref,
            groups,
            mixer: Mixer {
                gain,
                pan: mixer.pan,
                mute: mixer.mute,
                solo: mixer.solo,
            },
            voice,
            extra,
        })
    }
}

// ---------------------------------------------------------------------------
// Serialization (`write*` in `src/core/ProjectFile.cpp`)
// ---------------------------------------------------------------------------

/// Mirrors OpenSV's `retainedObject`: restores the preserved JSON blob, or an
/// empty object when the blob is empty or malformed. `Map::insert` keeps
/// existing-key positions, matching JUCE `setProperty` semantics (overwritten
/// keys stay put, new keys append) — so the writer reproduces the C++ key
/// order, including its append-on-first-save behavior for default keys.
fn retained_object(extra: &Extra) -> Map<String, Value> {
    if extra.is_empty() {
        return Map::new();
    }
    serde_json::from_str(extra.json()).unwrap_or_default()
}

/// Mirrors OpenSV's `objectProperty`: clones a nested object property, or an
/// empty object when it is absent or not an object.
fn object_property(parent: &Map<String, Value>, name: &str) -> Map<String, Value> {
    parent
        .get(name)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

/// Mirrors OpenSV's `setDefault`: inserts only when the key is missing.
fn set_default(object: &mut Map<String, Value>, name: &str, value: Value) {
    if !object.contains_key(name) {
        object.insert(name.to_owned(), value);
    }
}

/// Mirrors OpenSV's `defaultTakes`: `{activeTakeId: 0, takes: [{id: 0,
/// expr: 0.0, liked: false}]}`.
fn default_takes() -> Value {
    serde_json::json!({
        "activeTakeId": 0,
        "takes": [{ "id": 0, "expr": 0.0, "liked": false }],
    })
}

fn number(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// Mirrors OpenSV's `writePitchAttributes`: merges the preserved blob, then
/// sets present fields and removes absent ones.
fn write_pitch_attributes(attributes: &PitchAttributes) -> Map<String, Value> {
    let mut value = retained_object(&attributes.extra);
    for field in PitchAttribute::ALL {
        match attributes.get(field) {
            Some(amount) => {
                value.insert(field.name().to_owned(), number(amount));
            }
            None => {
                value.shift_remove(field.name());
            }
        }
    }
    value
}

/// Mirrors OpenSV's `writeCurve`: merges the preserved blob, then sets
/// `mode` and the flat `[position, value, ...]` points (positions as
/// integers).
fn write_curve(curve: &Curve) -> Map<String, Value> {
    let mut value = retained_object(&curve.extra);
    value.insert("mode".to_owned(), Value::String(curve.mode.as_str().into()));
    let mut points = Vec::with_capacity(curve.points.len() * 2);
    for point in &curve.points {
        points.push(Value::from(point.position));
        points.push(number(point.value));
    }
    value.insert("points".to_owned(), Value::Array(points));
    value
}

/// Converts a stored absolute path back to project-relative form, mirroring
/// `juce::File::getRelativePathFrom`: paths under the project directory lose
/// their prefix, anything else keeps its absolute form.
fn relativize(base_dir: &Path, path: &Path) -> String {
    path.strip_prefix(base_dir).map_or_else(
        |_| path.to_string_lossy().into_owned(),
        |relative| relative.to_string_lossy().into_owned(),
    )
}

fn write_note(note: &Note) -> Value {
    let mut item = retained_object(&note.extra);
    item.insert("onset".to_owned(), Value::from(note.onset));
    item.insert("duration".to_owned(), Value::from(note.duration));
    item.insert("pitch".to_owned(), Value::from(note.pitch));
    item.insert("lyrics".to_owned(), Value::String(note.lyrics.clone()));
    item.insert("phonemes".to_owned(), Value::String(note.phonemes.clone()));
    item.insert(
        "musicalType".to_owned(),
        Value::String(note.musical_type.clone()),
    );
    item.insert("accent".to_owned(), Value::String(note.accent.clone()));
    item.insert("detune".to_owned(), number(note.detune));
    item.insert("instantMode".to_owned(), Value::Bool(note.instant_mode));
    let mut attributes = write_pitch_attributes(&note.attributes);
    set_default(&mut attributes, "evenSyllableDuration", Value::Bool(true));
    item.insert("attributes".to_owned(), Value::Object(attributes));
    let system_attributes = write_pitch_attributes(&note.system_attributes);
    if system_attributes.is_empty() {
        item.shift_remove("systemAttributes");
    } else {
        item.insert(
            "systemAttributes".to_owned(),
            Value::Object(system_attributes),
        );
    }
    set_default(&mut item, "pitchTakes", default_takes());
    set_default(&mut item, "timbreTakes", default_takes());
    Value::Object(item)
}

fn write_group(group: &NoteGroup) -> Value {
    let mut value = retained_object(&group.extra);
    value.insert("uuid".to_owned(), Value::String(group.id.clone()));
    value.insert("name".to_owned(), Value::String(group.name.clone()));
    value.insert(
        "notes".to_owned(),
        Value::Array(group.notes.iter().map(write_note).collect()),
    );
    let mut parameters = object_property(&value, "parameters");
    for name in [
        "loudness",
        "tension",
        "breathiness",
        "voicing",
        "gender",
        "toneShift",
    ] {
        let mut curve = object_property(&parameters, name);
        set_default(&mut curve, "mode", Value::String("cubic".into()));
        set_default(&mut curve, "points", Value::Array(Vec::new()));
        parameters.insert(name.to_owned(), Value::Object(curve));
    }
    parameters.insert(
        "pitchDelta".to_owned(),
        Value::Object(write_curve(&group.pitch_delta)),
    );
    parameters.insert(
        "vibratoEnv".to_owned(),
        Value::Object(write_curve(&group.vibrato_env)),
    );
    value.insert("parameters".to_owned(), Value::Object(parameters));
    // `vocalModes` is rebuilt fresh from the members (C++ `writeGroup`),
    // so plain amounts normalize to `{enabled, amount}` objects here.
    let mut vocal_modes = Map::new();
    for (name, setting) in &group.vocal_modes {
        let mut mode = retained_object(&setting.extra);
        mode.insert("enabled".to_owned(), Value::Bool(setting.enabled));
        mode.insert("amount".to_owned(), number(setting.amount));
        if setting.has_curve {
            let curve = write_curve(&setting.curve);
            if let Some(mode_value) = curve.get("mode") {
                mode.insert("mode".to_owned(), mode_value.clone());
            }
            if let Some(points) = curve.get("points") {
                mode.insert("points".to_owned(), points.clone());
            }
        }
        vocal_modes.insert(name.clone(), Value::Object(mode));
    }
    value.insert("vocalModes".to_owned(), Value::Object(vocal_modes));
    Value::Object(value)
}

fn write_reference(reference: &GroupReference, base_dir: &Path) -> Value {
    let mut value = retained_object(&reference.extra);
    value.insert(
        "groupID".to_owned(),
        Value::String(reference.group_id.clone()),
    );
    value.insert("blickOffset".to_owned(), Value::from(reference.time_offset));
    value.insert(
        "pitchOffset".to_owned(),
        Value::from(reference.pitch_offset),
    );
    value.insert(
        "blickAbsoluteBegin".to_owned(),
        Value::from(reference.begin),
    );
    value.insert(
        "blickAbsoluteEnd".to_owned(),
        Value::from(reference.end.unwrap_or(-1)),
    );
    value.insert(
        "isInstrumental".to_owned(),
        Value::Bool(reference.instrumental),
    );
    let mut database = object_property(&value, "database");
    for name in [
        "name",
        "language",
        "phoneset",
        "languageOverride",
        "phonesetOverride",
        "backendType",
    ] {
        set_default(&mut database, name, Value::String(String::new()));
    }
    set_default(&mut database, "version", Value::String("-2".into()));
    value.insert("database".to_owned(), Value::Object(database));
    set_default(&mut value, "dictionary", Value::String(String::new()));
    let mut voice = write_pitch_attributes(&reference.voice_pitch);
    voice.insert(
        "vocalModeInherited".to_owned(),
        Value::Bool(reference.vocal_mode_inherited),
    );
    voice.insert(
        "vocalModePreset".to_owned(),
        Value::String(reference.vocal_mode_preset.clone()),
    );
    let mut mode_params = Map::new();
    for (name, amount) in &reference.vocal_mode_params {
        mode_params.insert(name.clone(), number(*amount));
    }
    voice.insert("vocalModeParams".to_owned(), Value::Object(mode_params));
    value.insert("voice".to_owned(), Value::Object(voice));
    set_default(&mut value, "pitchTakes", default_takes());
    set_default(&mut value, "timbreTakes", default_takes());
    // The imported system contour round-trips like the group curves (C++
    // `writeReference` always emits it, defaulting to the empty curve).
    value.insert(
        "systemPitchDelta".to_owned(),
        Value::Object(write_curve(&reference.system_pitch_delta)),
    );
    if let Some(audio_file) = reference.audio_file.as_ref() {
        let mut audio = object_property(&value, "audio");
        audio.insert(
            "filename".to_owned(),
            Value::String(relativize(base_dir, audio_file)),
        );
        audio.insert(
            "duration".to_owned(),
            number(reference.audio_duration_seconds),
        );
        value.insert("audio".to_owned(), Value::Object(audio));
    }
    Value::Object(value)
}

fn write_track(track: &Track, index: usize, base_dir: &Path) -> Value {
    let mut value = retained_object(&track.extra);
    value.insert("name".to_owned(), Value::String(track.name.clone()));
    set_default(&mut value, "dispColor", Value::String("ff7db235".into()));
    value.insert(
        "dispOrder".to_owned(),
        Value::from(i64::try_from(index).unwrap_or(i64::MAX)),
    );
    set_default(&mut value, "renderEnabled", Value::Bool(false));
    value.insert("mainGroup".to_owned(), write_group(&track.main_group));
    value.insert(
        "mainRef".to_owned(),
        write_reference(&track.main_ref, base_dir),
    );
    if track.voice.database.is_some()
        || track.voice.dictionary.is_some()
        || track.voice.language != Language::Japanese
    {
        let mut voice = Map::new();
        voice.insert(
            "databasePath".to_owned(),
            Value::String(
                track
                    .voice
                    .database
                    .as_deref()
                    .map_or_else(String::new, |path| relativize(base_dir, path)),
            ),
        );
        voice.insert(
            "language".to_owned(),
            Value::String(track.voice.language.as_str().into()),
        );
        voice.insert(
            "dictionaryDirectory".to_owned(),
            Value::String(
                track
                    .voice
                    .dictionary
                    .as_deref()
                    .map_or_else(String::new, |path| relativize(base_dir, path)),
            ),
        );
        value.insert("svVoice".to_owned(), Value::Object(voice));
    }
    let mut mixer = object_property(&value, "mixer");
    // Preserve the source dB value when no linear gain edit has occurred
    // (exact `f64` comparison, as in the C++).
    let gain_decibel = if track.mixer.gain > 0.0 {
        20.0 * track.mixer.gain.log10()
    } else {
        -100.0
    };
    let keep_decibel = mixer
        .get("gainDecibel")
        .and_then(Value::as_f64)
        .is_some_and(|original| 10.0_f64.powf(original / 20.0) == track.mixer.gain);
    if !mixer.contains_key("gainDecibel") || !keep_decibel {
        mixer.insert("gainDecibel".to_owned(), number(gain_decibel));
    }
    mixer.insert("pan".to_owned(), number(track.mixer.pan));
    mixer.insert("mute".to_owned(), Value::Bool(track.mixer.mute));
    mixer.insert("solo".to_owned(), Value::Bool(track.mixer.solo));
    set_default(&mut mixer, "display", Value::Bool(true));
    value.insert("mixer".to_owned(), Value::Object(mixer));
    value.insert(
        "groups".to_owned(),
        Value::Array(
            track
                .groups
                .iter()
                .map(|reference| write_reference(reference, base_dir))
                .collect(),
        ),
    );
    Value::Object(value)
}

/// Serializes a project to multi-line JSON, mirroring `saveProjectFile`
/// (`src/core/ProjectFile.cpp:899`) without the trailing NUL byte or the
/// atomic file replacement (see [`save`]). `base_dir` is the directory voice,
/// dictionary, and backing-track paths relativize against (the file's parent
/// on [`save`]).
pub fn serialize(project: &Project, base_dir: &Path) -> Result<String> {
    let mut root = retained_object(&project.extra);
    root.insert("version".to_owned(), Value::from(FORMAT_VERSION));
    let mut time = object_property(&root, "time");
    time.insert(
        "tempo".to_owned(),
        Value::Array(
            project
                .tempo_map
                .tempos()
                .iter()
                .map(|tempo| {
                    let mut value = retained_object(&tempo.extra);
                    value.insert("position".to_owned(), Value::from(tempo.position));
                    value.insert("bpm".to_owned(), number(tempo.bpm));
                    Value::Object(value)
                })
                .collect(),
        ),
    );
    time.insert(
        "meter".to_owned(),
        Value::Array(
            project
                .tempo_map
                .time_signatures()
                .iter()
                .map(|signature| {
                    let mut value = retained_object(&signature.extra);
                    value.insert("index".to_owned(), Value::from(signature.bar));
                    value.insert("numerator".to_owned(), Value::from(signature.numerator));
                    value.insert("denominator".to_owned(), Value::from(signature.denominator));
                    Value::Object(value)
                })
                .collect(),
        ),
    );
    root.insert("time".to_owned(), Value::Object(time));
    root.insert(
        "library".to_owned(),
        Value::Array(project.library.iter().map(write_group).collect()),
    );
    root.insert(
        "tracks".to_owned(),
        Value::Array(
            project
                .tracks
                .iter()
                .enumerate()
                .map(|(index, track)| write_track(track, index, base_dir))
                .collect(),
        ),
    );
    let mut render = object_property(&root, "renderConfig");
    set_default(&mut render, "destination", Value::String(String::new()));
    set_default(&mut render, "filename", Value::String(project.name.clone()));
    set_default(&mut render, "numChannels", Value::from(1));
    set_default(
        &mut render,
        "aspirationFormat",
        Value::String("noAspiration".into()),
    );
    set_default(&mut render, "bitDepth", Value::from(16));
    set_default(&mut render, "sampleRate", Value::from(44100));
    set_default(&mut render, "exportMixDown", Value::Bool(true));
    set_default(&mut render, "exportPitch", Value::Bool(false));
    root.insert("renderConfig".to_owned(), Value::Object(render));
    serde_json::to_string_pretty(&Value::Object(root)).map_err(SvpError::Json)
}

/// Saves a project to disk: multi-line JSON plus the trailing NUL byte
/// Synthesizer V appends, written atomically via a sibling temporary file
/// and rename, as in `saveProjectFile`.
pub fn save(path: &Path, project: &Project) -> Result<()> {
    let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
    let mut bytes = serialize(project, base_dir)?.into_bytes();
    bytes.push(0);
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, &bytes).map_err(|source| SvpError::Io {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| SvpError::Io {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use opensvr_core::{BLICKS_PER_QUARTER as Q, Language, sample_curve};
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

    /// Sets a JSON-pointer target, creating the leaf key when the sample omits
    /// it (crop bounds). Only the two crop leaves are ever absent, so the
    /// fallback inserts a final object key under an existing parent.
    fn set_pointer(value: &mut Value, pointer: &str, replacement: Value) {
        if let Some(target) = value.pointer_mut(pointer) {
            *target = replacement;
            return;
        }
        let parent_end = pointer.rfind('/').unwrap();
        let (parent, leaf) = pointer.split_at(parent_end);
        let leaf = leaf.trim_start_matches('/');
        let leaf: &str = leaf;
        value.pointer_mut(parent).unwrap()[leaf] = replacement;
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
    fn integer_valued_floats_load_like_integers() {
        let plain = parse_value(&sample()).unwrap();
        let mut value = sample();
        // One `.0` spelling per `readInteger` call site (note timing/pitch,
        // reference offsets, tempo position, meter fields). Crop bounds are
        // covered separately below so this stays value-identical to `plain`.
        value["tracks"][0]["mainGroup"]["notes"][0]["onset"] = json!(705_600_000.0);
        value["tracks"][0]["mainGroup"]["notes"][0]["duration"] = json!(705_600_000.0);
        value["tracks"][0]["mainGroup"]["notes"][0]["pitch"] = json!(62.0);
        value["tracks"][0]["mainGroup"]["notes"][1]["onset"] = json!(0.0);
        value["tracks"][0]["mainGroup"]["notes"][1]["duration"] = json!(705_600_000.0);
        value["tracks"][0]["mainGroup"]["notes"][1]["pitch"] = json!(60.0);
        value["tracks"][0]["groups"][0]["blickOffset"] = json!(0.0);
        value["tracks"][0]["groups"][0]["pitchOffset"] = json!(0.0);
        value["time"]["tempo"][0]["position"] = json!(0.0);
        value["time"]["meter"][0]["index"] = json!(0.0);
        value["time"]["meter"][0]["numerator"] = json!(3.0);
        value["time"]["meter"][0]["denominator"] = json!(4.0);
        let dotted = parse_value(&value).unwrap();
        assert_eq!(dotted.tracks, plain.tracks);
        assert_eq!(dotted.library, plain.library);
        assert_eq!(dotted.tempo_map, plain.tempo_map);
        // Integral crop doubles keep their values (the sample omits them, so
        // they are set explicitly here).
        let mut value = sample();
        value["tracks"][0]["groups"][0]["blickAbsoluteBegin"] = json!(705_600_000.0);
        value["tracks"][0]["groups"][0]["blickAbsoluteEnd"] = json!(2_116_800_000.0);
        let project = parse_value(&value).unwrap();
        assert_eq!(
            (
                project.tracks[0].groups[0].begin,
                project.tracks[0].groups[0].end
            ),
            (705_600_000, Some(2_116_800_000))
        );
        assert_eq!(project.tracks[0].main_group.notes[1].pitch, 62);
        // The `-1.0` sentinel still means unbounded, as in the C++ cast.
        let mut value = sample();
        value["tracks"][0]["groups"][0]["blickAbsoluteBegin"] = json!(0.0);
        value["tracks"][0]["groups"][0]["blickAbsoluteEnd"] = json!(-1.0);
        let project = parse_value(&value).unwrap();
        assert_eq!(project.tracks[0].groups[0].end, None);
    }

    #[test]
    fn rejects_non_integral_and_out_of_range_integers() {
        // Non-integral doubles fail with the field's `Invalid` error, never a
        // JSON error, exactly like C++ `readInteger` returning false.
        let pointers = [
            "/tracks/0/mainGroup/notes/0/onset",
            "/tracks/0/mainGroup/notes/0/duration",
            "/tracks/0/mainGroup/notes/0/pitch",
            "/tracks/0/groups/0/blickOffset",
            "/tracks/0/groups/0/pitchOffset",
            "/tracks/0/groups/0/blickAbsoluteBegin",
            "/tracks/0/groups/0/blickAbsoluteEnd",
            "/time/tempo/0/position",
            "/time/meter/0/index",
            "/time/meter/0/numerator",
            "/time/meter/0/denominator",
        ];
        for pointer in pointers {
            let mut value = sample();
            // `0.5` is the non-integral probe; the crop-end sentinel path
            // uses `-0.5` so it cannot be mistaken for `-1`. Crop keys are
            // absent from the sample, so they are created via the parent
            // object when the JSON pointer has no target yet.
            let probe = if pointer.ends_with("blickAbsoluteEnd") {
                json!(-0.5)
            } else {
                json!(0.5)
            };
            set_pointer(&mut value, pointer, probe);
            assert!(
                matches!(parse_value(&value), Err(SvpError::Invalid(_))),
                "{pointer} should reject 0.5 with Invalid"
            );
            let mut value = sample();
            set_pointer(&mut value, pointer, json!(4_611_686_018_427_387_904i64));
            assert!(
                matches!(parse_value(&value), Err(SvpError::Invalid(_))),
                "{pointer} should reject out-of-range magnitudes with Invalid"
            );
        }
        // Out-of-range doubles are rejected too, not just out-of-range ints.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["onset"] = json!(4_611_686_018_427_387_904.0);
        assert!(matches!(parse_value(&value), Err(SvpError::Invalid(_))));
    }

    #[test]
    fn missing_parameters_keep_empty_curves() {
        let project = parse_value(&sample()).unwrap();
        for group in project
            .library
            .iter()
            .chain(std::iter::once(&project.tracks[0].main_group))
        {
            assert_eq!(group.pitch_delta, Curve::default());
            assert_eq!(group.vibrato_env, Curve::default());
            assert_eq!(group.pitch_delta.mode, Interpolation::Cubic);
        }
    }

    #[test]
    fn loads_group_curves() {
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["parameters"] = json!({
            "pitchDelta": { "mode": "linear", "points": [0, 0.0, 1000, 10.0] },
            "vibratoEnv": { "mode": "cosine", "points": [0, 1.0, 500, 3.0] },
        });
        let project = parse_value(&value).unwrap();
        let group = &project.tracks[0].main_group;
        assert_eq!(group.pitch_delta.mode, Interpolation::Linear);
        assert_eq!(
            group.pitch_delta.points,
            vec![
                AutomationPoint {
                    position: 0,
                    value: 0.0
                },
                AutomationPoint {
                    position: 1000,
                    value: 10.0
                },
            ]
        );
        assert_eq!(group.vibrato_env.mode, Interpolation::Cosine);
        // Linear midpoint is exact arithmetic, so this checks the wiring
        // from file bytes to evaluated curve (interpolation itself is
        // covered by `opensvr-core` differential fixtures).
        assert_eq!(sample_curve(&group.pitch_delta, 500, 0.0), 5.0);
        // Library groups without `parameters` stay empty and inert.
        assert_eq!(project.library[0].pitch_delta, Curve::default());
    }

    #[test]
    fn rejects_malformed_curves() {
        let cases: [(&str, Value); 8] = [
            // Odd-length arrays are not time/value pairs.
            (
                "pitchDelta",
                json!({ "mode": "linear", "points": [0, 1.0, 500] }),
            ),
            // Non-numeric entries.
            (
                "pitchDelta",
                json!({ "mode": "linear", "points": [0, 1.0, "500", 2.0] }),
            ),
            (
                "vibratoEnv",
                json!({ "mode": "linear", "points": [0, "loud"] }),
            ),
            // Non-integer positions.
            (
                "pitchDelta",
                json!({ "mode": "linear", "points": [0.5, 1.0, 100, 2.0] }),
            ),
            // Positions beyond `maximumFileBlick`.
            (
                "pitchDelta",
                json!({ "mode": "linear", "points": [0, 1.0, 4_611_686_018_427_387_904i64, 2.0] }),
            ),
            // Unknown interpolation modes.
            (
                "pitchDelta",
                json!({ "mode": "smooth", "points": [0, 1.0, 100, 2.0] }),
            ),
            // Curves must be objects with a mode and points array.
            ("pitchDelta", json!(5.0)),
            ("vibratoEnv", json!({ "mode": "linear" })),
        ];
        for (name, replacement) in cases {
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["parameters"][name] = replacement;
            assert!(parse_value(&value).is_err(), "{name} should be rejected");
        }
    }

    #[test]
    fn curves_normalise_like_the_cpp() {
        // `readCurve` accepts unsorted and duplicated positions; the later
        // `normaliseCurve` stable-sorts them and lets the last point at each
        // position win (later editor edits overwrite).
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["parameters"] = json!({
            "pitchDelta": {
                "mode": "linear",
                "points": [1000, 1.0, 0, 2.0, 1000, 3.0],
            },
        });
        let project = parse_value(&value).unwrap();
        assert_eq!(
            project.tracks[0].main_group.pitch_delta.points,
            vec![
                AutomationPoint {
                    position: 0,
                    value: 2.0
                },
                AutomationPoint {
                    position: 1000,
                    value: 3.0
                },
            ]
        );
    }

    #[test]
    fn system_pitch_delta_reads_like_its_siblings() {
        // `readCurve` plus `normaliseCurve`, as in `ProjectFile.cpp:480` and
        // `Project.cpp:312`: unsorted points sort, and the last point at each
        // position wins.
        let mut value = sample();
        value["tracks"][0]["mainRef"]["systemPitchDelta"] = json!({
            "mode": "linear",
            "points": [1000, 1.0, 0, 2.0, 1000, 3.0],
        });
        let project = parse_value(&value).unwrap();
        let contour = &project.tracks[0].main_ref.system_pitch_delta;
        assert_eq!(contour.mode, Interpolation::Linear);
        assert_eq!(
            contour.points,
            vec![
                AutomationPoint {
                    position: 0,
                    value: 2.0
                },
                AutomationPoint {
                    position: 1000,
                    value: 3.0
                },
            ]
        );
        // Absent contours stay at the C++ default: an empty cubic curve.
        let plain = parse_value(&sample()).unwrap();
        assert_eq!(
            plain.tracks[0].main_ref.system_pitch_delta,
            Curve::default()
        );
        assert_eq!(
            plain.tracks[0].main_ref.system_pitch_delta.mode,
            Interpolation::Cubic
        );
    }

    #[test]
    fn non_default_system_pitch_delta_round_trips() {
        // The case that failed before this generation: `load -> save -> load`
        // dropped every non-default contour because the reader stripped it.
        let mut value = sample();
        value["tracks"][0]["mainRef"]["systemPitchDelta"] = json!({
            "mode": "linear",
            "points": [0, 10.0, 1000, -5.0],
        });
        let project = parse_value(&value).unwrap();
        let saved = reserialize(&project);
        let root: Value = serde_json::from_str(&saved).unwrap();
        let contour = &root["tracks"][0]["mainRef"]["systemPitchDelta"];
        assert_eq!(contour["mode"], json!("linear"));
        assert_eq!(contour["points"], json!([0, 10.0, 1000, -5.0]));
        let back = reparse(&saved);
        assert_eq!(
            back.tracks[0].main_ref.system_pitch_delta,
            project.tracks[0].main_ref.system_pitch_delta
        );
        // The second cycle is byte-stable too. (Handled keys appended after
        // retained vendor keys change key order on the first save, exactly as
        // the C++ `set` calls do, so byte-identity only holds from the
        // second save on.)
        let saved_twice = reserialize(&back);
        let as_value = |text: &str| serde_json::from_str::<Value>(text).unwrap();
        assert_eq!(as_value(&saved), as_value(&saved_twice));
        assert_eq!(reserialize(&reparse(&saved_twice)), saved_twice);
    }

    #[test]
    fn rejects_malformed_system_pitch_delta() {
        for bad in [
            json!({ "mode": "smooth", "points": [0, 1.0, 100, 2.0] }),
            json!({ "mode": "linear", "points": [0, 1.0, 500] }),
            json!({ "mode": "linear", "points": [0.5, 1.0] }),
            json!(5.0),
        ] {
            let mut value = sample();
            value["tracks"][0]["groups"][0]["systemPitchDelta"] = bad;
            assert!(
                parse_value(&value).is_err(),
                "a malformed systemPitchDelta should be rejected"
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

    #[test]
    fn missing_vocal_modes_keep_defaults() {
        let project = parse_value(&sample()).unwrap();
        let track = &project.tracks[0];
        assert!(track.main_group.vocal_modes.is_empty());
        assert!(project.library[0].vocal_modes.is_empty());
        assert!(track.main_ref.vocal_mode_inherited);
        assert_eq!(track.main_ref.vocal_mode_preset, "");
        assert!(track.main_ref.vocal_mode_params.is_empty());
        assert!(track.groups[0].vocal_mode_inherited);
    }

    #[test]
    fn loads_vocal_modes_and_voice_overrides() {
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["vocalModes"] = json!({
            "Power": 0.8,
            "Soft": { "enabled": false, "amount": 0.3 },
            "Airy": {
                "enabled": true,
                "amount": 0.5,
                "mode": "linear",
                "points": [0, 0.0, 1000, 1.0],
            },
        });
        value["tracks"][0]["groups"][0]["voice"] = json!({
            "vocalModeInherited": false,
            "vocalModePreset": "Chest",
            "vocalModeParams": { "Power": 0.9 },
        });
        let project = parse_value(&value).unwrap();
        let group = &project.tracks[0].main_group;
        assert_eq!(group.vocal_modes["Power"].amount, 0.8);
        assert!(group.vocal_modes["Power"].enabled);
        assert!(!group.vocal_modes["Power"].has_curve);
        assert!(!group.vocal_modes["Soft"].enabled);
        assert_eq!(group.vocal_modes["Soft"].amount, 0.3);
        let airy = &group.vocal_modes["Airy"];
        assert!(airy.has_curve);
        assert_eq!(sample_curve(&airy.curve, 500, 0.0), 0.5);
        // `std::map` ordering is mirrored with `BTreeMap`.
        let names: Vec<_> = group.vocal_modes.keys().cloned().collect();
        assert_eq!(names, ["Airy", "Power", "Soft"]);

        let reference = &project.tracks[0].groups[0];
        assert!(!reference.vocal_mode_inherited);
        assert_eq!(reference.vocal_mode_preset, "Chest");
        assert_eq!(reference.vocal_mode_params["Power"], 0.9);
    }

    #[test]
    fn vocal_mode_curves_stay_unnormalised() {
        // `normaliseProject` sorts `pitchDelta` but leaves per-mode curves in
        // file order, duplicates included.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["parameters"] = json!({
            "pitchDelta": {
                "mode": "linear",
                "points": [1000, 1.0, 0, 2.0],
            },
        });
        value["tracks"][0]["mainGroup"]["vocalModes"] = json!({
            "Power": {
                "amount": 0.5,
                "mode": "linear",
                "points": [1000, 1.0, 0, 2.0, 1000, 3.0],
            },
        });
        let project = parse_value(&value).unwrap();
        let group = &project.tracks[0].main_group;
        assert_eq!(group.pitch_delta.points[0].position, 0);
        assert_eq!(
            group.vocal_modes["Power"]
                .curve
                .points
                .iter()
                .map(|point| (point.position, point.value))
                .collect::<Vec<_>>(),
            [(1000, 1.0), (0, 2.0), (1000, 3.0)]
        );
    }

    #[test]
    fn missing_pitch_attributes_keep_defaults() {
        let project = parse_value(&sample()).unwrap();
        let track = &project.tracks[0];
        for note in &track.main_group.notes {
            assert_eq!(note.attributes, PitchAttributes::default());
            assert_eq!(note.system_attributes, PitchAttributes::default());
        }
        assert_eq!(track.main_ref.voice_pitch, PitchAttributes::default());
        assert_eq!(track.groups[0].voice_pitch, PitchAttributes::default());
        // JSON null parses to void in JUCE, i.e. absent.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] = json!(null);
        value["tracks"][0]["mainGroup"]["notes"][0]["systemAttributes"] = json!(null);
        value["tracks"][0]["groups"][0]["voice"] = json!(null);
        let project = parse_value(&value).unwrap();
        assert_eq!(
            project.tracks[0].main_group.notes[1].attributes,
            PitchAttributes::default()
        );
        assert_eq!(
            project.tracks[0].groups[0].voice_pitch,
            PitchAttributes::default()
        );
    }

    #[test]
    fn loads_pitch_attributes_through_the_shared_table() {
        use opensvr_core::PitchAttribute;
        let mut fields = serde_json::Map::new();
        for (index, field) in PitchAttribute::ALL.iter().enumerate() {
            fields.insert(field.name().into(), json!(index as f64 + 0.5));
        }
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] = json!(fields.clone());
        value["tracks"][0]["mainGroup"]["notes"][0]["systemAttributes"] = json!(fields.clone());
        // The reference `voice` object carries vocal-mode keys and pitch
        // attributes side by side, as in the C++ `readReference`.
        let mut voice = fields.clone();
        voice.insert("vocalModeInherited".into(), json!(false));
        voice.insert("vocalModePreset".into(), json!("Chest"));
        value["tracks"][0]["groups"][0]["voice"] = json!(voice);
        let project = parse_value(&value).unwrap();
        let track = &project.tracks[0];
        for attributes in [
            &track.main_group.notes[1].attributes,
            &track.main_group.notes[1].system_attributes,
            &track.groups[0].voice_pitch,
        ] {
            for (index, field) in PitchAttribute::ALL.iter().enumerate() {
                assert_eq!(attributes.get(*field), Some(index as f64 + 0.5));
            }
        }
        // The vocal-mode keys on the same object still parse.
        assert!(!track.groups[0].vocal_mode_inherited);
        assert_eq!(track.groups[0].vocal_mode_preset, "Chest");
        // Integer JSON counts as a finite number, as in C++ `readNumber`.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] = json!({ "tF0Offset": 1 });
        let project = parse_value(&value).unwrap();
        assert_eq!(
            project.tracks[0].main_group.notes[1].attributes.t_f0_offset,
            Some(1.0)
        );
        // Unlisted keys stay ignored until preserved fields land.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] =
            json!({ "tF0Offset": 0.25, "futureVendorKey": 1.0 });
        let project = parse_value(&value).unwrap();
        assert_eq!(
            project.tracks[0].main_group.notes[1].attributes.t_f0_offset,
            Some(0.25)
        );
    }

    #[test]
    fn rejects_malformed_pitch_attributes() {
        // One case per rule family (not 14 copies): non-object containers
        // and non-finite entries on each of the three paths.
        let bad_containers = [json!(5.0), json!([1.0])];
        for bad in bad_containers {
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] = bad.clone();
            assert!(
                parse_value(&value).is_err(),
                "malformed note attributes should be rejected"
            );
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["notes"][0]["systemAttributes"] = bad.clone();
            assert!(
                parse_value(&value).is_err(),
                "malformed note systemAttributes should be rejected"
            );
        }
        // A non-object `voice` is already rejected by the vocal-mode block;
        // here the object is fine but one pitch field is not a number.
        let mut value = sample();
        value["tracks"][0]["groups"][0]["voice"] =
            json!({ "vocalModeInherited": true, "tF0Offset": [1.0] });
        assert!(
            parse_value(&value).is_err(),
            "malformed reference voice pitch attributes should be rejected"
        );
        let bad_values = [json!("loud"), json!(true), json!(null), json!([0.5])];
        for bad in bad_values {
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["notes"][0]["attributes"] = json!({ "dF0Vbr": bad });
            assert!(
                parse_value(&value).is_err(),
                "non-numeric pitch attribute should be rejected"
            );
            let mut value = sample();
            value["tracks"][0]["groups"][0]["voice"] = json!({ "fF0Vbr": bad });
            assert!(
                parse_value(&value).is_err(),
                "non-numeric reference voice attribute should be rejected"
            );
        }
    }

    #[test]
    fn rejects_malformed_vocal_modes() {
        // Group-level shapes: non-object, non-boolean `enabled`,
        // neither-number-nor-object entries, and bad curves. (Plain JSON
        // numbers are always finite, so the finite-amount guard is
        // unreachable from text and needs no fixture.)
        let bad_groups = [
            json!(5.0),
            json!({ "Power": true }),
            json!({ "Power": null }),
            json!({ "Power": [0.5] }),
            json!({ "Power": { "enabled": "yes" } }),
            json!({ "Power": { "amount": "loud" } }),
            json!({ "Power": { "mode": "smooth", "points": [0, 1.0] } }),
            json!({ "Power": { "mode": "linear", "points": [0] } }),
        ];
        for bad in bad_groups {
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["vocalModes"] = bad;
            assert!(
                parse_value(&value).is_err(),
                "malformed vocalModes should be rejected"
            );
        }
        // Reference-level shapes: non-object voice, non-boolean inherited,
        // non-string preset, non-object params, non-finite param.
        let bad_voices = [
            json!(5.0),
            json!({ "vocalModeInherited": "yes" }),
            json!({ "vocalModePreset": 3 }),
            json!({ "vocalModeParams": 3.0 }),
            json!({ "vocalModeParams": { "Power": "loud" } }),
        ];
        for bad in bad_voices {
            let mut value = sample();
            value["tracks"][0]["groups"][0]["voice"] = bad;
            assert!(
                parse_value(&value).is_err(),
                "malformed voice overrides should be rejected"
            );
        }
    }

    #[test]
    fn loads_reference_audio_metadata() {
        // A full `audio` object loads on both instrumental and singing
        // references (the C++ block sits outside any instrumental check);
        // integer durations count, like C++ `isNumber`.
        for instrumental in [false, true] {
            let mut value = sample();
            value["tracks"][0]["groups"][0]["isInstrumental"] = json!(instrumental);
            value["tracks"][0]["groups"][0]["audio"] =
                json!({ "filename": "takes/vocal.wav", "duration": 90 });
            let project = parse_value(&value).unwrap();
            let reference = &project.tracks[0].groups[0];
            assert_eq!(reference.instrumental, instrumental);
            assert_eq!(
                reference.audio_file.as_deref(),
                Some(Path::new("/proj/takes/vocal.wav"))
            );
            assert_eq!(reference.audio_duration_seconds, 90.0);
        }
        // Absent and null `audio` keep the defaults; an empty filename stays
        // unset while still requiring a valid duration.
        let project = parse_value(&sample()).unwrap();
        assert_eq!(project.tracks[0].groups[0].audio_file, None);
        assert_eq!(project.tracks[0].groups[0].audio_duration_seconds, 0.0);
        let mut value = sample();
        value["tracks"][0]["groups"][0]["audio"] = json!(null);
        let project = parse_value(&value).unwrap();
        assert_eq!(project.tracks[0].groups[0].audio_file, None);
        let mut value = sample();
        value["tracks"][0]["groups"][0]["audio"] = json!({ "filename": "", "duration": 12.5 });
        let project = parse_value(&value).unwrap();
        assert_eq!(project.tracks[0].groups[0].audio_file, None);
        assert_eq!(project.tracks[0].groups[0].audio_duration_seconds, 12.5);
    }

    #[test]
    fn preserves_unknown_fields_at_every_level() {
        let value = json!({
            "version": 153,
            "vendorRoot": { "a": 1 },
            "time": {
                "tempo": [{ "position": 0, "bpm": 100.0, "vendorTempo": "t" }],
                "meter": [{ "index": 0, "numerator": 3, "denominator": 4, "vendorMeter": 2 }],
            },
            "library": [],
            "tracks": [{
                "name": "Lead",
                "vendorTrack": [1, 2],
                "mainGroup": {
                    "uuid": "main",
                    "name": "main",
                    "vendorGroup": true,
                    "notes": [{
                        "onset": 0,
                        "duration": Q,
                        "pitch": 60,
                        "lyrics": "do",
                        "accent": "strong",
                        "attributes": { "tF0Offset": 1.5, "vendorAttr": "a" },
                        "systemAttributes": { "vendorSys": 2 },
                        "vendorNote": "n",
                    }],
                    "parameters": {
                        "pitchDelta": { "mode": "linear", "points": [0, 0.0], "vendorCurve": "c" },
                        "vendorParam": 7,
                    },
                    "vocalModes": {
                        "Power": 0.5,
                        "Soft": { "enabled": false, "amount": 0.25, "vendorMode": "m" },
                    },
                },
                "mainRef": {
                    "groupID": "main",
                    "blickOffset": 0,
                    "pitchOffset": 0,
                    "vendorRef": "r",
                    "voice": { "vocalModePreset": "p", "vendorVoice": 1 },
                },
                "groups": [],
                "mixer": { "gainDecibel": 0.0, "pan": 0.0 },
                "svVoice": { "databasePath": "", "language": "japanese", "dictionaryDirectory": "" },
            }],
        });
        let project = parse_value(&value).unwrap();

        // Root keeps everything except tracks/library, in document order.
        assert_eq!(
            project.extra.json(),
            r#"{"version":153,"vendorRoot":{"a":1},"time":{"tempo":[{"position":0,"bpm":100.0,"vendorTempo":"t"}],"meter":[{"index":0,"numerator":3,"denominator":4,"vendorMeter":2}]}}"#
        );
        assert_eq!(
            project.tempo_map.tempos()[0].extra.json(),
            r#"{"vendorTempo":"t"}"#
        );
        assert_eq!(
            project.tempo_map.time_signatures()[0].extra.json(),
            r#"{"vendorMeter":2}"#
        );

        let track = &project.tracks[0];
        assert_eq!(
            track.extra.json(),
            r#"{"vendorTrack":[1,2],"mixer":{"gainDecibel":0.0,"pan":0.0}}"#
        );

        let group = &track.main_group;
        // The `parameters` blob survives minus the two curves Rust reads.
        assert_eq!(
            group.extra.json(),
            r#"{"vendorGroup":true,"parameters":{"vendorParam":7}}"#
        );
        // Handled curve keys are gone from the curve's own extra.
        assert_eq!(group.pitch_delta.extra.json(), r#"{"vendorCurve":"c"}"#);
        // A plain amount keeps the default empty string: the C++ `preserve`
        // call sits in the settings-object branch only.
        assert_eq!(group.vocal_modes["Power"].extra.json(), "");
        assert_eq!(
            group.vocal_modes["Soft"].extra.json(),
            r#"{"vendorMode":"m"}"#
        );

        let note = &group.notes[0];
        // The legacy `accent` key is stripped (a C++ member holds it) while
        // only truly unknown keys survive.
        assert_eq!(note.extra.json(), r#"{"vendorNote":"n"}"#);
        assert_eq!(note.attributes.get(PitchAttribute::TF0Offset), Some(1.5));
        assert_eq!(note.attributes.extra.json(), r#"{"vendorAttr":"a"}"#);
        assert_eq!(note.system_attributes.extra.json(), r#"{"vendorSys":2}"#);

        // The reference keeps `vendorRef`; the `voice` blob is stripped
        // (handled) while its own unknown keys land on `voice_pitch`.
        assert_eq!(track.main_ref.extra.json(), r#"{"vendorRef":"r"}"#);
        assert_eq!(track.main_ref.vocal_mode_preset, "p");
        assert_eq!(
            track.main_ref.voice_pitch.extra.json(),
            r#"{"vendorVoice":1}"#
        );
    }

    #[test]
    fn clean_objects_yield_the_cpp_empty_forms() {
        let project = parse_value(&sample()).unwrap();

        // Fully handled objects preserve to `"{}"`, exactly like JUCE's
        // `writeAsJSON` of an empty object; absent attributes/curves keep the
        // default empty string (C++ returns before its `preserve` call).
        assert_eq!(
            project.extra.json(),
            r#"{"version":153,"time":{"tempo":[{"position":0,"bpm":100.0}],"meter":[{"index":0,"numerator":3,"denominator":4}]}}"#
        );
        assert_eq!(project.tempo_map.tempos()[0].extra.json(), "{}");
        assert_eq!(project.tempo_map.time_signatures()[0].extra.json(), "{}");

        let track = &project.tracks[0];
        assert_eq!(
            track.extra.json(),
            r#"{"mixer":{"gainDecibel":-6.0,"pan":0.5,"mute":false,"solo":true}}"#
        );
        assert_eq!(track.main_ref.extra.json(), "{}");
        assert_eq!(track.main_ref.voice_pitch.extra.json(), "");

        // Groups always carry the (possibly empty) `parameters` blob: the C++
        // `set` writes it back even when the file omits it.
        for group in project
            .library
            .iter()
            .chain(std::iter::once(&track.main_group))
        {
            assert_eq!(group.extra.json(), r#"{"parameters":{}}"#);
            assert_eq!(group.pitch_delta.extra.json(), "");
        }
        let note = &track.main_group.notes[0];
        assert_eq!(note.extra.json(), "{}");
        assert_eq!(note.attributes.extra.json(), "");
    }

    #[test]
    fn rejects_malformed_reference_audio() {
        // Every malformed shape is the single error OpenSV's `readReference`
        // reports. `duration` is required: C++ `readNumber` fails on a void
        // (missing or null) value.
        let bad_audio = [
            json!(5.0),
            json!("take.wav"),
            json!({ "duration": 90.0 }),
            json!({ "filename": 3 }),
            json!({ "filename": "take.wav" }),
            json!({ "filename": "take.wav", "duration": "long" }),
            json!({ "filename": "take.wav", "duration": null }),
            json!({ "filename": "take.wav", "duration": -1.0 }),
        ];
        for bad in bad_audio {
            let mut value = sample();
            value["tracks"][0]["groups"][0]["audio"] = bad;
            assert!(
                parse_value(&value).is_err(),
                "malformed reference audio should be rejected"
            );
        }
        let mut value = sample();
        value["tracks"][0]["groups"][0]["audio"] = json!({ "filename": 3, "duration": 1.0 });
        match parse_value(&value) {
            Err(SvpError::Invalid(message)) => assert_eq!(
                message,
                "an instrumental reference has invalid audio metadata."
            ),
            other => panic!("expected the audio metadata error, got {other:?}"),
        }
    }

    #[test]
    fn reads_note_style_fields_with_defaults() {
        let project = parse_value(&sample()).unwrap();
        for note in &project.tracks[0].main_group.notes {
            assert_eq!(note.accent, "");
            assert!(note.instant_mode);
            assert_eq!(note.musical_type, "singing");
        }
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["accent"] = json!("strong");
        value["tracks"][0]["mainGroup"]["notes"][0]["instantMode"] = json!(false);
        value["tracks"][0]["mainGroup"]["notes"][0]["musicalType"] = json!("rap");
        let project = parse_value(&value).unwrap();
        let note = project.tracks[0]
            .main_group
            .notes
            .iter()
            .find(|note| note.onset == Q)
            .unwrap();
        assert_eq!(note.accent, "strong");
        assert!(!note.instant_mode);
        assert_eq!(note.musical_type, "rap");
        // JSON null parses to void in JUCE, i.e. absent.
        let mut value = sample();
        value["tracks"][0]["mainGroup"]["notes"][0]["accent"] = json!(null);
        value["tracks"][0]["mainGroup"]["notes"][0]["instantMode"] = json!(null);
        value["tracks"][0]["mainGroup"]["notes"][0]["musicalType"] = json!(null);
        let project = parse_value(&value).unwrap();
        let note = &project.tracks[0].main_group.notes[0];
        assert_eq!(note.accent, "");
        assert!(note.instant_mode);
        assert_eq!(note.musical_type, "singing");
    }

    #[test]
    fn rejects_malformed_note_style_fields() {
        let cases = [
            ("accent", json!(3)),
            ("instantMode", json!("yes")),
            ("musicalType", json!(7)),
        ];
        for (key, replacement) in cases {
            let mut value = sample();
            value["tracks"][0]["mainGroup"]["notes"][0][key] = replacement;
            assert!(parse_value(&value).is_err(), "{key} should be rejected");
        }
    }

    fn round_trip_fixture() -> Value {
        json!({
            "version": 153,
            "vendorRoot": { "a": 1 },
            "time": {
                "tempo": [{ "position": 0, "bpm": 100.0, "vendorTempo": "t" }],
                "meter": [{ "index": 0, "numerator": 3, "denominator": 4, "vendorMeter": 2 }],
            },
            "library": [],
            "tracks": [{
                "name": "Lead",
                "vendorTrack": [1, 2],
                "mainGroup": {
                    "uuid": "main",
                    "name": "main",
                    "vendorGroup": true,
                    "notes": [{
                        "onset": 0,
                        "duration": Q,
                        "pitch": 60,
                        "lyrics": "do",
                        "accent": "strong",
                        "instantMode": false,
                        "musicalType": "rap",
                        "attributes": { "tF0Offset": 1.5, "vendorAttr": "a" },
                        "systemAttributes": { "vendorSys": 2 },
                        "vendorNote": "n",
                    }],
                    "parameters": {
                        "pitchDelta": { "mode": "linear", "points": [0, 0.0], "vendorCurve": "c" },
                        "vendorParam": 7,
                    },
                    "vocalModes": {
                        "Power": 0.5,
                        "Soft": { "enabled": false, "amount": 0.25, "vendorMode": "m" },
                    },
                },
                "mainRef": {
                    "groupID": "main",
                    "blickOffset": 0,
                    "pitchOffset": 0,
                    "vendorRef": "r",
                    "voice": { "vocalModePreset": "p", "vendorVoice": 1 },
                    "database": { "name": "db", "vendorDb": 1 },
                    "audio": { "filename": "takes/vocal.wav", "duration": 90.0 },
                },
                "groups": [],
                "mixer": { "gainDecibel": 0.0, "pan": 0.0 },
                "svVoice": { "databasePath": "", "language": "japanese", "dictionaryDirectory": "" },
            }],
        })
    }

    fn reserialize(project: &Project) -> String {
        serialize(project, Path::new("/proj")).unwrap()
    }

    fn reparse(text: &str) -> Project {
        parse(text.as_bytes(), "song", Path::new("/proj")).unwrap()
    }

    #[test]
    fn saving_a_clean_project_adds_exactly_the_cpp_defaults() {
        let project = parse_value(&sample()).unwrap();
        let saved = reserialize(&project);
        let root: Value = serde_json::from_str(&saved).unwrap();

        assert_eq!(root["version"], json!(153));
        let render = &root["renderConfig"];
        assert_eq!(
            render,
            &json!({
                "destination": "",
                "filename": "song",
                "numChannels": 1,
                "aspirationFormat": "noAspiration",
                "bitDepth": 16,
                "sampleRate": 44100,
                "exportMixDown": true,
                "exportPitch": false,
            })
        );

        let track = &root["tracks"][0];
        assert_eq!(track["dispColor"], json!("ff7db235"));
        assert_eq!(track["dispOrder"], json!(0));
        assert_eq!(track["renderEnabled"], json!(false));
        assert_eq!(track["mixer"]["display"], json!(true));
        assert_eq!(
            track["svVoice"],
            json!({
                "databasePath": "voices/a.nofs",
                "language": "english",
                "dictionaryDirectory": "",
            })
        );

        // Notes are sorted by onset: the "do" note (detune 12.5) comes first.
        let note = &track["mainGroup"]["notes"][0];
        assert_eq!(note["lyrics"], json!("do"));
        assert_eq!(note["musicalType"], json!("singing"));
        assert_eq!(note["accent"], json!(""));
        assert_eq!(note["instantMode"], json!(true));
        assert_eq!(note["attributes"]["evenSyllableDuration"], json!(true));
        assert_eq!(note["pitchTakes"]["takes"][0]["liked"], json!(false));
        assert_eq!(note["timbreTakes"]["activeTakeId"], json!(0));
        assert!(note.get("systemAttributes").is_none());

        let parameters = &track["mainGroup"]["parameters"];
        for name in [
            "loudness",
            "tension",
            "breathiness",
            "voicing",
            "gender",
            "toneShift",
        ] {
            assert_eq!(
                parameters[name],
                json!({ "mode": "cubic", "points": [] }),
                "{name} should gain the C++ default curve"
            );
        }

        let reference = &track["groups"][0];
        assert_eq!(reference["database"]["version"], json!("-2"));
        assert_eq!(reference["database"]["name"], json!(""));
        assert_eq!(reference["dictionary"], json!(""));
        assert_eq!(reference["voice"]["vocalModeInherited"], json!(true));
        assert_eq!(reference["voice"]["vocalModeParams"], json!({}));
        assert_eq!(
            reference["systemPitchDelta"],
            json!({ "mode": "cubic", "points": [] })
        );
        assert!(reference.get("audio").is_none());

        // Handled values survive the trip.
        let back = reparse(&saved);
        let back_note = &back.tracks[0].main_group.notes[0];
        assert_eq!(back_note.lyrics, "do");
        assert_eq!(back_note.detune, 12.5);
        assert_eq!(back.tempo_map.tempo_at(0), 100.0);
        assert_eq!(back.tracks[0].voice.language, Language::English);
        assert!((back.tracks[0].mixer.gain - 0.501_187).abs() < 1e-5);
    }

    #[test]
    fn save_is_a_fixed_point() {
        let first = parse_value(&round_trip_fixture()).unwrap();
        assert_eq!(first.tracks[0].main_group.notes[0].accent, "strong");
        assert!(!first.tracks[0].main_group.notes[0].instant_mode);
        assert_eq!(first.tracks[0].main_group.notes[0].musical_type, "rap");
        assert_eq!(
            first.tracks[0].main_ref.audio_file.as_deref(),
            Some(Path::new("/proj/takes/vocal.wav"))
        );

        let saved_once = reserialize(&first);
        let second = reparse(&saved_once);
        // Style fields, vendor blobs, and the backing-track path survive.
        assert_eq!(second.tracks[0].main_group.notes[0].accent, "strong");
        assert!(!second.tracks[0].main_group.notes[0].instant_mode);
        assert_eq!(second.tracks[0].main_group.notes[0].musical_type, "rap");
        assert_eq!(
            second.tracks[0].main_ref.audio_file.as_deref(),
            Some(Path::new("/proj/takes/vocal.wav"))
        );
        assert_eq!(
            second.tracks[0].main_ref.voice_pitch.extra.json(),
            r#"{"vendorVoice":1}"#
        );

        // One cycle is semantically stable; the second cycle is byte-stable
        // too. (Handled keys appended after retained vendor keys change key
        // order on the first save, exactly as the C++ `set` calls do, so
        // byte-identity only holds from the second save on.)
        let saved_twice = reserialize(&second);
        let as_value = |text: &str| serde_json::from_str::<Value>(text).unwrap();
        assert_eq!(as_value(&saved_once), as_value(&saved_twice));
        let third = reparse(&saved_twice);
        assert_eq!(second.tracks, third.tracks);
        assert_eq!(second.library, third.library);
        assert_eq!(second.tempo_map, third.tempo_map);
        assert_eq!(reserialize(&third), saved_twice);

        // Vendor blobs survive both trips verbatim.
        let root = as_value(&saved_twice);
        assert_eq!(root["vendorRoot"], json!({ "a": 1 }));
        assert_eq!(root["tracks"][0]["vendorTrack"], json!([1, 2]));
        let group = &root["tracks"][0]["mainGroup"];
        assert_eq!(group["vendorGroup"], json!(true));
        assert_eq!(group["parameters"]["pitchDelta"]["vendorCurve"], json!("c"));
        assert_eq!(group["notes"][0]["vendorNote"], json!("n"));
        assert_eq!(group["notes"][0]["attributes"]["vendorAttr"], json!("a"));
        assert_eq!(
            root["tracks"][0]["mainRef"]["database"]["vendorDb"],
            json!(1)
        );
        assert_eq!(
            root["tracks"][0]["mainRef"]["audio"],
            json!({ "filename": "takes/vocal.wav", "duration": 90.0 })
        );
    }

    #[test]
    fn save_writes_a_nul_terminated_file_atomically() {
        let project = parse_value(&sample()).unwrap();
        let dir = std::env::temp_dir().join(format!("opensvr-svp-save-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("song.svp");
        save(&path, &project).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.last(), Some(&0));
        let mut orphan = path.as_os_str().to_owned();
        orphan.push(".tmp");
        assert!(!Path::new(&orphan).exists());

        let back = load(&path).unwrap();
        assert_eq!(back.name, "song");
        assert_eq!(back.tracks[0].main_group.notes.len(), 2);
        fs::remove_dir_all(&dir).unwrap();
    }
}
