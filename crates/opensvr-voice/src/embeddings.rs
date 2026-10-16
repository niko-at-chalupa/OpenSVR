//! Flat voice speaker/style embeddings from the voice document.
//!
//! This is the Rust counterpart of OpenSV+'s `src/synthesis/VoiceEmbeddings.*`
//! (the canonical fork has no Flat support, so document mode is ported from
//! the read-only snapshot with permission; behavior is reimplemented, never
//! copied). Split model files are shared across singers: the JSON document
//! carries the per-singer vectors as hex-encoded little-endian float32, and
//! these values override the serialized model defaults at load. Absent fields
//! retain the embedded model behavior (empty overrides); present-but-malformed
//! fields are errors.
//!
//! Error strings reproduce the reference failure text verbatim.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{Result, invalid};

/// Channel counts of the document embedding vectors.
pub const ACOUSTIC_CHANNELS: usize = 32;
pub const DURATION_CHANNELS: usize = 128;

/// Decoded Flat voice embeddings: speaker/style overrides plus named vocal
/// modes. Empty vectors mean "absent": the loader keeps the model-embedded
/// defaults.
#[derive(Debug, Clone, Default)]
pub struct VoiceEmbeddings {
    /// `(base)` style: the acoustic speaker override (32 channels).
    pub acoustic_speaker: Vec<f32>,
    /// `(default)` style: the acoustic default-style override (32 channels).
    pub acoustic_style: Vec<f32>,
    /// `f0_style`: the pitch speaker override (32 channels).
    pub pitch_speaker: Vec<f32>,
    /// `duration_style`: the duration voice override (128 channels).
    pub duration_voice: Vec<f32>,
    /// Named styles (other than `(base)`/`(default)`) as vocal modes.
    pub vocal_modes: BTreeMap<String, Vec<f32>>,
}

impl VoiceEmbeddings {
    /// Decodes the embedding fields of a Flat voice document.
    ///
    /// `styles[]` maps `(base)` to the acoustic speaker and `(default)` to
    /// the acoustic default style (they must pair: one without the other is
    /// an error); other named entries become vocal modes, skipping empty
    /// names and `(...)` names and rejecting duplicates. `extra` per-style
    /// fields are ignored. `f0_style` (32 channels) and `duration_style`
    /// (128 channels) override the pitch speaker and duration voice vectors.
    pub fn load(document: &Value) -> Result<Self> {
        let Value::Object(object) = document else {
            return Err(invalid("Voice embeddings require a NOFS JSON object"));
        };
        let mut candidate = Self::default();
        if let Some(styles) = object.get("styles") {
            let Value::Array(styles) = styles else {
                return Err(invalid("NOFS styles must be an array"));
            };
            for style in styles {
                let name = style
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let target = if name == "(base)" {
                    Some(&mut candidate.acoustic_speaker)
                } else if name == "(default)" {
                    Some(&mut candidate.acoustic_style)
                } else {
                    None
                };
                if let Some(target) = target {
                    if !target.is_empty() {
                        return Err(invalid("Duplicate NOFS speaker/default embedding"));
                    }
                    decode(style.get("data"), ACOUSTIC_CHANNELS, name, target)?;
                    continue;
                }
                if name.is_empty() || name.starts_with('(') {
                    continue;
                }
                if candidate.vocal_modes.contains_key(name) {
                    return Err(invalid("Duplicate NOFS vocal mode"));
                }
                let mut vector = Vec::new();
                decode(style.get("data"), ACOUSTIC_CHANNELS, name, &mut vector)?;
                candidate.vocal_modes.insert(name.to_owned(), vector);
            }
            if candidate.acoustic_speaker.is_empty() != candidate.acoustic_style.is_empty() {
                return Err(invalid(
                    "NOFS must provide both (base) and (default) embeddings",
                ));
            }
        }
        if let Some(value) = object.get("f0_style") {
            decode(
                Some(value),
                ACOUSTIC_CHANNELS,
                "f0_style",
                &mut candidate.pitch_speaker,
            )?;
        }
        if let Some(value) = object.get("duration_style") {
            decode(
                Some(value),
                DURATION_CHANNELS,
                "duration_style",
                &mut candidate.duration_voice,
            )?;
        }
        Ok(candidate)
    }
}

/// Decodes one hex vector: `channels` little-endian float32 values with a
/// finite check per value. Wrong length or non-hex digits are errors.
fn decode(
    value: Option<&Value>,
    channels: usize,
    field: &str,
    output: &mut Vec<f32>,
) -> Result<()> {
    let fail = || {
        invalid(format!(
            "Voice embedding '{field}' must be a finite little-endian float32 hex vector of the expected length"
        ))
    };
    let Some(Value::String(text)) = value else {
        return Err(fail());
    };
    if text.len() != channels * 8 {
        return Err(fail());
    }
    let nibble = |byte: u8| -> Option<u32> {
        match byte {
            b'0'..=b'9' => Some(u32::from(byte - b'0')),
            b'a'..=b'f' => Some(u32::from(byte - b'a') + 10),
            b'A'..=b'F' => Some(u32::from(byte - b'A') + 10),
            _ => None,
        }
    };
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(channels);
    for channel in 0..channels {
        let mut word = 0u32;
        for byte in 0..4 {
            let high = nibble(bytes[channel * 8 + byte * 2]).ok_or_else(&fail)?;
            let low = nibble(bytes[channel * 8 + byte * 2 + 1]).ok_or_else(&fail)?;
            word |= (high << 4 | low) << (byte * 8);
        }
        let value = f32::from_bits(word);
        if !value.is_finite() {
            return Err(fail());
        }
        result.push(value);
    }
    *output = result;
    Ok(())
}

/// Encodes floats as a little-endian hex vector (test helper parity).
#[cfg(test)]
pub(crate) fn encode(values: &[f32]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(values.len() * 8);
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            write!(text, "{byte:02x}").expect("writing to a String never fails");
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decodes_known_hex_with_endianness() {
        // 1.0f = 0x3F800000, little-endian bytes 00 00 80 3f.
        let mut output = Vec::new();
        decode(
            Some(&Value::String("0000803f".to_owned() + &"00".repeat(31 * 4))),
            32,
            "probe",
            &mut output,
        )
        .unwrap();
        assert_eq!(output.len(), 32);
        assert_eq!(output[0], 1.0);
        assert!(output[1..].iter().all(|&value| value == 0.0));

        // Uppercase hex decodes identically; byte order is little-endian:
        // "3f800000" big-endian spelling would read back as 2.2779...e-41, not 1.0.
        let mut upper = Vec::new();
        decode(
            Some(&Value::String("0000803F".to_owned() + &"00".repeat(31 * 4))),
            32,
            "probe",
            &mut upper,
        )
        .unwrap();
        assert_eq!(upper, output);
    }

    #[test]
    fn rejects_bad_vectors() {
        let good = encode(&[1.0; 32]);
        // Wrong length.
        assert!(decode_string(&good[2..], 32).is_err());
        // Non-hex digits.
        let mut bad = good.clone();
        bad.replace_range(0..1, "z");
        assert!(decode_string(&bad, 32).is_err());
        // Non-finite (+inf = 0x7F800000 LE "0000807f").
        let infinite = "0000807f".to_owned() + &"00".repeat(31 * 4);
        let error = decode_string(&infinite, 32).unwrap_err().to_string();
        assert_eq!(
            error,
            "Voice embedding 'probe' must be a finite little-endian float32 hex vector of the expected length"
        );
        // Non-string JSON.
        let mut output = Vec::new();
        assert!(decode(Some(&json!(1.0)), 32, "probe", &mut output).is_err());
        assert!(decode(None, 32, "probe", &mut output).is_err());
    }

    fn decode_string(text: &str, channels: usize) -> crate::Result<()> {
        decode(
            Some(&Value::String(text.to_owned())),
            channels,
            "probe",
            &mut Vec::new(),
        )
    }

    fn style(name: &str, channels: usize) -> Value {
        json!({"name": name, "data": encode(&vec![1.0; channels])})
    }

    #[test]
    fn maps_base_default_and_vocal_modes() {
        let document = json!({
            "styles": [style("(base)", 32), style("(default)", 32), style("Joyful", 32)],
        });
        let embeddings = VoiceEmbeddings::load(&document).unwrap();
        assert_eq!(embeddings.acoustic_speaker, vec![1.0; 32]);
        assert_eq!(embeddings.acoustic_style, vec![1.0; 32]);
        assert_eq!(embeddings.vocal_modes.len(), 1);
        assert_eq!(embeddings.vocal_modes["Joyful"], vec![1.0; 32]);
        assert!(embeddings.pitch_speaker.is_empty());
        assert!(embeddings.duration_voice.is_empty());

        // Empty names and `(...)` names are skipped; `extra` is ignored.
        let document = json!({
            "styles": [
                style("(base)", 32),
                style("(default)", 32),
                {"name": "", "data": encode(&[1.0; 32])},
                {"name": "(whisper)", "data": encode(&[1.0; 32])},
                {"name": "Joyful", "data": encode(&[1.0; 32]), "extra": {"gain": 0.5}},
            ],
        });
        let embeddings = VoiceEmbeddings::load(&document).unwrap();
        assert_eq!(embeddings.vocal_modes.len(), 1);
    }

    #[test]
    fn rejects_unpaired_and_duplicate_styles() {
        let error = VoiceEmbeddings::load(&json!({"styles": [style("(base)", 32)]}))
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "NOFS must provide both (base) and (default) embeddings"
        );
        let error = VoiceEmbeddings::load(&json!({
            "styles": [style("(base)", 32), style("(default)", 32), style("(base)", 32)],
        }))
        .unwrap_err()
        .to_string();
        assert_eq!(error, "Duplicate NOFS speaker/default embedding");
        let error = VoiceEmbeddings::load(&json!({
            "styles": [style("(base)", 32), style("(default)", 32), style("Joyful", 32), style("Joyful", 32)],
        }))
        .unwrap_err()
        .to_string();
        assert_eq!(error, "Duplicate NOFS vocal mode");
        let error = VoiceEmbeddings::load(&json!({"styles": {}}))
            .unwrap_err()
            .to_string();
        assert_eq!(error, "NOFS styles must be an array");
        let error = VoiceEmbeddings::load(&json!([1, 2]))
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Voice embeddings require a NOFS JSON object");
    }

    #[test]
    fn decodes_style_overrides_or_leaves_them_empty() {
        let document = json!({
            "f0_style": encode(&[2.0; 32]),
            "duration_style": encode(&[3.0; 128]),
        });
        let embeddings = VoiceEmbeddings::load(&document).unwrap();
        assert_eq!(embeddings.pitch_speaker, vec![2.0; 32]);
        assert_eq!(embeddings.duration_voice, vec![3.0; 128]);

        // Present-but-malformed is an error, with the field named.
        let error = VoiceEmbeddings::load(&json!({"f0_style": encode(&[1.0; 31])}))
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("Voice embedding 'f0_style'"), "{error}");
        let error = VoiceEmbeddings::load(&json!({"duration_style": Value::Null}))
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("Voice embedding 'duration_style'"),
            "{error}"
        );
    }
}
