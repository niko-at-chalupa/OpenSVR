/// Carrier for unknown SVP JSON fields preserved across edits.
///
/// This is the Rust counterpart of OpenSV's `preservedFieldsJson`
/// (`std::string`) members in `src/core/Project.h`: when the SVP reader meets
/// vendor fields it does not understand, it keeps their serialized JSON here
/// so a load-save round trip does not lose external editors' data.
///
/// The payload stays an opaque serialized-JSON string (empty when there is
/// nothing to preserve), mirroring the C++ type exactly. This keeps
/// `opensvr-core` free of `serde_json`: map-level capture and
/// `preserve_order` serialization live in `opensvr-svp`, which already
/// depends on it. Nothing populates `Extra` yet; the SVP-wiring PR does that.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extra(String);

impl Extra {
    /// Wraps already-serialized JSON of the unrecognized vendor fields.
    pub fn new(json: impl Into<String>) -> Self {
        Self(json.into())
    }

    /// The serialized JSON payload; `""` when empty.
    pub fn json(&self) -> &str {
        &self.0
    }

    /// Whether there are no preserved fields.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_empty_and_round_trips() {
        assert_eq!(Extra::default(), Extra::new(""));
        assert!(Extra::default().is_empty());
        assert_eq!(Extra::default().json(), "");
        let kept = Extra::new(r#"{"vendor":1}"#);
        assert!(!kept.is_empty());
        assert_eq!(kept.json(), r#"{"vendor":1}"#);
        assert_eq!(kept, Extra::new(kept.json().to_owned()));
    }
}
