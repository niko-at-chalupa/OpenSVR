//! Offline rendering for opensvr: voice backend trait, stereo mixer and WAV output.
//!
//! The neural voice engine of OpenSV is not ported yet. Rendering is therefore
//! split in two: a [`Renderer`] that does everything around the voice (timeline,
//! mixer, pan law, solo/mute, cancellation) and a [`VoiceBackend`] that turns a
//! phrase of notes into mono audio. [`ToneBackend`] is a placeholder that sings
//! sine tones, enough to exercise the whole pipeline end to end.

mod backend;
mod render;
mod wav;

pub use backend::{BackendError, TimedNote, ToneBackend, VoiceBackend};
pub use opensvr_core::CancelToken;
pub use render::{
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, REST_CONTEXT_SECONDS, RenderError, Renderer, StereoBuffer,
};
pub use wav::{WavError, write_wav};
