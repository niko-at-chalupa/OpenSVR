//! Offline rendering for opensvr: voice backend trait, stereo mixer and WAV output.
//!
//! A [`Renderer`] does everything around the voice (timeline, phrase splitting,
//! mixer, pan law, solo/mute, cancellation) and a [`VoiceBackend`] turns one
//! [`Phrase`] of notes into mono audio. [`ToneBackend`] is a placeholder that
//! sings sine tones; [`NeuralBackend`] sings whole phrases through the neural
//! voice when a voice database and dictionaries are configured.

mod backend;
mod cache;
mod neural;
mod phrase;
mod pitch;
mod render;
mod resample;
mod wav;

pub use backend::{BackendError, PhraseAudio, TimedNote, ToneBackend, VoiceBackend};
pub use neural::NeuralBackend;
pub use opensvr_core::CancelToken;
pub use phrase::{PHRASE_BREAK_SECONDS, Phrase, split_phrases};
pub use pitch::{
    build_log_f0, build_vibrato_envelope, build_vocal_mode_weights, local_curve_position,
    merge_continuation_phones, phrase_has_vocal_modes,
};
pub use render::{
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, NoProgress, ProgressSink, REST_CONTEXT_SECONDS, RenderError,
    Renderer, StereoBuffer,
};
pub use resample::resample_mono;
pub use wav::{WavError, write_wav};
