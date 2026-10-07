use std::f64::consts::FRAC_PI_4;

use opensvr_core::{Blick, Mixer, PlacedNote, Project, ProjectError};
use thiserror::Error;

use crate::{BackendError, CancelToken, TimedNote, VoiceBackend};

pub const MIN_SAMPLE_RATE: u32 = 8_000;
pub const MAX_SAMPLE_RATE: u32 = 384_000;
/// Silence kept after the last note so releases are not cut off.
pub const REST_CONTEXT_SECONDS: f64 = 0.1;

/// Upper bound on the stereo float buffer, to fail early rather than exhaust memory.
const MAX_AUDIO_BYTES: usize = 128 << 20;
/// How many frames are mixed between cancellation checks.
const MIX_CHUNK_FRAMES: usize = 4096;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("the output sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz")]
    InvalidSampleRate(u32),
    #[error("track {track:?}: {source}")]
    Track {
        track: String,
        #[source]
        source: ProjectError,
    },
    #[error("track {track:?}: overlapping singing notes require separate tracks")]
    Overlap { track: String },
    #[error("track {track:?}: {source}")]
    Backend {
        track: String,
        #[source]
        source: BackendError,
    },
    #[error("the project exceeds the 128 MiB stereo rendering buffer limit at this sample rate")]
    TooLong,
    #[error("rendering was cancelled")]
    Cancelled,
}

/// Planar stereo audio.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StereoBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl StereoBuffer {
    pub fn silent(frames: usize) -> Self {
        Self {
            left: vec![0.0; frames],
            right: vec![0.0; frames],
        }
    }

    pub fn frames(&self) -> usize {
        self.left.len()
    }
}

/// Mixes every audible track of a project into one stereo buffer.
#[derive(Debug)]
pub struct Renderer<B> {
    backend: B,
}

impl<B: VoiceBackend> Renderer<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Renders `project` at `sample_rate`.
    ///
    /// Each track is sung as a single phrase by the backend, then panned and summed.
    /// Muted and non-soloed tracks are skipped but still count towards the project length,
    /// so the mixer never changes the timeline.
    pub fn render(
        &mut self,
        project: &Project,
        sample_rate: u32,
        cancel: &CancelToken,
    ) -> Result<StereoBuffer, RenderError> {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
            return Err(RenderError::InvalidSampleRate(sample_rate));
        }

        let mut tracks = Vec::with_capacity(project.tracks.len());
        let mut score_end: Blick = 0;
        for track in &project.tracks {
            let placed = project
                .placed_notes(track)
                .map_err(|source| RenderError::Track {
                    track: track.name.clone(),
                    source,
                })?;
            if placed.windows(2).any(|pair| pair[1].start < pair[0].end) {
                return Err(RenderError::Overlap {
                    track: track.name.clone(),
                });
            }
            score_end = placed
                .iter()
                .map(|note| note.end)
                .fold(score_end, Blick::max);
            tracks.push((track, placed));
        }

        let seconds = if score_end > 0 {
            project.tempo_map.blick_to_seconds(score_end) + REST_CONTEXT_SECONDS
        } else {
            0.0
        };
        let frames = (seconds * f64::from(sample_rate)).ceil() as usize;
        if frames
            .checked_mul(2 * size_of::<f32>())
            .is_none_or(|bytes| bytes > MAX_AUDIO_BYTES)
        {
            return Err(RenderError::TooLong);
        }

        let has_solo = project.tracks.iter().any(|track| track.mixer.solo);
        let mut output = StereoBuffer::silent(frames);
        for (track, placed) in &tracks {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            if placed.is_empty() || !is_audible(&track.mixer, has_solo) {
                continue;
            }
            let notes = timed_notes(project, placed);
            let backend_error = |source| RenderError::Backend {
                track: track.name.clone(),
                source,
            };
            let samples = self
                .backend
                .render_phrase(&track.voice, &notes, sample_rate)
                .map_err(backend_error)?;
            if samples.iter().any(|sample| !sample.is_finite()) {
                let error = BackendError("the voice backend returned a non-finite sample".into());
                return Err(backend_error(error));
            }
            mix(
                &mut output,
                &samples,
                notes[0].start,
                sample_rate,
                &track.mixer,
                cancel,
            )?;
        }
        Ok(output)
    }
}

fn is_audible(mixer: &Mixer, has_solo: bool) -> bool {
    !mixer.mute && (!has_solo || mixer.solo) && mixer.gain != 0.0
}

fn timed_notes<'a>(project: &Project, placed: &[PlacedNote<'a>]) -> Vec<TimedNote<'a>> {
    let seconds = |position| project.tempo_map.blick_to_seconds(position);
    placed
        .iter()
        .map(|placed| TimedNote {
            start: seconds(placed.start),
            end: seconds(placed.end),
            pitch: placed.pitch,
            detune_cents: placed.note.detune,
            lyrics: &placed.note.lyrics,
            phonemes: &placed.note.phonemes,
        })
        .collect()
}

/// Adds mono `samples` to `output` starting at `start_seconds`, applying gain and a constant-power pan law.
fn mix(
    output: &mut StereoBuffer,
    samples: &[f32],
    start_seconds: f64,
    sample_rate: u32,
    mixer: &Mixer,
    cancel: &CancelToken,
) -> Result<(), RenderError> {
    let offset = (start_seconds * f64::from(sample_rate)).round() as usize;
    let count = samples.len().min(output.frames().saturating_sub(offset));
    if count == 0 {
        return Ok(());
    }

    let angle = (mixer.pan.clamp(-1.0, 1.0) + 1.0) * FRAC_PI_4;
    let left_gain = (mixer.gain * angle.cos()) as f32;
    let right_gain = (mixer.gain * angle.sin()) as f32;

    let span = offset..offset + count;
    let left = output.left[span.clone()].chunks_mut(MIX_CHUNK_FRAMES);
    let right = output.right[span].chunks_mut(MIX_CHUNK_FRAMES);
    let source = samples[..count].chunks(MIX_CHUNK_FRAMES);
    for (source, (left, right)) in source.zip(left.zip(right)) {
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        for ((&sample, left), right) in source.iter().zip(left).zip(right) {
            *left += sample * left_gain;
            *right += sample * right_gain;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use opensvr_core::{
        BLICKS_PER_QUARTER as Q, GroupReference, Note, NoteGroup, Track, VoiceSettings,
    };

    use super::*;

    /// Fills each phrase with 1.0 and records where phrases start.
    #[derive(Default)]
    struct Constant {
        starts: Vec<f64>,
    }

    impl VoiceBackend for Constant {
        fn render_phrase(
            &mut self,
            _: &VoiceSettings,
            notes: &[TimedNote<'_>],
            rate: u32,
        ) -> Result<Vec<f32>, BackendError> {
            self.starts.push(notes[0].start);
            let end = notes.last().unwrap().end;
            Ok(vec![
                1.0;
                ((end - notes[0].start) * f64::from(rate)).round()
                    as usize
            ])
        }
    }

    fn track(name: &str, onsets: &[Blick]) -> Track {
        let notes = onsets
            .iter()
            .map(|&onset| Note {
                onset,
                duration: Q,
                pitch: 60,
                lyrics: "la".into(),
                phonemes: String::new(),
                detune: 0.0,
            })
            .collect();
        Track::new(
            name,
            NoteGroup {
                id: name.into(),
                name: name.into(),
                notes,
            },
        )
    }

    fn render(
        tracks: Vec<Track>,
        backend: Constant,
    ) -> (Result<StereoBuffer, RenderError>, Constant) {
        let project = Project {
            tracks,
            ..Project::default()
        };
        let mut renderer = Renderer::new(backend);
        let result = renderer.render(&project, 8000, &CancelToken::new());
        (result, renderer.backend)
    }

    #[test]
    fn mixes_with_a_constant_power_pan_law() {
        let mut lead = track("lead", &[Q]);
        lead.mixer.pan = -1.0;
        lead.mixer.gain = 0.5;
        let (audio, backend) = render(vec![lead], Constant::default());
        let audio = audio.unwrap();

        // 120 BPM: the note spans 0.5 s..1.0 s, plus 0.1 s of rest context.
        assert_eq!(audio.frames(), 8800);
        assert!((backend.starts[0] - 0.5).abs() < 1e-9);
        assert!((audio.left[4000] - 0.5).abs() < 1e-6);
        assert!(audio.right[4000].abs() < 1e-6);
        assert_eq!(audio.left[3999], 0.0);
        assert_eq!(audio.left[8000], 0.0, "the note ends at 1.0 s");
    }

    #[test]
    fn centered_audio_keeps_equal_power_in_both_channels() {
        let (audio, _) = render(vec![track("lead", &[0])], Constant::default());
        let audio = audio.unwrap();
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((audio.left[10] - expected).abs() < 1e-6);
        assert!((audio.right[10] - expected).abs() < 1e-6);
    }

    #[test]
    fn solo_and_mute_choose_audible_tracks_but_not_the_length() {
        let mut short = track("short", &[0]);
        short.mixer.solo = true;
        let mut long = track("long", &[0, 2 * Q]);
        long.mixer.mute = true;
        let (audio, backend) = render(vec![short, long], Constant::default());
        assert_eq!(backend.starts.len(), 1);
        assert_eq!(
            audio.unwrap().frames(),
            12_800,
            "the muted track still sets the length"
        );
    }

    #[test]
    fn references_are_resolved_before_rendering() {
        let mut lead = track("lead", &[]);
        lead.groups.push(GroupReference {
            group_id: "lib".into(),
            time_offset: 2 * Q,
            ..GroupReference::default()
        });
        let library = vec![track("lib", &[0]).main_group];
        let project = Project {
            tracks: vec![lead],
            library,
            ..Project::default()
        };
        let mut renderer = Renderer::new(Constant::default());
        renderer
            .render(&project, 8000, &CancelToken::new())
            .unwrap();
        assert!((renderer.backend.starts[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_input() {
        let (result, _) = render(vec![track("lead", &[0, Q / 2])], Constant::default());
        assert!(matches!(result, Err(RenderError::Overlap { .. })));

        let project = Project::default();
        let mut renderer = Renderer::new(Constant::default());
        let error = renderer
            .render(&project, 100, &CancelToken::new())
            .unwrap_err();
        assert!(matches!(error, RenderError::InvalidSampleRate(100)));

        let token = CancelToken::new();
        token.cancel();
        let project = Project {
            tracks: vec![track("lead", &[0])],
            ..Project::default()
        };
        assert!(matches!(
            renderer.render(&project, 8000, &token),
            Err(RenderError::Cancelled)
        ));
    }

    #[test]
    fn an_empty_project_renders_no_audio() {
        let (audio, backend) = render(Vec::new(), Constant::default());
        assert_eq!(audio.unwrap().frames(), 0);
        assert!(backend.starts.is_empty());
    }
}
