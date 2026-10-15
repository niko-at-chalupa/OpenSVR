use std::f64::consts::FRAC_PI_4;

use opensvr_core::{Blick, Mixer, PlacedNote, Project, ProjectError, Track};
use thiserror::Error;

use crate::{BackendError, CancelToken, PhraseAudio, TimedNote, VoiceBackend, split_phrases};

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

/// Reports phrase completions during a render.
///
/// There is no C++ counterpart to port: OpenSV's `render` takes only a cancel
/// callback plus an optional editor-only `RenderVisualization` (out of scope
/// per the roadmap), and OpenSV+'s CLI prints JSON with no progress at all.
/// Workers report each completed phrase against a total known up front; they
/// never report audio, and a cancelled phrase reports nothing.
pub trait ProgressSink: Sync {
    /// Records that `done` of `total` phrases have completed.
    fn report(&self, done: usize, total: usize);
}

/// A [`ProgressSink`] that discards every report.
///
/// [`Renderer::render`] uses this, so library callers that pass nothing keep
/// today's behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoProgress;

impl ProgressSink for NoProgress {
    fn report(&self, _done: usize, _total: usize) {}
}

/// Shared phrase-completion reporting for the render workers.
///
/// Reports are serialized so sinks observe monotone `done` counts even though
/// worker threads complete phrases in nondeterministic order.
struct Reporter<'a> {
    sink: &'a dyn ProgressSink,
    completed: &'a std::sync::atomic::AtomicUsize,
    order: &'a std::sync::Mutex<()>,
    total: usize,
}

impl Reporter<'_> {
    /// Records one completed phrase. Callers return before reaching here for
    /// cancelled phrases, so cancellation never reports a completion.
    fn report_one(&self) {
        let _guard = self
            .order
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let done = self
            .completed
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        self.sink.report(done, self.total);
    }
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
///
/// Tracks sing concurrently on scoped threads through one shared backend and
/// are summed in track order, so multi-track renders stay bitwise identical
/// no matter how the workers interleave.
#[derive(Debug)]
pub struct Renderer<B> {
    backend: B,
}

impl<B: VoiceBackend + Sync> Renderer<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// The singing backend (which keeps its caches across renders).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Mutable access to the singing backend.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Renders `project` at `sample_rate`, discarding phrase-progress reports.
    ///
    /// This is [`render_with_progress`][Self::render_with_progress] with a
    /// [`NoProgress`] sink, so the samples are bitwise identical either way.
    pub fn render(
        &self,
        project: &Project,
        sample_rate: u32,
        cancel: &CancelToken,
    ) -> Result<StereoBuffer, RenderError> {
        self.render_with_progress(project, sample_rate, cancel, &NoProgress)
    }

    /// Renders `project` at `sample_rate`, reporting each completed phrase.
    ///
    /// Each audible track is sung on its own scoped thread as one phrase per
    /// rest-bounded phrase by the shared backend, then panned and summed in
    /// track order. Muted and non-soloed tracks are skipped but still count
    /// towards the project length, so the mixer never changes the timeline.
    /// A cancelled render publishes no audio, exactly as before: workers only
    /// return phrase audio, and mixing happens after they all join.
    /// Progress reporting never moves, skips or reorders any synthesis or
    /// summation: workers only report completions, so the samples are
    /// bitwise identical with any sink.
    pub fn render_with_progress(
        &self,
        project: &Project,
        sample_rate: u32,
        cancel: &CancelToken,
        progress: &dyn ProgressSink,
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
        // Audible work in track order. The cancellation check stays ahead of
        // the skips, exactly as in the sequential loop, so a cancelled token
        // fails even when every track is silent.
        let mut work = Vec::new();
        for (track, placed) in &tracks {
            if cancel.is_cancelled() {
                return Err(RenderError::Cancelled);
            }
            if placed.is_empty() || !is_audible(&track.mixer, has_solo) {
                continue;
            }
            work.push((*track, placed.as_slice()));
        }

        // The audible phrase total is known before spawning, so every report
        // carries a fixed total. Splitting is pure timeline geometry, and the
        // workers re-split identically when they sing.
        let total = work
            .iter()
            .map(|(track, placed)| split_phrases(&timed_notes(project, track, placed)).len())
            .sum();
        let completed = std::sync::atomic::AtomicUsize::new(0);
        let order = std::sync::Mutex::new(());
        let reporter = Reporter {
            sink: progress,
            completed: &completed,
            order: &order,
            total,
        };

        // Sing every audible track on its own scoped thread; workers only
        // return phrase audio, never touching the shared output buffer.
        let sung = sing_all(
            &self.backend,
            project,
            &work,
            sample_rate,
            cancel,
            &reporter,
        )?;

        // Sum in track order through the single mixing path, so scheduling
        // never changes a sample.
        let mut output = StereoBuffer::silent(frames);
        for ((track, _), phrases) in work.iter().zip(&sung) {
            for phrase in phrases {
                mix(
                    &mut output,
                    &phrase.audio.samples,
                    phrase.audio.start,
                    phrase.crop_start,
                    phrase.stop,
                    sample_rate,
                    &track.mixer,
                    cancel,
                )?;
            }
        }
        // The engine clamps the mixed output, exactly as in C++.
        for channel in [&mut output.left, &mut output.right] {
            for sample in channel.iter_mut() {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        Ok(output)
    }
}

/// Sings every track in `work` on its own scoped thread.
///
/// Workers only return phrase audio, never touching the shared output buffer.
/// Joins in track order so the first error in track order wins, exactly as
/// the sequential loop reported it.
fn sing_all<B: VoiceBackend + Sync>(
    backend: &B,
    project: &Project,
    work: &[(&Track, &[PlacedNote<'_>])],
    sample_rate: u32,
    cancel: &CancelToken,
    reporter: &Reporter<'_>,
) -> Result<Vec<Vec<SungPhrase>>, RenderError> {
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(work.len());
        for (track, placed) in work {
            handles.push(scope.spawn(|| {
                sing_track(
                    backend,
                    project,
                    track,
                    placed,
                    sample_rate,
                    cancel,
                    reporter,
                )
            }));
        }
        let mut sung = Vec::with_capacity(work.len());
        for (handle, (track, _)) in handles.into_iter().zip(work) {
            sung.push(handle.join().unwrap_or_else(|_| {
                Err(RenderError::Backend {
                    track: track.name.clone(),
                    source: BackendError("a render worker panicked".into()),
                })
            })?);
        }
        Ok(sung)
    })
}

/// One track's sung phrases: the audio plus its group-window crop bounds.
///
/// Per-phrase group-window crop, exactly as in C++: the front note's
/// `absoluteBegin` bounds the mixed output on the left (model-input context
/// is kept), the back note's `absoluteEnd` clamps the release tail on the
/// right.
struct SungPhrase {
    audio: PhraseAudio,
    crop_start: f64,
    stop: f64,
}

/// Sings one track's phrases on a worker thread: timeline geometry plus the
/// shared backend, without touching the mixed output.
fn sing_track<B: VoiceBackend>(
    backend: &B,
    project: &Project,
    track: &Track,
    placed: &[PlacedNote<'_>],
    sample_rate: u32,
    cancel: &CancelToken,
    reporter: &Reporter<'_>,
) -> Result<Vec<SungPhrase>, RenderError> {
    let notes = timed_notes(project, track, placed);
    let backend_error = |source| RenderError::Backend {
        track: track.name.clone(),
        source,
    };
    let mut sung = Vec::new();
    for phrase in split_phrases(&notes) {
        if cancel.is_cancelled() {
            return Err(RenderError::Cancelled);
        }
        let audio = backend
            .render_phrase(&track.voice, &phrase, sample_rate, cancel)
            .map_err(backend_error)?;
        if audio.samples.iter().any(|sample| !sample.is_finite()) {
            let error = BackendError("the voice backend returned a non-finite sample".into());
            return Err(backend_error(error));
        }
        let (Some(first), Some(last)) = (phrase.notes.first(), phrase.notes.last()) else {
            continue;
        };
        let release_end = last.end + REST_CONTEXT_SECONDS;
        let stop = last
            .window_end
            .map_or(release_end, |end| release_end.min(end));
        sung.push(SungPhrase {
            audio,
            crop_start: first.window_start,
            stop,
        });
        reporter.report_one();
    }
    Ok(sung)
}

fn is_audible(mixer: &Mixer, has_solo: bool) -> bool {
    !mixer.mute && (!has_solo || mixer.solo) && mixer.gain != 0.0
}

fn timed_notes<'a>(
    project: &'a Project,
    track: &'a Track,
    placed: &[PlacedNote<'a>],
) -> Vec<TimedNote<'a>> {
    let seconds = |position| project.tempo_map.blick_to_seconds(position);
    placed
        .iter()
        .map(|placed| TimedNote {
            start: seconds(placed.start),
            end: seconds(placed.end),
            pitch: placed.pitch,
            detune_cents: placed.note.detune,
            seconds_per_quarter: 60.0 / project.tempo_map.tempo_at(placed.start),
            lyrics: &placed.note.lyrics,
            phonemes: &placed.note.phonemes,
            note: placed.note,
            group: placed.group,
            reference: placed.reference,
            main_reference: &track.main_ref,
            tempo_map: &project.tempo_map,
            window_start: seconds(placed.window_start),
            window_end: placed.window_end.map(seconds),
        })
        .collect()
}

/// Adds mono `samples` to `output` starting at `start_seconds`, applying gain and a constant-power pan law.
///
/// Neural phrases start before zero when the leading context precedes the
/// project start; like C++ `mixPhrase`, whatever falls before zero is dropped
/// rather than shifting the timeline. The mixed output is additionally cropped
/// to the phrase's group window (`crop_start_seconds`/`stop_seconds`, from the
/// front/back notes' references): model-input context is kept, only the
/// placed audio is cut, after resampling into project sample coordinates.
#[allow(clippy::too_many_arguments)]
fn mix(
    output: &mut StereoBuffer,
    samples: &[f32],
    start_seconds: f64,
    crop_start_seconds: f64,
    stop_seconds: f64,
    sample_rate: u32,
    mixer: &Mixer,
    cancel: &CancelToken,
) -> Result<(), RenderError> {
    let rate = f64::from(sample_rate);
    let phrase_offset = (start_seconds * rate).round() as i64;
    let crop_offset = (crop_start_seconds * rate).ceil() as i64;
    let offset = phrase_offset.max(0).max(crop_offset) as usize;
    let stop = ((stop_seconds * rate).ceil().max(0.0) as usize).min(output.frames());
    let skipped = if phrase_offset < 0 {
        offset + phrase_offset.unsigned_abs() as usize
    } else {
        offset - phrase_offset as usize
    };
    let Some(source) = samples.get(skipped..) else {
        return Ok(());
    };
    let count = source.len().min(stop.saturating_sub(offset));
    if count == 0 {
        return Ok(());
    }

    let angle = (mixer.pan.clamp(-1.0, 1.0) + 1.0) * FRAC_PI_4;
    let left_gain = (mixer.gain * angle.cos()) as f32;
    let right_gain = (mixer.gain * angle.sin()) as f32;

    let span = offset..offset + count;
    let left = output.left[span.clone()].chunks_mut(MIX_CHUNK_FRAMES);
    let right = output.right[span].chunks_mut(MIX_CHUNK_FRAMES);
    let source = source[..count].chunks(MIX_CHUNK_FRAMES);
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
        BLICKS_PER_QUARTER as Q, Extra, GroupReference, Note, NoteGroup, PitchAttributes, Track,
        VoiceSettings,
    };

    use super::*;
    use crate::{Phrase, PhraseAudio};

    /// Emits 1.0 across the whole model-input span (leading context through
    /// the release tail), like `NeuralBackend`, so group-window cropping is
    /// observable: a backend that only sings note bodies would leave digital
    /// silence in exactly the cropped regions.
    #[derive(Default)]
    struct FullSpan;

    impl VoiceBackend for FullSpan {
        fn render_phrase(
            &self,
            _: &VoiceSettings,
            phrase: &Phrase<'_>,
            rate: u32,
            _: &CancelToken,
        ) -> Result<PhraseAudio, BackendError> {
            let first = phrase.notes.first().unwrap();
            let last = phrase.notes.last().unwrap();
            let start = first.start - REST_CONTEXT_SECONDS;
            let frames =
                ((last.end + REST_CONTEXT_SECONDS - start) * f64::from(rate)).round() as usize;
            Ok(PhraseAudio {
                start,
                samples: vec![1.0; frames],
                sample_rate: rate,
            })
        }
    }

    /// A track whose only singing note comes from a windowed library reference.
    fn windowed_track(
        name: &str,
        onset: Blick,
        time_offset: Blick,
        begin: Blick,
        end: Option<Blick>,
    ) -> (Track, NoteGroup) {
        let library = NoteGroup {
            id: format!("{name}-lib"),
            name: name.into(),
            notes: vec![Note {
                onset,
                duration: Q,
                pitch: 60,
                lyrics: "la".into(),
                phonemes: String::new(),
                detune: 0.0,
                instant_mode: true,
                musical_type: "singing".into(),
                accent: String::new(),
                attributes: PitchAttributes::default(),
                system_attributes: PitchAttributes::default(),
                extra: Extra::default(),
            }],
            ..NoteGroup::default()
        };
        let mut track = Track::new(
            name,
            NoteGroup {
                id: format!("{name}-main"),
                name: name.into(),
                notes: Vec::new(),
                ..NoteGroup::default()
            },
        );
        track.groups.push(GroupReference {
            group_id: library.id.clone(),
            time_offset,
            pitch_offset: 0,
            begin,
            end,
            instrumental: false,
            ..GroupReference::default()
        });
        (track, library)
    }

    #[test]
    fn group_windows_crop_the_release_tail_and_leading_context() {
        // At 120 BPM one quarter is 0.5 s. The lead note (0.0..0.5 s) sits in
        // a window ending mid-project at 0.5 s; the tail note (1.0..1.5 s)
        // sits in a window beginning at 1.0 s and sets the project end.
        let (lead, lead_lib) = windowed_track("lead", 0, 0, 0, Some(Q));
        let (tail, tail_lib) = windowed_track("tail", 0, 2 * Q, 2 * Q, None);
        let project = Project {
            tracks: vec![lead, tail],
            library: vec![lead_lib, tail_lib],
            ..Project::default()
        };
        let renderer = Renderer::new(FullSpan);
        let audio = renderer
            .render(&project, 8000, &CancelToken::new())
            .unwrap();
        assert_eq!(audio.frames(), 12_800);

        // The lead release tail (0.5..0.6 s) stops at the window edge.
        assert!(audio.left[3999] > 0.0);
        assert_eq!(audio.left[4000], 0.0);
        assert_eq!(audio.left[4799], 0.0);
        // The tail leading context (0.9..1.0 s) stops at the window begin.
        assert_eq!(audio.left[7200], 0.0);
        assert_eq!(audio.left[7999], 0.0);
        assert!(audio.left[8000] > 0.0);
        // The unbounded tail still rings to the project end.
        assert!(audio.left[11_999] > 0.0);
        assert!(audio.left[12_000] > 0.0);
        assert!(audio.left[12_799] > 0.0);
    }

    /// Fills each phrase with 1.0 and records where phrases start.
    #[derive(Default)]
    struct Constant {
        starts: std::sync::Mutex<Vec<f64>>,
    }

    impl VoiceBackend for Constant {
        fn render_phrase(
            &self,
            _: &VoiceSettings,
            phrase: &Phrase<'_>,
            rate: u32,
            _: &CancelToken,
        ) -> Result<PhraseAudio, BackendError> {
            self.starts
                .lock()
                .expect("test backend is never poisoned")
                .push(phrase.notes[0].start);
            let end = phrase.notes.last().unwrap().end;
            Ok(PhraseAudio {
                start: phrase.notes[0].start,
                samples: vec![
                    1.0;
                    ((end - phrase.notes[0].start) * f64::from(rate)).round() as usize
                ],
                sample_rate: rate,
            })
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
                instant_mode: true,
                musical_type: "singing".into(),
                accent: String::new(),
                attributes: PitchAttributes::default(),
                system_attributes: PitchAttributes::default(),
                extra: Extra::default(),
            })
            .collect();
        Track::new(
            name,
            NoteGroup {
                id: name.into(),
                name: name.into(),
                notes,
                ..NoteGroup::default()
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
        let renderer = Renderer::new(backend);
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
        assert!((backend.starts.lock().unwrap()[0] - 0.5).abs() < 1e-9);
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
        assert_eq!(backend.starts.lock().unwrap().len(), 1);
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
        let renderer = Renderer::new(Constant::default());
        renderer
            .render(&project, 8000, &CancelToken::new())
            .unwrap();
        assert!((renderer.backend.starts.lock().unwrap()[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_bad_input() {
        let (result, _) = render(vec![track("lead", &[0, Q / 2])], Constant::default());
        assert!(matches!(result, Err(RenderError::Overlap { .. })));

        let project = Project::default();
        let renderer = Renderer::new(Constant::default());
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
        assert!(backend.starts.lock().unwrap().is_empty());
    }

    #[test]
    fn parallel_tracks_sum_in_track_order_like_isolated_sums() {
        // Two identical centered tracks at half gain: the clamp stays out of
        // the way, so the joint render must equal the sample-wise sum of the
        // isolated renders no matter how the worker threads interleave.
        let mut lead = track("lead", &[0]);
        lead.mixer.gain = 0.5;
        let mut harmony = track("harmony", &[0]);
        harmony.mixer.gain = 0.5;

        let muted = |audible: &Track, silent: &Track| {
            let mut silent = silent.clone();
            silent.mixer.mute = true;
            Project {
                tracks: vec![audible.clone(), silent],
                ..Project::default()
            }
        };
        let cancel = CancelToken::new();
        let single = |project: &Project| {
            Renderer::new(Constant::default())
                .render(project, 8000, &cancel)
                .unwrap()
        };
        let first = single(&muted(&lead, &harmony));
        let second = single(&muted(&harmony, &lead));
        assert_eq!(first.frames(), second.frames());

        let joint = Project {
            tracks: vec![lead, harmony],
            ..Project::default()
        };
        // Repeated joint renders pin determinism: same bits every time.
        let mut previous = None;
        for _ in 0..3 {
            let audio = single(&joint);
            assert_eq!(audio.frames(), first.frames());
            for ((got, a), b) in audio
                .left
                .iter()
                .zip(&first.left)
                .zip(&second.left)
                .chain(audio.right.iter().zip(&first.right).zip(&second.right))
            {
                assert_eq!(*got, *a + *b, "joint tracks must sum in track order");
            }
            if let Some(previous) = previous {
                assert_eq!(audio, previous, "re-renders must be bitwise identical");
            }
            previous = Some(audio);
        }
    }

    #[test]
    fn a_cancelled_multi_track_render_publishes_nothing() {
        let token = CancelToken::new();
        token.cancel();
        let project = Project {
            tracks: vec![track("lead", &[0]), track("harmony", &[Q])],
            ..Project::default()
        };
        assert!(matches!(
            Renderer::new(Constant::default()).render(&project, 8000, &token),
            Err(RenderError::Cancelled)
        ));
    }

    /// A sink that records every report behind a mutex.
    #[derive(Debug, Default)]
    struct Recording {
        reports: std::sync::Mutex<Vec<(usize, usize)>>,
    }

    impl ProgressSink for Recording {
        fn report(&self, done: usize, total: usize) {
            self.reports
                .lock()
                .expect("test sink is never poisoned")
                .push((done, total));
        }
    }

    #[test]
    fn progress_reports_every_phrase_against_a_fixed_total() {
        // Lead sings one phrase; harmony's two notes sit a rest apart, so two
        // more: three phrases in total across two worker threads.
        let project = Project {
            tracks: vec![track("lead", &[0]), track("harmony", &[0, 2 * Q])],
            ..Project::default()
        };
        let sink = Recording::default();
        let audio = Renderer::new(Constant::default())
            .render_with_progress(&project, 8000, &CancelToken::new(), &sink)
            .unwrap();
        // Reports are serialized, so the observed order stays monotone even
        // though the workers race.
        assert_eq!(*sink.reports.lock().unwrap(), vec![(1, 3), (2, 3), (3, 3)]);
        // Reporting must not move a sample.
        let plain = Renderer::new(Constant::default())
            .render(&project, 8000, &CancelToken::new())
            .unwrap();
        assert_eq!(audio, plain);
    }

    #[test]
    fn a_cancelled_render_reports_no_completion() {
        let token = CancelToken::new();
        token.cancel();
        let project = Project {
            tracks: vec![track("lead", &[0]), track("harmony", &[Q])],
            ..Project::default()
        };
        let sink = Recording::default();
        assert!(matches!(
            Renderer::new(Constant::default()).render_with_progress(&project, 8000, &token, &sink),
            Err(RenderError::Cancelled)
        ));
        assert!(sink.reports.lock().unwrap().is_empty());
    }
}
