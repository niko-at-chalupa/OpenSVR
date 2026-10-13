//! Gated end-to-end test for [`NeuralBackend`](opensvr_audio::NeuralBackend):
//! first sung output through the real phrase path on a real voice.
//!
//! Set `OPENSVR_TEST_VOICE` to a voice archive (e.g. the GUMI AI voice) to
//! run; the test returns early when unset so CI stays green without
//! proprietary data. The SVP input and the C++ reference excerpts are
//! committed under `tests/golden/audio/`; the voice blob itself never is.
//!
//! Reference provenance: the excerpts are the first 3000 left-channel
//! samples of `opensv-cli` renders of `neural-basic.svp` from a clean
//! engine build (rendered channels are bitwise identical, so one channel
//! suffices). The 48 kHz native-rate excerpt pins the whole synthesis chain
//! at tight tolerance; the 8 kHz excerpt covers the resampler at the
//! tolerance its measured divergence requires (the `rubato` sinc kernel
//! audibly differs from the C++ `WindowedSincInterpolator`, while every
//! upstream stage agrees to ~1e-6).

use std::path::PathBuf;

use opensvr_audio::{CancelToken, NeuralBackend, Renderer, StereoBuffer};
use opensvr_core::{BLICKS_PER_QUARTER, Blick, GroupReference, NoteGroup, Project, Track};

/// Project rate for the resampling test: the voice renders natively at 48 kHz.
const SAMPLE_RATE: u32 = 8000;
/// Score length of `neural-basic.svp` (1.5 s) plus the 0.1 s rest context.
const EXPECTED_FRAMES: f64 = 12_800.0;
/// Measured 8 kHz divergence (max 0.081) plus margin: resampler kernels differ.
const RESAMPLED_TOLERANCE: f32 = 0.12;
/// Measured 48 kHz divergence (max 3.5e-7) plus wide margin: chain agrees.
const NATIVE_TOLERANCE: f32 = 1e-5;

fn voice_path() -> Option<PathBuf> {
    std::env::var("OPENSVR_TEST_VOICE").ok().map(PathBuf::from)
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/audio")
}

fn mean_abs(samples: &[f32]) -> f64 {
    samples.iter().map(|sample| f64::from(sample.abs())).sum::<f64>()
        / samples.len().max(1) as f64
}

fn render_basic(rate: u32) -> StereoBuffer {
    let voice = voice_path().expect("voice must be set");
    let mut project =
        opensvr_svp::load(&golden_dir().join("neural-basic.svp")).expect("basic SVP must parse");
    for track in &mut project.tracks {
        // Every note carries explicit phonemes, so no dictionary is needed.
        track.voice.database = Some(voice.clone());
    }
    Renderer::new(NeuralBackend::new())
        .render(&project, rate, &CancelToken::new())
        .expect("neural render must succeed")
}

struct Excerpt {
    rate: u32,
    frames: usize,
    samples: Vec<f32>,
}

fn load_excerpt(name: &str) -> Excerpt {
    let path: PathBuf = golden_dir().join(name);
    let text = std::fs::read_to_string(&path).expect("excerpt must read");
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let rate = lines
        .next()
        .expect("rate header")
        .strip_prefix("rate ")
        .expect("rate header")
        .parse()
        .expect("rate int");
    let frames = lines
        .next()
        .expect("frames header")
        .strip_prefix("frames ")
        .expect("frames header")
        .parse()
        .expect("frames int");
    let count: usize = lines
        .next()
        .expect("excerpt header")
        .strip_prefix("excerpt ")
        .expect("excerpt header")
        .parse()
        .expect("excerpt int");
    let samples: Vec<f32> = lines
        .take(count)
        .map(|line| line.parse().expect("sample float"))
        .collect();
    assert_eq!(samples.len(), count, "{}: excerpt truncated", path.display());
    Excerpt {
        rate,
        frames,
        samples,
    }
}

fn check_excerpt(name: &str, tolerance: f32) {
    let excerpt = load_excerpt(name);
    let audio = render_basic(excerpt.rate);
    assert_eq!(
        audio.frames(),
        excerpt.frames,
        "{name}: frame geometry must match"
    );
    let rendered = &audio.left[..excerpt.samples.len()];
    let (worst, max) = rendered
        .iter()
        .zip(&excerpt.samples)
        .enumerate()
        .map(|(index, (&got, &want))| (index, (got - want).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("excerpt nonempty");
    assert!(
        max <= tolerance,
        "{name}: frame {worst} differs {max} beyond {tolerance}"
    );
}

#[test]
fn neural_backend_sings_the_basic_phrase_deterministically() {
    let Some(voice) = voice_path() else {
        return;
    };
    let svp = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/audio/neural-basic.svp");
    let mut project = opensvr_svp::load(&svp).expect("basic SVP must parse");
    assert_eq!(project.tracks.len(), 1);
    for track in &mut project.tracks {
        // Every note carries explicit phonemes, so no dictionary is needed.
        track.voice.database = Some(voice.clone());
    }

    let cancel = CancelToken::new();
    let mut renderer = Renderer::new(NeuralBackend::new());
    let first = renderer
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("neural render must succeed");

    // Timeline: 1.5 s of score plus 0.1 s of rest context, up to float dust.
    assert!(
        (first.frames() as f64 - EXPECTED_FRAMES).abs() <= 2.0,
        "frames: got {}, want ~{EXPECTED_FRAMES}",
        first.frames()
    );
    assert!(first.left.iter().all(|sample| sample.is_finite()));
    assert!(first.right.iter().all(|sample| sample.is_finite()));

    // Sung, not silent: the voiced "k a" region carries far more energy
    // than the "sil" note at 1.0..1.2 s (measured on GUMI AI).
    let voiced = mean_abs(&first.left[800..2400]);
    let silence = mean_abs(&first.left[8400..9200]);
    let peak: f32 = first
        .left
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(peak > 0.05, "peak {peak}: the phrase must sing audibly");
    assert!(
        voiced > 10.0 * silence,
        "voiced {voiced} vs silence {silence}: vowels must dominate the rest note"
    );

    // Deterministic: same seed, same summation order, same output bits.
    let second = renderer
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("neural re-render must succeed");
    assert_eq!(first, second, "re-renders must be bitwise identical");
}

#[test]
fn neural_backend_matches_the_cpp_reference_excerpts() {
    if voice_path().is_none() {
        return;
    }
    check_excerpt("neural-basic-48k-ref.txt", NATIVE_TOLERANCE);
    check_excerpt("neural-basic-8k-ref.txt", RESAMPLED_TOLERANCE);
}

/// A two-phrase project loads each file once and matches a fresh render.
///
/// The `neural-basic` notes (explicit phonemes throughout, so no dictionary)
/// sing twice, 2 s apart: the gap splits two phrases with identical synthesis
/// inputs. One backend sings both phrases plus a full re-render; a second,
/// fresh backend re-sings from scratch. All three outputs must be bitwise
/// identical, with exactly one voice load across the shared backend's renders.
#[test]
fn repeated_phrases_load_once_and_match_fresh_renders() {
    let Some(voice) = voice_path() else {
        return;
    };
    let basic =
        opensvr_svp::load(&golden_dir().join("neural-basic.svp")).expect("basic SVP must parse");
    assert_eq!(basic.tracks.len(), 1);
    let source = basic.tracks[0].clone();

    let mut notes = source.main_group.notes.clone();
    notes.extend(source.main_group.notes.iter().map(|note| {
        let mut shifted = note.clone();
        shifted.onset += 4 * BLICKS_PER_QUARTER;
        shifted
    }));
    let mut project = basic.clone();
    project.tracks.clear();
    let mut track = Track::new(
        "lead",
        NoteGroup {
            id: "two-phrase-main".to_owned(),
            name: String::new(),
            notes,
        },
    );
    track.mixer = source.mixer.clone();
    track.voice = source.voice.clone();
    track.voice.database = Some(voice);
    project.tracks.push(track);

    let cancel = CancelToken::new();
    let mut cached = Renderer::new(NeuralBackend::new());
    let first = cached
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("cached render must succeed");
    assert_eq!(
        cached.backend().voice_loads(),
        1,
        "one render loads the voice once"
    );
    assert_eq!(cached.backend().dictionary_loads(), 0);
    assert_eq!(
        cached.backend().phrase_hits() + cached.backend().phrase_misses(),
        2,
        "two phrases synthesize or hit"
    );

    let (hits_before, misses_before) = (
        cached.backend().phrase_hits(),
        cached.backend().phrase_misses(),
    );
    let second = cached
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("re-render must succeed");
    assert_eq!(first, second, "re-renders must be bitwise identical");
    assert_eq!(
        cached.backend().voice_loads(),
        1,
        "re-renders must not reload the voice"
    );
    assert_eq!(
        cached.backend().phrase_hits(),
        hits_before + 2,
        "both re-rendered phrases must hit"
    );
    assert_eq!(
        cached.backend().phrase_misses(),
        misses_before,
        "re-renders must not re-synthesize"
    );

    let fresh = Renderer::new(NeuralBackend::new())
        .render(&project, SAMPLE_RATE, &cancel)
        .expect("fresh render must succeed");
    assert_eq!(first, fresh, "cached renders must match fresh renders");
}

/// Rehosts `source`'s main-group notes behind a library reference with the
/// given right window edge (`None` means unbounded); the track's own main
/// group is left empty so only the windowed notes sing.
fn windowed(basic: &Project, source: &Track, end: Option<Blick>) -> Project {
    let mut project = basic.clone();
    project.tracks.clear();
    project.library.push(NoteGroup {
        id: "windowed-lib".to_owned(),
        name: "windowed".to_owned(),
        notes: source.main_group.notes.clone(),
    });
    let mut track = Track::new(
        "lead",
        NoteGroup {
            id: "empty-main".to_owned(),
            name: String::new(),
            notes: Vec::new(),
        },
    );
    track.mixer = source.mixer.clone();
    track.voice = source.voice.clone();
    track.groups.push(GroupReference {
        group_id: "windowed-lib".to_owned(),
        time_offset: 0,
        pitch_offset: 0,
        begin: 0,
        end,
        instrumental: false,
    });
    project.tracks.push(track);
    project
}

/// A group reference ending mid-project rings no release tail past its edge.
///
/// The `neural-basic` score (1.5 s, explicit phonemes throughout) is moved
/// behind a library reference whose window ends exactly at the last note
/// end. The sung notes are untouched; only the 0.1 s release tail (1.5..1.6 s)
/// must go silent. An unbounded control render proves the tail is
/// voice-audible, so the exact-zero assertion is sensitive.
#[test]
fn group_windows_crop_the_neural_release_tail() {
    let Some(voice) = voice_path() else {
        return;
    };
    let svp = golden_dir().join("neural-basic.svp");
    let basic = opensvr_svp::load(&svp).expect("basic SVP must parse");
    assert_eq!(basic.tracks.len(), 1);
    let source = basic.tracks[0].clone();
    let score_end: Blick = source
        .main_group
        .notes
        .iter()
        .map(|note| note.onset + note.duration)
        .max()
        .expect("basic SVP has notes");

    let tail_start = (1.5 * f64::from(SAMPLE_RATE)) as usize;
    let cancel = CancelToken::new();
    let mut renderer = Renderer::new(NeuralBackend::new());

    // Control: without a window the voice emits audible tail audio, so the
    // cropped render's exact zeros below are a sensitive assertion.
    let mut open = windowed(&basic, &source, None);
    open.tracks[0].voice.database = Some(voice.clone());
    let open = renderer
        .render(&open, SAMPLE_RATE, &cancel)
        .expect("unbounded neural render must succeed");
    assert_eq!(open.frames(), tail_start + 800);
    let open_tail_max = open.left[tail_start..]
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(
        open_tail_max > 0.0,
        "the unbounded release tail must be voice-audible, got {open_tail_max}"
    );

    let mut shut = windowed(&basic, &source, Some(score_end));
    shut.tracks[0].voice.database = Some(voice.clone());
    let shut = renderer
        .render(&shut, SAMPLE_RATE, &cancel)
        .expect("windowed neural render must succeed");
    assert_eq!(shut.frames(), open.frames());
    assert!(
        shut.left[tail_start..].iter().all(|sample| *sample == 0.0),
        "no release tail may ring past the group edge"
    );
    // The window only crops the tail: the sung phrase still sings.
    let peak: f32 = shut
        .left
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!(peak > 0.05, "peak {peak}: the phrase must sing audibly");
}
