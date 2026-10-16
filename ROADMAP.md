# Roadmap

This document lists what is left to port from OpenSV (commit `84ef139`) and how to do it.
Line counts are the C++ sizes (`.cpp` + `.h`) and only indicate effort.
This file is public and self-contained: it never references gitignored
working files or local paths (see `AGENTS.md`).

## Goals

OpenSVR is a modern, cleaner, and potentially faster version of OpenSV's
engine — no JUCE dependency, typed errors, tested and deterministic output.
It is mostly a library with a thin CLI harness: it has no GUI and will not
gain one. It is not a successor to or competitor with standard SV (SV1 still
ships and works); it is a headless engine for offline rendering and
embedding. Parity with OpenSV comes first; speed comes from clean structure
(packed weight layouts, batch-level parallelism, no JUCE overhead), never
from relaxed numerics.

## Where things stand

| Area | OpenSV | opensvr | Status |
| --- | --- | --- | --- |
| Time units, tempo map | `core/Project.cpp` | `opensvr-core::tempo` | done |
| Project model, group placement | `core/Project.*`, `ProjectRenderer::collectNotes` | `opensvr-core::project` | done (no curves, no vocal modes, no note IDs) |
| SVP reading | `core/ProjectFile.cpp` (read half) | `opensvr-svp` | done, lossy by design |
| MIDI export | `core/MidiFile.cpp` (export half) | `opensvr-midi` | done |
| Mixer, pan law, solo/mute | `ProjectRenderer::mixPhrase` | `opensvr-audio::render` | done |
| WAV output | `audio/WaveFile.cpp` | `opensvr-audio::wav` | done |
| CLI | `cli/Main.cpp` | `opensvr-cli` | done, with subcommands instead of flat flags |
| Voice synthesis | `synthesis/*` (10.4k lines) | `opensvr-voice` (timing, pitch, acoustic, vocoder, `VoiceSynthesizer`), `opensvr-nofs`, `opensvr-dnni` | done (neural `NeuralBackend` sings voiced tracks; `ToneBackend` stays as the voiceless fallback) |
| Flat voicebank (document NOFS) | — (post-OpenSV format, not in the commit above) | `opensvr-nofs` + `opensvr-voice` | done (Phase 8) |
| Pronunciation (G2P) | `PhoneSet.*`, `PhonemeDictionary.*`, `resolvePhonemes` | `opensvr-g2p` | done (dictionary + resolution + `info --phonemes`; timing types live here, the duration model in `opensvr-voice`) |
| Phrase building, pitch/vibrato, caching | `ProjectRenderer.cpp` (1.6k lines) | `opensvr-audio` (phrase splitting, `buildPitch`/vibrato/vocal-mode weights, voice/dictionary/phrase caches) | done |
| Editor GUI | `ui/*`, `app/*`, `audio/PreviewEngine.*` (5.6k lines) | none | out of scope |

`opensvr render` sings through the neural voice when a voice database resolves,
and falls back to sine tones otherwise. All eight phases are ported; the only
known gap left is one small item (see "Remaining gaps").

## What was done

Every phase below is ported, in the
listed order, one PR-sized branch per
stage. Phase 2 (NOFS + voice config) lets `opensvr info` show singer metadata.
The `opensvr-dnni` reader parses all real model entries, scalar inference
matches per-op golden tensors, and blocked-`[f32; 16]` weights plus the
incremental dirty-range `Cache` keep those goldens green. The dictionary +
`resolvePhonemes` half of Phase 3 became `opensvr-g2p`, and
`opensvr info --phonemes` prints resolved phonemes per note. Phase 5 landed
stage by stage (timing → pitch → acoustic → vocoder) behind `VoiceSynthesizer`,
each gated on golden files from the previous stage's output, and Phase 6
replaced the sine placeholder with `NeuralBackend` (phrase splitting, pitch
and vibrato building, resampling with latency compensation, group-edge
cropping, caches, track parallelism, progress). Phase 7 finished the CLI
surface (compat shim, exit-code contract, completions/man, env defaults,
release automation) with hygiene (workspace clippy denial, `cargo fmt`).

## Remaining gaps

- **No note IDs.** Only needed for editing and incremental caches; add
  `NoteId(u64)` when an editor-side consumer needs it.

The editor (`ui/*`, `app/*`, `audio/PreviewEngine.*`) stays out of scope.

## Principles for the remaining work

1. **Golden tests against the C++ build.** Build `OpenSVEngine` once with CMake and dump
   intermediate values (parsed DNNI tensors, phoneme lists, F0 curves, vocoder output) from it.
   Check those files into `tests/golden/`, and port each stage until it reproduces them.
   Port a stage only when its inputs can be checked.
2. **One crate per concern, one stage per PR.** Keep dependencies pointing downwards
   (`core` <- `nofs`/`dnni` <- `voice` <- `audio`), and keep `core` free of I/O and heavy dependencies.
3. **Errors as types.** Each crate gets a `thiserror` enum, as in the existing crates. The C++
   `juce::Result` strings become variants; the CLI maps them to exit codes.
4. **Determinism.** Output should match the C++ engine closely enough to compare. Pin the RNG,
   avoid `fast-math`-style shortcuts, and be explicit about summation order in matrix kernels.
5. **No new public surface without a test.** The existing crates average roughly one test per 25 lines of code; keep that up.

## Phase 1: finish the project format (about 600 lines)

Done: `opensvr-core` + `opensvr-svp` + `opensvr-midi` read and write every
field OpenSV understands — curves (`Linear`/`Cosine`/`Cubic` evaluation plus
`pitchDelta`, `vibratoEnv` and the per-reference `systemPitchDelta` contour,
which also gates the neural residual in the pitch builder), vocal modes,
the 14 pitch attributes, audio metadata, unknown-field capture via `Extra`,
`load -> save -> load` fixed points, integer-valued floats, MIDI import,
and `opensvr convert`.

Original plan, kept for reference:

Goal: read and write every field OpenSV understands, with lossless round trips.

- **Preserved fields.** OpenSV keeps unknown JSON in `preservedFieldsJson` strings. In serde, add
  `#[serde(flatten)] extra: serde_json::Map<String, Value>` to each raw struct and carry it in the
  domain type (`Extra` newtype). Enable `serde_json`'s `preserve_order` so saved files diff cleanly.
- **Parameter curves** (`core/ParameterCurve.*`, 178 lines). Add `Curve { mode: Interpolation, points }`
  with `Linear`/`Cosine`/`Cubic` evaluation, plus `pitchDelta`, `vibratoEnv` and `systemPitchDelta`.
  Port the existing curve tests first; the cubic mode has SV-specific weight blending that is easy to get subtly wrong.
- **Vocal modes** (`vocalModes`, `vocalModeParams`, `vocalModePreset`), **pitch attributes** (14 optional doubles,
  table-driven in C++; use a macro or `[Option<f64>; 14]` indexed by an enum), and instrumental audio metadata.
- **Saving.** Derive `Serialize` on the same raw structs, so reading and writing share one definition.
  Test with golden `.svp` files: `load -> save -> load` must be a fixed point and unknown fields must survive.
- **Integer-valued floats.** SV writes integers, but OpenSV also accepts `1.0` for integer fields.
  Add a small `deserialize_with` helper if real-world files need it.
- **Note IDs.** Only needed for editing and incremental caches. Add `NoteId(u64)` when Phase 7 needs it.
- **MIDI import** (`importMidiFile`, about 250 lines). Use the `midly` crate for parsing. Port the logic:
  pair note-on/note-off, attach lyric meta events (type 5), build the tempo map, and convert PPQ ticks to blicks
  (`blicks_per_tick = BLICKS_PER_QUARTER / ppq`, rounding with `i128` to avoid overflow). Consider replacing the
  hand-written writer with `midly` as well, once import tests exist, to share one set of tests.
- **CLI.** Add `opensvr convert in.{svp,mid} out.{svp,mid}`.

## Phase 2: voice database (about 900 lines)

Done: new crate `opensvr-nofs` ports `VoiceDatabase.*` and `VoiceConfiguration.*`
(magic `0xf580`, version 10, entry table, metadata, `0x91` config blob with the
LCG name cipher, duration/acoustic/vocoder model references). `opensvr info`
prints the singer name, vendor and timbre styles. `VoiceDatabase::open` maps
the file read-only with `memmap2` and serves entries as borrowed slices, so
opening a 40 MiB voice costs under a millisecond with no owned copy.

Original plan, kept for reference:

New crate `opensvr-nofs`.

- **NOFS archive** (`VoiceDatabase.*`, 400 lines). Magic `0xf580`, version 10, entry table with key/offset/size,
  metadata table (name, vendor, language, phoneset, timbre styles). Read it with `memmap2` and expose entries as `&[u8]`
  slices, so large voices are not copied; keep `read_entry` lazy like the C++ version. Keep key bytes as `Vec<u8>`,
  since keys can be binary, and expose a lossy `name()` for printable ones.
- **Voice configuration** (`VoiceConfiguration.*`, 450 lines). Decode the per-voice configuration entries (phone sets,
  frame intervals, rap languages, vocal mode names) into a `VoiceConfig` struct. Make `opensvr info` print the voice
  metadata at this point; it is the first user-visible gain, and needs no neural code.
- **Tests.** Tests that need proprietary voice data are gated on an
  environment variable (`OPENSVR_TEST_VOICE`) and skip cleanly when it is
  unset, so CI stays green without proprietary data.

## Phase 3: pronunciation (about 1,500 lines)

Done: new crate `opensvr-g2p` ports `PhoneSet` (DNNI `_psv2` table),
`PhonemeDictionary` (generic, Japanese kana, Mandarin/CEDICT, CMUDict English
pipelines) and `ProjectRenderer::resolvePhonemes` with the `+`/`-` sequencing
around it (75 unit tests plus CLI tests). `opensvr info --phonemes` prints
resolved phonemes per note. `TimingSyllable`/`PhonemeDuration` are types only;
the duration model lands in Phase 5.

Original plan, kept for reference:

New crate `opensvr-g2p` (grapheme-to-phoneme).

- **Phone sets and dictionaries** (`PhoneSet.*`, `PhonemeDictionary.*`, 1,000 lines). The CLF dictionaries are data files,
  so the code is mostly parsing and lookup: `HashMap<String, Vec<String>>` per language.
  Languages and their quirks: Japanese (romaji tables for hiragana, katakana and small kana), Mandarin (pinyin and single
  characters), Cantonese (jyutping), Spanish (X-SAMPA; incomplete upstream), English (`cmudict-07b.txt`, with a project fallback).
- **Phoneme resolution** (`ProjectRenderer::resolvePhonemes`, about 400 lines). This is the logic that turns a note's
  lyrics into phonemes: explicit phonemes win, a leading `.` means "raw phonemes", `+` and `-` continue the previous
  vowel or phoneme, and Latin words in Japanese tracks fall back to the English dictionary. Port it as a pure function
  `fn resolve(note, context, &Dictionary) -> Result<Resolved>` with a table-driven test per rule listed in
  `BUILD_AND_USAGE.md`.
- **Phoneme timing** (`PhonemeTiming.*`, 520 lines). A duration model that depends on DNNI inference, so it lands in Phase 5;
  until then expose the `TimingSyllable` / `PhonemeDuration` types only.
- **Win.** `opensvr info --phonemes` can print resolved phonemes per note.

## Phase 4: neural network runtime (about 1,800 lines)

Done (reader half): `opensvr-dnni` parses the node tree and all `prim`
payloads (50 unit + 2 gated tests, golden `prim0` match).
Done (inference half, scalar): `opensvr-dnni` loads every layer type and runs
every kernel over plain row-major weights (129 unit + 1 golden test over 10
per-op tensors in `tests/golden/dnni/`, dumped from the C++ DNNI translation
units; `CancelToken` moved down to `opensvr-core`).
Done (SIMD + cache): weights pack into blocked `[f32; 16]` columns at load
(zero-padded tails) with block-outer/column-inner multiply lowering to packed
SIMD, plus the incremental dirty-range `Cache` (`DnniCache`,
`DnniRunStatistics`, `run_with_cache`; `run` delegates unchanged). Scalar
numerics unchanged: all per-op goldens green unmodified, packed-vs-scalar
matrix tests agree bitwise. Note there is
no `modl4` branch and no `moda6` in the C++ loader.

Original plan, kept for reference:

New crate `opensvr-dnni`. This is the largest single risk, so build it test-first.

- **Reader** (`DnniReader.*`, 520 lines). Node tree with 20-byte headers, versions 1 and 2 (v2 XORs the child count),
  `prim0`..`prim5` payloads: float, 8-bit and 16-bit row-scaled quantization, and BCSR sparse matrices.
  Port to `struct Node { type_id: u64, payload: Range<usize>, children: Vec<usize> }` over a `Vec<u8>`;
  decode lazily to `Matrix { rows, cols, values: Vec<f32> }`. Fuzz the parser with `cargo-fuzz`, since it reads untrusted files.
- **Inference** (`DnniInference.*`, 1,500 lines). Operations to port: dense, 1-D (dilated) convolution, gated convolution,
  WaveNet residual blocks (`_ncwnv0`/`_gnc1v0`), GRU and bidirectional GRU, and the activations ReLU, tanh, sigmoid, LeakyReLU, ELU, identity, SiLU.
  1. First write a straightforward scalar implementation over `Tensor { frames, channels, values }` and get the golden tests green.
  2. Then speed it up: C++ uses `juce::dsp::SIMDRegister` with 4-vector weight blocks. In Rust, pack weights into
     fixed `[f32; 16]` blocks and let LLVM auto-vectorize (check with `cargo asm`), or use the `wide` crate if it does not.
     Avoid nightly `std::simd`. Use `rayon` only across independent tracks or phrases, not inside kernels, to keep results deterministic.
  3. Finally port the incremental `Cache` (dirty-range inference with receptive-field expansion). It is only an
     optimization for editing, so it can wait until the first end-to-end render works.
- **Cancellation.** Pass `&CancelToken` (already in `opensvr-audio`; move it down into `core` or a tiny `opensvr-cancel` crate)
  and check it between layers, as the C++ code does.

## Phase 5: the models (about 4,500 lines)

Done: new crate `opensvr-voice`, one module per C++ stage, ported in order
(timing -> pitch -> acoustic -> vocoder) with `VoiceSynthesizer` as the
orchestrator implementing `predict`, `predict_pitch` and `render`. Gated
voice tests prove each stage against real model data; the suite stays green
without proprietary files.

Original plan, kept for reference:

New crate `opensvr-voice`, one module per C++ stage. Port in this order, since each stage's output is the next one's golden input:

1. **Timing**: `PhonemeTiming` -> phoneme durations.
2. **Pitch**: `PitchContext`, `PitchFeatures`, `PitchModel`, `PitchDecoder`, `PitchCurve`, `PitchPostprocess` (about 2,000 lines).
   Predicts the automatic F0 curve. Frame intervals differ between pitch and acoustic models; keep them as separate typed values.
3. **Acoustic**: `AcousticFeatures`, `FeatureNormalizer`, `AcousticModel` (about 900 lines).
4. **Vocoder**: `NeuralVocoder`, `VocoderFilterBank`, `VocoderResidual`, `VocoderSpectralHeads`,
   `PeriodicExcitation`, `LipRadiation` (about 1,700 lines). Needs an FFT: use `realfft`/`rustfft` and compare against
   `juce::dsp::FFT` output, including scaling conventions.
5. **`VoiceSynthesizer`** (302 lines) as the orchestrator that implements the public API: `predict`, `predict_pitch`, `render`.

Notes on exactness:

- **Random numbers.** Both models use `std::mt19937` with the default seed 5489, fed into a hand-written Box-Muller
  (not `std::normal_distribution`), so the stream is portable. Use `rand_mt::Mt19937GenRand32` or a 30-line port,
  and copy the Box-Muller rejection loop exactly. Do not substitute `rand_distr`.
- **Float behavior.** Use `f32` where C++ does, convert at the same points, and keep the same summation order in reductions.
  Expect small differences from FMA and different `exp`/`tanh` implementations; set tolerances per stage (start near `1e-5` relative)
  rather than demanding bit-identical output.
- **State.** `VoiceSynthesizer::State` holds per-phrase caches. In Rust make it an owned struct passed as `Option<&mut State>`.

## Phase 6: renderer parity (about 1,500 lines)

Done in `opensvr-audio`: `NeuralBackend` replaced the sine placeholder behind
the `VoiceBackend` seam (voiced projects sing, voiceless ones keep tones with
a warning), with `Phrase` splitting at rests, `buildPitch` /
`buildVibratoEnvelope` / `buildVocalModeWeights`, `rubato` resampling with
measured latency compensation, group-window cropping, file-stamp voice /
dictionary / phrase caches, scoped-thread track parallelism with fixed-order
summing, and a `ProgressSink` trait wired to `indicatif` behind `-q`.

Original plan, kept for reference:

Replace `ToneBackend` with `NeuralBackend` and widen the seam as needed. Work in `opensvr-audio`:

- **Phrase splitting.** `ProjectRenderer` cuts a track into phrases at rests and renders each separately, which is what makes
  incremental re-synthesis possible. Add `Phrase { notes, context_before, context_after }` and change
  `VoiceBackend::render_phrase` to take it. This is a breaking change to the trait, and it is expected: the note-level
  trait was designed for placeholders.
- **Pitch and vibrato building** (`buildPitch`, `buildVibratoEnvelope`, `buildVocalModeWeights`, about 250 lines). Needs the
  curves from Phase 1. `localCurvePosition` converts absolute to group-local positions with saturation; port it with `checked_*`/`saturating_*`.
- **Resampling.** C++ resamples native-rate phrase audio with `juce::WindowedSincInterpolator` and compensates its 100-sample latency.
  Use the `rubato` crate (sinc, fixed ratio), and measure and cancel the latency in a test with an impulse.
- **Cropping at group edges.** The C++ release tail is limited by `absoluteEnd` and `absoluteBegin`; the current code only clips notes.
- **Caching.** Add the voice, dictionary and phrase caches (`CachedVoice`, `CachedDictionary`, `CachedPhrase`) keyed by file
  stamp (size + mtime). They matter for re-rendering after small edits, less for one-shot CLI renders, so do this last.
- **Parallelism.** Tracks are independent: render them with `rayon` or scoped threads, then sum in a fixed order.
- **Progress.** Add a `ProgressSink` trait (phrases done / total) and hook up `indicatif` in the CLI behind `-q`.

## Phase 7: CLI polish

Done: the flat-form shim routes `opensvr project.svp -o out.wav --info` to
the matching subcommand, the exit-code table is tested and rides in `--help`
and `opensvr.1`, `complete`/`man` ship completions and the man page (a sync
test guards the checked-in page byte for byte), `OPENSVR_VOICE` /
`OPENSVR_DICT` default `--voice` / `--dict`, and CI builds, lints and uploads
per-OS artifacts on tags.

Original list, kept for reference:

- **Compatibility shim.** Accept the original flat form `opensvr project.svp -o out.wav --info` by routing it to `render`/`info`
  when the first argument is not a subcommand.
- **Exit-code table** in the README, kept in sync by the existing integration test.
- **Shell completions and man page** via `clap_complete` and `clap_mangen`.
- **Config.** `OPENSVR_VOICE` and `OPENSVR_DICT` environment variables as defaults for `--voice` and `--dict`.
- **Release engineering.** `cargo dist` or a plain CI matrix for Linux, macOS and Windows; cache `target/` and run `cargo clippy -D warnings`.

## Phase 8: Flat voicebanks (done)

Done: `opensvr info` and `opensvr render` accept Flat voices — a JSON voice
document beside sibling `.dnni` model files. `VoiceDatabase` probes document
mode, `VoiceConfiguration` resolves the fixed-architecture model references,
`VoiceEmbeddings` decodes the hex speaker/style vectors (`(base)`/`(default)`
pairing, named vocal modes, `f0_style`/`duration_style` overrides that replace
the model-embedded vectors at load), document vocal modes mix at render from
absolute style vectors in sorted-name order, and `--voice` takes the document
path with `--models-dir` (and its environment-variable default) overriding
model lookup. No Flat-only DNNI hashes were needed: the sampled voices use
the already-ported model families, and a gated Flat render test sings the
example end-to-end (vocal modes included).

Goal (kept for reference): `opensvr info` and `opensvr render` accept Flat voices — a JSON voice
document (singer metadata plus `acoustic`/`vocoder`/`duration`/`f0` model
IDs) beside a directory of sibling `.dnni` model files, with speaker and
style embeddings carried as hex-encoded little-endian float vectors in the
document (`styles` entries, `f0_style`, `duration_style`). The Flat models
are the same families the ported stages already run (acoustic `_rldtg0`,
duration `_pdwpv2`, vocoder `_nhntv2`, pitch wrapper roots), so this phase is
loader plus embedding work, not new kernels. One PR-sized branch per stage,
in this order:

- **Document-mode database** (in `opensvr-nofs`). Probe the voice file: a
  leading `{` (past any byte-order mark and whitespace) means a JSON voice
  document, anything else means the binary container from Phase 2, which stays
  byte-identical. Validate each model ID as a safe file name (bounded length,
  no separators or reserved device names, case-insensitive `.dnni` lookup)
  and resolve it against the document's sibling model directory, with an
  explicit models-directory override taking precedence. Serve model bytes
  through the same entry interface as container entries (same size caps);
  alias the `f0` entry as `f0model-dds`, which the pitch model resolves by
  fixed name. Map document metadata (name, vendor, language, phoneset,
  version, languages, timbre styles) onto the same fields `info` prints.
- **Voice configuration** (in `opensvr-nofs` or `opensvr-voice`, as fits).
  The document carries no architecture strings, so fix them per role:
  duration `gen2a`, acoustic `dds`, vocoder `nhv`.
- **Embeddings** (in `opensvr-voice`). Decode each hex vector with a
  finite-float check: `styles` `(base)` and `(default)` become the acoustic
  speaker and default style (they must pair), other named entries become
  vocal modes, `f0_style` (32 channels) overrides the pitch speaker vector,
  and `duration_style` (128 channels) overrides the duration voice vector.
  Extra per-style fields the engine does not consume are ignored.
- **Model coverage** (in `opensvr-dnni` + `opensvr-voice`). Extend the
  version-2 type-hash table for any Flat-only hashes (notably pitch wrapper
  roots) and thread the embedding overrides through the `timing`/`pitch`/
  `acoustic` loaders and the synthesizer, keeping container-voice renders
  unchanged.
- **CLI.** `--voice` accepts the voice document path; model lookup defaults
  to its sibling directory with a `--models-dir` flag and environment-variable
  default. No archive handling: packaged voices are extracted before use.
- **Tests.** Unit tests for document probing, safe-ID validation, embedding
  decode, and metadata mapping; tests needing a real Flat voice are gated on
  an environment variable (`OPENSVR_TEST_FLAT`) and skip cleanly when unset,
  so CI stays green without proprietary data.

## Out of scope: the editor

`ui/*`, `app/*` and `audio/PreviewEngine.*` (about 5,600 lines) are a JUCE desktop editor: piano roll, arrangement, parameter
view, playback and undo/redo (`ProjectDocument.*`). OpenSVR will not port them — there is no GUI here, at most the thin CLI.
Also out of scope, as editor-only adjuncts: `audio/PitchAudition.*` (realtime key-preview oscillator tied to the audio device thread)
and `audio/RenderVisualization.*` (`PhraseVisualization` waveform peaks for display).
Named-but-trivial engine adjuncts that ride along with their phases instead of getting their own:
`synthesis/SynthesisStatistics.h` and `audio/RenderStatistics.h` (profiling counters) land with Phases 5–6
alongside `ProgressSink`. An editor built on these crates would be a different project, not this one.

## Suggested order and sizing

Rows 1–7 shipped; row 8 is planned (the line counts were the original
estimates):

| Phase | Result you can see | Approx. new Rust lines |
| --- | --- | --- |
| 1 | Lossless `.svp` round trip, MIDI import, `convert` | 600 |
| 2 | `info` shows voice metadata | 900 |
| 3 | Resolved phonemes per note | 1,500 |
| 4 | DNNI files load and run, golden tensors match | 1,800 |
| 5 | First sung output (single phrase) | 4,500 |
| 6 | Full-project render matching `opensv-cli` | 1,500 |
| 7 | CLI polish | 300 |
| 8 | Flat voices load and sing | about 600 |

Row 8 is planned (Phase 8). The C++ scope the table estimated against was
about 14,400 lines without the editor; the Rust port landed at
a comparable size with roughly one test per 25 lines of
code, and the workspace suite (555 tests, green without proprietary data)
is the ground truth for what works.
