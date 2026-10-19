# Roadmap

OpenSVR is a headless rendering engine for Synthesizer V Studio projects: it
loads an `.svp`, resolves its lyrics to phonemes, sings them through a voice
model, and writes WAV or MIDI. It is mostly a library with a thin CLI
harness: it has no GUI and will not gain one. It is not a competitor with
Studio, which still ships and works — it is an offline engine for rendering
and embedding. This file is public and self-contained: it never references
gitignored working files or local paths (see `AGENTS.md`).

Its compatibility target is **Synthesizer V Studio (Flat)**: where the engine
and Studio disagree about what a note sounds like, Studio is right. Where
OpenSVR's own structure, speed and CLI are concerned, the engine is designed
on its own terms.

History: the engine began as a Rust port of OpenSV, the open-source C++
engine (commit `84ef139`), and the ported code still follows it closely — so
when a question is about how an already-ported stage works, OpenSV is the
reference to read. The port is complete; what remains is Studio parity, the
list under "Remaining gaps", which deliberately departs from the ported
behaviour wherever the two disagree.

## Goals

A clean, fast engine with typed errors, tested and deterministic output, and
no JUCE dependency. Studio parity is the target for *what a note sounds like*
— its phonemes, its syllable split, its timing and its pitch. Speed comes
from clean structure (packed weight layouts, batch-level parallelism, no JUCE
overhead) and from kernel arithmetic at least as accurate as the ported
baseline. The command-line interface is deliberately outside the parity
target: it is designed for its own users rather than cloned from the old
engine's grammar. What the engine trades for speed is bit-for-bit
reproducibility with the ported arithmetic, never accuracy — see the
determinism principle below.

## Where things stand

| Area | Ported from | opensvr | Status |
| --- | --- | --- | --- |
| Time units, tempo map | `core/Project.cpp` | `opensvr-core::tempo` | done |
| Project model, group placement, curves, vocal modes | `core/Project.*`, `core/ParameterCurve.*` | `opensvr-core::project` | done (no note IDs) |
| SVP reading | `core/ProjectFile.cpp` (read half) | `opensvr-svp` | done, lossy by design |
| MIDI import and export | `core/MidiFile.cpp` | `opensvr-midi` | done |
| Mixer, pan law, solo/mute | `ProjectRenderer::mixPhrase` | `opensvr-audio::render` | done |
| WAV output | `audio/WaveFile.cpp` | `opensvr-audio::wav` | done |
| CLI | `cli/Main.cpp` | `opensvr-cli` | done, redesigned on its own terms (subcommands, global flags, default outputs, `-` for stdout) |
| Voice synthesis | `synthesis/*` | `opensvr-voice` (timing, pitch, acoustic, vocoder, `VoiceSynthesizer`), `opensvr-nofs`, `opensvr-dnni` | done (neural `NeuralBackend` sings voiced tracks; `ToneBackend` stays as the voiceless fallback) |
| Flat voicebank (JSON voice document) | — (a format the engine reads, not a ported one) | `opensvr-nofs` + `opensvr-voice` | done |
| Pronunciation (G2P) | `PhoneSet.*`, `PhonemeDictionary.*`, `resolvePhonemes` | `opensvr-g2p` | done, ported parity (dictionary + resolution + `info --phonemes`; timing types live here, the duration model in `opensvr-voice`; Studio parity gaps under "Remaining gaps") |
| Phrase building, pitch/vibrato, caching | `ProjectRenderer.cpp` | `opensvr-audio` (phrase splitting, `buildPitch`/vibrato/vocal-mode weights, voice/dictionary/phrase caches) | done |
| Editor GUI | `ui/*`, `app/*`, `audio/PreviewEngine.*` | none | out of scope |

`opensvr render` sings through the neural voice when a voice database
resolves, and falls back to sine tones otherwise. Every stage of the port has
landed; what remains is the Synthesizer V parity work listed under "Remaining
gaps".

### Notes on the ported core

The decisions in the landed code that still constrain what comes next.

- `opensvr-nofs` maps a voice database read-only with `memmap2` and serves its
  entries as borrowed slices, so opening a 40 MiB voice costs under a
  millisecond and copies nothing.
- `opensvr-dnni` parses the node tree and every `prim` payload, then packs
  weights into blocked `[f32; 16]` columns at load (zero-padded tails), with
  the kernel chosen at runtime — fused AVX2 where the CPU reports AVX2 and
  FMA, the SSE2 baseline otherwise — so the shipped binary stays portable and
  needs no build flags. Each lane's columns are dealt round-robin over two
  partial sums, folded once per block. The loader has no `modl4` branch and no
  `moda6`.
- Noise is reproduced exactly: `mt19937` at the default seed 5489 through a
  hand-written Box-Muller, never the standard library's normal distribution.
- The `systemPitchDelta` curve is not only saved and loaded; it gates the
  neural residual in the pitch builder. Unknown JSON survives a round trip
  through the `Extra` capture, and `load -> save -> load` is a fixed point.
- Phrase audio is resampled with `rubato` (sinc, fixed ratio), and the
  interpolator's latency is measured and cancelled.
- The Flat voice format is a JSON voice document beside sibling `.dnni` model
  files, with speaker and style vectors as hex-encoded little-endian floats
  and fixed architectures per role (duration `gen2a`, acoustic `dds`, vocoder
  `nhv`). Model IDs are validated as safe file names and `--models-dir`
  overrides the sibling directory. Packaged voices are extracted before use:
  the engine does no archive handling.
- Tests that need proprietary data are gated on an environment variable
  (`OPENSVR_TEST_VOICE`, `OPENSVR_TEST_FLAT`) and skip cleanly when it is
  unset, so CI stays green without it.

## Remaining gaps

What is left is not "something OpenSV has and OpenSVR lacks" but the places
where the engine still diverges from Synthesizer V Studio. They were found by
rendering real projects through the engine and comparing the notes against
what Studio displays for the same notes. OpenSV behaves the same way, so
nothing already ported settles them: fixing them means departing from the
ported behaviour, recorded here rather than silently taken. The list is what
has been checked so far, not a claim of completeness.

- **No note IDs.** Only needed for editing and incremental caches; add
  `NoteId(u64)` when an editor-side consumer needs it.

- **Grapheme-to-phoneme parity, in every language.** Studio phonemizes
  whatever text sits on a note; the engine is dictionary-first, and its East
  Asian front ends take one dictionary unit per note, so text the shipped
  tables do not list is either invented as if it were English or aborts the
  render. What has been checked so far already spans more than one language:

  - *English.* Studio's front end is not CMUDict: it writes `iy` for a final
    `-y` (`happy` -> `hh ae p iy`, `evry` -> `eh v r iy`), a reduced `ax`
    where CMUDict has `ah` (`and` -> `ax n d`, `hello` -> `hh ax l ow`) and
    `ey` for `ay` (`day` -> `d ey`). Past CMUDict the engine uses 14 hardcoded
    words, then 11 digraph patterns (`sh ch th ph ng qu ee ea oo ow ai`), then
    a per-letter guess in which a final `y` is a consonant, so `evry` resolves
    to one syllable (`eh v r y`) where Studio resolves two. The hardcoded
    words disagree with Studio as well (`hello` is stored as `hh ah l ow`).
    For a word the tables do know the damage is the symbol alone: `hello`
    splits `hh ah` + `l ow` where Studio splits `hh ax` + `l ow`, so a
    following `+` still lands on the right syllable. The wrong syllable count
    is what then makes a following `+` fail.
  - *Japanese.* `lookup` romanizes the lyric character by character and then
    looks the concatenation up as one key, while the kana tables hold one mora
    per entry. A multi-mora lyric such as `あん` (`a N` in Studio) is not a
    key, and the long-vowel mark and kanji are not kana at all. The fallback
    is the Latin letter guess, which for Japanese either names phones the
    inventory lacks, aborting the render, or invents Latin phones for kana.
  - *Mandarin.* A note takes exactly one listed syllable; a multi-character
    lyric is rejected with "Split multi-character lyrics across notes".
  - *Cantonese, Spanish.* The CLF tables the ported engine uses, not yet
    compared with Studio at all.

  This is not one patch. It is a per-language front end that accepts the text
  Studio accepts, emits Studio's symbol inventory, and never turns an unknown
  input into either an abort or a wrong guess. The continuation bookkeeping is
  part of the same work: the resolver only fills its pending-syllable list
  when the previous note resolved to English, so a multi-syllable lyric in any
  other language could not be handed to a following `+` even once the front
  end can read it.

- **A `+` or `-` that cannot advance is silence, not an error.** Studio
  renders the continuation as silence and leaves the previous note's phonemes
  where they are; the engine aborts the whole render with "'+' advances to the
  next syllable, but no remaining syllable is available" (or, for `-`, "A
  legato lyric (-) must immediately follow a note with a resolved syllable").
  Measured on a probe project that follows every word with a `+` or `-`,
  Studio sings `cat` then `+` as `k ae t` and silence, `cat` with explicit
  `k ae t s` then `+` as `k ae t s` and silence, `smile` then `-` as `s m ay l`
  and silence, and `cat` with explicit `k ae t p` then `-` as `k ae t p` and
  silence; a leading `+` and `-` with nothing before them sing nothing at all.
  Even after G2P parity these must degrade to a silent note, never a hard
  failure.

- **`-` must only stretch the previous note's last phoneme.** The same probe
  exposes the other half of the `-` divergence: Studio never gives a `-` a
  phoneme of its own, it lets the previous syllable ring into the note. The
  engine instead re-sings the previous note's last vowel, and moves the
  previous note's trailing coda onto the `-` (`smile` resolves to `s m ay` and
  its `-` to `ay l`, so the stream gains a repeated vowel; a `cat` with
  explicit `k ae t p` resolves its `-` to `ae`). Explicit `phonemes` are not
  affected: they already take priority on their own note and are not re-split,
  which is what Studio does — a `hello` whose `phonemes` are `hh eh l ow`
  followed by `+` sings `hh eh l ow` and then the lyric's next syllable
  `l ow` in both.

- **`evenSyllableDuration` is parsed but never read.** `opensvr-svp`
  round-trips the per-note attribute; nothing in `opensvr-voice` or
  `opensvr-audio` consults it, so a project that sets it times its syllables
  the same as one that does not, where Studio re-distributes them.

- **A first-party dictionary format.** The engine reads the third-party CLF
  text set (`english-arpabet-phones.txt`, `cmudict-07b.txt`,
  `japanese-hira2romaji-dict.txt`, ...). Those files are not ours to
  redistribute, and Studio no longer ships them in that form anyway: its
  installed `clf-data` holds compiled inventories, so `--dictionary` pointed
  at Studio's directory fails before a note resolves. Define a dictionary
  format OpenSVR owns — a per-language file carrying the symbol inventory, the
  lookup entries and the front-end rules — so the engine ships usable defaults
  and stops depending on files the user has to supply. Two constraints: it
  must pronounce a word the same as the current tables wherever they agree,
  and it is where the per-language G2P work above lands (multi-mora Japanese,
  multi-character Mandarin, English reduced vowels).

- **Phrase splitting and its caps are engine bookkeeping, not Studio
  behaviour.** The engine inherits OpenSV's rest-based splitting (a phrase is
  a run of notes with no rest of at least 0.2 s) plus two hard caps,
  `maximumPhraseSeconds` (30 s) and `maximumPhrasePhonemes` (4096); a phrase
  over either is refused, so a real project with a 33 s legato passage fails
  to render. Studio never cuts a connected passage at rests, so a passage
  longer than either cap must render, the caps must go, and any splitting kept
  for incremental synthesis must stay invisible in the output.

The editor (`ui/*`, `app/*`, `audio/PreviewEngine.*`) stays out of scope.

## Principles for the remaining work

1. **Settle behaviour against a reference before writing the code.** The
   ported arithmetic keeps its goldens — intermediate values (parsed tensors,
   phoneme lists, F0 curves, vocoder output) dumped once from a build of the
   C++ engine and checked into `tests/golden/` — and a change to a kernel is
   only done when those stay green. Studio parity has no golden to inherit,
   so those gaps are settled by probing Studio itself: put the case on notes
   in a project, read what Studio resolves and sings, and write the
   observation down next to the gap.
2. **One crate per concern, one step per PR.** Keep dependencies pointing downwards
   (`core` <- `nofs`/`dnni` <- `voice` <- `audio`), and keep `core` free of I/O and heavy dependencies.
3. **Errors as types.** Each crate gets a `thiserror` enum, as in the existing crates. The
   `juce::Result` strings the ported code returned are its variants; the CLI maps them to exit codes.
4. **Determinism.** Output should match the ported baseline closely enough to compare. Pin the RNG, and
   be explicit about summation order in matrix kernels: the packed kernels may fuse a multiply and
   an add into a single-rounded operation, and may interleave several partial sums and fold them,
   when that is faster. Both change the last bits rather than the structure of the sum, and both
   stay inside the golden tolerance. Per-op goldens are necessary but not sufficient, though: some
   passages synthesize unstably and turn a last-place difference into an audible one, so compare
   end-to-end renders as well when a change touches the arithmetic.
5. **No new public surface without a test.** The existing crates average roughly one test per 25 lines of code; keep that up.

## Out of scope: the editor

`ui/*`, `app/*` and `audio/PreviewEngine.*` are a JUCE desktop editor: piano
roll, arrangement, parameter view, playback and undo/redo
(`ProjectDocument.*`). OpenSVR has no GUI and will not grow one, so none of it
is in scope. Also out of scope, as editor-only adjuncts:
`audio/PitchAudition.*` (realtime key-preview oscillator tied to the audio
device thread) and `audio/RenderVisualization.*` (`PhraseVisualization`
waveform peaks for display).

The two named-but-trivial engine adjuncts, `synthesis/SynthesisStatistics.h`
and `audio/RenderStatistics.h` (profiling counters), are the only engine
pieces still unported; they land whenever a caller wants the numbers. An
editor built on these crates would be a different project, not this one.

## What comes next

The remaining gaps, in the order to take them, one branch each:

1. **The per-language front ends, with the dictionary format that carries
   them.** G2P parity is the largest gap, and everything downstream of it —
   continuation syllables, multi-mora Japanese, multi-character Mandarin —
   waits on it, so the format and the front ends land together.
2. **Continuation rules.** `+` and `-` degrade to silence instead of aborting
   the render, `-` stops re-singing its predecessor's vowel, and the
   pending-syllable bookkeeping works in every language rather than only
   English.
3. **Phrase building.** Drop the rest-based splitting and the two hard caps so
   a long connected passage renders, keeping whatever splitting remains
   invisible in the output.
4. **`evenSyllableDuration`.** Read the per-note attribute and redistribute
   syllable timing the way Studio does.
5. **Note IDs.** Only when an editor-side consumer needs incremental caches.
