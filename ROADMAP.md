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
| Library facade | — | `opensvr` | done (one-call render from a file or string, unified errors, auto backend) |
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
- The pitch frontend's `cmpu1` z-score divides by the stored per-channel
  standard deviation, matching OpenSV+ `PitchFeatures.cpp`; the fork
  multiplied by it. The stored vectors are unmistakably deviations (the
  boolean channels equal `sqrt(p(1-p))`), so the product left every feature
  far outside the trained range and pinned the predicted curve to the voice's
  mean pitch. Long notes hid it because the pitch postprocess recentres
  anything past 0.3 s; fast passages sang a monotone instead of the score.
- Phrase audio is resampled with `rubato` (sinc, fixed ratio), and the
  interpolator's latency is measured and cancelled.
- The Flat voice format is a JSON voice document beside sibling `.dnni` model
  files, with speaker and style vectors as hex-encoded little-endian floats
  and fixed architectures per role (duration `gen2a`, acoustic `dds`, vocoder
  `nhv`). Model IDs are validated as safe file names and `--models-dir`
  overrides the sibling directory. Packaged voices are extracted before use:
  the engine does no archive handling.
- The unset track language is English. Studio sings a project that names no
  voice and no language in English (real English songs saved with an all-empty
  `mainRef.database`), so `Language`'s default is English and the SVP parse
  falls back to it. `svVoice` is written for any non-default language, so an
  explicit Japanese selection survives a save; before this, it was dropped and
  re-read as English.
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

- **`--models-dir` cannot read Studio's installed models directory.** Studio
  keeps its model files in one subdirectory per role (`acoustic`, `duration`,
  `f0`, `vocoder`), the same four names a Flat voice document uses for its
  model references. The engine looks for `<directory>/<id>.dnni` only, in the
  override, then the document's sibling `models` directory, then the
  document's own directory, so pointing `--models-dir` at Studio's own
  directory fails with "model file is missing" for a model that is installed
  one level down. Resolve each model as `<directory>/<role>/<id>.dnni` first
  and `<directory>/<id>.dnni` second, in every directory already searched, so
  a flat directory, an extracted voice package and Studio's installed
  directory all work without symlinks or copies. The same error names only
  the first path tried; it should list every location searched. Tests build
  both layouts from synthetic files, so no proprietary data is involved.

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
    a per-letter guess. A final `-y` is now the vowel `iy`, so `evry` resolves
    to two syllables (`eh v r iy`) as Studio does; the rest of the guess still
    differs from Studio's front end. The hardcoded words disagree with Studio
    as well (`hello` is stored as `hh ah l ow`). For a word the tables do know
    the damage is the symbol alone: `hello` splits `hh ah` + `l ow` where
    Studio splits `hh ax` + `l ow`, so a following `+` still lands on the right
    syllable. The wrong syllable count now makes a following `+` silent where
    Studio carries the missing syllable. Punctuation is stripped before the
    English lookup, so a lyric `score?` or `ces?` resolves as its word, but a
    lyric `=` strips to nothing and still misses. The engine now aliases `=` to
    `+` as a documented guess (`885e33d`), so `FLOP ERA Base.svp` renders; the
    phonemes for the `=` note have still not been read, and `=` may be a
    silence marker rather than a word.
  - *Japanese.* `lookup` now tries the whole romanization first and then, for
    an all-kana lyric, resolves one mora at a time (`そん` → `s o n`, `らい` →
    `r a i`, `にと` → `n i t o`). A small tsu contributes the inventory's `cl`
    closure (`たっ` → `t a cl`). That mapping is inferred from the inventory
    and Studio's explicit English codas (`it` → `ih t cl`), not read from a
    Japanese panel; `ん` → `n` and `うぉ` → `u o` are likewise unverified.
    Long-vowel marks and kanji are still not kana, so they still fall through
    to the Latin guess or an error. Studio's Japanese panel readings for
    `そん`, `たっ` and `うぉ` are the next parity step.
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
  where they are. Measured on a probe project that follows every word with a
  `+` or `-`, Studio sings `cat` then `+` as `k ae t` and silence, `cat` with
  explicit `k ae t s` then `+` as `k ae t s` and silence, `smile` then `-` as
  `s m ay l` and silence, and `cat` with explicit `k ae t p` then `-` as
  `k ae t p` and silence; a leading `+` and `-` with nothing before them sing
  nothing at all. The engine now resolves these cases to `sil` instead of
  aborting the render, pinned by `plus_without_pending_syllables_is_silent`
  and `legato_without_a_syllable_is_silent` (`crates/opensvr-g2p/src/resolve.rs`).

- **`=` is treated as an alias for `+`.** Studio renders the `expectations + = +`
  slot in FLOP as if `=` were a syllable break (observed 2026-10-25), but `=`
  is not a documented Studio lyric; this is a deliberate guess pending a finer
  observation.

- **SVR3 projects store vocal modes per axis.** A project saved with
  `backendType: SVR3` (Kasane Teto 105) writes `vocalModeParams` as
  `{pitch, timbre, pronunciation}` objects instead of scalars, and leaves
  `mainGroup.notes` empty with the arrangement in `library` groups referenced
  from `track.groups`. The render path already composed those groups
  (`Project::placed_notes`, `crates/opensvr-core/src/project.rs:576`); the
  loader now accepts the object when all three axes agree (`6c106db`) and
  rejects a divergent triple rather than averaging it. `info`/`--phonemes` now
  counts and resolves placed notes like the renderer (`ab44dac`), and Japanese
  multi-mora resolution (`390225b`) takes `Anoko_Teto_main.svp` end to end:
  216.10 s, mean −21.5 dBFS, peak −2.6 dBFS with the Teto 104 proxy. Studio
  renders it (observed 2026-10-25); parity is still open on the 104-vs-105
  voice and on the inferred Japanese `cl`/`ん`/`うぉ` readings. Divergent axes
  still need a per-axis mode model.

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

  Plan for the English half, in order:

  1. *Build the data from public sources.* Generate the English entries from
     upstream CMUdict (permissive license; keep its copyright notice with the
     generated file) instead of from any vendor's copy, so the shipped
     dictionary has no proprietary provenance.
  2. *Normalize with a small, documented rule set.* The CLF-style text set is
     CMUdict after light normalization: lowercase keys, stress digits
     stripped, apostrophes dropped from keys, lowercase space-separated
     ARPAbet, one entry per line, alternate pronunciations (`word(2)`)
     handled explicitly. Write these as a spec and implement them once, in a
     converter that both builds the first-party file and ingests other
     CMUdict-shaped sources.
  3. *Diff against the current tables.* Any entry that differs from the
     generated file is either a normalization rule not yet captured or an
     addition made by the table's author. Only the former is ours to adopt;
     record the diff result in the converter's tests, never the foreign
     entries themselves.
  4. *Fill out-of-vocabulary words with a permissively licensed G2P model*
     (check the model's license before bundling), normalized by the same
     converter, ahead of the hardcoded words and digraph rules described
     above.
  5. *Ship the converter, not converted vendor data.* Users convert
     dictionaries they legally hold on their own machines; the repository
     ships only data it has the right to distribute.

  **The same process, for any language.** English is the first instance, not a
  special case. Every language the engine reads (English, Japanese,
  Mandarin, Cantonese, Spanish, and any added later) gets its dictionary
  through these steps, so a new language never starts from someone's
  shipped file:

  1. *Pick a public source and record its license.* Write down the source,
     its version and its license before writing any code, and treat
     share-alike licenses as a constraint on what the generated file may be
     released under. As a starting point to verify, not a conclusion: English
     has CMUdict; Mandarin and Cantonese have CC-CEDICT and Jyutping
     resources; Japanese kana-to-romaji is a small published mapping, with
     word lists from open dictionaries; Spanish orthography is regular enough
     that rules may replace a word list entirely. Prefer rules over word
     lists wherever the language allows it, because rules carry no data
     license.
  2. *Define the target inventory.* State each language's phone inventory
     (the symbol set the engine and voice expect) as a first-party file, so
     the converter has a fixed target and the engine can validate against it.
  3. *Write the mapping from source symbols to inventory symbols* as a
     table, with one test per row, rather than as inline code.
  4. *Convert, then diff against the engine's current tables.* A difference is
     either a missing mapping rule (fix the converter) or a foreign addition
     (leave it out). Keep the diff gated behind an environment variable that
     points at the user's own copy, so no foreign data enters the repository
     or its tests.
  5. *Specify the front-end rules separately from the word list.* Rules
     (multi-mora Japanese, multi-character Mandarin, English reduced vowels)
     live in the first-party format alongside the lookup entries, and are
     tested against hand-written cases, not against a vendor's output.
  6. *Document the language* in one short page: source and license, inventory,
     normalization rules, known gaps, and how to regenerate the file.
     Regeneration must be a single command so a new contributor can add a
     language without reading the engine.
  7. *Ship the converter and the generated first-party file together* where
     the source license allows, and the converter alone where it does not.

- **The `vocalModeParams` amount scale is unverified.** Studio writes the
  vocal-mode slider amounts in `vocalModeParams`, and its files carry values up
  to at least 150; the engine clamps to `0..1`. Studio's `Customized` sentinel
  is handled (the params apply and no style lookup is attempted), but if 100 is
  the nominal amount and 150 a boost, the weights themselves need their own
  observation before mode fidelity can be called equal.

- **Phrase splitting is engine bookkeeping, not Studio behaviour.** The engine
  keeps OpenSV's rest-based splitting (a phrase is a run of notes with no rest
  of at least 0.2 s) as its synthesis, cache and progress boundary. The
  inherited `maximumPhraseSeconds` (30 s) refusal is gone: Studio has no such
  limit, so a connected passage is synthesised as one unit however long it is.
  A 40 s legato fixture renders as one pass at 48 kHz (40.1 s of audio,
  112 s wall time, 933 MiB peak RSS), and the longest run in the real songs
  measured so far is 36.6 s. The separate `maximumPhrasePhonemes` guard (4096)
  remains as a named model/tensor bound; the longest measured passage stays far
  below it, and it is not a musical limit. A rest of exactly 0.2 s still does
  not reliably split because the comparison is made on blick-converted seconds
  and can land just under the threshold, but that now only changes cache
  granularity, not whether the passage can be sung.

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

The `opensvr` facade is not one of these gaps: it is the embedding seam the
goals call for, and it landed first so each gap below is exercised through the
public path rather than only through the CLI. It deliberately models nothing
the gaps will change — a dictionary is still just a directory path, and notes
still have no IDs.

The remaining gaps, in the order to take them, one branch each:

1. **Models directory layout.** Resolve models under `<role>/` as well as flat
   so `--models-dir` works on Studio's installed directory, and list every
   searched path in the missing-model error. It is the smallest gap and
   independent of the rest, so it goes first.
2. **The per-language front ends, with the dictionary format that carries
   them.** G2P parity is the largest gap, and everything downstream of it —
   continuation syllables, multi-mora Japanese, multi-character Mandarin —
   waits on it, so the format and the front ends land together.
3. **Continuation rules.** A `+` or `-` that cannot advance is silence now
   (done); the remaining half is that `-` stops re-singing its predecessor's
   vowel, and the pending-syllable bookkeeping works in every language rather
   than only English.
4. **`evenSyllableDuration`.** Read the per-note attribute and redistribute
   syllable timing the way Studio does.
5. **Note IDs.** Only when an editor-side consumer needs incremental caches.
