<div align="center">

# Open _(Singing Voice)_

</div>

A Rust port of [OpenSV](https://github.com/cubeww/OpenSV)'s CLI and engine.

It is not an SV Studio replacement or competitor: SV1 still ships and works. OpenSVR is a headless
engine/library for offline rendering and embedding, not an editor.

Goals are modernization and maintainability, not raw speed: no JUCE dependency, typed errors,
tested/deterministic output, and CPU-efficient offline synthesis at parity with OpenSV.

The engine sings. `render` mixes the project (timeline, tempo map, group
placement, transposition, cropping, gain, pan, solo/mute) and renders tracks
whose voice database resolves to a real `.nofs` file through the neural voice
(timing → pitch → acoustic → vocoder); without one, notes fall back to sine
tones with a warning. Voice, dictionary and phrase files are cached by file
stamp, tracks render concurrently, and phrase progress reports to stderr
(`-q` silences it; the output bytes are unchanged).

Subcommands are `info` (tracks, voice/dictionary presence, singer metadata,
`--phonemes` per-note resolution), `render` (24-bit stereo WAV), `midi` (SMF
export), `convert` (`.svp` ↔ `.mid`), `complete` (shell completions) and `man`
(troff page, checked in as `opensvr.1`). The flat form
`opensvr project.svp -o out.wav` still routes to the matching subcommand, and
`OPENSVR_VOICE` / `OPENSVR_DICT` default `--voice` / `--dict` when those flags
are absent. See `opensvr --help` and `opensvr.1` for flag details; exit codes
are `0 ok, 1 usage, 2 load, 3 render/write, 130 cancelled`.

Tests that need proprietary voice or dictionary data are gated on environment
variables and skip cleanly without them, so the suite is green on a bare
checkout. [ROADMAP.md](ROADMAP.md) records what was ported and the small gaps
that remain.

Porting is cross-checked against a second reference, OpenSV+ (used with permission):
MysticLD's research engine with additional fixes, which has no public
repository of its own. Credit and thanks to MysticLD for letting this project use that research.
Where the two references disagree, behavior is verified by measurement against committed golden
tests, never copied sight unseen.

## Releases

Releases are manual: no automation cuts tags or publishes anything. To cut
one, tag `vX.Y.Z`, let the `ci` workflow finish on the tag, then attach the
per-OS artifacts it uploads (the `opensvr` binary plus `opensvr.1`) to the
release notes with the exit-code table (`0 ok, 1 usage, 2 load, 3
render/write, 130 cancelled`). Voice databases and dictionaries are never
bundled; users point at files they own.