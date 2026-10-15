<div align="center">

# Open _(Singing Voice)_

</div>

A Rust port of [OpenSV](https://github.com/cubeww/OpenSV)'s CLI and engine.

It is not an SV Studio replacement or competitor: SV1 still ships and works. OpenSVR is a headless
engine/library for offline rendering and embedding, not an editor.

Goals are modernization and maintainability, not raw speed: no JUCE dependency, typed errors,
tested/deterministic output, and CPU-efficient offline synthesis at parity with OpenSV.

It's very minimal and implements little-to-nothing, for now.

> [!IMPORTANT]
> **The neural voice engine sings when it has a voice.** `render` mixes the project correctly (timeline, tempo map, group
> placement, transposition, cropping, gain, pan, solo/mute) and sings tracks whose voice database resolves to a real
> `.nofs` file through the neural voice (voice, dictionary and phrase files are cached by file stamp); without one,
> notes fall back to sine tones. `info` does read voice databases (`opensvr-nofs`): it prints the singer name, vendor and vocal
> modes when `--voice` points at a `.nofs` file. See [ROADMAP.md](ROADMAP.md) for what remains of the synthesis port.

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