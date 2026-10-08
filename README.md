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
> **The neural voice engine is not ported yet.** `render` mixes the project correctly (timeline, tempo map, group
> placement, transposition, cropping, gain, pan, solo/mute) but sings sine tones instead of using a voice
> database. `info` does read voice databases (`opensvr-nofs`): it prints the singer name, vendor and vocal
> modes when `--voice` points at a `.nofs` file. See [ROADMAP.md](ROADMAP.md) for how to port the neural synthesis.