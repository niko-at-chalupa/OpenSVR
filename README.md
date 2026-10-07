<div align="center">

# Open _(Singing Voice)_

</div>

A Rust port of [OpenSV](https://github.com/cubeww/OpenSV)'s CLI

It's very minimal and implements little-to-nothing, for now.

> [!IMPORTANT]
> **The voice engine is not ported.** `render` mixes the project correctly (timeline, tempo map, group
> placement, transposition, cropping, gain, pan, solo/mute) but sings sine tones instead of using a voice
> database. See [ROADMAP.md](ROADMAP.md) for how to port the neural synthesis.