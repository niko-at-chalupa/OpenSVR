<div align="center">

# Open _(Singing Voice)_

</div>

A Rust port of [OpenSV](https://github.com/cubeww/OpenSV)'s CLI

It's very minimal and implements little-to-nothing, for now.

> [!IMPORTANT]
> **The neural voice engine is not ported yet.** `render` mixes the project correctly (timeline, tempo map, group
> placement, transposition, cropping, gain, pan, solo/mute) but sings sine tones instead of using a voice
> database. `info` does read voice databases (`opensvr-nofs`): it prints the singer name, vendor and vocal
> modes when `--voice` points at a `.nofs` file. See [ROADMAP.md](ROADMAP.md) for how to port the neural synthesis.