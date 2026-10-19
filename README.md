<div align="center">

# Open _(Singing Voice)_

</div>

An engine that uses _(Singing Voice)_ databases to render singing.

[Click here](assets/sounds/watertherosesvr.wav) to listen to a snippet of [FLAVOR FOLEY's "Water the Roses,"](https://youtu.be/NA3MJmcyPpE) rendered by OpenSVR_

OpenSVR comes with a CLI but it's meant to be a library. I find it frivolous to come with an editor, this shouldn't act as a direct competitor to a certain other piece of software when such works _perfectly fine_.

## As a library

The `opensvr` crate is the one-call facade: load a project from a file or a
string, render it, and write WAV. Voice databases, dictionaries and neural
models are not bundled; point at files you own.

```rust
use opensvr::{CancelToken, Options, encode_wav, render_svp};

let svp = std::fs::read_to_string("song.svp")?;
let options = Options::default(); // 48 kHz, auto backend
let cancel = CancelToken::new();
let audio = render_svp(&svp, &options, &cancel)?;
let wav = encode_wav(&audio, options.sample_rate, &cancel)?;
std::fs::write("song.wav", wav)?;
```

With no voice database the project renders as placeholder tones, exactly like
`opensvr render`; set `backend: Backend::Neural` to make a missing voice an
error instead. The per-crate APIs remain public for callers that need a single
stage.

Tests that need proprietary voice or dictionary data are gated on environment
variables and skip cleanly without them, so the suite is green on a bare
checkout. [ROADMAP.md](ROADMAP.md) records where the engine is going and the
small gaps that remain.
