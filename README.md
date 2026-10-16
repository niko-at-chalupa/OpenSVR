<div align="center">

# Open _(Singing Voice)_

</div>

An engine that uses _(Singing Voice)_ databases to render singing.

<div align="center">
<audio controls src="assets/sounds/watertherosesvr.wav"></audio> <br />

_[FLAVOR FOLEY's "Water the Roses,"](https://youtu.be/NA3MJmcyPpE) rendered by OpenSVR_

</div>

OpenSVR comes with a CLI but it's meant to be a library. I find it frivolous to come with an editor, this shouldn't act as a direct competitor to a certain other piece of software when such works _perfectly fine_.

Tests that need proprietary voice or dictionary data are gated on environment
variables and skip cleanly without them, so the suite is green on a bare
checkout. [ROADMAP.md](ROADMAP.md) records what was ported and the small gaps
that remain.
