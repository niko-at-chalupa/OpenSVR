use std::{
    ffi::OsString,
    fs,
    io::{self, Cursor, Seek, Write},
    path::{Path, PathBuf},
};

use hound::{SampleFormat, WavSpec, WavWriter};
use thiserror::Error;

use crate::{CancelToken, StereoBuffer};

const BITS_PER_SAMPLE: u16 = 24;
const FULL_SCALE: f64 = 8_388_607.0; // 2^23 - 1
const CHUNK_FRAMES: usize = 4096;

#[derive(Debug, Error)]
pub enum WavError {
    #[error("cannot write {}: {source}", .path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("cannot replace {}: {source}", .path.display())]
    Replace {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("cannot encode WAV: {source}")]
    Encode {
        #[source]
        source: hound::Error,
    },
    #[error("WAV export was cancelled")]
    Cancelled,
}

fn spec(sample_rate: u32) -> WavSpec {
    WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: BITS_PER_SAMPLE,
        sample_format: SampleFormat::Int,
    }
}

/// Writes `audio` as a 24-bit stereo WAV file.
///
/// The data goes to a sibling temporary file that replaces `path` only on success,
/// so a failure or cancellation never leaves a truncated file behind.
pub fn write_wav(
    path: &Path,
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<(), WavError> {
    let mut temporary = OsString::from(path);
    temporary.push(".part");
    let temporary = PathBuf::from(temporary);

    let result = write_temporary(&temporary, path, audio, sample_rate, cancel);
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Writes the temporary file and replaces `path` with it on success.
fn write_temporary(
    temporary: &Path,
    path: &Path,
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<(), WavError> {
    let writer = WavWriter::create(temporary, spec(sample_rate)).map_err(|source| WavError::Write {
        path: path.to_owned(),
        source,
    })?;
    write_samples(writer, audio, cancel).map_err(|source| match source {
        Some(source) => WavError::Write {
            path: path.to_owned(),
            source,
        },
        None => WavError::Cancelled,
    })?;
    fs::rename(temporary, path).map_err(|source| WavError::Replace {
        path: path.to_owned(),
        source,
    })
}

/// Encodes `audio` as a 24-bit stereo WAV in memory.
///
/// The CLI uses this to stream a render to stdout, where there is no file
/// path to replace atomically; everything else should prefer [`write_wav`].
pub fn encode_wav(
    audio: &StereoBuffer,
    sample_rate: u32,
    cancel: &CancelToken,
) -> Result<Vec<u8>, WavError> {
    let mut bytes = Cursor::new(Vec::new());
    let writer = WavWriter::new(&mut bytes, spec(sample_rate)).map_err(|source| WavError::Encode {
        source,
    })?;
    write_samples(writer, audio, cancel).map_err(|source| match source {
        Some(source) => WavError::Encode { source },
        None => WavError::Cancelled,
    })?;
    Ok(bytes.into_inner())
}

/// Writes every sample into `writer` and finalizes it; `Err(None)` means cancelled.
fn write_samples<W: Write + Seek>(
    mut writer: WavWriter<W>,
    audio: &StereoBuffer,
    cancel: &CancelToken,
) -> Result<(), Option<hound::Error>> {
    let frames = audio
        .left
        .chunks(CHUNK_FRAMES)
        .zip(audio.right.chunks(CHUNK_FRAMES));
    for (left, right) in frames {
        if cancel.is_cancelled() {
            return Err(None);
        }
        for (&left, &right) in left.iter().zip(right) {
            writer.write_sample(to_i24(left)).map_err(Some)?;
            writer.write_sample(to_i24(right)).map_err(Some)?;
        }
    }
    writer.finalize().map_err(Some)
}

fn to_i24(sample: f32) -> i32 {
    (f64::from(sample.clamp(-1.0, 1.0)) * FULL_SCALE).round() as i32
}

#[cfg(test)]
mod tests {
    use hound::WavReader;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("opensvr-{}-{name}", std::process::id()))
    }

    #[test]
    fn writes_24_bit_stereo_and_clips() {
        let path = scratch("out.wav");
        let audio = StereoBuffer {
            left: vec![0.5, -2.0],
            right: vec![1.0, 0.0],
        };
        write_wav(&path, &audio, 48_000, &CancelToken::new()).unwrap();

        let mut reader = WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (2, 48_000, 24)
        );
        let samples: Vec<i32> = reader.samples::<i32>().map(Result::unwrap).collect();
        assert_eq!(samples, [4_194_304, 8_388_607, -8_388_607, 0]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn encodes_a_wav_in_memory() {
        let audio = StereoBuffer {
            left: vec![0.5, -1.0],
            right: vec![0.0, 1.0],
        };
        let bytes = encode_wav(&audio, 44_100, &CancelToken::new()).unwrap();

        let mut reader = hound::WavReader::new(Cursor::new(bytes)).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (2, 44_100, 24)
        );
        let samples: Vec<i32> = reader.samples::<i32>().map(Result::unwrap).collect();
        assert_eq!(samples, [4_194_304, 0, -8_388_607, 8_388_607]);
    }

    #[test]
    fn cancellation_leaves_no_file() {
        let path = scratch("cancelled.wav");
        let token = CancelToken::new();
        token.cancel();
        let result = write_wav(&path, &StereoBuffer::silent(10), 48_000, &token);
        assert!(matches!(result, Err(WavError::Cancelled)));
        assert!(!path.exists());
        assert!(!path.with_extension("wav.part").exists());
    }
}
