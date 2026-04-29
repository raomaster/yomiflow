use crate::error::{Result, YomiError};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tracing::debug;

/// Resamples stereo interleaved f32 audio (e.g. 48kHz) down to 16kHz mono
/// for Whisper input.
pub struct AudioResampler {
    resampler: SincFixedIn<f32>,
    input_channels: usize,
    /// Pre-allocated output buffer (channels x frames)
    output_buf: Vec<Vec<f32>>,
}

impl AudioResampler {
    /// Create a new resampler.
    ///
    /// - `input_rate`: source sample rate (e.g. 48000)
    /// - `output_rate`: target sample rate (must be 16000 for Whisper)
    /// - `channels`: number of input channels (1 or 2)
    pub fn new(input_rate: u32, output_rate: u32, channels: u16) -> Result<Self> {
        let ratio = output_rate as f64 / input_rate as f64;
        let channels = channels as usize;

        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::Blackman2,
        };

        // Fixed input chunk size — ~21ms at 48kHz
        let chunk_size = 1024;

        let resampler = SincFixedIn::<f32>::new(ratio, 1.1, params, chunk_size, channels)
            .map_err(|e| YomiError::Resample(format!("init resampler: {e}")))?;

        let output_buf = resampler.output_buffer_allocate(true);

        debug!(
            input_rate,
            output_rate,
            channels,
            chunk_size,
            ratio,
            "resampler created"
        );

        Ok(Self {
            resampler,
            input_channels: channels,
            output_buf,
        })
    }

    /// Number of interleaved input samples needed for the next `process` call.
    /// This is `input_frames_next() * channels`.
    pub fn input_samples_needed(&self) -> usize {
        self.resampler.input_frames_next() * self.input_channels
    }

    /// Process a chunk of interleaved f32 samples and return 16kHz mono output.
    ///
    /// `interleaved` must contain exactly `input_samples_needed()` samples.
    pub fn process(&mut self, interleaved: &[f32]) -> Result<Vec<f32>> {
        let frames = interleaved.len() / self.input_channels;
        let deinterleaved = deinterleave(interleaved, self.input_channels, frames);

        let (_, out_frames) = self
            .resampler
            .process_into_buffer(&deinterleaved, &mut self.output_buf, None)
            .map_err(|e| YomiError::Resample(format!("process: {e}")))?;

        Ok(to_mono(&self.output_buf, out_frames))
    }

    /// Process remaining samples at end-of-stream (may be less than a full chunk).
    pub fn process_partial(&mut self, interleaved: &[f32]) -> Result<Vec<f32>> {
        let frames = interleaved.len() / self.input_channels;
        let deinterleaved = deinterleave(interleaved, self.input_channels, frames);

        let (_, out_frames) = self
            .resampler
            .process_partial_into_buffer(Some(&deinterleaved), &mut self.output_buf, None)
            .map_err(|e| YomiError::Resample(format!("process partial: {e}")))?;

        Ok(to_mono(&self.output_buf, out_frames))
    }

    /// Flush any remaining samples from internal buffers.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let (_, out_frames) = self
            .resampler
            .process_partial_into_buffer(None::<&[Vec<f32>]>, &mut self.output_buf, None)
            .map_err(|e| YomiError::Resample(format!("flush: {e}")))?;

        Ok(to_mono(&self.output_buf, out_frames))
    }
}

/// Split interleaved samples into per-channel vectors.
fn deinterleave(interleaved: &[f32], channels: usize, frames: usize) -> Vec<Vec<f32>> {
    let mut out = vec![Vec::with_capacity(frames); channels];
    for frame in interleaved.chunks_exact(channels) {
        for (ch, sample) in frame.iter().enumerate() {
            out[ch].push(*sample);
        }
    }
    out
}

/// Average all channels down to mono.
fn to_mono(channel_bufs: &[Vec<f32>], frames: usize) -> Vec<f32> {
    if channel_bufs.len() == 1 {
        return channel_bufs[0][..frames].to_vec();
    }
    (0..frames)
        .map(|i| {
            let sum: f32 = channel_bufs.iter().map(|ch| ch[i]).sum();
            sum / channel_bufs.len() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_sine_wave() {
        let input_rate = 48000u32;
        let output_rate = 16000u32;
        let channels = 2u16;

        let mut resampler = AudioResampler::new(input_rate, output_rate, channels).unwrap();
        let needed = resampler.input_samples_needed();

        // Generate a 440Hz sine wave, stereo interleaved
        let mut input = Vec::with_capacity(needed);
        for i in 0..needed / 2 {
            let t = i as f32 / input_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            input.push(sample); // left
            input.push(sample); // right (same for test)
        }

        let output = resampler.process(&input).unwrap();

        // Output should be roughly 1/3 of input frames (48k → 16k).
        // Sinc resampler has an initial delay so allow generous tolerance.
        let expected_frames = needed / 2 / 3;
        assert!(
            output.len() > expected_frames / 2 && output.len() <= expected_frames + 20,
            "expected ~{} output frames, got {}",
            expected_frames,
            output.len()
        );

        // Samples should be in valid range
        assert!(output.iter().all(|s| *s >= -1.5 && *s <= 1.5));
    }
}
