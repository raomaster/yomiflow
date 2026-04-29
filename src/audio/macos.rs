use crate::error::{Result, YomiError};
use super::AudioCaptureSource;
use crossbeam_channel::Sender;

pub struct ScreenCaptureAudio {
    sample_rate: u32,
    channels: u16,
}

impl ScreenCaptureAudio {
    pub fn new() -> Result<Self> {
        // TODO: initialize ScreenCaptureKit
        Ok(Self {
            sample_rate: 48000,
            channels: 2,
        })
    }
}

impl AudioCaptureSource for ScreenCaptureAudio {
    fn start(&mut self, _sender: Sender<Vec<f32>>) -> Result<()> {
        // TODO: implement ScreenCaptureKit audio capture
        Err(YomiError::AudioCapture("not yet implemented".into()))
    }

    fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}
