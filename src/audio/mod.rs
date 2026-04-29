#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "macos")]
pub mod macos;

/// Trait for platform-specific audio capture sources.
pub trait AudioCaptureSource {
    /// Start capturing system audio. Sends f32 PCM samples (mono or stereo)
    /// at the system's native sample rate through the provided sender.
    fn start(&mut self, sender: crossbeam_channel::Sender<Vec<f32>>) -> crate::error::Result<()>;

    /// Stop capturing audio.
    fn stop(&mut self) -> crate::error::Result<()>;

    /// Returns the sample rate of the captured audio.
    fn sample_rate(&self) -> u32;

    /// Returns the number of channels in the captured audio.
    fn channels(&self) -> u16;
}

/// Create the platform-appropriate audio capture source.
#[cfg(target_os = "windows")]
pub fn create_capture() -> crate::error::Result<windows::WasapiCapture> {
    windows::WasapiCapture::new()
}

#[cfg(target_os = "macos")]
pub fn create_capture() -> crate::error::Result<macos::ScreenCaptureAudio> {
    macos::ScreenCaptureAudio::new()
}
