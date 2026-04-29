use crate::error::{Result, YomiError};
use super::AudioCaptureSource;
use crossbeam_channel::Sender;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

pub struct WasapiCapture {
    sample_rate: u32,
    channels: u16,
    running: Arc<AtomicBool>,
    capture_thread: Option<std::thread::JoinHandle<()>>,
}

impl WasapiCapture {
    pub fn new() -> Result<Self> {
        // Initialize COM for this thread
        wasapi::initialize_mta()
            .ok()
            .map_err(|e| YomiError::AudioCapture(format!("COM init failed: {e}")))?;

        // Probe the default render device to get its format
        let enumerator = wasapi::DeviceEnumerator::new()
            .map_err(|e| YomiError::AudioCapture(format!("device enumerator: {e}")))?;
        let device = enumerator
            .get_default_device(&wasapi::Direction::Render)
            .map_err(|e| YomiError::AudioCapture(format!("no default render device: {e}")))?;

        let name = device.get_friendlyname().unwrap_or_default();
        info!(device = %name, "WASAPI loopback device selected");

        Ok(Self {
            sample_rate: 48000,
            channels: 2,
            running: Arc::new(AtomicBool::new(false)),
            capture_thread: None,
        })
    }
}

impl AudioCaptureSource for WasapiCapture {
    fn start(&mut self, sender: Sender<Vec<f32>>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(YomiError::AudioCapture("already running".into()));
        }

        self.running.store(true, Ordering::SeqCst);
        let running = Arc::clone(&self.running);
        let sample_rate = self.sample_rate as usize;
        let channels = self.channels as usize;

        let handle = std::thread::Builder::new()
            .name("wasapi-capture".into())
            .spawn(move || {
                if let Err(e) = capture_loop(running.clone(), sender, sample_rate, channels) {
                    warn!(error = %e, "capture loop exited with error");
                    running.store(false, Ordering::SeqCst);
                }
            })
            .map_err(|e| YomiError::AudioCapture(format!("spawn thread: {e}")))?;

        self.capture_thread = Some(handle);
        info!("WASAPI loopback capture started");
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.capture_thread.take() {
            handle
                .join()
                .map_err(|_| YomiError::AudioCapture("capture thread panicked".into()))?;
        }
        info!("WASAPI loopback capture stopped");
        Ok(())
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u16 {
        self.channels
    }
}

impl Drop for WasapiCapture {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Runs the WASAPI loopback capture on a dedicated thread.
fn capture_loop(
    running: Arc<AtomicBool>,
    sender: Sender<Vec<f32>>,
    sample_rate: usize,
    channels: usize,
) -> Result<()> {
    // COM must be initialized per-thread
    wasapi::initialize_mta()
        .ok()
        .map_err(|e| YomiError::AudioCapture(format!("COM init (capture thread): {e}")))?;

    let enumerator = wasapi::DeviceEnumerator::new()
        .map_err(|e| YomiError::AudioCapture(format!("device enumerator: {e}")))?;
    let device = enumerator
        .get_default_device(&wasapi::Direction::Render)
        .map_err(|e| YomiError::AudioCapture(format!("no default render device: {e}")))?;

    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|e| YomiError::AudioCapture(format!("get audio client: {e}")))?;

    let desired_format = wasapi::WaveFormat::new(
        32,
        32,
        &wasapi::SampleType::Float,
        sample_rate,
        channels,
        None,
    );
    let blockalign = desired_format.get_blockalign() as usize;

    let (_, min_time) = audio_client
        .get_device_period()
        .map_err(|e| YomiError::AudioCapture(format!("get device period: {e}")))?;

    // Direction::Capture on a Render device activates AUDCLNT_STREAMFLAGS_LOOPBACK
    audio_client
        .initialize_client(
            &desired_format,
            &wasapi::Direction::Capture,
            &wasapi::StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: min_time,
            },
        )
        .map_err(|e| YomiError::AudioCapture(format!("initialize client: {e}")))?;

    let h_event = audio_client
        .set_get_eventhandle()
        .map_err(|e| YomiError::AudioCapture(format!("set event handle: {e}")))?;

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| YomiError::AudioCapture(format!("get capture client: {e}")))?;

    audio_client
        .start_stream()
        .map_err(|e| YomiError::AudioCapture(format!("start stream: {e}")))?;

    debug!(
        sample_rate,
        channels,
        blockalign,
        "WASAPI capture loop running"
    );

    let mut byte_queue: VecDeque<u8> = VecDeque::new();
    // Send chunks of ~100ms worth of samples
    let chunk_frames = sample_rate / 10;
    let chunk_bytes = chunk_frames * blockalign;

    while running.load(Ordering::SeqCst) {
        // Wait for audio event (timeout 200ms so we can check `running`)
        if h_event.wait_for_event(200_000).is_err() {
            continue;
        }

        // Drain all available packets
        loop {
            let next = capture_client
                .get_next_packet_size()
                .map_err(|e| YomiError::AudioCapture(format!("get packet size: {e}")))?;

            match next {
                Some(0) | None => break,
                Some(_) => {
                    capture_client
                        .read_from_device_to_deque(&mut byte_queue)
                        .map_err(|e| YomiError::AudioCapture(format!("read device: {e}")))?;
                }
            }
        }

        // Send complete chunks
        while byte_queue.len() >= chunk_bytes {
            let raw: Vec<u8> = byte_queue.drain(..chunk_bytes).collect();
            let samples = bytes_to_f32(&raw);

            if sender.send(samples).is_err() {
                // Receiver dropped — stop capture
                debug!("channel receiver dropped, stopping capture");
                running.store(false, Ordering::SeqCst);
                break;
            }
        }
    }

    audio_client
        .stop_stream()
        .map_err(|e| YomiError::AudioCapture(format!("stop stream: {e}")))?;

    info!("WASAPI capture loop finished");
    Ok(())
}

/// Convert little-endian f32 bytes to Vec<f32>.
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
