use crate::audio::{self, AudioCaptureSource};
use crate::error::{Result, YomiError};
use crate::inference::model::{self, ModelSize};
use crate::inference::translator::{self, Translator};
use crate::inference::whisper::{Segment, WhisperEngine};
use crate::resample::AudioResampler;
use crossbeam_channel::{bounded, Receiver, Sender};
use tracing::{debug, error};

/// Configuration for the pipeline.
pub struct PipelineConfig {
    pub model_size: ModelSize,
    /// Source language (e.g. "es", "en", "ja"). None = auto-detect.
    pub language: Option<String>,
    /// Target output language (e.g. "es"). None = same as source.
    pub target: Option<String>,
}

/// Runs the capture → resample → inference pipeline.
/// Blocks until Ctrl+C or an error occurs.
/// Calls `on_segment` for each transcribed segment.
pub fn run(config: PipelineConfig, on_segment: impl Fn(&Segment) + Send + 'static) -> Result<()> {
    eprintln!("loading whisper model...");
    let model_path = model::ensure_model(config.model_size)?;

    // Determine if we need EN→ES translation
    let source_lang = config.language.as_deref();
    let target_lang = config.target.as_deref();
    let need_translation = match target_lang {
        Some("es") => source_lang != Some("es"),
        _ => false,
    };

    // If target is ES and source is not EN, Whisper must translate to EN first
    let whisper_translate = need_translation && source_lang != Some("en");

    // Load translation model if needed
    let translator = if need_translation {
        let tl_dir = translator::ensure_translation_model()?;
        Some(Translator::new(&tl_dir)?)
    } else {
        None
    };

    let mut capture = audio::create_capture()?;
    let sample_rate = capture.sample_rate();
    let channels = capture.channels();

    let (cap_tx, cap_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(64);
    let (inf_tx, inf_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(16);

    capture.start(cap_tx)?;

    let resample_handle = std::thread::Builder::new()
        .name("resample".into())
        .spawn(move || {
            if let Err(e) = resample_loop(sample_rate, channels, cap_rx, inf_tx) {
                error!(error = %e, "resample thread failed");
            }
        })
        .map_err(|e| YomiError::Pipeline(format!("spawn resample thread: {e}")))?;

    let language = config.language.clone();
    let inference_handle = std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            if let Err(e) = inference_loop(
                &model_path,
                language.as_deref(),
                whisper_translate,
                translator,
                inf_rx,
                on_segment,
            ) {
                error!(error = %e, "inference thread failed");
            }
        })
        .map_err(|e| YomiError::Pipeline(format!("spawn inference thread: {e}")))?;

    eprintln!("listening... (Ctrl+C to stop)");
    let (stop_tx, stop_rx) = bounded::<()>(1);
    ctrlc_handler(stop_tx);
    let _ = stop_rx.recv();

    capture.stop()?;
    drop(capture);
    let _ = resample_handle.join();
    let _ = inference_handle.join();

    Ok(())
}

/// Resample loop: reads capture chunks, resamples to 16kHz mono, sends to inference.
fn resample_loop(
    sample_rate: u32,
    channels: u16,
    cap_rx: Receiver<Vec<f32>>,
    inf_tx: Sender<Vec<f32>>,
) -> Result<()> {
    let mut resampler = AudioResampler::new(sample_rate, 16000, channels)?;
    let needed = resampler.input_samples_needed();
    let mut buffer: Vec<f32> = Vec::with_capacity(needed * 2);

    let whisper_chunk_samples = 16000 * 5; // 5 seconds — longer chunks = better context for Whisper
    let mut inference_buffer: Vec<f32> = Vec::with_capacity(whisper_chunk_samples);

    for chunk in cap_rx {
        buffer.extend_from_slice(&chunk);

        while buffer.len() >= needed {
            let input: Vec<f32> = buffer.drain(..needed).collect();
            let resampled = resampler.process(&input)?;
            inference_buffer.extend_from_slice(&resampled);
        }

        if inference_buffer.len() >= whisper_chunk_samples {
            let to_send: Vec<f32> = inference_buffer.drain(..).collect();
            if inf_tx.send(to_send).is_err() {
                break;
            }
        }
    }

    // Flush remaining audio
    if !buffer.is_empty() {
        if let Ok(resampled) = resampler.process_partial(&buffer) {
            inference_buffer.extend_from_slice(&resampled);
        }
    }
    if let Ok(flushed) = resampler.flush() {
        inference_buffer.extend_from_slice(&flushed);
    }
    if !inference_buffer.is_empty() {
        let _ = inf_tx.send(inference_buffer);
    }

    debug!("resample loop finished");
    Ok(())
}

/// Inference loop: receives 16kHz mono audio chunks, runs Whisper, optionally translates.
fn inference_loop(
    model_path: &std::path::Path,
    language: Option<&str>,
    translate: bool,
    mut translator: Option<Translator>,
    inf_rx: Receiver<Vec<f32>>,
    on_segment: impl Fn(&Segment),
) -> Result<()> {
    let mut engine = WhisperEngine::new(model_path, language, translate)?;
    eprintln!("model ready");

    for audio_chunk in inf_rx {
        let segments = engine.transcribe(&audio_chunk)?;
        for seg in &segments {
            if let Some(ref mut tl) = translator {
                match tl.translate(&seg.text) {
                    Ok(translated) => {
                        let translated_seg = Segment {
                            start_ms: seg.start_ms,
                            end_ms: seg.end_ms,
                            text: translated,
                        };
                        on_segment(&translated_seg);
                    }
                    Err(e) => {
                        error!(error = %e, "translation failed, using original");
                        on_segment(seg);
                    }
                }
            } else {
                on_segment(seg);
            }
        }
    }

    debug!("inference loop finished");
    Ok(())
}

/// Register a Ctrl+C handler that sends a signal on the channel.
fn ctrlc_handler(stop_tx: Sender<()>) {
    let _ = ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    });
}
