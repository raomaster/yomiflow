use crate::error::{Result, YomiError};
use std::path::Path;
use tracing::info;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Result of a single transcription segment.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in milliseconds.
    pub start_ms: i64,
    /// End time in milliseconds.
    pub end_ms: i64,
    /// Transcribed/translated text.
    pub text: String,
}

/// Suppress all whisper.cpp and ggml internal logging to stderr.
pub fn suppress_whisper_logs() {
    unsafe extern "C" fn noop(
        _level: whisper_rs_sys::ggml_log_level,
        _text: *const std::os::raw::c_char,
        _user_data: *mut std::os::raw::c_void,
    ) {
    }
    unsafe {
        whisper_rs_sys::whisper_log_set(Some(noop), std::ptr::null_mut());
        whisper_rs_sys::ggml_log_set(Some(noop), std::ptr::null_mut());
    }
}

/// Wrapper around whisper-rs for real-time transcription.
pub struct WhisperEngine {
    ctx: WhisperContext,
    state: WhisperState,
    /// Source language (e.g. "es", "en", "ja"). None = auto-detect.
    language: Option<String>,
    /// If true, translate to English.
    translate: bool,
}

impl WhisperEngine {
    /// Load a Whisper model and create the engine.
    pub fn new(model_path: &Path, language: Option<&str>, translate: bool) -> Result<Self> {
        suppress_whisper_logs();

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);

        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| {
                YomiError::Inference("model path contains invalid UTF-8".into())
            })?,
            ctx_params,
        )
        .map_err(|e| YomiError::Inference(format!("load model: {e}")))?;

        let state = ctx
            .create_state()
            .map_err(|e| YomiError::Inference(format!("create state: {e}")))?;

        info!("whisper model loaded");

        Ok(Self {
            ctx,
            state,
            language: language.map(String::from),
            translate,
        })
    }

    /// Transcribe a chunk of 16kHz mono f32 audio.
    /// Returns the recognized segments (text + timestamps).
    pub fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<Segment>> {
        self.state = self
            .ctx
            .create_state()
            .map_err(|e| YomiError::Inference(format!("create state: {e}")))?;

        // Beam search gives much better accuracy than greedy
        let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: 1.0 });

        params.set_language(self.language.as_deref());
        params.set_translate(self.translate);

        params.set_n_threads(4);
        params.set_single_segment(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_suppress_blank(true);
        params.set_no_context(true);

        // Temperature fallback: start at 0 (deterministic), raise if decoding fails
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);

        self.state
            .full(params, audio_16k_mono)
            .map_err(|e| YomiError::Inference(format!("inference: {e}")))?;

        let mut segments = Vec::new();
        for seg in self.state.as_iter() {
            let text = seg
                .to_str_lossy()
                .map_err(|e| YomiError::Inference(format!("decode text: {e}")))?
                .trim()
                .to_string();

            if text.is_empty() {
                continue;
            }

            segments.push(Segment {
                start_ms: seg.start_timestamp() * 10,
                end_ms: seg.end_timestamp() * 10,
                text,
            });
        }

        Ok(segments)
    }
}
