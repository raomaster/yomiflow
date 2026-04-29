use thiserror::Error;

#[derive(Debug, Error)]
pub enum YomiError {
    #[error("audio capture failed: {0}")]
    AudioCapture(String),

    #[error("resampling failed: {0}")]
    Resample(String),

    #[error("whisper inference failed: {0}")]
    Inference(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, YomiError>;
