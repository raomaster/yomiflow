use crate::error::{Result, YomiError};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tracing::{debug, info};

/// Supported Whisper model sizes.
#[derive(Debug, Clone, Copy)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl ModelSize {
    /// Parse from CLI string.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "base" => Ok(Self::Base),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err(YomiError::Model(format!(
                "unknown model size '{s}' — use tiny, base, small, medium, or large"
            ))),
        }
    }

    /// Filename for the ggml model.
    fn filename(self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
            Self::Medium => "ggml-medium.bin",
            Self::Large => "ggml-large-v3.bin",
        }
    }

    /// Download URL (Hugging Face ggerganov/whisper.cpp).
    fn url(self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.filename()
        )
    }

    /// Approximate download size for progress display.
    fn approx_size_mb(self) -> u64 {
        match self {
            Self::Tiny => 75,
            Self::Base => 142,
            Self::Small => 466,
            Self::Medium => 1533,
            Self::Large => 3095,
        }
    }
}

/// Directory where models are cached.
pub(crate) fn models_dir() -> Result<PathBuf> {
    let home = dirs_next().ok_or_else(|| YomiError::Model("cannot find home directory".into()))?;
    let dir = home.join(".yomiflow").join("models");
    Ok(dir)
}

/// Cross-platform home directory.
fn dirs_next() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Ensure a Whisper model is available locally. Downloads if missing.
/// Returns the path to the model file.
pub fn ensure_model(size: ModelSize) -> Result<PathBuf> {
    let dir = models_dir()?;
    let path = dir.join(size.filename());

    if path.exists() {
        info!(model = %size.filename(), "model already cached");
        return Ok(path);
    }

    fs::create_dir_all(&dir).map_err(|e| YomiError::Model(format!("create cache dir: {e}")))?;

    let url = size.url();
    info!(
        model = %size.filename(),
        url = %url,
        "downloading model (~{} MB)",
        size.approx_size_mb()
    );

    download_model(&url, &path, size.approx_size_mb())?;

    info!(path = %path.display(), "model downloaded");
    Ok(path)
}

/// Download a file with progress bar.
fn download_model(url: &str, dest: &PathBuf, approx_mb: u64) -> Result<()> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| YomiError::Model(format!("download request failed: {e}")))?;

    if !response.status().is_success() {
        return Err(YomiError::Model(format!(
            "download failed: HTTP {}",
            response.status()
        )));
    }

    let total = response.content_length().unwrap_or(approx_mb * 1_000_000);

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    // Write to a temp file first, then rename (atomic-ish)
    let tmp_path = dest.with_extension("bin.part");
    let mut file =
        fs::File::create(&tmp_path).map_err(|e| YomiError::Model(format!("create file: {e}")))?;

    let bytes = response
        .bytes()
        .map_err(|e| YomiError::Model(format!("read response: {e}")))?;

    pb.set_position(bytes.len() as u64);
    file.write_all(&bytes)
        .map_err(|e| YomiError::Model(format!("write file: {e}")))?;

    file.flush()
        .map_err(|e| YomiError::Model(format!("flush file: {e}")))?;
    drop(file);

    fs::rename(&tmp_path, dest).map_err(|e| YomiError::Model(format!("rename file: {e}")))?;

    pb.finish_with_message("done");
    debug!(path = %dest.display(), bytes = bytes.len(), "model saved");
    Ok(())
}
