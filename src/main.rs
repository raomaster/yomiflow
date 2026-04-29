mod audio;
mod error;
mod inference;
mod pipeline;
mod resample;

use clap::Parser;
use inference::model::ModelSize;
use pipeline::PipelineConfig;

#[derive(Parser, Debug)]
#[command(name = "yomiflow", about = "Real-time audio translation engine")]
struct Cli {
    /// Whisper model size (tiny, base, small, medium, large)
    #[arg(long, default_value = "small")]
    model: String,

    /// Source language (en, es, ja)
    #[arg(long)]
    language: Option<String>,

    /// Target output language (e.g. "es" for Spanish subtitles)
    #[arg(long)]
    target: Option<String>,

    /// Enable verbose logging
    #[arg(long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Only show logs in verbose mode
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
            )
            .init();
    }

    // Translation to non-English targets requires explicit source language
    if cli.target.is_some() && cli.language.is_none() {
        eprintln!("error: --target requires --language (e.g. --language en --target es)");
        std::process::exit(1);
    }

    let model_size = ModelSize::from_str(&cli.model)?;

    eprintln!(
        "yomiflow — model: {}, language: {}, target: {}",
        cli.model,
        cli.language.as_deref().unwrap_or("auto"),
        cli.target.as_deref().unwrap_or("same"),
    );

    let config = PipelineConfig {
        model_size,
        language: cli.language,
        target: cli.target,
    };

    pipeline::run(config, |segment| {
        println!("{}", segment.text);
    })?;

    Ok(())
}
