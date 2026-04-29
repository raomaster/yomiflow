//! Benchmark: measures real latency of each pipeline stage.
//! Run with: cargo test --release bench_ -- --nocapture

use std::path::PathBuf;
use std::time::Instant;

fn model_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap();
    PathBuf::from(home)
        .join(".yomiflow")
        .join("models")
        .join("ggml-base.bin")
}

fn read_debug_wav() -> Option<Vec<f32>> {
    let path = std::path::Path::new("debug_capture.wav");
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read(path).ok()?;
    let pcm = &raw[44..];
    Some(
        pcm.chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
    )
}

/// Generate 5 seconds of 48kHz stereo silence with a tone burst.
fn generate_test_audio(duration_secs: f32) -> Vec<f32> {
    let rate = 48000;
    let channels = 2;
    let total_samples = (rate as f32 * duration_secs) as usize * channels;
    let mut audio = Vec::with_capacity(total_samples);
    for i in 0..total_samples / channels {
        let t = i as f32 / rate as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
        audio.push(sample); // left
        audio.push(sample); // right
    }
    audio
}

#[test]
fn bench_resample() {
    let audio = generate_test_audio(5.0);
    let mut resampler = yomiflow::resample::AudioResampler::new(48000, 16000, 2).unwrap();
    let needed = resampler.input_samples_needed();

    let start = Instant::now();
    let mut total_out = 0usize;
    let mut pos = 0;
    while pos + needed <= audio.len() {
        let chunk = &audio[pos..pos + needed];
        let out = resampler.process(chunk).unwrap();
        total_out += out.len();
        pos += needed;
    }
    let elapsed = start.elapsed();

    println!("=== RESAMPLE BENCHMARK ===");
    println!("  Input:  {} samples (5s stereo 48kHz)", audio.len());
    println!("  Output: {} samples (16kHz mono)", total_out);
    println!("  Time:   {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Speed:  {:.0}x realtime", 5.0 / elapsed.as_secs_f64());
}

#[test]
fn bench_whisper_inference() {
    let mp = model_path();
    if !mp.exists() {
        println!("SKIP: model not found at {}", mp.display());
        return;
    }

    // Suppress whisper logs
    yomiflow::inference::whisper::suppress_whisper_logs();

    // Use debug WAV if available, otherwise generate audio
    let audio_16k = match read_debug_wav() {
        Some(a) => {
            println!("Using debug_capture.wav ({} samples)", a.len());
            a
        }
        None => {
            println!("No debug_capture.wav, generating sine wave");
            // Generate and resample
            let raw = generate_test_audio(5.0);
            let mut resampler = yomiflow::resample::AudioResampler::new(48000, 16000, 2).unwrap();
            let needed = resampler.input_samples_needed();
            let mut mono = Vec::new();
            let mut pos = 0;
            while pos + needed <= raw.len() {
                mono.extend_from_slice(&resampler.process(&raw[pos..pos + needed]).unwrap());
                pos += needed;
            }
            mono
        }
    };

    let duration_s = audio_16k.len() as f64 / 16000.0;

    // Benchmark: transcribe (Spanish, no translate)
    let mut engine =
        yomiflow::inference::whisper::WhisperEngine::new(&mp, Some("es"), false).unwrap();

    let start = Instant::now();
    let segments = engine.transcribe(&audio_16k).unwrap();
    let elapsed = start.elapsed();

    println!("=== WHISPER INFERENCE (transcribe) ===");
    println!("  Audio:    {:.1}s ({} samples)", duration_s, audio_16k.len());
    println!("  Time:     {:.0}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Speed:    {:.1}x realtime", duration_s / elapsed.as_secs_f64());
    println!("  Segments: {}", segments.len());
    for seg in &segments {
        println!("    [{}ms-{}ms] {}", seg.start_ms, seg.end_ms, seg.text);
    }

    // Benchmark: translate (Spanish → English)
    let mut engine_translate =
        yomiflow::inference::whisper::WhisperEngine::new(&mp, Some("es"), true).unwrap();

    let start = Instant::now();
    let segments = engine_translate.transcribe(&audio_16k).unwrap();
    let elapsed = start.elapsed();

    println!("=== WHISPER INFERENCE (translate ES→EN) ===");
    println!("  Audio:    {:.1}s ({} samples)", duration_s, audio_16k.len());
    println!("  Time:     {:.0}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Speed:    {:.1}x realtime", duration_s / elapsed.as_secs_f64());
    println!("  Segments: {}", segments.len());
    for seg in &segments {
        println!("    [{}ms-{}ms] {}", seg.start_ms, seg.end_ms, seg.text);
    }
}

#[test]
fn bench_full_pipeline_latency() {
    let mp = model_path();
    if !mp.exists() {
        println!("SKIP: model not found");
        return;
    }

    yomiflow::inference::whisper::suppress_whisper_logs();

    // Simulate full pipeline: capture → resample → inference
    let pipeline_start = Instant::now();

    // Stage 1: Resample (skip if using debug WAV — already 16kHz mono)
    let resample_start = Instant::now();
    let mono = if let Some(wav) = read_debug_wav() {
        // debug_capture.wav is already 16kHz mono, take ~3s worth
        let samples_3s = (16000 * 3).min(wav.len());
        wav[..samples_3s].to_vec()
    } else {
        let raw_audio = generate_test_audio(3.0);
        let mut resampler = yomiflow::resample::AudioResampler::new(48000, 16000, 2).unwrap();
        let needed = resampler.input_samples_needed();
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + needed <= raw_audio.len() {
            out.extend_from_slice(&resampler.process(&raw_audio[pos..pos + needed]).unwrap());
            pos += needed;
        }
        out
    };
    let resample_time = resample_start.elapsed();

    // Stage 2: Inference
    let inference_start = Instant::now();
    let mut engine =
        yomiflow::inference::whisper::WhisperEngine::new(&mp, Some("es"), true).unwrap();
    let segments = engine.transcribe(&mono).unwrap();
    let inference_time = inference_start.elapsed();

    let total_time = pipeline_start.elapsed();

    println!("=== FULL PIPELINE LATENCY (3s chunk) ===");
    println!("  Resample:  {:.0}ms", resample_time.as_secs_f64() * 1000.0);
    println!("  Inference: {:.0}ms", inference_time.as_secs_f64() * 1000.0);
    println!("  Total:     {:.0}ms", total_time.as_secs_f64() * 1000.0);
    println!("  Segments:  {}", segments.len());
    for seg in &segments {
        println!("    {}", seg.text);
    }
}
