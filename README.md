# yomiflow

Cross-platform, local-first real-time audio translation engine built in Rust.

`yomiflow` captures system audio, resamples it to Whisper's 16 kHz input, transcribes it locally with `whisper.cpp` via `whisper-rs`, and can translate output to Spanish without calling cloud APIs.

## Current status

- Windows audio capture uses WASAPI loopback and is implemented.
- macOS audio capture is planned via ScreenCaptureKit, but the capture backend is not implemented yet.
- Whisper models are downloaded on demand into `~/.yomiflow/models`.
- EN -> ES translation is supported through a local ONNX model that is also downloaded on demand.

## Features

- Local-first transcription and translation
- System-wide audio capture on Windows
- Automatic resampling to 16 kHz mono
- Whisper transcription with optional translate-to-English mode
- EN -> ES post-translation pipeline for Spanish output

## Requirements

### Runtime

- Rust 1.8x+ toolchain recommended
- Windows 10/11 for the current working capture pipeline
- A local Whisper model download on first run
- Internet only for the initial model downloads

### GPU / inference notes

- CPU fallback works by default
- `--features vulkan` enables Vulkan support for `whisper-rs`
- `--features metal` is intended for macOS builds

## Build

```bash
cargo build
```

Windows with Vulkan:

```bash
cargo build --release --features vulkan
```

macOS with Metal:

```bash
cargo build --release --features metal
```

## Run

Basic transcription:

```bash
cargo run -- --model small
```

Explicit source language:

```bash
cargo run -- --model small --language es
```

Translate captured audio to Spanish subtitles:

```bash
cargo run -- --model small --language en --target es
```

Verbose logs:

```bash
cargo run -- --model small --language en --target es --verbose
```

## CLI options

- `--model <tiny|base|small|medium|large>` selects the Whisper model
- `--language <en|es|ja>` sets the source language; if omitted, Whisper auto-detects
- `--target <lang>` sets the output language; today the pipeline only applies post-translation for `es`
- `--verbose` enables tracing output

Note: `--target` currently requires `--language`.

## Model storage

Downloaded models are cached here:

- Windows: `%USERPROFILE%\.yomiflow\models`
- macOS/Linux-style home layouts: `$HOME/.yomiflow/models`

This folder stores:

- Whisper model files such as `ggml-small.bin`
- The EN -> ES ONNX translation model
- The translation tokenizer

## Environment variables

These are the variables that matter today:

- `USERPROFILE` or `HOME`: used to resolve the local model cache directory. One of them must exist for downloads/cache lookup to work.
- `RUST_LOG`: optional; only used when running with `--verbose`. Example: `RUST_LOG=debug`.
- `ORT_DYLIB_PATH`: may be required when using ONNX Runtime with the `ort` crate's `load-dynamic` feature, especially on Windows if the ONNX Runtime DLL is not already discoverable on `PATH`.
- `VULKAN_SDK`: may be required for Vulkan-enabled builds on Windows depending on your local SDK/toolchain setup.

### Which ones do you actually need?

- For normal local use: usually none beyond a valid home directory environment (`USERPROFILE` on Windows).
- For verbose debugging: optionally set `RUST_LOG`.
- For EN -> ES translation on some Windows setups: set `ORT_DYLIB_PATH` to your `onnxruntime.dll` location if the translator fails to load.
- For Vulkan builds: set `VULKAN_SDK` if your environment does not already expose the Vulkan SDK.

Example on Windows:

```powershell
$env:ORT_DYLIB_PATH = "C:\tmp\onnxruntime-win-x64-1.23.0\lib\onnxruntime.dll"
$env:RUST_LOG = "debug"
cargo run --release --features vulkan -- --model small --language en --target es --verbose
```

## Tests

```bash
cargo test
```

Other useful checks:

```bash
cargo clippy
cargo fmt --check
```

## Known limitations

- macOS capture backend is still a stub
- Spanish output currently relies on an EN -> ES translation stage
- First run downloads models, so startup is slower until the cache is populated
