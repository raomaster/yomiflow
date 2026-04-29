use crate::error::{Result, YomiError};
use ort::{inputs, session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const ENCODER_URL: &str =
    "https://huggingface.co/Xenova/opus-mt-en-es/resolve/main/onnx/encoder_model_quantized.onnx";
const DECODER_URL: &str =
    "https://huggingface.co/Xenova/opus-mt-en-es/resolve/main/onnx/decoder_model_quantized.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/Xenova/opus-mt-en-es/resolve/main/tokenizer.json";

const DECODER_START_TOKEN_ID: i64 = 65000;
const EOS_TOKEN_ID: i64 = 0;
const PAD_TOKEN_ID: i64 = 65000;
const MAX_LENGTH: usize = 256;

/// Opus-MT EN→ES translator using ONNX Runtime.
pub struct Translator {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
}

impl Translator {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let encoder = Session::builder()
            .map_err(|e| YomiError::Inference(format!("ort session builder: {e}")))?
            .commit_from_file(model_dir.join("encoder_model.onnx"))
            .map_err(|e| YomiError::Inference(format!("load encoder: {e}")))?;

        let decoder = Session::builder()
            .map_err(|e| YomiError::Inference(format!("ort session builder: {e}")))?
            .commit_from_file(model_dir.join("decoder_model.onnx"))
            .map_err(|e| YomiError::Inference(format!("load decoder: {e}")))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| YomiError::Inference(format!("load tokenizer: {e}")))?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Translate English text to Spanish.
    pub fn translate(&mut self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| YomiError::Inference(format!("tokenize: {e}")))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_array((vec![1i64, seq_len as i64], input_ids))
            .map_err(|e| YomiError::Inference(format!("input tensor: {e}")))?;
        let attention_mask_tensor =
            Tensor::from_array((vec![1i64, seq_len as i64], attention_mask.clone()))
                .map_err(|e| YomiError::Inference(format!("attention mask tensor: {e}")))?;

        // Run encoder
        let encoder_out = self
            .encoder
            .run(inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e| YomiError::Inference(format!("encoder: {e}")))?;

        // Keep encoder hidden states for decoder input
        let hidden_states_ref = &encoder_out["last_hidden_state"];

        // Greedy decoding
        let mut decoder_ids: Vec<i64> = vec![DECODER_START_TOKEN_ID];

        for _ in 0..MAX_LENGTH {
            let dec_len = decoder_ids.len();
            let decoder_input =
                Tensor::from_array((vec![1i64, dec_len as i64], decoder_ids.clone()))
                    .map_err(|e| YomiError::Inference(format!("decoder input: {e}")))?;
            let attn_mask =
                Tensor::from_array((vec![1i64, seq_len as i64], attention_mask.clone()))
                    .map_err(|e| YomiError::Inference(format!("attn mask: {e}")))?;

            let decoder_out = self
                .decoder
                .run(inputs![
                    "input_ids" => decoder_input,
                    "encoder_attention_mask" => attn_mask,
                    "encoder_hidden_states" => hidden_states_ref,
                ])
                .map_err(|e| YomiError::Inference(format!("decoder: {e}")))?;

            let (logits_shape, logits_data) = decoder_out["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| YomiError::Inference(format!("extract logits: {e}")))?;

            // Argmax on last position, skipping pad token
            let vocab_size = logits_shape[2] as usize;
            let offset = (dec_len - 1) * vocab_size;
            let last_logits = &logits_data[offset..offset + vocab_size];

            let mut best_token = EOS_TOKEN_ID;
            let mut best_score = f32::NEG_INFINITY;
            for (i, &score) in last_logits.iter().enumerate() {
                if i as i64 == PAD_TOKEN_ID {
                    continue;
                }
                if score > best_score {
                    best_score = score;
                    best_token = i as i64;
                }
            }

            if best_token == EOS_TOKEN_ID {
                break;
            }

            decoder_ids.push(best_token);
        }

        // Detokenize (skip start token)
        let output_ids: Vec<u32> = decoder_ids[1..].iter().map(|&id| id as u32).collect();
        let decoded = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| YomiError::Inference(format!("detokenize: {e}")))?;

        Ok(decoded)
    }
}

/// Ensure translation model files are downloaded. Returns path to model directory.
pub fn ensure_translation_model() -> Result<PathBuf> {
    let dir = super::model::models_dir()?.join("opus-mt-en-es");

    if dir.join("encoder_model.onnx").exists()
        && dir.join("decoder_model.onnx").exists()
        && dir.join("tokenizer.json").exists()
    {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir).map_err(|e| YomiError::Model(format!("create dir: {e}")))?;

    eprintln!("downloading translation model (EN→ES, ~76 MB)...");
    download_file(ENCODER_URL, &dir.join("encoder_model.onnx"))?;
    download_file(DECODER_URL, &dir.join("decoder_model.onnx"))?;
    download_file(TOKENIZER_URL, &dir.join("tokenizer.json"))?;
    patch_tokenizer(&dir.join("tokenizer.json"))?;
    eprintln!("translation model ready");

    Ok(dir)
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let fname = dest
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    eprintln!("  downloading {fname}...");

    let response = reqwest::blocking::get(url)
        .map_err(|e| YomiError::Model(format!("download {fname}: {e}")))?;

    if !response.status().is_success() {
        return Err(YomiError::Model(format!(
            "download {fname}: HTTP {}",
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|e| YomiError::Model(format!("read {fname}: {e}")))?;

    let tmp = dest.with_extension("part");
    std::fs::write(&tmp, &bytes)
        .map_err(|e| YomiError::Model(format!("write {fname}: {e}")))?;
    std::fs::rename(&tmp, dest).map_err(|e| YomiError::Model(format!("rename {fname}: {e}")))?;

    eprintln!(
        "  {} ({:.1} MB)",
        fname,
        bytes.len() as f64 / 1_000_000.0
    );
    Ok(())
}

/// Remove the unsupported "Precompiled" normalizer from tokenizer.json.
/// The Precompiled normalizer with null charsmap is a no-op anyway.
fn patch_tokenizer(path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| YomiError::Model(format!("read tokenizer: {e}")))?;
    let patched = content.replace(
        r#""normalizer":{"type":"Precompiled","precompiled_charsmap":null}"#,
        r#""normalizer":null"#,
    );
    // Also try pretty-printed format
    let patched = if patched == content {
        // Try with whitespace variations
        let mut val: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| YomiError::Model(format!("parse tokenizer json: {e}")))?;
        if let Some(obj) = val.as_object_mut() {
            obj.insert("normalizer".to_string(), serde_json::Value::Null);
        }
        serde_json::to_string_pretty(&val)
            .map_err(|e| YomiError::Model(format!("serialize tokenizer json: {e}")))?
    } else {
        patched
    };
    std::fs::write(path, patched)
        .map_err(|e| YomiError::Model(format!("write tokenizer: {e}")))?;
    Ok(())
}
