use std::path::PathBuf;
use std::time::Instant;

fn translation_model_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap();
    PathBuf::from(home)
        .join(".yomiflow")
        .join("models")
        .join("opus-mt-en-es")
}

#[test]
fn test_translation_download_and_run() {
    // Download model if needed
    let dir = yomiflow::inference::translator::ensure_translation_model().unwrap();
    println!("Model dir: {}", dir.display());

    // Load translator
    let mut translator = yomiflow::inference::translator::Translator::new(&dir).unwrap();
    println!("Translator loaded");

    // Test translations
    let tests = [
        "Hello, how are you?",
        "I sleep on a bed of nails",
        "In the past, another example, nobody will remember but years ago",
        "The quick brown fox jumps over the lazy dog",
    ];

    for text in &tests {
        let start = Instant::now();
        let result = translator.translate(text).unwrap();
        let elapsed = start.elapsed();
        println!("  EN: {}", text);
        println!("  ES: {}", result);
        println!("  Time: {:.0}ms", elapsed.as_secs_f64() * 1000.0);
        println!();
    }
}
