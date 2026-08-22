// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Token-counting calibration against the REAL API.
//!
//! Compares the library's heuristic `count_messages_tokens` with the
//! provider's ground truth: `usage.prompt_tokens` returned by a real
//! `chat/completions` call. Prints per-sample estimate/actual/ratio and a
//! summary so the heuristic's accuracy can be measured (and re-measured).
//!
//! Credentials come from the environment:
//!   LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME
//!
//! Run:
//!   export LLM_API_KEY=... LLM_API_URL=... LLM_MODEL_NAME=...
//!   cargo run --example token_calibration

use std::env;
use std::process::exit;

use openai_client_rs::*;

fn env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("missing required env var: {name}");
        exit(2);
    })
}

const SYSTEM: &str = "You are a terse assistant.";

const SAMPLES: &[&str] = &[
    "",
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "fn main() { println!(\"hello world\"); }",
    "fn fib(n: u64) -> u64 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } }",
    "你好，世界！这是一个中文测试。",
    "日本語のテキストでトークン数を確認します。",
    "Mixed 中文 and English text with 12345 numbers and code()!",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
    "1234567890123456789012345678901234567890",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
];

fn main() {
    let key = env("LLM_API_KEY");
    let base = env("LLM_API_URL");
    let model = env("LLM_MODEL_NAME");

    println!("== openai-rs token calibration ==");
    println!("  endpoint: {base}");
    println!("  model:    {model}");
    println!("  heuristic: char-based (ASCII=1pt, other=3pt) / divisor");
    println!();
    println!("{:<30} {:>8} {:>8} {:>8}", "sample", "estimate", "actual", "ratio");

    let client = OpenAiClient::with_base_url(&key, &model, &base).with_total_timeout(60);
    let mut ratios: Vec<f64> = Vec::new();
    let mut diffs: Vec<f64> = Vec::new();

    for s in SAMPLES {
        let msgs = vec![
            ChatMessage::system(SYSTEM),
            ChatMessage::user((*s).to_string()),
        ];
        let estimate = openai_client_rs::tokens::count_messages_tokens(&msgs);
        let actual = match client.chat_create_raw(&msgs, None) {
            Ok(cc) => cc.usage.map(|u| u.prompt_tokens as i64).unwrap_or(-1),
            Err(e) => {
                eprintln!("  request failed for sample {s:?}: {e}");
                exit(1);
            }
        };
        let ratio = if actual > 0 {
            estimate as f64 / actual as f64
        } else {
            f64::NAN
        };
        ratios.push(ratio);
        if actual > 0 {
            diffs.push((actual - estimate as i64) as f64);
        }
        let label = if s.chars().count() > 28 {
            let t: String = s.chars().take(26).collect();
            format!("{t}…")
        } else {
            s.to_string()
        };
        println!(
            "{label:<30} {estimate:>8} {actual:>8} {ratio:>8.2}",
        );
    }

    let valid: Vec<f64> = ratios.iter().copied().filter(|r| r.is_finite() && *r > 0.0).collect();
    let n = valid.len() as f64;
    let mean_ratio: f64 = valid.iter().sum::<f64>() / n;
    let mean_abs_err: f64 =
        valid.iter().map(|r| (r - 1.0).abs()).sum::<f64>() / n;
    let max_abs_err: f64 = valid.iter().map(|r| (r - 1.0).abs()).fold(0.0, f64::max);

    // Median of (actual − estimate) ≈ the provider's fixed chat-template
    // overhead the heuristic cannot model. This is the value to pass to
    // `count_messages_tokens_with_overhead` / `PROMPT_OVERHEAD`.
    let mut sorted = diffs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_overhead = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    println!();
    println!("summary (over {n} samples):");
    println!("  mean ratio      : {mean_ratio:.2}  (<1 = underestimate, >1 = overestimate)");
    println!("  mean |error|    : {:.2}%", mean_abs_err * 100.0);
    println!("  max  |error|    : {:.2}%", max_abs_err * 100.0);
    println!(
        "  implied overhead : ~{median_overhead:.0} tokens → set `PROMPT_OVERHEAD` / use `count_messages_tokens_with_overhead`"
    );
}
