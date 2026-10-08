// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Token counting utility.
//!
//! Provides approximate token counts for messages using a character-based
//! heuristic (≈4 chars per token for English, ≈2 chars per token for CJK).
//! This is a **fallback for rough context budgeting only** — it cannot match
//! a provider's real tokenizer:
//!
//! - OpenAI: accurate counts come from `usage.prompt_tokens` in a real
//!   response (or the separate `tiktoken` library for known encodings).
//! - Provider-specific chat-template overhead (DeepSeek ≈ 70 tokens,
//!   MiniMax ≈ 160, OpenAI gpt-4o ≈ 4–10) is NOT modelled.
//!
//! Measure against a real provider with the `token_calibration` example.

use crate::types::{ChatMessage, ContentPart};

/// Approximate token count for a string.
///
/// Character heuristic: ASCII ≈ 1 token per 4 chars; non-ASCII (CJK etc.)
/// ≈ 2 tokens per 4 chars (≈2 chars/token). Calibrated against DeepSeek /
/// MiniMax via the `token_calibration` example.
pub fn count_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    let mut tokens = 0;
    for ch in text.chars() {
        if ch.is_ascii() {
            tokens += 1;
        } else {
            tokens += 2; // CJK roughly 2 chars per token
        }
    }
    ((tokens as f64) / 4.0).ceil() as u64
}

/// Approximate per-request chat-template overhead, not modelled by
/// [`count_messages_tokens`]. Provider-specific (DeepSeek ≈ 70,
/// MiniMax ≈ 160, OpenAI gpt-4o ≈ 4–10). Add it if you want estimates closer
/// to the server's `prompt_tokens` for a known provider.
pub const PROMPT_OVERHEAD: u64 = 0;

/// Approximate token count for a message (including role and formatting overhead).
pub fn count_message_tokens(msg: &ChatMessage) -> u64 {
    let mut total = 4; // role + formatting overhead
    total += count_tokens(&msg.content);

    if let Some(ref parts) = msg.content_parts {
        for part in parts {
            match part {
                ContentPart::Text { text } => total += count_tokens(text),
                ContentPart::ImageUrl { .. } => total += 85, // image tokens vary, ~85 is low detail
                ContentPart::InputAudio { .. } => total += 200,
                ContentPart::File { .. } => total += 100,
            }
        }
    }

    if let Some(ref tcs) = msg.tool_calls {
        for tc in tcs {
            total += count_tokens(&tc.function.name);
            total += count_tokens(&tc.function.arguments);
            total += 10; // JSON structure overhead
        }
    }

    if let Some(ref rc) = msg.reasoning_content {
        total += count_tokens(rc);
    }

    total
}

/// Approximate total token count for a list of messages.
pub fn count_messages_tokens(messages: &[ChatMessage]) -> u64 {
    count_messages_tokens_with_overhead(messages, 0)
}

/// Like [`count_messages_tokens`] but adds a provider-specific chat-template
/// overhead (see [`PROMPT_OVERHEAD`]).
pub fn count_messages_tokens_with_overhead(messages: &[ChatMessage], overhead: u64) -> u64 {
    overhead
        + messages
            .iter()
            .map(|m| count_message_tokens(m) + 1)
            .sum::<u64>()
}

/// Estimate max_tokens for a completion based on desired response length.
pub fn estimate_completion_tokens(desired_chars: u64) -> u64 {
    count_tokens(&"x".repeat(desired_chars as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn test_count_english() {
        let n = count_tokens("Hello world");
        assert!(n >= 2 && n <= 5, "got {n}");
    }

    #[test]
    fn test_count_cjk() {
        let n = count_tokens("你好世界");
        assert!(n >= 2 && n <= 4, "got {n}");
        // measured: DeepSeek/MiniMax tokenize CJK at ~2 chars/token
        assert_eq!(count_tokens("你好世界"), 2);
    }

    #[test]
    fn test_count_message() {
        let msg = ChatMessage::user("Hello world");
        let n = count_message_tokens(&msg);
        assert!(n >= 5, "got {n}");
    }

    #[test]
    fn test_count_messages() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let n = count_messages_tokens(&msgs);
        assert!(n >= 10, "got {n}");
    }
}
