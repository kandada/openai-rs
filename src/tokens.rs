// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Token counting utility.
//!
//! Provides approximate token counts for messages. Uses character-based
//! heuristics (≈4 chars per token for English, ≈1.5 chars for CJK).
//! For production accuracy, consider using a tiktoken-compatible crate.

use crate::types::{ChatMessage, ContentPart};

/// Approximate token count for a string.
///
/// Uses a simple heuristic: ~4 characters per token for ASCII,
/// ~1.5 characters per token for CJK.
pub fn count_tokens(text: &str) -> u64 {
    if text.is_empty() { return 0; }
    let mut tokens = 0;
    for ch in text.chars() {
        if ch.is_ascii() {
            tokens += 1;
        } else {
            tokens += 3; // CJK characters roughly 3x
        }
    }
    ((tokens as f64) / 4.0).ceil() as u64
}

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
    messages.iter().map(|m| count_message_tokens(m) + 1).sum()
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
        assert!(n >= 3 && n <= 6, "got {n}");
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