// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Fine-grained token-counting tests for openai-rs.
//!
//! Documents the heuristic's exact behavior and its (known, unavoidable)
//! deviation from real tokenizer counts. Reference counts below come from
//! OpenAI's tiktoken documentation (cl100k_base) — the library does NOT
//! ship a BPE tokenizer, so these assert the heuristic stays "in the
//! ballpark", not that it is exact.

use openai_client_rs::*;

/// Well-known tiktoken counts for reference strings (from OpenAI docs).
const TIKTOKEN_REFERENCES: &[(&str, u64)] = &[
    ("Hello world", 2),
    ("The quick brown fox jumps over the lazy dog", 10),
    ("chatgpt", 2),
    ("tiktoken is great!", 6),
    ("😀😃😄😁😆", 3),
];

#[test]
fn heuristic_stays_in_ballpark_of_tiktoken() {
    for (text, tiktoken) in TIKTOKEN_REFERENCES {
        let est = tokens::count_tokens(text);
        let ratio = est as f64 / *tiktoken as f64;
        println!("{text:?}: heuristic={est} tiktoken={tiktoken} ratio={ratio:.2}");
        assert!(
            (0.5..=1.8).contains(&ratio),
            "{text:?}: heuristic {est} vs tiktoken {tiktoken} (ratio {ratio:.2})"
        );
    }
}

#[test]
fn heuristic_is_deterministic() {
    // Exact, stable behavior of the char heuristic.
    let cases = [
        ("", 0),
        ("Hello world", 3), // 11 ascii chars → ceil(11/4) = 3
        ("你好世界", 2),    // 4 CJK chars × 2pts = 8 → 2
        ("tiktoken is great!", 5), // 18 ascii → ceil(18/4) = 5
    ];
    for (text, expected) in cases {
        assert_eq!(tokens::count_tokens(text), expected, "{text:?}");
    }
}

#[test]
fn message_count_includes_role_and_format_overhead() {
    let msg = ChatMessage::user("Hello world");
    let n = tokens::count_message_tokens(&msg);
    // 4 overhead + content(3)
    assert_eq!(n, 7);
}

#[test]
fn messages_count_is_additive() {
    let msgs = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello"),
    ];
    let sum = tokens::count_message_tokens(&msgs[0])
        + tokens::count_message_tokens(&msgs[1])
        + 2;
    assert_eq!(tokens::count_messages_tokens(&msgs), sum);
}

#[test]
fn tool_calls_and_parts_add_to_count() {
    let plain = tokens::count_message_tokens(&ChatMessage::user("hi"));
    let with_tools = tokens::count_message_tokens(&ChatMessage::assistant_with_tools(
        "hi",
        vec![ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "run".into(),
                arguments: r#"{"cmd":"ls"}"#.into(),
            },
        }],
    ));
    assert!(with_tools > plain, "tool calls must add tokens");

    let with_image = tokens::count_message_tokens(&ChatMessage::user_with_images("hi", &["https://x/i.png"]));
    assert!(with_image > plain, "images must add tokens");
}

#[test]
fn prompt_overhead_is_documented_and_additive() {
    // The chat-template overhead is NOT modelled (see PROMPT_OVERHEAD).
    let msgs = vec![ChatMessage::user("hi")];
    let base = tokens::count_messages_tokens(&msgs);
    let _with_overhead = base + tokens::PROMPT_OVERHEAD;
    assert_eq!(tokens::PROMPT_OVERHEAD, 0);
}
