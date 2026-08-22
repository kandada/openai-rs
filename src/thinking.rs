// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Thinking / reasoning extraction for OpenAI-compatible streams.
//!
//! Different providers ship model "thinking" through different channels:
//!
//! 1. **Field strategies** — a dedicated delta field, in priority order:
//!    - `reasoning_content` — DeepSeek / Kimi-K2-thinking convention
//!    - `reasoning` — some private-provider convention
//!    - `thinking` — another private-provider convention
//! 2. **Inline `<think>...</think>` tags** — Qwen-family models put their
//!    reasoning inside `content` between `<think>` / `</think>` markers.
//!
//! Field strategies take priority for *reasoning accumulation* (the field
//! value becomes `reasoning`), but `<think>` tags in `content` are always
//! stripped from the visible text. When both formats appear in the same
//! response the tag content is discarded so the same reasoning is not
//! double-counted. The tag parser is stateful so it survives tags split
//! across SSE chunks.
//!
//! This mirrors the extractor used by aacode-rs.

use serde_json::Value;

/// Field names tried in priority order to extract thinking from a delta.
pub const OPENAI_THINKING_FIELDS: &[&str] = &["reasoning_content", "reasoning", "thinking"];

/// `<think>` opening tag (Qwen-style inline reasoning).
const TAG_OPEN: &str = "<think>";
/// `</think>` closing tag.
const TAG_CLOSE: &str = "</think>";

/// Stateful `<think>...</think>` parser that survives chunk boundaries.
#[derive(Debug, Clone, Default)]
pub struct ThinkTagParser {
    /// Cross-chunk hold-back buffer. Always flushed by [`ThinkTagParser::flush`].
    tag_buffer: String,
    /// Whether the parser is currently inside a `<think>` block.
    in_think: bool,
    /// Whether the very first fragment of the current thinking segment
    /// is still pending (used to drop formatting whitespace after `<think>`).
    first_think_pending: bool,
    /// Same as `first_think_pending`, for the content channel.
    first_content_pending: bool,
}

impl ThinkTagParser {
    /// Feed one content chunk into the parser.
    ///
    /// Returns `(think_chunk, content_chunk)` to append to the respective
    /// accumulators. Both may be empty.
    ///
    /// Hold-back model: only the suffix starting at the **first `<`** in the
    /// residual buffer can possibly form the next tag; everything before it is
    /// guaranteed not to start a tag and is emitted right away. The suffix
    /// stays buffered for the next call. Whitespace around tag boundaries is
    /// trimmed so model formatting newlines do not leak into either channel.
    pub fn feed(&mut self, content: &str) -> (String, String) {
        if content.is_empty() {
            return (String::new(), String::new());
        }
        let mut full = std::mem::take(&mut self.tag_buffer);
        full.push_str(content);

        let mut out_think = String::new();
        let mut out_content = String::new();
        let mut last_safe_idx = 0;

        loop {
            let search_from = &full[last_safe_idx..];
            let (needle, _needle_len) = if self.in_think {
                (TAG_CLOSE, TAG_CLOSE.len())
            } else {
                (TAG_OPEN, TAG_OPEN.len())
            };
            if let Some(rel_idx) = search_from.find(needle) {
                let abs_idx = last_safe_idx + rel_idx;
                let segment = &full[last_safe_idx..abs_idx];
                if self.in_think {
                    push_segment(&mut out_think, segment, true, self.first_think_pending);
                    self.first_think_pending = false;
                } else {
                    push_segment(&mut out_content, segment, false, self.first_content_pending);
                    self.first_content_pending = false;
                }
                self.in_think = !self.in_think;
                last_safe_idx = abs_idx + needle.len();
                if self.in_think {
                    self.first_think_pending = true;
                } else {
                    self.first_content_pending = true;
                }
            } else {
                break;
            }
        }

        let remaining = &full[last_safe_idx..];
        if remaining.is_empty() {
            return (out_think, out_content);
        }

        // Emit everything up to the first `<` (guaranteed not to start a
        // tag) and hold back the rest. `find` always lands on a byte
        // boundary because `<` is 1-byte ASCII.
        let lt_idx = remaining.find('<');
        let (emit_part, hold_part) = match lt_idx {
            Some(i) => (&remaining[..i], &remaining[i..]),
            None => (remaining, ""),
        };

        if !emit_part.is_empty() {
            if self.in_think {
                push_segment(&mut out_think, emit_part, false, self.first_think_pending);
                self.first_think_pending = false;
            } else {
                push_segment(&mut out_content, emit_part, false, self.first_content_pending);
                self.first_content_pending = false;
            }
        }
        if !hold_part.is_empty() {
            self.tag_buffer.push_str(hold_part);
        }

        (out_think, out_content)
    }

    /// Flush any residual buffered content at end-of-stream.
    ///
    /// If still inside `<think>`, the residue is treated as thinking content
    /// (best-effort recovery from an unclosed tag); otherwise as content.
    pub fn flush(&mut self) -> (String, String) {
        if self.tag_buffer.is_empty() {
            return (String::new(), String::new());
        }
        let out = std::mem::take(&mut self.tag_buffer);
        if self.in_think {
            (out, String::new())
        } else {
            (String::new(), out)
        }
    }
}

/// Extract thinking from one streaming `delta` object.
///
/// Returns `(thinking_chunk, content_chunk)`.
///
/// When a reasoning field is present, the field value is returned as
/// `thinking_chunk` and the `content` is still run through the tag parser
/// so that inline `<think>...</think>` tags are **stripped from the visible
/// text** — but the tag content is discarded (not appended to reasoning),
/// avoiding double-counting when a provider emits the same reasoning in
/// both a field and inline tags.
pub fn extract_thinking(delta: &Value, parser: &mut ThinkTagParser) -> (String, String) {
    let mut field_hit = false;
    let mut rc = String::new();
    for &field in OPENAI_THINKING_FIELDS {
        if let Some(v) = delta.get(field).and_then(|v| v.as_str()) {
            if !v.is_empty() {
                rc.push_str(v);
            }
            field_hit = true;
            break;
        }
    }
    let content = delta.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if field_hit {
        // Feed content through the parser anyway so tags are stripped and
        // cross-chunk parser state stays consistent; discard the tag content.
        let (_, stripped) = parser.feed(content);
        return (rc, stripped);
    }
    parser.feed(content)
}

/// One-shot split of a complete content string into `(thinking, content)`.
///
/// Used for non-streaming responses that may embed `<think>...</think>`.
pub fn split_thinking(content: &str) -> (String, String) {
    let mut parser = ThinkTagParser::default();
    let (t, c) = parser.feed(content);
    let (t2, c2) = parser.flush();
    (t + &t2, c + &c2)
}

/// Push a non-empty `seg` into `out`, trimming formatting whitespace around
/// tag boundaries:
///   • `is_before_tag_close` — the segment is immediately followed by
///     `</think>`; trim trailing whitespace.
///   • `is_first` — this is the very first emit of the current segment
///     type in this stream; trim leading whitespace.
fn push_segment(out: &mut String, seg: &str, is_before_tag_close: bool, is_first: bool) {
    if seg.is_empty() {
        return;
    }
    let to_emit: std::borrow::Cow<'_, str> = if is_first && is_before_tag_close {
        std::borrow::Cow::Owned(seg.trim().to_string())
    } else if is_first {
        std::borrow::Cow::Owned(seg.trim_start().to_string())
    } else if is_before_tag_close {
        std::borrow::Cow::Owned(seg.trim_end().to_string())
    } else {
        std::borrow::Cow::Borrowed(seg)
    };
    if to_emit.is_empty() {
        return;
    }
    out.push_str(&to_emit);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta_json(content: &str) -> Value {
        serde_json::json!({"content": content})
    }

    fn extract(content: &str) -> (String, String) {
        let mut p = ThinkTagParser::default();
        let (t, c) = extract_thinking(&delta_json(content), &mut p);
        let (t2, c2) = p.flush();
        (t + &t2, c + &c2)
    }

    #[test]
    fn fields_priority_reasoning_content_first() {
        let d = serde_json::json!({"reasoning_content":"R","reasoning":"Re","thinking":"T","content":"C"});
        let mut p = ThinkTagParser::default();
        let (t, c) = extract_thinking(&d, &mut p);
        assert_eq!(t, "R");
        assert_eq!(c, "C"); // content passes through verbatim
    }

    #[test]
    fn field_thinking_recognized() {
        let d = serde_json::json!({"thinking":"plan A","content":"go"});
        let mut p = ThinkTagParser::default();
        let (t, c) = extract_thinking(&d, &mut p);
        assert_eq!(t, "plan A");
        assert_eq!(c, "go");
    }

    #[test]
    fn complete_tag_in_one_chunk() {
        let (t, c) = extract("<think>plan</think>result");
        assert_eq!(t, "plan");
        assert_eq!(c, "result");
    }

    #[test]
    fn tag_open_split_across_chunks() {
        let mut p = ThinkTagParser::default();
        let (t1, c1) = extract_thinking(&delta_json("<thi"), &mut p);
        let (t2, c2) = extract_thinking(&delta_json("nk>hello</think>world"), &mut p);
        let (t3, c3) = p.flush();
        assert_eq!(format!("{t1}{t2}{t3}"), "hello");
        assert_eq!(format!("{c1}{c2}{c3}"), "world");
    }

    #[test]
    fn tag_close_split_across_chunks() {
        let mut p = ThinkTagParser::default();
        let (t1, c1) = extract_thinking(&delta_json("<think>hello</th"), &mut p);
        let (t2, c2) = extract_thinking(&delta_json("ink>world"), &mut p);
        let (t3, c3) = p.flush();
        assert_eq!(format!("{t1}{t2}{t3}"), "hello");
        assert_eq!(format!("{c1}{c2}{c3}"), "world");
    }

    #[test]
    fn one_char_per_chunk() {
        let mut p = ThinkTagParser::default();
        let mut t = String::new();
        let mut c = String::new();
        for ch in "<think>thinking</think>answer".chars() {
            let (a, b) = extract_thinking(&delta_json(&ch.to_string()), &mut p);
            t.push_str(&a);
            c.push_str(&b);
        }
        let (a, b) = p.flush();
        t.push_str(&a);
        c.push_str(&b);
        assert_eq!(t, "thinking");
        assert_eq!(c, "answer");
    }

    #[test]
    fn unclosed_tag_flushed_as_thinking() {
        let (t, c) = extract("<think>still thinking");
        assert_eq!(t, "still thinking");
        assert_eq!(c, "");
    }

    #[test]
    fn multiple_blocks() {
        let (t, c) = extract("<think>a</think>X<think>b</think>Y");
        assert_eq!(t, "ab");
        assert_eq!(c, "XY");
    }

    #[test]
    fn plain_content_untouched() {
        let (t, c) = extract("hello world");
        assert_eq!(t, "");
        assert_eq!(c, "hello world");
    }

    #[test]
    fn no_phantom_newline_from_think_open() {
        let (t, c) = extract("<think>\nT");
        assert_eq!(t, "T");
        assert_eq!(c, "");
    }

    #[test]
    fn no_leading_newline_after_tag() {
        let (t, c) = extract("<think>plan\n</think>\nresult");
        assert_eq!(t, "plan");
        assert_eq!(c, "result");
    }

    #[test]
    fn split_thinking_one_shot() {
        let (t, c) = split_thinking("<think>deep</think>answer text");
        assert_eq!(t, "deep");
        assert_eq!(c, "answer text");
    }

    #[test]
    fn field_hit_strips_tags_but_does_not_reuse_tag_content() {
        let d = serde_json::json!({"reasoning_content":"R","content":"<think>T</think>C"});
        let mut p = ThinkTagParser::default();
        let (t, c) = extract_thinking(&d, &mut p);
        // reasoning comes from the field only (no double-count of tag content)
        assert_eq!(t, "R");
        // but the visible text is still cleaned of the inline tags
        assert_eq!(c, "C");
    }
}
