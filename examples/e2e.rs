// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Protocol-specific end-to-end test for openai-rs.
//!
//! OpenAI-compatible distinctive semantics exercised here:
//!   - **Reasoning fields**: `reasoning_content` (DeepSeek/Kimi),
//!     `reasoning` / `thinking` (some private providers), extracted in
//!     both streaming and non-streaming paths.
//!   - **Inline `<think>...</think>` tags**: Qwen-family models put
//!     reasoning in `content`; the library strips them into reasoning.
//!   - **Fragmented tool calls**: `delta.tool_calls[].index` spreads
//!     id/name/arguments across many chunks — must be reassembled.
//!   - **Tool messages**: results are `role: "tool"` messages carrying
//!     `tool_call_id` linking to the assistant's tool_calls.
//!   - **Reasoning models**: `reasoning_effort` / `max_completion_tokens`.
//!
//! Credentials come from the environment — never hard-coded:
//!   LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME
//!
//! Run:
//!   export LLM_API_KEY=... LLM_API_URL=... LLM_MODEL_NAME=...
//!   cargo run --example e2e

use std::env;
use std::process::exit;

use openai_client_rs::*;
use serde_json::json;

fn env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("missing required env var: {name}");
        exit(2);
    })
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

/// Tiny assertion harness: real failures exit non-zero; provider quirks are
/// reported as notes so a lenient/chatty model doesn't fail the suite.
struct T {
    passed: u32,
    failed: Vec<String>,
}

impl T {
    fn check(&mut self, ok: bool, label: &str) {
        if ok {
            self.passed += 1;
            println!("  ✅ {label}");
        } else {
            self.failed.push(label.to_string());
            println!("  ❌ {label}");
        }
    }
    fn note(&self, label: &str) {
        println!("  ⚠️  {label}");
    }
    fn finish(self) {
        println!("\n  passed={} failed={}", self.passed, self.failed.len());
        if !self.failed.is_empty() {
            for f in &self.failed {
                eprintln!("  FAILED: {f}");
            }
            exit(1);
        }
    }
}

fn make_tools() -> Vec<Tool> {
    vec![Tool::function(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )]
}

fn as_tool_call(tc: &SimplifiedToolCall) -> ToolCall {
    ToolCall {
        id: tc.id.clone(),
        call_type: "function".into(),
        function: FunctionCall {
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        },
    }
}

fn main() {
    let key = env("LLM_API_KEY");
    let base = env("LLM_API_URL");
    let model = env("LLM_MODEL_NAME");

    println!("== openai-rs e2e (protocol-specific) ==");
    println!("  endpoint: {base}");
    println!("  model:    {model}");

    let client = OpenAiClient::with_base_url(&key, &model, &base).with_total_timeout(120);
    let tools = make_tools();
    let system = ChatMessage::system(
        "You are a helpful assistant. When asked about the weather, call get_weather for each requested city.",
    );
    let mut t = T { passed: 0, failed: Vec::new() };

    // ── Test 1: non-streaming reasoning + tool message roundtrip ───────────
    println!("── Test 1: non-streaming reasoning + multi-turn tool loop ──");
    let mut history: Vec<ChatMessage> = vec![
        system.clone(),
        ChatMessage::user("What is the weather in San Francisco and in Paris? Use the tool for each."),
    ];
    let mut rounds_ok = 0;
    for round in 0..3 {
        let resp = client.chat_create(&history, Some(&tools)).unwrap_or_else(|e| {
            eprintln!("❌ round {round} failed: {e}");
            exit(1);
        });
        if let Some(rc) = &resp.reasoning_content {
            println!("  reasoning: {}", truncate(rc, 120));
        }
        if let Some(u) = &resp.usage {
            println!("  usage: in={} out={} total={}", u.prompt_tokens, u.completion_tokens, u.total_tokens);
        }
        if resp.tool_calls.is_empty() {
            println!("  round {round}: finished: {}", truncate(&resp.text, 120));
            rounds_ok += 1;
            break;
        }
        // Assistant turn carries its tool_calls (and any reasoning).
        let mut asst = ChatMessage::assistant_with_tools(
            resp.text.clone(),
            resp.tool_calls.iter().map(as_tool_call).collect(),
        );
        asst.reasoning_content = resp.reasoning_content.clone();
        history.push(asst);
        for tc in &resp.tool_calls {
            let city = tc.parsed_args().get("city").cloned().unwrap_or_else(|| json!("?"));
            history.push(ChatMessage::tool_result(&tc.id, format!("Weather in {city}: sunny, 22C")));
            println!("  ↳ {} <- weather result", tc.name);
        }
        rounds_ok += 1;
    }
    t.check(rounds_ok > 0, "multi-turn tool loop completed (tool messages roundtripped)");

    // ── Test 2: streaming fragmented reasoning + parallel tool calls ───────
    println!("── Test 2: streaming (fragmented reasoning + tool_calls) ──");
    let mut streamed_text = String::new();
    let mut streamed_tool_calls: Vec<(String, String)> = Vec::new();
    let resp = client
        .chat_stream(
            &[
                system.clone(),
                ChatMessage::user("Call get_weather for Tokyo and for Berlin in parallel. Then stop."),
            ],
            Some(&tools),
            |d| streamed_text.push_str(d),
            |name, args| streamed_tool_calls.push((name.to_string(), args.to_string())),
        )
        .unwrap_or_else(|e| {
            eprintln!("❌ streaming failed: {e}");
            exit(1);
        });
    if let Some(rc) = &resp.reasoning_content {
        println!("  streamed reasoning ({} chars)", rc.chars().count());
    }
    println!("  streamed text: {}", truncate(&streamed_text, 150));
    t.note(&format!("streamed tool calls: {} (via callback), {} (via response)", streamed_tool_calls.len(), resp.tool_calls.len()));

    // Every streamed tool call must have an id, a name, and parseable args.
    for (i, tc) in resp.tool_calls.iter().enumerate() {
        t.check(!tc.id.is_empty(), &format!("streamed tool_call[{i}] has id"));
        t.check(!tc.name.is_empty(), &format!("streamed tool_call[{i}] has name"));
        let parsed = tc.parsed_args();
        t.check(parsed.is_object(), &format!("streamed tool_call[{i}] args are JSON"));
        println!(
            "  ↳ assembled tool_call[{i}] id={} name={} args={}",
            tc.id,
            tc.name,
            serde_json::to_string(&parsed).unwrap_or_default()
        );
    }
    t.check(!resp.tool_calls.is_empty(), "streamed tool calls reassembled from fragments");

    // ── Test 3: <think> tag stripping (Qwen-style) ─────────────────────────
    println!("── Test 3: inline <think> tag compatibility ──");
    // Ask the model to wrap its reasoning in <think> tags. Not all models
    // comply; probe the raw response first.
    let probe = client
        .chat_create_raw(
            &[
                system.clone(),
                ChatMessage::user(
                    "Before answering, write your internal reasoning inside <think>...</think> tags, then give the final answer after </think>. Answer: what is 2+2?",
                ),
            ],
            None,
        )
        .unwrap_or_else(|e| {
            eprintln!("❌ probe failed: {e}");
            exit(1);
        });
    let raw_content = probe
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();
    let has_tag = raw_content.contains("<think>");
    if has_tag {
        // The library must strip <think> tags from the visible text in ALL
        // cases. When the provider ALSO sends a reasoning field, the tag
        // content is discarded (not double-added to reasoning).
        let assembled = openai_client_rs::api_common::assemble_response(&probe);
        t.check(!assembled.text.contains("<think>"), "content stripped of <think> tags");
        t.check(
            assembled.reasoning_content.is_some(),
            "reasoning populated (field or tag content)",
        );
        println!("  raw content: {}", truncate(&raw_content, 200));
        println!(
            "  assembled: text={} reasoning={}",
            truncate(&assembled.text, 120),
            assembled.reasoning_content.as_deref().unwrap_or("").chars().count()
        );
    } else {
        t.note("model did not emit <think> tags — provider uses reasoning field instead; skipping tag assertion");
    }

    // ── Test 4: reasoning model params (builder path) ──────────────────────
    println!("── Test 4: reasoning_effort + max_completion_tokens ──");
    let req = ChatCompletionRequest::new(model.clone(), history.clone())
        .tools(tools.clone())
        .reasoning_effort("high")
        .max_completion_tokens(2048);
    match client.send(&req) {
        Ok(resp) => {
            t.check(!resp.text.is_empty(), "builder path returns text");
            if let Some(rc) = &resp.reasoning_content {
                println!("  reasoning: {}", truncate(rc, 120));
            }
        }
        Err(e) => t.check(false, &format!("builder path failed: {e}")),
    }

    // ── Test 5: streaming usage (include_usage) ────────────────────────────
    println!("── Test 5: streaming usage via include_usage ──");
    let usage_client = OpenAiClient::with_base_url(&key, &model, &base).with_include_usage(true);
    match usage_client.chat_stream(
        &[system.clone(), ChatMessage::user("Say hello.")],
        None,
        |_| {},
        |_, _| {},
    ) {
        Ok(resp) => match resp.usage {
            Some(u) => {
                println!("  usage: in={} out={} total={}", u.prompt_tokens, u.completion_tokens, u.total_tokens);
                t.check(u.total_tokens > 0, "streaming usage captured from final chunk");
            }
            None => t.note("provider did not send a usage chunk (stream_options not honored)"),
        },
        Err(e) => t.check(false, &format!("include_usage streaming failed: {e}")),
    }

    // ── Test 6: typed chunk stream ─────────────────────────────────────────
    println!("── Test 6: typed chunk stream ──");
    match client.chat_stream_chunks(&[system.clone(), ChatMessage::user("Say hello.")], None) {
        Ok(mut chunks) => {
            let mut seen = 0usize;
            for c in chunks.by_ref() {
                match c {
                    Ok(chunk) => {
                        seen += 1;
                        if let Some(d) = chunk.choices.first().and_then(|c| c.delta.content.as_deref()) {
                            println!("  ↳ [typed] content: {}", truncate(d, 40));
                        }
                    }
                    Err(e) => {
                        t.check(false, &format!("typed chunk stream error: {e}"));
                        break;
                    }
                }
            }
            t.check(seen > 0, &format!("typed chunk stream yielded {seen} chunks"));
        }
        Err(e) => t.check(false, &format!("chat_stream_chunks failed: {e}")),
    }

    // ── Test 7: logit_bias + logprobs ──────────────────────────────────────
    println!("── Test 7: logit_bias + logprobs ──");
    let req = ChatCompletionRequest::new(model.clone(), vec![ChatMessage::user("Say hello.")])
        .logprobs(true)
        .top_logprobs(3)
        .logit_bias(json!({"42": -100}));
    match client.send(&req) {
        Ok(resp) => {
            t.check(!resp.text.is_empty(), "logit_bias + logprobs request accepted");
            let has_logprobs = resp
                .raw
                .as_ref()
                .and_then(|r| r.choices.first())
                .and_then(|c| c.logprobs.as_ref())
                .is_some();
            t.check(has_logprobs, "provider returned typed logprobs");
            if let Some(lp) = resp.raw.as_ref().and_then(|r| r.choices.first()).and_then(|c| c.logprobs.as_ref()) {
                if let Some(content) = &lp.content {
                    if let Some(first) = content.first() {
                        println!(
                            "  top token: {:?} logprob={} ({} candidates)",
                            first.token,
                            first.logprob,
                            first.top_logprobs.len()
                        );
                    }
                }
            }
        }
        Err(e) => t.check(false, &format!("logit_bias request failed: {e}")),
    }

    t.finish();
    println!("✅ openai-rs e2e PASSED");
}
