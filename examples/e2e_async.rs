// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async end-to-end test for openai-rs (`OpenAiAsyncClient`).
//!
//! Same protocol-specific coverage as `e2e.rs` but through the async
//! `reqwest` client: reasoning extraction, multi-turn tool message
//! roundtrip, streaming (fragmented reasoning + tool_calls) and usage.
//!
//! Credentials come from the environment — never hard-coded:
//!   LLM_API_KEY / LLM_API_URL / LLM_MODEL_NAME
//!
//! Run:
//!   export LLM_API_KEY=... LLM_API_URL=... LLM_MODEL_NAME=...
//!   cargo run --example e2e_async --features async

#[cfg(feature = "async")]
use std::env;
use std::process::exit;
#[cfg(feature = "async")]
use openai_client_rs::*;
#[cfg(feature = "async")]
use serde_json::json;

#[cfg(not(feature = "async"))]
fn main() {
    eprintln!("this example requires the `async` feature: cargo run --example e2e_async --features async");
    exit(2);
}

#[cfg(feature = "async")]
fn env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("missing required env var: {name}");
        exit(2);
    })
}

#[cfg(feature = "async")]
fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

#[cfg(feature = "async")]
struct T {
    passed: u32,
    failed: Vec<String>,
}

#[cfg(feature = "async")]
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

#[cfg(feature = "async")]
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

#[cfg(feature = "async")]
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

#[cfg(feature = "async")]
#[tokio::main]
async fn main() {
    let key = env("LLM_API_KEY");
    let base = env("LLM_API_URL");
    let model = env("LLM_MODEL_NAME");

    println!("== openai-rs e2e (ASYNC) ==");
    println!("  endpoint: {base}");
    println!("  model:    {model}");

    let client = OpenAiAsyncClient::with_base_url(&key, &model, &base);
    let tools = make_tools();
    let system = ChatMessage::system(
        "You are a helpful assistant. When asked about the weather, call get_weather for each requested city.",
    );
    let mut t = T { passed: 0, failed: Vec::new() };

    // ── Test 1: async non-streaming reasoning + tool loop ──────────────────
    println!("── Test 1: async reasoning + multi-turn tool loop ──");
    let mut history: Vec<ChatMessage> = vec![
        system.clone(),
        ChatMessage::user("What is the weather in San Francisco and in Paris? Use the tool for each."),
    ];
    let mut rounds_ok = 0;
    for round in 0..3 {
        let resp = client.chat_create(&history, Some(&tools)).await.unwrap_or_else(|e| {
            eprintln!("❌ async round {round} failed: {e}");
            exit(1);
        });
        if let Some(rc) = &resp.reasoning_content {
            println!("  reasoning: {}", truncate(rc, 100));
        }
        if let Some(u) = &resp.usage {
            println!("  usage: in={} out={} total={}", u.prompt_tokens, u.completion_tokens, u.total_tokens);
        }
        if resp.tool_calls.is_empty() {
            println!("  round {round}: finished: {}", truncate(&resp.text, 120));
            rounds_ok += 1;
            break;
        }
        let mut asst = ChatMessage::assistant_with_tools(
            resp.text.clone(),
            resp.tool_calls.iter().map(as_tool_call).collect(),
        );
        asst.reasoning_content = resp.reasoning_content.clone();
        history.push(asst);
        for tc in &resp.tool_calls {
            let city = tc.parsed_args().get("city").cloned().unwrap_or_else(|| json!("?"));
            history.push(ChatMessage::tool_result(&tc.id, format!("Weather in {city}: sunny, 22C")));
        }
        rounds_ok += 1;
    }
    t.check(rounds_ok > 0, "async multi-turn tool loop completed");

    // ── Test 2: async streaming reasoning + fragmented tool calls ──────────
    println!("── Test 2: async streaming (reasoning + tool_calls) ──");
    let mut streamed_text = String::new();
    let mut streamed_reasoning = String::new();
    let resp = client
        .chat_stream(
            &[
                system.clone(),
                ChatMessage::user("Call get_weather for Tokyo and for Berlin in parallel. Then stop."),
            ],
            Some(&tools),
            |d| streamed_text.push_str(d),
            |name, args| println!("  ↳ [stream] tool {name}: {args}"),
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("❌ async streaming failed: {e}");
            exit(1);
        });
    if let Some(rc) = &resp.reasoning_content {
        streamed_reasoning.push_str(rc);
    }
    if !streamed_reasoning.is_empty() {
        t.check(true, "async streamed reasoning captured");
    } else {
        t.note("no reasoning field emitted in stream");
    }
    println!("  streamed text: {}", truncate(&streamed_text, 150));
    for (i, tc) in resp.tool_calls.iter().enumerate() {
        t.check(!tc.id.is_empty(), &format!("async tool_call[{i}] has id"));
        t.check(!tc.name.is_empty(), &format!("async tool_call[{i}] has name"));
        t.check(tc.parsed_args().is_object(), &format!("async tool_call[{i}] args JSON"));
        println!(
            "  ↳ assembled tool_call[{i}] id={} name={} args={}",
            tc.id,
            tc.name,
            serde_json::to_string(&tc.parsed_args()).unwrap_or_default()
        );
    }
    t.check(!resp.tool_calls.is_empty(), "async streamed tool calls reassembled");

    // ── Test 3: async builder path (reasoning_effort) ──────────────────────
    println!("── Test 3: async request-builder path ──");
    let req = ChatCompletionRequest::new(model.clone(), history.clone())
        .tools(tools)
        .reasoning_effort("high");
    match client.send(&req).await {
        Ok(resp) => {
            t.check(!resp.text.is_empty(), "async builder path returns text");
            if let Some(rc) = &resp.reasoning_content {
                println!("  reasoning: {}", truncate(rc, 100));
            }
        }
        Err(e) => t.check(false, &format!("async builder path failed: {e}")),
    }

    // ── Test 4: async streaming usage ──────────────────────────────────────
    println!("── Test 4: async streaming usage ──");
    let usage_client = client.with_include_usage(true);
    match usage_client.chat_stream(&[system, ChatMessage::user("Say hello.")], None, |_| {}, |_, _| {}).await {
        Ok(resp) => match resp.usage {
            Some(u) => {
                println!("  usage: in={} out={} total={}", u.prompt_tokens, u.completion_tokens, u.total_tokens);
                t.check(u.total_tokens > 0, "async streaming usage captured");
            }
            None => t.note("no usage chunk from provider"),
        },
        Err(e) => t.check(false, &format!("async include_usage failed: {e}")),
    }

    t.finish();
    println!("✅ openai-rs async e2e PASSED");
}
