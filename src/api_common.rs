// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared API logic used by both sync and async clients.
//!
//! Message building, body construction, response assembly.

use serde_json::{json, Value};

use crate::types::{ChatMessage, ChatCompletion, LlmResponse, SimplifiedToolCall, Tool};

/// Build the OpenAI chat messages JSON array from internal ChatMessages.
pub fn build_messages_json(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let mut obj = json!({ "role": m.role });
        let map = obj.as_object_mut().unwrap();
        if let Some(ref parts) = m.content_parts {
            map.insert("content".into(), serde_json::to_value(parts).unwrap_or_default());
        } else {
            map.insert("content".into(), Value::String(m.content.clone()));
        }
        if let Some(tcs) = &m.tool_calls {
            let arr: Vec<Value> = tcs
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": tc.call_type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                })
                .collect();
            map.insert("tool_calls".into(), Value::Array(arr));
        }
        if let Some(id) = &m.tool_call_id {
            map.insert("tool_call_id".into(), Value::String(id.clone()));
        }
        if let Some(name) = &m.name {
            map.insert("name".into(), Value::String(name.clone()));
        }
        if let Some(rc) = &m.reasoning_content {
            map.insert("reasoning_content".into(), Value::String(rc.clone()));
        }
        out.push(obj);
    }
    out
}

/// Build the request body for a chat completion.
pub fn build_chat_body(
    model: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    stream: bool,
    max_tokens: u64,
    temperature: Option<f64>,
) -> Value {
    let model_lower = model.to_lowercase();
    let temp = if model_lower.contains("kimi") || model_lower.contains("moonshot") {
        1.0
    } else {
        temperature.unwrap_or(0.7)
    };
    let msgs = build_messages_json(messages);
    let mut body = json!({
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temp,
    });
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let arr: Vec<Value> = tools.iter().map(|t| serde_json::to_value(t).unwrap_or_default()).collect();
            body["tools"] = Value::Array(arr);
            body["tool_choice"] = json!("auto");
        }
    }
    body
}

/// Assemble a non-streaming ChatCompletion into a LlmResponse.
pub fn assemble_response(raw: &ChatCompletion) -> LlmResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<SimplifiedToolCall> = Vec::new();
    let mut reasoning: Option<String> = None;
    let mut finish_reason: Option<String> = None;
    if let Some(choice) = raw.choices.first() {
        if let Some(ref content) = choice.message.content {
            text.push_str(content);
        }
        finish_reason = choice.finish_reason.clone();
        if let Some(ref rc) = choice.message.reasoning_content {
            reasoning = Some(rc.clone());
        }
        if let Some(ref tcs) = choice.message.tool_calls {
            for tc in tcs {
                tool_calls.push(SimplifiedToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                });
            }
        }
    }
    LlmResponse {
        text,
        tool_calls,
        reasoning_content: reasoning,
        finish_reason,
        raw: Some(raw.clone()),
    }
}

pub(crate) fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}