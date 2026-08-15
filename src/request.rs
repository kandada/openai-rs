// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Request builder for OpenAI Chat Completions API.
//!
//! Provides full control over all OpenAI API parameters.

use serde_json::{json, Value};

use crate::types::{ChatMessage, Tool, ToolChoice};

/// Builder for constructing a chat completion request body.
///
/// Mirrors all parameters in the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub n: Option<i32>,
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    pub max_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub response_format: Option<ResponseFormat>,
    pub stream_options: Option<StreamOptions>,
    pub reasoning_effort: Option<String>,
    pub user: Option<String>,
}

/// Response format configuration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaObject>,
}

impl ResponseFormat {
    pub fn json_object() -> Self {
        ResponseFormat { format_type: "json_object".into(), json_schema: None }
    }
    pub fn json_schema(name: impl Into<String>, schema: Value, strict: bool) -> Self {
        ResponseFormat {
            format_type: "json_schema".into(),
            json_schema: Some(JsonSchemaObject {
                name: name.into(),
                schema,
                strict: Some(strict),
                description: None,
            }),
        }
    }
}

/// JSON Schema object for structured outputs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct JsonSchemaObject {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Streaming options.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        ChatCompletionRequest {
            model: String::new(),
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            n: None,
            stream: false,
            stop: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
            stream_options: None,
            reasoning_effort: None,
            user: None,
        }
    }
}

impl ChatCompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        ChatCompletionRequest {
            model: model.into(),
            messages,
            ..Default::default()
        }
    }

    pub fn temperature(mut self, v: f64) -> Self { self.temperature = Some(v); self }
    pub fn top_p(mut self, v: f64) -> Self { self.top_p = Some(v); self }
    pub fn n(mut self, v: i32) -> Self { self.n = Some(v); self }
    pub fn stream(mut self, v: bool) -> Self { self.stream = v; self }
    pub fn stop(mut self, v: Vec<String>) -> Self { self.stop = Some(v); self }
    pub fn max_tokens(mut self, v: u64) -> Self { self.max_tokens = Some(v); self }
    pub fn max_completion_tokens(mut self, v: u64) -> Self { self.max_completion_tokens = Some(v); self }
    pub fn presence_penalty(mut self, v: f64) -> Self { self.presence_penalty = Some(v); self }
    pub fn frequency_penalty(mut self, v: f64) -> Self { self.frequency_penalty = Some(v); self }
    pub fn seed(mut self, v: i64) -> Self { self.seed = Some(v); self }
    pub fn tools(mut self, v: Vec<Tool>) -> Self { self.tools = Some(v); self }
    pub fn tool_choice(mut self, v: ToolChoice) -> Self { self.tool_choice = Some(v); self }
    pub fn parallel_tool_calls(mut self, v: bool) -> Self { self.parallel_tool_calls = Some(v); self }
    pub fn response_format(mut self, v: ResponseFormat) -> Self { self.response_format = Some(v); self }
    pub fn stream_options(mut self, v: StreamOptions) -> Self { self.stream_options = Some(v); self }
    pub fn reasoning_effort(mut self, v: impl Into<String>) -> Self { self.reasoning_effort = Some(v.into()); self }
    pub fn user(mut self, v: impl Into<String>) -> Self { self.user = Some(v.into()); self }

    /// Build the JSON body for this request.
    pub fn build_body(&self) -> Value {
        let model_lower = self.model.to_lowercase();
        let temperature = if model_lower.contains("kimi") || model_lower.contains("moonshot") {
            1.0
        } else {
            self.temperature.unwrap_or(1.0)
        };
        let max_tokens = self.max_completion_tokens.or(self.max_tokens).unwrap_or(4096);

        let msgs = super::api_common::build_messages_json(&self.messages);
        let mut body = json!({
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": self.stream,
        });

        if let Some(v) = self.top_p { body["top_p"] = json!(v); }
        if let Some(v) = self.n { body["n"] = json!(v); }
        if let Some(ref v) = self.stop { body["stop"] = json!(v); }
        if let Some(v) = self.presence_penalty { body["presence_penalty"] = json!(v); }
        if let Some(v) = self.frequency_penalty { body["frequency_penalty"] = json!(v); }
        if let Some(v) = self.seed { body["seed"] = json!(v); }
        if let Some(ref v) = self.tools {
            let arr: Vec<Value> = v.iter().map(|t| serde_json::to_value(t).unwrap_or_default()).collect();
            body["tools"] = Value::Array(arr);
            body["tool_choice"] = serde_json::to_value(self.tool_choice.as_ref().unwrap_or(&ToolChoice::auto())).unwrap_or_default();
        }
        if let Some(v) = self.parallel_tool_calls { body["parallel_tool_calls"] = json!(v); }
        if let Some(ref v) = self.response_format { body["response_format"] = serde_json::to_value(v).unwrap_or_default(); }
        if let Some(ref v) = self.stream_options { body["stream_options"] = serde_json::to_value(v).unwrap_or_default(); }
        if let Some(ref v) = self.reasoning_effort { body["reasoning_effort"] = json!(v); }
        if let Some(ref v) = self.user { body["user"] = json!(v); }

        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tool;
    use serde_json::json;

    fn make_messages() -> Vec<ChatMessage> {
        vec![ChatMessage::user("Hello")]
    }

    #[test]
    fn test_minimal_body() {
        let body = ChatCompletionRequest::new("gpt-4o", make_messages()).build_body();
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn test_full_body() {
        let tools = vec![Tool::function("fn", "desc", json!({"type":"object","properties":{}}))];
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .temperature(0.5)
            .top_p(0.9)
            .max_tokens(2048)
            .seed(42)
            .stop(vec!["END".into()])
            .presence_penalty(0.1)
            .frequency_penalty(0.2)
            .tools(tools)
            .parallel_tool_calls(false)
            .reasoning_effort("medium")
            .user("test-user")
            .build_body();

        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["max_tokens"], 2048);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["stop"].as_array().unwrap()[0], "END");
        assert_eq!(body["presence_penalty"], 0.1);
        assert_eq!(body["frequency_penalty"], 0.2);
        assert!(body["tools"].is_array());
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["reasoning_effort"], "medium");
        assert_eq!(body["user"], "test-user");
    }

    #[test]
    fn test_response_format_json_object() {
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .response_format(ResponseFormat::json_object())
            .build_body();
        let rf = &body["response_format"];
        assert_eq!(rf["type"], "json_object");
    }

    #[test]
    fn test_response_format_json_schema() {
        let schema = json!({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]});
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .response_format(ResponseFormat::json_schema("MySchema", schema, true))
            .build_body();
        let rf = &body["response_format"];
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "MySchema");
        assert_eq!(rf["json_schema"]["strict"], true);
    }

    #[test]
    fn test_stream_options() {
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .stream(true)
            .stream_options(StreamOptions { include_usage: Some(true) })
            .build_body();
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn test_kimi_forces_temp_1() {
        let body = ChatCompletionRequest::new("kimi-k2", make_messages())
            .temperature(0.1)
            .build_body();
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn test_max_completion_tokens_priority() {
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .max_tokens(1000)
            .max_completion_tokens(2000)
            .build_body();
        assert_eq!(body["max_tokens"], 2000);
    }

    #[test]
    fn test_body_serializable() {
        let body = ChatCompletionRequest::new("gpt-4o", make_messages())
            .temperature(0.5)
            .tools(vec![Tool::function("f", "d", json!({"type":"object","properties":{}}))])
            .build_body();
        let s = serde_json::to_string(&body).unwrap();
        assert!(s.contains("gpt-4o"));
        assert!(s.contains("tools"));
    }
}