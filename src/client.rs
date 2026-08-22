// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

use std::time::Duration;

use crate::api_common::truncate;
use crate::error::{OpenAiError, Result};
use crate::retry::RetryConfig;

/// Connect timeout: fail fast on dead links.
const CONNECT_TIMEOUT_SECS: u64 = 10;
/// Per-read socket timeout for the SSE stream.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 30;
const WRITE_TIMEOUT_SECS: u64 = 30;

fn build_agent(read_timeout_secs: u64, total_timeout_secs: u64, proxy: Option<&str>) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(read_timeout_secs))
        .timeout_write(Duration::from_secs(WRITE_TIMEOUT_SECS))
        .timeout(Duration::from_secs(total_timeout_secs));
    if let Some(p) = proxy {
        b = b.proxy(ureq::Proxy::new(p).expect("invalid proxy URL"));
    }
    b.build()
}

/// An OpenAI-compatible API client.
///
/// Works with OpenAI, DeepSeek, Kimi, MiniMax, and any OpenAI-compatible provider.
pub struct OpenAiClient {
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) agent: ureq::Agent,
    /// organization header (for OpenAI).
    organization: Option<String>,
    /// default max_tokens for requests.
    default_max_tokens: u64,
    /// default temperature.
    temperature: Option<f64>,
    /// `reasoning_effort` injected into every convenience request.
    reasoning_effort: Option<String>,
    /// `max_completion_tokens` override for reasoning models.
    max_completion_tokens: Option<u64>,
    /// Send `stream_options.include_usage` on convenience stream requests.
    include_usage: bool,
    /// Automatic retry configuration (on by default).
    retry_config: RetryConfig,
    /// HTTP proxy URL (e.g. `http://localhost:8080`).
    proxy: Option<String>,
}

impl OpenAiClient {
    /// Create a client for OpenAI's default API.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        OpenAiClient {
            api_key: api_key.clone(),
            model: model.into(),
            base_url: if api_key.starts_with("sk-ant") {
                "https://api.deepseek.com/v1".into()
            } else {
                Self::detect_base_url(&api_key)
            },
            agent: build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, None),
            organization: None,
            default_max_tokens: 4096,
            temperature: None,
            reasoning_effort: None,
            max_completion_tokens: None,
            include_usage: false,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    /// Create a client with a custom base URL for non-OpenAI providers.
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        OpenAiClient {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            agent: build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, None),
            organization: None,
            default_max_tokens: 4096,
            temperature: None,
            reasoning_effort: None,
            max_completion_tokens: None,
            include_usage: false,
            retry_config: RetryConfig::default(),
            proxy: None,
        }
    }

    /// Route requests through an HTTP proxy.
    pub fn with_proxy(mut self, proxy: impl Into<String>) -> Self {
        self.proxy = Some(proxy.into());
        self.agent = build_agent(DEFAULT_READ_TIMEOUT_SECS, 300, self.proxy.as_deref());
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set the default max_tokens.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Set the default temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set `reasoning_effort` for reasoning models (o1/o3/gpt-5, ...).
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Override `max_tokens` with `max_completion_tokens` (required for
    /// reasoning models). When set, `max_tokens` is omitted from the body.
    pub fn with_max_completion_tokens(mut self, tokens: u64) -> Self {
        self.max_completion_tokens = Some(tokens);
        self
    }

    /// Ask the server to include token usage in the final stream chunk
    /// (`stream_options.include_usage`). Off by default for maximum
    /// provider compatibility.
    pub fn with_include_usage(mut self, enable: bool) -> Self {
        self.include_usage = enable;
        self
    }

    /// Set the maximum number of automatic retries (0 disables retry).
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.retry_config.max_retries = max_retries;
        self
    }

    /// Set a full retry configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Override the read timeout (seconds).
    pub fn with_read_timeout(mut self, secs: u64) -> Self {
        let total = 300;
        self.agent = build_agent(secs, total, self.proxy.as_deref());
        self
    }

    /// Override the total request timeout (seconds).
    pub fn with_total_timeout(mut self, secs: u64) -> Self {
        let read = DEFAULT_READ_TIMEOUT_SECS;
        self.agent = build_agent(read, secs, self.proxy.as_deref());
        self
    }

    /// Change the model after construction.
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    /// The current model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Detect the default base URL based on the API key prefix.
    fn detect_base_url(api_key: &str) -> String {
        if api_key.starts_with("sk-") && api_key.len() > 50 {
            "https://api.deepseek.com/v1".into()
        } else if api_key.starts_with("sk-ant") {
            "https://api.anthropic.com".into()
        } else {
            "https://api.openai.com/v1".into()
        }
    }

    pub fn endpoint(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }

    pub(crate) fn chat_body_options(&self) -> crate::api_common::ChatBodyOptions {
        crate::api_common::ChatBodyOptions {
            reasoning_effort: self.reasoning_effort.clone(),
            max_completion_tokens: self.max_completion_tokens,
            include_usage: self.include_usage,
        }
    }

    pub(crate) fn post(&self, path: &str, body: serde_json::Value) -> Result<ureq::Response> {
        let cfg = &self.retry_config;
        let mut attempt = 0u32;
        loop {
            match self.post_once(path, &body) {
                Ok(r) => return Ok(r),
                Err(e) => {
                    if attempt >= cfg.max_retries || !e.is_retryable() {
                        return Err(e);
                    }
                    let delay = match e.retry_after_secs() {
                        Some(secs) => Duration::from_secs(secs),
                        None => Duration::from_millis(cfg.delay_ms(attempt)),
                    };
                    attempt += 1;
                    std::thread::sleep(delay);
                }
            }
        }
    }

    fn post_once(&self, path: &str, body: &serde_json::Value) -> Result<ureq::Response> {
        let mut req = self
            .agent
            .post(&self.endpoint(path))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json");
        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }
        match req.send_json(body.clone()) {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, r)) => {
                let retry_after = parse_retry_after(&r);
                let msg = r.into_string().unwrap_or_default();
                Err(OpenAiError::api(code, retry_after, truncate(&msg, 500)))
            }
            Err(e) => Err(OpenAiError::Network(e.to_string())),
        }
    }

    pub(crate) fn post_stream(&self, path: &str, body: serde_json::Value) -> Result<impl std::io::Read> {
        let cfg = &self.retry_config;
        let mut attempt = 0u32;
        loop {
            match self.post_stream_once(path, &body) {
                Ok(r) => return Ok(r.into_reader()),
                Err(e) => {
                    if attempt >= cfg.max_retries || !e.is_retryable() {
                        return Err(e);
                    }
                    let delay = match e.retry_after_secs() {
                        Some(secs) => Duration::from_secs(secs),
                        None => Duration::from_millis(cfg.delay_ms(attempt)),
                    };
                    attempt += 1;
                    std::thread::sleep(delay);
                }
            }
        }
    }

    fn post_stream_once(&self, path: &str, body: &serde_json::Value) -> Result<ureq::Response> {
        let mut req = self
            .agent
            .post(&self.endpoint(path))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream");
        if let Some(ref org) = self.organization {
            req = req.set("OpenAI-Organization", org);
        }
        match req.send_json(body.clone()) {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, r)) => {
                let retry_after = parse_retry_after(&r);
                let msg = r.into_string().unwrap_or_default();
                Err(OpenAiError::api(code, retry_after, truncate(&msg, 500)))
            }
            Err(e) => Err(OpenAiError::Network(e.to_string())),
        }
    }

    pub(crate) fn default_max_tokens(&self) -> u64 {
        self.default_max_tokens
    }

    pub(crate) fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    pub fn validate(&self) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": false,
        });
        self.post("chat/completions", body).map(|_| ())
    }
}

fn parse_retry_after(r: &ureq::Response) -> Option<u64> {
    r.header("Retry-After")
        .and_then(|v| v.trim().parse::<u64>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_new_free() {
        let client = OpenAiClient::new("sk-test12345678", "gpt-4o");
        assert_eq!(client.model(), "gpt-4o");
        // sk-test12345678 isn't >50 chars, so it defaults to openai
        assert_eq!(
            client.endpoint("chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn client_with_base_url() {
        let client = OpenAiClient::with_base_url(
            "sk-test",
            "deepseek-chat",
            "https://api.deepseek.com/v1",
        );
        assert_eq!(
            client.endpoint("chat/completions"),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[test]
    fn client_custom_timeout() {
        let _client = OpenAiClient::new("sk-test", "test")
            .with_read_timeout(60)
            .with_total_timeout(600);
        // construction should not panic
    }

    #[test]
    fn client_reasoning_config() {
        let client = OpenAiClient::new("sk-test", "o3")
            .with_reasoning_effort("high")
            .with_max_completion_tokens(8000)
            .with_include_usage(true);
        let opts = client.chat_body_options();
        assert_eq!(opts.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(opts.max_completion_tokens, Some(8000));
        assert!(opts.include_usage);
    }

    #[test]
    fn client_retry_default_on() {
        let client = OpenAiClient::new("sk-test", "test");
        assert_eq!(client.retry_config.max_retries, 3);
        assert_eq!(client.with_retries(0).retry_config.max_retries, 0);
    }
}
