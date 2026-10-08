// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async OpenAI client using `reqwest`.

use std::time::Duration;

use crate::api_common::{truncate, ChatBodyOptions};
use crate::error::{OpenAiError, Result};
use crate::retry::RetryConfig;

fn build_http_client(timeout_secs: u64, proxy: Option<&str>) -> reqwest::Client {
    let mut b = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(10));
    if let Some(p) = proxy {
        b = b.proxy(reqwest::Proxy::all(p).expect("invalid proxy URL"));
    }
    b.build().expect("failed to build reqwest client")
}

/// An async OpenAI-compatible API client using `reqwest`.
pub struct OpenAiAsyncClient {
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) client: reqwest::Client,
    pub(crate) organization: Option<String>,
    pub(crate) default_max_tokens: u64,
    pub(crate) temperature: Option<f64>,
    reasoning_effort: Option<String>,
    max_completion_tokens: Option<u64>,
    include_usage: bool,
    retry_config: RetryConfig,
    proxy: Option<String>,
}

impl OpenAiAsyncClient {
    /// Create an async client for OpenAI's API.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        OpenAiAsyncClient {
            base_url: Self::detect_base_url(&api_key),
            api_key,
            model: model.into(),
            client: build_http_client(300, None),
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

    /// Create a client with custom base URL.
    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        OpenAiAsyncClient {
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            client: build_http_client(300, None),
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
        self.client = build_http_client(300, self.proxy.as_deref());
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Set default max_tokens.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = max_tokens;
        self
    }

    /// Set default temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set `reasoning_effort` for reasoning models.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Override `max_tokens` with `max_completion_tokens`.
    pub fn with_max_completion_tokens(mut self, tokens: u64) -> Self {
        self.max_completion_tokens = Some(tokens);
        self
    }

    /// Ask the server to include token usage in the final stream chunk.
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

    /// Override the request timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.client = build_http_client(secs, self.proxy.as_deref());
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    pub(crate) fn temperature_val(&self) -> Option<f64> {
        self.temperature
    }

    pub(crate) fn default_max_tokens_u64(&self) -> u64 {
        self.default_max_tokens
    }

    pub(crate) fn chat_body_options(&self) -> ChatBodyOptions {
        ChatBodyOptions {
            reasoning_effort: self.reasoning_effort.clone(),
            max_completion_tokens: self.max_completion_tokens,
            include_usage: self.include_usage,
        }
    }

    pub(crate) fn endpoint(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }

    fn detect_base_url(api_key: &str) -> String {
        if api_key.starts_with("sk-") && api_key.len() > 50 {
            "https://api.deepseek.com/v1".into()
        } else if api_key.starts_with("sk-ant") {
            "https://api.anthropic.com".into()
        } else {
            "https://api.openai.com/v1".into()
        }
    }

    pub(crate) async fn post(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<reqwest::Response> {
        let cfg = &self.retry_config;
        let mut attempt = 0u32;
        loop {
            match self.post_once(path, &body).await {
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
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn post_once(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(self.endpoint(path))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json");
        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req
            .json(body)
            .send()
            .await
            .map_err(|e| OpenAiError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::api(code, retry_after, truncate(&msg, 500)));
        }
        Ok(resp)
    }

    pub(crate) async fn post_stream(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<reqwest::Response> {
        let cfg = &self.retry_config;
        let mut attempt = 0u32;
        loop {
            match self.post_stream_once(path, &body).await {
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
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    async fn post_stream_once(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(self.endpoint(path))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req
            .json(body)
            .send()
            .await
            .map_err(|e| OpenAiError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::api(code, retry_after, truncate(&msg, 500)));
        }
        Ok(resp)
    }

    /// Validate the API key.
    pub async fn validate(&self) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": false,
        });
        self.post("chat/completions", body).await.map(|_| ())
    }
}
