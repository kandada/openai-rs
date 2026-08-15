// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async OpenAI client using `reqwest`.

use std::time::Duration;

use crate::api_common::truncate;
use crate::error::{OpenAiError, Result};

/// An async OpenAI-compatible API client using `reqwest`.
pub struct OpenAiAsyncClient {
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
    pub(crate) client: reqwest::Client,
    pub(crate) organization: Option<String>,
    pub(crate) default_max_tokens: u64,
    pub(crate) temperature: Option<f64>,
}

impl OpenAiAsyncClient {
    /// Create an async client for OpenAI's API.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let api_key = api_key.into();
        OpenAiAsyncClient {
            base_url: Self::detect_base_url(&api_key),
            api_key,
            model: model.into(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("failed to build reqwest client"),
            organization: None,
            default_max_tokens: 4096,
            temperature: None,
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
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("failed to build reqwest client"),
            organization: None,
            default_max_tokens: 4096,
            temperature: None,
        }
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

    /// Override the request timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(Duration::from_secs(secs))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client");
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

    pub(crate) async fn post(&self, path: &str, body: serde_json::Value) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(self.endpoint(path))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json");
        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req.json(&body).send().await.map_err(|e| {
            OpenAiError::Network(e.to_string())
        })?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
        }
        Ok(resp)
    }

    pub(crate) async fn post_stream(&self, path: &str, body: serde_json::Value) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(self.endpoint(path))
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req.json(&body).send().await.map_err(|e| {
            OpenAiError::Network(e.to_string())
        })?;
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::Api(format!("HTTP {code}: {}", truncate(&msg, 500))));
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