// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

use std::fmt;

/// Structured payload for `OpenAiError::Api`.
///
/// Mirrors the typed error shape of the official OpenAI SDK, which
/// attaches an HTTP status code (and, when present, a `Retry-After`
/// hint) to every API error.
#[derive(Debug, Clone)]
pub struct ApiError {
    /// HTTP status code returned by the server (None for in-stream errors).
    pub status_code: Option<u16>,
    /// Value of the `Retry-After` header, if the server sent one.
    pub retry_after_secs: Option<u64>,
    /// Human-readable error message.
    pub message: String,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        ApiError {
            status_code: None,
            retry_after_secs: None,
            message: message.into(),
        }
    }

    /// Whether this API error is worth retrying.
    ///
    /// Retryable when the server told us to (429/5xx, timeout keywords,
    /// rate limiting) — including in-stream overloaded/rate-limit events
    /// which carry no status code.
    pub fn is_retryable(&self) -> bool {
        if let Some(code) = self.status_code {
            return matches!(code, 408 | 409 | 429 | 500 | 502 | 503 | 504);
        }
        let l = self.message.to_lowercase();
        l.contains("timeout")
            || l.contains("timed out")
            || l.contains("connection")
            || l.contains("rate limit")
            || l.contains("overloaded")
            || l.contains(" 500")
            || l.contains(" 502")
            || l.contains(" 503")
            || l.contains(" 504")
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.status_code {
            Some(code) => write!(f, "HTTP {code}: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

/// Central error type for the openai-client-rs client.
#[derive(Debug)]
pub enum OpenAiError {
    /// Configuration problem (missing api key, invalid base_url, etc.)
    Config(String),
    /// Network / HTTP transport failure (DNS, connection, timeout).
    Network(String),
    /// API returned an error status (4xx, 5xx) or a streamed error event.
    Api(ApiError),
    /// JSON (de)serialization failure.
    Json(String),
    /// I/O failure (reading response body, writing to sink).
    Io(String),
    /// The operation was cancelled by the caller.
    Cancelled,
}

impl fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenAiError::Config(m) => write!(f, "config error: {m}"),
            OpenAiError::Network(m) => write!(f, "network error: {m}"),
            OpenAiError::Api(e) => write!(f, "api error: {e}"),
            OpenAiError::Json(m) => write!(f, "json error: {m}"),
            OpenAiError::Io(m) => write!(f, "io error: {m}"),
            OpenAiError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for OpenAiError {}

impl From<serde_json::Error> for OpenAiError {
    fn from(e: serde_json::Error) -> Self {
        OpenAiError::Json(e.to_string())
    }
}

impl From<std::io::Error> for OpenAiError {
    fn from(e: std::io::Error) -> Self {
        OpenAiError::Io(e.to_string())
    }
}

#[cfg(feature = "async")]
impl From<reqwest::Error> for OpenAiError {
    fn from(e: reqwest::Error) -> Self {
        OpenAiError::Network(e.to_string())
    }
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, OpenAiError>;

impl OpenAiError {
    /// Build an API error from a status code and body text.
    ///
    /// `retry_after` is the parsed `Retry-After` header value (in seconds),
    /// if the server supplied one.
    pub fn api(status_code: u16, retry_after: Option<u64>, message: impl Into<String>) -> Self {
        OpenAiError::Api(ApiError {
            status_code: Some(status_code),
            retry_after_secs: retry_after,
            message: message.into(),
        })
    }

    /// Build an in-stream API error (no HTTP status code).
    pub fn stream_error(message: impl Into<String>) -> Self {
        OpenAiError::Api(ApiError::new(message))
    }

    /// The `Retry-After` hint carried by this error, if any.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            OpenAiError::Api(e) => e.retry_after_secs,
            _ => None,
        }
    }

    /// Whether this error is worth retrying.
    pub fn is_retryable(&self) -> bool {
        match self {
            OpenAiError::Network(_) => true,
            OpenAiError::Api(e) => e.is_retryable(),
            // A JSON parse failure means the request already succeeded and
            // the response was malformed — retrying cannot change the result.
            OpenAiError::Json(_) => false,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_formats() {
        assert_eq!(OpenAiError::Cancelled.to_string(), "cancelled");
        assert!(OpenAiError::Config("x".into()).to_string().contains("config"));
        let api = OpenAiError::api(503, None, "unavailable");
        assert!(api.to_string().contains("HTTP 503"));
    }

    #[test]
    fn retryable_classification() {
        assert!(OpenAiError::Network("reset".into()).is_retryable());
        assert!(OpenAiError::api(503, None, "unavailable").is_retryable());
        assert!(OpenAiError::api(429, None, "rate limit").is_retryable());
        assert!(OpenAiError::stream_error("overloaded").is_retryable());
        assert!(OpenAiError::stream_error("rate limit exceeded").is_retryable());
        assert!(!OpenAiError::api(401, None, "unauthorized").is_retryable());
        assert!(!OpenAiError::api(400, None, "bad request").is_retryable());
        assert!(!OpenAiError::Config("no key".into()).is_retryable());
        // JSON parse failures must NOT be retried.
        assert!(!OpenAiError::Json("malformed response".into()).is_retryable());
    }

    #[test]
    fn retry_after_exposed() {
        let e = OpenAiError::api(503, Some(12), "busy");
        assert_eq!(e.retry_after_secs(), Some(12));
        assert!(e.is_retryable());
        assert_eq!(OpenAiError::Network("x".into()).retry_after_secs(), None);
    }

    #[test]
    fn json_error_converts() {
        let e: OpenAiError = serde_json::from_str::<i32>("not json").unwrap_err().into();
        matches!(e, OpenAiError::Json(_));
    }
}
