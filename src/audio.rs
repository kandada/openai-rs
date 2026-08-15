// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Audio API — TTS (text-to-speech) and STT (speech-to-text).

use serde::{Deserialize, Serialize};

use crate::client::OpenAiClient;
use crate::error::Result;

/// TTS request.
#[derive(Debug, Clone, Serialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
}

impl SpeechRequest {
    pub fn new(model: impl Into<String>, input: impl Into<String>, voice: impl Into<String>) -> Self {
        SpeechRequest {
            model: model.into(),
            input: input.into(),
            voice: voice.into(),
            speed: None,
            response_format: None,
        }
    }
    pub fn speed(mut self, v: f64) -> Self { self.speed = Some(v); self }
    pub fn format(mut self, v: impl Into<String>) -> Self { self.response_format = Some(v.into()); self }
}

/// Transcription request.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionRequest {
    pub file: String,      // base64 audio data
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

impl TranscriptionRequest {
    pub fn new(file_b64: impl Into<String>, model: impl Into<String>) -> Self {
        TranscriptionRequest {
            file: file_b64.into(),
            model: model.into(),
            language: None,
            prompt: None,
            response_format: None,
            temperature: None,
        }
    }
    pub fn language(mut self, v: impl Into<String>) -> Self { self.language = Some(v.into()); self }
    pub fn prompt(mut self, v: impl Into<String>) -> Self { self.prompt = Some(v.into()); self }
    pub fn temperature(mut self, v: f64) -> Self { self.temperature = Some(v); self }
}

/// Transcription / Translation response.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub segments: Option<Vec<TranscriptionSegment>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionSegment {
    pub id: i64,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub seek: Option<i64>,
    pub tokens: Option<Vec<i64>>,
    pub temperature: Option<f64>,
    pub avg_logprob: Option<f64>,
    pub compression_ratio: Option<f64>,
    pub no_speech_prob: Option<f64>,
}

impl OpenAiClient {
    /// Generate speech from text (TTS). Returns raw audio bytes.
    pub fn audio_speech(&self, request: &SpeechRequest) -> Result<Vec<u8>> {
        let body = serde_json::to_value(request)?;
        let resp = self.agent
            .post(&self.endpoint("audio/speech"))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf)?;
        Ok(buf)
    }

    /// Transcribe audio to text. `file_b64` should be base64-encoded audio.
    pub fn audio_transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("audio/transcriptions", body)?;
        Ok(resp.into_json()?)
    }

    /// Translate audio to English text.
    pub fn audio_translate(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("audio/translations", body)?;
        Ok(resp.into_json()?)
    }
}

#[cfg(feature = "async")]
impl crate::async_client::OpenAiAsyncClient {
    pub async fn audio_speech(&self, request: &SpeechRequest) -> Result<Vec<u8>> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("audio/speech", body).await?;
        Ok(resp.bytes().await?.to_vec())
    }

    pub async fn audio_transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("audio/transcriptions", body).await?;
        Ok(resp.json().await?)
    }

    pub async fn audio_translate(&self, request: &TranscriptionRequest) -> Result<TranscriptionResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("audio/translations", body).await?;
        Ok(resp.json().await?)
    }
}

use std::io::Read;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_request_serde() {
        let req = SpeechRequest::new("tts-1", "Hello", "alloy").speed(1.2);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("tts-1"));
        assert!(json.contains("alloy"));
        assert!(json.contains("1.2"));
    }

    #[test]
    fn test_transcription_response_deserialize() {
        let json = r#"{"text":"hello world","language":"en"}"#;
        let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "hello world");
    }
}