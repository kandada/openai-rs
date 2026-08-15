// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Images API — DALL-E image generation, editing, variations.

use serde::{Deserialize, Serialize};
use crate::client::OpenAiClient;
use crate::error::Result;

/// Image generation request parameters.
#[derive(Debug, Clone, Serialize)]
pub struct ImageRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl ImageRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        ImageRequest {
            prompt: prompt.into(),
            model: None,
            n: None,
            size: None,
            quality: None,
            style: None,
            response_format: None,
            user: None,
        }
    }
    pub fn model(mut self, v: impl Into<String>) -> Self { self.model = Some(v.into()); self }
    pub fn n(mut self, v: i32) -> Self { self.n = Some(v); self }
    pub fn size(mut self, v: impl Into<String>) -> Self { self.size = Some(v.into()); self }
    pub fn quality(mut self, v: impl Into<String>) -> Self { self.quality = Some(v.into()); self }
    pub fn style(mut self, v: impl Into<String>) -> Self { self.style = Some(v.into()); self }
    pub fn response_format(mut self, v: impl Into<String>) -> Self { self.response_format = Some(v.into()); self }
    pub fn user(mut self, v: impl Into<String>) -> Self { self.user = Some(v.into()); self }
}

/// Image generation response.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageResponse {
    pub created: i64,
    pub data: Vec<ImageData>,
}

/// A single generated image.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageData {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub b64_json: Option<String>,
    #[serde(default)]
    pub revised_prompt: Option<String>,
}

impl OpenAiClient {
    /// Generate images from a prompt using DALL-E.
    pub fn images_generate(&self, request: &ImageRequest) -> Result<ImageResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("images/generations", body)?;
        Ok(resp.into_json()?)
    }

    /// Edit an existing image (masked region replaced by prompt).
    /// Requires sending image + mask as multipart. For simplicity, use with_base64.
    pub fn images_edit(
        &self,
        image_b64: &str,
        mask_b64: Option<&str>,
        prompt: &str,
        n: Option<i32>,
        size: Option<&str>,
    ) -> Result<ImageResponse> {
        let mut body = serde_json::json!({
            "image": image_b64,
            "prompt": prompt,
        });
        if let Some(m) = mask_b64 { body["mask"] = serde_json::json!(m); }
        if let Some(n) = n { body["n"] = serde_json::json!(n); }
        if let Some(s) = size { body["size"] = serde_json::json!(s); }
        let resp = self.post("images/edits", body)?;
        Ok(resp.into_json()?)
    }

    /// Create a variation of an existing image.
    pub fn images_variation(
        &self,
        image_b64: &str,
        n: Option<i32>,
        size: Option<&str>,
    ) -> Result<ImageResponse> {
        let mut body = serde_json::json!({ "image": image_b64 });
        if let Some(n) = n { body["n"] = serde_json::json!(n); }
        if let Some(s) = size { body["size"] = serde_json::json!(s); }
        let resp = self.post("images/variations", body)?;
        Ok(resp.into_json()?)
    }
}

#[cfg(feature = "async")]
impl crate::async_client::OpenAiAsyncClient {
    pub async fn images_generate(&self, request: &ImageRequest) -> Result<ImageResponse> {
        let body = serde_json::to_value(request)?;
        let resp = self.post("images/generations", body).await?;
        Ok(resp.json().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_request_serde() {
        let req = ImageRequest::new("a cat")
            .n(2)
            .size("1024x1024")
            .quality("hd")
            .response_format("b64_json");
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("a cat"));
        assert!(json.contains("1024x1024"));
    }

    #[test]
    fn test_image_response_deserialize() {
        let json = r#"{"created":123,"data":[{"url":"http://x.com/a.png","revised_prompt":"a cat"}]}"#;
        let resp: ImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].url.as_deref(), Some("http://x.com/a.png"));
    }
}