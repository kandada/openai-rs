// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Embeddings API.

use crate::async_client::OpenAiAsyncClient;
use crate::error::Result;
use crate::types::EmbeddingResponse;
use serde_json::Value;

impl OpenAiAsyncClient {
    /// Create embeddings for the given input(s).
    pub async fn embeddings_create(
        &self,
        input: &[impl AsRef<str>],
        model: &str,
    ) -> Result<EmbeddingResponse> {
        let input_arr: Vec<&str> = input.iter().map(|s| s.as_ref()).collect();
        let input_value = if input_arr.len() == 1 {
            Value::String(input_arr[0].to_string())
        } else {
            Value::Array(
                input_arr
                    .iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect(),
            )
        };
        let body = serde_json::json!({
            "model": model,
            "input": input_value,
        });
        let resp = self.post("embeddings", body).await?;
        Ok(resp.json().await?)
    }

    /// Create a single embedding.
    pub async fn embedding_create(&self, input: impl AsRef<str>, model: &str) -> Result<Vec<f64>> {
        let resp = self.embeddings_create(&[input], model).await?;
        Ok(resp
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .unwrap_or_default())
    }
}
