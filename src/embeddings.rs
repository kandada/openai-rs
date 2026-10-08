// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Embeddings API — generate text embeddings.

use crate::client::OpenAiClient;
use crate::error::Result;
use crate::types::EmbeddingResponse;
use serde_json::Value;

impl OpenAiClient {
    /// Create embeddings for the given input(s).
    ///
    /// `input` can be a single string or multiple strings.
    /// `model` is the embedding model ID (e.g. "text-embedding-3-small").
    pub fn embeddings_create(
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
        let resp = self.post("embeddings", body)?;
        Ok(resp.into_json()?)
    }

    /// Create an embedding for a single string (convenience method).
    pub fn embedding_create(&self, input: impl AsRef<str>, model: &str) -> Result<Vec<f64>> {
        let resp = self.embeddings_create(&[input], model)?;
        Ok(resp
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_response_deserializes() {
        let json = r#"{
            "object":"list",
            "data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],
            "model":"text-embedding-3-small",
            "usage":{"prompt_tokens":5,"total_tokens":5}
        }"#;
        let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].embedding.len(), 3);
        assert_eq!(resp.usage.prompt_tokens, 5);
    }
}
