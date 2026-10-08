// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Models API.

use crate::api_common::truncate;
use crate::async_client::OpenAiAsyncClient;
use crate::error::OpenAiError;
use crate::error::Result;
use crate::models::ModelList;
use crate::types::ModelInfo;
use serde_json::Value;

impl OpenAiAsyncClient {
    /// List available models.
    pub async fn models_list(&self) -> Result<ModelList> {
        let resp = self
            .client
            .get(self.endpoint("models"))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| OpenAiError::Network(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::api(status.as_u16(), None, truncate(&msg, 500)));
        }
        Ok(resp.json().await?)
    }

    /// Retrieve a model by ID.
    pub async fn models_retrieve(&self, model_id: &str) -> Result<ModelInfo> {
        let resp = self
            .client
            .get(self.endpoint(&format!("models/{model_id}")))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| OpenAiError::Network(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::api(status.as_u16(), None, truncate(&msg, 500)));
        }
        Ok(resp.json().await?)
    }

    /// Delete a fine-tuned model.
    pub async fn models_delete(&self, model_id: &str) -> Result<Value> {
        let resp = self
            .client
            .delete(self.endpoint(&format!("models/{model_id}")))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| OpenAiError::Network(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let msg = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::api(status.as_u16(), None, truncate(&msg, 500)));
        }
        Ok(resp.json().await?)
    }
}
