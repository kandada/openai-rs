// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Models API — list and retrieve model information.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client::OpenAiClient;
use crate::error::{OpenAiError, Result};
use crate::types::ModelInfo;

/// Response from listing models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl OpenAiClient {
    /// List all available models.
    pub fn models_list(&self) -> Result<ModelList> {
        match self
            .agent
            .get(&self.endpoint("models"))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
        {
            Ok(r) => Ok(r.into_json()?),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_string().unwrap_or_default();
                Err(OpenAiError::Api(format!("HTTP {code}: {msg}")))
            }
            Err(e) => Err(OpenAiError::Network(e.to_string())),
        }
    }

    /// Retrieve a specific model by ID.
    pub fn models_retrieve(&self, model_id: &str) -> Result<ModelInfo> {
        match self
            .agent
            .get(&self.endpoint(&format!("models/{model_id}")))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
        {
            Ok(r) => Ok(r.into_json()?),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_string().unwrap_or_default();
                Err(OpenAiError::Api(format!("HTTP {code}: {msg}")))
            }
            Err(e) => Err(OpenAiError::Network(e.to_string())),
        }
    }

    /// Delete a fine-tuned model.
    pub fn models_delete(&self, model_id: &str) -> Result<Value> {
        match self
            .agent
            .delete(&self.endpoint(&format!("models/{model_id}")))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
        {
            Ok(r) => Ok(r.into_json()?),
            Err(ureq::Error::Status(code, r)) => {
                let msg = r.into_string().unwrap_or_default();
                Err(OpenAiError::Api(format!("HTTP {code}: {msg}")))
            }
            Err(e) => Err(OpenAiError::Network(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_list_deserializes() {
        let json = r#"{"object":"list","data":[{"id":"gpt-4o","object":"model","created":1,"owned_by":"openai"}]}"#;
        let list: ModelList = serde_json::from_str(json).unwrap();
        assert_eq!(list.data.len(), 1);
        assert_eq!(list.data[0].id, "gpt-4o");
    }
}