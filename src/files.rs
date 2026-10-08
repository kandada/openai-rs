// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Files API — upload, list, retrieve, delete files.

use crate::client::OpenAiClient;
use crate::error::Result;
use serde::Deserialize;
use serde_json::Value;

/// File object returned by the API.
#[derive(Debug, Clone, Deserialize)]
pub struct FileObject {
    pub id: String,
    pub object: String,
    pub bytes: i64,
    pub created_at: i64,
    pub filename: String,
    pub purpose: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub status_details: Option<String>,
}

/// Response from listing files.
#[derive(Debug, Clone, Deserialize)]
pub struct FileList {
    pub object: String,
    pub data: Vec<FileObject>,
}

impl OpenAiClient {
    /// List all uploaded files.
    pub fn files_list(&self) -> Result<FileList> {
        let resp = self
            .agent
            .get(&self.endpoint("files"))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.into_json()?)
    }

    /// Upload a file. `file_b64` should be base64-encoded file content.
    pub fn files_upload(
        &self,
        filename: &str,
        file_b64: &str,
        purpose: &str,
    ) -> Result<FileObject> {
        let body = serde_json::json!({
            "file": file_b64,
            "filename": filename,
            "purpose": purpose,
        });
        let resp = self.post("files", body)?;
        Ok(resp.into_json()?)
    }

    /// Delete a file.
    pub fn files_delete(&self, file_id: &str) -> Result<Value> {
        let resp = self
            .agent
            .delete(&self.endpoint(&format!("files/{file_id}")))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.into_json()?)
    }

    /// Retrieve file metadata.
    pub fn files_retrieve(&self, file_id: &str) -> Result<FileObject> {
        let resp = self
            .agent
            .get(&self.endpoint(&format!("files/{file_id}")))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.into_json()?)
    }

    /// Download file content as raw bytes.
    pub fn files_content(&self, file_id: &str) -> Result<Vec<u8>> {
        let resp = self
            .agent
            .get(&self.endpoint(&format!("files/{file_id}/content")))
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .call()
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf)?;
        Ok(buf)
    }
}

#[cfg(feature = "async")]
impl crate::async_client::OpenAiAsyncClient {
    pub async fn files_list(&self) -> Result<FileList> {
        let resp = self
            .client
            .get(self.endpoint("files"))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.json().await?)
    }

    pub async fn files_upload(
        &self,
        filename: &str,
        file_b64: &str,
        purpose: &str,
    ) -> Result<FileObject> {
        let body =
            serde_json::json!({ "file": file_b64, "filename": filename, "purpose": purpose });
        let resp = self.post("files", body).await?;
        Ok(resp.json().await?)
    }

    pub async fn files_delete(&self, file_id: &str) -> Result<Value> {
        let resp = self
            .client
            .delete(self.endpoint(&format!("files/{file_id}")))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.json().await?)
    }

    pub async fn files_retrieve(&self, file_id: &str) -> Result<FileObject> {
        let resp = self
            .client
            .get(self.endpoint(&format!("files/{file_id}")))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| crate::error::OpenAiError::Network(e.to_string()))?;
        Ok(resp.json().await?)
    }
}

use std::io::Read;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_object_deserialize() {
        let json = r#"{"id":"file-1","object":"file","bytes":100,"created_at":1,"filename":"test.jsonl","purpose":"fine-tune"}"#;
        let f: FileObject = serde_json::from_str(json).unwrap();
        assert_eq!(f.id, "file-1");
        assert_eq!(f.purpose, "fine-tune");
    }
}
