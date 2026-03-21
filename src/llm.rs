use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::config::LlmConfig;

#[derive(Clone)]
pub struct LlmClient {
    client: Client,
    config: LlmConfig,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f64,
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f64,
    max_tokens: i32,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessageResponse,
}

#[derive(Deserialize)]
struct OpenAiMessageResponse {
    content: String,
}

impl LlmClient {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            client: Client::new(),
            config: config.clone(),
        }
    }

    /// Send a prompt and get a text response.
    /// Routes to Ollama or OpenAI-compatible endpoint based on config.provider.
    pub async fn complete(&self, prompt: &str) -> Result<String> {
        if self.config.provider == "ollama" {
            self.ollama_complete(prompt).await
        } else {
            self.openai_complete(prompt).await
        }
    }

    async fn ollama_complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/api/generate", self.config.base_url);
        let req = OllamaRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            options: OllamaOptions {
                temperature: 0.3,
                num_predict: 2048,
            },
        };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("ollama returned {status}: {body}");
        }

        let data: OllamaResponse = resp.json().await?;
        Ok(data.response)
    }

    async fn openai_complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);
        let req = OpenAiRequest {
            model: self.config.model.clone(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: 0.3,
            max_tokens: 2048,
        };

        let mut builder = self.client.post(&url).json(&req);
        if !self.config.api_key.is_empty() {
            builder = builder.bearer_auth(&self.config.api_key);
        }

        let resp = builder.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("openai-compat returned {status}: {body}");
        }

        let data: OpenAiResponse = resp.json().await?;
        data.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("empty response from LLM"))
    }

    /// Structured extraction: send a prompt that asks for JSON, attempt to parse it.
    pub async fn complete_json<T: serde::de::DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let raw = self.complete(prompt).await?;

        // Try to extract JSON from the response (LLMs love wrapping in ```json blocks)
        let cleaned = extract_json_block(&raw);
        serde_json::from_str(&cleaned)
            .with_context(|| format!("failed to parse LLM JSON response:\n{raw}"))
    }
}

fn extract_json_block(s: &str) -> String {
    // Strip ```json ... ``` fences if present
    if let Some(start) = s.find("```json") {
        let after = &s[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = s.find("```") {
        let after = &s[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    // Try to find raw JSON object or array
    if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            return s[start..=end].to_string();
        }
    }
    s.trim().to_string()
}
