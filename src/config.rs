use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub watcher: WatcherConfig,
    pub consolidation: ConsolidationConfig,
    pub llm: LlmConfig,
    pub database: DatabaseConfig,
    pub traits: TraitsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WatcherConfig {
    pub inbox_dir: PathBuf,
    pub poll_interval_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ConsolidationConfig {
    pub interval_minutes: u64,
    pub min_unconsolidated: usize,
    pub edge_threshold: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub path: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TraitsConfig {
    pub dimensions: usize,
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config from {path}"))?;
        let config: Config =
            toml::from_str(&content).with_context(|| "failed to parse config")?;
        Ok(config)
    }
}
