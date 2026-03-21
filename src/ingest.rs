use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;
use tracing::{info, warn, error};

use crate::db::MemoryStore;
use crate::llm::LlmClient;
use crate::revision::RevisionResult;

/// Structured extraction result from the LLM.
#[derive(Debug, serde::Deserialize)]
struct Extraction {
    summary: String,
    topics: Vec<String>,
    entities: Vec<String>,
    importance: f64,
    traits: Vec<f64>,
}

/// Result of an ingestion attempt, including revision detection details.
#[derive(Debug, serde::Serialize)]
pub struct IngestResult {
    pub id: i64,
    pub revision: RevisionResult,
}

/// Ingest raw text content into the memory store.
pub async fn ingest_text(
    store: &MemoryStore,
    llm: &LlmClient,
    content: &str,
    source: &str,
    collection: &str,
    trait_dims: usize,
) -> Result<IngestResult> {
    let hash = sha256_hash(content);

    if store.has_file_hash(&hash).await? {
        info!("skipping exact duplicate: {source}");
        return Ok(IngestResult {
            id: -1,
            revision: RevisionResult {
                match_id: None,
                classification: crate::revision::Classification::StrongMatch,
                flight_status: crate::revision::FlightStatus::Landed,
                regime_scores: std::collections::HashMap::new(),
                total_score: 1.0,
                fired_regime: Some("file_hash".to_string()),
                explanation: "exact duplicate (hash match)".to_string(),
            },
        });
    }

    let extraction = extract_structured(llm, content, trait_dims).await?;

    // -- Revision detection (QX cascade, 9D core) --
    let existing = store.memories_by_collection(collection).await?;

    let incoming = crate::revision::IncomingContent {
        content: content.to_string(),
        summary: extraction.summary.clone(),
        source: source.to_string(),
        topics: extraction.topics.clone(),
        entities: extraction.entities.clone(),
        traits: extraction.traits.clone(),
        importance: extraction.importance,
        created_at: None,
        domain_meta: std::collections::HashMap::new(),
    };

    let engine = crate::revision::CascadeEngine::with_core_regimes();
    let rev = engine.detect(&incoming, &existing);

    match &rev.classification {
        crate::revision::Classification::StrongMatch => {
            info!(
                source,
                match_id = rev.match_id,
                score = format!("{:.3}", rev.total_score),
                "flight=LANDED — near-duplicate detected, skipping"
            );
            return Ok(IngestResult { id: -1, revision: rev });
        }
        crate::revision::Classification::SemanticMatch => {
            info!(
                source,
                match_id = rev.match_id,
                score = format!("{:.3}", rev.total_score),
                "flight=FALLING — revision detected, ingesting as new version"
            );
        }
        crate::revision::Classification::DriftMatch => {
            info!(
                source,
                match_id = rev.match_id,
                score = format!("{:.3}", rev.total_score),
                "flight=FALLING — drift detected, ingesting with link"
            );
        }
        crate::revision::Classification::Novel => {
            info!(source, "flight=CRASHED — novel content");
        }
    }
    // ─────────────────────────────────────────────────────────────

    let id = store
        .insert_memory(
            collection,
            content,
            &extraction.summary,
            source,
            &hash,
            extraction.importance.clamp(0.0, 1.0),
            &extraction.traits,
            &extraction.topics,
            &extraction.entities,
        )
        .await?;

    // If revision/drift, store the edge linking old → new
    if let Some(parent_id) = rev.match_id {
        let relationship = format!(
            "{}:{}",
            serde_json::to_string(&rev.classification).unwrap_or_default(),
            rev.explanation
        );
        if let Err(e) = store
            .insert_edge(parent_id, id, rev.total_score, &relationship)
            .await
        {
            warn!("failed to link revision edge {parent_id}->{id}: {e}");
        }
    }

    info!(
        id,
        source,
        topics = ?extraction.topics,
        flight = ?rev.flight_status,
        "ingested memory"
    );

    Ok(IngestResult { id, revision: rev })
}

/// Ingest a file from disk.
pub async fn ingest_file(
    store: &MemoryStore,
    llm: &LlmClient,
    path: &Path,
    collection: &str,
    trait_dims: usize,
) -> Result<IngestResult> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let content = match ext.as_str() {
        "txt" | "md" | "json" | "csv" | "log" | "xml" | "yaml" | "yml" | "toml" | "rs"
        | "py" | "js" | "ts" | "html" | "css" => {
            std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {path:?}"))?
        }
        _ => {
            warn!("unsupported file type: {ext} ({path:?})");
            return Ok(IngestResult {
                id: -1,
                revision: RevisionResult {
                    match_id: None,
                    classification: crate::revision::Classification::Novel,
                    flight_status: crate::revision::FlightStatus::Crashed,
                    regime_scores: std::collections::HashMap::new(),
                    total_score: 0.0,
                    fired_regime: None,
                    explanation: format!("unsupported file type: {ext}"),
                },
            });
        }
    };

    if content.trim().is_empty() {
        return Ok(IngestResult {
            id: -1,
            revision: RevisionResult {
                match_id: None,
                classification: crate::revision::Classification::Novel,
                flight_status: crate::revision::FlightStatus::Crashed,
                regime_scores: std::collections::HashMap::new(),
                total_score: 0.0,
                fired_regime: None,
                explanation: "empty file".to_string(),
            },
        });
    }

    let source = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    ingest_text(store, llm, &content, source, collection, trait_dims).await
}

/// Spawn the file watcher task. Sends file paths through the channel.
/// Returns the debouncer so the caller can keep it alive for the program lifetime.
pub fn spawn_watcher(
    inbox: PathBuf,
    tx: mpsc::Sender<PathBuf>,
) -> Result<notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>> {
    use notify::{RecursiveMode, Watcher};
    use notify_debouncer_mini::new_debouncer;
    use std::time::Duration;

    // Ensure inbox exists
    std::fs::create_dir_all(&inbox)?;

    let tx_clone = tx.clone();
    let mut debouncer = new_debouncer(Duration::from_secs(2), move |res: Result<Vec<notify_debouncer_mini::DebouncedEvent>, Vec<notify::Error>>| {
        match res {
            Ok(events) => {
                for event in events {
                    let path = event.path;
                    if path.is_file() {
                        if let Err(e) = tx_clone.blocking_send(path.clone()) {
                            error!("failed to send file event for {path:?}: {e}");
                        }
                    }
                }
            }
            Err(errors) => error!("watcher error: {errors:?}"),
        }
    })?;

    debouncer.watcher().watch(&inbox, RecursiveMode::NonRecursive)?;

    info!("watching inbox: {inbox:?}");
    Ok(debouncer)
}

/// Scan inbox for any existing files on startup.
pub async fn scan_existing(inbox: &Path, tx: &mpsc::Sender<PathBuf>) -> Result<()> {
    if !inbox.exists() {
        return Ok(());
    }
    let mut entries = tokio::fs::read_dir(inbox).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_file() {
            tx.send(path).await?;
        }
    }
    Ok(())
}

/// Ask the LLM to extract structured information from content.
async fn extract_structured(
    llm: &LlmClient,
    content: &str,
    trait_dims: usize,
) -> Result<Extraction> {
    // Truncate very long content for the extraction prompt
    let truncated = if content.len() > 8000 {
        &content[..8000]
    } else {
        content
    };

    let prompt = format!(
        r#"Analyze the following content and extract structured information.
Return ONLY a JSON object with these fields:
- "summary": a 1-2 sentence summary
- "topics": array of 2-6 topic keywords (lowercase)
- "entities": array of named entities (people, companies, technologies, places)
- "importance": float 0.0-1.0 (how significant/actionable is this information)
- "traits": array of exactly {trait_dims} floats between -1.0 and 1.0, representing the content's character along these axes:
  [technical-creative, abstract-concrete, theoretical-practical, broad-focused,
   objective-subjective, static-dynamic, historical-forward, individual-collective,
   simple-complex, cautious-bold, formal-casual, analytical-intuitive,
   incremental-revolutionary, local-global, passive-active, certain-speculative]
  (use only the first {trait_dims} axes)

Content:
{truncated}

JSON:"#
    );

    match llm.complete_json::<Extraction>(&prompt).await {
        Ok(extraction) => Ok(extraction),
        Err(e) => {
            warn!("LLM extraction failed, using defaults: {e}");
            // Fallback: create a minimal extraction
            Ok(Extraction {
                summary: content.chars().take(200).collect(),
                topics: vec![],
                entities: vec![],
                importance: 0.5,
                traits: vec![0.0; trait_dims],
            })
        }
    }
}

fn sha256_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}
