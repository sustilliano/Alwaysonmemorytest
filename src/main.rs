mod api;
mod config;
mod consolidate;
mod db;
mod edges;
mod ingest;
mod llm;
mod localrecall;
mod query;
mod revision;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "always_on_memory=info".into()),
        )
        .init();

    // Load config
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());
    let config = config::Config::load(&config_path)?;

    info!("always-on-memory agent starting");
    info!("  database: {:?}", config.database.path);
    info!("  llm: {} ({})", config.llm.base_url, config.llm.model);
    info!("  inbox: {:?}", config.watcher.inbox_dir);
    info!("  consolidation: every {} min", config.consolidation.interval_minutes);
    info!("  trait dimensions: {}", config.traits.dimensions);

    // Open the memory store
    let store = db::MemoryStore::open(&config.database.path)?;
    let llm_client = llm::LlmClient::new(&config.llm);

    // Shared state
    let state = Arc::new(api::AppState {
        store: store.clone(),
        llm: llm_client.clone(),
        config: config.clone(),
    });

    // File watcher channel
    let (file_tx, mut file_rx) = mpsc::channel::<std::path::PathBuf>(100);

    // Spawn inbox file watcher
    let inbox = config.watcher.inbox_dir.clone();
    ingest::spawn_watcher(inbox.clone(), file_tx.clone())?;

    // Scan existing files on startup
    ingest::scan_existing(&inbox, &file_tx).await?;

    // Spawn file ingestion task
    let ingest_store = store.clone();
    let ingest_llm = llm_client.clone();
    let trait_dims = config.traits.dimensions;
    tokio::spawn(async move {
        while let Some(path) = file_rx.recv().await {
            info!("ingesting file: {path:?}");
            match ingest::ingest_file(&ingest_store, &ingest_llm, &path, "default", trait_dims).await {
                Ok(id) if id > 0 => info!("ingested {path:?} -> memory #{id}"),
                Ok(_) => info!("skipped {path:?} (duplicate or unsupported)"),
                Err(e) => error!("failed to ingest {path:?}: {e:?}"),
            }
        }
    });

    // Spawn consolidation timer
    let cons_store = store.clone();
    let cons_llm = llm_client.clone();
    let cons_interval = config.consolidation.interval_minutes;
    let cons_min = config.consolidation.min_unconsolidated;
    let cons_threshold = config.consolidation.edge_threshold;
    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(cons_interval * 60));
        // Skip first tick (fires immediately)
        interval.tick().await;

        loop {
            interval.tick().await;
            info!("consolidation timer fired");

            // Check if we have enough unconsolidated memories
            match cons_store.unconsolidated_memories().await {
                Ok(uncons) if uncons.len() >= cons_min => {
                    match consolidate::consolidate(&cons_store, &cons_llm, cons_threshold).await {
                        Ok(result) => {
                            info!(
                                "consolidation done: {} memories, {} edges, avg_edge={:.3}",
                                result.memories_processed,
                                result.edges_detected,
                                result.avg_edge_score
                            );
                        }
                        Err(e) => error!("consolidation failed: {e:?}"),
                    }
                }
                Ok(uncons) => {
                    info!(
                        "skipping consolidation: only {} unconsolidated (need {})",
                        uncons.len(),
                        cons_min
                    );
                }
                Err(e) => error!("failed to check unconsolidated: {e:?}"),
            }
        }
    });

    // Start Axum HTTP server
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("API server listening on {addr}");
    info!("native endpoints:");
    info!("  GET  /status         - memory stats");
    info!("  GET  /memories       - list all memories");
    info!("  GET  /edges          - list detected edges");
    info!("  GET  /consolidations - list consolidation insights");
    info!("  POST /ingest         - ingest text");
    info!("  GET  /query?q=...    - query memory");
    info!("  POST /consolidate    - trigger manual consolidation");
    info!("  POST /delete         - delete memory");
    info!("  POST /clear          - clear all memories");
    info!("localrecall-compatible endpoints:");
    info!("  POST /api/collections                          - create collection");
    info!("  GET  /api/collections                          - list collections");
    info!("  POST /api/collections/{{name}}/upload            - add content");
    info!("  GET  /api/collections/{{name}}/entries           - list entries");
    info!("  GET  /api/collections/{{name}}/entries/{{file}}    - get entry");
    info!("  POST /api/collections/{{name}}/search            - search (topology-aware)");
    info!("  POST /api/collections/{{name}}/reset             - reset collection");
    info!("  DEL  /api/collections/{{name}}/entry/delete      - delete entry");

    // Merge native API + LocalRecall compat API
    let app = api::router(state.clone()).merge(localrecall::router(state));
    axum::serve(listener, app).await?;

    Ok(())
}
