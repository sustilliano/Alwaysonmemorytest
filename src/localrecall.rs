//! LocalRecall-compatible API endpoints.
//!
//! This module implements the same REST API contract as mudler/LocalRecall,
//! allowing always-on-memory to serve as a drop-in replacement for LocalAGI's
//! knowledge base backend — but with Rgano edge detection and trait extraction
//! underneath instead of flat vector similarity.
//!
//! LocalRecall endpoints:
//!   POST   /api/collections                         -> create collection
//!   GET    /api/collections                         -> list collections
//!   POST   /api/collections/:name/upload            -> upload file
//!   GET    /api/collections/:name/entries            -> list entries
//!   GET    /api/collections/:name/entries/:filename  -> get entry content
//!   POST   /api/collections/:name/search            -> search
//!   POST   /api/collections/:name/reset             -> reset collection
//!   DELETE /api/collections/:name/entry/delete       -> delete entry

use axum::{
    extract::{Path, State},
    response::Json,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::api::AppState;
use crate::ingest;
use crate::query;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/collections", post(create_collection))
        .route("/api/collections", get(list_collections))
        .route("/api/collections/{name}/upload", post(upload_text))
        .route("/api/collections/{name}/entries", get(list_entries))
        .route(
            "/api/collections/{name}/entries/{filename}",
            get(get_entry),
        )
        .route("/api/collections/{name}/search", post(search_collection))
        .route("/api/collections/{name}/reset", post(reset_collection))
        .route(
            "/api/collections/{name}/entry/delete",
            delete(delete_entry),
        )
        .with_state(state)
}

// --- Request/Response types matching LocalRecall's contract ---

#[derive(Deserialize)]
struct CreateCollectionReq {
    name: String,
}

#[derive(Deserialize)]
struct SearchReq {
    query: String,
    max_results: Option<usize>,
}

#[derive(Deserialize)]
struct DeleteEntryReq {
    entry: String,
}

#[derive(Serialize)]
struct ChunkResponse {
    id: i64,
    content: String,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct EntryDetailResponse {
    collection: String,
    entry: String,
    chunks: Vec<ChunkResponse>,
    count: usize,
}

#[derive(Serialize)]
struct SearchResult {
    content: String,
    source: String,
    similarity: f64,
    metadata: serde_json::Value,
}

// --- Handlers ---

/// Create a collection. In our system this is a no-op since collections
/// are created implicitly on first insert, but we acknowledge it for compat.
async fn create_collection(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionReq>,
) -> Json<serde_json::Value> {
    info!("localrecall compat: created collection '{}'", req.name);
    Json(serde_json::json!({
        "status": "ok",
        "collection": req.name
    }))
}

/// List all collections.
async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, String> {
    let collections = state
        .store
        .list_collections()
        .await
        .map_err(|e| e.to_string())?;
    Ok(Json(serde_json::json!({ "collections": collections })))
}

/// Upload/add text content to a collection.
/// LocalRecall supports file upload via multipart; we accept JSON text for now.
async fn upload_text(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, String> {
    // Accept either {"content": "..."} or {"text": "..."} for flexibility
    let content = body
        .get("content")
        .or_else(|| body.get("text"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing 'content' or 'text' field".to_string())?;

    let source = body
        .get("source")
        .or_else(|| body.get("filename"))
        .and_then(|v| v.as_str())
        .unwrap_or("upload");

    let id = ingest::ingest_text(
        &state.store,
        &state.llm,
        content,
        source,
        &collection,
        state.config.traits.dimensions,
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(Json(serde_json::json!({
        "status": if id > 0 { "ok" } else { "duplicate" },
        "id": id
    })))
}

/// List all entries (sources) in a collection.
async fn list_entries(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<Json<serde_json::Value>, String> {
    let memories = state
        .store
        .memories_by_collection(&collection)
        .await
        .map_err(|e| e.to_string())?;

    let entries: Vec<serde_json::Value> = memories
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m.source,
                "id": m.id,
                "summary": m.summary,
                "importance": m.importance,
                "topics": m.topics,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "collection": collection,
        "entries": entries,
        "count": entries.len()
    })))
}

/// Get a specific entry's content by source filename.
async fn get_entry(
    State(state): State<Arc<AppState>>,
    Path((collection, filename)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, String> {
    let memories = state
        .store
        .memories_by_collection(&collection)
        .await
        .map_err(|e| e.to_string())?;

    let matching: Vec<_> = memories.iter().filter(|m| m.source == filename).collect();

    let chunks: Vec<ChunkResponse> = matching
        .iter()
        .map(|m| ChunkResponse {
            id: m.id,
            content: m.content.clone(),
            metadata: serde_json::json!({
                "summary": m.summary,
                "topics": m.topics,
                "entities": m.entities,
                "importance": m.importance,
                "traits": m.traits,
            }),
        })
        .collect();

    let count = chunks.len();
    let resp = EntryDetailResponse {
        collection,
        entry: filename,
        chunks,
        count,
    };

    Ok(Json(serde_json::to_value(resp).unwrap()))
}

/// Search a collection. This is where our system shines:
/// instead of flat vector similarity, we use the topology-aware query engine.
async fn search_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<SearchReq>,
) -> Result<Json<serde_json::Value>, String> {
    // Use our topology-aware query engine
    let result = query::query(&state.store, &state.llm, &req.query)
        .await
        .map_err(|e| e.to_string())?;

    // Also get collection-specific memories for direct results
    let memories = state
        .store
        .memories_by_collection(&collection)
        .await
        .map_err(|e| e.to_string())?;

    let max = req.max_results.unwrap_or(5);

    // Return both the synthesized answer and the raw matching memories
    // in a format LocalRecall consumers expect
    let results: Vec<SearchResult> = memories
        .iter()
        .take(max)
        .map(|m| SearchResult {
            content: m.content.clone(),
            source: m.source.clone(),
            similarity: m.importance, // use importance as relevance proxy
            metadata: serde_json::json!({
                "summary": m.summary,
                "topics": m.topics,
                "entities": m.entities,
                "traits": m.traits,
                "edge_aware_answer": &result.answer,
            }),
        })
        .collect();

    Ok(Json(serde_json::json!({
        "results": results,
        "answer": result.answer,
        "edges": result.relevant_edges,
    })))
}

/// Reset (clear) a collection.
async fn reset_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<Json<serde_json::Value>, String> {
    state
        .store
        .clear_collection(&collection)
        .await
        .map_err(|e| e.to_string())?;

    info!("localrecall compat: reset collection '{collection}'");

    Ok(Json(serde_json::json!({
        "status": "ok",
        "collection": collection
    })))
}

/// Delete a specific entry from a collection.
async fn delete_entry(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<DeleteEntryReq>,
) -> Result<Json<serde_json::Value>, String> {
    let deleted = state
        .store
        .delete_entry(&collection, &req.entry)
        .await
        .map_err(|e| e.to_string())?;

    Ok(Json(serde_json::json!({
        "status": if deleted { "ok" } else { "not_found" },
        "entry": req.entry
    })))
}
