use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::error;

use crate::config::Config;
use crate::consolidate;
use crate::db::MemoryStore;
use crate::ingest;
use crate::llm::LlmClient;
use crate::query;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub store: MemoryStore,
    pub llm: LlmClient,
    pub config: Config,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/status", get(status))
        .route("/memories", get(list_memories))
        .route("/edges", get(list_edges))
        .route("/consolidations", get(list_consolidations))
        .route("/ingest", post(ingest_text))
        .route("/query", get(query_memory))
        .route("/consolidate", post(trigger_consolidate))
        .route("/delete", post(delete_memory))
        .route("/clear", post(clear_all))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// --- Handlers ---

async fn status(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, AppError> {
    let stats = state.store.stats().await?;
    Ok(Json(stats))
}

async fn list_memories(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let memories = state.store.all_memories().await?;
    Ok(Json(serde_json::json!({ "memories": memories })))
}

async fn list_edges(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let edges = state.store.all_edges().await?;
    Ok(Json(serde_json::json!({ "edges": edges })))
}

async fn list_consolidations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let consolidations = state.store.all_consolidations().await?;
    Ok(Json(serde_json::json!({ "consolidations": consolidations })))
}

#[derive(Deserialize)]
struct IngestRequest {
    text: String,
    source: Option<String>,
    collection: Option<String>,
}

async fn ingest_text(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let source = req.source.unwrap_or_else(|| "api".to_string());
    let collection = req.collection.unwrap_or_else(|| "default".to_string());
    let id = ingest::ingest_text(
        &state.store,
        &state.llm,
        &req.text,
        &source,
        &collection,
        state.config.traits.dimensions,
    )
    .await?;

    Ok(Json(serde_json::json!({
        "id": id,
        "status": if id > 0 { "ingested" } else { "duplicate" }
    })))
}

#[derive(Deserialize)]
struct QueryParams {
    q: String,
}

async fn query_memory(
    State(state): State<Arc<AppState>>,
    Query(params): Query<QueryParams>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = query::query(&state.store, &state.llm, &params.q).await?;
    Ok(Json(serde_json::json!(result)))
}

async fn trigger_consolidate(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let result = consolidate::consolidate(
        &state.store,
        &state.llm,
        state.config.consolidation.edge_threshold,
    )
    .await?;
    Ok(Json(serde_json::json!(result)))
}

#[derive(Deserialize)]
struct DeleteRequest {
    memory_id: i64,
}

async fn delete_memory(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let deleted = state.store.delete_memory(req.memory_id).await?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn clear_all(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    state.store.clear_all().await?;
    Ok(Json(serde_json::json!({ "status": "cleared" })))
}

// --- Error type for Axum ---

struct AppError(anyhow::Error);

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!("request error: {:?}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}
