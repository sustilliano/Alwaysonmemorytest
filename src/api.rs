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
use crate::flightscore;
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
        .route("/flightscore", get(flight_score))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// --- Handlers ---

async fn status(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, AppError> {
    let stats = state.store.stats().await?;
    Ok(Json(stats))
}

#[derive(Deserialize)]
struct PaginationParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize { 50 }

async fn list_memories(
    State(state): State<Arc<AppState>>,
    Query(page): Query<PaginationParams>,
) -> Result<Json<serde_json::Value>, AppError> {
    let memories = state.store.paginated_memories(page.limit, page.offset).await?;
    Ok(Json(serde_json::json!({ "memories": memories, "limit": page.limit, "offset": page.offset })))
}

async fn list_edges(
    State(state): State<Arc<AppState>>,
    Query(page): Query<PaginationParams>,
) -> Result<Json<serde_json::Value>, AppError> {
    let edges = state.store.paginated_edges(page.limit, page.offset).await?;
    Ok(Json(serde_json::json!({ "edges": edges, "limit": page.limit, "offset": page.offset })))
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
    let result = ingest::ingest_text(
        &state.store,
        &state.llm,
        &req.text,
        &source,
        &collection,
        state.config.traits.dimensions,
    )
    .await?;

    Ok(Json(serde_json::json!({
        "id": result.id,
        "status": if result.id > 0 { "ingested" } else { "duplicate" },
        "revision": result.revision,
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

#[derive(Deserialize)]
struct ClearRequest {
    confirm: Option<bool>,
}

async fn clear_all(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ClearRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if req.confirm != Some(true) {
        return Err(StatusCode::BAD_REQUEST);
    }
    state.store.clear_all().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({ "status": "cleared" })))
}

async fn flight_score(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let report = flightscore::score(&state.store).await?;
    Ok(Json(serde_json::to_value(report).map_err(anyhow::Error::from)?))
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
