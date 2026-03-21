use anyhow::{Context, Result};
use duckdb::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A single memory unit extracted from ingested content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: i64,
    pub collection: String,
    pub content: String,
    pub summary: String,
    pub source: String,
    pub file_hash: String,
    pub importance: f64,
    pub traits: Vec<f64>,
    pub topics: Vec<String>,
    pub entities: Vec<String>,
    pub created_at: String,
    pub consolidated: bool,
}

/// A consolidation insight linking multiple memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consolidation {
    pub id: i64,
    pub memory_ids: Vec<i64>,
    pub insight: String,
    pub edge_score: f64,
    pub connections: Vec<Connection_>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection_ {
    pub from_id: i64,
    pub to_id: i64,
    pub relationship: String,
    pub strength: f64,
}

/// An edge between two memories detected by Rgano consensus analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: i64,
    pub memory_a: i64,
    pub memory_b: i64,
    pub edge_score: f64,
    pub relationship: String,
    pub created_at: String,
}

/// Thread-safe handle to the DuckDB store.
#[derive(Clone)]
pub struct MemoryStore {
    conn: Arc<RwLock<Connection>>,
}

impl MemoryStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn =
            Connection::open(path).with_context(|| format!("failed to open db at {path:?}"))?;
        let store = Self {
            conn: Arc::new(RwLock::new(conn)),
        };
        store.init_schema_blocking()?;
        Ok(store)
    }

    fn init_schema_blocking(&self) -> Result<()> {
        // We're still in sync init, so use try_write
        let conn = self.conn.try_write().expect("lock during init");
        conn.execute_batch(
            "
            CREATE SEQUENCE IF NOT EXISTS mem_seq START 1;
            CREATE TABLE IF NOT EXISTS memories (
                id              BIGINT DEFAULT nextval('mem_seq') PRIMARY KEY,
                collection      TEXT NOT NULL DEFAULT 'default',
                content         TEXT NOT NULL,
                summary         TEXT NOT NULL DEFAULT '',
                source          TEXT NOT NULL DEFAULT '',
                file_hash       TEXT NOT NULL DEFAULT '',
                importance      DOUBLE DEFAULT 0.5,
                traits          TEXT DEFAULT '[]',
                topics          TEXT DEFAULT '[]',
                entities        TEXT DEFAULT '[]',
                created_at      TIMESTAMP DEFAULT current_timestamp,
                consolidated    BOOLEAN DEFAULT false
            );

            CREATE SEQUENCE IF NOT EXISTS cons_seq START 1;
            CREATE TABLE IF NOT EXISTS consolidations (
                id              BIGINT DEFAULT nextval('cons_seq') PRIMARY KEY,
                memory_ids      TEXT NOT NULL DEFAULT '[]',
                insight         TEXT NOT NULL,
                edge_score      DOUBLE DEFAULT 0.0,
                connections     TEXT DEFAULT '[]',
                created_at      TIMESTAMP DEFAULT current_timestamp
            );

            CREATE SEQUENCE IF NOT EXISTS edge_seq START 1;
            CREATE TABLE IF NOT EXISTS edges (
                id              BIGINT DEFAULT nextval('edge_seq') PRIMARY KEY,
                memory_a        BIGINT NOT NULL,
                memory_b        BIGINT NOT NULL,
                edge_score      DOUBLE NOT NULL,
                relationship    TEXT DEFAULT '',
                created_at      TIMESTAMP DEFAULT current_timestamp
            );
            ",
        )
        .with_context(|| "failed to init schema")?;
        Ok(())
    }

    /// Insert a new memory, returning its ID.
    pub async fn insert_memory(
        &self,
        collection: &str,
        content: &str,
        summary: &str,
        source: &str,
        file_hash: &str,
        importance: f64,
        traits: &[f64],
        topics: &[String],
        entities: &[String],
    ) -> Result<i64> {
        let conn = self.conn.write().await;
        let traits_json = serde_json::to_string(traits)?;
        let topics_json = serde_json::to_string(topics)?;
        let entities_json = serde_json::to_string(entities)?;

        let id: i64 = conn.query_row(
            "INSERT INTO memories (collection, content, summary, source, file_hash, importance, traits, topics, entities)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
             RETURNING id",
            params![collection, content, summary, source, file_hash, importance, traits_json, topics_json, entities_json],
            |row| row.get(0),
        ).with_context(|| "failed to insert memory")?;

        Ok(id)
    }

    /// Check if a file has already been ingested by its hash.
    pub async fn has_file_hash(&self, hash: &str) -> Result<bool> {
        let conn = self.conn.read().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE file_hash = ?",
            params![hash],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get all unconsolidated memories.
    pub async fn unconsolidated_memories(&self) -> Result<Vec<Memory>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, collection, content, summary, source, file_hash, importance, traits, topics, entities,
                    created_at::TEXT, consolidated
             FROM memories WHERE consolidated = false ORDER BY created_at ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    collection: row.get(1)?,
                    content: row.get(2)?,
                    summary: row.get(3)?,
                    source: row.get(4)?,
                    file_hash: row.get(5)?,
                    importance: row.get(6)?,
                    traits: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                    topics: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                    entities: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                    created_at: row.get(10)?,
                    consolidated: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get all memories.
    pub async fn all_memories(&self) -> Result<Vec<Memory>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, collection, content, summary, source, file_hash, importance, traits, topics, entities,
                    created_at::TEXT, consolidated
             FROM memories ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    collection: row.get(1)?,
                    content: row.get(2)?,
                    summary: row.get(3)?,
                    source: row.get(4)?,
                    file_hash: row.get(5)?,
                    importance: row.get(6)?,
                    traits: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                    topics: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                    entities: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                    created_at: row.get(10)?,
                    consolidated: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get memories by collection.
    pub async fn memories_by_collection(&self, collection: &str) -> Result<Vec<Memory>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, collection, content, summary, source, file_hash, importance, traits, topics, entities,
                    created_at::TEXT, consolidated
             FROM memories WHERE collection = ? ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map(params![collection], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    collection: row.get(1)?,
                    content: row.get(2)?,
                    summary: row.get(3)?,
                    source: row.get(4)?,
                    file_hash: row.get(5)?,
                    importance: row.get(6)?,
                    traits: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                    topics: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                    entities: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                    created_at: row.get(10)?,
                    consolidated: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// List all distinct collections.
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare("SELECT DISTINCT collection FROM memories ORDER BY collection")?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        Ok(rows)
    }

    /// Delete all memories in a collection.
    pub async fn clear_collection(&self, collection: &str) -> Result<()> {
        let conn = self.conn.write().await;
        conn.execute("DELETE FROM memories WHERE collection = ?", params![collection])?;
        Ok(())
    }

    /// Delete a memory by source name within a collection.
    pub async fn delete_entry(&self, collection: &str, source: &str) -> Result<bool> {
        let conn = self.conn.write().await;
        let affected = conn.execute(
            "DELETE FROM memories WHERE collection = ? AND source = ?",
            params![collection, source],
        )?;
        Ok(affected > 0)
    }

    /// Mark memories as consolidated.
    pub async fn mark_consolidated(&self, ids: &[i64]) -> Result<()> {
        let conn = self.conn.write().await;
        for id in ids {
            conn.execute(
                "UPDATE memories SET consolidated = true WHERE id = ?",
                params![id],
            )?;
        }
        Ok(())
    }

    /// Insert a consolidation record.
    pub async fn insert_consolidation(
        &self,
        memory_ids: &[i64],
        insight: &str,
        edge_score: f64,
        connections: &[Connection_],
    ) -> Result<i64> {
        let conn = self.conn.write().await;
        let ids_json = serde_json::to_string(memory_ids)?;
        let conn_json = serde_json::to_string(connections)?;

        let id: i64 = conn.query_row(
            "INSERT INTO consolidations (memory_ids, insight, edge_score, connections)
             VALUES (?, ?, ?, ?)
             RETURNING id",
            params![ids_json, insight, edge_score, conn_json],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Insert an edge between two memories.
    pub async fn insert_edge(
        &self,
        memory_a: i64,
        memory_b: i64,
        edge_score: f64,
        relationship: &str,
    ) -> Result<i64> {
        let conn = self.conn.write().await;
        let id: i64 = conn.query_row(
            "INSERT INTO edges (memory_a, memory_b, edge_score, relationship)
             VALUES (?, ?, ?, ?)
             RETURNING id",
            params![memory_a, memory_b, edge_score, relationship],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Get all consolidation insights.
    pub async fn all_consolidations(&self) -> Result<Vec<Consolidation>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, memory_ids, insight, edge_score, connections, created_at::TEXT
             FROM consolidations ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Consolidation {
                    id: row.get(0)?,
                    memory_ids: serde_json::from_str(&row.get::<_, String>(1)?)
                        .unwrap_or_default(),
                    insight: row.get(2)?,
                    edge_score: row.get(3)?,
                    connections: serde_json::from_str(&row.get::<_, String>(4)?)
                        .unwrap_or_default(),
                    created_at: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get all edges.
    pub async fn all_edges(&self) -> Result<Vec<Edge>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, memory_a, memory_b, edge_score, relationship, created_at::TEXT
             FROM edges ORDER BY edge_score DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Edge {
                    id: row.get(0)?,
                    memory_a: row.get(1)?,
                    memory_b: row.get(2)?,
                    edge_score: row.get(3)?,
                    relationship: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get a paginated page of memories.
    pub async fn paginated_memories(&self, limit: usize, offset: usize) -> Result<Vec<Memory>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, collection, content, summary, source, file_hash, importance, traits, topics, entities,
                    created_at::TEXT, consolidated
             FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
        )?;
        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok(Memory {
                    id: row.get(0)?,
                    collection: row.get(1)?,
                    content: row.get(2)?,
                    summary: row.get(3)?,
                    source: row.get(4)?,
                    file_hash: row.get(5)?,
                    importance: row.get(6)?,
                    traits: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                    topics: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                    entities: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or_default(),
                    created_at: row.get(10)?,
                    consolidated: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Get a paginated page of edges.
    pub async fn paginated_edges(&self, limit: usize, offset: usize) -> Result<Vec<Edge>> {
        let conn = self.conn.read().await;
        let mut stmt = conn.prepare(
            "SELECT id, memory_a, memory_b, edge_score, relationship, created_at::TEXT
             FROM edges ORDER BY edge_score DESC LIMIT ? OFFSET ?",
        )?;
        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok(Edge {
                    id: row.get(0)?,
                    memory_a: row.get(1)?,
                    memory_b: row.get(2)?,
                    edge_score: row.get(3)?,
                    relationship: row.get(4)?,
                    created_at: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Memory count and stats.
    pub async fn stats(&self) -> Result<serde_json::Value> {
        let conn = self.conn.read().await;
        let total: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
        let unconsolidated: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidated = false",
            [],
            |r| r.get(0),
        )?;
        let consolidations: i64 =
            conn.query_row("SELECT COUNT(*) FROM consolidations", [], |r| r.get(0))?;
        let edges: i64 =
            conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;

        Ok(serde_json::json!({
            "total_memories": total,
            "unconsolidated": unconsolidated,
            "consolidations": consolidations,
            "edges": edges
        }))
    }

    /// Delete a memory by ID.
    pub async fn delete_memory(&self, id: i64) -> Result<bool> {
        let conn = self.conn.write().await;
        let affected = conn.execute("DELETE FROM memories WHERE id = ?", params![id])?;
        Ok(affected > 0)
    }

    /// Clear everything.
    pub async fn clear_all(&self) -> Result<()> {
        let conn = self.conn.write().await;
        conn.execute_batch(
            "DELETE FROM edges; DELETE FROM consolidations; DELETE FROM memories;",
        )?;
        Ok(())
    }
}
