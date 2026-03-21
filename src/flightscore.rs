//! Rgano-powered flight score for the memory knowledge graph.
//!
//! Applies the core Rgano formula to every memory in the store:
//!
//!   edge_score(x) = neighborhood(x, neighbors) × (1 − trait_consensus(x, neighbors))
//!
//! where:
//!   neighborhood = average Jaccard similarity of topics + entities with k-nearest neighbors
//!   trait_consensus = cosine similarity of trait vectors with those neighbors
//!
//! Each memory is then classified by its edge score:
//!   Landed  — edge_score ≤ 0.30, has ≥ 1 neighbor → stable, well-integrated knowledge
//!   Falling — 0.30 < edge_score ≤ 0.60           → active revision zone, in flux
//!   Crashed — edge_score > 0.60 OR no neighbors   → frontier / isolated / fragmented
//!
//! The aggregate of all per-memory scores produces a graph-level FlightStatus:
//!   Landed  — avg_consensus ≥ 0.70 → knowledge is converged
//!   Falling — avg_consensus ≥ 0.40 → knowledge is in active revision
//!   Crashed — avg_consensus < 0.40 → knowledge is fragmented or too sparse

use anyhow::Result;
use serde::Serialize;

use crate::db::{Memory, MemoryStore};
use crate::edges::{find_neighbors, memory_edge_score};
use crate::math::jaccard_similarity;
use crate::revision::FlightStatus;

/// Edge score thresholds for per-memory classification.
const LANDED_EDGE_MAX: f64 = 0.30;
const FALLING_EDGE_MAX: f64 = 0.60;

/// Average consensus thresholds for graph-level classification.
const GRAPH_LANDED_MIN: f64 = 0.70;
const GRAPH_FALLING_MIN: f64 = 0.40;

/// Per-memory Rgano flight classification.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryScore {
    pub id: i64,
    pub source: String,
    pub collection: String,
    pub topics: Vec<String>,
    /// Rgano edge score: high = diverges from neighbors, low = consensus with neighbors.
    pub edge_score: f64,
    /// Consensus level = 1 - edge_score.
    pub consensus: f64,
    /// Number of topically related neighbors found (neighborhood > 0.05).
    pub neighbor_count: usize,
    /// Flight classification based on edge_score and connectivity.
    pub flight_status: FlightStatus,
    pub importance: f64,
}

/// Graph connectivity and topology statistics.
#[derive(Debug, Clone, Serialize)]
pub struct TopologyStats {
    /// Memories with at least one neighbor (connected nodes in the graph).
    pub connected: usize,
    /// Memories with no neighbors (isolated nodes — knowledge islands).
    pub isolated: usize,
    /// Total neighbor-pairs above the minimum neighborhood threshold (0.05).
    pub edge_pairs: usize,
    /// Average neighbor count across connected memories.
    pub avg_neighbors: f64,
}

/// Full Rgano flight score report for the knowledge graph.
#[derive(Debug, Serialize)]
pub struct FlightScoreReport {
    pub total_memories: usize,
    /// Stable memories: low divergence, integrated into the graph.
    pub landed: Vec<MemoryScore>,
    /// Revision-zone memories: moderate divergence from neighbors.
    pub falling: Vec<MemoryScore>,
    /// Frontier memories: isolated OR extreme divergence — novel or fragmented.
    pub crashed: Vec<MemoryScore>,
    /// Average Rgano edge score across all memories (0 = full consensus, 1 = full divergence).
    pub avg_edge_score: f64,
    /// Average consensus level across all memories (1 - avg_edge_score).
    pub avg_consensus: f64,
    /// Graph-level FlightStatus derived from average consensus.
    pub overall_status: FlightStatus,
    /// Knowledge graph topology.
    pub topology: TopologyStats,
    /// Top 5 most "edgy" memories — highest divergence from their neighbor cluster.
    pub hottest_edges: Vec<MemoryScore>,
}

/// Compute the Rgano flight score for the entire memory store.
pub async fn score(store: &MemoryStore) -> Result<FlightScoreReport> {
    let memories = store.all_memories().await?;

    if memories.is_empty() {
        return Ok(FlightScoreReport {
            total_memories: 0,
            landed: vec![],
            falling: vec![],
            crashed: vec![],
            avg_edge_score: 0.0,
            avg_consensus: 1.0,
            overall_status: FlightStatus::Crashed,
            topology: TopologyStats {
                connected: 0,
                isolated: 0,
                edge_pairs: 0,
                avg_neighbors: 0.0,
            },
            hottest_edges: vec![],
        });
    }

    let total = memories.len();
    let mut scored: Vec<MemoryScore> = Vec::with_capacity(total);
    let mut total_edge = 0.0_f64;
    let mut connected_count = 0_usize;
    let mut total_neighbor_sum = 0_usize;

    for mem in &memories {
        let neighbors = find_neighbors(mem, &memories, 10);
        let neighbor_count = neighbors.len();

        // Isolated memories get edge_score=0 — no neighbors means no divergence measurable.
        let edge = if neighbor_count == 0 {
            0.0
        } else {
            memory_edge_score(mem, &memories)
        };

        let flight_status = classify(edge, neighbor_count);

        if neighbor_count > 0 {
            connected_count += 1;
            total_neighbor_sum += neighbor_count;
        }
        total_edge += edge;

        scored.push(MemoryScore {
            id: mem.id,
            source: mem.source.clone(),
            collection: mem.collection.clone(),
            topics: mem.topics.clone(),
            edge_score: edge,
            consensus: (1.0 - edge).clamp(0.0, 1.0),
            neighbor_count,
            flight_status,
            importance: mem.importance,
        });
    }

    let avg_edge_score = total_edge / total as f64;
    let avg_consensus = (1.0 - avg_edge_score).clamp(0.0, 1.0);

    let overall_status = if avg_consensus >= GRAPH_LANDED_MIN {
        FlightStatus::Landed
    } else if avg_consensus >= GRAPH_FALLING_MIN {
        FlightStatus::Falling
    } else {
        FlightStatus::Crashed
    };

    let avg_neighbors = if connected_count > 0 {
        total_neighbor_sum as f64 / connected_count as f64
    } else {
        0.0
    };

    let topology = TopologyStats {
        connected: connected_count,
        isolated: total - connected_count,
        edge_pairs: count_edge_pairs(&memories),
        avg_neighbors,
    };

    // Top-5 hottest edges — sorted by edge_score descending.
    let mut for_hot = scored.clone();
    for_hot.sort_by(|a, b| {
        b.edge_score
            .partial_cmp(&a.edge_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let hottest_edges: Vec<MemoryScore> = for_hot.into_iter().take(5).collect();

    let mut landed = vec![];
    let mut falling = vec![];
    let mut crashed = vec![];

    for s in scored {
        match s.flight_status {
            FlightStatus::Landed => landed.push(s),
            FlightStatus::Falling => falling.push(s),
            FlightStatus::Crashed => crashed.push(s),
        }
    }

    // Landed: most stable first (highest consensus).
    landed.sort_by(|a, b| {
        b.consensus
            .partial_cmp(&a.consensus)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Falling: most divergent first.
    falling.sort_by(|a, b| {
        b.edge_score
            .partial_cmp(&a.edge_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Crashed: most important isolated memories first.
    crashed.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(FlightScoreReport {
        total_memories: total,
        landed,
        falling,
        crashed,
        avg_edge_score,
        avg_consensus,
        overall_status,
        topology,
        hottest_edges,
    })
}

fn classify(edge_score: f64, neighbor_count: usize) -> FlightStatus {
    if neighbor_count == 0 {
        FlightStatus::Crashed
    } else if edge_score <= LANDED_EDGE_MAX {
        FlightStatus::Landed
    } else if edge_score <= FALLING_EDGE_MAX {
        FlightStatus::Falling
    } else {
        FlightStatus::Crashed
    }
}

/// Count distinct memory pairs that are actual neighbors (neighborhood > 0.05).
/// This is the number of real Rgano edges in the knowledge graph.
fn count_edge_pairs(memories: &[Memory]) -> usize {
    let mut count = 0;
    for (i, a) in memories.iter().enumerate() {
        for b in memories.iter().skip(i + 1) {
            let topic_sim = jaccard_similarity(&a.topics, &b.topics);
            let entity_sim = jaccard_similarity(&a.entities, &b.entities);
            if (topic_sim + entity_sim) / 2.0 > 0.05 {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: i64, topics: &[&str], traits: &[f64], importance: f64) -> Memory {
        Memory {
            id,
            collection: "default".to_string(),
            content: String::new(),
            summary: String::new(),
            source: format!("source-{id}"),
            file_hash: String::new(),
            importance,
            traits: traits.to_vec(),
            topics: topics.iter().map(|s| s.to_string()).collect(),
            entities: vec![],
            created_at: String::new(),
            consolidated: false,
        }
    }

    #[test]
    fn classify_isolated_is_crashed() {
        assert_eq!(classify(0.0, 0), FlightStatus::Crashed);
    }

    #[test]
    fn classify_low_edge_is_landed() {
        assert_eq!(classify(0.1, 3), FlightStatus::Landed);
        assert_eq!(classify(0.3, 1), FlightStatus::Landed);
    }

    #[test]
    fn classify_mid_edge_is_falling() {
        assert_eq!(classify(0.31, 2), FlightStatus::Falling);
        assert_eq!(classify(0.6, 1), FlightStatus::Falling);
    }

    #[test]
    fn classify_high_edge_is_crashed() {
        assert_eq!(classify(0.61, 2), FlightStatus::Crashed);
        assert_eq!(classify(1.0, 5), FlightStatus::Crashed);
    }

    #[test]
    fn count_edge_pairs_unrelated_topics() {
        let a = mem(1, &["rust", "programming"], &[], 0.5);
        let b = mem(2, &["cooking", "recipes"], &[], 0.5);
        assert_eq!(count_edge_pairs(&[a, b]), 0);
    }

    #[test]
    fn count_edge_pairs_shared_topics() {
        let a = mem(1, &["ai", "memory"], &[], 0.5);
        let b = mem(2, &["ai", "agents"], &[], 0.5);
        let c = mem(3, &["cooking", "recipes"], &[], 0.5);
        // a-b share "ai" → count=1; a-c and b-c share nothing → count stays 1
        assert_eq!(count_edge_pairs(&[a, b, c]), 1);
    }
}
