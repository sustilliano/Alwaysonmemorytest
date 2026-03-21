use crate::db::Memory;
use crate::math::{jaccard_similarity, cosine_similarity};

/// Rgano edge detection applied to memory space.
///
/// Core formula: edge_score(x) = 1 - consensus(x, neighbors(x))
///
/// In memory space:
/// - A "neighbor" is a memory that shares topics or entities
/// - "Consensus" is measured by trait vector similarity
/// - High edge scores = interesting boundaries where related memories disagree
///   in character/tone/perspective — these are the most valuable consolidation targets

/// Compute the edge score between two memories.
/// Returns a value in [0, 1] where:
///   0 = perfect consensus (memories agree completely)
///   1 = maximum edge (memories that share context but diverge in character)
pub fn edge_score(a: &Memory, b: &Memory) -> f64 {
    let topic_overlap = jaccard_similarity(&a.topics, &b.topics);
    let entity_overlap = jaccard_similarity(&a.entities, &b.entities);
    let neighborhood = (topic_overlap + entity_overlap) / 2.0;

    // If memories aren't neighbors at all, edge score is 0
    // (no edge exists between unrelated memories)
    if neighborhood < 0.05 {
        return 0.0;
    }

    // Trait consensus: how similar are their trait vectors?
    let trait_consensus = if a.traits.is_empty() || b.traits.is_empty() {
        0.5 // default middle ground if traits unavailable
    } else {
        cosine_similarity(&a.traits, &b.traits).clamp(0.0, 1.0)
    };

    // Importance-weighted: edges between important memories matter more
    let importance_weight = (a.importance + b.importance) / 2.0;

    // Edge score = neighborhood strength * (1 - consensus) * importance
    // High neighborhood + low consensus = strong edge
    let raw_edge = neighborhood * (1.0 - trait_consensus);
    (raw_edge * importance_weight).clamp(0.0, 1.0)
}

/// Find the k-nearest neighbors of a memory by topic/entity overlap.
pub fn find_neighbors(target: &Memory, all: &[Memory], k: usize) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = all
        .iter()
        .enumerate()
        .filter(|(_, m)| m.id != target.id)
        .map(|(i, m)| {
            let topic_sim = jaccard_similarity(&target.topics, &m.topics);
            let entity_sim = jaccard_similarity(&target.entities, &m.entities);
            (i, (topic_sim + entity_sim) / 2.0)
        })
        .filter(|(_, sim)| *sim > 0.0)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Compute the aggregate edge score for a memory against all its neighbors.
/// This tells us how "edgy" a particular memory is in the knowledge space.
pub fn memory_edge_score(target: &Memory, all: &[Memory]) -> f64 {
    let neighbors = find_neighbors(target, all, 5);
    if neighbors.is_empty() {
        return 0.0;
    }

    let total: f64 = neighbors
        .iter()
        .map(|(i, _)| edge_score(target, &all[*i]))
        .sum();

    total / neighbors.len() as f64
}

/// Find all significant edges in the memory space.
/// Returns pairs of memory IDs with their edge scores, sorted by score descending.
pub fn detect_all_edges(memories: &[Memory], threshold: f64) -> Vec<(i64, i64, f64)> {
    let mut edges = Vec::new();

    for (i, a) in memories.iter().enumerate() {
        for b in memories.iter().skip(i + 1) {
            let score = edge_score(a, b);
            if score >= threshold {
                edges.push((a.id, b.id, score));
            }
        }
    }

    edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(id: i64, topics: &[&str], traits: &[f64], importance: f64) -> Memory {
        Memory {
            id,
            content: String::new(),
            summary: String::new(),
            source: String::new(),
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
    fn unrelated_memories_have_zero_edge() {
        let a = make_memory(1, &["rust", "programming"], &[1.0, 0.0], 0.8);
        let b = make_memory(2, &["cooking", "recipes"], &[0.0, 1.0], 0.8);
        assert_eq!(edge_score(&a, &b), 0.0);
    }

    #[test]
    fn related_but_different_have_high_edge() {
        let a = make_memory(1, &["ai", "memory", "agents"], &[1.0, 0.0, 0.5], 0.9);
        let b = make_memory(2, &["ai", "memory", "databases"], &[0.0, 1.0, 0.2], 0.9);
        let score = edge_score(&a, &b);
        assert!(score > 0.2, "expected high edge, got {score}");
    }

    #[test]
    fn identical_traits_have_low_edge() {
        let a = make_memory(1, &["ai", "memory"], &[1.0, 0.5, 0.3], 0.8);
        let b = make_memory(2, &["ai", "memory"], &[1.0, 0.5, 0.3], 0.8);
        let score = edge_score(&a, &b);
        assert!(score < 0.05, "expected low edge, got {score}");
    }
}
