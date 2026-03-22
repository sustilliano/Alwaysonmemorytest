use anyhow::Result;
use tracing::info;

use crate::db::{Memory, MemoryStore};
use crate::llm::LlmClient;

// Maximum items to include in the LLM prompt context
const MAX_MEMORIES: usize = 20;
const MAX_CONSOLIDATIONS: usize = 10;
const MAX_EDGES: usize = 15;

/// Query the memory store with a natural language question.
/// Unlike flat retrieval, this is topology-aware:
/// it knows about edges and consolidation insights.
pub async fn query(
    store: &MemoryStore,
    llm: &LlmClient,
    question: &str,
) -> Result<QueryResult> {
    let all_memories = store.all_memories().await?;
    let consolidations = store.all_consolidations().await?;
    let edges = store.all_edges().await?;

    if all_memories.is_empty() {
        return Ok(QueryResult {
            answer: "No memories stored yet. Feed me some information first!".to_string(),
            sources: vec![],
            relevant_edges: vec![],
        });
    }

    // Rank memories by topic keyword overlap with the query
    let relevant_memories = top_k_memories(&all_memories, question, MAX_MEMORIES);
    let relevant_ids: std::collections::HashSet<i64> =
        relevant_memories.iter().map(|m| m.id).collect();

    // Build context with top-K knowledge
    let mut context = String::from("## Stored Memories\n");
    for m in &relevant_memories {
        context.push_str(&format!(
            "Memory #{} [topics: {}] [importance: {:.1}]: {}\n",
            m.id,
            m.topics.join(", "),
            m.importance,
            m.summary,
        ));
    }

    let top_consolidations: Vec<_> = consolidations.iter().take(MAX_CONSOLIDATIONS).collect();
    if !top_consolidations.is_empty() {
        context.push_str("\n## Consolidation Insights\n");
        for c in &top_consolidations {
            context.push_str(&format!(
                "Insight (from memories {:?}, edge_score={:.3}): {}\n",
                c.memory_ids, c.edge_score, c.insight
            ));
        }
    }

    // Only show edges that touch memories relevant to this query.
    let relevant_edges: Vec<_> = edges
        .iter()
        .filter(|e| relevant_ids.contains(&e.memory_a) || relevant_ids.contains(&e.memory_b))
        .take(MAX_EDGES)
        .collect();
    if !relevant_edges.is_empty() {
        context.push_str("\n## Knowledge Boundaries (edges)\n");
        for e in &relevant_edges {
            context.push_str(&format!(
                "Edge: Memory #{} <-> Memory #{} (score={:.3}){}\n",
                e.memory_a,
                e.memory_b,
                e.edge_score,
                if e.relationship.is_empty() {
                    String::new()
                } else {
                    format!(": {}", e.relationship)
                },
            ));
        }
    }

    let prompt = format!(
        r#"You are a memory query agent with access to a structured knowledge store.
You know not just facts, but the TOPOLOGY of knowledge — where ideas connect
and where they diverge (edges). Use this to give nuanced, insightful answers.

When answering:
- Cite memory IDs for specific claims
- Highlight relevant edges (contradictions/tensions) when they inform the answer
- Distinguish between confident knowledge and speculative connections
- Be direct and useful

{context}

Question: {question}

Answer:"#
    );

    let answer = llm.complete(&prompt).await?;

    info!(question, "query answered");

    Ok(QueryResult {
        answer,
        sources: relevant_memories.iter().map(|m| m.id).collect(),
        relevant_edges: relevant_edges
            .iter()
            .take(5)
            .map(|e| (e.memory_a, e.memory_b, e.edge_score))
            .collect(),
    })
}

/// Return up to `k` memories most relevant to the query by topic keyword overlap.
/// Falls back to ordering by importance when no topic keywords match.
fn top_k_memories<'a>(memories: &'a [Memory], query: &str, k: usize) -> Vec<&'a Memory> {
    let query_words: std::collections::HashSet<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    let mut scored: Vec<(&Memory, f64)> = memories
        .iter()
        .map(|m| {
            let topic_matches = m.topics.iter()
                .filter(|t| query_words.contains(&t.to_lowercase()))
                .count() as f64;
            // Blend topic overlap with importance so high-importance memories aren't buried
            let score = topic_matches + m.importance * 0.3;
            (m, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(m, _)| m).collect()
}

#[derive(Debug, serde::Serialize)]
pub struct QueryResult {
    pub answer: String,
    pub sources: Vec<i64>,
    pub relevant_edges: Vec<(i64, i64, f64)>,
}
