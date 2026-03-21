use anyhow::Result;
use tracing::info;

use crate::db::MemoryStore;
use crate::llm::LlmClient;

/// Query the memory store with a natural language question.
/// Unlike flat retrieval, this is topology-aware:
/// it knows about edges and consolidation insights.
pub async fn query(
    store: &MemoryStore,
    llm: &LlmClient,
    question: &str,
) -> Result<QueryResult> {
    let memories = store.all_memories().await?;
    let consolidations = store.all_consolidations().await?;
    let edges = store.all_edges().await?;

    if memories.is_empty() {
        return Ok(QueryResult {
            answer: "No memories stored yet. Feed me some information first!".to_string(),
            sources: vec![],
            relevant_edges: vec![],
        });
    }

    // Build context with all available knowledge
    let mut context = String::from("## Stored Memories\n");
    for m in &memories {
        context.push_str(&format!(
            "Memory #{} [topics: {}] [importance: {:.1}]: {}\n",
            m.id,
            m.topics.join(", "),
            m.importance,
            m.summary,
        ));
    }

    if !consolidations.is_empty() {
        context.push_str("\n## Consolidation Insights\n");
        for c in &consolidations {
            context.push_str(&format!(
                "Insight (from memories {:?}, edge_score={:.3}): {}\n",
                c.memory_ids, c.edge_score, c.insight
            ));
        }
    }

    if !edges.is_empty() {
        context.push_str("\n## Knowledge Boundaries (edges)\n");
        for e in edges.iter().take(15) {
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
        sources: memories.iter().map(|m| m.id).collect(),
        relevant_edges: edges
            .iter()
            .take(5)
            .map(|e| (e.memory_a, e.memory_b, e.edge_score))
            .collect(),
    })
}

#[derive(Debug, serde::Serialize)]
pub struct QueryResult {
    pub answer: String,
    pub sources: Vec<i64>,
    pub relevant_edges: Vec<(i64, i64, f64)>,
}
