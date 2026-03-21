use anyhow::Result;
use tracing::{info, warn};

use crate::db::{Connection_, MemoryStore, Memory};
use crate::edges;
use crate::llm::LlmClient;

/// Run a single consolidation cycle.
///
/// 1. Fetch unconsolidated memories
/// 2. Detect edges (interesting boundaries) between them
/// 3. Ask the LLM to find cross-cutting insights
/// 4. Store consolidation records and edge data
/// 5. Mark memories as consolidated
pub async fn consolidate(
    store: &MemoryStore,
    llm: &LlmClient,
    edge_threshold: f64,
) -> Result<ConsolidationResult> {
    let unconsolidated = store.unconsolidated_memories().await?;

    if unconsolidated.len() < 2 {
        info!("not enough unconsolidated memories to consolidate ({})", unconsolidated.len());
        return Ok(ConsolidationResult::default());
    }

    info!(
        count = unconsolidated.len(),
        "starting consolidation cycle"
    );

    // Also grab all memories for full-graph edge detection
    let all_memories = store.all_memories().await?;

    // 1. Detect edges between the new memories and everything
    let new_edges = edges::detect_all_edges(&all_memories, edge_threshold);
    info!(edge_count = new_edges.len(), "detected edges");

    // Store edges
    for (a, b, score) in &new_edges {
        // Generate a brief relationship description for significant edges
        let relationship = if *score > 0.6 {
            describe_edge(llm, &all_memories, *a, *b).await.unwrap_or_default()
        } else {
            String::new()
        };

        if let Err(e) = store.insert_edge(*a, *b, *score, &relationship).await {
            warn!("failed to insert edge {a}<->{b}: {e}");
        }
    }

    // 2. Build the consolidation prompt with edge-aware context
    let insight = generate_insight(llm, &unconsolidated, &new_edges).await?;

    // 3. Build connection records from the strongest edges
    let connections: Vec<Connection_> = new_edges
        .iter()
        .take(10)
        .map(|(a, b, score)| Connection_ {
            from_id: *a,
            to_id: *b,
            relationship: format!("edge_score={score:.3}"),
            strength: *score,
        })
        .collect();

    // 4. Compute aggregate edge score for this consolidation
    let avg_edge = if new_edges.is_empty() {
        0.0
    } else {
        new_edges.iter().map(|(_, _, s)| s).sum::<f64>() / new_edges.len() as f64
    };

    // 5. Store the consolidation
    let memory_ids: Vec<i64> = unconsolidated.iter().map(|m| m.id).collect();
    let cons_id = store
        .insert_consolidation(&memory_ids, &insight, avg_edge, &connections)
        .await?;

    // 6. Mark as consolidated
    store.mark_consolidated(&memory_ids).await?;

    info!(
        consolidation_id = cons_id,
        memories = memory_ids.len(),
        edges = new_edges.len(),
        avg_edge_score = format!("{avg_edge:.3}"),
        "consolidation complete"
    );

    Ok(ConsolidationResult {
        consolidation_id: cons_id,
        memories_processed: memory_ids.len(),
        edges_detected: new_edges.len(),
        avg_edge_score: avg_edge,
        insight,
    })
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ConsolidationResult {
    pub consolidation_id: i64,
    pub memories_processed: usize,
    pub edges_detected: usize,
    pub avg_edge_score: f64,
    pub insight: String,
}

/// Ask the LLM to synthesize insights from memories, with edge awareness.
async fn generate_insight(
    llm: &LlmClient,
    memories: &[Memory],
    detected_edges: &[(i64, i64, f64)],
) -> Result<String> {
    let mut context = String::new();

    for m in memories {
        context.push_str(&format!(
            "Memory #{}: [topics: {}] [importance: {:.1}] {}\n",
            m.id,
            m.topics.join(", "),
            m.importance,
            m.summary
        ));
    }

    // Include edge information so the LLM knows where the interesting boundaries are
    if !detected_edges.is_empty() {
        context.push_str("\nDetected knowledge boundaries (high edge scores = related but divergent):\n");
        for (a, b, score) in detected_edges.iter().take(10) {
            context.push_str(&format!("  Memory #{a} <-> Memory #{b}: edge_score={score:.3}\n"));
        }
    }

    let prompt = format!(
        r#"You are a memory consolidation agent. Your job is like the brain during sleep:
review memories, find connections, compress related information, and generate insights.

Pay special attention to "edges" — places where related memories DIVERGE in perspective
or approach. These boundaries are where the most interesting insights live.

Memories to consolidate:
{context}

Produce a concise consolidation report:
1. Key connections between memories
2. Cross-cutting themes
3. Tensions or contradictions (high-edge-score pairs)
4. One actionable insight that emerges from the synthesis

Be direct and specific. Reference memory IDs."#
    );

    llm.complete(&prompt).await
}

/// Ask the LLM to briefly describe the relationship at an edge.
async fn describe_edge(
    llm: &LlmClient,
    all: &[Memory],
    id_a: i64,
    id_b: i64,
) -> Result<String> {
    let a = all.iter().find(|m| m.id == id_a);
    let b = all.iter().find(|m| m.id == id_b);

    let (a, b) = match (a, b) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(String::new()),
    };

    let prompt = format!(
        r#"In one sentence, describe the relationship between these two pieces of information.
Focus on where they connect AND where they diverge.

A: {}
B: {}

One sentence:"#,
        a.summary, b.summary
    );

    llm.complete(&prompt).await
}
