//! File revision detection via extensible QX regime cascade.
//!
//! Each regime is a comparison dimension. The cascade walks regimes in priority
//! order until one fires, but ALL regimes contribute to the total score vector —
//! producing an N-dimensional revision fingerprint.
//!
//! Core regimes (always present):
//!   1. Content   — structural similarity (word overlap, length ratio)
//!   2. Semantic  — topic/entity overlap (Jaccard + importance delta)
//!   3. Trait     — character vector consensus (cosine similarity)
//!   4. Temporal  — recency/cadence clustering (ingestion time proximity)
//!   5. Provenance — source path/author similarity
//!   6. Narrative — summary/description similarity
//!   7. Knowledge — shared entity graph overlap
//!   8. Structural — topic hierarchy alignment
//!   9. Consensus — multi-signal agreement (meta-regime)
//!
//! Domain regimes (registered by adapters):
//!   - NFL/Sports: archetype fit, combine measurables, scout consensus
//!   - Code: AST similarity, import graph overlap, call graph alignment
//!   - Research: citation overlap, methodology similarity, claim alignment
//!
//! The regime count IS the dimensionality. 9 core = 9D. Add 3 NFL = 12D.

use crate::db::Memory;
use crate::math::{jaccard_similarity, jaccard_set, cosine_similarity};
use serde::Serialize;
use std::collections::HashMap;

// -- Regime trait --

pub trait Regime: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64;
    fn threshold(&self) -> f64;
    fn weight(&self) -> f64;
    fn next_on_fail(&self) -> Option<&str>;
}

// -- Cascade results --

#[derive(Debug, Clone, Serialize)]
pub struct RevisionResult {
    pub match_id: Option<i64>,
    pub classification: Classification,
    pub flight_status: FlightStatus,
    pub regime_scores: HashMap<String, f64>,
    pub total_score: f64,
    pub fired_regime: Option<String>,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum Classification {
    StrongMatch,
    SemanticMatch,
    DriftMatch,
    Novel,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum FlightStatus {
    Landed,
    Falling,
    Crashed,
}

// -- Incoming content --

pub struct IncomingContent {
    pub content: String,
    pub summary: String,
    pub source: String,
    pub topics: Vec<String>,
    pub entities: Vec<String>,
    pub traits: Vec<f64>,
    pub importance: f64,
    pub created_at: Option<String>,
    pub domain_meta: HashMap<String, serde_json::Value>,
}

// -- Cascade engine --

pub struct CascadeEngine {
    regimes: Vec<Box<dyn Regime>>,
    strong_floor: f64,
    semantic_floor: f64,
    drift_floor: f64,
}

impl CascadeEngine {
    pub fn new() -> Self {
        Self {
            regimes: Vec::new(),
            strong_floor: 0.85,
            semantic_floor: 0.55,
            drift_floor: 0.35,
        }
    }

    pub fn with_core_regimes() -> Self {
        let mut engine = Self::new();
        engine.register(Box::new(ContentRegime::default()));
        engine.register(Box::new(SemanticRegime::default()));
        engine.register(Box::new(TraitRegime::default()));
        engine.register(Box::new(TemporalRegime::default()));
        engine.register(Box::new(ProvenanceRegime::default()));
        engine.register(Box::new(NarrativeRegime::default()));
        engine.register(Box::new(KnowledgeRegime::default()));
        engine.register(Box::new(StructuralRegime::default()));
        engine.register(Box::new(ConsensusRegime::default()));
        engine
    }

    pub fn register(&mut self, regime: Box<dyn Regime>) {
        self.regimes.push(regime);
    }

    pub fn dimensions(&self) -> usize {
        self.regimes.len()
    }

    pub fn detect(&self, incoming: &IncomingContent, existing: &[Memory]) -> RevisionResult {
        if existing.is_empty() {
            return RevisionResult {
                match_id: None,
                classification: Classification::Novel,
                flight_status: FlightStatus::Crashed,
                regime_scores: HashMap::new(),
                total_score: 0.0,
                fired_regime: None,
                explanation: "no existing memories to compare against".to_string(),
            };
        }

        let mut best_result: Option<RevisionResult> = None;

        for mem in existing {
            let result = self.evaluate_pair(incoming, mem);
            match &best_result {
                Some(prev) if result.total_score <= prev.total_score => {}
                _ => best_result = Some(result),
            }
        }

        best_result.unwrap_or(RevisionResult {
            match_id: None,
            classification: Classification::Novel,
            flight_status: FlightStatus::Crashed,
            regime_scores: HashMap::new(),
            total_score: 0.0,
            fired_regime: None,
            explanation: "no match found".to_string(),
        })
    }

    fn evaluate_pair(&self, incoming: &IncomingContent, existing: &Memory) -> RevisionResult {
        let mut scores = HashMap::new();
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut fired_regime: Option<String> = None;

        for regime in &self.regimes {
            let score = regime.evaluate(incoming, existing);
            scores.insert(regime.name().to_string(), score);
            weighted_sum += score * regime.weight();
            total_weight += regime.weight();
        }

        let total_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        for regime in &self.regimes {
            let score = scores.get(regime.name()).copied().unwrap_or(0.0);
            if score >= regime.threshold() {
                fired_regime = Some(regime.name().to_string());
                break;
            }
        }

        let classification = match &fired_regime {
            Some(name) if is_content_class(name) && total_score >= self.strong_floor => {
                Classification::StrongMatch
            }
            Some(name) if is_semantic_class(name) && total_score >= self.semantic_floor => {
                Classification::SemanticMatch
            }
            Some(_) if total_score >= self.drift_floor => Classification::DriftMatch,
            None if total_score >= self.drift_floor => Classification::DriftMatch,
            _ => Classification::Novel,
        };

        let flight_status = if total_score >= 0.9 {
            FlightStatus::Landed
        } else if total_score >= 0.5 {
            FlightStatus::Falling
        } else {
            FlightStatus::Crashed
        };

        let explanation = build_explanation(
            &classification, &fired_regime, existing.id, &scores, total_score,
        );

        RevisionResult {
            match_id: if classification == Classification::Novel { None } else { Some(existing.id) },
            classification,
            flight_status,
            regime_scores: scores,
            total_score,
            fired_regime,
            explanation,
        }
    }
}

fn is_content_class(name: &str) -> bool {
    matches!(name, "content" | "provenance")
}

fn is_semantic_class(name: &str) -> bool {
    matches!(name, "semantic" | "knowledge" | "structural" | "narrative")
}

fn build_explanation(
    class: &Classification, fired: &Option<String>, mem_id: i64,
    scores: &HashMap<String, f64>, total: f64,
) -> String {
    let fired_str = fired.as_deref().unwrap_or("none");
    let top_scores: Vec<String> = {
        let mut pairs: Vec<_> = scores.iter().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.iter().take(3).map(|(k, v)| format!("{k}={v:.2}")).collect()
    };
    match class {
        Classification::StrongMatch => format!(
            "near-duplicate of #{mem_id} (fired={fired_str}, total={total:.3}, top: {})",
            top_scores.join(", ")
        ),
        Classification::SemanticMatch => format!(
            "revision of #{mem_id} (fired={fired_str}, total={total:.3}, top: {})",
            top_scores.join(", ")
        ),
        Classification::DriftMatch => format!(
            "drift from #{mem_id} (fired={fired_str}, total={total:.3}, top: {})",
            top_scores.join(", ")
        ),
        Classification::Novel => format!(
            "novel content (total={total:.3}, top: {})", top_scores.join(", ")
        ),
    }
}

// ====================================================================
// CORE REGIMES (9 dimensions)
// ====================================================================

// -- 1. Content --
pub struct ContentRegime { pub threshold: f64, pub weight: f64 }
impl Default for ContentRegime { fn default() -> Self { Self { threshold: 0.85, weight: 1.0 } } }
impl Regime for ContentRegime {
    fn name(&self) -> &str { "content" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("semantic") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let a = &incoming.content; let b = &existing.content;
        if a.is_empty() || b.is_empty() { return 0.0; }
        let len_ratio = a.len().min(b.len()) as f64 / a.len().max(b.len()) as f64;
        let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
        0.3 * len_ratio + 0.7 * jaccard_set(&wa, &wb)
    }
}

// -- 2. Semantic --
pub struct SemanticRegime { pub threshold: f64, pub weight: f64 }
impl Default for SemanticRegime { fn default() -> Self { Self { threshold: 0.55, weight: 1.0 } } }
impl Regime for SemanticRegime {
    fn name(&self) -> &str { "semantic" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("trait") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let t = jaccard_similarity(&incoming.topics, &existing.topics);
        let i = 1.0 - (incoming.importance - existing.importance).abs();
        if incoming.entities.is_empty() && existing.entities.is_empty() {
            // Neither has entities — redistribute entity weight to available signals.
            (0.45 * t + 0.20 * i) / 0.65
        } else {
            let e = jaccard_similarity(&incoming.entities, &existing.entities);
            (0.45 * t + 0.35 * e + 0.20 * i).clamp(0.0, 1.0)
        }
    }
}

// -- 3. Trait --
pub struct TraitRegime { pub threshold: f64, pub weight: f64 }
impl Default for TraitRegime { fn default() -> Self { Self { threshold: 0.40, weight: 1.0 } } }
impl Regime for TraitRegime {
    fn name(&self) -> &str { "trait" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("temporal") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        if incoming.traits.is_empty() || existing.traits.is_empty() { return 0.5; }
        cosine_similarity(&incoming.traits, &existing.traits).clamp(0.0, 1.0)
    }
}

// -- 4. Temporal --
pub struct TemporalRegime { pub threshold: f64, pub weight: f64 }
impl Default for TemporalRegime { fn default() -> Self { Self { threshold: 0.50, weight: 0.6 } } }
impl Regime for TemporalRegime {
    fn name(&self) -> &str { "temporal" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("provenance") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let a = match &incoming.created_at { Some(ts) => ts.as_str(), None => return 0.5 };
        let b = existing.created_at.as_str();
        if a.is_empty() || b.is_empty() { return 0.5; }
        let shared = a.chars().zip(b.chars()).take_while(|(ca, cb)| ca == cb).count();
        let max_len = a.len().max(b.len());
        if max_len == 0 { 0.5 } else { (shared as f64 / max_len as f64).clamp(0.0, 1.0) }
    }
}

// -- 5. Provenance --
pub struct ProvenanceRegime { pub threshold: f64, pub weight: f64 }
impl Default for ProvenanceRegime { fn default() -> Self { Self { threshold: 0.80, weight: 0.8 } } }
impl Regime for ProvenanceRegime {
    fn name(&self) -> &str { "provenance" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("narrative") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let a = incoming.source.to_lowercase();
        let b = existing.source.to_lowercase();
        if a == b { return 1.0; }
        let pa: std::collections::HashSet<&str> = a.split(&['/', '\\', '.'][..]).collect();
        let pb: std::collections::HashSet<&str> = b.split(&['/', '\\', '.'][..]).collect();
        jaccard_set(&pa, &pb)
    }
}

// -- 6. Narrative --
pub struct NarrativeRegime { pub threshold: f64, pub weight: f64 }
impl Default for NarrativeRegime { fn default() -> Self { Self { threshold: 0.50, weight: 0.7 } } }
impl Regime for NarrativeRegime {
    fn name(&self) -> &str { "narrative" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("knowledge") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let a = &incoming.summary; let b = &existing.summary;
        if a.is_empty() || b.is_empty() { return 0.0; }
        let wa: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let wb: std::collections::HashSet<&str> = b.split_whitespace().collect();
        jaccard_set(&wa, &wb)
    }
}

// -- 7. Knowledge --
pub struct KnowledgeRegime { pub threshold: f64, pub weight: f64 }
impl Default for KnowledgeRegime { fn default() -> Self { Self { threshold: 0.45, weight: 0.8 } } }
impl Regime for KnowledgeRegime {
    fn name(&self) -> &str { "knowledge" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("structural") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let total = incoming.entities.len() + existing.entities.len();
        if total == 0 {
            // No entities in either — fall back to direct content word overlap.
            let wa: std::collections::HashSet<&str> = incoming.content.split_whitespace().collect();
            let wb: std::collections::HashSet<&str> = existing.content.split_whitespace().collect();
            return jaccard_set(&wa, &wb);
        }
        let entity_sim = jaccard_similarity(&incoming.entities, &existing.entities);
        let il = incoming.content.to_lowercase();
        let el = existing.content.to_lowercase();
        let im = existing.entities.iter().filter(|e| il.contains(&e.to_lowercase())).count();
        let em = incoming.entities.iter().filter(|e| el.contains(&e.to_lowercase())).count();
        let cross = (im + em) as f64 / total as f64;
        (0.5 * entity_sim + 0.5 * cross).clamp(0.0, 1.0)
    }
}

// -- 8. Structural --
pub struct StructuralRegime { pub threshold: f64, pub weight: f64 }
impl Default for StructuralRegime { fn default() -> Self { Self { threshold: 0.45, weight: 0.7 } } }
impl Regime for StructuralRegime {
    fn name(&self) -> &str { "structural" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { Some("consensus") }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        if incoming.topics.is_empty() || existing.topics.is_empty() { return 0.0; }
        let mut score = 0.0;
        let max_k = incoming.topics.len().max(existing.topics.len());
        for (i, topic) in incoming.topics.iter().enumerate() {
            let pw = 1.0 / (i as f64 + 1.0);
            if let Some(j) = existing.topics.iter().position(|t| t.to_lowercase() == topic.to_lowercase()) {
                let rd = (i as f64 - j as f64).abs() / max_k as f64;
                score += pw * (1.0 - rd);
            }
        }
        let norm: f64 = (0..incoming.topics.len()).map(|i| 1.0 / (i as f64 + 1.0)).sum();
        if norm > 0.0 { (score / norm).clamp(0.0, 1.0) } else { 0.0 }
    }
}

// -- 9. Consensus (meta-regime) --
pub struct ConsensusRegime { pub threshold: f64, pub weight: f64 }
impl Default for ConsensusRegime { fn default() -> Self { Self { threshold: 0.40, weight: 0.5 } } }
impl Regime for ConsensusRegime {
    fn name(&self) -> &str { "consensus" }
    fn threshold(&self) -> f64 { self.threshold }
    fn weight(&self) -> f64 { self.weight }
    fn next_on_fail(&self) -> Option<&str> { None }
    fn evaluate(&self, incoming: &IncomingContent, existing: &Memory) -> f64 {
        let sigs = [
            jaccard_similarity(&incoming.topics, &existing.topics),
            jaccard_similarity(&incoming.entities, &existing.entities),
            if incoming.traits.is_empty() || existing.traits.is_empty() { 0.5 }
            else { cosine_similarity(&incoming.traits, &existing.traits).clamp(0.0, 1.0) },
            1.0 - (incoming.importance - existing.importance).abs(),
        ];
        let agreeing = sigs.iter().filter(|&&s| s > 0.3).count();
        let avg: f64 = sigs.iter().sum::<f64>() / sigs.len() as f64;
        (agreeing as f64 / sigs.len() as f64 * avg).clamp(0.0, 1.0)
    }
}

// ====================================================================
// TESTS
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(id: i64, content: &str, topics: &[&str], traits: &[f64]) -> Memory {
        Memory {
            id, collection: "test".to_string(),
            content: content.to_string(),
            summary: content.chars().take(80).collect(),
            source: "test.txt".to_string(),
            file_hash: String::new(), importance: 0.7,
            traits: traits.to_vec(),
            topics: topics.iter().map(|s| s.to_string()).collect(),
            entities: vec![],
            created_at: "2026-03-06T12:00:00".to_string(),
            consolidated: false,
        }
    }

    fn make_incoming(content: &str, topics: &[&str], traits: &[f64]) -> IncomingContent {
        IncomingContent {
            content: content.to_string(),
            summary: content.chars().take(80).collect(),
            source: "test.txt".to_string(),
            topics: topics.iter().map(|s| s.to_string()).collect(),
            entities: vec![], traits: traits.to_vec(),
            importance: 0.7,
            created_at: Some("2026-03-06T12:00:00".to_string()),
            domain_meta: HashMap::new(),
        }
    }

    #[test]
    fn engine_has_9_core_dimensions() {
        let engine = CascadeEngine::with_core_regimes();
        assert_eq!(engine.dimensions(), 9);
    }

    #[test]
    fn exact_duplicate_is_strong_match() {
        let engine = CascadeEngine::with_core_regimes();
        let existing = vec![make_memory(1, "AI agents need persistent memory to function effectively", &["ai", "memory", "agents"], &[0.8, 0.3, 0.6])];
        let incoming = make_incoming("AI agents need persistent memory to function effectively", &["ai", "memory", "agents"], &[0.8, 0.3, 0.6]);
        let result = engine.detect(&incoming, &existing);
        assert_eq!(result.classification, Classification::StrongMatch);
        assert_eq!(result.flight_status, FlightStatus::Landed);
        assert_eq!(result.regime_scores.len(), 9);
    }

    #[test]
    fn novel_content_is_novel() {
        let engine = CascadeEngine::with_core_regimes();
        let existing = vec![make_memory(1, "AI agents need persistent memory", &["ai", "memory"], &[0.8, 0.3, 0.6])];
        let incoming = make_incoming("The best sourdough recipe requires a 72 hour cold ferment", &["cooking", "bread", "fermentation"], &[-0.5, 0.9, -0.2]);
        let result = engine.detect(&incoming, &existing);
        assert_eq!(result.classification, Classification::Novel);
        assert_eq!(result.match_id, None);
    }

    #[test]
    fn custom_regime_adds_dimension() {
        struct NflArchetype;
        impl Regime for NflArchetype {
            fn name(&self) -> &str { "nfl_archetype" }
            fn evaluate(&self, _: &IncomingContent, _: &Memory) -> f64 { 0.42 }
            fn threshold(&self) -> f64 { 0.5 }
            fn weight(&self) -> f64 { 1.0 }
            fn next_on_fail(&self) -> Option<&str> { None }
        }
        let mut engine = CascadeEngine::with_core_regimes();
        engine.register(Box::new(NflArchetype));
        assert_eq!(engine.dimensions(), 10);
    }
}
