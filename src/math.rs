/// Shared math primitives used by edge detection and revision scoring.

/// Jaccard similarity between two string slices (case-insensitive).
/// Returns 0.0 if both are empty.
pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let a_set: std::collections::HashSet<String> = a.iter().map(|s| s.to_lowercase()).collect();
    let b_set: std::collections::HashSet<String> = b.iter().map(|s| s.to_lowercase()).collect();
    let intersection = a_set.intersection(&b_set).count() as f64;
    let union = a_set.union(&b_set).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

/// Jaccard similarity between two pre-built HashSets (borrows, no allocation).
pub fn jaccard_set<T: std::hash::Hash + Eq>(
    a: &std::collections::HashSet<T>,
    b: &std::collections::HashSet<T>,
) -> f64 {
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

/// Cosine similarity between two f64 slices. Operates on the shorter length.
/// Returns 0.0 if either slice is empty or has zero magnitude.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { 0.0 } else { dot / (mag_a * mag_b) }
}
