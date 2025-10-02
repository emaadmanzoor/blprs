//! Contraction solver configuration and diagnostics.

/// Configuration for the BLP fixed-point contraction that recovers mean utilities.
#[derive(Clone, Debug)]
pub struct ContractionOptions {
    /// Supremum norm tolerance for convergence.
    pub tolerance: f64,
    /// Maximum number of iterations allowed before aborting.
    pub max_iterations: usize,
    /// Damping factor applied to the log-share update (1.0 is standard BLP).
    pub damping: f64,
    /// Lower bound enforced on predicted shares to avoid taking `ln(0)`.
    pub minimum_share: f64,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-9,
            max_iterations: 1_000,
            damping: 1.0,
            minimum_share: 1e-16,
        }
    }
}

/// Diagnostics returned alongside the contracted mean utilities.
#[derive(Clone, Debug)]
pub struct ContractionSummary {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Maximum absolute change observed in the final iteration.
    pub max_gap: f64,
}
