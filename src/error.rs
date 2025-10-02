use thiserror::Error;

/// Unified error type for `blprs` operations.
#[derive(Debug, Error)]
pub enum BlpError {
    /// Raised when provided arrays or matrices have incompatible dimensions.
    #[error("dimension mismatch in {context}: expected {expected} but found {found}")]
    DimensionMismatch {
        /// Human-readable context describing the operation.
        context: &'static str,
        /// The required dimension, often the model-implied value.
        expected: usize,
        /// The dimension that was actually supplied.
        found: usize,
    },

    /// Raised when the supplied market ids are not grouped contiguously.
    #[error("market identifiers must appear in contiguous blocks; market `{market_id}` is split")]
    NonContiguousMarket { market_id: String },

    /// Raised when product shares are missing or non-positive.
    #[error("product share at index {index} must be positive, found {share}")]
    NonPositiveShare { index: usize, share: f64 },

    /// Raised when the outside good share becomes non-positive.
    #[error("outside share for market `{market_id}` must be positive, found {share}")]
    NonPositiveOutsideShare { market_id: String, share: f64 },

    /// Raised when a normalization or weight vector is invalid.
    #[error("weights must be strictly positive and sum to one (slack {slack})")]
    InvalidWeights { slack: f64 },

    /// Raised when linear algebra operations encounter a singular system.
    #[error("matrix in {context} is singular")]
    SingularMatrix { context: &'static str },

    /// Raised when the contraction mapping fails to meet the tolerance.
    #[error(
        "BLP contraction did not converge after {iterations} iterations; best max gap {max_gap}"
    )]
    ContractionDidNotConverge {
        /// Number of iterations performed before termination.
        iterations: usize,
        /// Maximum absolute change in the last iteration.
        max_gap: f64,
    },

    /// Raised when numerical routines produce NaN.
    #[error("encountered NaN during {context}")]
    NumericalError { context: &'static str },

    /// Raised when a required component has not been provided to a builder or solver.
    #[error("{component} must be provided before solving the problem")]
    MissingComponent { component: &'static str },
}

impl BlpError {
    /// Helper to format a [`DimensionMismatch`](BlpError::DimensionMismatch) error.
    pub fn dimension_mismatch(context: &'static str, expected: usize, found: usize) -> Self {
        Self::DimensionMismatch {
            context,
            expected,
            found,
        }
    }

    /// Helper to raise when a matrix factorization fails due to singularity.
    pub fn singular(context: &'static str) -> Self {
        Self::SingularMatrix { context }
    }

    /// Helper for bubbling up missing component errors from builders.
    pub fn missing_component(component: &'static str) -> Self {
        Self::MissingComponent { component }
    }
}

/// Type alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, BlpError>;
