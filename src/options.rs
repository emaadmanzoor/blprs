//! Configuration structures that mirror pyBLP's solver and GMM options while remaining idiomatic Rust.

use nalgebra::DMatrix;

use crate::solving::ContractionOptions;

/// Choice of weighting matrix used in the GMM objective.
#[derive(Clone, Debug)]
pub enum WeightingMatrix {
    /// Use the inverse of `Z'Z`, matching the canonical two-step BLP estimator.
    InverseZTZ,
    /// Provide a custom positive-definite weighting matrix.
    Provided(DMatrix<f64>),
}

/// Controls the outer GMM loop and weighting updates.
#[derive(Clone, Debug)]
pub struct GmmOptions {
    /// Maximum number of outer iterations (weighting updates).
    pub max_iterations: usize,
    /// Convergence tolerance for the GMM objective (not yet enforced by the minimal implementation).
    pub tolerance: f64,
    /// Whether to update the weighting matrix between iterations.
    pub update_weighting: bool,
    /// Strategy for constructing the weighting matrix.
    pub weighting: WeightingMatrix,
}

impl Default for GmmOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1,
            tolerance: 1e-10,
            update_weighting: false,
            weighting: WeightingMatrix::InverseZTZ,
        }
    }
}

/// Aggregated solver configuration used when estimating a [`Problem`](crate::Problem).
#[derive(Clone, Debug)]
pub struct ProblemOptions {
    /// Configuration for the contraction mapping that recovers mean utilities.
    pub contraction: ContractionOptions,
    /// Configuration for the outer GMM iterations.
    pub gmm: GmmOptions,
}

impl Default for ProblemOptions {
    fn default() -> Self {
        Self {
            contraction: ContractionOptions::default(),
            gmm: GmmOptions::default(),
        }
    }
}

impl ProblemOptions {
    /// Override the contraction settings while preserving other defaults.
    pub fn with_contraction(mut self, contraction: ContractionOptions) -> Self {
        self.contraction = contraction;
        self
    }

    /// Override the weighting configuration while preserving other defaults.
    pub fn with_weighting(mut self, weighting: WeightingMatrix) -> Self {
        self.gmm.weighting = weighting;
        self
    }

    /// Set the maximum number of outer GMM iterations that should be attempted.
    pub fn with_max_gmm_iterations(mut self, max_iterations: usize) -> Self {
        self.gmm.max_iterations = max_iterations.max(1);
        self
    }

    /// Set the convergence tolerance for the GMM objective (stored for diagnostics).
    pub fn with_gmm_tolerance(mut self, tolerance: f64) -> Self {
        self.gmm.tolerance = tolerance;
        self
    }

    /// Enable or disable weighting matrix updates between GMM iterations.
    pub fn with_weighting_updates(mut self, update: bool) -> Self {
        self.gmm.update_weighting = update;
        self
    }
}

/// Backwards-compatible alias for users migrating from earlier versions.
pub type EstimationOptions = ProblemOptions;
