//! Monte Carlo integration helpers for simulating heterogeneous consumer tastes.

use nalgebra::{DMatrix, DVector};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{BlpError, Result};

/// Represents simulated consumer heterogeneity used in BLP demand estimation.
#[derive(Clone, Debug)]
pub struct SimulationDraws {
    draws: DMatrix<f64>,
    weights: DVector<f64>,
}

impl SimulationDraws {
    /// Builds simulation draws from matrices of draws and weights.
    pub fn new(draws: DMatrix<f64>, weights: DVector<f64>) -> Result<Self> {
        if draws.nrows() == 0 {
            return Err(BlpError::dimension_mismatch("simulation draws", 1, 0));
        }
        if draws.nrows() != weights.len() {
            return Err(BlpError::dimension_mismatch(
                "draw weight length",
                draws.nrows(),
                weights.len(),
            ));
        }
        for weight in weights.iter() {
            if *weight <= 0.0 {
                return Err(BlpError::InvalidWeights { slack: *weight });
            }
        }
        let sum: f64 = weights.iter().sum();
        let slack = (sum - 1.0).abs();
        if slack > 1e-8 {
            return Err(BlpError::InvalidWeights { slack });
        }

        Ok(Self { draws, weights })
    }

    /// Generates standard normal draws with uniform weights.
    pub fn standard_normal(draws: usize, dimension: usize, seed: u64) -> Self {
        assert!(draws > 0, "at least one draw is required");
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut values = Vec::with_capacity(draws * dimension);
        for _ in 0..(draws * dimension) {
            values.push(StandardNormal.sample(&mut rng));
        }
        let matrix = DMatrix::from_vec(draws, dimension, values);
        let weight = 1.0 / draws as f64;
        let weights = DVector::from_element(draws, weight);
        Self::new(matrix, weights).expect("validated gaussian draws")
    }

    /// Number of Monte Carlo draws.
    pub fn draw_count(&self) -> usize {
        self.draws.nrows()
    }

    /// Dimension of the random coefficients.
    pub fn dimension(&self) -> usize {
        self.draws.ncols()
    }

    /// Returns a view of the draw matrix.
    pub fn draws(&self) -> &DMatrix<f64> {
        &self.draws
    }

    /// Returns the associated integration weights (normalized to sum to one).
    pub fn weights(&self) -> &DVector<f64> {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_generates_expected_shapes() {
        let draws = SimulationDraws::standard_normal(128, 2, 7);
        assert_eq!(draws.draw_count(), 128);
        assert_eq!(draws.dimension(), 2);
        let weights_sum: f64 = draws.weights.iter().sum();
        assert!((weights_sum - 1.0).abs() < 1e-10);
    }
}
