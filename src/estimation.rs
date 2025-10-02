//! High-level demand estimation pipeline that mirrors `pyBLP.Problem`.

use nalgebra::{DMatrix, DVector};

use crate::data::ProductData;
use crate::demand::{predict_shares, solve_delta};
use crate::error::{BlpError, Result};
use crate::integration::SimulationDraws;
use crate::solving::{ContractionOptions, ContractionSummary};

/// High-level wrapper that mirrors `pyBLP.Problem` on the demand side.
#[derive(Clone, Debug)]
pub struct BlpProblem {
    data: ProductData,
    draws: SimulationDraws,
}

impl BlpProblem {
    /// Constructs a new BLP estimation problem.
    pub fn new(data: ProductData, draws: SimulationDraws) -> Result<Self> {
        if data.nonlinear_dim() == 0 && draws.dimension() != 0 {
            return Err(BlpError::dimension_mismatch(
                "draw dimension",
                0,
                draws.dimension(),
            ));
        }
        Ok(Self { data, draws })
    }

    /// Accessor for product data.
    pub fn data(&self) -> &ProductData {
        &self.data
    }

    /// Accessor for simulation draws.
    pub fn draws(&self) -> &SimulationDraws {
        &self.draws
    }

    /// Estimates the model for a given nonlinear parameter matrix `sigma`.
    pub fn estimate(
        &self,
        sigma: &DMatrix<f64>,
        options: &EstimationOptions,
    ) -> Result<EstimationResult> {
        let (delta, contraction) =
            solve_delta(&self.data, &self.draws, sigma, &options.contraction)?;

        let weighting = match &options.weighting {
            WeightingMatrix::InverseZTZ => inverse_ztz(self.data.instruments())?,
            WeightingMatrix::Provided(matrix) => matrix.clone(),
        };

        let beta = compute_linear_parameters(&self.data, &delta, &weighting)?;
        let xi = &delta - self.data.x1() * &beta;
        let predicted_shares =
            predict_shares(&delta, &self.data, sigma, &self.draws, &options.contraction)?;
        let gmm_value = compute_gmm_objective(&self.data, &xi, &weighting);

        Ok(EstimationResult {
            delta,
            beta,
            xi,
            predicted_shares,
            gmm_value,
            contraction,
            weighting_matrix: weighting,
        })
    }
}

/// Configuration knobs for demand-side estimation.
#[derive(Clone, Debug)]
pub struct EstimationOptions {
    /// Options for the contraction mapping.
    pub contraction: ContractionOptions,
    /// Choice of GMM weighting matrix.
    pub weighting: WeightingMatrix,
}

impl Default for EstimationOptions {
    fn default() -> Self {
        Self {
            contraction: ContractionOptions::default(),
            weighting: WeightingMatrix::InverseZTZ,
        }
    }
}

impl EstimationOptions {
    /// Overrides the weighting matrix while keeping other defaults.
    pub fn with_weighting(mut self, weighting: WeightingMatrix) -> Self {
        self.weighting = weighting;
        self
    }

    /// Overrides the contraction options.
    pub fn with_contraction(mut self, contraction: ContractionOptions) -> Self {
        self.contraction = contraction;
        self
    }
}

/// Describes the result of a BLP estimation run.
#[derive(Clone, Debug)]
pub struct EstimationResult {
    /// Mean utilities recovered by the contraction mapping.
    pub delta: DVector<f64>,
    /// Linear taste parameters (equivalent to `beta` in BLP).
    pub beta: DVector<f64>,
    /// Structural error term implied by the demand system (`xi`).
    pub xi: DVector<f64>,
    /// Model-implied market shares corresponding to `delta`.
    pub predicted_shares: DVector<f64>,
    /// Value of the GMM objective at the solution.
    pub gmm_value: f64,
    /// Diagnostics from the contraction mapping.
    pub contraction: ContractionSummary,
    /// Weighting matrix used during estimation.
    pub weighting_matrix: DMatrix<f64>,
}

/// Computes the optimal linear parameters via two-stage least squares.
fn compute_linear_parameters(
    data: &ProductData,
    delta: &DVector<f64>,
    weighting: &DMatrix<f64>,
) -> Result<DVector<f64>> {
    let x1 = data.x1();
    let z = data.instruments();

    let z_t = z.transpose();
    let zx = &z_t * x1;
    let xz = zx.transpose();
    let ztz = &z_t * z;

    if ztz.nrows() != weighting.nrows() {
        return Err(BlpError::dimension_mismatch(
            "weighting rows",
            ztz.nrows(),
            weighting.nrows(),
        ));
    }

    let xzwzx = &xz * weighting * &zx;
    let rhs = xz * (weighting * (z_t * delta));

    let cholesky =
        nalgebra::linalg::Cholesky::new(xzwzx).ok_or_else(|| BlpError::singular("X'ZWZX"))?;
    Ok(cholesky.solve(&rhs))
}

/// Evaluates the standard BLP GMM objective.
fn compute_gmm_objective(data: &ProductData, xi: &DVector<f64>, weighting: &DMatrix<f64>) -> f64 {
    let z = data.instruments();
    let z_t = z.transpose();
    let ztxi = &z_t * xi;
    let w_ztxi = weighting * &ztxi;
    ztxi.dot(&w_ztxi)
}

fn inverse_ztz(z: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let z_t = z.transpose();
    let ztz = &z_t * z;
    let cholesky =
        nalgebra::linalg::Cholesky::new(ztz).ok_or_else(|| BlpError::singular("Z'Z inversion"))?;
    Ok(cholesky.inverse())
}

/// Weighting matrix strategies for the GMM estimator.
#[derive(Clone, Debug)]
pub enum WeightingMatrix {
    /// Uses the inverse of `Z'Z`, yielding the conventional two-step GMM weighting.
    InverseZTZ,
    /// Supplies a custom positive-definite weighting matrix.
    Provided(DMatrix<f64>),
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::data::ProductDataBuilder;

    #[test]
    fn estimate_linear_logit_matches_closed_form() {
        let market_ids = vec!["m1".to_string(), "m1".to_string()];
        let shares = DVector::from_vec(vec![0.2, 0.3]);
        let x1 = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 2.0]);
        let data = ProductDataBuilder::new(market_ids, shares)
            .x1(x1.clone())
            .build()
            .unwrap();
        let draws = SimulationDraws::standard_normal(1, 0, 42);
        let sigma = DMatrix::<f64>::zeros(0, 0);
        let problem = BlpProblem::new(data, draws).unwrap();
        let options = EstimationOptions::default();

        let result = problem.estimate(&sigma, &options).unwrap();
        assert_eq!(result.contraction.iterations, 1);
        assert!(result.gmm_value >= 0.0);

        // Homogeneous logit reduces to simple IV regression with instruments = X.
        let outside = 0.5_f64;
        let delta_0 = (0.2_f64 / outside).ln();
        assert_relative_eq!(result.delta[0], delta_0, epsilon = 1e-9);
    }
}
