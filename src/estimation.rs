//! High-level demand estimation pipeline that mirrors `pyBLP.Problem`.

use nalgebra::{DMatrix, DVector};

use crate::data::ProductData;
use crate::demand::{predict_shares, solve_delta};
use crate::error::{BlpError, Result};
use crate::integration::SimulationDraws;
use crate::options::{ProblemOptions, WeightingMatrix};
use crate::solving::ContractionSummary;

/// High-level wrapper that mirrors `pyBLP.Problem` on the demand side.
#[derive(Clone, Debug)]
pub struct Problem {
    data: ProductData,
    draws: SimulationDraws,
    options: ProblemOptions,
}

impl Problem {
    /// Construct a new BLP estimation problem with default solver options.
    pub fn new(data: ProductData, draws: SimulationDraws) -> Result<Self> {
        Self::with_options(data, draws, ProblemOptions::default())
    }

    /// Construct a new problem with explicit solver options.
    pub fn with_options(
        data: ProductData,
        draws: SimulationDraws,
        options: ProblemOptions,
    ) -> Result<Self> {
        if data.nonlinear_dim() == 0 && draws.dimension() != 0 {
            return Err(BlpError::dimension_mismatch(
                "draw dimension",
                0,
                draws.dimension(),
            ));
        }
        Ok(Self {
            data,
            draws,
            options,
        })
    }

    /// Start building a problem fluently, mirroring the ergonomics of pyBLP's kwargs.
    pub fn builder() -> ProblemBuilder {
        ProblemBuilder::default()
    }

    /// Accessor for product data.
    pub fn data(&self) -> &ProductData {
        &self.data
    }

    /// Accessor for simulation draws.
    pub fn draws(&self) -> &SimulationDraws {
        &self.draws
    }

    /// Accessor for the default solver options carried by the problem.
    pub fn options(&self) -> &ProblemOptions {
        &self.options
    }

    /// Solve the model for a given nonlinear parameter matrix `sigma` using the stored options.
    pub fn solve(&self, sigma: &DMatrix<f64>) -> Result<ProblemResults> {
        self.solve_with_options(sigma, &self.options)
    }

    /// Solve the model with an explicit options override.
    pub fn solve_with_options(
        &self,
        sigma: &DMatrix<f64>,
        options: &ProblemOptions,
    ) -> Result<ProblemResults> {
        let (delta, contraction) =
            solve_delta(&self.data, &self.draws, sigma, &options.contraction)?;

        let weighting = match &options.gmm.weighting {
            WeightingMatrix::InverseZTZ => inverse_ztz(self.data.instruments())?,
            WeightingMatrix::Provided(matrix) => matrix.clone(),
        };

        let beta = compute_linear_parameters(&self.data, &delta, &weighting)?;
        let xi = &delta - self.data.x1() * &beta;
        let predicted_shares =
            predict_shares(&delta, &self.data, sigma, &self.draws, &options.contraction)?;
        let gmm_value = compute_gmm_objective(&self.data, &xi, &weighting);

        Ok(ProblemResults {
            delta,
            beta,
            xi,
            predicted_shares,
            gmm_value,
            contraction,
            weighting_matrix: weighting,
            options_used: options.clone(),
        })
    }

    /// Backwards-compatible helper for earlier API versions that called `estimate` directly.
    pub fn estimate(
        &self,
        sigma: &DMatrix<f64>,
        options: &ProblemOptions,
    ) -> Result<ProblemResults> {
        self.solve_with_options(sigma, options)
    }
}

/// Fluent builder for [`Problem`], mirroring pyBLP's keyword-heavy constructors.
#[derive(Clone, Debug, Default)]
pub struct ProblemBuilder {
    products: Option<ProductData>,
    draws: Option<SimulationDraws>,
    options: ProblemOptions,
}

impl ProblemBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide validated product data.
    pub fn products(mut self, products: ProductData) -> Self {
        self.products = Some(products);
        self
    }

    /// Provide simulation draws for heterogeneous consumers.
    pub fn draws(mut self, draws: SimulationDraws) -> Self {
        self.draws = Some(draws);
        self
    }

    /// Override solver options carried by the constructed problem.
    pub fn options(mut self, options: ProblemOptions) -> Self {
        self.options = options;
        self
    }

    /// Finalise the builder into a fully-configured problem.
    pub fn build(self) -> Result<Problem> {
        let products = self
            .products
            .ok_or_else(|| BlpError::missing_component("product data"))?;
        let draws = self
            .draws
            .ok_or_else(|| BlpError::missing_component("simulation draws"))?;
        Problem::with_options(products, draws, self.options)
    }
}

/// Describes the result of a BLP estimation run.
#[derive(Clone, Debug)]
pub struct ProblemResults {
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
    /// Options that were in effect during estimation.
    pub options_used: ProblemOptions,
}

/// Backwards-compatible alias for earlier versions of the crate.
pub type BlpProblem = Problem;
/// Backwards-compatible alias for earlier versions of the crate.
pub type EstimationResult = ProblemResults;

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
        let problem = Problem::new(data, draws).unwrap();
        let options = ProblemOptions::default();

        let result = problem.solve_with_options(&sigma, &options).unwrap();
        assert_eq!(result.contraction.iterations, 1);
        assert!(result.gmm_value >= 0.0);

        // Homogeneous logit reduces to simple IV regression with instruments = X.
        let outside = 0.5_f64;
        let delta_0 = (0.2_f64 / outside).ln();
        assert_relative_eq!(result.delta[0], delta_0, epsilon = 1e-9);
    }

    #[test]
    fn builder_requires_components() {
        let market_ids = vec!["m1".to_string(), "m1".to_string()];
        let shares = DVector::from_vec(vec![0.2, 0.3]);
        let x1 = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 2.0]);
        let products = ProductDataBuilder::new(market_ids, shares)
            .x1(x1)
            .build()
            .unwrap();
        let draws = SimulationDraws::standard_normal(1, 0, 7);

        let problem = Problem::builder()
            .products(products.clone())
            .draws(draws.clone())
            .build()
            .expect("builder succeeds");
        assert_eq!(problem.data().product_count(), 2);

        let err = Problem::builder()
            .products(products)
            .build()
            .expect_err("missing draws");
        assert!(matches!(err, BlpError::MissingComponent { .. }));

        let err = Problem::builder()
            .draws(draws)
            .build()
            .expect_err("missing products");
        assert!(matches!(err, BlpError::MissingComponent { .. }));
    }
}
