//! Demand-side primitives: share prediction and the BLP contraction mapping.

use nalgebra::{DMatrix, DVector};

use crate::data::ProductData;
use crate::error::{BlpError, Result};
use crate::integration::SimulationDraws;
use crate::solving::{ContractionOptions, ContractionSummary};

/// Computes model-implied product shares given mean utilities `delta` and
/// nonlinear parameters `sigma`.
pub fn predict_shares(
    delta: &DVector<f64>,
    data: &ProductData,
    sigma: &DMatrix<f64>,
    draws: &SimulationDraws,
    options: &ContractionOptions,
) -> Result<DVector<f64>> {
    let n = delta.len();
    if n != data.product_count() {
        return Err(BlpError::dimension_mismatch(
            "delta length",
            data.product_count(),
            n,
        ));
    }

    let k2 = data.nonlinear_dim();
    if k2 == 0 {
        return predict_simple_logit(delta, data, options);
    }

    if sigma.nrows() != k2 || sigma.ncols() != k2 {
        return Err(BlpError::dimension_mismatch(
            "sigma dimension",
            k2,
            sigma.nrows(),
        ));
    }
    if draws.dimension() != k2 {
        return Err(BlpError::dimension_mismatch(
            "draw dimension",
            k2,
            draws.dimension(),
        ));
    }

    let mut predicted = DVector::zeros(n);
    let draws_matrix = draws.draws();
    let weights = draws.weights();

    for (draw_index, weight) in weights.iter().enumerate() {
        let draw = draws_matrix.row(draw_index).transpose();
        let taste = sigma * draw;

        for market in data.partition().markets() {
            let range = market.range();
            let mut exp_utilities = Vec::with_capacity(range.len());
            let mut denominator = 1.0_f64;

            for product_index in range.clone() {
                let mu = data.x2().row(product_index).dot(&taste);
                let utility = delta[product_index] + mu;
                let exp_u = utility.exp();
                if !exp_u.is_finite() {
                    return Err(BlpError::NumericalError {
                        context: "utility exponentiation",
                    });
                }
                exp_utilities.push(exp_u);
                denominator += exp_u;
            }

            for (offset, product_index) in range.enumerate() {
                let share = *weight * exp_utilities[offset] / denominator;
                if share < options.minimum_share {
                    return Err(BlpError::NumericalError {
                        context: "predicted share underflow",
                    });
                }
                predicted[product_index] += share;
            }
        }
    }

    Ok(predicted)
}

fn predict_simple_logit(
    delta: &DVector<f64>,
    data: &ProductData,
    options: &ContractionOptions,
) -> Result<DVector<f64>> {
    let mut predicted = DVector::zeros(delta.len());

    for market in data.partition().markets() {
        let range = market.range();
        let mut exp_utilities = Vec::with_capacity(range.len());
        let mut denominator = 1.0_f64;

        for product_index in range.clone() {
            let utility = delta[product_index];
            let exp_u = utility.exp();
            if !exp_u.is_finite() {
                return Err(BlpError::NumericalError {
                    context: "utility exponentiation",
                });
            }
            exp_utilities.push(exp_u);
            denominator += exp_u;
        }

        for (offset, product_index) in range.enumerate() {
            let share = exp_utilities[offset] / denominator;
            if share < options.minimum_share {
                return Err(BlpError::NumericalError {
                    context: "predicted share underflow",
                });
            }
            predicted[product_index] = share;
        }
    }

    Ok(predicted)
}

/// Solves the BLP fixed-point equation for mean utilities `delta`.
pub fn solve_delta(
    data: &ProductData,
    draws: &SimulationDraws,
    sigma: &DMatrix<f64>,
    options: &ContractionOptions,
) -> Result<(DVector<f64>, ContractionSummary)> {
    let n = data.product_count();
    let mut delta = DVector::zeros(n);

    // Initialize using the standard log share ratio: delta = log(s_j) - log(s_0)
    for (product_index, share) in data.shares().iter().enumerate() {
        let outside = data.outside_share_for_product(product_index);
        delta[product_index] = (share / outside).ln();
    }

    let mut max_gap = f64::INFINITY;
    let mut iteration = 0usize;

    while iteration < options.max_iterations {
        let predicted = predict_shares(&delta, data, sigma, draws, options)?;
        max_gap = 0.0;

        for product_index in 0..n {
            let observed = data.shares()[product_index];
            let model = predicted[product_index];
            if model < options.minimum_share {
                return Err(BlpError::NumericalError {
                    context: "predicted share underflow",
                });
            }
            let update = (observed / model).ln();
            let damped = options.damping * update;
            delta[product_index] += damped;
            max_gap = max_gap.max(damped.abs());
        }

        iteration += 1;
        if max_gap < options.tolerance {
            return Ok((
                delta,
                ContractionSummary {
                    iterations: iteration,
                    max_gap,
                },
            ));
        }
    }

    Err(BlpError::ContractionDidNotConverge {
        iterations: iteration,
        max_gap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ProductDataBuilder;
    use approx::assert_relative_eq;

    /// Reproduces the homogeneous logit solution where the contraction converges in one step.
    #[test]
    fn contraction_with_zero_sigma_matches_logit() {
        let market_ids = vec!["m1".to_string(), "m1".to_string()];
        let shares = DVector::from_vec(vec![0.2, 0.3]);
        let x1 = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 2.0]);
        let data = ProductDataBuilder::new(market_ids, shares)
            .x1(x1.clone())
            .build()
            .unwrap();
        let draws = SimulationDraws::standard_normal(1, 0, 123);
        let sigma = DMatrix::<f64>::zeros(0, 0);
        let options = ContractionOptions::default();

        let (delta, summary) = solve_delta(&data, &draws, &sigma, &options).unwrap();
        assert_eq!(summary.iterations, 1);

        let outside = data.outside_share_for_product(0);
        let expected_delta0 = (data.shares()[0] / outside).ln();
        assert_relative_eq!(delta[0], expected_delta0, epsilon = 1e-9);
    }
}
