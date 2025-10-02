use approx::assert_relative_eq;
use blprs::data::ProductDataBuilder;
use blprs::demand::predict_shares;
use blprs::integration::SimulationDraws;
use blprs::{ContractionOptions, Problem, ProblemOptions};
use nalgebra::{DMatrix, DVector};

/// Reproduces a simple logit comparison whose closed-form delta matches values reported in
/// `pyblp/tests/test_blp.py` when running with the same product shares.
#[test]
fn logit_delta_matches_pyblp_reference() {
    let market_ids = vec!["0".to_string(), "0".to_string(), "1".to_string()];
    let shares = DVector::from_vec(vec![0.3, 0.2, 0.4]);
    let x1 = DMatrix::from_row_slice(3, 2, &[1.0, 10.0, 1.0, 15.0, 1.0, 12.0]);
    let instruments = x1.clone();

    let data = ProductDataBuilder::new(market_ids, shares.clone())
        .x1(x1)
        .instruments(instruments)
        .build()
        .unwrap();

    // No heterogeneity, so we supply zero-dimensional draws and a 0x0 sigma matrix.
    let draws = SimulationDraws::standard_normal(1, 0, 123);
    let sigma = DMatrix::<f64>::zeros(0, 0);

    let problem = Problem::builder()
        .products(data.clone())
        .draws(draws.clone())
        .build()
        .unwrap();
    let options = ProblemOptions::default();
    let result = problem.solve_with_options(&sigma, &options).unwrap();

    let expected_delta = DVector::from_vec(vec![
        -0.510_825_623_765_9907,
        -0.916_290_731_874_155,
        -0.405_465_108_108_1644,
    ]);
    assert_relative_eq!(result.delta, expected_delta, epsilon = 1e-12);

    // Ensure that predicted shares match the observed shares within numerical tolerance.
    let predicted = predict_shares(
        &result.delta,
        &data,
        &sigma,
        problem.draws(),
        &ContractionOptions::default(),
    )
    .unwrap();
    assert_relative_eq!(predicted, shares, epsilon = 1e-12);
}

/// Mirror of the Gauss-Hermite moment check in `pyblp/tests/test_integration.py` limited to Monte Carlo draws.
#[test]
fn gaussian_second_moment_is_unity() {
    let draws = SimulationDraws::standard_normal(50_000, 2, 42);
    let nodes = draws.draws();
    let weights = draws.weights();

    // Estimate E[x^2] under the standard normal using Monte Carlo draws.
    let mut moment = 0.0;
    for (row, weight) in nodes.row_iter().zip(weights.iter()) {
        let squared = row.iter().fold(1.0, |acc, value| acc * value.powi(2));
        moment += weight * squared;
    }
    assert_relative_eq!(moment, 1.0, epsilon = 1e-2);
}

/// Ensure weights sum to one, mirroring the checks in `pyblp/tests/test_integration.py`.
#[test]
fn integration_weights_sum_to_one() {
    let draws = SimulationDraws::standard_normal(1_000, 3, 7);
    let weights_sum: f64 = draws.weights().iter().sum();
    assert_relative_eq!(weights_sum, 1.0, epsilon = 1e-12);
}
