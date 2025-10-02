# blprs
<img src="assets/logo.svg" height="180" align="right" alt="blprs logo"/>

*Fast Berry–Levinsohn–Pakes estimation in safe, expressive Rust*

[![docs.rs](https://img.shields.io/docsrs/blprs?label=docs.rs)](https://docs.rs/blprs)
[![crates.io](https://img.shields.io/crates/v/blprs.svg)](https://crates.io/crates/blprs)
[![license](https://img.shields.io/crates/l/blprs.svg)](LICENSE)

`blprs` is a Rust-first translation of the pyBLP API for estimating differentiated
product demand. It keeps pyBLP’s concepts—problems, formulations, integration schemes,
moments—while leaning on Rust’s performance, deterministic parallelism, and type safety.
The crate currently implements the demand side of random-coefficient logit models and is
actively expanding toward full parity.

## Overview

`blprs` implements the demand side of Berry–Levinsohn–Pakes (BLP) estimation for
random coefficients logit models. The API tracks pyBLP concepts (problems,
formulations, integrations, moments) so users can port notebooks and scripts with minimal
friction. Supply-side estimation, optimal instruments, counterfactual engines, and other
advanced features are actively under development.

## Installation

Add the crate from crates.io:

```sh
cargo add blprs
```

or depend on the repository while the API stabilises:

```toml
[dependencies]
blprs = { git = "https://github.com/emaadmanzoor/blprs" }
```

## Quick example

```rust
use blprs::{ContractionOptions, Problem, ProblemOptions, WeightingMatrix};
use blprs::data::ProductDataBuilder;
use blprs::integration::SimulationDraws;
use nalgebra::{DMatrix, DVector};

let market_ids = vec!["m1".to_string(), "m1".to_string(), "m2".to_string()];
let shares = DVector::from_vec(vec![0.3, 0.2, 0.4]);
let x1 = DMatrix::from_row_slice(3, 2, &[1.0, 10.0, 1.0, 15.0, 1.0, 12.0]);
let x2 = DMatrix::from_row_slice(3, 1, &[10.0, 15.0, 12.0]);

let products = ProductDataBuilder::new(market_ids, shares)
    .x1(x1)
    .x2(x2)
    .build()
    .expect("validated product data");

let draws = SimulationDraws::standard_normal(200, 1, 1234);
let problem = Problem::builder()
    .products(products)
    .draws(draws)
    .options(ProblemOptions::default())
    .build()
    .unwrap();

let sigma = DMatrix::from_row_slice(1, 1, &[2.0]);
let options = ProblemOptions::default()
    .with_contraction(ContractionOptions { tolerance: 1e-10, ..Default::default() })
    .with_weighting(WeightingMatrix::InverseZTZ);

let results = problem.solve_with_options(&sigma, &options).unwrap();
println!("beta = {:?}", results.beta);
```

## Features

- R/pyBLP-style builder surface for configuring problems
- Validated product data with contiguous market partitioning
- Monte Carlo integration with reproducible seeds
- BLP contraction with configurable damping and diagnostics
- Two-step GMM estimator with customizable weighting matrices
- Rich error reporting for data shape issues and solver failures

Planned parity items include:

- Supply-side markups, marginal costs, and conduct alternatives
- Optimal instruments and demographic interactions
- Micro moment support and importance sampling
- Counterfactual engines (mergers, taxes, welfare analysis)
- Extended integration schemes (Halton, Sobol, sparse grids)
- Analytic gradients, clustered standard errors, and bootstrapping

The project roadmap tracks which pyBLP features have landed and what is in
progress.

## Contributing & support

Feedback and pull requests are welcome. Please use the
[GitHub issue tracker](https://github.com/emaadmanzoor/blprs/issues) for bugs or feature
requests.

Development quickstart:

```sh
cargo fmt
cargo clippy
cargo test
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
