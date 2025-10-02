# blprs

Fast Berry–Levinsohn–Pakes (BLP) estimation for random coefficients logit models, written in Rust. The goal of `blprs` is to provide a feature-complete and performant alternative to [pyBLP](https://github.com/jeffgortmaker/pyblp), accompanied by rich inline documentation and an ergonomic Rust-first API.

> ⚠️ This is an early preview. The demand-side estimator is implemented, but advanced functionality (supply, optimal instruments, bootstrapping, simulation aides, and the full suite of `pyBLP` conveniences) is still under construction.

## Highlights
- idiomatic Rust translation of the core BLP demand estimator: product data validation, simulation draws, contraction mapping, and two-step GMM.
- ready-to-read inline documentation and unit tests that mirror `pyBLP` notebooks.
- modular architecture prepared for supply-side extensions, optimal instruments, and equilibrium pricing models.
- no `unsafe` code; builds on `nalgebra`, `rayon`, and `serde`.

## Getting started
Add the crate once it is published:

```sh
cargo add blprs
```

For now, use a Git dependency while the crate incubates:

```toml
[dependencies]
blprs = { git = "https://github.com/eam398/blprs" }
```

### Example
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

let result = problem.solve_with_options(&sigma, &options).unwrap();
println!("beta = {:?}", result.beta);
println!("GMM objective = {}", result.gmm_value);
```

## API surface
- `data`: builders and validation for product-level matrices (X1, X2, instruments).
- `integration`: Monte Carlo draws with shape checking and seeded reproducibility.
- `demand`: share prediction and the BLP contraction mapping.
- `solving`: solver configuration and diagnostics.
- `estimation`: high-level `BlpProblem` wrapper that mirrors `pyblp.Problem` for the demand side.

## Roadmap
1. Implement supply-side marginal cost recovery and equilibrium pricing.
2. Add optimal instrument routines and support for demographic heterogeneity.
3. Provide full pyBLP-style convenience APIs (formulations, problems, counterfactuals).
4. Publish crate releases, CI, benchmarks, and docs.rs integrations.

Contributions and feature requests are welcome via pull requests or GitHub discussions.

## Development
```sh
cargo fmt
cargo clippy
cargo test
```

## License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.
