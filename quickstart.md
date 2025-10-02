---
layout: page
title: Quickstart
permalink: /quickstart/
---

## Install

```sh
cargo add blprs
```

Until the crate is published on crates.io, depend directly on Git:

```toml
[dependencies]
blprs = { git = "https://github.com/eam398/blprs" }
```

## Prepare data

blprs expects product-level observations grouped by market:

```rust
use blprs::data::ProductDataBuilder;
use nalgebra::{DMatrix, DVector};

let market_ids = vec!["nyc-2010".into(), "nyc-2010".into(), "sf-2010".into()];
let shares = DVector::from_vec(vec![0.30, 0.24, 0.42]);
let x1 = DMatrix::from_row_slice(3, 2, &[1.0, 10.0, 1.0, 12.0, 1.0, 15.0]);
let x2 = DMatrix::from_row_slice(3, 1, &[10.0, 12.0, 15.0]);

let data = ProductDataBuilder::new(market_ids, shares)
    .x1(x1)
    .x2(x2)
    .build()?;
```

The builder enforces contiguous market identifiers, positive shares, and matching
matrix shapes to prevent subtle numerical failures later.

## Simulate heterogeneous tastes

```rust
use blprs::integration::SimulationDraws;

let draws = SimulationDraws::standard_normal(2000, 2, 42);
```

Weights always sum to one and are validated for positivity.

## Solve the demand system

```rust
use blprs::{BlpProblem, ContractionOptions, EstimationOptions};
use nalgebra::DMatrix;

let problem = BlpProblem::new(data, draws)?;
let sigma = DMatrix::from_diagonal_element(2, 2, 1.5);
let options = EstimationOptions::default()
    .with_contraction(ContractionOptions { damping: 0.7, ..Default::default() });

let result = problem.estimate(&sigma, &options)?;
println!("beta = {:?}", result.beta);
println!("GMM objective = {}", result.gmm_value);
```

`result.contraction` reports iteration counts and fixed-point gaps so you can
profile performance or loosen tolerances for large counterfactual sweeps.

## What's next?

- Explore the [guides]({{ '/guides/' | relative_url }}) for design notes and roadmap items.
- Read the [full API reference](https://docs.rs/blprs) for every struct and method.
- File an [issue](https://github.com/eam398/blprs/issues) if you need a missing
  pyBLP feature.

