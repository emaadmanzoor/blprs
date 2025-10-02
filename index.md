---
layout: page
homepage: true
title: Home
---

{: style="margin-top: 0.5rem; margin-bottom: 0;"}
# blprs

{: style="margin-top: 0.25rem;"}
*Fast Berry–Levinsohn–Pakes estimation in safe, expressive Rust*

blprs mirrors the ergonomics of [pyBLP](https://github.com/jeffgortmaker/pyblp) while
leveraging Rust's performance and safety guarantees. The crate packages market
validation, heterogeneous-consumer integration, the BLP contraction mapping, and
a two-step GMM estimator into a cohesive API that feels natural to Rustaceans.

- Demand-side pipeline compatible with `pyBLP.Problem`
- Strong typing and shape validation on `nalgebra` matrices
- Clear diagnostics and logging around contraction convergence
- Ready for docs.rs, crates.io, and high-performance workflows

<div class="cta-buttons">
  <a class="button primary" href="https://docs.rs/blprs">Read the docs</a>
  <a class="button" href="https://github.com/eam398/blprs">Browse the code</a>
</div>

---

## Why blprs?

- **Performance:** Parallel-ready routines built on `rayon` and cache-friendly
  linear algebra from `nalgebra`.
- **Trustworthy numerics:** Aggressive validation + `thiserror` reporting help
  surface data issues before they reach the optimizer.
- **pyBLP parity roadmap:** Demand-side estimation is here today, with supply,
  counterfactuals, and optimal instruments on the way.

## At a glance

```rust
use blprs::{BlpProblem, ContractionOptions, EstimationOptions};
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
let problem = BlpProblem::new(products, draws).unwrap();

let sigma = DMatrix::from_row_slice(1, 1, &[2.0]);
let options = EstimationOptions::default()
    .with_contraction(ContractionOptions { tolerance: 1e-10, ..Default::default() });

let result = problem.estimate(&sigma, &options).unwrap();
println!("beta = {:?}", result.beta);
```

Need the full API surface? Jump into the [quickstart]({{ '/quickstart/' | relative_url }}) or dive into
the [reference documentation](https://docs.rs/blprs).

## Stay in the loop

- Follow along on [GitHub](https://github.com/eam398/blprs)
- Chat about roadmap ideas in the issue tracker
- Subscribe to crates.io once the inaugural release ships
