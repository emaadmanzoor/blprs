---
layout: page
title: Guides & Roadmap
permalink: /guides/
---

## Design pillars

1. **Parity with pyBLP.** blprs intentionally mirrors the concepts and naming in
   pyBLP so existing notebooks and papers translate with minimal friction.
2. **Deterministic numerics.** Every routine guards against silent shape
   mismatches, negative shares, and NaNs by returning structured `BlpError`
   variants.
3. **Composable building blocks.** Each module (`data`, `integration`, `demand`,
   `solving`, `estimation`) can be used independently for custom workflows or
   plugged together for the end-to-end estimator.

## Current status

- [x] Demand-side estimation with two-step GMM
- [x] Monte Carlo integration helpers and contraction diagnostics
- [ ] Supply-side marginal cost recovery and pricing
- [ ] Optimal instruments and demographic heterogeneity
- [ ] Counterfactual solvers and equilibrium simulation utilities

## Roadmap highlights

| Milestone | Description |
| --- | --- |
| `v0.1` | Publish demand-side crate on crates.io with docs.rs coverage |
| `v0.2` | Add supply-side pricing, recover marginal costs, expose markup tooling |
| `v0.3` | Implement optimal instruments, bootstrapping, and simulation aides |
| `v0.4` | Counterfactual engine with price, cost, and merger experiments |

## Learning resources

- [Docs.rs reference](https://docs.rs/blprs) — canonical API documentation.
- [GitHub repository](https://github.com/eam398/blprs) — issues, discussions, and
  release notes.
- [pyBLP documentation](https://pyblp.readthedocs.io/en/stable/) — the Python
  inspiration with detailed theory derivations.

Have ideas or want to help? Reach out via a GitHub issue or open a pull request.
