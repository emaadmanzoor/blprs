---
layout: page
title: Theory
permalink: /theory/
---

## Overview

Berry–Levinsohn–Pakes (BLP) demand estimation models differentiated products by
combining aggregate market shares with heterogeneous consumer preferences. The
`blprs` crate keeps the standard BLP structure while exposing ergonomic Rust
abstractions for each component: product data validation, integration over
random coefficients, the contraction mapping that recovers mean utilities, and a
two-step GMM estimator.

This page sketches the key identities that guide the implementation. Wherever
possible we point to the corresponding `blprs` module so theory and code stay in
sync.

## Consumers and utility

Each consumer draw $i$ who observes product $j$ in market $t$ has indirect
utility

$$
U_{ijt} = \delta_{jt} + \mu_{ijt}(\theta_2, \nu_i) + \varepsilon_{ijt},
$$

where:

- $\delta_{jt}$ is the mean utility of product $j$ in market $t$.
- $\mu_{ijt}$ captures deviations from the market mean via random coefficients
  with parameters $\theta_2$ (called `sigma` in the library) and consumer draw
  $\nu_i$.
- $\varepsilon_{ijt}$ follows an i.i.d. Type-I Extreme Value distribution.

In `blprs`, market-level primitives live in [`data::ProductData`](https://docs.rs/blprs/latest/blprs/data/struct.ProductData.html), and the
heterogeneous draws $\nu_i$ are packaged by [`integration::SimulationDraws`](https://docs.rs/blprs/latest/blprs/integration/struct.SimulationDraws.html).

## Market shares

Conditional on a draw $\nu_i$, the probability that the consumer purchases
product $j$ in market $t$ is

$$
P_{ijt}(\delta, \theta_2, \nu_i) = \frac{\exp\left(\delta_{jt} + \mu_{ijt}\right)}{1 + \sum_{k \in \mathcal{J}_t} \exp\left(\delta_{kt} + \mu_{ikt}\right)}.
$$

Aggregate shares integrate these probabilities over the distribution of draws:

$$
\hat{s}_{jt}(\delta, \theta_2)
  = \int P_{ijt}(\delta, \theta_2, \nu) \, dF(\nu)
  \approx \sum_{i=1}^I w_i P_{ijt}(\delta, \theta_2, \nu_i),
$$

with weights $w_i$ that sum to one. The approximation above is exactly what
[`demand::predict_shares`](https://docs.rs/blprs/latest/blprs/demand/fn.predict_shares.html) computes using the draws supplied to the
[`BlpProblem`](https://docs.rs/blprs/latest/blprs/struct.BlpProblem.html).

## Contraction mapping

Observed shares $s_{jt}$ give us equilibrium conditions that pin down mean
utilities. BLP proposes iterating on

$$
\delta^{(m+1)}_{jt}
  = \delta^{(m)}_{jt}
    + \ln s_{jt}
    - \ln \hat{s}_{jt}(\delta^{(m)}, \theta_2).
$$

Under standard assumptions this fixed-point iteration converges to the unique
vector $\delta(\theta_2)$ that rationalizes the data. In `blprs`,
[`demand::solve_delta`](https://docs.rs/blprs/latest/blprs/demand/fn.solve_delta.html) implements this contraction with configurable tolerance,
maximum iterations, damping, and minimum share guards exposed through
[`ContractionOptions`](https://docs.rs/blprs/latest/blprs/struct.ContractionOptions.html). Convergence diagnostics are returned as a
[`ContractionSummary`](https://docs.rs/blprs/latest/blprs/struct.ContractionSummary.html).

## Linear parameters and unobserved quality

Once we obtain $\delta$, linear taste parameters $\beta$ follow from the
moment condition

$$
E[Z_t^\top \xi_t] = 0, \qquad
\xi_t = \delta_t - X_{1t} \beta,
$$

where $X_{1t}$ are the observed linear characteristics and $Z_t$ are instruments.
`blprs` solves for $\beta$ via two-stage least squares inside
[`estimation::compute_linear_parameters`](https://docs.rs/blprs/latest/blprs/estimation/fn.compute_linear_parameters.html) (invoked internally by
[`BlpProblem::estimate`](https://docs.rs/blprs/latest/blprs/struct.BlpProblem.html#method.estimate)). The resulting structural shocks $\xi$ are
exposed to users so supply-side models or diagnostics can be layered on later.

## GMM objective

The demand-side estimator minimizes the quadratic form

$$
Q(\theta_2) = g(\theta_2)^\top W g(\theta_2), \qquad
 g(\theta_2) = \frac{1}{T} \sum_{t} Z_t^\top \xi_t(\theta_2),
$$

with $W$ a positive-definite weighting matrix. The current implementation offers
an inverse $Z^\top Z$ weighting by default and accepts custom positive-definite
matrices through [`WeightingMatrix`](https://docs.rs/blprs/latest/blprs/enum.WeightingMatrix.html). Future releases will incorporate heteroskedasticity-robust and optimal weighting strategies.

## Connecting theory to the codebase

| Theory object | Library representation |
| --- | --- |
| Product characteristics $(X_1, X_2)$ | `ProductData::x1`, `ProductData::x2` |
| Market shares $s_{jt}$ | `ProductData::shares` |
| Draws $\nu_i$, weights $w_i$ | `SimulationDraws` |
| Mean utility $\delta$ | Output of `solve_delta`, stored in `EstimationResult::delta` |
| Structural error $\xi$ | `EstimationResult::xi` |
| GMM objective $Q$ | `EstimationResult::gmm_value` |

The [`guides`]({{ '/guides/' | relative_url }}) page tracks upcoming additions such as supply-side
pricing, optimal instruments, and counterfactual engines.

## Further reading

- Berry, Levinsohn, and Pakes (1995), *Automobile Prices in Market Equilibrium*.
- [pyBLP documentation](https://pyblp.readthedocs.io/en/stable/theory.html) for a
  comprehensive Python treatment.
- Nevo (2000), *A Practitioner's Guide to Estimation of Random Coefficients Logit Models*.

