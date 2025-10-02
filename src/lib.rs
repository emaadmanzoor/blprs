//! Fast Berry–Levinsohn–Pakes (BLP) estimation for random coefficients logit models.
//!
//! This crate provides a demand-side estimator that mirrors the public API of
//! [pyBLP](https://github.com/jeffgortmaker/pyblp) while embracing idiomatic Rust.
//! It offers tools to
//!
//! - manage product-level market data (`data` module),
//! - describe simulation draws for heterogeneous consumers (`integration` module),
//! - solve the BLP contraction mapping (`solving` module), and
//! - assemble a two-step GMM estimator (`estimation` module).
//!
//! The implementation focuses on clarity and extensibility. Heavy inline
//! documentation and unit tests illustrate the essential ingredients of BLP:
//! market partitioning, heterogeneous consumer integration, fixed-point solving,
//! and GMM objective evaluation. The goal is to provide a high-quality Rust
//! alternative to `pyBLP` while maintaining feature parity over time.
//!
//! # Quick start
//!
//! ```no_run
//! use blprs::data::{ProductData, ProductDataBuilder};
//! use blprs::integration::SimulationDraws;
//! use blprs::estimation::{BlpProblem, EstimationOptions};
//! use nalgebra::{DMatrix, DVector};
//!
//! // Assume we have N products and K1 linear, K2 nonlinear characteristics.
//! let market_ids = vec!["m1".to_string(), "m1".to_string(), "m2".to_string()];
//! let shares = DVector::from_vec(vec![0.3, 0.2, 0.4]);
//! let x1 = DMatrix::from_row_slice(3, 2, &[1.0, 10.0, 1.0, 15.0, 1.0, 12.0]);
//! let x2 = DMatrix::from_row_slice(3, 1, &[10.0, 15.0, 12.0]);
//! let instruments = x1.clone();
//!
//! let products = ProductDataBuilder::new(market_ids, shares)
//!     .x1(x1)
//!     .x2(x2)
//!     .instruments(instruments)
//!     .build()
//!     .expect("validated product data");
//!
//! let draws = SimulationDraws::standard_normal(200, 1, 1234);
//!
//! let problem = BlpProblem::new(products, draws).expect("well-formed problem");
//! let options = EstimationOptions::default();
//! let sigma = DMatrix::from_row_slice(1, 1, &[2.0]);
//!
//! let result = problem.estimate(&sigma, &options).expect("converged");
//! println!("Estimated betas: {:?}", result.beta);
//! ```
//!
//! The crate is still under heavy development. Supply-side estimation,
//! optimal instruments, and many advanced `pyBLP` options are tracked in the
//! public roadmap.

pub mod data;
pub mod demand;
pub mod error;
pub mod estimation;
pub mod integration;
pub mod solving;

pub use estimation::{BlpProblem, EstimationOptions, EstimationResult, WeightingMatrix};
pub use solving::{ContractionOptions, ContractionSummary};
