//! Product-level data containers and validation utilities used by the BLP estimator.

use std::collections::HashSet;

use nalgebra::{DMatrix, DVector};

use crate::error::{BlpError, Result};

/// Represents product-level data required for BLP estimation.
#[derive(Clone, Debug)]
pub struct ProductData {
    market_ids: Vec<String>,
    shares: DVector<f64>,
    x1: DMatrix<f64>,
    x2: DMatrix<f64>,
    instruments: DMatrix<f64>,
    partition: MarketPartition,
}

impl ProductData {
    /// Creates a `ProductData` instance from validated components.
    pub fn new(
        market_ids: Vec<String>,
        shares: DVector<f64>,
        x1: DMatrix<f64>,
        x2: DMatrix<f64>,
        instruments: DMatrix<f64>,
    ) -> Result<Self> {
        let builder = ProductDataBuilder::new(market_ids, shares)
            .x1(x1)
            .x2(x2)
            .instruments(instruments);
        builder.build()
    }

    /// Number of products across all markets.
    pub fn product_count(&self) -> usize {
        self.shares.len()
    }

    /// Number of linear characteristics (`X1`).
    pub fn linear_dim(&self) -> usize {
        self.x1.ncols()
    }

    /// Number of nonlinear characteristics (`X2`).
    pub fn nonlinear_dim(&self) -> usize {
        self.x2.ncols()
    }

    /// Number of instruments.
    pub fn instrument_dim(&self) -> usize {
        self.instruments.ncols()
    }

    /// Returns a read-only view of the linear characteristics matrix (`X1`).
    pub fn x1(&self) -> &DMatrix<f64> {
        &self.x1
    }

    /// Returns a read-only view of the nonlinear characteristics matrix (`X2`).
    pub fn x2(&self) -> &DMatrix<f64> {
        &self.x2
    }

    /// Returns a read-only view of the instrument matrix (`Z`).
    pub fn instruments(&self) -> &DMatrix<f64> {
        &self.instruments
    }

    /// Returns a read-only view of product market shares.
    pub fn shares(&self) -> &DVector<f64> {
        &self.shares
    }

    /// Provides access to the precomputed market partition.
    pub fn partition(&self) -> &MarketPartition {
        &self.partition
    }

    /// Returns the outside share for the market containing product `i`.
    pub fn outside_share_for_product(&self, product_index: usize) -> f64 {
        let market_idx = self.partition.market_of(product_index);
        self.partition.markets[market_idx].outside_share
    }

    /// Returns the market identifier for a given product index.
    pub fn market_id(&self, product_index: usize) -> &str {
        &self.market_ids[product_index]
    }
}

/// Builder that validates dimensions and market structure before constructing [`ProductData`].
#[derive(Debug)]
pub struct ProductDataBuilder {
    market_ids: Vec<String>,
    shares: DVector<f64>,
    x1: Option<DMatrix<f64>>,
    x2: Option<DMatrix<f64>>,
    instruments: Option<DMatrix<f64>>,
}

impl ProductDataBuilder {
    /// Start building product data from market identifiers and observed shares.
    pub fn new(market_ids: Vec<String>, shares: DVector<f64>) -> Self {
        Self {
            market_ids,
            shares,
            x1: None,
            x2: None,
            instruments: None,
        }
    }

    /// Sets the linear characteristics matrix (`X1`).
    pub fn x1(mut self, matrix: DMatrix<f64>) -> Self {
        self.x1 = Some(matrix);
        self
    }

    /// Sets the nonlinear characteristics matrix (`X2`).
    pub fn x2(mut self, matrix: DMatrix<f64>) -> Self {
        self.x2 = Some(matrix);
        self
    }

    /// Sets the instrument matrix (`Z`).
    pub fn instruments(mut self, matrix: DMatrix<f64>) -> Self {
        self.instruments = Some(matrix);
        self
    }

    /// Finalizes construction after validating shapes and market structure.
    pub fn build(self) -> Result<ProductData> {
        let n = self.market_ids.len();
        if self.shares.len() != n {
            return Err(BlpError::dimension_mismatch(
                "shares length",
                n,
                self.shares.len(),
            ));
        }

        for (index, share) in self.shares.iter().enumerate() {
            if *share <= 0.0 {
                return Err(BlpError::NonPositiveShare {
                    index,
                    share: *share,
                });
            }
        }

        let x1 = self
            .x1
            .ok_or_else(|| BlpError::dimension_mismatch("X1", n, 0))?;
        if x1.nrows() != n {
            return Err(BlpError::dimension_mismatch("X1 rows", n, x1.nrows()));
        }

        let x2 = self.x2.unwrap_or_else(|| DMatrix::zeros(n, 0));
        if x2.nrows() != n {
            return Err(BlpError::dimension_mismatch("X2 rows", n, x2.nrows()));
        }

        let instruments = self.instruments.unwrap_or_else(|| x1.clone());
        if instruments.nrows() != n {
            return Err(BlpError::dimension_mismatch(
                "Z rows",
                n,
                instruments.nrows(),
            ));
        }

        let partition = MarketPartition::new(&self.market_ids, &self.shares)?;

        Ok(ProductData {
            market_ids: self.market_ids,
            shares: self.shares,
            x1,
            x2,
            instruments,
            partition,
        })
    }
}

/// Describes the markets contained in the product data.
#[derive(Clone, Debug)]
pub struct MarketPartition {
    markets: Vec<MarketSegment>,
    product_to_market: Vec<usize>,
}

impl MarketPartition {
    /// Constructs a partition by scanning contiguous market identifiers.
    fn new(market_ids: &[String], shares: &DVector<f64>) -> Result<Self> {
        let n = market_ids.len();
        let mut markets = Vec::new();
        let mut product_to_market = vec![0usize; n];
        let mut seen = HashSet::new();

        let mut start = 0usize;
        while start < n {
            let market_id = market_ids[start].clone();
            if !seen.insert(market_id.clone()) {
                return Err(BlpError::NonContiguousMarket { market_id });
            }

            let mut end = start + 1;
            while end < n && market_ids[end] == market_id {
                end += 1;
            }

            let mut total_share = 0.0f64;
            for product_idx in start..end {
                product_to_market[product_idx] = markets.len();
                total_share += shares[product_idx];
                // Avoid extremely small or negative totals due to rounding errors.
                if !shares[product_idx].is_finite() {
                    return Err(BlpError::NumericalError {
                        context: "share validation",
                    });
                }
            }
            let outside_share = 1.0 - total_share;
            if outside_share <= 0.0 {
                return Err(BlpError::NonPositiveOutsideShare {
                    market_id: market_id.clone(),
                    share: outside_share,
                });
            }

            markets.push(MarketSegment {
                market_id,
                start,
                end,
                outside_share,
            });
            start = end;
        }

        Ok(Self {
            markets,
            product_to_market,
        })
    }

    /// Returns the number of distinct markets.
    pub fn market_count(&self) -> usize {
        self.markets.len()
    }

    /// Iterates over market segments.
    pub fn markets(&self) -> impl Iterator<Item = &MarketSegment> {
        self.markets.iter()
    }

    /// Finds the index of the market containing `product_index`.
    pub fn market_of(&self, product_index: usize) -> usize {
        self.product_to_market[product_index]
    }
}

/// Metadata for a single market.
#[derive(Clone, Debug)]
pub struct MarketSegment {
    /// Identifier carried from the original data.
    market_id: String,
    /// Start index (inclusive) of this market in the flattened product arrays.
    pub(crate) start: usize,
    /// End index (exclusive) of this market.
    pub(crate) end: usize,
    /// Observed outside option share: `1 - sum_j s_j`.
    pub outside_share: f64,
}

impl MarketSegment {
    /// Returns the identifier of the market.
    pub fn id(&self) -> &str {
        &self.market_id
    }

    /// Returns the range of product indices that belong to this market.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    /// Number of products inside the market.
    pub fn product_count(&self) -> usize {
        self.end - self.start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_validates_and_constructs_partition() {
        let market_ids = vec!["m1".to_string(), "m1".to_string(), "m2".to_string()];
        let shares = DVector::from_vec(vec![0.3, 0.2, 0.4]);
        let x1 = DMatrix::from_row_slice(3, 2, &[1.0, 10.0, 1.0, 11.0, 1.0, 12.0]);
        let x2 = DMatrix::from_row_slice(3, 1, &[10.0, 11.0, 12.0]);
        let instruments = x1.clone();

        let data = ProductDataBuilder::new(market_ids, shares)
            .x1(x1)
            .x2(x2)
            .instruments(instruments)
            .build()
            .expect("valid data");

        assert_eq!(data.product_count(), 3);
        assert_eq!(data.partition.market_count(), 2);
        let mut iter = data.partition.markets();
        let first = iter.next().unwrap();
        assert_eq!(first.id(), "m1");
        assert_eq!(first.product_count(), 2);
        assert!(iter.next().is_some());
    }

    #[test]
    fn builder_detects_non_contiguous_market() {
        let market_ids = vec!["m1".to_string(), "m2".to_string(), "m1".to_string()];
        let shares = DVector::from_vec(vec![0.3, 0.3, 0.3]);
        let x1 = DMatrix::from_row_slice(3, 1, &[10.0, 11.0, 12.0]);

        let result = ProductDataBuilder::new(market_ids, shares).x1(x1).build();
        assert!(matches!(result, Err(BlpError::NonContiguousMarket { .. })));
    }
}
