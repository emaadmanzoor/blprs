//! Lightweight placeholder for pyBLP-style formulas.
//!
//! A full translation of pyBLP will eventually parse and evaluate expressions like
//! `"0 + prices + x1 + x2"`. For now, this type stores the raw expression so that
//! builders and configuration structs can accept user intent with parity to the
//! Python API.

/// Represents a symbolic specification of linear or nonlinear characteristics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Formulation {
    expression: String,
}

impl Formulation {
    /// Creates a new formulation from any string-like expression.
    pub fn new<S: Into<String>>(expression: S) -> Self {
        Self {
            expression: expression.into(),
        }
    }

    /// Returns the original expression string.
    pub fn expression(&self) -> &str {
        &self.expression
    }
}

impl From<&str> for Formulation {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Formulation {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::Formulation;

    #[test]
    fn stores_expression() {
        let f = Formulation::new("0 + prices + x1");
        assert_eq!(f.expression(), "0 + prices + x1");
    }
}
