//! Self-contained mathematical functions for `no_std` environments.

#[inline]
pub fn pow_f64_u32(base: f64, exp: u32) -> f64 {
    #[cfg(feature = "std")]
    {
        base.powi(exp as i32)
    }
    #[cfg(not(feature = "std"))]
    {
        let mut b = base;
        let mut e = exp;
        let mut result = 1.0;
        while e > 0 {
            if e & 1 == 1 {
                result *= b;
            }
            b *= b;
            e >>= 1;
        }
        result
    }
}

/// Compute 1 - exp(-x) for x >= 0.
///
/// In `no_std`, uses a high-precision Padé approximant for exp(-x) when x is small,
/// and converges smoothly for large x without external libraries.
#[inline]
pub fn one_minus_exp_neg(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    #[cfg(feature = "std")]
    {
        1.0 - (-x).exp()
    }
    #[cfg(not(feature = "std"))]
    {
        if x > 30.0 {
            return 1.0;
        }
        // Padé (3,3) approximation for exp(-x)
        // exp(-x) ≈ (1 - x/2 + x^2/10 - x^3/120) / (1 + x/2 + x^2/10 + x^3/120)
        // For larger x, reduce via e^(-x) = (e^(-x/4))^4
        let x_scaled = x / 4.0;
        let x2 = x_scaled * x_scaled;
        let x3 = x2 * x_scaled;
        let p = 1.0 - 0.5 * x_scaled + 0.1 * x2 - (1.0 / 120.0) * x3;
        let q = 1.0 + 0.5 * x_scaled + 0.1 * x2 + (1.0 / 120.0) * x3;
        let r = p / q;
        let exp_neg = r * r * r * r;
        (1.0 - exp_neg).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow() {
        assert!((pow_f64_u32(1.5, 0) - 1.0).abs() < 1e-12);
        assert!((pow_f64_u32(1.5, 1) - 1.5).abs() < 1e-12);
        assert!((pow_f64_u32(1.5, 2) - 2.25).abs() < 1e-12);
        assert!((pow_f64_u32(2.0, 10) - 1024.0).abs() < 1e-12);
    }

    #[test]
    fn test_one_minus_exp() {
        assert!((one_minus_exp_neg(0.0) - 0.0).abs() < 1e-12);
        let val1 = one_minus_exp_neg(1.0);
        let expected1 = 1.0 - (-1.0_f64).exp();
        assert!((val1 - expected1).abs() < 1e-4);

        let val_large = one_minus_exp_neg(50.0);
        assert_eq!(val_large, 1.0);
    }
}
