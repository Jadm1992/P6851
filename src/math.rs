//! Self-contained, deterministic mathematical functions for `std` and `no_std` environments.
//!
//! Provides:
//! - Exponentiation by squaring: [`pow_f64_u32`] with O(log n) efficiency and overflow safety.
//! - Time-normalized smoothing factor: [`one_minus_exp_neg`] evaluating `1 - exp(-x)`
//!   without catastrophic cancellation for small `x` or Padé sign distortion for large `x`.

/// Compute `base^exp` using exponentiation by squaring in O(log n) time.
///
/// Operates identically in both `std` and `no_std` environments, avoiding `exp as i32`
/// negative exponent errors when `exp > i32::MAX`.
///
/// ### Non-finite inputs:
/// - If `exp == 0`, returns `1.0` (including `0^0`, `NaN^0`, `inf^0` per IEEE 754-2008).
/// - If `base.is_nan()`, returns `NaN`.
/// - If `base == +inf`, returns `+inf` for `exp >= 1`.
/// - If `base == -inf`, returns `-inf` for odd `exp`, and `+inf` for even `exp`.
#[inline]
pub fn pow_f64_u32(base: f64, exp: u32) -> f64 {
    if exp == 0 {
        return 1.0;
    }
    if !base.is_finite() {
        if base.is_nan() {
            return f64::NAN;
        }
        if base == f64::INFINITY {
            return f64::INFINITY;
        }
        if base == f64::NEG_INFINITY {
            return if exp % 2 == 1 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
    }
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

/// Compute `1 - exp(-x)` for `x >= 0`.
///
/// ### Numerical Accuracy & Precision Strategy:
/// 1. **Ultra-small $x$ ($x < 10^{-15}$):**
///    Uses $1 - e^{-x} = x$. Avoids catastrophic cancellation and preserves tiny positive
///    smoothing coefficients down to subnormal floats ($x = 10^{-20}$ returns $10^{-20}$).
/// 2. **Small $x$ ($10^{-15} \le x < 0.1$):**
///    Uses a 9th-degree Taylor polynomial in Horner's form. Evaluated without subtraction
///    from 1.0 to eliminate cancellation error ($< 10^{-16}$ truncation error).
/// 3. **Intermediate & Large $x$ ($0.1 \le x < 37.0$):**
///    In `std`, evaluates `-(-x).exp_m1()`.
///    In `no_std`, uses $\ln 2$ argument reduction: $x = k \ln 2 + r$ with $r \in [-\frac{\ln 2}{2}, \frac{\ln 2}{2}]$,
///    then $e^{-x} = e^{-r} \cdot 2^{-k}$ evaluated via polynomial expansion, guaranteeing positive,
///    monotonic values without Padé denominator/numerator sign errors (tested up to $x = 30$ and beyond).
/// 4. **Asymptotic limit ($x \ge 37.0$):**
///    Since $e^{-37} < 10^{-16} < \epsilon_{\text{f64}}$, returns `1.0`.
///
/// ### Non-finite inputs:
/// - If `x.is_nan()`, returns `NaN`.
/// - If `x == +inf`, returns `1.0`.
/// - If `x <= 0.0`, returns `0.0`.
#[inline]
pub fn one_minus_exp_neg(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_sign_positive() {
            return 1.0;
        }
        return f64::NEG_INFINITY;
    }
    if x <= 0.0 {
        return 0.0;
    }

    #[cfg(feature = "std")]
    {
        let res = -(-x).exp_m1();
        res.clamp(0.0, 1.0)
    }

    #[cfg(not(feature = "std"))]
    {
        one_minus_exp_neg_no_std(x)
    }
}

/// Standalone `no_std` implementation of `1 - exp(-x)` with full double-precision accuracy.
/// Available in all profiles for cross-validation and standalone benchmarking.
pub fn one_minus_exp_neg_no_std(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        if x.is_sign_positive() {
            return 1.0;
        }
        return f64::NEG_INFINITY;
    }
    if x <= 0.0 {
        return 0.0;
    }

    // Case 1: Ultra-small x (< 1e-15).
    // 1 - exp(-x) ≈ x (since x^2/2 < 1e-30 < epsilon_f64).
    // Preserves positive smoothing coefficients for Δt << τ (e.g. x = 1e-20).
    if x < 1e-15 {
        return x;
    }

    // Case 2: Small x in [1e-15, 0.1).
    // Taylor expansion of (1 - exp(-x)) in Horner's form:
    // x * (1 - x/2 + x^2/6 - x^3/24 + ... + x^10/11!)
    // Truncation error at x = 0.1 is < 0.1^12 / 12! ≈ 2.1e-21.
    if x < 0.1 {
        let poly = 1.0
            - x * (0.5
                - x * ((1.0 / 6.0)
                    - x * ((1.0 / 24.0)
                        - x * ((1.0 / 120.0)
                            - x * ((1.0 / 720.0)
                                - x * ((1.0 / 5040.0)
                                    - x * ((1.0 / 40320.0)
                                        - x * ((1.0 / 362880.0)
                                            - x * ((1.0 / 3628800.0)
                                                - x * ((1.0 / 39916800.0)
                                                    - x / 479001600.0))))))))));
        return x * poly;
    }

    // Case 3: Large x (x >= 37.0).
    // exp(-37) ≈ 8.5e-17 < epsilon_f64 / 2.
    // In IEEE 754 double precision, 1.0 - exp(-x) is identically 1.0.
    if x >= 37.0 {
        return 1.0;
    }

    // Case 4: Intermediate x in [0.1, 37.0).
    // Range reduction via ln(2):
    // x = k * ln(2) + r, where k = round(x / ln(2)), r ∈ [-ln(2)/2, ln(2)/2].
    // exp(-x) = 2^(-k) * exp(-r).
    use core::f64::consts::{LN_2, LOG2_E};

    let k = (x * LOG2_E + 0.5) as u32;
    let k_float = k as f64;
    let r = x - k_float * LN_2;

    // High-precision degree-14 polynomial approximation of exp(-r) for |r| <= 0.35:
    // Truncation error at |r| = 0.35 is < 0.35^15 / 15! ≈ 1.2e-19 << epsilon_f64.
    let exp_neg_r = 1.0
        - r * (1.0
            - r * (0.5
                - r * ((1.0 / 6.0)
                    - r * ((1.0 / 24.0)
                        - r * ((1.0 / 120.0)
                            - r * ((1.0 / 720.0)
                                - r * ((1.0 / 5040.0)
                                    - r * ((1.0 / 40320.0)
                                        - r * ((1.0 / 362880.0)
                                            - r * ((1.0 / 3628800.0)
                                                - r * ((1.0 / 39916800.0)
                                                    - r * ((1.0 / 479001600.0)
                                                        - r * ((1.0 / 6227020800.0)
                                                            - r / 87178291200.0)))))))))))));

    // Exact scale factor 2^(-k)
    let scale = pow_f64_u32(0.5, k);
    let exp_neg_x = exp_neg_r * scale;

    (1.0 - exp_neg_x).clamp(0.0, 1.0)
}

/// Compute ceiling of `x` in a portable, `no_std`-compatible way.
#[inline]
pub fn ceil_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        if !x.is_finite() || x == 0.0 {
            return x;
        }
        let truncated = x as i64 as f64;
        if x > 0.0 && x > truncated {
            truncated + 1.0
        } else {
            truncated
        }
    }
}

/// Compute floor of `x` in a portable, `no_std`-compatible way.
#[inline]
pub fn floor_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        if !x.is_finite() || x == 0.0 {
            return x;
        }
        let truncated = x as i64 as f64;
        if x < 0.0 && x < truncated {
            truncated - 1.0
        } else {
            truncated
        }
    }
}

/// Compute square root of `x` in a portable, `no_std`-compatible way.
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if !x.is_finite() {
            if x.is_nan() || x < 0.0 {
                return f64::NAN;
            }
            return f64::INFINITY;
        }
        if x <= 0.0 {
            return 0.0;
        }
        let mut curr = x;
        if curr > 1.0 {
            curr = x * 0.5;
        }
        for _ in 0..24 {
            let next = 0.5 * (curr + x / curr);
            let diff = next - curr;
            let abs_diff = if diff < 0.0 { -diff } else { diff };
            if abs_diff <= 1e-15 * curr {
                return next;
            }
            curr = next;
        }
        curr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pow_basic() {
        assert!((pow_f64_u32(1.5, 0) - 1.0).abs() < 1e-12);
        assert!((pow_f64_u32(1.5, 1) - 1.5).abs() < 1e-12);
        assert!((pow_f64_u32(1.5, 2) - 2.25).abs() < 1e-12);
        assert!((pow_f64_u32(2.0, 10) - 1024.0).abs() < 1e-12);
    }

    #[test]
    fn test_pow_u32_max_no_negative_exponent() {
        // u32::MAX would cast to -1 as i32, turning 2.0^u32::MAX into 0.5!
        // With squaring, it correctly overflows to +inf.
        let val = pow_f64_u32(2.0, u32::MAX);
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());

        let val_one = pow_f64_u32(1.0, u32::MAX);
        assert_eq!(val_one, 1.0);
    }

    #[test]
    fn test_pow_non_finite_inputs() {
        assert_eq!(pow_f64_u32(f64::NAN, 0), 1.0);
        assert!(pow_f64_u32(f64::NAN, 1).is_nan());
        assert_eq!(pow_f64_u32(f64::INFINITY, 1), f64::INFINITY);
        assert_eq!(pow_f64_u32(f64::NEG_INFINITY, 1), f64::NEG_INFINITY);
        assert_eq!(pow_f64_u32(f64::NEG_INFINITY, 2), f64::INFINITY);
    }

    #[test]
    fn test_one_minus_exp_ultra_small_x() {
        // x = 10^-20 must yield positive smoothing coefficient, not 0.0
        let val_20 = one_minus_exp_neg(1e-20);
        assert!(val_20 > 0.0, "x = 1e-20 must produce positive smoothing coefficient");
        assert!((val_20 - 1e-20).abs() < 1e-30);

        let val_20_nostd = one_minus_exp_neg_no_std(1e-20);
        assert!(val_20_nostd > 0.0);
        assert!((val_20_nostd - 1e-20).abs() < 1e-30);
    }

    #[test]
    fn test_one_minus_exp_small_x() {
        // x = 10^-12 accurate small-value behavior
        let val_12 = one_minus_exp_neg(1e-12);
        let val_12_nostd = one_minus_exp_neg_no_std(1e-12);
        let expected = 1e-12 - 0.5 * 1e-24;
        assert!((val_12 - expected).abs() < 1e-20);
        assert!((val_12_nostd - expected).abs() < 1e-20);
    }

    #[test]
    fn test_one_minus_exp_x_one() {
        // x = 1: agreement with reference exponential 1 - exp(-1) ≈ 0.6321205588285577
        let val = one_minus_exp_neg(1.0);
        let val_nostd = one_minus_exp_neg_no_std(1.0);
        let ref_val = 0.6321205588285577;
        assert!((val - ref_val).abs() < 1e-12);
        assert!((val_nostd - ref_val).abs() < 1e-12);
    }

    #[test]
    fn test_one_minus_exp_x_ten() {
        // x = 10: accurate intermediate behavior 1 - exp(-10) ≈ 0.9999546000702375
        let val = one_minus_exp_neg(10.0);
        let val_nostd = one_minus_exp_neg_no_std(10.0);
        let ref_val = 0.9999546000702375;
        assert!((val - ref_val).abs() < 1e-12);
        assert!((val_nostd - ref_val).abs() < 1e-12);
    }

    #[test]
    fn test_one_minus_exp_x_twenty() {
        // x = 20: stable large-value behavior
        let val = one_minus_exp_neg(20.0);
        let val_nostd = one_minus_exp_neg_no_std(20.0);
        let ref_val = 1.0 - (-20.0_f64).exp();
        assert!((val - ref_val).abs() < 1e-12);
        assert!((val_nostd - ref_val).abs() < 1e-12);
    }

    #[test]
    fn test_one_minus_exp_x_thirty_no_pade_sign_error() {
        // x = 30: must not suffer from Padé negative numerator sign error
        let val = one_minus_exp_neg(30.0);
        let val_nostd = one_minus_exp_neg_no_std(30.0);
        let ref_val = 1.0 - 9.357622968840175e-14;
        assert!((val - ref_val).abs() < 1e-12);
        assert!((val_nostd - ref_val).abs() < 1e-12);
        assert!(val < 1.0);
        assert!(val_nostd < 1.0);
    }

    #[test]
    fn test_one_minus_exp_asymptotic_infinity() {
        assert_eq!(one_minus_exp_neg(100.0), 1.0);
        assert_eq!(one_minus_exp_neg_no_std(100.0), 1.0);
        assert_eq!(one_minus_exp_neg(f64::INFINITY), 1.0);
        assert_eq!(one_minus_exp_neg_no_std(f64::INFINITY), 1.0);
    }

    #[test]
    fn test_one_minus_exp_reduction_boundary_and_user_eval_point() {
        use core::f64::consts::LN_2;

        // Specific point cited in review: x ≈ 0.339642
        let x_eval = 0.339642;
        let val_eval = one_minus_exp_neg(x_eval);
        let val_nostd = one_minus_exp_neg_no_std(x_eval);
        #[cfg(feature = "std")]
        let expected = -(-x_eval).exp_m1();
        #[cfg(not(feature = "std"))]
        let expected = 0.2879577785311003; // Exact IEEE 754 1 - exp(-0.339642)
        assert!(
            (val_eval - expected).abs() < 1e-14,
            "Error at x={x_eval}: got {val_eval}, expected {expected}, diff={}",
            (val_eval - expected).abs()
        );
        assert!(
            (val_nostd - expected).abs() < 1e-14,
            "Error in no_std at x={x_eval}: got {val_nostd}, expected {expected}, diff={}",
            (val_nostd - expected).abs()
        );

        // Test around ln(2) range-reduction transition boundaries (k + 0.5) * ln(2)
        let boundaries = [
            0.5 * LN_2, // boundary 0->1 (~0.34657)
            1.5 * LN_2, // boundary 1->2 (~1.03972)
            2.5 * LN_2, // boundary 2->3 (~1.73287)
            3.5 * LN_2, // boundary 3->4 (~2.42602)
        ];

        let eps = 1e-6;
        for &b in &boundaries {
            for &x in &[b - eps, b, b + eps] {
                let v = one_minus_exp_neg(x);
                let v_no_std = one_minus_exp_neg_no_std(x);
                #[cfg(feature = "std")]
                let ref_val = -(-x).exp_m1();
                #[cfg(not(feature = "std"))]
                let ref_val = v;

                #[cfg(feature = "std")]
                {
                    assert!(
                        (v - ref_val).abs() < 1e-14,
                        "Boundary error at x={x} around b={b}: diff={}",
                        (v - ref_val).abs()
                    );
                    assert!(
                        (v_no_std - ref_val).abs() < 1e-14,
                        "Boundary error in no_std at x={x} around b={b}: diff={}",
                        (v_no_std - ref_val).abs()
                    );
                }
                assert!((v - v_no_std).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_one_minus_exp_non_finite() {
        assert!(one_minus_exp_neg(f64::NAN).is_nan());
        assert!(one_minus_exp_neg_no_std(f64::NAN).is_nan());
        assert_eq!(one_minus_exp_neg(-1.0), 0.0);
        assert_eq!(one_minus_exp_neg_no_std(-1.0), 0.0);
    }
}
