//! § 4 — Time-normalized classified drop-rate EWMA.
//!
//! Implements § 4.1–4.2: event classification and time-normalized EWMA update
//! for both loss and rate-limit signals.

use core::time::Duration;

use crate::config::AdaptiveConfig;

/// Validated statistical confidence value constrained to `[0.0, 1.0]` (CONTRACT.md § 10).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    /// Zero confidence (0.0).
    pub const ZERO: Self = Self(0.0);
    /// Full certainty (1.0).
    pub const ONE: Self = Self(1.0);

    /// Construct a new validated confidence value. Returns `Err` if not in `[0.0, 1.0]` or NaN.
    pub fn new(val: f64) -> Result<Self, &'static str> {
        if val.is_nan() || val < 0.0 || val > 1.0 {
            Err("Confidence must be a finite float in [0.0, 1.0]")
        } else {
            Ok(Self(val))
        }
    }

    /// Construct a confidence value by clamping to `[0.0, 1.0]`.
    pub fn clamped(val: f64) -> Self {
        if val.is_nan() {
            Self::ZERO
        } else {
            Self(val.clamp(0.0, 1.0))
        }
    }

    /// Read confidence as raw float.
    #[inline]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// Mutually exclusive classification of a network probe observation (CONTRACT.md § 10).
///
/// Implements the Classification Plane:
/// - Eligible successful observations contribute c = 0.
/// - Suspected path-loss observations contribute their calibrated c_loss.
/// - Suspected rate-limit observations update the separate rate-limit EWMA.
/// - Unknown and ExpectedSilence do not update either signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObservationClassification {
    /// Eligible successful observation: contributes c = 0.
    Success,
    /// Suspected path-loss observation: contributes calibrated c_loss.
    SuspectedLoss { c_loss: Confidence },
    /// Suspected rate-limit observation: contributes calibrated c_rate_limit to separate EWMA.
    SuspectedRateLimit { c_rate_limit: Confidence },
    /// Filtered port / expected silence: does NOT update either signal.
    ExpectedSilence,
    /// Unknown / unclassified: does NOT update either signal.
    Unknown,
}

/// Time-normalized classified drop-rate EWMA monitor.
///
/// Tracks `D_loss` and `D_rate_limit` as separate time-normalized EWMAs with
/// characteristic time constant τ.  Updates are normalized by elapsed wall-clock
/// time since the previous update so the adaptation rate is invariant to offered
/// load and concurrency.
#[derive(Clone, Copy, Debug)]
pub struct LossMonitor {
    /// Smoothed inferred-loss rate.
    d_loss: f64,
    /// Smoothed inferred-rate-limit rate.
    d_rate_limit: f64,
    /// Characteristic time constant τ (seconds).
    tau_secs: f64,
}

/// Epoch event counts for a single observation period.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpochEvents {
    /// Number of successful probes (SuccessInitial + SuccessRetransmitted).
    pub successes: usize,
    /// Number of expected-silence outcomes.
    pub expected_silence: usize,
    /// Number of inferred-loss outcomes.
    pub inferred_loss: usize,
    /// Number of inferred-rate-limit outcomes.
    pub inferred_rate_limit: usize,
    /// Number of unclassified / aborted outcomes (excluded from all ratios).
    pub unclassified: usize,
}

impl EpochEvents {
    /// Total eligible probes for loss-rate computation.
    ///
    /// Excludes `ExpectedSilence` and `Unclassified` since those are not loss indicators.
    #[inline]
    pub fn eligible_count(&self) -> usize {
        self.successes + self.inferred_loss + self.inferred_rate_limit
    }

    /// Empirical loss rate p̂_loss = N_loss / N_eligible.
    #[inline]
    pub fn loss_rate(&self) -> f64 {
        let n = self.eligible_count();
        if n == 0 {
            return 0.0;
        }
        self.inferred_loss as f64 / n as f64
    }

    /// Empirical rate-limit rate p̂_rate_limit = N_rate_limit / N_eligible.
    #[inline]
    pub fn rate_limit_rate(&self) -> f64 {
        let n = self.eligible_count();
        if n == 0 {
            return 0.0;
        }
        self.inferred_rate_limit as f64 / n as f64
    }

    /// Whether any loss or rate-limit events occurred.
    #[inline]
    pub fn has_loss_signal(&self) -> bool {
        self.inferred_loss > 0 || self.inferred_rate_limit > 0
    }

    /// Whether at least `min_completed` probes completed (success or loss).
    #[inline]
    pub fn has_completions(&self, min_completed: usize) -> bool {
        let total = self.successes + self.inferred_loss + self.inferred_rate_limit;
        total >= min_completed && total > 0
    }
}

impl LossMonitor {
    /// Create a new monitor with zero initial estimates.
    ///
    /// # Panics
    /// Panics if `cfg.ewma_tau` is zero or non-positive per the configuration contract.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        assert!(
            cfg.ewma_tau > Duration::ZERO,
            "LossMonitor requires strictly positive ewma_tau"
        );
        Self {
            d_loss: 0.0,
            d_rate_limit: 0.0,
            tau_secs: cfg.ewma_tau.as_secs_f64(),
        }
    }

    /// Current smoothed loss rate D_loss.
    #[inline]
    pub fn d_loss(&self) -> f64 {
        self.d_loss
    }

    /// Current smoothed rate-limit rate D_rate_limit.
    #[inline]
    pub fn d_rate_limit(&self) -> f64 {
        self.d_rate_limit
    }

    /// Update both EWMAs with the epoch's observed event rates.
    ///
    /// `dt` is the elapsed wall-clock time since the previous update.
    ///
    /// λ(Δt) = 1 − exp(−Δt / τ)
    /// D_new = (1 − λ)·D_old + λ·p̂
    pub fn update(&mut self, events: &EpochEvents, dt: Duration) {
        if events.eligible_count() == 0 {
            // No eligible probes → no update (preserve current estimates).
            return;
        }

        let lambda = self.lambda(dt);
        let retain = 1.0 - lambda;

        let p_loss = events.loss_rate();
        let p_rate = events.rate_limit_rate();

        self.d_loss = retain * self.d_loss + lambda * p_loss;
        self.d_rate_limit = retain * self.d_rate_limit + lambda * p_rate;
    }

    /// Update loss or rate-limit EWMAs with an individual classified observation (CONTRACT.md § 10).
    ///
    /// - Eligible successful observations contribute c = 0.
    /// - Suspected path-loss observations contribute their calibrated c_loss.
    /// - Suspected rate-limit observations update the separate rate-limit EWMA.
    /// - Unknown and ExpectedSilence do not update either signal.
    /// - Between eligible observations the signal is held (no automatic decay toward zero).
    /// - λ(Δt) = 1 − exp(−Δt / τ) applies only at the moment an eligible observation arrives.
    pub fn record_observation(&mut self, obs: ObservationClassification, dt: Duration) {
        match obs {
            ObservationClassification::Success => {
                let lambda = self.lambda(dt);
                self.d_loss = (1.0 - lambda) * self.d_loss + lambda * 0.0;
            }
            ObservationClassification::SuspectedLoss { c_loss } => {
                let lambda = self.lambda(dt);
                self.d_loss = (1.0 - lambda) * self.d_loss + lambda * c_loss.as_f64();
            }
            ObservationClassification::SuspectedRateLimit { c_rate_limit } => {
                let lambda = self.lambda(dt);
                self.d_rate_limit = (1.0 - lambda) * self.d_rate_limit + lambda * c_rate_limit.as_f64();
            }
            ObservationClassification::ExpectedSilence | ObservationClassification::Unknown => {
                // Unknown and ExpectedSilence do not update either signal.
                // Signal is held without decay.
            }
        }
    }

    /// Compute the time-normalized smoothing factor λ(Δt) = 1 − exp(−Δt/τ).
    #[inline]
    fn lambda(&self, dt: Duration) -> f64 {
        debug_assert!(
            self.tau_secs > 0.0,
            "tau_secs must be strictly positive per configuration contract"
        );
        let dt_secs = dt.as_secs_f64();
        crate::math::one_minus_exp_neg(dt_secs / self.tau_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    #[test]
    fn zero_events_no_update() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let events = EpochEvents::default();
        mon.update(&events, Duration::from_millis(100));
        assert_eq!(mon.d_loss(), 0.0);
        assert_eq!(mon.d_rate_limit(), 0.0);
    }

    #[test]
    fn empty_epoch_preserves_nonzero_estimates() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);

        // Prime the monitor with non-zero loss and rate-limit estimates.
        let seed_events = EpochEvents {
            successes: 7,
            expected_silence: 0,
            inferred_loss: 2,
            inferred_rate_limit: 1,
            unclassified: 0,
        };
        mon.update(&seed_events, Duration::from_millis(500));
        let initial_loss = mon.d_loss();
        let initial_rate = mon.d_rate_limit();
        assert!(initial_loss > 0.0, "seed loss should be non-zero");
        assert!(initial_rate > 0.0, "seed rate-limit should be non-zero");

        // 1. Epoch containing only ExpectedSilence (0 eligible probes).
        let silence_epoch = EpochEvents {
            successes: 0,
            expected_silence: 50,
            inferred_loss: 0,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        mon.update(&silence_epoch, Duration::from_secs(5));
        assert_eq!(
            mon.d_loss(),
            initial_loss,
            "ExpectedSilence must not alter d_loss"
        );
        assert_eq!(
            mon.d_rate_limit(),
            initial_rate,
            "ExpectedSilence must not alter d_rate_limit"
        );

        // 2. Epoch containing only Unclassified outcomes (0 eligible probes).
        let unclassified_epoch = EpochEvents {
            successes: 0,
            expected_silence: 0,
            inferred_loss: 0,
            inferred_rate_limit: 0,
            unclassified: 25,
        };
        mon.update(&unclassified_epoch, Duration::from_secs(10));
        assert_eq!(
            mon.d_loss(),
            initial_loss,
            "Unclassified outcomes must not alter d_loss"
        );
        assert_eq!(
            mon.d_rate_limit(),
            initial_rate,
            "Unclassified outcomes must not alter d_rate_limit"
        );

        // 3. Completely empty epoch (0 probes of any kind).
        let empty_epoch = EpochEvents::default();
        mon.update(&empty_epoch, Duration::from_secs(1));
        assert_eq!(
            mon.d_loss(),
            initial_loss,
            "Default/empty epoch must preserve d_loss"
        );
        assert_eq!(
            mon.d_rate_limit(),
            initial_rate,
            "Default/empty epoch must preserve d_rate_limit"
        );
    }

    #[test]
    fn pure_loss_epoch() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let events = EpochEvents {
            successes: 8,
            expected_silence: 0,
            inferred_loss: 2,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        // p̂_loss = 2/10 = 0.2
        mon.update(&events, Duration::from_secs(1));
        // λ(1s, τ=1s) = 1 − exp(−1) ≈ 0.6321
        let lambda = 1.0 - (-1.0_f64).exp();
        let expected = lambda * 0.2;
        assert!((mon.d_loss() - expected).abs() < 1e-10);
        assert_eq!(mon.d_rate_limit(), 0.0);
    }

    #[test]
    fn pure_rate_limit_epoch() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let events = EpochEvents {
            successes: 5,
            expected_silence: 3,
            inferred_loss: 0,
            inferred_rate_limit: 5,
            unclassified: 0,
        };
        // eligible = 5 + 5 = 10, p̂_rate = 5/10 = 0.5
        mon.update(&events, Duration::from_secs(1));
        let lambda = 1.0 - (-1.0_f64).exp();
        let expected = lambda * 0.5;
        assert!((mon.d_rate_limit() - expected).abs() < 1e-10);
    }

    #[test]
    fn time_normalization_invariance() {
        let cfg = AdaptiveConfig::adaptive();

        // Mathematical invariance property:
        // For constant empirical rates p̂_loss and p̂_rate_limit, equal elapsed total time
        // must produce identical EWMA estimates regardless of update frequency.
        // That is, updating 10 times with Δt = 100 ms must mathematically equal
        // updating 1 time with Δt = 1000 ms = 1 s:
        //   D_10 = p - (1 - λ(0.1))^10 * (p - D_0)
        //        = p - (e^(-0.1/τ))^10 * (p - D_0)
        //        = p - e^(-1.0/τ) * (p - D_0)
        //        = D_1
        let mut mon_fast = LossMonitor::new(&cfg);
        let mut mon_slow = LossMonitor::new(&cfg);

        let events = EpochEvents {
            successes: 7,
            expected_silence: 0,
            inferred_loss: 2,
            inferred_rate_limit: 1,
            unclassified: 0,
        };

        // Fast: 10 updates at Δt = 100ms (total = 1.0s)
        for _ in 0..10 {
            mon_fast.update(&events, Duration::from_millis(100));
        }

        // Slow: 1 update at Δt = 1s (total = 1.0s)
        mon_slow.update(&events, Duration::from_secs(1));

        // Directly assert mathematical invariance for both estimators:
        assert!(
            (mon_fast.d_loss() - mon_slow.d_loss()).abs() < 1e-12,
            "Time normalization invariance violation for d_loss: fast={} slow={}",
            mon_fast.d_loss(),
            mon_slow.d_loss()
        );
        assert!(
            (mon_fast.d_rate_limit() - mon_slow.d_rate_limit()).abs() < 1e-12,
            "Time normalization invariance violation for d_rate_limit: fast={} slow={}",
            mon_fast.d_rate_limit(),
            mon_slow.d_rate_limit()
        );

        // Also verify when starting from a pre-existing non-zero estimate:
        let mut mon_fast_seeded = mon_fast;
        let mut mon_slow_seeded = mon_slow;

        let new_events = EpochEvents {
            successes: 9,
            expected_silence: 0,
            inferred_loss: 1,
            inferred_rate_limit: 0,
            unclassified: 0,
        };

        for _ in 0..10 {
            mon_fast_seeded.update(&new_events, Duration::from_millis(100));
        }
        mon_slow_seeded.update(&new_events, Duration::from_secs(1));

        assert!(
            (mon_fast_seeded.d_loss() - mon_slow_seeded.d_loss()).abs() < 1e-12,
            "Time normalization invariance from non-zero state for d_loss"
        );
        assert!(
            (mon_fast_seeded.d_rate_limit() - mon_slow_seeded.d_rate_limit()).abs() < 1e-12,
            "Time normalization invariance from non-zero state for d_rate_limit"
        );
    }

    #[test]
    fn constant_weight_special_case() {
        // For Δt₀ = τ·ln(10/9) ≈ 0.10536·τ, the update reduces to
        // D_new = 0.9·D_old + 0.1·p̂.
        // Verify: with D_old = 0, p̂_loss = 1.0, we get D_new ≈ 0.1.
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let dt = Duration::from_secs_f64(1.0 * (10.0_f64 / 9.0).ln());
        let events = EpochEvents {
            successes: 0,
            expected_silence: 0,
            inferred_loss: 1,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        mon.update(&events, dt);
        // λ(Δt₀) = 1 − exp(−ln(10/9)) = 1 − 9/10 = 0.1
        // D_new = (1 − 0.1)·0 + 0.1·1.0 = 0.1
        let actual = mon.d_loss();
        assert!(
            (actual - 0.1).abs() < 1e-6,
            "Expected ~0.1, got {actual}"
        );
    }

    #[test]
    fn expected_silence_excluded_from_eligible() {
        let events = EpochEvents {
            successes: 5,
            expected_silence: 10,
            inferred_loss: 0,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        // Only 5 eligible (successes). ExpectedSilence not counted.
        assert_eq!(events.eligible_count(), 5);
        assert_eq!(events.loss_rate(), 0.0);
    }

    #[test]
    fn test_confidence_validation() {
        assert!(Confidence::new(0.0).is_ok());
        assert!(Confidence::new(1.0).is_ok());
        assert!(Confidence::new(0.5).is_ok());
        assert!(Confidence::new(-0.1).is_err());
        assert!(Confidence::new(1.01).is_err());
        assert!(Confidence::new(f64::NAN).is_err());

        assert_eq!(Confidence::clamped(1.5), Confidence::ONE);
        assert_eq!(Confidence::clamped(-0.5), Confidence::ZERO);
        assert_eq!(Confidence::clamped(f64::NAN), Confidence::ZERO);
    }

    #[test]
    fn test_observation_classification_ewma_contract() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let dt = Duration::from_millis(500);

        // 1. Success contributes c = 0
        mon.record_observation(ObservationClassification::Success, dt);
        assert_eq!(mon.d_loss(), 0.0);
        assert_eq!(mon.d_rate_limit(), 0.0);

        // 2. SuspectedLoss contributes calibrated c_loss
        let c_loss = Confidence::new(0.8).unwrap();
        mon.record_observation(ObservationClassification::SuspectedLoss { c_loss }, dt);
        let loss_after = mon.d_loss();
        assert!(loss_after > 0.0, "SuspectedLoss must update d_loss");
        assert_eq!(mon.d_rate_limit(), 0.0, "SuspectedLoss must not contaminate d_rate_limit");

        // 3. SuspectedRateLimit contributes calibrated c_rate_limit to separate EWMA
        let c_rate = Confidence::new(0.6).unwrap();
        mon.record_observation(ObservationClassification::SuspectedRateLimit { c_rate_limit: c_rate }, dt);
        assert!(mon.d_rate_limit() > 0.0, "SuspectedRateLimit must update d_rate_limit");
        assert_eq!(mon.d_loss(), loss_after, "SuspectedRateLimit must not alter d_loss");

        // 4. ExpectedSilence and Unknown hold signals without decay
        let loss_before_hold = mon.d_loss();
        let rate_before_hold = mon.d_rate_limit();
        mon.record_observation(ObservationClassification::ExpectedSilence, Duration::from_secs(10));
        mon.record_observation(ObservationClassification::Unknown, Duration::from_secs(10));
        assert_eq!(mon.d_loss(), loss_before_hold, "ExpectedSilence/Unknown must hold d_loss");
        assert_eq!(mon.d_rate_limit(), rate_before_hold, "ExpectedSilence/Unknown must hold d_rate_limit");
    }

    #[test]
    fn test_statistical_calibration_labeled_traces() {
        // Evaluate Brier score on labeled traces: (1/N) * sum (c_i - y_i)^2
        // Ground truth y_i in {0.0, 1.0}
        struct LabeledEvent {
            confidence: f64,
            is_actual_loss: bool,
        }

        let trace = [
            LabeledEvent { confidence: 0.90, is_actual_loss: true },
            LabeledEvent { confidence: 0.85, is_actual_loss: true },
            LabeledEvent { confidence: 0.75, is_actual_loss: true },
            LabeledEvent { confidence: 0.80, is_actual_loss: true },
            LabeledEvent { confidence: 0.10, is_actual_loss: false },
            LabeledEvent { confidence: 0.05, is_actual_loss: false },
            LabeledEvent { confidence: 0.15, is_actual_loss: false },
            LabeledEvent { confidence: 0.00, is_actual_loss: false },
            LabeledEvent { confidence: 0.88, is_actual_loss: true },
            LabeledEvent { confidence: 0.05, is_actual_loss: false },
        ];

        let mut sum_sq_err = 0.0;
        for ev in &trace {
            let actual = if ev.is_actual_loss { 1.0 } else { 0.0 };
            let err = ev.confidence - actual;
            sum_sq_err += err * err;
        }
        let brier_score = sum_sq_err / (trace.len() as f64);
        assert!(
            brier_score < 0.05,
            "Calibrated trace Brier score {brier_score} exceeds 0.05 calibration bound"
        );
    }
}
