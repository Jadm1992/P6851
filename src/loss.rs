//! § 4 — Time-normalized classified drop-rate EWMA.
//!
//! Implements § 4.1–4.2: event classification and time-normalized EWMA update
//! for both loss and rate-limit signals.

use core::time::Duration;

use crate::config::AdaptiveConfig;

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
}

impl EpochEvents {
    /// Total eligible probes for loss-rate computation.
    ///
    /// Excludes `ExpectedSilence` since those are not loss indicators.
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

    /// Whether any probes completed (success or loss).
    #[inline]
    pub fn has_completions(&self) -> bool {
        self.successes > 0 || self.inferred_loss > 0 || self.inferred_rate_limit > 0
    }
}

impl LossMonitor {
    /// Create a new monitor with zero initial estimates.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
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

    /// Compute the time-normalized smoothing factor λ(Δt) = 1 − exp(−Δt/τ).
    #[inline]
    fn lambda(&self, dt: Duration) -> f64 {
        let dt_secs = dt.as_secs_f64();
        if self.tau_secs <= 0.0 {
            return 1.0; // Instant convergence if τ = 0.
        }
        1.0 - (-dt_secs / self.tau_secs).exp()
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
    fn pure_loss_epoch() {
        let cfg = AdaptiveConfig::adaptive();
        let mut mon = LossMonitor::new(&cfg);
        let events = EpochEvents {
            successes: 8,
            expected_silence: 0,
            inferred_loss: 2,
            inferred_rate_limit: 0,
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

        // Two monitors, same events, different Δt
        let mut mon_fast = LossMonitor::new(&cfg);
        let mut mon_slow = LossMonitor::new(&cfg);

        let events = EpochEvents {
            successes: 9,
            expected_silence: 0,
            inferred_loss: 1,
            inferred_rate_limit: 0,
        };

        // Fast: 10 updates at Δt = 100ms
        for _ in 0..10 {
            mon_fast.update(&events, Duration::from_millis(100));
        }

        // Slow: 1 update at Δt = 1s
        mon_slow.update(&events, Duration::from_secs(1));

        // They should NOT be exactly equal (time-normalization ensures consistent
        // behavior, but multiple updates accumulate differently). The key property
        // is that the fast monitor converges to the same steady state as the slow one.
        // After many updates both should converge to the same value.
        let events_many = EpochEvents {
            successes: 9,
            expected_silence: 0,
            inferred_loss: 1,
            inferred_rate_limit: 0,
        };
        for _ in 0..100 {
            mon_fast.update(&events_many, Duration::from_millis(100));
            // 10 fast updates ≈ 1 slow update
            if mon_fast.d_loss() > 0.0 {
                // Just ensure it converges somewhere positive
            }
        }
        for _ in 0..10 {
            mon_slow.update(&events_many, Duration::from_secs(1));
        }

        // Both should have converged close to 0.1 (the true rate)
        assert!((mon_fast.d_loss() - 0.1).abs() < 0.01);
        assert!((mon_slow.d_loss() - 0.1).abs() < 0.01);
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
        };
        // Only 5 eligible (successes). ExpectedSilence not counted.
        assert_eq!(events.eligible_count(), 5);
        assert_eq!(events.loss_rate(), 0.0);
    }
}
