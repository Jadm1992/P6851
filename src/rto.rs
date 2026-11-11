//! § 2 — Bounded RTO computation & per-probe retry schedule.
//!
//! Implements §§ 2.1–2.3: base RTO calculation, latched probe-local backoff
//! schedule, Karn reset semantics, and destination-level backoff integration.

use core::time::Duration;

use crate::config::AdaptiveConfig;

/// Latched per-probe timeout schedule created at initial transmission (n == 0).
///
/// The base RTO is frozen at dispatch time so that global SRTT/RTTVAR updates
/// only affect newly dispatched probes (§ 2.2).
#[derive(Clone, Copy, Debug)]
pub struct ProbeRetrySchedule {
    /// The base RTO latched at n=0 transmission.
    pub base_rto: Duration,
    /// Global RTO ceiling.
    pub rto_max: Duration,
    /// Backoff multiplier (2.0 for Reference, γ ∈ [1.4, 1.6] for Adaptive).
    pub gamma: f64,
}

impl ProbeRetrySchedule {
    /// Create a new retry schedule by latching the current RTO from the config
    /// and RTT estimator state (§ 2.1 & § 2.2).
    ///
    /// If the estimator is uninitialized, uses `rto_init`.
    pub fn new(cfg: &AdaptiveConfig, srtt: Option<Duration>, rttvar: Option<Duration>) -> Self {
        let base_rto = match (srtt, rttvar) {
            (Some(s), Some(v)) => cfg.compute_rto(s, v),
            _ => cfg.rto_init,
        };
        Self {
            base_rto,
            rto_max: cfg.rto_max,
            gamma: cfg.backoff_gamma,
        }
    }

    /// Create a retry schedule for a specific destination incorporating its
    /// destination-level backoff state (§ 2.3).
    ///
    /// When destination `d` has accumulated `consecutive_backoffs` unacknowledged losses/retries,
    /// newly dispatched probes latch an escalated base timeout:
    /// `base_rto = min(rto_max, base_rto * gamma^consecutive_backoffs)`.
    ///
    /// This backed-off timeout multiplier is cleared and reset to fresh RTO_0
    /// only upon receiving an eligible (n = 0) acknowledgment (§ 2.3).
    pub fn for_destination(
        cfg: &AdaptiveConfig,
        srtt: Option<Duration>,
        rttvar: Option<Duration>,
        consecutive_backoffs: u32,
    ) -> Self {
        let mut sched = Self::new(cfg, srtt, rttvar);
        if consecutive_backoffs > 0 {
            sched.base_rto = sched.rto_for_attempt(consecutive_backoffs);
        }
        sched
    }

    /// Apply destination-level consecutive backoffs (§ 2.3) to escalate the base timeout.
    pub fn with_destination_backoff(mut self, consecutive_backoffs: u32) -> Self {
        if consecutive_backoffs > 0 {
            self.base_rto = self.rto_for_attempt(consecutive_backoffs);
        }
        self
    }

    /// Compute the timeout for retry attempt `n` (§ 2.2).
    ///
    /// - n=0: base_rto (initial transmission)
    /// - n≥1: min(rto_max, base_rto · γⁿ)
    ///
    /// Reference profile (γ=2): RTO_n = min(RTO_max, RTO_0 · 2ⁿ)
    /// Adaptive profile (γ=1.5): RTO_n = min(RTO_max, RTO_0 · 1.5ⁿ)
    ///
    /// ### Safety & Overflow Protection:
    /// Pre-checks whether `gamma^n` exceeds `rto_max / base_rto` in floating-point before
    /// performing `Duration` arithmetic. This prevents `Duration::mul_f64` panics on extreme
    /// retry attempts (e.g., n = 64, n = u32::MAX).
    #[inline]
    pub fn rto_for_attempt(&self, n: u32) -> Duration {
        if n == 0 {
            return self.base_rto.min(self.rto_max);
        }

        let base_secs = self.base_rto.as_secs_f64();
        let max_secs = self.rto_max.as_secs_f64();

        if base_secs <= 0.0 || max_secs <= 0.0 {
            return self.rto_max;
        }

        let max_factor = max_secs / base_secs;

        // If n is large, gamma^n will easily exceed max_factor for gamma >= 1.4.
        // Cap n to 64 to avoid float overflow or unnecessary cycles in pow computation.
        let clamped_n = n.min(64);
        let factor = crate::math::pow_f64_u32(self.gamma, clamped_n);

        // Pre-check: if the scaling factor exceeds max_factor, clamp immediately
        // without attempting Duration multiplication.
        if !factor.is_finite() || factor >= max_factor {
            return self.rto_max;
        }

        let scaled_secs = base_secs * factor;
        if !scaled_secs.is_finite() || scaled_secs >= max_secs {
            return self.rto_max;
        }

        Duration::from_secs_f64(scaled_secs).min(self.rto_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    #[test]
    fn uninitialized_uses_rto_init() {
        let cfg = AdaptiveConfig::reference();
        let sched = ProbeRetrySchedule::new(&cfg, None, None);
        assert_eq!(sched.base_rto, Duration::from_secs(1));
    }

    #[test]
    fn reference_backoff_doubles() {
        let cfg = AdaptiveConfig::reference();
        let sched = ProbeRetrySchedule::new(&cfg, None, None);

        assert_eq!(sched.rto_for_attempt(0), Duration::from_secs(1));
        assert_eq!(sched.rto_for_attempt(1), Duration::from_secs(2));
        assert_eq!(sched.rto_for_attempt(2), Duration::from_secs(4));
        assert_eq!(sched.rto_for_attempt(3), Duration::from_secs(8));
        assert_eq!(sched.rto_for_attempt(4), Duration::from_secs(16));
        assert_eq!(sched.rto_for_attempt(5), Duration::from_secs(32));
        // n=6: 2^6 = 64 > 60 → clamped to 60
        assert_eq!(sched.rto_for_attempt(6), Duration::from_secs(60));
    }

    #[test]
    fn adaptive_backoff_gamma() {
        let cfg = AdaptiveConfig::adaptive();
        // SRTT = 200ms, RTTVAR = 50ms → RTO = clamp(200 + max(1, 200), 100, 3000) = 400ms
        let sched = ProbeRetrySchedule::new(
            &cfg,
            Some(Duration::from_millis(200)),
            Some(Duration::from_millis(50)),
        );

        let rto0 = sched.rto_for_attempt(0);
        assert_eq!(rto0, Duration::from_millis(400));

        // n=1: 400 * 1.5 = 600ms
        let rto1 = sched.rto_for_attempt(1);
        assert_eq!(rto1, Duration::from_millis(600));

        // n=2: 400 * 2.25 = 900ms
        let rto2 = sched.rto_for_attempt(2);
        assert_eq!(rto2, Duration::from_millis(900));

        // n=3: 400 * 3.375 = 1350ms
        let rto3 = sched.rto_for_attempt(3);
        assert_eq!(rto3, Duration::from_millis(1350));

        // n=4: 400 * 5.0625 = 2025ms
        let rto4 = sched.rto_for_attempt(4);
        assert_eq!(rto4, Duration::from_millis(2025));

        // n=5: 400 * 7.59375 = 3037.5ms > 3000 → clamped
        let rto5 = sched.rto_for_attempt(5);
        assert_eq!(rto5, Duration::from_secs(3));
    }

    #[test]
    fn rto_computation_with_estimator_state() {
        let cfg = AdaptiveConfig::adaptive();
        // SRTT = 100ms, RTTVAR = 25ms
        // RTO = clamp(100 + max(1, 100), 100, 3000) = clamp(200, 100, 3000) = 200ms
        let sched = ProbeRetrySchedule::new(
            &cfg,
            Some(Duration::from_millis(100)),
            Some(Duration::from_millis(25)),
        );
        assert_eq!(sched.base_rto, Duration::from_millis(200));
    }

    #[test]
    fn rto_respects_rto_min_floor() {
        let cfg = AdaptiveConfig::adaptive();
        // Very low SRTT and RTTVAR → RTO should be clamped to rto_min = 100ms
        let sched = ProbeRetrySchedule::new(
            &cfg,
            Some(Duration::from_millis(5)),
            Some(Duration::from_micros(100)),
        );
        assert_eq!(sched.base_rto, cfg.rto_min);
    }

    #[test]
    fn extreme_attempt_counts_do_not_panic() {
        let cfg = AdaptiveConfig::reference();
        let sched = ProbeRetrySchedule::new(&cfg, None, None);

        // n = 64: 2^64 would panic Duration::mul_f64 if not pre-checked
        let rto64 = sched.rto_for_attempt(64);
        assert_eq!(rto64, sched.rto_max);

        // n = u32::MAX
        let rto_max_u32 = sched.rto_for_attempt(u32::MAX);
        assert_eq!(rto_max_u32, sched.rto_max);

        // n = 1000
        let rto1000 = sched.rto_for_attempt(1000);
        assert_eq!(rto1000, sched.rto_max);
    }

    #[test]
    fn destination_level_backoff_schedule() {
        let cfg = AdaptiveConfig::adaptive();
        // base RTO = 400ms, gamma = 1.5
        let sched_normal = ProbeRetrySchedule::new(
            &cfg,
            Some(Duration::from_millis(200)),
            Some(Duration::from_millis(50)),
        );
        assert_eq!(sched_normal.base_rto, Duration::from_millis(400));

        // Destination with 2 consecutive backoffs: 400 * 1.5^2 = 900ms
        let sched_dest = ProbeRetrySchedule::for_destination(
            &cfg,
            Some(Duration::from_millis(200)),
            Some(Duration::from_millis(50)),
            2,
        );
        assert_eq!(sched_dest.base_rto, Duration::from_millis(900));

        // Retry 1 of this probe will be: min(3000, 900 * 1.5) = 1350ms
        assert_eq!(sched_dest.rto_for_attempt(1), Duration::from_millis(1350));
    }
}
