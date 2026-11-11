//! § 1 — Outlier-resistant RTT estimator with Karn's algorithm.
//!
//! Implements §§ 1.1–1.3 of the spec: initialization, steady-state EWMA with
//! optional outlier clamping, and Karn's algorithm for retransmission ambiguity.

use core::time::Duration;

use crate::config::AdaptiveConfig;

/// Internal estimator state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// No eligible sample received yet; using `rto_init`.
    Uninitialized,
    /// At least one eligible sample has been processed.
    Initialized,
}

/// RTT estimator implementing § 1.1–1.3.
///
/// All state is deterministic and allocation-free.  Pass RTT samples through
/// [`record_sample`](RttEstimator::record_sample) — only *eligible* samples
/// (initial-transmission, n=0) should be fed; the caller enforces Karn's rule.
#[derive(Clone, Debug)]
pub struct RttEstimator {
    state: State,
    srtt: Duration,
    rttvar: Duration,
    // Snapshot of config params we need — avoids holding a reference.
    alpha: f64,
    beta: f64,
    clamp_k: f64,
    rttvar_min: Duration,
}

/// Snapshot of the current estimator values.
#[derive(Clone, Copy, Debug)]
pub struct RttSnapshot {
    pub srtt: Duration,
    pub rttvar: Duration,
    pub initialized: bool,
}

impl RttEstimator {
    /// Create a new uninitialized estimator from the given config.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        Self {
            state: State::Uninitialized,
            srtt: Duration::ZERO,
            rttvar: Duration::ZERO,
            alpha: cfg.alpha,
            beta: cfg.beta,
            clamp_k: cfg.clamp_k,
            rttvar_min: cfg.rttvar_min,
        }
    }

    /// Whether at least one eligible sample has been recorded.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.state == State::Initialized
    }

    /// Current SRTT (zero if uninitialized).
    #[inline]
    pub fn srtt(&self) -> Duration {
        self.srtt
    }

    /// Current RTTVAR (zero if uninitialized).
    #[inline]
    pub fn rttvar(&self) -> Duration {
        self.rttvar
    }

    /// Return a snapshot of the current state.
    #[inline]
    pub fn snapshot(&self) -> RttSnapshot {
        RttSnapshot {
            srtt: self.srtt,
            rttvar: self.rttvar,
            initialized: self.is_initialized(),
        }
    }

    /// Record an eligible RTT sample (must be n=0; caller enforces Karn's rule).
    ///
    /// Returns the previous snapshot for diagnostics.
    pub fn record_sample(&mut self, rtt: Duration) -> RttSnapshot {
        let prev = self.snapshot();
        match self.state {
            State::Uninitialized => self.initialize(rtt),
            State::Initialized => self.update(rtt),
        }
        prev
    }

    /// Force a re-anchor (used by the change detector § 1.5).
    ///
    /// Sets SRTT and RTTVAR directly, bypassing normal EWMA.
    pub fn force_reanchor(&mut self, new_srtt: Duration, new_rttvar: Duration) {
        self.srtt = new_srtt;
        self.rttvar = if new_rttvar < self.rttvar_min {
            self.rttvar_min
        } else {
            new_rttvar
        };
        self.state = State::Initialized;
    }

    // ── private ──────────────────────────────────────────────────────────

    /// § 1.2 — First eligible sample.
    fn initialize(&mut self, r0: Duration) {
        self.srtt = r0;
        let half = r0 / 2;
        self.rttvar = if half < self.rttvar_min {
            self.rttvar_min
        } else {
            half
        };
        self.state = State::Initialized;
    }

    /// § 1.3 — Steady-state EWMA update.
    fn update(&mut self, r: Duration) {
        let srtt_secs = self.srtt.as_secs_f64();
        let rttvar_secs = self.rttvar.as_secs_f64();
        let r_secs = r.as_secs_f64();

        // e = R − SRTT_old
        let e = r_secs - srtt_secs;

        // e_c = clamp(e, −k·RTTVAR, +k·RTTVAR)
        let e_c = if self.clamp_k.is_infinite() {
            e // Reference profile: unclamped
        } else {
            let bound = self.clamp_k * rttvar_secs;
            clamp_f64(e, -bound, bound)
        };

        // SRTT_new = SRTT_old + α·e_c
        let new_srtt = (srtt_secs + self.alpha * e_c).max(0.0);

        // RTTVAR_new = max(RTTVAR_min, (1−β)·RTTVAR_old + β·|e|)
        // NOTE: RTTVAR uses the UNCLAMPED |e|, not |e_c|.
        let abs_e = e.abs();
        let new_rttvar_secs = (1.0 - self.beta) * rttvar_secs + self.beta * abs_e;
        let rttvar_min_secs = self.rttvar_min.as_secs_f64();
        let new_rttvar = new_rttvar_secs.max(rttvar_min_secs);

        self.srtt = Duration::from_secs_f64(new_srtt);
        self.rttvar = Duration::from_secs_f64(new_rttvar);
    }
}

#[inline]
fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    #[test]
    fn initialization_reference() {
        let cfg = AdaptiveConfig::reference();
        let mut est = RttEstimator::new(&cfg);
        assert!(!est.is_initialized());

        let r0 = Duration::from_millis(100);
        est.record_sample(r0);
        assert!(est.is_initialized());
        assert_eq!(est.srtt(), r0);
        assert_eq!(est.rttvar(), r0 / 2); // 50 ms
    }

    #[test]
    fn initialization_adaptive_rttvar_floor() {
        let cfg = AdaptiveConfig::adaptive();
        let mut est = RttEstimator::new(&cfg);

        // R0 = 1 ms → R0/2 = 0.5 ms < rttvar_min = 1 ms → RTTVAR = 1 ms
        let r0 = Duration::from_millis(1);
        est.record_sample(r0);
        assert_eq!(est.srtt(), r0);
        assert_eq!(est.rttvar(), Duration::from_millis(1));
    }

    #[test]
    fn steady_state_reference_unclamped() {
        let cfg = AdaptiveConfig::reference();
        let mut est = RttEstimator::new(&cfg);
        est.record_sample(Duration::from_millis(100));

        // Send a spike sample 1000 ms — should NOT be clamped in Reference mode
        est.record_sample(Duration::from_millis(1000));

        // SRTT_new = 100 + (1/8)*(1000 - 100) = 100 + 112.5 = 212.5
        let expected_srtt = Duration::from_secs_f64(0.2125);
        let diff = if est.srtt() > expected_srtt {
            est.srtt() - expected_srtt
        } else {
            expected_srtt - est.srtt()
        };
        assert!(diff < Duration::from_micros(10));
    }

    #[test]
    fn steady_state_adaptive_clamped() {
        let cfg = AdaptiveConfig::adaptive();
        let mut est = RttEstimator::new(&cfg);
        est.record_sample(Duration::from_millis(100));

        // RTTVAR = max(1ms, 50ms) = 50ms, k = 2.5 → clamp bound = 125ms
        // e = 1000 - 100 = 900 → e_c = clamp(900, -125, +125) = 125
        // SRTT_new = 100 + (1/8)*125 = 115.625 ms
        est.record_sample(Duration::from_millis(1000));

        let expected_srtt = Duration::from_secs_f64(0.115625);
        let diff = if est.srtt() > expected_srtt {
            est.srtt() - expected_srtt
        } else {
            expected_srtt - est.srtt()
        };
        assert!(diff < Duration::from_micros(10));

        // But RTTVAR still uses unclamped |e| = 900:
        // RTTVAR_new = max(1ms, 0.75*50 + 0.25*900) = max(1, 262.5) = 262.5 ms
        let expected_rttvar = Duration::from_secs_f64(0.2625);
        let diff_v = if est.rttvar() > expected_rttvar {
            est.rttvar() - expected_rttvar
        } else {
            expected_rttvar - est.rttvar()
        };
        assert!(diff_v < Duration::from_micros(10));
    }

    #[test]
    fn force_reanchor() {
        let cfg = AdaptiveConfig::adaptive();
        let mut est = RttEstimator::new(&cfg);
        est.record_sample(Duration::from_millis(100));

        est.force_reanchor(Duration::from_millis(500), Duration::from_millis(50));
        assert_eq!(est.srtt(), Duration::from_millis(500));
        assert_eq!(est.rttvar(), Duration::from_millis(50));
    }

    #[test]
    fn force_reanchor_respects_rttvar_min() {
        let cfg = AdaptiveConfig::adaptive();
        let mut est = RttEstimator::new(&cfg);
        est.record_sample(Duration::from_millis(100));

        // Try to set RTTVAR below floor
        est.force_reanchor(Duration::from_millis(200), Duration::from_micros(500));
        assert_eq!(est.rttvar(), Duration::from_millis(1)); // clamped to rttvar_min
    }
}
