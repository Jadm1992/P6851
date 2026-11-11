//! Configuration types, profile constructors, and probe outcome/retry types.

use core::time::Duration;

// ─── Profile Mode ──────────────────────────────────────────────────────────

/// Selects between strict RFC 6298 semantics and the adaptive scanner profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileMode {
    /// Strict RFC 6298: unclamped SRTT, 1 s RTO floor, 2× backoff, no change detector.
    Reference,
    /// Scanner-optimized: clamped SRTT, aggressive RTO bounds, γ-backoff, change detector.
    Adaptive,
}

// ─── Adaptive Config ────────────────────────────────────────────────────────

/// Complete parameterization of the estimator, backoff, concurrency, and loss
/// tracking subsystems.  All fields are `pub` for trace-driven replay.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    pub mode: ProfileMode,

    // § 1.3 — EWMA coefficients
    pub alpha: f64, // 1/8
    pub beta: f64,  // 1/4

    // § 1.3 — Outlier clamp
    pub clamp_k: f64,          // ∞ for Reference; [2.0, 3.0] for Adaptive
    pub rttvar_min: Duration,  // 0 for Reference; e.g. 1 ms for Adaptive

    // § 2.1 — Clock granularity & RTO bounds
    pub clock_granularity: Duration,
    pub rto_init: Duration,
    pub rto_min: Duration,
    pub rto_max: Duration,

    // § 2.2 — Backoff multiplier
    pub backoff_gamma: f64, // 2.0 for Reference; [1.4, 1.6] for Adaptive

    // § 1.5 — Change detector
    pub change_c: f64,          // deviation threshold multiplier (default 3.0)
    pub change_m_epochs: u32,   // consecutive violations before re-anchor (default 3)
    pub min_epoch_samples: usize, // minimum samples for a valid epoch (default 3)

    // § 3 — AIMD concurrency
    pub aimd_delta: f64, // multiplicative decrease factor [0.4, 0.5]
    pub c_min_dest: usize,
    pub c_max_dest: usize,
    pub c_min_global: usize,
    pub c_max_global: usize,

    // § 4 — Time-normalized loss EWMA
    pub ewma_tau: Duration, // characteristic time constant (default 1 s)

    // § 1.4 — P95 rolling window
    pub p95_window_size: usize, // W ∈ [64, 128], default 64

    // § 3.1 — Epoch duration bounds
    pub epoch_e_min: Duration,
    pub epoch_e_max: Duration,
    pub epoch_e_init: Duration,
}

impl AdaptiveConfig {
    /// Strict RFC 6298 reference profile.
    pub fn reference() -> Self {
        Self {
            mode: ProfileMode::Reference,
            alpha: 0.125,
            beta: 0.25,
            clamp_k: f64::INFINITY,
            rttvar_min: Duration::ZERO,
            clock_granularity: Duration::from_millis(1),
            rto_init: Duration::from_secs(1),
            rto_min: Duration::from_secs(1),
            rto_max: Duration::from_secs(60),
            backoff_gamma: 2.0,
            change_c: 3.0,
            change_m_epochs: 3,
            min_epoch_samples: 3,
            aimd_delta: 0.5,
            c_min_dest: 1,
            c_max_dest: 256,
            c_min_global: 1,
            c_max_global: 1024,
            ewma_tau: Duration::from_secs(1),
            p95_window_size: 64,
            epoch_e_min: Duration::from_millis(10),
            epoch_e_max: Duration::from_secs(1),
            epoch_e_init: Duration::from_millis(100),
        }
    }

    /// Scanner-optimized adaptive profile with sensible defaults.
    pub fn adaptive() -> Self {
        Self {
            mode: ProfileMode::Adaptive,
            alpha: 0.125,
            beta: 0.25,
            clamp_k: 2.5,
            rttvar_min: Duration::from_millis(1),
            clock_granularity: Duration::from_millis(1),
            rto_init: Duration::from_millis(500),
            rto_min: Duration::from_millis(100),
            rto_max: Duration::from_secs(3),
            backoff_gamma: 1.5,
            change_c: 3.0,
            change_m_epochs: 3,
            min_epoch_samples: 3,
            aimd_delta: 0.5,
            c_min_dest: 1,
            c_max_dest: 256,
            c_min_global: 1,
            c_max_global: 1024,
            ewma_tau: Duration::from_secs(1),
            p95_window_size: 64,
            epoch_e_min: Duration::from_millis(10),
            epoch_e_max: Duration::from_secs(1),
            epoch_e_init: Duration::from_millis(100),
        }
    }

    /// Compute the base RTO from current SRTT / RTTVAR.
    ///
    /// RTO₀ = clamp(SRTT + max(G, 4·RTTVAR), RTO_min, RTO_max)
    #[inline]
    pub fn compute_rto(&self, srtt: Duration, rttvar: Duration) -> Duration {
        let four_var = rttvar.saturating_mul(4);
        let jitter = if self.clock_granularity > four_var {
            self.clock_granularity
        } else {
            four_var
        };
        let raw = srtt.saturating_add(jitter);
        clamp_duration(raw, self.rto_min, self.rto_max)
    }

    /// Epoch duration = clamp(SRTT, E_min, E_max).
    /// Returns `epoch_e_init` when SRTT is `None` (uninitialized).
    #[inline]
    pub fn epoch_duration(&self, srtt: Option<Duration>) -> Duration {
        match srtt {
            Some(s) => clamp_duration(s, self.epoch_e_min, self.epoch_e_max),
            None => self.epoch_e_init,
        }
    }
}

// ─── Probe Outcome ─────────────────────────────────────────────────────────

/// Mutually exclusive outcome of a completed or expired probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Successful response on initial transmission (n == 0). Eligible for RTT update.
    SuccessInitial { rtt: Duration },
    /// Successful response on retransmitted probe (n >= 1). Karn's rule rejects RTT sample.
    SuccessRetransmitted { attempt: u32 },
    /// Expected silence (e.g., filtered port). Excluded from loss & AIMD backoff.
    ExpectedSilence,
    /// Inferred network packet drop. Triggers AIMD decrease (once per epoch) & D_loss update.
    InferredLoss,
    /// Inferred target/middlebox rate limit. Triggers AIMD decrease & D_rate_limit update.
    InferredRateLimit,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Clamp a `Duration` to `[lo, hi]`.
#[inline]
pub(crate) fn clamp_duration(v: Duration, lo: Duration, hi: Duration) -> Duration {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
