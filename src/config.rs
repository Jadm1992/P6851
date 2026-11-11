//! Configuration types, profile constructors, and probe outcome/retry types.

use core::time::Duration;

// ─── Profile Mode ──────────────────────────────────────────────────────────

/// Selects between strict RFC 6298 semantics and the adaptive scanner profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileMode {
    /// Strict RFC 6298: unclamped SRTT, 1 s RTO floor, 2× backoff, no change detector,
    /// fixed concurrency (caller-supplied ceiling), passive loss telemetry only.
    Reference,
    /// Scanner-optimized: clamped SRTT, aggressive RTO bounds, γ-backoff, change detector,
    /// epoch-based AIMD concurrency with dual ceilings, time-normalized loss EWMA.
    Adaptive,
}

// ─── Policy for Rate-Limit vs Packet-Loss in AIMD ──────────────────────────

/// Policy determining how rate-limiting vs packet-loss events affect AIMD window backoff.
///
/// ### Precedence Rule (§ 3.2):
/// When both inferred packet loss and rate-limiting signals occur in the **same observation epoch**:
/// - Under `Shared`: both events use the standard `aimd_delta`.
/// - Under `SeparateRateLimit { rate_limit_delta }`: the more conservative (smaller) multiplier,
///   `aimd_delta.min(rate_limit_delta)`, is applied. This alleviates both network congestion and
///   target rate-limiting while strictly preserving **at most one multiplicative decrease per epoch**.
/// - Under `LossOnly`: only packet loss triggers reduction; rate-limiting is ignored for window backoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AimdLossPolicy {
    /// Both packet loss and rate-limiting trigger window decrease using `aimd_delta`.
    /// This is the baseline contract specified in § 3.2.
    Shared,
    /// Rate-limit signals trigger window reduction with a separate delta multiplier
    /// (e.g. 0.7 instead of 0.5), allowing distinct response tuning.
    SeparateRateLimit { rate_limit_delta: f64 },
    /// Only inferred packet drops trigger AIMD window decrease; rate-limiting is
    /// tracked in telemetry only (or handled at higher application layers).
    LossOnly,
}

// ─── Re-Anchor Strategy for Change Detector ────────────────────────────────

/// Strategy for computing new RTTVAR during a forced unclamped re-anchor (§ 1.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReanchorVariationStrategy {
    /// Spec default: RTTVAR = max(RTTVAR_min, |M_e - SRTT_baseline| / 2)
    MedianStep,
    /// Empirical spread: RTTVAR = max(RTTVAR_min, MAD(samples, M_e))
    /// where MAD is the Mean Absolute Deviation of the violating epoch samples around M_e.
    EpochSampleVariation,
}

// ─── Configuration Error ───────────────────────────────────────────────────

/// Errors encountered when validating an [`AdaptiveConfig`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConfigError {
    /// `alpha` must be finite and in `(0.0, 1.0]`.
    InvalidAlpha(f64),
    /// `beta` must be finite and in `(0.0, 1.0]`.
    InvalidBeta(f64),
    /// `clamp_k` must be in `[2.0, 3.0]` for Adaptive, or positive infinity for Reference.
    InvalidClampK { mode: ProfileMode, value: f64 },
    /// `backoff_gamma` must be in `[1.4, 1.6]` for Adaptive, or exactly `2.0` for Reference.
    InvalidBackoffGamma { mode: ProfileMode, value: f64 },
    /// `aimd_delta` must be in `[0.4, 0.5]`.
    InvalidAimdDelta(f64),
    /// `rate_limit_delta` must be finite and in `(0.0, 1.0]`.
    InvalidRateLimitDelta(f64),
    /// `rto_min` must be greater than zero.
    ZeroRtoMin,
    /// `rto_max` must be greater than or equal to `rto_min`.
    InvalidRtoBounds { min: Duration, max: Duration },
    /// `rto_init` must be within `[rto_min, rto_max]`.
    InvalidRtoInit { init: Duration, min: Duration, max: Duration },
    /// `ewma_tau` must be greater than zero.
    ZeroEwmaTau,
    /// `change_m_epochs` must be at least 1.
    ZeroChangeMEpochs,
    /// `min_epoch_samples` must be at least 1.
    ZeroMinEpochSamples,
    /// `min_completed_probes` must be at least 1.
    ZeroMinCompletedProbes,
    /// Destination concurrency limits must satisfy `1 <= c_min <= c_max`.
    InvalidConcurrencyDest { min: usize, max: usize },
    /// Global concurrency limits must satisfy `1 <= c_min <= c_max`.
    InvalidConcurrencyGlobal { min: usize, max: usize },
    /// Epoch durations must be positive and ordered: `0 < e_min <= e_max` and `e_min <= e_init <= e_max`.
    InvalidEpochDurations { min: Duration, max: Duration, init: Duration },
    /// Clock granularity must be greater than zero.
    ZeroClockGranularity,
    /// `change_c` must be finite and in `[2.5, 4.0]` for Adaptive (or `>= 0.0` for Reference).
    InvalidChangeC { mode: ProfileMode, value: f64 },
    /// `rttvar_min` must be `0` for Reference, or `> 0` for Adaptive.
    InvalidRttvarMin { mode: ProfileMode, min: Duration },
    /// `p95_window_size` must be in `[64, 128]` (§ 1.4).
    InvalidP95WindowSize(usize),
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAlpha(v) => write!(f, "alpha must be finite and in (0, 1], got {v}"),
            Self::InvalidBeta(v) => write!(f, "beta must be finite and in (0, 1], got {v}"),
            Self::InvalidClampK { mode, value } => write!(
                f,
                "clamp_k for {mode:?} must be {} got {value}",
                if *mode == ProfileMode::Reference {
                    "positive infinity,"
                } else {
                    "in [2.0, 3.0],"
                }
            ),
            Self::InvalidBackoffGamma { mode, value } => write!(
                f,
                "backoff_gamma for {mode:?} must be {} got {value}",
                if *mode == ProfileMode::Reference {
                    "2.0,"
                } else {
                    "in [1.4, 1.6],"
                }
            ),
            Self::InvalidAimdDelta(v) => write!(f, "aimd_delta must be in [0.4, 0.5], got {v}"),
            Self::InvalidRateLimitDelta(v) => {
                write!(f, "rate_limit_delta must be finite and in (0, 1], got {v}")
            }
            Self::ZeroRtoMin => write!(f, "rto_min must be greater than zero"),
            Self::InvalidRtoBounds { min, max } => {
                write!(f, "rto_max ({max:?}) must be >= rto_min ({min:?})")
            }
            Self::InvalidRtoInit { init, min, max } => {
                write!(f, "rto_init ({init:?}) must be within [{min:?}, {max:?}]")
            }
            Self::ZeroEwmaTau => write!(f, "ewma_tau must be greater than zero"),
            Self::ZeroChangeMEpochs => write!(f, "change_m_epochs must be at least 1"),
            Self::ZeroMinEpochSamples => write!(f, "min_epoch_samples must be at least 1"),
            Self::ZeroMinCompletedProbes => write!(f, "min_completed_probes must be at least 1"),
            Self::InvalidConcurrencyDest { min, max } => {
                write!(f, "c_min_dest ({min}) must be >= 1 and <= c_max_dest ({max})")
            }
            Self::InvalidConcurrencyGlobal { min, max } => {
                write!(f, "c_min_global ({min}) must be >= 1 and <= c_max_global ({max})")
            }
            Self::InvalidEpochDurations { min, max, init } => write!(
                f,
                "epoch durations must satisfy 0 < min ({min:?}) <= init ({init:?}) <= max ({max:?})"
            ),
            Self::ZeroClockGranularity => write!(f, "clock_granularity must be greater than zero"),
            Self::InvalidChangeC { mode, value } => write!(
                f,
                "change_c for {mode:?} must be {} got {value}",
                if *mode == ProfileMode::Adaptive {
                    "in [2.5, 4.0],"
                } else {
                    ">= 0.0,"
                }
            ),
            Self::InvalidRttvarMin { mode, min } => write!(
                f,
                "rttvar_min for {mode:?} must be {} got {min:?}",
                if *mode == ProfileMode::Reference {
                    "zero,"
                } else {
                    "greater than zero,"
                }
            ),
            Self::InvalidP95WindowSize(w) => {
                write!(f, "p95_window_size must be in [64, 128], got {w}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

// ─── Adaptive Config ────────────────────────────────────────────────────────

/// Complete parameterization of the estimator, backoff, concurrency, and loss
/// tracking subsystems. All fields are `pub` for trace-driven replay.
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
    pub reanchor_strategy: ReanchorVariationStrategy,

    // § 3 — AIMD concurrency
    pub aimd_delta: f64, // multiplicative decrease factor [0.4, 0.5]
    pub aimd_loss_policy: AimdLossPolicy,
    pub min_completed_probes: usize, // minimum completions in epoch to qualify for +1 increase (default 1)
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
        let cfg = Self {
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
            reanchor_strategy: ReanchorVariationStrategy::MedianStep,
            aimd_delta: 0.5,
            aimd_loss_policy: AimdLossPolicy::Shared,
            min_completed_probes: 1,
            c_min_dest: 1,
            c_max_dest: 256,
            c_min_global: 1,
            c_max_global: 1024,
            ewma_tau: Duration::from_secs(1),
            p95_window_size: 64,
            epoch_e_min: Duration::from_millis(10),
            epoch_e_max: Duration::from_secs(1),
            epoch_e_init: Duration::from_millis(100),
        };
        debug_assert!(cfg.validate().is_ok());
        cfg
    }

    /// Scanner-optimized adaptive profile with sensible defaults.
    pub fn adaptive() -> Self {
        let cfg = Self {
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
            reanchor_strategy: ReanchorVariationStrategy::MedianStep,
            aimd_delta: 0.5,
            aimd_loss_policy: AimdLossPolicy::Shared,
            min_completed_probes: 1,
            c_min_dest: 1,
            c_max_dest: 256,
            c_min_global: 1,
            c_max_global: 1024,
            ewma_tau: Duration::from_secs(1),
            p95_window_size: 64,
            epoch_e_min: Duration::from_millis(10),
            epoch_e_max: Duration::from_secs(1),
            epoch_e_init: Duration::from_millis(100),
        };
        debug_assert!(cfg.validate().is_ok());
        cfg
    }

    /// Validate configuration and return `Ok(cfg)` if all constraints are satisfied.
    pub fn try_new(cfg: Self) -> Result<Self, ConfigError> {
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate that all configuration parameters satisfy their mathematical constraints.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // EWMA coefficients: finite, in (0, 1]
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(ConfigError::InvalidAlpha(self.alpha));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 || self.beta > 1.0 {
            return Err(ConfigError::InvalidBeta(self.beta));
        }

        // Clamp k: Adaptive in [2.0, 3.0]; Reference: positive infinity
        match self.mode {
            ProfileMode::Reference => {
                if !self.clamp_k.is_infinite() || !self.clamp_k.is_sign_positive() {
                    return Err(ConfigError::InvalidClampK {
                        mode: self.mode,
                        value: self.clamp_k,
                    });
                }
            }
            ProfileMode::Adaptive => {
                if !self.clamp_k.is_finite() || self.clamp_k < 2.0 || self.clamp_k > 3.0 {
                    return Err(ConfigError::InvalidClampK {
                        mode: self.mode,
                        value: self.clamp_k,
                    });
                }
            }
        }

        // Backoff gamma: Adaptive in [1.4, 1.6]; Reference: exactly 2.0
        match self.mode {
            ProfileMode::Reference => {
                if (self.backoff_gamma - 2.0).abs() > 1e-6 {
                    return Err(ConfigError::InvalidBackoffGamma {
                        mode: self.mode,
                        value: self.backoff_gamma,
                    });
                }
            }
            ProfileMode::Adaptive => {
                if !self.backoff_gamma.is_finite()
                    || self.backoff_gamma < 1.4
                    || self.backoff_gamma > 1.6
                {
                    return Err(ConfigError::InvalidBackoffGamma {
                        mode: self.mode,
                        value: self.backoff_gamma,
                    });
                }
            }
        }

        // AIMD delta: in [0.4, 0.5]
        if !self.aimd_delta.is_finite() || self.aimd_delta < 0.4 || self.aimd_delta > 0.5 {
            return Err(ConfigError::InvalidAimdDelta(self.aimd_delta));
        }

        // Rate limit delta (when configured)
        if let AimdLossPolicy::SeparateRateLimit { rate_limit_delta } = self.aimd_loss_policy
            && (!rate_limit_delta.is_finite() || rate_limit_delta <= 0.0 || rate_limit_delta > 1.0)
        {
            return Err(ConfigError::InvalidRateLimitDelta(rate_limit_delta));
        }

        // RTO bounds
        if self.rto_min == Duration::ZERO {
            return Err(ConfigError::ZeroRtoMin);
        }
        if self.rto_max < self.rto_min {
            return Err(ConfigError::InvalidRtoBounds {
                min: self.rto_min,
                max: self.rto_max,
            });
        }
        if self.rto_init < self.rto_min || self.rto_init > self.rto_max {
            return Err(ConfigError::InvalidRtoInit {
                init: self.rto_init,
                min: self.rto_min,
                max: self.rto_max,
            });
        }

        // Loss EWMA time constant
        if self.ewma_tau == Duration::ZERO {
            return Err(ConfigError::ZeroEwmaTau);
        }

        // Change detector parameters
        if self.change_m_epochs == 0 {
            return Err(ConfigError::ZeroChangeMEpochs);
        }
        if self.min_epoch_samples == 0 {
            return Err(ConfigError::ZeroMinEpochSamples);
        }
        if self.min_completed_probes == 0 {
            return Err(ConfigError::ZeroMinCompletedProbes);
        }

        // Concurrency bounds: c_min >= 1, c_min <= c_max
        if self.c_min_dest == 0 || self.c_min_dest > self.c_max_dest {
            return Err(ConfigError::InvalidConcurrencyDest {
                min: self.c_min_dest,
                max: self.c_max_dest,
            });
        }
        if self.c_min_global == 0 || self.c_min_global > self.c_max_global {
            return Err(ConfigError::InvalidConcurrencyGlobal {
                min: self.c_min_global,
                max: self.c_max_global,
            });
        }

        // Epoch durations: positive and ordered
        if self.epoch_e_min == Duration::ZERO
            || self.epoch_e_max < self.epoch_e_min
            || self.epoch_e_init < self.epoch_e_min
            || self.epoch_e_init > self.epoch_e_max
        {
            return Err(ConfigError::InvalidEpochDurations {
                min: self.epoch_e_min,
                max: self.epoch_e_max,
                init: self.epoch_e_init,
            });
        }

        // Clock granularity
        if self.clock_granularity == Duration::ZERO {
            return Err(ConfigError::ZeroClockGranularity);
        }

        // Change detector threshold multiplier c
        match self.mode {
            ProfileMode::Adaptive => {
                if !self.change_c.is_finite() || self.change_c < 2.5 || self.change_c > 4.0 {
                    return Err(ConfigError::InvalidChangeC {
                        mode: self.mode,
                        value: self.change_c,
                    });
                }
            }
            ProfileMode::Reference => {
                if !self.change_c.is_finite() || self.change_c < 0.0 {
                    return Err(ConfigError::InvalidChangeC {
                        mode: self.mode,
                        value: self.change_c,
                    });
                }
            }
        }

        // RTTVAR minimum floor: 0 for Reference, > 0 for Adaptive
        match self.mode {
            ProfileMode::Reference => {
                if self.rttvar_min != Duration::ZERO {
                    return Err(ConfigError::InvalidRttvarMin {
                        mode: self.mode,
                        min: self.rttvar_min,
                    });
                }
            }
            ProfileMode::Adaptive => {
                if self.rttvar_min == Duration::ZERO {
                    return Err(ConfigError::InvalidRttvarMin {
                        mode: self.mode,
                        min: self.rttvar_min,
                    });
                }
            }
        }

        // Rolling P95 window: W ∈ [64, 128] (§ 1.4)
        if self.p95_window_size < 64 || self.p95_window_size > 128 {
            return Err(ConfigError::InvalidP95WindowSize(self.p95_window_size));
        }

        Ok(())
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
    /// Unclassified or aborted probe (e.g. caller cancelled, socket unroutable).
    /// Excluded from loss calculations and AIMD transitions.
    Unclassified,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_defaults_pass_validation() {
        assert!(AdaptiveConfig::reference().validate().is_ok());
        assert!(AdaptiveConfig::adaptive().validate().is_ok());
        assert!(AdaptiveConfig::try_new(AdaptiveConfig::adaptive()).is_ok());
    }

    #[test]
    fn invalid_alpha_beta() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.alpha = 0.0;
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidAlpha(0.0)));
        cfg.alpha = 1.5;
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidAlpha(1.5)));

        cfg.alpha = 0.125;
        cfg.beta = -0.1;
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidBeta(-0.1)));
    }

    #[test]
    fn invalid_clamp_k() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.clamp_k = 1.5; // Adaptive must be [2.0, 3.0]
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidClampK { .. })));

        let mut ref_cfg = AdaptiveConfig::reference();
        ref_cfg.clamp_k = 2.5; // Reference must be infinity
        assert!(matches!(ref_cfg.validate(), Err(ConfigError::InvalidClampK { .. })));
    }

    #[test]
    fn invalid_backoff_gamma() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.backoff_gamma = 1.2; // Adaptive must be [1.4, 1.6]
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidBackoffGamma { .. })));

        let mut ref_cfg = AdaptiveConfig::reference();
        ref_cfg.backoff_gamma = 1.5; // Reference must be 2.0
        assert!(matches!(ref_cfg.validate(), Err(ConfigError::InvalidBackoffGamma { .. })));
    }

    #[test]
    fn invalid_aimd_delta() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.aimd_delta = 0.6; // must be in [0.4, 0.5]
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidAimdDelta(0.6)));
    }

    #[test]
    fn invalid_rate_limit_delta() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.aimd_loss_policy = AimdLossPolicy::SeparateRateLimit {
            rate_limit_delta: 0.0,
        };
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidRateLimitDelta(0.0)));
    }

    #[test]
    fn invalid_rto_bounds() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.rto_min = Duration::ZERO;
        assert_eq!(cfg.validate(), Err(ConfigError::ZeroRtoMin));

        cfg.rto_min = Duration::from_secs(5);
        cfg.rto_max = Duration::from_secs(3); // max < min
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidRtoBounds { .. })));

        cfg.rto_min = Duration::from_millis(100);
        cfg.rto_max = Duration::from_secs(3);
        cfg.rto_init = Duration::from_millis(50); // init < min
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidRtoInit { .. })));
    }

    #[test]
    fn invalid_concurrency_limits() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.c_min_dest = 0;
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidConcurrencyDest { .. })));

        cfg.c_min_dest = 100;
        cfg.c_max_dest = 50; // min > max
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidConcurrencyDest { .. })));
    }

    #[test]
    fn invalid_epoch_durations() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.epoch_e_min = Duration::ZERO;
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidEpochDurations { .. })));

        cfg.epoch_e_min = Duration::from_secs(2);
        cfg.epoch_e_max = Duration::from_secs(1); // max < min
        assert!(matches!(cfg.validate(), Err(ConfigError::InvalidEpochDurations { .. })));
    }

    #[test]
    fn invalid_p95_window_size() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.p95_window_size = 32; // < 64
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidP95WindowSize(32)));
        cfg.p95_window_size = 256; // > 128
        assert_eq!(cfg.validate(), Err(ConfigError::InvalidP95WindowSize(256)));
    }

    #[test]
    fn invalid_rttvar_min() {
        let mut ref_cfg = AdaptiveConfig::reference();
        ref_cfg.rttvar_min = Duration::from_millis(1); // Reference must have 0
        assert!(matches!(ref_cfg.validate(), Err(ConfigError::InvalidRttvarMin { .. })));

        let mut adapt_cfg = AdaptiveConfig::adaptive();
        adapt_cfg.rttvar_min = Duration::ZERO; // Adaptive must have > 0
        assert!(matches!(adapt_cfg.validate(), Err(ConfigError::InvalidRttvarMin { .. })));
    }
}
