//! § 1.5 — Reproducible regime-change detector.
//!
//! Detects sustained step-increases in baseline path latency that the clamped
//! EWMA cannot track, and forces an unclamped re-anchoring.

use core::time::Duration;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::config::{AdaptiveConfig, ReanchorVariationStrategy};

/// Per-epoch accumulator for collecting eligible RTT samples.
#[derive(Clone, Debug, Default)]
pub struct EpochAccumulator {
    samples: Vec<Duration>,
}

impl EpochAccumulator {
    /// Create a new empty epoch accumulator.
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    /// Record an eligible RTT sample in this epoch.
    pub fn push(&mut self, rtt: Duration) {
        self.samples.push(rtt);
    }

    /// Number of samples collected in this epoch.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether any samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Compute the median of collected samples.
    ///
    /// Returns `None` if no samples are present.
    pub fn median(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        if n % 2 == 1 {
            Some(sorted[n / 2])
        } else {
            // Average the two middle elements
            let a = sorted[n / 2 - 1];
            let b = sorted[n / 2];
            Some((a + b) / 2)
        }
    }

    /// Compute Mean Absolute Deviation (MAD) of samples around the given median.
    ///
    /// \[ \text{MAD} = \frac{1}{N} \sum_{i=1}^N |R_i - M_e| \]
    ///
    /// Evaluated in full nanosecond precision with checked accumulation and safe
    /// conversion back to [`Duration`] without integer truncation.
    pub fn mean_absolute_deviation(&self, median: Duration) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let med_nanos = median.as_nanos();
        let mut sum_diff_nanos: u128 = 0;
        for &s in &self.samples {
            let diff = s.as_nanos().abs_diff(med_nanos);
            sum_diff_nanos = sum_diff_nanos.saturating_add(diff);
        }
        let avg_diff_nanos = sum_diff_nanos / (self.samples.len() as u128);
        let secs = (avg_diff_nanos / 1_000_000_000) as u64;
        let subsec_nanos = (avg_diff_nanos % 1_000_000_000) as u32;
        Duration::new(secs, subsec_nanos)
    }

    /// Drain and reset for the next epoch.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Reference to the collected samples in this epoch.
    #[inline]
    pub fn samples(&self) -> &[Duration] {
        &self.samples
    }
}

/// Regime-change detector state.
///
/// Tracks consecutive violation epochs and triggers forced re-anchoring when
/// `m` consecutive epochs have median RTT significantly above the baseline.
#[derive(Clone, Debug)]
pub struct ChangeDetector {
    /// Consecutive violation counter `v`.
    violations: u32,
    /// Deviation threshold multiplier `c`.
    change_c: f64,
    /// Required consecutive violations `m`.
    change_m: u32,
    /// Minimum samples per epoch to be considered valid.
    min_samples: usize,
    /// RTTVAR floor for re-anchoring.
    rttvar_min: Duration,
    /// Strategy for computing RTTVAR upon forced re-anchoring.
    reanchor_strategy: ReanchorVariationStrategy,
    /// Whether the detector is enabled (disabled for Reference profile).
    enabled: bool,
    /// Retained per-epoch medians M_1..M_m across consecutive violating epochs.
    retained_medians: Vec<Duration>,
    /// Retained eligible samples R_1..R_N across consecutive violating epochs.
    retained_samples: Vec<Duration>,
}

/// Result of closing an epoch through the change detector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochCloseResult {
    /// Not enough samples; epoch was skipped (counter preserved).
    InsufficientSamples,
    /// Epoch did not violate the baseline; counter was reset to 0.
    Normal,
    /// Epoch violated the baseline; counter incremented but threshold not reached.
    Violation { consecutive: u32 },
    /// Threshold reached: force a re-anchoring with these values.
    ReAnchor {
        new_srtt: Duration,
        new_rttvar: Duration,
    },
}

impl ChangeDetector {
    /// Create from config.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        Self {
            violations: 0,
            change_c: cfg.change_c,
            change_m: cfg.change_m_epochs,
            min_samples: cfg.min_epoch_samples,
            rttvar_min: cfg.rttvar_min,
            reanchor_strategy: cfg.reanchor_strategy,
            enabled: cfg.mode == crate::config::ProfileMode::Adaptive,
            retained_medians: Vec::new(),
            retained_samples: Vec::new(),
        }
    }

    /// Current consecutive violation count.
    #[inline]
    pub fn violation_count(&self) -> u32 {
        self.violations
    }

    /// Current consecutive violation count (alias for [`violation_count`](Self::violation_count)).
    #[inline]
    pub fn consecutive_violations(&self) -> u32 {
        self.violations
    }

    /// Whether the detector is active (Adaptive mode only).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Close an epoch: evaluate the accumulated samples against the snapshotted
    /// baseline SRTT and RTTVAR (established at the beginning of the epoch).
    ///
    /// If a re-anchoring is triggered, the caller must apply it to the RTT
    /// estimator via `force_reanchor`.
    pub fn close_epoch(
        &mut self,
        acc: &EpochAccumulator,
        baseline_srtt: Duration,
        baseline_rttvar: Duration,
    ) -> EpochCloseResult {
        if !self.enabled {
            return EpochCloseResult::Normal;
        }

        // Epochs with fewer than n_min samples preserve the counter (§ 1.5 item 1).
        //
        // Formal Interpretation:
        // The detector enforces `m` consecutive *valid observation epochs* that violate the baseline.
        // Under § 1.5 item 1 of the frozen mathematical contract:
        //   "(Epochs with fewer than n_min eligible samples preserve the current violation counter
        //    without incrementing or resetting it.)"
        // Sparse/idle epochs containing insufficient samples act as transparent pauses, preserving
        // the counter rather than resetting it, to prevent low probe volume from masking a real
        // sustained shift. Any subsequent valid epoch that does not violate the baseline resets `v <- 0`.
        if acc.len() < self.min_samples {
            return EpochCloseResult::InsufficientSamples;
        }

        let median = match acc.median() {
            Some(m) => m,
            None => return EpochCloseResult::InsufficientSamples,
        };

        // Violation condition (§ 1.5 item 2): M_e > baseline_SRTT + c · baseline_RTTVAR
        let threshold_secs =
            baseline_srtt.as_secs_f64() + self.change_c * baseline_rttvar.as_secs_f64();
        let threshold = Duration::from_secs_f64(threshold_secs.max(0.0));

        if median > threshold {
            self.violations += 1;
            self.retained_medians.push(median);
            self.retained_samples.extend_from_slice(acc.samples());

            if self.violations >= self.change_m {
                // Forced unclamped re-anchoring (CONTRACT.md § 2):
                // SRTT_new = median(M_1, ..., M_m)
                let mut medians = self.retained_medians.clone();
                medians.sort_unstable();
                let n_m = medians.len();
                let new_srtt = if n_m % 2 == 1 {
                    medians[n_m / 2]
                } else {
                    let a = medians[n_m / 2 - 1];
                    let b = medians[n_m / 2];
                    (a + b) / 2
                };

                let candidate_rttvar = match self.reanchor_strategy {
                    ReanchorVariationStrategy::MedianStep => {
                        // Spec default: RTTVAR = max(RTTVAR_min, |M_e − baseline_SRTT| / 2)
                        let diff_secs = (new_srtt.as_secs_f64() - baseline_srtt.as_secs_f64()).abs();
                        Duration::from_secs_f64(diff_secs / 2.0)
                    }
                    ReanchorVariationStrategy::EpochSampleVariation => {
                        // CONTRACT.md § 2:
                        // RTTVAR_new = max(RTTVAR_min, 1/N * sum_{i=1}^N |R_i - SRTT_new|)
                        // across all retained samples R_1..R_N
                        compute_mad_nanos(&self.retained_samples, new_srtt)
                    }
                };

                let new_rttvar = if candidate_rttvar < self.rttvar_min {
                    self.rttvar_min
                } else {
                    candidate_rttvar
                };

                self.violations = 0;
                self.retained_medians.clear();
                self.retained_samples.clear();

                EpochCloseResult::ReAnchor {
                    new_srtt,
                    new_rttvar,
                }
            } else {
                EpochCloseResult::Violation {
                    consecutive: self.violations,
                }
            }
        } else {
            // Non-violating epoch: reset counter and retained history (CONTRACT.md § 2).
            self.violations = 0;
            self.retained_medians.clear();
            self.retained_samples.clear();
            EpochCloseResult::Normal
        }
    }

    /// Reset the violation counter and retained samples (e.g., after manual re-anchoring).
    pub fn reset(&mut self) {
        self.violations = 0;
        self.retained_medians.clear();
        self.retained_samples.clear();
    }
}

/// Compute Mean Absolute Deviation (MAD) of samples around the given center duration.
fn compute_mad_nanos(samples: &[Duration], center: Duration) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let center_nanos = center.as_nanos();
    let mut sum_diff_nanos: u128 = 0;
    for &s in samples {
        let diff = s.as_nanos().abs_diff(center_nanos);
        sum_diff_nanos = sum_diff_nanos.saturating_add(diff);
    }
    let avg_diff_nanos = sum_diff_nanos / (samples.len() as u128);
    let secs = (avg_diff_nanos / 1_000_000_000) as u64;
    let subsec_nanos = (avg_diff_nanos % 1_000_000_000) as u32;
    Duration::new(secs, subsec_nanos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    fn make_acc(samples: &[u64]) -> EpochAccumulator {
        let mut acc = EpochAccumulator::new();
        for &ms in samples {
            acc.push(Duration::from_millis(ms));
        }
        acc
    }

    #[test]
    fn insufficient_samples_preserves_counter() {
        let cfg = AdaptiveConfig::adaptive();
        let mut det = ChangeDetector::new(&cfg);

        // First, create a violation to set counter = 1
        let acc = make_acc(&[500, 500, 500]);
        let srtt = Duration::from_millis(100);
        let rttvar = Duration::from_millis(10);
        let result = det.close_epoch(&acc, srtt, rttvar);
        assert!(matches!(result, EpochCloseResult::Violation { consecutive: 1 }));

        // Now close with insufficient samples — should preserve counter at 1
        let acc_short = make_acc(&[500, 500]); // only 2, min is 3
        let result2 = det.close_epoch(&acc_short, srtt, rttvar);
        assert_eq!(result2, EpochCloseResult::InsufficientSamples);
        assert_eq!(det.violation_count(), 1); // preserved!
    }

    #[test]
    fn normal_resets_counter() {
        let cfg = AdaptiveConfig::adaptive();
        let mut det = ChangeDetector::new(&cfg);

        let srtt = Duration::from_millis(100);
        let rttvar = Duration::from_millis(50);

        // Violation
        let acc = make_acc(&[500, 500, 500]);
        det.close_epoch(&acc, srtt, rttvar);
        assert_eq!(det.violation_count(), 1);

        // Normal epoch (median 100, threshold = 100 + 3*50 = 250)
        let acc_normal = make_acc(&[100, 100, 100]);
        let result = det.close_epoch(&acc_normal, srtt, rttvar);
        assert_eq!(result, EpochCloseResult::Normal);
        assert_eq!(det.violation_count(), 0);
    }

    #[test]
    fn three_violations_trigger_reanchor() {
        let cfg = AdaptiveConfig::adaptive();
        let mut det = ChangeDetector::new(&cfg);

        let srtt = Duration::from_millis(100);
        let rttvar = Duration::from_millis(10);
        // threshold = 100 + 3*10 = 130 ms

        for i in 1..=3 {
            let acc = make_acc(&[200, 200, 200]); // median 200 > 130
            let result = det.close_epoch(&acc, srtt, rttvar);
            if i < 3 {
                assert!(matches!(result, EpochCloseResult::Violation { .. }));
            } else {
                match result {
                    EpochCloseResult::ReAnchor {
                        new_srtt,
                        new_rttvar,
                    } => {
                        assert_eq!(new_srtt, Duration::from_millis(200));
                        // |200 - 100| / 2 = 50 ms
                        assert_eq!(new_rttvar, Duration::from_millis(50));
                    }
                    other => panic!("Expected ReAnchor, got {:?}", other),
                }
                assert_eq!(det.violation_count(), 0);
            }
        }
    }

    #[test]
    fn reanchor_sample_variation_strategy() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.reanchor_strategy = ReanchorVariationStrategy::EpochSampleVariation;
        let mut det = ChangeDetector::new(&cfg);

        let srtt = Duration::from_millis(100);
        let rttvar = Duration::from_millis(10);

        // Epoch samples: [280, 300, 320] -> median 300, MAD = (20 + 0 + 20) / 3 = 13.33ms
        for i in 1..=3 {
            let acc = make_acc(&[280, 300, 320]);
            let result = det.close_epoch(&acc, srtt, rttvar);
            if i == 3 {
                match result {
                    EpochCloseResult::ReAnchor {
                        new_srtt,
                        new_rttvar,
                    } => {
                        assert_eq!(new_srtt, Duration::from_millis(300));
                        // MAD is 13.33ms (13333 micros)
                        assert_eq!(new_rttvar.as_millis(), 13);
                    }
                    other => panic!("Expected ReAnchor, got {:?}", other),
                }
            }
        }
    }

    #[test]
    fn reference_mode_always_normal() {
        let cfg = AdaptiveConfig::reference();
        let mut det = ChangeDetector::new(&cfg);
        assert!(!det.is_enabled());

        let acc = make_acc(&[1000, 1000, 1000]);
        let result = det.close_epoch(&acc, Duration::from_millis(100), Duration::from_millis(10));
        assert_eq!(result, EpochCloseResult::Normal);
    }

    #[test]
    fn mad_nanosecond_precision() {
        let mut acc = EpochAccumulator::new();
        // 100ns and 300ns around median 200ns -> MAD = (100ns + 100ns)/2 = 100ns
        acc.push(Duration::from_nanos(100));
        acc.push(Duration::from_nanos(300));
        let mad = acc.mean_absolute_deviation(Duration::from_nanos(200));
        assert_eq!(mad, Duration::from_nanos(100));
    }

    #[test]
    fn median_computation() {
        let acc = make_acc(&[10, 30, 20]); // sorted: [10, 20, 30] → median = 20
        assert_eq!(acc.median(), Some(Duration::from_millis(20)));

        let acc2 = make_acc(&[10, 30, 20, 40]); // sorted: [10, 20, 30, 40] → (20+30)/2 = 25
        assert_eq!(acc2.median(), Some(Duration::from_millis(25)));
    }

    #[test]
    fn multi_epoch_varying_retained_samples_contract() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.reanchor_strategy = ReanchorVariationStrategy::EpochSampleVariation;
        let mut det = ChangeDetector::new(&cfg);

        let baseline_srtt = Duration::from_millis(50);
        let baseline_rttvar = Duration::from_millis(10);
        // threshold = 50 + 3*10 = 80ms

        // Epoch 1: [190, 200, 210] -> median 200ms
        let acc1 = make_acc(&[190, 200, 210]);
        let res1 = det.close_epoch(&acc1, baseline_srtt, baseline_rttvar);
        assert_eq!(res1, EpochCloseResult::Violation { consecutive: 1 });

        // Epoch 2: [240, 250, 260] -> median 250ms
        let acc2 = make_acc(&[240, 250, 260]);
        let res2 = det.close_epoch(&acc2, baseline_srtt, baseline_rttvar);
        assert_eq!(res2, EpochCloseResult::Violation { consecutive: 2 });

        // Epoch 3: [290, 300, 310] -> median 300ms
        let acc3 = make_acc(&[290, 300, 310]);
        let res3 = det.close_epoch(&acc3, baseline_srtt, baseline_rttvar);

        // Medians: [200, 250, 300] -> median of medians = 250ms
        // Samples R_1..R_9: [190, 200, 210, 240, 250, 260, 290, 300, 310]
        // Diff from 250: |190-250|=60, |200-250|=50, |210-250|=40, |240-250|=10, |250-250|=0,
        //               |260-250|=10, |290-250|=40, |300-250|=50, |310-250|=60.
        // Sum = 60+50+40+10+0+10+40+50+60 = 320ms.
        // MAD = 320ms / 9 = 35.555... ms = 35_555_555 ns.
        match res3 {
            EpochCloseResult::ReAnchor {
                new_srtt,
                new_rttvar,
            } => {
                assert_eq!(new_srtt, Duration::from_millis(250));
                assert_eq!(new_rttvar.as_millis(), 35);
            }
            other => panic!("Expected ReAnchor, got {:?}", other),
        }
    }
}
