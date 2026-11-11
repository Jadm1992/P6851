//! § 1.5 — Reproducible regime-change detector.
//!
//! Detects sustained step-increases in baseline path latency that the clamped
//! EWMA cannot track, and forces an unclamped re-anchoring.

use core::time::Duration;

use crate::config::AdaptiveConfig;

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

    /// Drain and reset for the next epoch.
    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

/// Regime-change detector state.
///
/// Tracks consecutive violation epochs and triggers forced re-anchoring when
/// `m` consecutive epochs have median RTT significantly above the current baseline.
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
    /// Whether the detector is enabled (disabled for Reference profile).
    enabled: bool,
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
            enabled: cfg.mode == crate::config::ProfileMode::Adaptive,
        }
    }

    /// Current consecutive violation count.
    #[inline]
    pub fn violation_count(&self) -> u32 {
        self.violations
    }

    /// Whether the detector is active (Adaptive mode only).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Close an epoch: evaluate the accumulated samples against the current
    /// SRTT and RTTVAR baseline.
    ///
    /// If a re-anchoring is triggered, the caller must apply it to the RTT
    /// estimator via `force_reanchor`.
    pub fn close_epoch(
        &mut self,
        acc: &EpochAccumulator,
        current_srtt: Duration,
        current_rttvar: Duration,
    ) -> EpochCloseResult {
        if !self.enabled {
            return EpochCloseResult::Normal;
        }

        // Epochs with fewer than n_min samples preserve the counter.
        if acc.len() < self.min_samples {
            return EpochCloseResult::InsufficientSamples;
        }

        let median = match acc.median() {
            Some(m) => m,
            None => return EpochCloseResult::InsufficientSamples,
        };

        // Violation condition: M_e > SRTT + c·RTTVAR
        let threshold_secs =
            current_srtt.as_secs_f64() + self.change_c * current_rttvar.as_secs_f64();
        let threshold = Duration::from_secs_f64(threshold_secs.max(0.0));

        if median > threshold {
            self.violations += 1;

            if self.violations >= self.change_m {
                // Forced unclamped re-anchoring:
                // SRTT ← M_e
                // RTTVAR ← max(RTTVAR_min, |M_e − SRTT_old| / 2)
                let diff_secs = (median.as_secs_f64() - current_srtt.as_secs_f64()).abs();
                let new_rttvar_secs = diff_secs / 2.0;
                let new_rttvar = Duration::from_secs_f64(new_rttvar_secs);
                let clamped_rttvar = if new_rttvar < self.rttvar_min {
                    self.rttvar_min
                } else {
                    new_rttvar
                };

                self.violations = 0;

                EpochCloseResult::ReAnchor {
                    new_srtt: median,
                    new_rttvar: clamped_rttvar,
                }
            } else {
                EpochCloseResult::Violation {
                    consecutive: self.violations,
                }
            }
        } else {
            // Non-violating epoch: reset counter.
            self.violations = 0;
            EpochCloseResult::Normal
        }
    }

    /// Reset the violation counter (e.g., after manual re-anchoring).
    pub fn reset(&mut self) {
        self.violations = 0;
    }
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
    fn reference_mode_always_normal() {
        let cfg = AdaptiveConfig::reference();
        let mut det = ChangeDetector::new(&cfg);
        assert!(!det.is_enabled());

        let acc = make_acc(&[1000, 1000, 1000]);
        let result = det.close_epoch(&acc, Duration::from_millis(100), Duration::from_millis(10));
        assert_eq!(result, EpochCloseResult::Normal);
    }

    #[test]
    fn median_computation() {
        let acc = make_acc(&[10, 30, 20]); // sorted: [10, 20, 30] → median = 20
        assert_eq!(acc.median(), Some(Duration::from_millis(20)));

        let acc2 = make_acc(&[10, 30, 20, 40]); // sorted: [10, 20, 30, 40] → (20+30)/2 = 25
        assert_eq!(acc2.median(), Some(Duration::from_millis(25)));
    }
}
