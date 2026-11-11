//! Integrated adaptive engine — orchestrates all subsystems.
//!
//! `AdaptiveEngine` ties together the RTT estimator, P95 tracker, change
//! detector, RTO schedule, concurrency controller, and loss monitor into a
//! single coherent state machine driven by explicit timestamps.

use core::time::Duration;

use crate::change_detector::{ChangeDetector, EpochAccumulator, EpochCloseResult};
use crate::concurrency::ConcurrencyController;
use crate::config::{AdaptiveConfig, ConfigError, ProbeOutcome};
use crate::loss::{EpochEvents, LossMonitor};
use crate::p95::RollingP95;
use crate::rto::ProbeRetrySchedule;
use crate::rtt::RttEstimator;

/// Default destination ID used when callers don't provide a destination key.
pub const DEFAULT_DESTINATION_ID: u64 = 0;

/// Instant representation for trace-driven replay.
///
/// We use a simple monotonic counter instead of `std::time::Instant` to allow
/// deterministic, allocation-free benchmarking without a real clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(Duration);

impl Timestamp {
    /// Create a timestamp from a `Duration` since an arbitrary epoch.
    pub fn from_duration(d: Duration) -> Self {
        Self(d)
    }

    /// Duration since another (earlier) timestamp.
    pub fn duration_since(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    /// The underlying duration value.
    pub fn as_duration(self) -> Duration {
        self.0
    }

    /// A zero timestamp.
    pub const ZERO: Self = Self(Duration::ZERO);
}

impl core::ops::Add<Duration> for Timestamp {
    type Output = Timestamp;
    fn add(self, rhs: Duration) -> Self::Output {
        Timestamp(self.0 + rhs)
    }
}

/// Full telemetry snapshot exported after each event.
#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    pub srtt: Duration,
    pub rttvar: Duration,
    pub rtt_initialized: bool,
    pub rto: Duration,
    pub t95: Option<Duration>,
    pub d_loss: f64,
    pub d_rate_limit: f64,
    pub global_window: usize,
    pub violation_count: u32,
    pub in_flight: usize,
    pub epoch_duration: Duration,
}

/// The integrated adaptive engine.
///
/// Call [`record_outcome`] or [`record_dest_outcome`] for each probe completion
/// and [`tick`] to advance the epoch clock. The engine handles all internal state
/// transitions.
#[derive(Clone, Debug)]
pub struct AdaptiveEngine {
    cfg: AdaptiveConfig,
    rtt: RttEstimator,
    p95: RollingP95,
    change_detector: ChangeDetector,
    concurrency: ConcurrencyController,
    loss: LossMonitor,

    // Epoch tracking
    epoch_start: Timestamp,
    /// Observation epoch duration latched when the epoch opens (§ 3.1).
    /// Retained until the epoch closes for reproducible observation windows.
    current_epoch_duration: Duration,
    epoch_acc: EpochAccumulator,
    epoch_events: EpochEvents,
    /// Snapshotted baseline SRTT at epoch start for change detection (§ 1.5).
    baseline_srtt: Duration,
    /// Snapshotted baseline RTTVAR at epoch start for change detection (§ 1.5).
    baseline_rttvar: Duration,
}

impl AdaptiveEngine {
    /// Create a new engine from the given configuration, starting at `now`.
    ///
    /// Validates the configuration before initializing engine components.
    ///
    /// # Errors
    /// Returns [`ConfigError`] if configuration validation fails.
    pub fn try_new(cfg: AdaptiveConfig, now: Timestamp) -> Result<Self, ConfigError> {
        cfg.validate()?;
        let rtt = RttEstimator::new(&cfg);
        let p95 = RollingP95::new(cfg.p95_window_size);
        let change_detector = ChangeDetector::new(&cfg);
        let concurrency = ConcurrencyController::new(&cfg);
        let loss = LossMonitor::new(&cfg);
        let current_epoch_duration = cfg.epoch_duration(None);

        Ok(Self {
            cfg,
            rtt,
            p95,
            change_detector,
            concurrency,
            loss,
            epoch_start: now,
            current_epoch_duration,
            epoch_acc: EpochAccumulator::new(),
            epoch_events: EpochEvents::default(),
            baseline_srtt: Duration::ZERO,
            baseline_rttvar: Duration::ZERO,
        })
    }

    /// Create a new engine with the given configuration, starting at `now`.
    ///
    /// # Panics
    /// Panics if configuration validation fails. Use [`try_new`](Self::try_new) for safe error handling.
    pub fn new(cfg: AdaptiveConfig, now: Timestamp) -> Self {
        Self::try_new(cfg, now).expect("invalid AdaptiveConfig")
    }

    /// Access the configuration.
    pub fn config(&self) -> &AdaptiveConfig {
        &self.cfg
    }

    /// Access the RTT estimator.
    pub fn rtt(&self) -> &RttEstimator {
        &self.rtt
    }

    /// Access the P95 tracker.
    pub fn p95_tracker(&self) -> &RollingP95 {
        &self.p95
    }

    /// Access the concurrency controller.
    pub fn concurrency(&self) -> &ConcurrencyController {
        &self.concurrency
    }

    /// Access the concurrency controller mutably.
    pub fn concurrency_mut(&mut self) -> &mut ConcurrencyController {
        &mut self.concurrency
    }

    /// Access the change detector.
    pub fn change_detector(&self) -> &ChangeDetector {
        &self.change_detector
    }

    /// Access the loss monitor.
    pub fn loss_monitor(&self) -> &LossMonitor {
        &self.loss
    }

    /// The snapshotted baseline SRTT used by the change detector for the active epoch.
    #[inline]
    pub fn baseline_srtt(&self) -> Duration {
        self.baseline_srtt
    }

    /// The snapshotted baseline RTTVAR used by the change detector for the active epoch.
    #[inline]
    pub fn baseline_rttvar(&self) -> Duration {
        self.baseline_rttvar
    }

    /// The duration of the currently active observation epoch (§ 3.1).
    ///
    /// Fixed when the epoch opens and retained until it closes.
    #[inline]
    pub fn current_epoch_duration(&self) -> Duration {
        self.current_epoch_duration
    }

    /// Check if a probe can be dispatched to a destination under dual ceilings (§ 3.3).
    #[inline]
    pub fn can_dispatch(&self, dest_id: u64) -> bool {
        self.concurrency.can_dispatch(dest_id)
    }

    /// Record that a probe has been dispatched to a destination.
    #[inline]
    pub fn on_dispatch(&mut self, dest_id: u64) {
        self.concurrency.on_dispatch(dest_id);
    }

    /// Attempt to complete a probe for destination `dest_id`.
    ///
    /// Returns `true` if an active in-flight probe was decremented on `dest_id` and globally.
    #[inline]
    pub fn try_complete(&mut self, dest_id: u64) -> bool {
        self.concurrency.try_complete(dest_id)
    }

    /// Get the current RTO (or `rto_init` if uninitialized).
    pub fn current_rto(&self) -> Duration {
        if self.rtt.is_initialized() {
            self.cfg.compute_rto(self.rtt.srtt(), self.rtt.rttvar())
        } else {
            self.cfg.rto_init
        }
    }

    /// Create a latched per-probe retry schedule at dispatch time (n=0).
    pub fn latch_retry_schedule(&self) -> ProbeRetrySchedule {
        if self.rtt.is_initialized() {
            ProbeRetrySchedule::new(
                &self.cfg,
                Some(self.rtt.srtt()),
                Some(self.rtt.rttvar()),
            )
        } else {
            ProbeRetrySchedule::new(&self.cfg, None, None)
        }
    }

    /// Create a latched per-probe retry schedule for a specific destination.
    ///
    /// Uses the destination's isolated RTT estimate if available (preventing
    /// high-latency targets from skewing timeouts of unrelated targets), falling back
    /// to the global baseline, and incorporates destination-level backoff state (§ 2.3).
    pub fn latch_dest_retry_schedule(&self, dest_id: u64) -> ProbeRetrySchedule {
        let (srtt, rttvar, backoffs) = if let Some(dest) = self.concurrency.get_dest(dest_id) {
            let (s, r) = if let Some(dest_rtt) = &dest.rtt
                && dest_rtt.is_initialized()
            {
                (Some(dest_rtt.srtt()), Some(dest_rtt.rttvar()))
            } else if self.rtt.is_initialized() {
                (Some(self.rtt.srtt()), Some(self.rtt.rttvar()))
            } else {
                (None, None)
            };
            (s, r, dest.consecutive_backoffs)
        } else if self.rtt.is_initialized() {
            (Some(self.rtt.srtt()), Some(self.rtt.rttvar()), 0)
        } else {
            (None, None, 0)
        };

        ProbeRetrySchedule::for_destination(&self.cfg, srtt, rttvar, backoffs)
    }

    /// Access the destination-specific RTT estimator for `dest_id`, if initialized.
    #[inline]
    pub fn dest_rtt(&self, dest_id: u64) -> Option<&RttEstimator> {
        self.concurrency.get_dest(dest_id).and_then(|d| d.rtt.as_ref())
    }

    /// Record a probe outcome for the default global destination.
    #[inline]
    pub fn record_outcome(&mut self, outcome: ProbeOutcome) {
        self.record_dest_outcome(DEFAULT_DESTINATION_ID, outcome);
    }

    /// Record a probe outcome for a specific destination (§ 2.3 & § 3.3).
    ///
    /// Routes the outcome to the RTT estimator, tail tracker, change detector,
    /// destination-level concurrency tracker, and loss monitor.
    pub fn record_dest_outcome(&mut self, dest_id: u64, outcome: ProbeOutcome) {
        // Complete the in-flight probe on concurrency tracker
        self.concurrency.on_completion(dest_id);
        self.concurrency.record_dest_outcome(dest_id, outcome);

        match outcome {
            ProbeOutcome::SuccessInitial { rtt } => {
                // § 1.1 — Eligible sample (n=0), Karn's rule satisfied.
                // 1. Destination-isolated RTT estimator tracking:
                self.concurrency
                    .get_or_create_dest(dest_id)
                    .record_rtt_sample(rtt, &self.cfg);

                // 2. Global aggregate RTT estimator, P95, and change detector:
                self.rtt.record_sample(rtt);
                self.p95.push(rtt);
                self.epoch_acc.push(rtt);
                self.epoch_events.successes += 1;
            }
            ProbeOutcome::SuccessRetransmitted { .. } => {
                // § 1.1 — Karn's rule: reject RTT sample from retransmitted probe.
                self.epoch_events.successes += 1;
            }
            ProbeOutcome::ExpectedSilence => {
                // § 4.1 — Excluded from loss indicators and AIMD.
                self.epoch_events.expected_silence += 1;
            }
            ProbeOutcome::InferredLoss => {
                self.epoch_events.inferred_loss += 1;
            }
            ProbeOutcome::InferredRateLimit => {
                self.epoch_events.inferred_rate_limit += 1;
            }
            ProbeOutcome::Unclassified => {
                // Excluded from all loss metrics and AIMD calculations
                self.epoch_events.unclassified += 1;
            }
        }
    }

    /// Advance the epoch clock. Call this periodically.
    ///
    /// Uses the reproducible epoch duration latched when the epoch opened.
    /// If sufficient time has elapsed (≥ epoch duration), closes the current
    /// epoch, opens a new one, and latches the next epoch duration.
    ///
    /// Returns `true` if an epoch was closed.
    pub fn tick(&mut self, now: Timestamp) -> bool {
        let elapsed = now.duration_since(self.epoch_start);

        if elapsed < self.current_epoch_duration {
            return false;
        }

        self.close_epoch(elapsed);
        self.epoch_start = now;
        self.current_epoch_duration = self.next_epoch_duration();
        true
    }

    /// Force-close the current epoch (for shutdown, flush, or manual stepping).
    pub fn force_close_epoch(&mut self, now: Timestamp) {
        let elapsed = now.duration_since(self.epoch_start);
        self.close_epoch(elapsed);
        self.epoch_start = now;
        self.current_epoch_duration = self.next_epoch_duration();
    }

    /// Get a complete telemetry snapshot.
    pub fn telemetry(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            srtt: self.rtt.srtt(),
            rttvar: self.rtt.rttvar(),
            rtt_initialized: self.rtt.is_initialized(),
            rto: self.current_rto(),
            t95: self.p95.p95(),
            d_loss: self.loss.d_loss(),
            d_rate_limit: self.loss.d_rate_limit(),
            global_window: self.concurrency.global.window(),
            violation_count: self.change_detector.violation_count(),
            in_flight: self.concurrency.global_in_flight,
            epoch_duration: self.current_epoch_duration,
        }
    }

    // ── Private ──────────────────────────────────────────────────────────

    /// Calculate the epoch duration for an opening epoch (§ 3.1).
    #[inline]
    fn next_epoch_duration(&self) -> Duration {
        let srtt_opt = if self.rtt.is_initialized() {
            Some(self.rtt.srtt())
        } else {
            None
        };
        self.cfg.epoch_duration(srtt_opt)
    }

    fn close_epoch(&mut self, elapsed: Duration) {
        // If this is the first epoch with samples, seed the baseline
        if self.baseline_srtt == Duration::ZERO && self.rtt.is_initialized() {
            self.baseline_srtt = self.rtt.srtt();
            self.baseline_rttvar = self.rtt.rttvar();
        }

        // § 1.5 — Change detection against snapshotted baseline (Adaptive mode only)
        let change_result = if self.change_detector.is_enabled() && self.baseline_srtt > Duration::ZERO {
            self.change_detector.close_epoch(
                &self.epoch_acc,
                self.baseline_srtt,
                self.baseline_rttvar,
            )
        } else {
            EpochCloseResult::Normal
        };

        // Baseline action policy strictly enforced per contract:
        // - Normal: Refresh baseline from current RTT estimator state
        // - Violation: Preserve existing baseline (do NOT contaminate reference point)
        // - InsufficientSamples: Preserve existing baseline
        // - ReAnchor: Replace baseline with re-anchored values
        match change_result {
            EpochCloseResult::ReAnchor {
                new_srtt,
                new_rttvar,
            } => {
                self.rtt.force_reanchor(new_srtt, new_rttvar);
                self.baseline_srtt = new_srtt;
                self.baseline_rttvar = new_rttvar;
            }
            EpochCloseResult::Normal => {
                if self.rtt.is_initialized() {
                    self.baseline_srtt = self.rtt.srtt();
                    self.baseline_rttvar = self.rtt.rttvar();
                }
            }
            EpochCloseResult::Violation { .. } | EpochCloseResult::InsufficientSamples => {
                // Strictly preserve existing baseline without updating.
            }
        }

        // § 3 — AIMD concurrency updates across all active destinations & global
        // In Reference profile (§ 5), concurrency is fixed at caller ceiling
        self.concurrency.apply_epoch();

        // § 4.2 — Time-normalized loss EWMA update
        // In Reference profile, acts as passive telemetry
        self.loss.update(&self.epoch_events, elapsed);

        // Reset epoch accumulators
        self.epoch_acc.reset();
        self.epoch_events = EpochEvents::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    fn ts(ms: u64) -> Timestamp {
        Timestamp::from_duration(Duration::from_millis(ms))
    }

    #[test]
    fn basic_lifecycle() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        assert!(!engine.rtt().is_initialized());
        assert_eq!(engine.current_rto(), Duration::from_millis(500));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });
        assert!(engine.rtt().is_initialized());
        assert_eq!(engine.current_rto(), Duration::from_millis(300));
    }

    #[test]
    fn reference_profile_isolation() {
        let cfg = AdaptiveConfig::reference();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        assert_eq!(engine.concurrency().global.window(), cfg.c_max_global);

        // Record drops and close epoch
        engine.record_outcome(ProbeOutcome::InferredLoss);
        engine.tick(ts(1000));

        // Window must remain completely fixed in Reference mode
        assert_eq!(engine.concurrency().global.window(), cfg.c_max_global);
    }

    #[test]
    fn unclassified_outcome_ignored_by_loss_and_aimd() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        let initial_win = engine.concurrency().global.window();
        engine.record_outcome(ProbeOutcome::Unclassified);
        engine.tick(ts(100));

        // Unclassified does not trigger loss reduction or additive increase
        assert_eq!(engine.loss_monitor().d_loss(), 0.0);
        assert_eq!(engine.loss_monitor().d_rate_limit(), 0.0);
        assert_eq!(engine.concurrency().global.window(), initial_win);
    }

    #[test]
    fn min_completed_probes_enforcement() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.min_completed_probes = 5;
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Only 2 completions (below threshold of 5)
        for _ in 0..2 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            });
        }
        let win_before = engine.concurrency().global.window();
        engine.tick(ts(100));

        // Should NOT increment window because min_completed_probes not met
        assert_eq!(engine.concurrency().global.window(), win_before);

        // Now provide 5 completions
        for _ in 0..5 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            });
        }
        engine.tick(ts(200));
        assert_eq!(engine.concurrency().global.window(), win_before + 1);
    }

    #[test]
    fn per_destination_aimd_tracking() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        let d1 = 101;
        let d2 = 102;

        engine.on_dispatch(d1);
        engine.record_dest_outcome(d1, ProbeOutcome::InferredLoss);

        engine.on_dispatch(d2);
        for _ in 0..3 {
            engine.record_dest_outcome(d2, ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            });
        }

        engine.tick(ts(100));

        let dest1 = engine.concurrency().get_dest(d1).unwrap();
        let dest2 = engine.concurrency().get_dest(d2).unwrap();

        // Dest 1 experienced loss -> floor/reduction
        assert_eq!(dest1.window.window(), cfg.c_min_dest);
        // Dest 2 was lossless -> window increased
        assert!(dest2.window.window() > cfg.c_min_dest);
    }

    #[test]
    fn change_detection_with_snapshotted_baseline() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Seed with 50ms baseline
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(50),
        });
        engine.tick(ts(100)); // Establishes baseline 50ms

        // 3 consecutive epochs with samples at 300ms
        let mut t = 100;
        for _ in 0..3 {
            for _ in 0..5 {
                engine.record_outcome(ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(300),
                });
            }
            t += 100;
            engine.tick(ts(t));
        }

        // Re-anchor should have kicked in
        assert!(engine.rtt().srtt() >= Duration::from_millis(250));
    }

    #[test]
    fn baseline_preservation_during_violations() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Initialize at 50ms
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(50),
        });
        engine.tick(ts(100));

        let baseline_at_start = engine.baseline_srtt();
        assert_eq!(baseline_at_start, Duration::from_millis(50));

        // Epoch with a violation (samples at 400ms > 50 + 3*25 = 125ms)
        for _ in 0..5 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(400),
            });
        }
        engine.tick(ts(200));

        // RTT estimator has updated its internal SRTT (clamped)
        assert!(engine.rtt().srtt() > Duration::from_millis(50));
        // BUT the baseline must strictly be preserved!
        assert_eq!(
            engine.baseline_srtt(),
            baseline_at_start,
            "Baseline must NOT be contaminated during a violation epoch"
        );
    }

    #[test]
    fn reproducible_epoch_duration() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Uninitialized: latches epoch_e_init (100ms)
        assert_eq!(engine.current_epoch_duration(), Duration::from_millis(100));

        // In the middle of epoch, record a sample at 500ms
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(500),
        });

        // Even though SRTT changed to 500ms, current_epoch_duration remains 100ms
        assert_eq!(engine.current_epoch_duration(), Duration::from_millis(100));

        // Advance by 99ms: tick should NOT close epoch yet
        assert!(!engine.tick(ts(99)));

        // Advance to 100ms: closes epoch
        assert!(engine.tick(ts(100)));

        // NEW epoch now opens, latching the new duration based on SRTT (500ms)
        assert_eq!(engine.current_epoch_duration(), Duration::from_millis(500));
    }

    #[test]
    fn destination_retry_schedule_backoff() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        let d1 = 42;
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });

        let sched_normal = engine.latch_dest_retry_schedule(d1);

        // Record drops on d1
        engine.record_dest_outcome(d1, ProbeOutcome::InferredLoss);
        let sched_backed_off = engine.latch_dest_retry_schedule(d1);

        // Backed off base RTO should be higher than normal base RTO
        assert!(
            sched_backed_off.base_rto > sched_normal.base_rto,
            "Destination retry schedule should reflect destination backoff state"
        );
    }

    #[test]
    fn try_new_enforces_validation() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.alpha = -0.5; // Invalid alpha

        let res = AdaptiveEngine::try_new(cfg, ts(0));
        assert!(res.is_err(), "try_new should reject invalid alpha");

        let mut cfg_ref = AdaptiveConfig::reference();
        cfg_ref.rto_min = Duration::from_millis(100); // Invalid for Reference (< 1s)
        let res_ref = AdaptiveEngine::try_new(cfg_ref, ts(0));
        assert!(res_ref.is_err(), "try_new should reject invalid Reference rto_min");
    }

    #[test]
    fn destination_rtt_isolation() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        let fast_dest = 1;
        let slow_dest = 2;

        // Fast destination receives 20ms samples
        for _ in 0..5 {
            engine.record_dest_outcome(fast_dest, ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(20),
            });
        }

        // Slow destination receives 300ms samples
        for _ in 0..5 {
            engine.record_dest_outcome(slow_dest, ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(300),
            });
        }

        // Fast dest schedule should compute tight timeouts around 20ms
        let sched_fast = engine.latch_dest_retry_schedule(fast_dest);
        let sched_slow = engine.latch_dest_retry_schedule(slow_dest);

        assert!(
            sched_fast.base_rto < sched_slow.base_rto,
            "Fast destination timeout ({:?}) should be lower than slow destination timeout ({:?})",
            sched_fast.base_rto,
            sched_slow.base_rto
        );

        let fast_srtt = engine.dest_rtt(fast_dest).unwrap().srtt();
        let slow_srtt = engine.dest_rtt(slow_dest).unwrap().srtt();

        assert!(fast_srtt < Duration::from_millis(50));
        assert!(slow_srtt > Duration::from_millis(200));
    }

    #[test]
    #[should_panic(expected = "invalid AdaptiveConfig")]
    fn new_panics_on_invalid_config() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.alpha = 2.0; // Invalid alpha > 1.0
        let _ = AdaptiveEngine::new(cfg, ts(0));
    }
}
