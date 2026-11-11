//! Integrated adaptive engine — orchestrates all subsystems.
//!
//! `AdaptiveEngine` ties together the RTT estimator, P95 tracker, change
//! detector, RTO schedule, concurrency controller, and loss monitor into a
//! single coherent state machine driven by explicit timestamps.

use core::time::Duration;

use crate::change_detector::{ChangeDetector, EpochAccumulator, EpochCloseResult};
use crate::concurrency::{ConcurrencyController, EpochOutcome};
use crate::config::{AdaptiveConfig, ProbeOutcome};
use crate::loss::{EpochEvents, LossMonitor};
use crate::p95::RollingP95;
use crate::rto::ProbeRetrySchedule;
use crate::rtt::RttEstimator;

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
}

/// The integrated adaptive engine.
///
/// Call [`record_outcome`] for each probe completion and [`tick`] to advance
/// the epoch clock.  The engine handles all internal state transitions.
#[derive(Clone, Debug)]
pub struct AdaptiveEngine {
    cfg: AdaptiveConfig,
    rtt: RttEstimator,
    p95: RollingP95,
    change_detector: ChangeDetector,
    concurrency: ConcurrencyController,
    loss: LossMonitor,

    // Epoch state
    epoch_start: Timestamp,
    epoch_acc: EpochAccumulator,
    epoch_events: EpochEvents,
    epoch_has_loss: bool,
}

impl AdaptiveEngine {
    /// Create a new engine with the given configuration, starting at `now`.
    pub fn new(cfg: AdaptiveConfig, now: Timestamp) -> Self {
        let rtt = RttEstimator::new(&cfg);
        let p95 = RollingP95::new(cfg.p95_window_size);
        let change_detector = ChangeDetector::new(&cfg);
        let concurrency = ConcurrencyController::new(&cfg);
        let loss = LossMonitor::new(&cfg);

        Self {
            cfg,
            rtt,
            p95,
            change_detector,
            concurrency,
            loss,
            epoch_start: now,
            epoch_acc: EpochAccumulator::new(),
            epoch_events: EpochEvents::default(),
            epoch_has_loss: false,
        }
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

    /// Access the loss monitor.
    pub fn loss_monitor(&self) -> &LossMonitor {
        &self.loss
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

    /// Record a probe outcome event.
    ///
    /// This is the primary entry point.  Call this for every completed or
    /// expired probe.  The engine routes the event to the appropriate
    /// subsystems based on the outcome classification.
    pub fn record_outcome(&mut self, outcome: ProbeOutcome) {
        match outcome {
            ProbeOutcome::SuccessInitial { rtt } => {
                // § 1.1 — Eligible sample (n=0), Karn's rule satisfied.
                self.rtt.record_sample(rtt);
                self.p95.push(rtt);
                self.epoch_acc.push(rtt);
                self.epoch_events.successes += 1;
            }
            ProbeOutcome::SuccessRetransmitted { .. } => {
                // § 1.1 — Karn's rule: reject RTT sample from retransmitted probe.
                // Still counts as a successful completion for concurrency/loss.
                self.epoch_events.successes += 1;
            }
            ProbeOutcome::ExpectedSilence => {
                // § 4.1 — Excluded from loss indicators and AIMD.
                self.epoch_events.expected_silence += 1;
            }
            ProbeOutcome::InferredLoss => {
                self.epoch_events.inferred_loss += 1;
                self.epoch_has_loss = true;
            }
            ProbeOutcome::InferredRateLimit => {
                self.epoch_events.inferred_rate_limit += 1;
                self.epoch_has_loss = true;
            }
        }
    }

    /// Advance the epoch clock.  Call this periodically (e.g., once per tick).
    ///
    /// If sufficient time has elapsed (≥ epoch duration), closes the current
    /// epoch and opens a new one, applying all state transitions.
    ///
    /// Returns `true` if an epoch was closed.
    pub fn tick(&mut self, now: Timestamp) -> bool {
        let srtt_opt = if self.rtt.is_initialized() {
            Some(self.rtt.srtt())
        } else {
            None
        };
        let epoch_dur = self.cfg.epoch_duration(srtt_opt);
        let elapsed = now.duration_since(self.epoch_start);

        if elapsed < epoch_dur {
            return false;
        }

        self.close_epoch(elapsed);
        self.epoch_start = now;
        true
    }

    /// Force-close the current epoch (for shutdown or flush).
    pub fn force_close_epoch(&mut self, now: Timestamp) {
        let elapsed = now.duration_since(self.epoch_start);
        self.close_epoch(elapsed);
        self.epoch_start = now;
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
        }
    }

    // ── Private ──────────────────────────────────────────────────────────

    fn close_epoch(&mut self, elapsed: Duration) {
        // § 1.5 — Change detection (Adaptive mode only).
        if self.change_detector.is_enabled() {
            let result = self.change_detector.close_epoch(
                &self.epoch_acc,
                self.rtt.srtt(),
                self.rtt.rttvar(),
            );
            if let EpochCloseResult::ReAnchor {
                new_srtt,
                new_rttvar,
            } = result
            {
                self.rtt.force_reanchor(new_srtt, new_rttvar);
            }
        }

        // § 3.2 — AIMD concurrency update.
        let epoch_outcome = if self.epoch_has_loss {
            EpochOutcome::Loss
        } else if self.epoch_events.has_completions() {
            EpochOutcome::Lossless
        } else {
            EpochOutcome::Idle
        };
        self.concurrency.global.apply_epoch(epoch_outcome);

        // § 4.2 — Time-normalized loss EWMA update.
        self.loss.update(&self.epoch_events, elapsed);

        // Reset epoch state.
        self.epoch_acc.reset();
        self.epoch_events = EpochEvents::default();
        self.epoch_has_loss = false;
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

        // Before any samples
        assert!(!engine.rtt().is_initialized());
        assert_eq!(engine.current_rto(), Duration::from_millis(500));

        // First eligible sample
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });
        assert!(engine.rtt().is_initialized());

        // RTO = clamp(100 + max(1, 200), 100, 3000) = 300ms
        // Wait, RTTVAR = max(1, 50) = 50, 4*50 = 200
        // max(G=1, 200) = 200
        // 100 + 200 = 300
        assert_eq!(engine.current_rto(), Duration::from_millis(300));
    }

    #[test]
    fn karn_rejection() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });
        let srtt_after_init = engine.rtt().srtt();

        // Retransmitted response should NOT update SRTT
        engine.record_outcome(ProbeOutcome::SuccessRetransmitted { attempt: 1 });
        assert_eq!(engine.rtt().srtt(), srtt_after_init);
    }

    #[test]
    fn epoch_tick_increments_window_on_lossless() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });

        let win_before = engine.concurrency().global.window();

        // Tick to close epoch (epoch_e_init = 100ms since SRTT was just set)
        // Actually, SRTT = 100ms, so epoch_dur = clamp(100, 10, 1000) = 100ms
        let closed = engine.tick(ts(100));
        assert!(closed);

        assert_eq!(
            engine.concurrency().global.window(),
            win_before + 1,
            "Lossless epoch should increment window"
        );
    }

    #[test]
    fn epoch_tick_decreases_window_on_loss() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Build up the window first
        for i in 0..10 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(100),
            });
            engine.tick(ts((i + 1) * 100));
        }

        let win_before = engine.concurrency().global.window();
        assert!(win_before > 1);

        // Record a loss
        engine.record_outcome(ProbeOutcome::InferredLoss);
        engine.tick(ts(1100));

        let win_after = engine.concurrency().global.window();
        assert!(
            win_after < win_before,
            "Loss epoch should decrease window: {} -> {}",
            win_before,
            win_after
        );
    }

    #[test]
    fn retry_schedule_latching() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });

        let sched = engine.latch_retry_schedule();
        let rto0 = sched.rto_for_attempt(0);

        // Now update SRTT with a very different sample
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(500),
        });

        // The latched schedule should still use the old RTO
        assert_eq!(sched.rto_for_attempt(0), rto0);

        // But a new schedule should reflect the updated SRTT
        let sched2 = engine.latch_retry_schedule();
        assert_ne!(sched2.rto_for_attempt(0), rto0);
    }

    #[test]
    fn change_detection_integration() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        // Initialize at 50ms RTT
        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(50),
        });
        let srtt_init = engine.rtt().srtt();
        // SRTT=50, RTTVAR=max(1, 25)=25

        // Three consecutive epochs with median >> SRTT + c·RTTVAR
        // threshold = 50 + 3*25 = 125 ms
        // We need median > 125 ms consistently for 3 epochs
        let mut t = 100;
        for epoch in 0..3 {
            // Push 5 samples at 300ms (well above 125ms threshold)
            for _ in 0..5 {
                engine.record_outcome(ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(300),
                });
            }
            t += 100;
            engine.tick(ts(t));

            if epoch < 2 {
                // SRTT should not have jumped to 300 yet (clamped updates)
                // In Adaptive mode, it moves slowly due to clamping
                assert!(engine.rtt().srtt() < Duration::from_millis(250));
            }
        }

        // After 3 violating epochs, change detector should have re-anchored
        // SRTT to ~300 ms
        let srtt_final = engine.rtt().srtt();
        assert!(
            srtt_final > srtt_init,
            "Re-anchor should have moved SRTT up: {srtt_init:?} -> {srtt_final:?}"
        );
    }

    #[test]
    fn telemetry_snapshot() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });

        let tel = engine.telemetry();
        assert!(tel.rtt_initialized);
        assert_eq!(tel.srtt, Duration::from_millis(100));
        assert!(tel.t95.is_some());
        assert_eq!(tel.d_loss, 0.0);
        assert_eq!(tel.d_rate_limit, 0.0);
    }

    #[test]
    fn p95_tracks_through_engine() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        for i in 1..=20 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(i * 10),
            });
        }

        let t95 = engine.p95_tracker().p95().unwrap();
        // 20 samples, ⌈0.95·20⌉ = 19, sorted samples: 10,20,...,200
        // 19th = 190 ms
        assert_eq!(t95, Duration::from_millis(190));
    }

    #[test]
    fn expected_silence_excluded() {
        let cfg = AdaptiveConfig::adaptive();
        let mut engine = AdaptiveEngine::new(cfg, ts(0));

        engine.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(100),
        });

        // Many expected-silence outcomes should not trigger loss
        for _ in 0..50 {
            engine.record_outcome(ProbeOutcome::ExpectedSilence);
        }

        let win_before = engine.concurrency().global.window();
        engine.tick(ts(100));
        // Should be lossless (expected silence is excluded)
        assert!(engine.concurrency().global.window() >= win_before);
    }
}
