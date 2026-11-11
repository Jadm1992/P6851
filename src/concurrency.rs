//! § 3 — Epoch-based AIMD concurrency control.
//!
//! Implements §§ 3.1–3.3: observation epochs, additive-increase /
//! multiplicative-decrease state transitions, dual-ceiling enforcement,
//! and destination-level AIMD integration.

use core::time::Duration;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::config::{AdaptiveConfig, AimdLossPolicy, ProbeOutcome, ProfileMode};
use crate::loss::EpochEvents;
use crate::rtt::RttEstimator;

/// AIMD concurrency window for a single scope (per-destination or global).
#[derive(Clone, Copy, Debug)]
pub struct AimdWindow {
    /// Current concurrency window size.
    window: usize,
    /// Minimum window (≥ 1).
    c_min: usize,
    /// Maximum window.
    c_max: usize,
    /// Multiplicative decrease factor δ ∈ [0.4, 0.5].
    delta: f64,
}

/// What happened in an epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochOutcome {
    /// At least `min_completed_probes` completed, zero loss/rate-limit events.
    Lossless,
    /// One or more inferred-loss events occurred.
    Loss,
    /// One or more rate-limit events occurred (without packet loss).
    RateLimit,
    /// Both packet loss and rate-limit events occurred.
    LossAndRateLimit,
    /// Insufficient completions or idle.
    Idle,
}

impl AimdWindow {
    /// Create a new AIMD window with validated bounds.
    ///
    /// Guarantees:
    /// - `c_min ≥ 1` (never zero-sized window)
    /// - `c_max ≥ c_min` (prevents clamp panics)
    /// - `initial` safely clamped to `[c_min, c_max]`
    /// - `delta` clamped to `(0.0, 1.0]`
    pub fn new(initial: usize, c_min: usize, c_max: usize, delta: f64) -> Self {
        let safe_min = c_min.max(1);
        let safe_max = c_max.max(safe_min);
        let safe_delta = delta.clamp(0.01, 1.0);
        Self {
            window: initial.clamp(safe_min, safe_max),
            c_min: safe_min,
            c_max: safe_max,
            delta: safe_delta,
        }
    }

    /// Current window value.
    #[inline]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Set a fixed window (used in Reference profile to enforce caller ceiling).
    #[inline]
    pub fn set_window(&mut self, win: usize) {
        self.window = win.clamp(self.c_min, self.c_max);
    }

    /// Apply an epoch state transition with configured policy.
    ///
    /// - **Lossless**: C_new = min(C_max, C_old.saturating_add(1))
    /// - **Loss / RateLimit**: C_new = max(C_min, ⌊C_old · δ_eff⌋)
    /// - **LossAndRateLimit**: uses conservative `delta.min(rate_limit_delta)`
    ///   to alleviate both conditions while preserving at most one multiplicative decrease per epoch.
    /// - **Idle**: C_new = C_old (no change)
    pub fn apply_epoch(&mut self, outcome: EpochOutcome, policy: AimdLossPolicy) {
        match outcome {
            EpochOutcome::Lossless => {
                self.window = self.window.saturating_add(1).min(self.c_max);
            }
            EpochOutcome::Loss => {
                let reduced = (self.window as f64 * self.delta) as usize;
                self.window = reduced.max(self.c_min);
            }
            EpochOutcome::RateLimit => {
                let eff_delta = match policy {
                    AimdLossPolicy::Shared => Some(self.delta),
                    AimdLossPolicy::SeparateRateLimit { rate_limit_delta } => {
                        Some(rate_limit_delta)
                    }
                    AimdLossPolicy::LossOnly => None,
                };
                if let Some(d) = eff_delta {
                    let reduced = (self.window as f64 * d) as usize;
                    self.window = reduced.max(self.c_min);
                }
            }
            EpochOutcome::LossAndRateLimit => {
                let eff_delta = match policy {
                    AimdLossPolicy::Shared => self.delta,
                    AimdLossPolicy::SeparateRateLimit { rate_limit_delta } => {
                        // Apply the more conservative (smaller) multiplier to alleviate
                        // both loss and rate limiting while enforcing at most one multiplicative
                        // decrease in this epoch.
                        self.delta.min(rate_limit_delta)
                    }
                    AimdLossPolicy::LossOnly => self.delta,
                };
                let reduced = (self.window as f64 * eff_delta) as usize;
                self.window = reduced.max(self.c_min);
            }
            EpochOutcome::Idle => {
                // No change.
            }
        }
    }

    /// Check whether a new probe can be dispatched given current in-flight count.
    #[inline]
    pub fn can_dispatch(&self, in_flight: usize) -> bool {
        in_flight < self.window
    }
}

/// Destination-level state tracking concurrency and backoff (§ 2.3 & § 3.3).
#[derive(Clone, Debug)]
pub struct DestinationTracker {
    /// Destination identifier.
    pub id: u64,
    /// Per-destination AIMD window C_d.
    pub window: AimdWindow,
    /// Current number of probes in-flight to this destination.
    pub in_flight: usize,
    /// Accumulated epoch events for this destination.
    pub events: EpochEvents,
    /// Destination-level backoff counter (§ 2.3). Cleared to 0 on eligible n=0 ACK.
    pub consecutive_backoffs: u32,
    /// Active flag.
    pub active: bool,
    /// Destination-specific RTT estimator (§ 1.1–1.3).
    /// Tracks latency independently for this destination so cross-target latency
    /// variation does not distort timeout calculations.
    pub rtt: Option<RttEstimator>,
}

impl DestinationTracker {
    pub fn new(id: u64, c_min: usize, c_max: usize, delta: f64) -> Self {
        let safe_min = c_min.max(1);
        let safe_max = c_max.max(safe_min);
        Self {
            id,
            window: AimdWindow::new(safe_min, safe_min, safe_max, delta),
            in_flight: 0,
            events: EpochEvents::default(),
            consecutive_backoffs: 0,
            active: true,
            rtt: None,
        }
    }

    /// Record an RTT sample for this destination, initializing destination RTT state if needed.
    pub fn record_rtt_sample(&mut self, sample: Duration, cfg: &AdaptiveConfig) {
        let estimator = self.rtt.get_or_insert_with(|| RttEstimator::new(cfg));
        estimator.record_sample(sample);
    }

    /// Record a probe dispatch to this destination.
    #[inline]
    pub fn on_dispatch(&mut self) {
        self.in_flight = self.in_flight.saturating_add(1);
    }

    /// Attempt to record a probe completion/expiration from this destination.
    ///
    /// Returns `true` if an active in-flight probe was decremented, or `false`
    /// if there were no outstanding in-flight probes (e.g. mismatched/duplicate completion).
    #[inline]
    pub fn try_complete(&mut self) -> bool {
        if self.in_flight > 0 {
            self.in_flight -= 1;
            true
        } else {
            false
        }
    }

    /// Record a probe completion/expiration from this destination.
    #[inline]
    pub fn on_completion(&mut self) {
        self.try_complete();
    }

    /// Destination-level timeout multiplier (§ 2.3).
    ///
    /// Computes gamma^k where k is the number of consecutive backed-off losses
    /// on this destination, capped at 64 to prevent float exponent overflow.
    pub fn backoff_multiplier(&self, gamma: f64) -> f64 {
        if self.consecutive_backoffs == 0 {
            1.0
        } else {
            crate::math::pow_f64_u32(gamma, self.consecutive_backoffs.min(64))
        }
    }

    /// Calculate the backed-off base RTO for this destination given base RTO (§ 2.3).
    ///
    /// Cleared and reset to fresh RTO_0 upon receiving an eligible (n=0) ACK.
    /// Uses pre-multiplication ceiling check to safely prevent Duration::mul_f64 overflow.
    pub fn backed_off_rto(&self, base_rto: Duration, rto_max: Duration, gamma: f64) -> Duration {
        if self.consecutive_backoffs == 0 {
            base_rto
        } else {
            let factor = self.backoff_multiplier(gamma);
            let base_secs = base_rto.as_secs_f64();
            let max_secs = rto_max.as_secs_f64();
            let max_factor = if base_secs > 0.0 {
                max_secs / base_secs
            } else {
                f64::INFINITY
            };

            if factor >= max_factor {
                rto_max
            } else {
                base_rto.mul_f64(factor).min(rto_max)
            }
        }
    }

    /// Record an outcome for this destination (§ 2.3 Karn reset semantics included).
    pub fn record_outcome(&mut self, outcome: ProbeOutcome) {
        match outcome {
            ProbeOutcome::SuccessInitial { .. } => {
                self.events.successes += 1;
                // § 2.3: Reset destination-level backoff state upon eligible ACK
                self.consecutive_backoffs = 0;
            }
            ProbeOutcome::SuccessRetransmitted { .. } => {
                self.events.successes += 1;
                // Ambiguous response: do not reset backoff state (Karn's rule § 2.3)
            }
            ProbeOutcome::ExpectedSilence => {
                self.events.expected_silence += 1;
            }
            ProbeOutcome::InferredLoss => {
                self.events.inferred_loss += 1;
                self.consecutive_backoffs = self.consecutive_backoffs.saturating_add(1);
            }
            ProbeOutcome::InferredRateLimit => {
                self.events.inferred_rate_limit += 1;
            }
            ProbeOutcome::Unclassified => {
                self.events.unclassified += 1;
            }
        }
    }
}

/// Dual-ceiling concurrency controller (§ 3.3).
///
/// Enforces both per-destination and global concurrency limits via independent
/// AIMD windows.
#[derive(Clone, Debug)]
pub struct ConcurrencyController {
    /// Global AIMD window C_global.
    pub global: AimdWindow,
    /// Total in-flight probes across all destinations: Σ in_flight_d.
    pub global_in_flight: usize,
    /// Tracked destinations.
    pub destinations: Vec<DestinationTracker>,
    /// Profile mode (Reference enforces static ceiling; Adaptive runs AIMD).
    mode: ProfileMode,
    /// Per-destination AIMD parameters.
    dest_delta: f64,
    dest_c_min: usize,
    dest_c_max: usize,
    loss_policy: AimdLossPolicy,
    min_completed_probes: usize,
}

impl ConcurrencyController {
    /// Create from config with validated bounds.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        let safe_min_global = cfg.c_min_global.max(1);
        let safe_max_global = cfg.c_max_global.max(safe_min_global);
        let safe_min_dest = cfg.c_min_dest.max(1);
        let safe_max_dest = cfg.c_max_dest.max(safe_min_dest);
        let safe_min_completed = cfg.min_completed_probes.max(1);

        let initial_global = if cfg.mode == ProfileMode::Reference {
            safe_max_global
        } else {
            safe_min_global
        };
        let initial_dest = if cfg.mode == ProfileMode::Reference {
            safe_max_dest
        } else {
            safe_min_dest
        };

        Self {
            global: AimdWindow::new(
                initial_global,
                safe_min_global,
                safe_max_global,
                cfg.aimd_delta,
            ),
            global_in_flight: 0,
            destinations: Vec::new(),
            mode: cfg.mode,
            dest_delta: cfg.aimd_delta,
            dest_c_min: initial_dest,
            dest_c_max: safe_max_dest,
            loss_policy: cfg.aimd_loss_policy,
            min_completed_probes: safe_min_completed,
        }
    }

    /// Find or create a destination tracker.
    pub fn get_or_create_dest(&mut self, dest_id: u64) -> &mut DestinationTracker {
        if let Some(pos) = self.destinations.iter().position(|d| d.id == dest_id) {
            return &mut self.destinations[pos];
        }
        let tracker = DestinationTracker::new(
            dest_id,
            self.dest_c_min,
            self.dest_c_max,
            self.dest_delta,
        );
        self.destinations.push(tracker);
        self.destinations.last_mut().unwrap()
    }

    /// Find an existing destination tracker.
    pub fn get_dest(&self, dest_id: u64) -> Option<&DestinationTracker> {
        self.destinations.iter().find(|d| d.id == dest_id)
    }

    /// Get the destination-backed-off RTO for a specific destination (§ 2.3).
    ///
    /// Applies the destination's latched consecutive backoff multiplier to `base_rto`.
    pub fn dest_backed_off_rto(
        &self,
        dest_id: u64,
        base_rto: Duration,
        rto_max: Duration,
        gamma: f64,
    ) -> Duration {
        match self.get_dest(dest_id) {
            Some(d) => d.backed_off_rto(base_rto, rto_max, gamma),
            None => base_rto,
        }
    }

    /// Current total number of in-flight probes across all destinations.
    #[inline]
    pub fn global_in_flight(&self) -> usize {
        self.global_in_flight
    }

    /// Dual ceiling check (§ 3.3):
    ///
    /// in_flight_d < C_d  AND  Σ in_flight_d < C_global.
    #[inline]
    pub fn can_dispatch(&self, dest_id: u64) -> bool {
        if !self.global.can_dispatch(self.global_in_flight) {
            return false;
        }
        match self.get_dest(dest_id) {
            Some(d) => d.window.can_dispatch(d.in_flight),
            None => true, // Window allows at least initial probe
        }
    }

    /// Record a probe dispatch to destination `dest_id`.
    pub fn on_dispatch(&mut self, dest_id: u64) {
        self.global_in_flight = self.global_in_flight.saturating_add(1);
        let dest = self.get_or_create_dest(dest_id);
        dest.on_dispatch();
    }

    /// Attempt to complete a probe for destination `dest_id`.
    ///
    /// Only decrements the global in-flight counter if a matching destination exists
    /// and had an active in-flight probe. Returns `true` on successful completion,
    /// or `false` on duplicate/mismatched completion.
    pub fn try_complete(&mut self, dest_id: u64) -> bool {
        if let Some(dest) = self.destinations.iter_mut().find(|d| d.id == dest_id)
            && dest.try_complete()
        {
            self.global_in_flight = self.global_in_flight.saturating_sub(1);
            return true;
        }
        false
    }

    /// Record a probe completion/expiration for destination `dest_id`.
    ///
    /// Ensures global in-flight counters are only decremented when an actual
    /// in-flight probe completes.
    #[inline]
    pub fn on_completion(&mut self, dest_id: u64) {
        self.try_complete(dest_id);
    }

    /// Record an outcome for destination `dest_id`.
    pub fn record_dest_outcome(&mut self, dest_id: u64, outcome: ProbeOutcome) {
        let dest = self.get_or_create_dest(dest_id);
        dest.record_outcome(outcome);
    }

    /// Close an epoch and apply AIMD state transitions to all active destinations
    /// and the global window (§ 3.2 & § 3.3).
    ///
    /// If `ProfileMode::Reference`, windows are fixed at their ceilings.
    pub fn apply_epoch(&mut self) {
        if self.mode == ProfileMode::Reference {
            // Reference Profile (§ 5): Concurrency is fixed / caller-supplied ceiling.
            // No dynamic window modifications.
            for dest in &mut self.destinations {
                dest.events = EpochEvents::default();
            }
            return;
        }

        let mut global_has_loss = false;
        let mut global_has_rate_limit = false;
        let mut global_completions: usize = 0;

        // Apply per-destination transitions
        for dest in &mut self.destinations {
            let has_loss = dest.events.inferred_loss > 0;
            let has_rate = dest.events.inferred_rate_limit > 0;
            let completions = dest.events.successes + dest.events.inferred_loss + dest.events.inferred_rate_limit;

            if has_loss {
                global_has_loss = true;
            }
            if has_rate {
                global_has_rate_limit = true;
            }
            global_completions = global_completions.saturating_add(completions);

            let dest_outcome = if has_loss && has_rate {
                EpochOutcome::LossAndRateLimit
            } else if has_loss {
                EpochOutcome::Loss
            } else if has_rate {
                EpochOutcome::RateLimit
            } else if completions >= self.min_completed_probes && completions > 0 {
                EpochOutcome::Lossless
            } else {
                EpochOutcome::Idle
            };

            dest.window.apply_epoch(dest_outcome, self.loss_policy);
            dest.events = EpochEvents::default();
        }

        // Apply global transition
        let global_outcome = if global_has_loss && global_has_rate_limit {
            EpochOutcome::LossAndRateLimit
        } else if global_has_loss {
            EpochOutcome::Loss
        } else if global_has_rate_limit {
            EpochOutcome::RateLimit
        } else if global_completions >= self.min_completed_probes && global_completions > 0 {
            EpochOutcome::Lossless
        } else {
            EpochOutcome::Idle
        };

        self.global.apply_epoch(global_outcome, self.loss_policy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    #[test]
    fn lossless_increments() {
        let mut w = AimdWindow::new(10, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Lossless, AimdLossPolicy::Shared);
        assert_eq!(w.window(), 11);
    }

    #[test]
    fn lossless_capped_at_max() {
        let mut w = AimdWindow::new(100, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Lossless, AimdLossPolicy::Shared);
        assert_eq!(w.window(), 100);
    }

    #[test]
    fn loss_halves() {
        let mut w = AimdWindow::new(20, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Loss, AimdLossPolicy::Shared);
        assert_eq!(w.window(), 10);
    }

    #[test]
    fn loss_respects_floor() {
        let mut w = AimdWindow::new(1, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Loss, AimdLossPolicy::Shared);
        assert_eq!(w.window(), 1);
    }

    #[test]
    fn separate_rate_limit_policy() {
        let mut w = AimdWindow::new(20, 1, 100, 0.5);
        let policy = AimdLossPolicy::SeparateRateLimit {
            rate_limit_delta: 0.8,
        };
        // 20 * 0.8 = 16
        w.apply_epoch(EpochOutcome::RateLimit, policy);
        assert_eq!(w.window(), 16);
    }

    #[test]
    fn loss_only_policy_ignores_rate_limit() {
        let mut w = AimdWindow::new(20, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::RateLimit, AimdLossPolicy::LossOnly);
        assert_eq!(w.window(), 20); // Unchanged!
    }

    #[test]
    fn loss_and_rate_limit_precedence_separate_rate_limit() {
        let mut w = AimdWindow::new(20, 1, 100, 0.5);
        let policy = AimdLossPolicy::SeparateRateLimit {
            rate_limit_delta: 0.8,
        };
        // Both loss (0.5) and rate limit (0.8) occurred in the same epoch.
        // Precedence applies the more conservative multiplier: min(0.5, 0.8) = 0.5 -> window = 10
        w.apply_epoch(EpochOutcome::LossAndRateLimit, policy);
        assert_eq!(w.window(), 10);

        let mut w2 = AimdWindow::new(20, 1, 100, 0.5);
        let stricter_rate_policy = AimdLossPolicy::SeparateRateLimit {
            rate_limit_delta: 0.3,
        };
        // Stricter rate limit (0.3) vs loss (0.5) -> min(0.5, 0.3) = 0.3 -> window = 6
        w2.apply_epoch(EpochOutcome::LossAndRateLimit, stricter_rate_policy);
        assert_eq!(w2.window(), 6);
    }

    #[test]
    fn try_complete_guards_against_mismatched_completions() {
        let cfg = AdaptiveConfig::adaptive();
        let mut ctrl = ConcurrencyController::new(&cfg);
        let dest_id = 42;

        ctrl.on_dispatch(dest_id);
        assert_eq!(ctrl.global_in_flight, 1);
        assert_eq!(ctrl.get_dest(dest_id).unwrap().in_flight, 1);

        // Mismatched completion for a nonexistent destination
        assert!(!ctrl.try_complete(999));
        assert_eq!(ctrl.global_in_flight, 1, "Global in-flight must not decrement for nonexistent dest");

        // Valid completion
        assert!(ctrl.try_complete(dest_id));
        assert_eq!(ctrl.global_in_flight, 0);

        // Duplicate completion for the same destination (now at 0)
        assert!(!ctrl.try_complete(dest_id));
        assert_eq!(ctrl.global_in_flight, 0, "Global in-flight must not underflow or decrement on duplicate completion");
    }

    #[test]
    fn edge_cases_construction_sanitization() {
        // c_min = 0, c_min > c_max, min_completed = 0
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.c_min_global = 0;
        cfg.c_max_global = 0;
        cfg.c_min_dest = 50;
        cfg.c_max_dest = 10; // c_min > c_max
        cfg.min_completed_probes = 0;

        let ctrl = ConcurrencyController::new(&cfg);
        // c_min_global should be sanitized to at least 1
        assert!(ctrl.global.c_min >= 1);
        assert!(ctrl.global.c_max >= ctrl.global.c_min);
        // dest c_max should be clamped to at least dest c_min
        assert!(ctrl.dest_c_max >= ctrl.dest_c_min);
        assert!(ctrl.min_completed_probes >= 1);
    }

    #[test]
    fn destination_backed_off_rto_calculation() {
        let mut tracker = DestinationTracker::new(1, 1, 10, 0.5);
        let base_rto = Duration::from_millis(100);
        let rto_max = Duration::from_secs(3);
        let gamma = 1.5;

        // No backoffs -> base RTO
        assert_eq!(tracker.backed_off_rto(base_rto, rto_max, gamma), base_rto);

        // One loss -> 100 * 1.5 = 150ms
        tracker.record_outcome(ProbeOutcome::InferredLoss);
        assert_eq!(tracker.consecutive_backoffs, 1);
        assert_eq!(
            tracker.backed_off_rto(base_rto, rto_max, gamma),
            Duration::from_millis(150)
        );

        // Second loss -> 100 * 2.25 = 225ms
        tracker.record_outcome(ProbeOutcome::InferredLoss);
        assert_eq!(tracker.consecutive_backoffs, 2);
        assert_eq!(
            tracker.backed_off_rto(base_rto, rto_max, gamma),
            Duration::from_millis(225)
        );

        // Success initial resets backoff -> back to 100ms
        tracker.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(50),
        });
        assert_eq!(tracker.consecutive_backoffs, 0);
        assert_eq!(tracker.backed_off_rto(base_rto, rto_max, gamma), base_rto);
    }

    #[test]
    fn dual_ceiling_controller() {
        let cfg = AdaptiveConfig::adaptive();
        let mut ctrl = ConcurrencyController::new(&cfg);
        let dest_id = 42;

        assert!(ctrl.can_dispatch(dest_id));
        ctrl.on_dispatch(dest_id);
        // Dest window starts at 1, so 1 in-flight should block further dispatch to this dest
        assert!(!ctrl.can_dispatch(dest_id));

        // Another destination still has dest capacity, but global capacity is checked
        let other_dest = 99;
        // Global window starts at 1, so global capacity is full!
        assert!(!ctrl.can_dispatch(other_dest));

        // Complete the probe
        ctrl.on_completion(dest_id);
        assert!(ctrl.can_dispatch(dest_id));
        assert!(ctrl.can_dispatch(other_dest));
    }

    #[test]
    fn destination_level_karn_reset() {
        let mut tracker = DestinationTracker::new(1, 1, 10, 0.5);
        tracker.record_outcome(ProbeOutcome::InferredLoss);
        assert_eq!(tracker.consecutive_backoffs, 1);

        // Ambiguous retransmitted probe should NOT reset backoff
        tracker.record_outcome(ProbeOutcome::SuccessRetransmitted { attempt: 1 });
        assert_eq!(tracker.consecutive_backoffs, 1);

        // Eligible n=0 response resets backoff
        tracker.record_outcome(ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(50),
        });
        assert_eq!(tracker.consecutive_backoffs, 0);
    }

    #[test]
    fn reference_mode_isolation_fixed_concurrency() {
        let cfg = AdaptiveConfig::reference();
        let mut ctrl = ConcurrencyController::new(&cfg);
        let initial_global = ctrl.global.window();
        assert_eq!(initial_global, cfg.c_max_global);

        // Record loss on a destination
        ctrl.record_dest_outcome(1, ProbeOutcome::InferredLoss);
        ctrl.apply_epoch();

        // Reference mode must NEVER decrease or increase window
        assert_eq!(ctrl.global.window(), initial_global);
    }

    #[test]
    fn backed_off_rto_extreme_backoffs_do_not_panic() {
        let mut tracker = DestinationTracker::new(1, 1, 10, 0.5);
        tracker.consecutive_backoffs = 64; // factor = 2^64
        let rto = tracker.backed_off_rto(Duration::from_secs(1), Duration::from_secs(60), 2.0);
        assert_eq!(rto, Duration::from_secs(60));

        tracker.consecutive_backoffs = u32::MAX;
        let rto_max = tracker.backed_off_rto(Duration::from_secs(1), Duration::from_secs(60), 2.0);
        assert_eq!(rto_max, Duration::from_secs(60));
    }
}
