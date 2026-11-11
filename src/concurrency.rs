//! § 3 — Epoch-based AIMD concurrency control.
//!
//! Implements §§ 3.1–3.3: observation epochs, additive-increase /
//! multiplicative-decrease state transitions, dual-ceiling enforcement,
//! and destination-level AIMD integration.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::config::{AdaptiveConfig, AimdLossPolicy, ProbeOutcome, ProfileMode};
use crate::loss::EpochEvents;

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
    /// Create a new AIMD window.
    pub fn new(initial: usize, c_min: usize, c_max: usize, delta: f64) -> Self {
        Self {
            window: initial.clamp(c_min, c_max),
            c_min: c_min.max(1),
            c_max,
            delta,
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
    /// - **Lossless**: C_new = min(C_max, C_old + 1)
    /// - **Loss / RateLimit**: C_new = max(C_min, ⌊C_old · δ_eff⌋)
    /// - **Idle**: C_new = C_old (no change)
    pub fn apply_epoch(&mut self, outcome: EpochOutcome, policy: AimdLossPolicy) {
        match outcome {
            EpochOutcome::Lossless => {
                self.window = (self.window + 1).min(self.c_max);
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
                let reduced = (self.window as f64 * self.delta) as usize;
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
}

impl DestinationTracker {
    pub fn new(id: u64, c_min: usize, c_max: usize, delta: f64) -> Self {
        Self {
            id,
            window: AimdWindow::new(c_min, c_min, c_max, delta),
            in_flight: 0,
            events: EpochEvents::default(),
            consecutive_backoffs: 0,
            active: true,
        }
    }

    /// Record a probe dispatch to this destination.
    #[inline]
    pub fn on_dispatch(&mut self) {
        self.in_flight += 1;
    }

    /// Record a probe completion/expiration from this destination.
    #[inline]
    pub fn on_completion(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
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
    /// Create from config.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        let initial_global = if cfg.mode == ProfileMode::Reference {
            cfg.c_max_global
        } else {
            cfg.c_min_global
        };
        let initial_dest = if cfg.mode == ProfileMode::Reference {
            cfg.c_max_dest
        } else {
            cfg.c_min_dest
        };

        Self {
            global: AimdWindow::new(
                initial_global,
                cfg.c_min_global,
                cfg.c_max_global,
                cfg.aimd_delta,
            ),
            global_in_flight: 0,
            destinations: Vec::new(),
            mode: cfg.mode,
            dest_delta: cfg.aimd_delta,
            dest_c_min: initial_dest,
            dest_c_max: cfg.c_max_dest,
            loss_policy: cfg.aimd_loss_policy,
            min_completed_probes: cfg.min_completed_probes,
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
        self.global_in_flight += 1;
        let dest = self.get_or_create_dest(dest_id);
        dest.on_dispatch();
    }

    /// Record a probe completion/expiration for destination `dest_id`.
    pub fn on_completion(&mut self, dest_id: u64) {
        self.global_in_flight = self.global_in_flight.saturating_sub(1);
        if let Some(dest) = self.destinations.iter_mut().find(|d| d.id == dest_id) {
            dest.on_completion();
        }
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
        let mut global_completions = 0;

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
            global_completions += completions;

            let dest_outcome = if has_loss && has_rate {
                EpochOutcome::LossAndRateLimit
            } else if has_loss {
                EpochOutcome::Loss
            } else if has_rate {
                EpochOutcome::RateLimit
            } else if completions >= self.min_completed_probes {
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
        } else if global_completions >= self.min_completed_probes {
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
            rtt: core::time::Duration::from_millis(50),
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
}
