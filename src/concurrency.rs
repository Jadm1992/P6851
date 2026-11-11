//! § 3 — Epoch-based AIMD concurrency control.
//!
//! Implements §§ 3.1–3.3: observation epochs, additive-increase /
//! multiplicative-decrease state transitions, and dual-ceiling enforcement.

use crate::config::AdaptiveConfig;

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
    /// At least one probe completed, zero loss/rate-limit events.
    Lossless,
    /// One or more inferred-loss or rate-limit events occurred.
    Loss,
    /// No probes completed, no losses.
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

    /// Apply an epoch state transition.
    ///
    /// - **Lossless**: C_new = min(C_max, C_old + 1)
    /// - **Loss**: C_new = max(C_min, ⌊C_old · δ⌋)
    /// - **Idle**: C_new = C_old (no change)
    pub fn apply_epoch(&mut self, outcome: EpochOutcome) {
        match outcome {
            EpochOutcome::Lossless => {
                self.window = (self.window + 1).min(self.c_max);
            }
            EpochOutcome::Loss => {
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

/// Dual-ceiling concurrency controller (§ 3.3).
///
/// Enforces both per-destination and global concurrency limits via independent
/// AIMD windows.
#[derive(Clone, Debug)]
pub struct ConcurrencyController {
    /// Global AIMD window.
    pub global: AimdWindow,
    /// Per-destination AIMD window defaults.
    dest_delta: f64,
    dest_c_min: usize,
    dest_c_max: usize,
    dest_initial: usize,
}

impl ConcurrencyController {
    /// Create from config, starting at the minimum window sizes.
    pub fn new(cfg: &AdaptiveConfig) -> Self {
        Self {
            global: AimdWindow::new(
                cfg.c_min_global,
                cfg.c_min_global,
                cfg.c_max_global,
                cfg.aimd_delta,
            ),
            dest_delta: cfg.aimd_delta,
            dest_c_min: cfg.c_min_dest,
            dest_c_max: cfg.c_max_dest,
            dest_initial: cfg.c_min_dest,
        }
    }

    /// Create a new per-destination AIMD window with the controller's defaults.
    pub fn new_dest_window(&self) -> AimdWindow {
        AimdWindow::new(
            self.dest_initial,
            self.dest_c_min,
            self.dest_c_max,
            self.dest_delta,
        )
    }

    /// Check whether a probe can be dispatched to destination `d` given its
    /// per-destination in-flight count and the global in-flight total.
    ///
    /// Enforces: in_flight_d ≤ C_d AND Σ in_flight_d ≤ C_global.
    #[inline]
    pub fn can_dispatch(&self, dest_window: &AimdWindow, dest_in_flight: usize, global_in_flight: usize) -> bool {
        dest_window.can_dispatch(dest_in_flight) && self.global.can_dispatch(global_in_flight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AdaptiveConfig;

    #[test]
    fn lossless_increments() {
        let mut w = AimdWindow::new(10, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Lossless);
        assert_eq!(w.window(), 11);
    }

    #[test]
    fn lossless_capped_at_max() {
        let mut w = AimdWindow::new(100, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Lossless);
        assert_eq!(w.window(), 100);
    }

    #[test]
    fn loss_halves() {
        let mut w = AimdWindow::new(20, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Loss);
        assert_eq!(w.window(), 10);
    }

    #[test]
    fn loss_respects_floor() {
        let mut w = AimdWindow::new(1, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Loss);
        assert_eq!(w.window(), 1);
    }

    #[test]
    fn loss_with_delta_0_4() {
        let mut w = AimdWindow::new(10, 1, 100, 0.4);
        w.apply_epoch(EpochOutcome::Loss);
        // ⌊10 * 0.4⌋ = 4
        assert_eq!(w.window(), 4);
    }

    #[test]
    fn idle_unchanged() {
        let mut w = AimdWindow::new(15, 1, 100, 0.5);
        w.apply_epoch(EpochOutcome::Idle);
        assert_eq!(w.window(), 15);
    }

    #[test]
    fn dual_ceiling_enforcement() {
        let cfg = AdaptiveConfig::adaptive();
        let ctrl = ConcurrencyController::new(&cfg);
        let dw = ctrl.new_dest_window();

        // Both windows at minimum (1), should allow dispatch with 0 in flight
        assert!(ctrl.can_dispatch(&dw, 0, 0));
        // Cannot dispatch if dest is full
        assert!(!ctrl.can_dispatch(&dw, 1, 0));
    }

    #[test]
    fn dispatch_respects_global_ceiling() {
        let cfg = AdaptiveConfig::adaptive();
        let mut ctrl = ConcurrencyController::new(&cfg);

        // Grow global window
        for _ in 0..5 {
            ctrl.global.apply_epoch(EpochOutcome::Lossless);
        }

        let mut dw = ctrl.new_dest_window();
        // Grow dest window too
        for _ in 0..5 {
            dw.apply_epoch(EpochOutcome::Lossless);
        }

        let global_win = ctrl.global.window();
        // Dest allows, but global is full
        assert!(!ctrl.can_dispatch(&dw, 0, global_win));
        // Dest allows and global has room
        assert!(ctrl.can_dispatch(&dw, 0, global_win - 1));
    }
}
