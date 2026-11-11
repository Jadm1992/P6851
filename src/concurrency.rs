//! § 3 — Epoch-based AIMD concurrency control.
//!
//! Implements §§ 3.1–3.3: observation epochs, additive-increase /
//! multiplicative-decrease state transitions, dual-ceiling enforcement,
//! and destination-level AIMD integration.

use core::time::Duration;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

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

/// Independent leaky-bucket / rate pacer (§ 7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pacer {
    /// Transmission interval (minimum spacing between packets).
    pub interval: Duration,
    /// Earliest timestamp (duration since epoch) when transmission is permitted.
    pub next_tx: Duration,
}

impl Pacer {
    /// Create a new pacer with the given rate interval.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_tx: Duration::ZERO,
        }
    }

    /// Check if transmission is allowed at `now`.
    /// Returns `None` if allowed, or `Some(wake_at)` if pacing delay is required.
    #[inline]
    pub fn check(&self, now: Duration) -> Option<Duration> {
        if now >= self.next_tx {
            None
        } else {
            Some(self.next_tx)
        }
    }

    /// Advance the pacer by one transmission.
    #[inline]
    pub fn consume(&mut self, now: Duration) {
        let base = if now > self.next_tx { now } else { self.next_tx };
        self.next_tx = base.saturating_add(self.interval);
    }
}

/// Group-level state tracking concurrency and pacing (§ 3 & § 7).
#[derive(Clone, Debug)]
pub struct GroupTracker {
    /// Group identifier.
    pub id: u64,
    /// Per-group AIMD window C_g.
    pub window: AimdWindow,
    /// In-flight probes in this group.
    pub in_flight: usize,
    /// Accumulated epoch events for this group.
    pub events: EpochEvents,
    /// Independent group pacer (§ 7).
    pub pacer: Option<Pacer>,
    /// Active flag.
    pub active: bool,
}

impl GroupTracker {
    pub fn new(id: u64, c_min: usize, c_max: usize, delta: f64) -> Self {
        let safe_min = c_min.max(1);
        let safe_max = c_max.max(safe_min);
        Self {
            id,
            window: AimdWindow::new(safe_min, safe_min, safe_max, delta),
            in_flight: 0,
            events: EpochEvents::default(),
            pacer: None,
            active: true,
        }
    }

    /// Record an outcome for this group.
    pub fn record_outcome(&mut self, outcome: ProbeOutcome) {
        match outcome {
            ProbeOutcome::SuccessInitial { .. } | ProbeOutcome::SuccessRetransmitted { .. } => {
                self.events.successes += 1;
            }
            ProbeOutcome::ExpectedSilence => {
                self.events.expected_silence += 1;
            }
            ProbeOutcome::InferredLoss => {
                self.events.inferred_loss += 1;
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

/// Destination-level state tracking concurrency and backoff (§ 2.3 & § 3.3).
#[derive(Clone, Debug)]
pub struct DestinationTracker {
    /// Destination identifier.
    pub id: u64,
    /// Group to which this destination belongs (if any).
    pub group_id: Option<u64>,
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
    /// Independent destination pacer (§ 7).
    pub pacer: Option<Pacer>,
}

impl DestinationTracker {
    pub fn new(id: u64, c_min: usize, c_max: usize, delta: f64) -> Self {
        let safe_min = c_min.max(1);
        let safe_max = c_max.max(safe_min);
        Self {
            id,
            group_id: None,
            window: AimdWindow::new(safe_min, safe_min, safe_max, delta),
            in_flight: 0,
            events: EpochEvents::default(),
            consecutive_backoffs: 0,
            active: true,
            rtt: None,
            pacer: None,
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

/// Dual/triple-ceiling concurrency controller with independent pacing (§ 3.3 & § 7).
///
/// Enforces per-destination, per-group, and global concurrency limits via independent
/// AIMD windows, and provides atomic hierarchical pacing across all scopes.
#[derive(Clone, Debug)]
pub struct ConcurrencyController {
    /// Global AIMD window C_global.
    pub global: AimdWindow,
    /// Total in-flight probes across all destinations: Σ in_flight_d.
    pub global_in_flight: usize,
    /// Tracked destinations.
    pub destinations: Vec<DestinationTracker>,
    /// Tracked destination groups (§ 3).
    pub groups: Vec<GroupTracker>,
    /// Independent global pacer (§ 7).
    pub global_pacer: Option<Pacer>,
    /// Minimum distinct groups required for global evidence (§ 3): G_min = 4.
    pub g_min: usize,
    /// Maximum fractional contribution per group (§ 3): f_max = 0.25.
    pub f_max: f64,
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
            groups: Vec::new(),
            global_pacer: None,
            g_min: 4,
            f_max: 0.25,
            mode: cfg.mode,
            dest_delta: cfg.aimd_delta,
            dest_c_min: initial_dest,
            dest_c_max: safe_max_dest,
            loss_policy: cfg.aimd_loss_policy,
            min_completed_probes: safe_min_completed,
        }
    }

    /// Find or create a group tracker (§ 3).
    pub fn get_or_create_group(&mut self, group_id: u64) -> &mut GroupTracker {
        if let Some(pos) = self.groups.iter().position(|g| g.id == group_id) {
            return &mut self.groups[pos];
        }
        let tracker = GroupTracker::new(
            group_id,
            self.dest_c_min,
            self.dest_c_max,
            self.dest_delta,
        );
        self.groups.push(tracker);
        self.groups.last_mut().unwrap()
    }

    /// Find an existing group tracker.
    pub fn get_group(&self, group_id: u64) -> Option<&GroupTracker> {
        self.groups.iter().find(|g| g.id == group_id)
    }

    /// Find an existing group tracker mutably.
    pub fn get_group_mut(&mut self, group_id: u64) -> Option<&mut GroupTracker> {
        self.groups.iter_mut().find(|g| g.id == group_id)
    }

    /// Assign destination `dest_id` to group `group_id`.
    pub fn set_dest_group(&mut self, dest_id: u64, group_id: u64) {
        let _ = self.get_or_create_group(group_id);
        let dest = self.get_or_create_dest(dest_id);
        dest.group_id = Some(group_id);
    }

    /// Configure a transmission pacer on global, group, or destination scope (§ 7).
    pub fn set_pacer(&mut self, dest_id: Option<u64>, group_id: Option<u64>, interval: Duration) {
        if let Some(did) = dest_id {
            let dest = self.get_or_create_dest(did);
            dest.pacer = Some(Pacer::new(interval));
        } else if let Some(gid) = group_id {
            let group = self.get_or_create_group(gid);
            group.pacer = Some(Pacer::new(interval));
        } else {
            self.global_pacer = Some(Pacer::new(interval));
        }
    }

    /// Atomically check and acquire pacing reservation across all applicable scopes (§ 7).
    ///
    /// If all scopes permit transmission at `now`:
    ///   Consumes atomic reservation across all applicable pacers and returns `Ok(())`.
    /// If ANY scope requires pacing:
    ///   NONE of the pacers are consumed, and returns `Err(wake_at)` with the maximum
    ///   required wake timestamp.
    pub fn try_acquire_pacing(&mut self, dest_id: u64, now: Duration) -> Result<(), Duration> {
        let mut wake_at = now;
        let mut must_wait = false;

        // 1. Check global pacer
        if let Some(p) = &self.global_pacer
            && let Some(w) = p.check(now)
        {
            must_wait = true;
            if w > wake_at {
                wake_at = w;
            }
        }

        // Find destination and group
        let group_id = self.get_dest(dest_id).and_then(|d| d.group_id);

        // 2. Check group pacer
        if let Some(gid) = group_id
            && let Some(g) = self.get_group(gid)
            && let Some(p) = &g.pacer
            && let Some(w) = p.check(now)
        {
            must_wait = true;
            if w > wake_at {
                wake_at = w;
            }
        }

        // 3. Check destination pacer
        if let Some(d) = self.get_dest(dest_id)
            && let Some(p) = &d.pacer
            && let Some(w) = p.check(now)
        {
            must_wait = true;
            if w > wake_at {
                wake_at = w;
            }
        }

        if must_wait {
            // Atomic failure: NONE of the tokens are consumed
            return Err(wake_at);
        }

        // Atomic success: consume all applicable pacers
        if let Some(p) = &mut self.global_pacer {
            p.consume(now);
        }
        if let Some(gid) = group_id
            && let Some(g) = self.get_group_mut(gid)
            && let Some(p) = &mut g.pacer
        {
            p.consume(now);
        }
        if let Some(d) = self.get_dest_mut(dest_id)
            && let Some(p) = &mut d.pacer
        {
            p.consume(now);
        }

        Ok(())
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

    /// Find an existing destination tracker mutably.
    pub fn get_dest_mut(&mut self, dest_id: u64) -> Option<&mut DestinationTracker> {
        self.destinations.iter_mut().find(|d| d.id == dest_id)
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

    /// Hierarchical ceiling check (§ 3.3):
    ///
    /// Checks global ceiling, group ceiling (if assigned), and destination ceiling.
    #[inline]
    pub fn can_dispatch(&self, dest_id: u64) -> bool {
        if !self.global.can_dispatch(self.global_in_flight) {
            return false;
        }
        if let Some(d) = self.get_dest(dest_id) {
            if let Some(gid) = d.group_id
                && let Some(g) = self.get_group(gid)
                && !g.window.can_dispatch(g.in_flight)
            {
                return false;
            }
            d.window.can_dispatch(d.in_flight)
        } else {
            true // Window allows at least initial probe
        }
    }

    /// Record a probe dispatch to destination `dest_id`.
    pub fn on_dispatch(&mut self, dest_id: u64) {
        self.global_in_flight = self.global_in_flight.saturating_add(1);
        let dest = self.get_or_create_dest(dest_id);
        dest.on_dispatch();
        let group_id = dest.group_id;
        if let Some(gid) = group_id
            && let Some(g) = self.groups.iter_mut().find(|g| g.id == gid)
        {
            g.in_flight = g.in_flight.saturating_add(1);
        }
    }

    /// Attempt to complete a probe for destination `dest_id`.
    ///
    /// Only decrements in-flight counters if a matching destination exists
    /// and had an active in-flight probe. Returns `true` on successful completion,
    /// or `false` on duplicate/mismatched completion.
    pub fn try_complete(&mut self, dest_id: u64) -> bool {
        if let Some(dest) = self.destinations.iter_mut().find(|d| d.id == dest_id)
            && dest.try_complete()
        {
            let group_id = dest.group_id;
            if let Some(gid) = group_id
                && let Some(g) = self.groups.iter_mut().find(|g| g.id == gid)
            {
                g.in_flight = g.in_flight.saturating_sub(1);
            }
            self.global_in_flight = self.global_in_flight.saturating_sub(1);
            return true;
        }
        false
    }

    /// Record a probe completion/expiration for destination `dest_id`.
    #[inline]
    pub fn on_completion(&mut self, dest_id: u64) {
        self.try_complete(dest_id);
    }

    /// Record an outcome for destination `dest_id`.
    pub fn record_dest_outcome(&mut self, dest_id: u64, outcome: ProbeOutcome) {
        let dest = self.get_or_create_dest(dest_id);
        dest.record_outcome(outcome);
        let group_id = dest.group_id;
        if let Some(gid) = group_id
            && let Some(g) = self.groups.iter_mut().find(|g| g.id == gid)
        {
            g.record_outcome(outcome);
        }
    }

    /// Close an epoch and apply AIMD state transitions to all active destinations,
    /// groups, and global window (§ 3.2 & § 3.3).
    ///
    /// If `ProfileMode::Reference`, windows are fixed at their ceilings.
    pub fn apply_epoch(&mut self) {
        if self.mode == ProfileMode::Reference {
            // Reference Profile (§ 5): Concurrency is fixed / caller-supplied ceiling.
            // No dynamic window modifications.
            for dest in &mut self.destinations {
                dest.events = EpochEvents::default();
            }
            for group in &mut self.groups {
                group.events = EpochEvents::default();
            }
            return;
        }

        let mut global_completions: usize = 0;

        // 1. Apply per-destination transitions (isolated to destination)
        for dest in &mut self.destinations {
            let has_loss = dest.events.inferred_loss > 0;
            let has_rate = dest.events.inferred_rate_limit > 0;
            let completions = dest.events.successes + dest.events.inferred_loss + dest.events.inferred_rate_limit;
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

        // 2. Apply per-group transitions (isolated to group) and gather global evidence
        let mut groups_satisfying_trigger = 0;
        let mut group_raw_confidences = Vec::new();
        let mut group_has_rate = false;

        for group in &mut self.groups {
            let has_loss = group.events.inferred_loss > 0;
            let has_rate = group.events.inferred_rate_limit > 0;
            let eligible = group.events.eligible_count();

            if has_rate {
                group_has_rate = true;
            }

            let raw_c_loss = group.events.inferred_loss as f64;
            group_raw_confidences.push(raw_c_loss);

            // Loss trigger per group tier: n_eligible >= N_min, sum c_loss >= E_min, sum c_loss / n >= p_min
            let trigger_met = eligible >= 1 && raw_c_loss >= 1.0 && (raw_c_loss / (eligible as f64)) >= 0.05;
            if trigger_met {
                groups_satisfying_trigger += 1;
            }

            let group_outcome = if has_loss && has_rate {
                EpochOutcome::LossAndRateLimit
            } else if has_loss {
                EpochOutcome::Loss
            } else if has_rate {
                EpochOutcome::RateLimit
            } else if eligible >= self.min_completed_probes && eligible > 0 {
                EpochOutcome::Lossless
            } else {
                EpochOutcome::Idle
            };

            group.window.apply_epoch(group_outcome, self.loss_policy);
            group.events = EpochEvents::default();
        }

        // 3. Apply global transition (§ 3: Global Evidence & Water-Filling)
        let global_outcome = if !self.groups.is_empty() {
            // Hierarchical Mode:
            // Global evidence holds in an epoch iff:
            // - At least G_min distinct groups each independently satisfy their own loss-trigger conditions
            // - The consistent capped sum S meets global evidence and loss-rate thresholds
            let (capped_s, _) = solve_water_filling(&group_raw_confidences, self.f_max);
            let global_loss_trigger = groups_satisfying_trigger >= self.g_min
                && capped_s >= 1.0
                && (global_completions > 0 && (capped_s / (global_completions as f64)) >= 0.05);

            if global_loss_trigger {
                if group_has_rate {
                    EpochOutcome::LossAndRateLimit
                } else {
                    EpochOutcome::Loss
                }
            } else if group_has_rate && groups_satisfying_trigger >= self.g_min {
                EpochOutcome::RateLimit
            } else if global_completions >= self.min_completed_probes && global_completions > 0 {
                EpochOutcome::Lossless
            } else {
                EpochOutcome::Idle
            }
        } else {
            // Flat/single-tier mode (no groups configured)
            // Destination decrease is isolated to destination; global maintains or increases on lossless
            if global_completions >= self.min_completed_probes && global_completions > 0 {
                EpochOutcome::Lossless
            } else {
                EpochOutcome::Idle
            }
        };

        self.global.apply_epoch(global_outcome, self.loss_policy);
    }
}

/// Solves the fixed-point water-filling equation (CONTRACT.md § 3):
///
/// \[ S = \sum_g \min(c_g,\, f_{\max} S) \]
///
/// Returns `(S, [c'_g])` where `c'_g = min(c_g, f_max * S)` is each group's capped contribution.
/// A single group cannot exceed `f_max` of the final capped total `S`.
pub fn solve_water_filling(c_list: &[f64], f_max: f64) -> (f64, Vec<f64>) {
    if c_list.is_empty() || f_max <= 0.0 || f_max > 1.0 {
        return (0.0, Vec::new());
    }

    let mut indexed: Vec<(usize, f64)> = c_list.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let k_total = indexed.len();
    let s_total: f64 = c_list.iter().sum();
    if s_total <= 0.0 {
        return (0.0, vec![0.0; k_total]);
    }

    // Check if uncapped total already satisfies c_g <= f_max * S for all g:
    let all_uncapped = indexed.iter().all(|&(_, c)| c <= f_max * s_total + 1e-9);
    if all_uncapped {
        return (s_total, c_list.to_vec());
    }

    // Search for consistent number of uncapped groups k:
    // S = (sum_{j=0}^{k-1} c_(j)) / (1.0 - (K - k) * f_max)
    let mut best_s = 0.0;
    for k in 0..k_total {
        let uncapped_sum: f64 = indexed[..k].iter().map(|&(_, c)| c).sum();
        let denom = 1.0 - ((k_total - k) as f64) * f_max;
        if denom > 0.0 {
            let s_cand = uncapped_sum / denom;
            let mut consistent = true;
            if k > 0 && indexed[k - 1].1 > f_max * s_cand + 1e-9 {
                consistent = false;
            }
            if k < k_total && indexed[k].1 < f_max * s_cand - 1e-9 {
                consistent = false;
            }
            if consistent {
                best_s = s_cand;
                break;
            }
        }
    }

    let mut capped = vec![0.0; k_total];
    for (orig_idx, &c) in c_list.iter().enumerate() {
        capped[orig_idx] = c.min(f_max * best_s);
    }
    (best_s, capped)
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

    #[test]
    fn test_solve_water_filling_contract() {
        // 4 equal groups: [10, 10, 10, 10]
        let c_equal = [10.0, 10.0, 10.0, 10.0];
        let (s_eq, capped_eq) = solve_water_filling(&c_equal, 0.25);
        assert!((s_eq - 40.0).abs() < 1e-9);
        assert_eq!(capped_eq, vec![10.0, 10.0, 10.0, 10.0]);

        // Dominant group: [100, 1, 1, 1]
        // S = 4.0, dominant group is capped at 0.25 * 4.0 = 1.0
        let c_dom = [100.0, 1.0, 1.0, 1.0];
        let (s_dom, capped_dom) = solve_water_filling(&c_dom, 0.25);
        assert!((s_dom - 4.0).abs() < 1e-9);
        assert!((capped_dom[0] - 1.0).abs() < 1e-9);
        assert_eq!(capped_dom[1..], [1.0, 1.0, 1.0]);

        // Capped sum equals S
        let c_mixed = [5.0, 5.0, 5.0, 5.0, 20.0];
        let (s_mixed, capped_mixed) = solve_water_filling(&c_mixed, 0.25);
        let sum_capped: f64 = capped_mixed.iter().sum();
        assert!((s_mixed - sum_capped).abs() < 1e-9);
        // And no single group exceeds f_max * S
        for &c_prime in &capped_mixed {
            assert!(c_prime <= 0.25 * s_mixed + 1e-9);
        }
    }

    #[test]
    fn test_hierarchical_group_isolation_and_global_evidence() {
        let mut cfg = AdaptiveConfig::adaptive();
        cfg.c_min_dest = 1;
        cfg.c_max_dest = 20;
        cfg.c_min_global = 1;
        cfg.c_max_global = 100;
        let mut ctrl = ConcurrencyController::new(&cfg);

        // Create 4 groups
        let g1 = 1;
        let g2 = 2;
        let g3 = 3;
        let g4 = 4;

        let d1 = 10;
        let d2 = 20;
        let d3 = 30;
        let d4 = 40;

        ctrl.set_dest_group(d1, g1);
        ctrl.set_dest_group(d2, g2);
        ctrl.set_dest_group(d3, g3);
        ctrl.set_dest_group(d4, g4);

        // Initialize windows: ramp all groups to window 10
        for _ in 0..10 {
            for &did in &[d1, d2, d3, d4] {
                ctrl.record_dest_outcome(did, ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                });
            }
            ctrl.apply_epoch();
        }

        let g1_win_before = ctrl.get_group(g1).unwrap().window.window();
        let g2_win_before = ctrl.get_group(g2).unwrap().window.window();
        let global_win_before = ctrl.global.window();
        assert!(g1_win_before > 1);

        // Test Isolation: G1 experiences packet loss, G2..G4 are lossless
        ctrl.record_dest_outcome(d1, ProbeOutcome::InferredLoss);
        for &did in &[d2, d3, d4] {
            ctrl.record_dest_outcome(did, ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(20),
            });
        }
        ctrl.apply_epoch();

        // G1 window decreased
        let g1_win_after = ctrl.get_group(g1).unwrap().window.window();
        assert!(g1_win_after < g1_win_before, "G1 window must decrease on G1 loss");

        // G2 window did NOT decrease (group loss is isolated)
        let g2_win_after = ctrl.get_group(g2).unwrap().window.window();
        assert!(g2_win_after >= g2_win_before, "G2 window must not decrease from G1 loss");

        // Global window did NOT decrease because G_min = 4 groups are required for global evidence
        let global_win_after = ctrl.global.window();
        assert!(
            global_win_after >= global_win_before,
            "Global window must NOT decrease from a single group's localized loss"
        );

        // Now: ALL 4 groups experience loss -> global evidence condition holds!
        ctrl.record_dest_outcome(d1, ProbeOutcome::InferredLoss);
        ctrl.record_dest_outcome(d2, ProbeOutcome::InferredLoss);
        ctrl.record_dest_outcome(d3, ProbeOutcome::InferredLoss);
        ctrl.record_dest_outcome(d4, ProbeOutcome::InferredLoss);
        ctrl.apply_epoch();

        // Now global window must decrease!
        let global_win_final = ctrl.global.window();
        assert!(
            global_win_final < global_win_after,
            "Global window must decrease when all 4 groups exhibit loss satisfying global evidence"
        );
    }

    #[test]
    fn test_atomic_hierarchical_pacing() {
        let cfg = AdaptiveConfig::adaptive();
        let mut ctrl = ConcurrencyController::new(&cfg);

        let d1 = 100;
        let g1 = 10;
        ctrl.set_dest_group(d1, g1);

        // Set global pacer = 10ms interval, group pacer = 20ms interval, dest pacer = 50ms interval
        ctrl.set_pacer(None, None, Duration::from_millis(10));
        ctrl.set_pacer(None, Some(g1), Duration::from_millis(20));
        ctrl.set_pacer(Some(d1), None, Duration::from_millis(50));

        let t0 = Duration::ZERO;
        // First transmit at t0 should succeed across all pacers atomically
        let res1 = ctrl.try_acquire_pacing(d1, t0);
        assert!(res1.is_ok(), "First acquisition should succeed");

        // Immediate retry at t0 should fail because dest requires 50ms, group requires 20ms, global requires 10ms
        let res2 = ctrl.try_acquire_pacing(d1, t0);
        assert_eq!(
            res2,
            Err(Duration::from_millis(50)),
            "Should require waiting for slowest pacer (50ms)"
        );

        // Attempt at 25ms: global (ready) and group (ready), but dest (needs 50ms) -> atomic failure!
        let res3 = ctrl.try_acquire_pacing(d1, Duration::from_millis(25));
        assert_eq!(res3, Err(Duration::from_millis(50)));

        // Because it failed atomically, global and group next_tx must NOT have been consumed at 25ms!
        // Another destination d2 in group g2 (no dest pacer, no group pacer, only global pacer)
        let d2 = 200;
        // Global was set at t0 + 10ms = 10ms. At 25ms, global IS ready!
        let res_d2 = ctrl.try_acquire_pacing(d2, Duration::from_millis(25));
        assert!(res_d2.is_ok(), "d2 should succeed because global pacer was not corrupted");

        // Advance to 50ms: d1 now succeeds across all scopes!
        let res4 = ctrl.try_acquire_pacing(d1, Duration::from_millis(50));
        assert!(res4.is_ok(), "d1 should succeed at 50ms");
    }
}
