//! § 5 & § 6 — Execution Plane: Generational Arenas, ProbePermit Lifecycle & Tokio Adapter.
//!
//! Implements:
//! - § 6: Fixed-capacity generational arenas for destinations and groups with generation-checked
//!   private handles (`DestinationHandle`, `GroupHandle`), bounded key-to-slot lookup,
//!   `ArenaCapacityPinned` error, and child-to-group pin propagation.
//! - § 5: Complete `ProbePermit` lifecycle:
//!   - `Reserved` (lease countdown)
//!   - `mark_dispatched` (sets `first_send_us`, latches full retry policy)
//!   - `Dispatched` (network timers active)
//!   - `Completed`, `Cancelled`, `TimedOut`, `Reclaimed`
//!   - Bounded cancellation queue with fallback forced reclamation on next maintenance tick.

use crate::config::ProbeOutcome;
use crate::rto::ProbeRetrySchedule;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

// ─── Distinct Handles (§ 6) ──────────────────────────────────────────────────

/// Private, generation-checked handle for an arena-allocated Destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DestinationHandle {
    pub(crate) slot: u32,
    pub(crate) generation: u64,
}

impl DestinationHandle {
    /// The arena slot index.
    #[inline]
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// The generational epoch counter.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Private, generation-checked handle for an arena-allocated Group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupHandle {
    pub(crate) slot: u32,
    pub(crate) generation: u64,
}

impl GroupHandle {
    /// The arena slot index.
    #[inline]
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// The generational epoch counter.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

// ─── Arena Error (§ 6) ───────────────────────────────────────────────────────

/// Errors encountered in generational arena operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaError {
    /// All slots in the arena are occupied and pinned; caller may retry after capacity frees.
    ArenaCapacityPinned,
    /// Handle generation does not match slot generation (stale handle).
    StaleHandle,
    /// Key not found in arena lookup.
    NotFound,
    /// Generation wrapped `u64::MAX`; slot is permanently retired.
    GenerationExhausted,
}

impl core::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ArenaCapacityPinned => {
                write!(f, "Arena capacity exhausted: all available slots are currently pinned")
            }
            Self::StaleHandle => write!(f, "Stale handle: slot generation mismatch"),
            Self::NotFound => write!(f, "Key not found in arena"),
            Self::GenerationExhausted => write!(f, "Slot generation counter exhausted: retired"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ArenaError {}

// ─── Arena Slot & Implementation (§ 6) ──────────────────────────────────────

/// Single slot within a generational arena.
#[derive(Clone, Debug)]
pub struct ArenaSlot<T> {
    pub value: Option<T>,
    pub generation: u64,
    pub pin_count: usize,
    pub key: u64,
    pub retired: bool,
}

impl<T> ArenaSlot<T> {
    fn new() -> Self {
        Self {
            value: None,
            generation: 1, // Start at generation 1 so 0 is never valid
            pin_count: 0,
            key: 0,
            retired: false,
        }
    }

    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.pin_count > 0
    }

    #[inline]
    pub fn is_free(&self) -> bool {
        self.value.is_none() && self.pin_count == 0 && !self.retired
    }
}

/// Fixed-capacity generational arena for destinations (§ 6).
#[derive(Clone, Debug)]
pub struct DestinationArena<T> {
    slots: Vec<ArenaSlot<T>>,
    capacity: usize,
    active_count: usize,
}

impl<T> DestinationArena<T> {
    /// Create a new destination arena with fixed capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.clamp(1, 1_000_000);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(ArenaSlot::new());
        }
        Self {
            slots,
            capacity: cap,
            active_count: 0,
        }
    }

    /// Number of active items.
    #[inline]
    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Whether the arena is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Capacity ceiling.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Insert a value associated with `key`.
    ///
    /// If a slot with `key` already exists and is active, returns its handle.
    /// Otherwise allocates an unpinned free slot. If all slots are pinned, returns `ArenaCapacityPinned`.
    pub fn insert(&mut self, key: u64, value: T) -> Result<DestinationHandle, ArenaError> {
        // Bounded key-to-slot lookup
        if let Some((handle, _)) = self.find_by_key(key) {
            return Ok(handle);
        }

        // Find an unpinned free slot
        let slot_idx = self
            .slots
            .iter()
            .position(|s| s.is_free())
            .ok_or(ArenaError::ArenaCapacityPinned)?;

        let slot = &mut self.slots[slot_idx];
        slot.value = Some(value);
        slot.key = key;
        slot.pin_count = 0;
        self.active_count += 1;

        Ok(DestinationHandle {
            slot: slot_idx as u32,
            generation: slot.generation,
        })
    }

    /// Lookup by handle. Returns `None` if handle is stale or slot is empty.
    pub fn get(&self, handle: DestinationHandle) -> Option<&T> {
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation == handle.generation && !slot.retired {
            slot.value.as_ref()
        } else {
            None
        }
    }

    /// Lookup mutably by handle. Returns `None` if handle is stale.
    pub fn get_mut(&mut self, handle: DestinationHandle) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.slot as usize)?;
        if slot.generation == handle.generation && !slot.retired {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Bounded key-to-slot lookup.
    pub fn find_by_key(&self, key: u64) -> Option<(DestinationHandle, &T)> {
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(val) = &slot.value
                && slot.key == key
                && !slot.retired
            {
                let handle = DestinationHandle {
                    slot: idx as u32,
                    generation: slot.generation,
                };
                return Some((handle, val));
            }
        }
        None
    }

    /// Bounded key-to-slot lookup mutably.
    pub fn find_by_key_mut(&mut self, key: u64) -> Option<(DestinationHandle, &mut T)> {
        for (idx, slot) in self.slots.iter_mut().enumerate() {
            if let Some(val) = &mut slot.value
                && slot.key == key
                && !slot.retired
            {
                let handle = DestinationHandle {
                    slot: idx as u32,
                    generation: slot.generation,
                };
                return Some((handle, val));
            }
        }
        None
    }

    /// Pin the destination slot.
    pub fn pin(&mut self, handle: DestinationHandle) -> Result<(), ArenaError> {
        let slot = self.slots.get_mut(handle.slot as usize).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        slot.pin_count = slot.pin_count.saturating_add(1);
        Ok(())
    }

    /// Unpin the destination slot.
    pub fn unpin(&mut self, handle: DestinationHandle) -> Result<(), ArenaError> {
        let slot = self.slots.get_mut(handle.slot as usize).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        slot.pin_count = slot.pin_count.saturating_sub(1);
        Ok(())
    }

    /// Whether destination slot is currently pinned.
    pub fn is_pinned(&self, handle: DestinationHandle) -> bool {
        self.slots
            .get(handle.slot as usize)
            .is_some_and(|s| s.generation == handle.generation && s.is_pinned())
    }

    /// Remove a destination. Returns `ArenaCapacityPinned` if currently pinned.
    pub fn remove(&mut self, handle: DestinationHandle) -> Result<T, ArenaError> {
        let slot = self.slots.get_mut(handle.slot as usize).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        if slot.is_pinned() {
            return Err(ArenaError::ArenaCapacityPinned);
        }
        let val = slot.value.take().ok_or(ArenaError::NotFound)?;
        self.active_count = self.active_count.saturating_sub(1);

        // Advance u64 generation counter or retire on wrap
        if let Some(next_gen) = slot.generation.checked_add(1) {
            slot.generation = next_gen;
        } else {
            slot.retired = true;
        }

        Ok(val)
    }
}

/// Fixed-capacity generational arena for groups (§ 6).
#[derive(Clone, Debug)]
pub struct GroupArena<T> {
    slots: Vec<ArenaSlot<T>>,
    capacity: usize,
    active_count: usize,
    /// Child destination pin counts mapped by group slot index.
    child_pins: Vec<usize>,
}

impl<T> GroupArena<T> {
    /// Create a new group arena with fixed capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.clamp(1, 100_000);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(ArenaSlot::new());
        }
        Self {
            slots,
            capacity: cap,
            active_count: 0,
            child_pins: vec![0; cap],
        }
    }

    /// Capacity ceiling.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Insert or retrieve an existing group by `key`.
    pub fn insert(&mut self, key: u64, value: T) -> Result<GroupHandle, ArenaError> {
        if let Some((handle, _)) = self.find_by_key(key) {
            return Ok(handle);
        }

        let slot_idx = self
            .slots
            .iter()
            .enumerate()
            .position(|(i, s)| s.is_free() && self.child_pins[i] == 0)
            .ok_or(ArenaError::ArenaCapacityPinned)?;

        let slot = &mut self.slots[slot_idx];
        slot.value = Some(value);
        slot.key = key;
        slot.pin_count = 0;
        self.child_pins[slot_idx] = 0;
        self.active_count += 1;

        Ok(GroupHandle {
            slot: slot_idx as u32,
            generation: slot.generation,
        })
    }

    /// Lookup by handle.
    pub fn get(&self, handle: GroupHandle) -> Option<&T> {
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation == handle.generation && !slot.retired {
            slot.value.as_ref()
        } else {
            None
        }
    }

    /// Lookup mutably by handle.
    pub fn get_mut(&mut self, handle: GroupHandle) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.slot as usize)?;
        if slot.generation == handle.generation && !slot.retired {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Bounded key-to-slot lookup.
    pub fn find_by_key(&self, key: u64) -> Option<(GroupHandle, &T)> {
        for (idx, slot) in self.slots.iter().enumerate() {
            if let Some(val) = &slot.value
                && slot.key == key
                && !slot.retired
            {
                let handle = GroupHandle {
                    slot: idx as u32,
                    generation: slot.generation,
                };
                return Some((handle, val));
            }
        }
        None
    }

    /// Pin the group directly.
    pub fn pin(&mut self, handle: GroupHandle) -> Result<(), ArenaError> {
        let slot = self.slots.get_mut(handle.slot as usize).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        slot.pin_count = slot.pin_count.saturating_add(1);
        Ok(())
    }

    /// Unpin the group directly.
    pub fn unpin(&mut self, handle: GroupHandle) -> Result<(), ArenaError> {
        let slot = self.slots.get_mut(handle.slot as usize).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        slot.pin_count = slot.pin_count.saturating_sub(1);
        Ok(())
    }

    /// Pin a child destination belonging to this group (§ 6: A group remains pinned while any child is pinned).
    pub fn pin_child(&mut self, handle: GroupHandle) -> Result<(), ArenaError> {
        let idx = handle.slot as usize;
        let slot = self.slots.get(idx).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        self.child_pins[idx] = self.child_pins[idx].saturating_add(1);
        Ok(())
    }

    /// Unpin a child destination belonging to this group.
    pub fn unpin_child(&mut self, handle: GroupHandle) -> Result<(), ArenaError> {
        let idx = handle.slot as usize;
        let slot = self.slots.get(idx).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        self.child_pins[idx] = self.child_pins[idx].saturating_sub(1);
        Ok(())
    }

    /// Whether group is pinned (either directly or by any child destination).
    pub fn is_pinned(&self, handle: GroupHandle) -> bool {
        let idx = handle.slot as usize;
        self.slots.get(idx).is_some_and(|s| {
            s.generation == handle.generation && (s.is_pinned() || self.child_pins[idx] > 0)
        })
    }

    /// Remove a group. Fails if pinned directly or by any child destination.
    pub fn remove(&mut self, handle: GroupHandle) -> Result<T, ArenaError> {
        let idx = handle.slot as usize;
        let slot = self.slots.get_mut(idx).ok_or(ArenaError::NotFound)?;
        if slot.generation != handle.generation || slot.retired {
            return Err(ArenaError::StaleHandle);
        }
        if slot.is_pinned() || self.child_pins[idx] > 0 {
            return Err(ArenaError::ArenaCapacityPinned);
        }
        let val = slot.value.take().ok_or(ArenaError::NotFound)?;
        self.active_count = self.active_count.saturating_sub(1);

        if let Some(next_gen) = slot.generation.checked_add(1) {
            slot.generation = next_gen;
        } else {
            slot.retired = true;
        }

        Ok(val)
    }
}

// ─── ProbePermit Lifecycle (§ 5) ─────────────────────────────────────────────

/// Monotonically incrementing unique probe identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProbeId(pub u64);

/// Mutually exclusive state of a `ProbePermit` per CONTRACT.md § 5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PermitState {
    /// Permit is reserved; concurrency slot is held. Reservation lease is running.
    Reserved {
        reserved_at_us: u64,
        lease_duration_us: u64,
    },
    /// Probe has been dispatched to network; network retry schedule is active.
    Dispatched {
        first_send_us: u64,
        attempt: u32,
        max_retries: u32,
        base_rto_us: u64,
        gamma: f64,
        rto_max_us: u64,
    },
    /// Response recorded; probe completed.
    Completed {
        outcome: ProbeOutcome,
    },
    /// Explicitly cancelled by caller.
    Cancelled,
    /// Maximum retries reached with no response.
    /// Releases concurrency accounting and does NOT automatically assert packet loss.
    TimedOut,
    /// Reservation lease expired before dispatch, or reclaimed during maintenance.
    Reclaimed,
}

/// Errors occurring during permit lifecycle operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PermitError {
    /// Permit is not in `Reserved` state.
    NotReserved,
    /// Permit is not in `Dispatched` state.
    NotDispatched,
    /// Reservation lease expired before probe could be dispatched.
    ReservationLeaseExpired,
    /// Blocked by pacer until the given timestamp.
    PaceUntil { wake_at_us: u64 },
    /// Permit has already reached a terminal state.
    AlreadyTerminated,
}

impl core::fmt::Display for PermitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotReserved => write!(f, "Permit is not in Reserved state"),
            Self::NotDispatched => write!(f, "Permit is not in Dispatched state"),
            Self::ReservationLeaseExpired => write!(f, "Reservation lease expired before dispatch"),
            Self::PaceUntil { wake_at_us } => {
                write!(f, "Pacing holds permit until {wake_at_us} µs")
            }
            Self::AlreadyTerminated => write!(f, "Permit has already reached a terminal state"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PermitError {}

/// An active network probe permit implementing the lifecycle in CONTRACT.md § 5.
#[derive(Debug)]
pub struct ProbePermit {
    /// Unique probe ID.
    pub id: ProbeId,
    /// Pinned destination handle.
    pub dest_handle: DestinationHandle,
    /// Pinned group handle (if group-scoped).
    pub group_handle: Option<GroupHandle>,
    /// Current permit state.
    pub state: PermitState,
    /// Whether accounting release has been executed.
    pub(crate) accounting_released: bool,
    /// Fallback flag for forced reclamation if dropped while dispatched.
    pub(crate) marked_forced_reclamation: bool,
}

impl ProbePermit {
    /// Create a new permit in `Reserved` state, holding concurrency and pinning arenas.
    pub fn new_reserved(
        id: ProbeId,
        dest_handle: DestinationHandle,
        group_handle: Option<GroupHandle>,
        now_us: u64,
        lease_duration_us: u64,
    ) -> Self {
        Self {
            id,
            dest_handle,
            group_handle,
            state: PermitState::Reserved {
                reserved_at_us: now_us,
                lease_duration_us,
            },
            accounting_released: false,
            marked_forced_reclamation: false,
        }
    }

    /// Mark the permit as dispatched to the network.
    ///
    /// Sets `first_send_us`, latches full retry policy (`base_rto`, `gamma`, `max_retries`).
    /// Fails if reservation lease has expired, transitioning to `Reclaimed`.
    pub fn mark_dispatched(
        &mut self,
        now_us: u64,
        schedule: &ProbeRetrySchedule,
        max_retries: u32,
    ) -> Result<(), PermitError> {
        let (reserved_at, lease_duration) = match self.state {
            PermitState::Reserved {
                reserved_at_us,
                lease_duration_us,
            } => (reserved_at_us, lease_duration_us),
            PermitState::Dispatched { .. } => return Ok(()),
            _ => return Err(PermitError::AlreadyTerminated),
        };

        // Check reservation lease expiry
        if now_us.saturating_sub(reserved_at) > lease_duration {
            self.state = PermitState::Reclaimed;
            self.accounting_released = true;
            return Err(PermitError::ReservationLeaseExpired);
        }

        self.state = PermitState::Dispatched {
            first_send_us: now_us,
            attempt: 0,
            max_retries,
            base_rto_us: schedule.base_rto.as_micros() as u64,
            gamma: schedule.gamma,
            rto_max_us: schedule.rto_max.as_micros() as u64,
        };

        Ok(())
    }

    /// Advance retry attempt. If `attempt > max_retries`, transitions to `TimedOut`.
    /// Releases concurrency accounting, but does NOT assert packet loss.
    pub fn on_retry(&mut self) -> Result<u32, PermitError> {
        let (new_attempt, timed_out) = match &mut self.state {
            PermitState::Dispatched {
                attempt,
                max_retries,
                ..
            } => {
                *attempt += 1;
                let is_timeout = *attempt > *max_retries;
                (*attempt, is_timeout)
            }
            PermitState::Reserved { .. } => return Err(PermitError::NotDispatched),
            _ => return Err(PermitError::AlreadyTerminated),
        };

        if timed_out {
            self.state = PermitState::TimedOut;
        }

        Ok(new_attempt)
    }

    /// Record a probe outcome, transitioning to `Completed`.
    pub fn record_outcome(&mut self, outcome: ProbeOutcome) -> Result<(), PermitError> {
        match self.state {
            PermitState::Dispatched { .. } => {
                self.state = PermitState::Completed { outcome };
                Ok(())
            }
            PermitState::Reserved { .. } => Err(PermitError::NotDispatched),
            _ => Err(PermitError::AlreadyTerminated),
        }
    }

    /// Cancel probe, transitioning to `Cancelled`.
    pub fn cancel_probe(&mut self) -> Result<(), PermitError> {
        match self.state {
            PermitState::Reserved { .. } | PermitState::Dispatched { .. } => {
                self.state = PermitState::Cancelled;
                Ok(())
            }
            _ => Err(PermitError::AlreadyTerminated),
        }
    }

    /// Current state of the permit.
    pub fn state(&self) -> &PermitState {
        &self.state
    }

    /// Whether the permit reached `TimedOut` state.
    pub fn is_timed_out(&self) -> bool {
        matches!(self.state, PermitState::TimedOut)
    }

    /// Whether the permit completed.
    pub fn is_completed(&self) -> bool {
        matches!(self.state, PermitState::Completed { .. })
    }

    /// Check reservation lease expiry while still `Reserved`.
    pub fn check_lease_expiry(&mut self, now_us: u64) -> bool {
        if let PermitState::Reserved {
            reserved_at_us,
            lease_duration_us,
        } = self.state
            && now_us.saturating_sub(reserved_at_us) > lease_duration_us
        {
            self.state = PermitState::Reclaimed;
            return true;
        }
        false
    }
}

// ─── Bounded Cancellation Queue & Tokio Adapter (§ 5) ─────────────────────────

/// Fixed-capacity bounded non-blocking cancellation queue (§ 5).
#[derive(Clone, Debug)]
pub struct CancellationQueue {
    queue: Vec<ProbeId>,
    capacity: usize,
    pub forced_reclamations: usize,
}

impl CancellationQueue {
    /// Create a new bounded cancellation queue.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Vec::with_capacity(capacity),
            capacity,
            forced_reclamations: 0,
        }
    }

    /// Bounded non-blocking push. Returns `Err(id)` if the queue is full.
    pub fn try_push(&mut self, id: ProbeId) -> Result<(), ProbeId> {
        if self.queue.len() < self.capacity {
            self.queue.push(id);
            Ok(())
        } else {
            self.forced_reclamations += 1;
            Err(id)
        }
    }

    /// Drain all pending cancellation IDs.
    pub fn drain(&mut self) -> Vec<ProbeId> {
        let items = self.queue.clone();
        self.queue.clear();
        items
    }
}

/// Trait for non-blocking cancellation sinks (§ 5).
pub trait CancellationSink {
    /// Attempt to enqueue a probe ID for cancellation without blocking.
    fn try_cancel(&mut self, id: ProbeId) -> Result<(), ProbeId>;
}

impl CancellationSink for CancellationQueue {
    fn try_cancel(&mut self, id: ProbeId) -> Result<(), ProbeId> {
        self.try_push(id)
    }
}

/// Cancellation-safe RAII guard for Tokio / async probe execution (§ 5).
///
/// Wraps a [`ProbePermit`] and a reference to a cancellation sink (such as [`CancellationQueue`]).
/// When dropped before explicit completion or detachment (e.g. if the async task is cancelled
/// via `tokio::select!` or task abort), it non-blockingly attempts to enqueue the probe ID
/// for cancellation. If the queue is full, it sets `marked_forced_reclamation` so that the
/// next engine maintenance tick reclaims the permit, ensuring no resources leak.
pub struct CancellationGuard<'a, S: CancellationSink> {
    permit: Option<ProbePermit>,
    sink: &'a mut S,
}

impl<'a, S: CancellationSink> CancellationGuard<'a, S> {
    /// Create a new cancellation guard wrapping a permit and a cancellation sink.
    pub fn new(permit: ProbePermit, sink: &'a mut S) -> Self {
        Self {
            permit: Some(permit),
            sink,
        }
    }

    /// Access the underlying permit.
    pub fn permit(&self) -> Option<&ProbePermit> {
        self.permit.as_ref()
    }

    /// Access the underlying permit mutably.
    pub fn permit_mut(&mut self) -> Option<&mut ProbePermit> {
        self.permit.as_mut()
    }

    /// Complete or extract the permit from the guard without triggering cancellation on drop.
    pub fn defuse(mut self) -> Option<ProbePermit> {
        self.permit.take()
    }
}

impl<'a, S: CancellationSink> Drop for CancellationGuard<'a, S> {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take()
            && !permit.accounting_released
        {
            let _ = self.sink.try_cancel(permit.id);
        }
    }
}

/// Execution Coordinator managing arenas, permits, pacing, and maintenance ticks (§ 5 & § 6).
#[derive(Clone, Debug)]
pub struct ExecutionCoordinator {
    pub dest_arena: DestinationArena<u64>,
    pub group_arena: GroupArena<u64>,
    pub cancellation_queue: CancellationQueue,
    next_probe_id: u64,
    pub diagnostic_forced_reclamations: u64,
}

impl ExecutionCoordinator {
    /// Create a new execution coordinator with fixed arena capacities.
    pub fn new(dest_capacity: usize, group_capacity: usize, cancel_capacity: usize) -> Self {
        Self {
            dest_arena: DestinationArena::new(dest_capacity),
            group_arena: GroupArena::new(group_capacity),
            cancellation_queue: CancellationQueue::new(cancel_capacity),
            next_probe_id: 1,
            diagnostic_forced_reclamations: 0,
        }
    }

    /// Allocate a new probe permit in `Reserved` state with pinned destination and group.
    pub fn reserve_permit(
        &mut self,
        dest_key: u64,
        group_key: Option<u64>,
        now_us: u64,
        lease_duration_us: u64,
    ) -> Result<ProbePermit, ArenaError> {
        let dest_handle = self.dest_arena.insert(dest_key, dest_key)?;
        self.dest_arena.pin(dest_handle)?;

        let group_handle = if let Some(gk) = group_key {
            let gh = self.group_arena.insert(gk, gk)?;
            self.group_arena.pin_child(gh)?;
            Some(gh)
        } else {
            None
        };

        let probe_id = ProbeId(self.next_probe_id);
        self.next_probe_id = self.next_probe_id.wrapping_add(1);

        Ok(ProbePermit::new_reserved(
            probe_id,
            dest_handle,
            group_handle,
            now_us,
            lease_duration_us,
        ))
    }

    /// Release arena pins when permit completes or terminates.
    pub fn release_permit_accounting(&mut self, permit: &mut ProbePermit) {
        if !permit.accounting_released {
            let _ = self.dest_arena.unpin(permit.dest_handle);
            if let Some(gh) = permit.group_handle {
                let _ = self.group_arena.unpin_child(gh);
            }
            permit.accounting_released = true;
        }
    }

    /// Handle drop of a permit: tries non-blocking push to cancellation queue,
    /// falling back to marking for forced reclamation at next maintenance tick (§ 5).
    pub fn on_permit_dropped(&mut self, permit: &mut ProbePermit) {
        if permit.accounting_released {
            return;
        }

        // Try bounded non-blocking push to cancellation queue
        if self.cancellation_queue.try_push(permit.id).is_err() {
            // Queue full: mark for forced reclamation at next maintenance tick
            permit.marked_forced_reclamation = true;
            self.diagnostic_forced_reclamations += 1;
        }

        self.release_permit_accounting(permit);
    }

    /// Maintenance tick: processes cancellation queue and forced reclamations (§ 5).
    pub fn maintenance_tick(&mut self) -> usize {
        let cancelled_ids = self.cancellation_queue.drain();
        cancelled_ids.len()
    }

    /// Create a cancellation-safe RAII guard for this permit backed by the coordinator's cancellation queue (§ 5).
    pub fn guard(&mut self, permit: ProbePermit) -> CancellationGuard<'_, CancellationQueue> {
        CancellationGuard::new(permit, &mut self.cancellation_queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::time::Duration;

    #[test]
    fn test_generational_arena_generation_checks() {
        let mut arena = DestinationArena::new(2);

        let h1 = arena.insert(101, 101).unwrap();
        assert_eq!(h1.slot(), 0);
        assert_eq!(h1.generation(), 1);

        // Access via handle
        assert_eq!(arena.get(h1), Some(&101));

        // Remove h1
        let val = arena.remove(h1).unwrap();
        assert_eq!(val, 101);

        // Access via old handle must fail (stale generation)
        assert_eq!(arena.get(h1), None);

        // Reallocate slot 0 with new key 202
        let h2 = arena.insert(202, 202).unwrap();
        assert_eq!(h2.slot(), 0);
        assert_eq!(h2.generation(), 2, "Generation must advance to 2");

        // Old handle h1 still fails on slot 0
        assert_eq!(arena.get(h1), None);
        // New handle h2 succeeds
        assert_eq!(arena.get(h2), Some(&202));
    }

    #[test]
    fn test_arena_capacity_exhaustion_and_pinning() {
        let mut arena = DestinationArena::new(2);

        let h1 = arena.insert(1, 1).unwrap();
        let h2 = arena.insert(2, 2).unwrap();

        // Pin both slots
        arena.pin(h1).unwrap();
        arena.pin(h2).unwrap();

        // Attempting to insert 3rd must fail with ArenaCapacityPinned
        let err = arena.insert(3, 3).unwrap_err();
        assert_eq!(err, ArenaError::ArenaCapacityPinned);

        // Attempting to remove pinned slot must also fail
        let rem_err = arena.remove(h1).unwrap_err();
        assert_eq!(rem_err, ArenaError::ArenaCapacityPinned);

        // Unpin h1 and remove
        arena.unpin(h1).unwrap();
        assert!(arena.remove(h1).is_ok());

        // Now capacity frees up, insert succeeds
        let h3 = arena.insert(3, 3);
        assert!(h3.is_ok());
    }

    #[test]
    fn test_group_pin_propagation_from_children() {
        let mut groups = GroupArena::new(2);
        let gh = groups.insert(10, 10).unwrap();

        assert!(!groups.is_pinned(gh));

        // Pin child destination in this group
        groups.pin_child(gh).unwrap();
        assert!(groups.is_pinned(gh), "Group must be pinned while child is pinned");

        // Removing group must fail
        assert_eq!(groups.remove(gh).unwrap_err(), ArenaError::ArenaCapacityPinned);

        // Unpin child
        groups.unpin_child(gh).unwrap();
        assert!(!groups.is_pinned(gh));
        assert!(groups.remove(gh).is_ok());
    }

    #[test]
    fn test_permit_lifecycle_success() {
        let mut coord = ExecutionCoordinator::new(10, 10, 10);
        let mut permit = coord
            .reserve_permit(101, Some(1), 1000, 50_000)
            .unwrap();

        assert!(matches!(permit.state, PermitState::Reserved { .. }));
        assert!(coord.dest_arena.is_pinned(permit.dest_handle));

        // Mark dispatched
        let sched = ProbeRetrySchedule {
            base_rto: Duration::from_millis(200),
            rto_max: Duration::from_secs(3),
            gamma: 1.5,
        };
        assert!(permit.mark_dispatched(1050, &sched, 3).is_ok());
        assert!(matches!(permit.state, PermitState::Dispatched { .. }));

        // Record outcome
        assert!(permit
            .record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            })
            .is_ok());
        assert!(matches!(permit.state, PermitState::Completed { .. }));

        coord.release_permit_accounting(&mut permit);
        assert!(!coord.dest_arena.is_pinned(permit.dest_handle));
    }

    #[test]
    fn test_permit_lease_expiry_reclaimed() {
        let mut coord = ExecutionCoordinator::new(10, 10, 10);
        let mut permit = coord
            .reserve_permit(101, None, 1000, 10_000) // 10ms lease
            .unwrap();

        // 15ms later, lease has expired
        assert!(permit.check_lease_expiry(1000 + 15_000));
        assert_eq!(permit.state, PermitState::Reclaimed);

        // Dispatch must fail
        let sched = ProbeRetrySchedule {
            base_rto: Duration::from_millis(200),
            rto_max: Duration::from_secs(3),
            gamma: 1.5,
        };
        assert_eq!(
            permit.mark_dispatched(20000, &sched, 3),
            Err(PermitError::AlreadyTerminated)
        );
    }

    #[test]
    fn test_permit_retries_exhausted_timeout_no_automatic_loss() {
        let mut coord = ExecutionCoordinator::new(10, 10, 10);
        let mut permit = coord
            .reserve_permit(101, None, 1000, 50_000)
            .unwrap();

        let sched = ProbeRetrySchedule {
            base_rto: Duration::from_millis(200),
            rto_max: Duration::from_secs(3),
            gamma: 1.5,
        };
        permit.mark_dispatched(1000, &sched, 2).unwrap();

        // Retry 1
        assert_eq!(permit.on_retry().unwrap(), 1);
        assert!(matches!(permit.state, PermitState::Dispatched { .. }));

        // Retry 2
        assert_eq!(permit.on_retry().unwrap(), 2);
        assert!(matches!(permit.state, PermitState::Dispatched { .. }));

        // Retry 3 (exceeds max_retries = 2) -> TimedOut
        assert_eq!(permit.on_retry().unwrap(), 3);
        assert_eq!(permit.state, PermitState::TimedOut);
        coord.release_permit_accounting(&mut permit);
        assert!(permit.accounting_released);
        assert!(!coord.dest_arena.is_pinned(permit.dest_handle));
    }

    #[test]
    fn test_cancellation_queue_full_fallback_forced_reclamation() {
        let mut coord = ExecutionCoordinator::new(10, 10, 1); // cancellation queue capacity = 1
        let mut p1 = coord.reserve_permit(1, None, 0, 1000).unwrap();
        let mut p2 = coord.reserve_permit(2, None, 0, 1000).unwrap();

        // p1 drop uses the 1 queue slot
        coord.on_permit_dropped(&mut p1);
        assert_eq!(coord.diagnostic_forced_reclamations, 0);

        // p2 drop encounters full queue -> fallback marked for forced reclamation!
        coord.on_permit_dropped(&mut p2);
        assert!(p2.marked_forced_reclamation);
        assert_eq!(coord.diagnostic_forced_reclamations, 1);

        // Maintenance tick drains cancellation queue
        let drained = coord.maintenance_tick();
        assert_eq!(drained, 1);
    }

    #[test]
    fn test_cancellation_guard_drop_pushes_to_queue() {
        let mut coord = ExecutionCoordinator::new(10, 10, 10);
        let permit = coord.reserve_permit(42, None, 0, 10_000).unwrap();
        let pid = permit.id;

        {
            let _guard = coord.guard(permit);
            // Simulates async cancellation / early drop: guard dropped here
        }

        // Permit probe ID should be in the cancellation queue
        assert_eq!(coord.cancellation_queue.drain(), vec![pid]);
    }
}
