//! § 1.4 — Deterministic bounded rolling-window P95 tail tracker.
//!
//! Fixed-capacity ring buffer of the last W eligible RTT samples with
//! exact order-statistic P95 computation via a sorted auxiliary buffer.

use core::time::Duration;

/// Rolling-window P95 tail tracker.
///
/// Maintains a ring buffer of the last `W` eligible RTT samples and computes
/// `T_95 = R_(⌈0.95 · n_W⌉)` on demand.
#[derive(Clone, Debug)]
pub struct RollingP95 {
    /// Ring buffer of raw samples in insertion order.
    buf: Vec<Duration>,
    /// Maximum window size W.
    capacity: usize,
    /// Write cursor (next insertion index, wrapping).
    cursor: usize,
    /// Number of valid samples currently stored (≤ capacity).
    count: usize,
}

impl RollingP95 {
    /// Create a new tracker with window size `w` (clamped to [1, 1024]).
    pub fn new(w: usize) -> Self {
        let capacity = w.clamp(1, 1024);
        Self {
            buf: vec![Duration::ZERO; capacity],
            capacity,
            cursor: 0,
            count: 0,
        }
    }

    /// Push an eligible RTT sample into the ring buffer.
    pub fn push(&mut self, rtt: Duration) {
        self.buf[self.cursor] = rtt;
        self.cursor = (self.cursor + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Number of valid samples in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Compute T_95 = R_(⌈0.95 · n_W⌉).
    ///
    /// Returns `None` if no samples are present.
    pub fn p95(&self) -> Option<Duration> {
        if self.count == 0 {
            return None;
        }

        // Collect active samples and sort.
        let mut sorted = self.active_samples();
        sorted.sort_unstable();

        // Index: ⌈0.95 · n_W⌉ − 1 (0-based)
        let rank = ceil_mul_95(self.count);
        let idx = rank.saturating_sub(1).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Return a sorted copy of the active window for external use.
    pub fn active_samples(&self) -> Vec<Duration> {
        if self.count < self.capacity {
            // Buffer not yet full: samples are at indices [0, count).
            self.buf[..self.count].to_vec()
        } else {
            // Buffer full: all `capacity` entries are valid.
            self.buf.clone()
        }
    }
}

/// Compute ⌈0.95 · n⌉ without floating point.
///
/// ⌈0.95·n⌉ = ⌈(19·n)/20⌉ = (19·n + 19) / 20  (integer division).
#[inline]
fn ceil_mul_95(n: usize) -> usize {
    (19 * n).div_ceil(20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        let t = RollingP95::new(64);
        assert!(t.p95().is_none());
    }

    #[test]
    fn single_sample() {
        let mut t = RollingP95::new(64);
        t.push(Duration::from_millis(100));
        // ⌈0.95 · 1⌉ = 1 → index 0
        assert_eq!(t.p95(), Some(Duration::from_millis(100)));
    }

    #[test]
    fn twenty_sequential_samples() {
        let mut t = RollingP95::new(64);
        for i in 1..=20 {
            t.push(Duration::from_millis(i));
        }
        // ⌈0.95 · 20⌉ = ⌈19⌉ = 19 → 19th order stat = 19 ms
        assert_eq!(t.p95(), Some(Duration::from_millis(19)));
    }

    #[test]
    fn wrapping_evicts_old() {
        let mut t = RollingP95::new(4);
        // Fill: [10, 20, 30, 40]
        for &v in &[10, 20, 30, 40] {
            t.push(Duration::from_millis(v));
        }
        assert_eq!(t.len(), 4);

        // Push 5 → evicts 10: [5, 20, 30, 40]
        t.push(Duration::from_millis(5));
        assert_eq!(t.len(), 4);

        // Sorted: [5, 20, 30, 40], ⌈0.95 · 4⌉ = ⌈3.8⌉ = 4 → index 3 → 40
        assert_eq!(t.p95(), Some(Duration::from_millis(40)));
    }

    #[test]
    fn ceil_mul_95_spot_checks() {
        assert_eq!(ceil_mul_95(1), 1);
        assert_eq!(ceil_mul_95(20), 19);
        assert_eq!(ceil_mul_95(64), 61); // ⌈0.95·64⌉ = ⌈60.8⌉ = 61
        assert_eq!(ceil_mul_95(100), 95);
    }
}
