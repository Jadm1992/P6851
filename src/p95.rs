//! § 1.4 & § 8 — 64-Bucket Geometric Histogram, Block-Sampled P95 Tail Tracker & Confidence Gate.
//!
//! Implements:
//! - § 8: 64-bucket geometric log-spaced histogram covering 100 µs to 10 s with published boundaries
//!   and representative values (geometric mean), rolling window of fixed capacity W.
//! - § 4: Non-overlapping block-based sampling (block duration = max(SRTT, min_block_duration)),
//!   block medians as representative observations, and distribution-free order-statistic confidence gate.

use core::time::Duration;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec;

use crate::math::{ceil_f64, floor_f64, sqrt_f64};

// ─── Constant Tables (§ 8) ───────────────────────────────────────────────────

/// 63 boundary values (in nanoseconds) defining the geometric progression from 100 µs to 10 s
/// across the 62 ordinary buckets (buckets 1 through 62).
///
/// Ratio r = (10_000_000_000 / 100_000)^(1/62) ≈ 1.2040517723.
pub const HISTOGRAM_BOUNDARIES_NANOS: [u64; 63] = [
    100_000,        // B_0: 100 µs
    120_405,        // B_1
    144_974,        // B_2
    174_556,        // B_3
    210_175,        // B_4
    253_061,        // B_5
    304_699,        // B_6
    366_873,        // B_7
    441_734,        // B_8
    531_871,        // B_9
    640_400,        // B_10
    771_075,        // B_11
    928_415,        // B_12
    1_117_859,      // B_13
    1_345_960,      // B_14
    1_620_606,      // B_15
    1_951_293,      // B_16
    2_349_458,      // B_17
    2_828_869,      // B_18
    3_406_105,      // B_19
    4_101_127,      // B_20
    4_937_969,      // B_21
    5_945_571,      // B_22
    7_158_775,      // B_23
    8_619_536,      // B_24
    10_378_367,     // B_25
    12_496_091,     // B_26
    15_045_941,     // B_27
    18_116_092,     // B_28
    21_812_713,     // B_29
    26_263_635,     // B_30
    31_622_777,     // B_31
    38_075_460,     // B_32
    45_844_825,     // B_33
    55_199_543,     // B_34
    66_463_108,     // B_35
    80_025_023,     // B_36
    96_354_271,     // B_37
    116_015_530,    // B_38
    139_688_705,    // B_39
    168_192_432,    // B_40
    202_512_396,    // B_41
    243_835_410,    // B_42
    293_590_457,    // B_43
    353_498_111,    // B_44
    425_630_026,    // B_45
    512_480_588,    // B_46
    617_053_160,    // B_47
    742_963_951,    // B_48
    894_567_062,    // B_49
    1_077_105_056,  // B_50
    1_296_890_252,  // B_51
    1_561_523_006,  // B_52
    1_880_154_543,  // B_53
    2_263_803_410,  // B_54
    2_725_736_507,  // B_55
    3_281_927_873,  // B_56
    3_951_611_071,  // B_57
    4_757_944_314,  // B_58
    5_728_811_284,  // B_59
    6_897_785_379,  // B_60
    8_305_290_711,  // B_61
    10_000_000_000, // B_62: 10 s
];

/// 64 representative values (in nanoseconds) for buckets 0..63.
/// Bucket 0: underflow (100 µs).
/// Buckets 1..62: geometric mean of bucket lower and upper boundaries.
/// Bucket 63: overflow (10 s).
pub const HISTOGRAM_REPRESENTATIVES_NANOS: [u64; 64] = [
    100_000,        // Bucket 0 (underflow)
    109_729,        // Bucket 1: sqrt(B_0 * B_1)
    132_120,        // Bucket 2
    159_079,        // Bucket 3
    191_539,        // Bucket 4
    230_623,        // Bucket 5
    277_682,        // Bucket 6
    334_344,        // Bucket 7
    402_567,        // Bucket 8
    484_712,        // Bucket 9
    583_618,        // Bucket 10
    702_707,        // Bucket 11
    846_096,        // Bucket 12
    1_018_743,      // Bucket 13
    1_226_619,      // Bucket 14
    1_476_913,      // Bucket 15
    1_778_279,      // Bucket 16
    2_141_140,      // Bucket 17
    2_578_044,      // Bucket 18
    3_104_098,      // Bucket 19
    3_737_495,      // Bucket 20
    4_500_138,      // Bucket 21
    5_418_399,      // Bucket 22
    6_524_033,      // Bucket 23
    7_855_273,      // Bucket 24
    9_458_156,      // Bucket 25
    11_388_109,     // Bucket 26
    13_711_873,     // Bucket 27
    16_509_805,     // Bucket 28
    19_878_660,     // Bucket 29
    23_934_935,     // Bucket 30
    28_818_901,     // Bucket 31
    34_699_449,     // Bucket 32
    41_779_933,     // Bucket 33
    50_305_202,     // Bucket 34
    60_570_068,     // Bucket 35
    72_929_498,     // Bucket 36
    87_810_892,     // Bucket 37
    105_728_860,    // Bucket 38
    127_303_021,    // Bucket 39
    153_279_428,    // Bucket 40
    184_556_366,    // Bucket 41
    222_215_421,    // Bucket 42
    267_558_871,    // Bucket 43
    322_154_733,    // Bucket 44
    387_890_977,    // Bucket 45
    467_040_818,    // Bucket 46
    562_341_325,    // Bucket 47
    677_088_069,    // Bucket 48
    815_249_090,    // Bucket 49
    981_602_112,    // Bucket 50
    1_181_899_762,  // Bucket 51
    1_423_068_503,  // Bucket 52
    1_713_448_153,  // Bucket 53
    2_063_080_286,  // Bucket 54
    2_484_055_474,  // Bucket 55
    2_990_931_396,  // Bucket 56
    3_601_236_248,  // Bucket 57
    4_336_074_887,  // Bucket 58
    5_220_858_653,  // Bucket 59
    6_286_184_114,  // Bucket 60
    7_568_891_123,  // Bucket 61
    9_113_336_771,  // Bucket 62
    10_000_000_000, // Bucket 63 (overflow)
];

/// Total bucket count.
pub const TOTAL_BUCKETS: usize = 64;
/// Underflow bucket index.
pub const UNDERFLOW_BUCKET: usize = 0;
/// Overflow bucket index.
pub const OVERFLOW_BUCKET: usize = 63;

// ─── 64-Bucket Geometric Histogram (§ 8) ────────────────────────────────────

/// Outcome of a quantile lookup on [`RttHistogram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantileEstimate {
    /// Representative value for the quantile.
    pub value: Duration,
    /// Bucket ID where quantile fell (0..63).
    pub bucket: usize,
    /// Whether the estimate is bounded (i.e. Not in underflow or overflow).
    pub is_bounded: bool,
    /// Lower edge of the containing bucket (if bounded).
    pub lower_edge: Option<Duration>,
    /// Upper edge of the containing bucket (if bounded).
    pub upper_edge: Option<Duration>,
}

/// Fixed-capacity rolling 64-bucket geometric log-histogram.
///
/// - Bucket 0 = underflow (< 100 µs)
/// - Bucket 63 = overflow (≥ 10 s)
/// - Buckets 1..62 = ordinary log-spaced range
///
/// Stores exact bucket IDs in a ring buffer; eviction decrements the exact bucket.
#[derive(Clone, Debug)]
pub struct RttHistogram {
    /// Bucket frequency counts (u32, no scaling).
    buckets: [u32; TOTAL_BUCKETS],
    /// Ring buffer storing exact bucket IDs (0..=63) for exact eviction.
    ring: Vec<u8>,
    /// Capacity W.
    capacity: usize,
    /// Write cursor in ring.
    cursor: usize,
    /// Number of active items in the window (≤ capacity).
    count: usize,
}

impl RttHistogram {
    /// Create a new histogram with rolling window capacity `w` (clamped to [1, 65536]).
    pub fn new(w: usize) -> Self {
        let capacity = w.clamp(1, 65536);
        Self {
            buckets: [0; TOTAL_BUCKETS],
            ring: vec![0; capacity],
            capacity,
            cursor: 0,
            count: 0,
        }
    }

    /// Total samples currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the histogram is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Map an RTT duration to its exact bucket index (0..63).
    #[inline]
    pub fn bucket_for(rtt: Duration) -> usize {
        let nanos = rtt.as_nanos() as u64;
        if nanos < HISTOGRAM_BOUNDARIES_NANOS[0] {
            return UNDERFLOW_BUCKET;
        }
        if nanos >= HISTOGRAM_BOUNDARIES_NANOS[62] {
            return OVERFLOW_BUCKET;
        }
        // Binary search over the 63 boundaries
        HISTOGRAM_BOUNDARIES_NANOS.partition_point(|&b| b <= nanos)
    }

    /// Get representative duration for a bucket.
    #[inline]
    pub fn representative_for_bucket(bucket: usize) -> Duration {
        let idx = bucket.min(TOTAL_BUCKETS - 1);
        Duration::from_nanos(HISTOGRAM_REPRESENTATIVES_NANOS[idx])
    }

    /// Get lower and upper boundary edges for ordinary bucket `bucket` (1..62).
    #[inline]
    pub fn edges_for_bucket(bucket: usize) -> Option<(Duration, Duration)> {
        if bucket == 0 || bucket >= OVERFLOW_BUCKET {
            None
        } else {
            let lower = Duration::from_nanos(HISTOGRAM_BOUNDARIES_NANOS[bucket - 1]);
            let upper = Duration::from_nanos(HISTOGRAM_BOUNDARIES_NANOS[bucket]);
            Some((lower, upper))
        }
    }

    /// Push an eligible sample into the histogram.
    pub fn push(&mut self, rtt: Duration) {
        let bucket = Self::bucket_for(rtt) as u8;

        if self.count == self.capacity {
            // Evict oldest sample from exact bucket
            let old_bucket = self.ring[self.cursor] as usize;
            self.buckets[old_bucket] = self.buckets[old_bucket].saturating_sub(1);
        } else {
            self.count += 1;
        }

        self.ring[self.cursor] = bucket;
        self.buckets[bucket as usize] = self.buckets[bucket as usize].saturating_add(1);
        self.cursor = (self.cursor + 1) % self.capacity;
    }

    /// Read raw bucket count.
    #[inline]
    pub fn bucket_count(&self, bucket: usize) -> u32 {
        if bucket < TOTAL_BUCKETS {
            self.buckets[bucket]
        } else {
            0
        }
    }

    /// Compute quantile `q` ∈ (0.0, 1.0].
    /// Returns `None` if histogram is empty.
    pub fn quantile(&self, q: f64) -> Option<QuantileEstimate> {
        if self.count == 0 || !q.is_finite() || q <= 0.0 {
            return None;
        }

        let clamped_q = q.min(1.0);
        let target_rank = ceil_f64((self.count as f64) * clamped_q) as u32;
        let rank = target_rank.clamp(1, self.count as u32);

        let mut cum: u32 = 0;
        let mut target_bucket = OVERFLOW_BUCKET;

        for (bucket_idx, &cnt) in self.buckets.iter().enumerate() {
            cum = cum.saturating_add(cnt);
            if cum >= rank {
                target_bucket = bucket_idx;
                break;
            }
        }

        let is_bounded = target_bucket != UNDERFLOW_BUCKET && target_bucket != OVERFLOW_BUCKET;
        let edges = Self::edges_for_bucket(target_bucket);
        let value = Self::representative_for_bucket(target_bucket);

        Some(QuantileEstimate {
            value,
            bucket: target_bucket,
            is_bounded,
            lower_edge: edges.map(|(lo, _)| lo),
            upper_edge: edges.map(|(_, hi)| hi),
        })
    }
}

// ─── P95 Confidence Gate & Block-Based Sampling (§ 4) ──────────────────────

/// Confidence gate evaluation result (§ 4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct P95ConfidenceResult {
    /// The estimated P95 duration (if available).
    pub p95: Option<Duration>,
    /// Half-width of the 95% confidence interval (after bucket width).
    pub ci_half_width: Option<Duration>,
    /// The maximum allowed uncertainty threshold = max(20% of P95, 2 ms).
    pub max_allowed_uncertainty: Option<Duration>,
    /// Number of block representatives evaluated.
    pub block_representatives_count: usize,
    /// Whether the confidence gate passes.
    pub gate_passed: bool,
    /// Tail guard duration η · T_95 (enabled only when gate passes).
    pub tail_guard: Option<Duration>,
}

/// Block-based sampler and distribution-free order-statistic confidence gate.
#[derive(Clone, Debug)]
pub struct BlockBasedP95Gate {
    /// Rolling window capacity for block representatives.
    capacity: usize,
    /// Ring of block representatives (medians of observation blocks).
    reps: Vec<Duration>,
    reps_cursor: usize,
    reps_count: usize,

    /// Samples accumulating in the current open block.
    current_block_samples: Vec<Duration>,
    /// Monotonic duration elapsed in current block.
    current_block_elapsed: Duration,
    /// Configured tail guard factor η (e.g. 1.5).
    tail_guard_eta: f64,
    /// Minimum duration for an observation block (default 10 ms).
    min_block_duration: Duration,
}

impl BlockBasedP95Gate {
    /// Create a new gate with window size `w` and tail guard multiplier `eta`.
    pub fn new(w: usize, eta: f64) -> Self {
        let capacity = w.clamp(16, 512);
        let safe_eta = if eta.is_finite() && eta >= 1.0 { eta } else { 1.5 };
        Self {
            capacity,
            reps: vec![Duration::ZERO; capacity],
            reps_cursor: 0,
            reps_count: 0,
            current_block_samples: Vec::new(),
            current_block_elapsed: Duration::ZERO,
            tail_guard_eta: safe_eta,
            min_block_duration: Duration::from_millis(10),
        }
    }

    /// Record an eligible unambiguous RTT sample, advancing block accumulation.
    ///
    /// `estimated_srtt`: current path SRTT estimate used to size the block duration.
    /// `dt`: elapsed time since previous sample.
    pub fn record_sample(&mut self, rtt: Duration, estimated_srtt: Duration, dt: Duration) {
        self.current_block_samples.push(rtt);
        self.current_block_elapsed += dt;

        let block_len = estimated_srtt.max(self.min_block_duration);
        if self.current_block_elapsed >= block_len {
            self.finalize_current_block();
        }
    }

    /// Directly record a block representative observation (e.g. In simulation or test).
    pub fn record_representative(&mut self, rep: Duration) {
        self.reps[self.reps_cursor] = rep;
        self.reps_cursor = (self.reps_cursor + 1) % self.capacity;
        if self.reps_count < self.capacity {
            self.reps_count += 1;
        }
    }

    /// Force finalization of current block if it contains samples.
    pub fn finalize_current_block(&mut self) {
        if !self.current_block_samples.is_empty() {
            let mut sorted = self.current_block_samples.clone();
            sorted.sort_unstable();
            let n = sorted.len();
            let median = if n % 2 == 1 {
                sorted[n / 2]
            } else {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2
            };
            self.record_representative(median);
            self.current_block_samples.clear();
        }
        self.current_block_elapsed = Duration::ZERO;
    }

    /// Number of active block representatives.
    #[inline]
    pub fn representative_count(&self) -> usize {
        self.reps_count
    }

    /// Active block representatives sorted.
    pub fn active_sorted_representatives(&self) -> Vec<Duration> {
        let mut s = if self.reps_count < self.capacity {
            self.reps[..self.reps_count].to_vec()
        } else {
            self.reps.clone()
        };
        s.sort_unstable();
        s
    }

    /// Evaluate the 95% confidence gate per CONTRACT.md § 4.
    pub fn evaluate(&self) -> P95ConfidenceResult {
        let m = self.reps_count;
        if m < 20 {
            // Sample gate: Need at least 20 block representatives for two-sided 95% CI
            return P95ConfidenceResult {
                p95: None,
                ci_half_width: None,
                max_allowed_uncertainty: None,
                block_representatives_count: m,
                gate_passed: false,
                tail_guard: None,
            };
        }

        let sorted = self.active_sorted_representatives();
        let p95_rank = ceil_f64((m as f64) * 0.95) as usize;
        let p95_idx = p95_rank.saturating_sub(1).min(m - 1);
        let p95_val = sorted[p95_idx];

        // Distribution-free binomial order-statistic confidence ranks for Binomial(M, 0.95):
        let (l_rank, u_rank) = binomial_95_ci_ranks(m);
        let l_idx = l_rank.saturating_sub(1).min(m - 1);
        let u_idx = u_rank.saturating_sub(1).min(m - 1);

        let lower_obs = sorted[l_idx];
        let upper_obs = sorted[u_idx];

        // Map to histogram buckets to account for bucket width (§ 4 & § 8)
        let lower_bucket = RttHistogram::bucket_for(lower_obs);
        let upper_bucket = RttHistogram::bucket_for(upper_obs);

        // If either falls in underflow (0) or overflow (63), interval is unbounded
        if lower_bucket == UNDERFLOW_BUCKET || upper_bucket == OVERFLOW_BUCKET {
            return P95ConfidenceResult {
                p95: Some(p95_val),
                ci_half_width: None,
                max_allowed_uncertainty: None,
                block_representatives_count: m,
                gate_passed: false,
                tail_guard: None,
            };
        }

        // Account for bucket width: lower edge of lower bucket, upper edge of upper bucket
        let (lower_edge, _) = match RttHistogram::edges_for_bucket(lower_bucket) {
            Some(e) => e,
            None => return P95ConfidenceResult {
                p95: Some(p95_val),
                ci_half_width: None,
                max_allowed_uncertainty: None,
                block_representatives_count: m,
                gate_passed: false,
                tail_guard: None,
            },
        };
        let (_, upper_edge) = match RttHistogram::edges_for_bucket(upper_bucket) {
            Some(e) => e,
            None => return P95ConfidenceResult {
                p95: Some(p95_val),
                ci_half_width: None,
                max_allowed_uncertainty: None,
                block_representatives_count: m,
                gate_passed: false,
                tail_guard: None,
            },
        };

        let ci_width = upper_edge.saturating_sub(lower_edge);
        let half_width = ci_width / 2;

        // Bound: half-width <= max(20% of current P95, 2 ms)
        let p95_20pct = p95_val.mul_f64(0.20);
        let max_allowed = p95_20pct.max(Duration::from_millis(2));

        let gate_passed = half_width <= max_allowed;
        let tail_guard = if gate_passed {
            Some(p95_val.mul_f64(self.tail_guard_eta))
        } else {
            None
        };

        P95ConfidenceResult {
            p95: Some(p95_val),
            ci_half_width: Some(half_width),
            max_allowed_uncertainty: Some(max_allowed),
            block_representatives_count: m,
            gate_passed,
            tail_guard,
        }
    }
}

/// Compute optimal binomial order statistic ranks [L, U] for Binomial(M, 0.95) covering >= 95%.
///
/// Uses standard closed-form binomial order-statistic rank derivation for M >= 20.
#[inline]
fn binomial_95_ci_ranks(m: usize) -> (usize, usize) {
    let p = 0.95_f64;
    let mean = (m as f64) * p;
    let var = (m as f64) * p * (1.0 - p);
    let std_dev = sqrt_f64(var);

    // 1.96 standard deviations for two-sided 95% confidence interval
    let l_float = floor_f64(mean - 1.96 * std_dev);
    let u_float = ceil_f64(mean + 1.96 * std_dev);

    let l = (l_float as usize).clamp(1, m);
    let u = (u_float as usize).clamp(l + 1, m);
    (l, u)
}

// ─── Rolling P95 Wrapper (Backwards-Compatible & Histogram-Backed) ──────────

/// Rolling-window P95 tail tracker incorporating the § 8 histogram and § 4 confidence gate.
#[derive(Clone, Debug)]
pub struct RollingP95 {
    /// Underlying 64-bucket geometric histogram.
    histogram: RttHistogram,
    /// Ring buffer of raw samples for backwards-compatible order-statistic computation.
    raw_buf: Vec<Duration>,
    capacity: usize,
    cursor: usize,
    count: usize,
    /// Block-based confidence gate.
    gate: BlockBasedP95Gate,
}

impl RollingP95 {
    /// Create a new tracker with window size `w` (clamped to [16, 1024]).
    pub fn new(w: usize) -> Self {
        let capacity = w.clamp(16, 1024);
        Self {
            histogram: RttHistogram::new(capacity),
            raw_buf: vec![Duration::ZERO; capacity],
            capacity,
            cursor: 0,
            count: 0,
            gate: BlockBasedP95Gate::new(capacity, 1.5),
        }
    }

    /// Push an eligible RTT sample into the ring buffer, histogram, and gate.
    pub fn push(&mut self, rtt: Duration) {
        self.histogram.push(rtt);
        self.raw_buf[self.cursor] = rtt;
        self.cursor = (self.cursor + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        // Also feed into block gate with default 10ms pacing
        self.gate.record_sample(rtt, Duration::from_millis(10), Duration::from_millis(5));
    }

    /// Access the underlying 64-bucket histogram.
    #[inline]
    pub fn histogram(&self) -> &RttHistogram {
        &self.histogram
    }

    /// Access the block-based confidence gate.
    #[inline]
    pub fn gate(&self) -> &BlockBasedP95Gate {
        &self.gate
    }

    /// Access the block-based confidence gate mutably.
    #[inline]
    pub fn gate_mut(&mut self) -> &mut BlockBasedP95Gate {
        &mut self.gate
    }

    /// Number of valid samples currently stored.
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
        let mut sorted = self.active_samples();
        sorted.sort_unstable();
        let rank = (19 * self.count).div_ceil(20);
        let idx = rank.saturating_sub(1).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Return a sorted copy of the active window.
    pub fn active_samples(&self) -> Vec<Duration> {
        if self.count < self.capacity {
            self.raw_buf[..self.count].to_vec()
        } else {
            self.raw_buf.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        let t = RollingP95::new(64);
        assert!(t.p95().is_none());
        assert!(t.histogram().is_empty());
    }

    #[test]
    fn single_sample() {
        let mut t = RollingP95::new(64);
        t.push(Duration::from_millis(100));
        assert_eq!(t.p95(), Some(Duration::from_millis(100)));
        assert_eq!(t.histogram().len(), 1);
    }

    #[test]
    fn twenty_sequential_samples() {
        let mut t = RollingP95::new(64);
        for i in 1..=20 {
            t.push(Duration::from_millis(i));
        }
        assert_eq!(t.p95(), Some(Duration::from_millis(19)));
    }

    #[test]
    fn wrapping_evicts_old() {
        let mut t = RollingP95::new(16); // minimum clamped to 16
        for &v in &[10, 20, 30, 40] {
            t.push(Duration::from_millis(v));
        }
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn ceil_mul_95_spot_checks() {
        let c = |n: usize| (19 * n).div_ceil(20);
        assert_eq!(c(1), 1);
        assert_eq!(c(20), 19);
        assert_eq!(c(64), 61);
        assert_eq!(c(100), 95);
    }

    // ─── Contract § 8 Tests: Monotonicity, Boundaries, Inverses, Quantization ──

    #[test]
    fn test_histogram_monotonicity() {
        // Boundaries must strictly increase: B_0 < B_1 < ... < B_62
        for i in 0..62 {
            assert!(
                HISTOGRAM_BOUNDARIES_NANOS[i] < HISTOGRAM_BOUNDARIES_NANOS[i + 1],
                "Boundaries must strictly increase at index {i}"
            );
        }
        // Representatives must strictly increase: Rep_0 < Rep_1 < ... < Rep_63
        for i in 0..63 {
            assert!(
                HISTOGRAM_REPRESENTATIVES_NANOS[i] < HISTOGRAM_REPRESENTATIVES_NANOS[i + 1],
                "Representatives must strictly increase at index {i}"
            );
        }
    }

    #[test]
    fn test_histogram_boundary_behavior() {
        // Underflow: < 100 µs (99_999 ns)
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(99_999)),
            UNDERFLOW_BUCKET
        );
        assert_eq!(
            RttHistogram::bucket_for(Duration::ZERO),
            UNDERFLOW_BUCKET
        );

        // Exact lower bound B_0 = 100 µs -> Bucket 1
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(100_000)),
            1
        );

        // Interior of Bucket 1
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(110_000)),
            1
        );

        // Exact boundary B_1 = 120_405 ns -> Bucket 2
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(120_405)),
            2
        );

        // B_61 boundary -> Bucket 62
        let b61 = HISTOGRAM_BOUNDARIES_NANOS[61];
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(b61)),
            62
        );

        // Upper boundary B_62 = 10s (10_000_000_000 ns) -> Overflow (Bucket 63)
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_nanos(10_000_000_000)),
            OVERFLOW_BUCKET
        );
        // Above 10s -> Overflow
        assert_eq!(
            RttHistogram::bucket_for(Duration::from_secs(12)),
            OVERFLOW_BUCKET
        );
    }

    #[test]
    fn test_histogram_inverse_mapping() {
        // Mapping a bucket's representative value back to bucket_for must return that exact bucket
        for bucket in 1..=62 {
            let rep = RttHistogram::representative_for_bucket(bucket);
            let mapped_bucket = RttHistogram::bucket_for(rep);
            assert_eq!(
                mapped_bucket, bucket,
                "Representative for bucket {bucket} ({rep:?}) mapped to {mapped_bucket}"
            );
        }
    }

    #[test]
    fn test_histogram_quantization_bound() {
        // CONTRACT.md § 8: maximum relative error <= 1.5x within the covered range [100 µs, 10 s]
        for bucket in 1..=62 {
            let (lower, upper) = RttHistogram::edges_for_bucket(bucket).unwrap();
            let rep = RttHistogram::representative_for_bucket(bucket);

            let lower_secs = lower.as_secs_f64();
            let upper_secs = upper.as_secs_f64();
            let rep_secs = rep.as_secs_f64();

            // Check lower edge ratio
            let ratio_lower = rep_secs / lower_secs;
            assert!(
                ratio_lower <= 1.5,
                "Bucket {bucket} lower ratio {ratio_lower} exceeds 1.5"
            );

            // Check upper edge ratio
            let ratio_upper = upper_secs / rep_secs;
            assert!(
                ratio_upper <= 1.5,
                "Bucket {bucket} upper ratio {ratio_upper} exceeds 1.5"
            );
        }
    }

    #[test]
    fn test_histogram_eviction_ring() {
        let mut hist = RttHistogram::new(4);
        // Push 4 samples: 100 µs (b1), 1 ms (b13), 10 ms (b25), 100 ms (b38)
        let s1 = Duration::from_micros(100);
        let s2 = Duration::from_millis(1);
        let s3 = Duration::from_millis(10);
        let s4 = Duration::from_millis(100);

        hist.push(s1);
        hist.push(s2);
        hist.push(s3);
        hist.push(s4);

        assert_eq!(hist.len(), 4);
        let b1 = RttHistogram::bucket_for(s1);
        assert_eq!(hist.bucket_count(b1), 1);

        // Push 5th sample -> evicts s1
        let s5 = Duration::from_millis(200);
        hist.push(s5);

        assert_eq!(hist.len(), 4);
        assert_eq!(hist.bucket_count(b1), 0, "Evicted sample should decrement exact bucket count");
        let b5 = RttHistogram::bucket_for(s5);
        assert_eq!(hist.bucket_count(b5), 1);
    }

    // ─── Contract § 4 Tests: P95 Confidence Gate & Block Sampling ──────────────

    #[test]
    fn test_confidence_gate_sample_count_threshold() {
        let mut gate = BlockBasedP95Gate::new(64, 1.5);
        // < 20 representatives -> gate fails
        for _ in 0..19 {
            gate.record_representative(Duration::from_millis(50));
        }
        let res = gate.evaluate();
        assert!(!res.gate_passed, "Gate must not pass with < 20 block representatives");
        assert!(res.tail_guard.is_none());

        // 20th representative -> gate evaluated
        gate.record_representative(Duration::from_millis(50));
        assert_eq!(gate.representative_count(), 20);
        let res2 = gate.evaluate();
        assert!(res2.gate_passed, "Tight 50ms samples should pass gate");
        assert!(res2.tail_guard.is_some());
    }

    #[test]
    fn test_confidence_gate_unbounded_underflow() {
        let mut gate = BlockBasedP95Gate::new(64, 1.5);
        // Samples in underflow (< 100 µs)
        for _ in 0..30 {
            gate.record_representative(Duration::from_micros(50));
        }
        let res = gate.evaluate();
        // Underflow bucket is unbounded on left -> gate must not activate tail guard
        assert!(!res.gate_passed);
        assert!(res.tail_guard.is_none());
    }
}
