# P6851 Frozen Contract

Architecture is frozen. This document is the implementation contract.

Four planes:

- **Measurement**: RTT (Reference/Adaptive branches), RTTVAR, P95 (block-sampled), timestamps, censored observations (separate).
- **Classification**: Eligible population, expected silence, suspected loss, rate limiting, calibrated confidence, unknown.
- **Control**: Latched RTO, hierarchical AIMD (isolated + correlated global), independent pacing.
- **Execution**: ProbePermit lifecycle, generational arenas, retry schedule, cancellation, Tokio integration.

Censored-data / survival estimation is a separately validated extension and is outside the frozen control path.

---

## 1. RTT estimators

### Reference (RFC 6298)

Unclamped residual: \(e_c = e = R - \text{SRTT}_{\text{old}}\).

Initialization and update order exactly as RFC 6298.

Before any RTT sample exists, RTO = 1 s.

After the first unambiguous sample, initialize SRTT and RTTVAR per RFC 6298, compute RTO from those values, then apply the Reference-mode minimum (1 s). 1 s is never treated as a post-first-sample estimate.

No variance floor, no clamp, no Adaptive-only logic.

### Adaptive

Only unambiguous first-attempt (\(n=0\)) samples update either estimator.

\[
e_c = \operatorname{clamp}(e,\, -k\cdot\text{RTTVAR}_{\text{old}},\, k\cdot\text{RTTVAR}_{\text{old}}),\quad k\in[2,3]
\]
\[
\text{SRTT}_{\text{new}} = \text{SRTT}_{\text{old}} + \alpha\, e_c
\]
\[
\text{RTTVAR}_{\text{new}} = \max\bigl(\text{RTTVAR}_{\min},\, (1-\beta)\,\text{RTTVAR}_{\text{old}} + \beta\,|e|\bigr)
\]

\(\alpha=1/8\), \(\beta=1/4\), \(\text{RTTVAR}_{\min}\) floor (1 ms).

Retransmitted successes are excluded from both estimators.

---

## 2. Change-detector re-anchor

Representative sample per epoch = median of eligible unambiguous samples in that epoch.

Violation counter advances only on **eligible violating epochs**. Zero-sample epochs leave the counter unchanged.

On \(m\) consecutive eligible violating epochs, re-anchor from the retained eligible samples \(R_1\dots R_N\) across those epochs and their per-epoch medians \(M_1\dots M_m\):

\[
\text{SRTT}_{\text{new}} = \operatorname{median}(M_1,\dots,M_m)
\]
\[
\text{RTTVAR}_{\text{new}} = \max\left(
  \text{RTTVAR}_{\min},\,
  \frac{1}{N}\sum_{i=1}^{N}|R_i - \text{SRTT}_{\text{new}}|
\right)
\]

After re-anchor:

- Violation counter is reset to 0.
- Detector baseline moves to the new SRTT / RTTVAR.
- The samples \(R_1\dots R_N\) are used solely for the re-anchor calculation. They are not fed back into the normal Adaptive EWMA update path.

---

## 3. Hierarchical AIMD and global evidence

Decrease rules are isolated:

- Destination loss decreases only the destination window.
- Group loss decreases only the group window.
- Global decrease requires global evidence.

### Global evidence (fixed-point water-filling)

\[
G_{\min} = 4,\qquad f_{\max} = 0.25
\]

Let \(c_g\) be the raw confidence sum contributed by group \(g\).

Solve for the consistent capped aggregate \(S\):

\[
S = \sum_g \min(c_g,\, f_{\max} S)
\]

Then the capped contribution of each group is:

\[
c'_g = \min(c_g,\, f_{\max} S)
\]

Global evidence holds in an epoch iff:

- At least \(G_{\min}\) distinct groups each independently satisfy their own loss-trigger conditions, and
- The capped sum \(S\) meets the global evidence and loss-rate thresholds.

A single group cannot exceed \(f_{\max}\) of the final capped total used for the decision.

Loss trigger (per tier, on that tier's eligible population):

\[
n_{\text{eligible}} \ge N_{\min},\qquad
\sum_i c_{\text{loss},i} \ge E_{\min},\qquad
\frac{\sum_i c_{\text{loss},i}}{n_{\text{eligible}}} \ge p_{\min}
\]

---

## 4. P95 confidence gate

Histogram receives only unambiguous first-attempt (\(n=0\)) RTT samples.

Independence is not assumed for consecutive samples. Use block-based sampling:

- Partition eligible RTT samples into non-overlapping observation blocks of length equal to one estimated RTT (or a fixed minimum block duration).
- Use the median of each block as one representative observation.
- Apply the distribution-free order-statistic confidence calculation to those block representatives.

Gate parameters:

- Confidence level: 95 %
- Maximum quantile uncertainty: half-width of the 95 % CI ≤ max(20 % of current P95, 2 ms), after accounting for bucket width
- Sample gate: smallest number of block representatives such that the interval meets the uncertainty bound
- Tail guard (\(\eta\cdot T_{95}\)) is disabled until the gate passes

When a quantile falls in the underflow or overflow bucket, the confidence interval is reported as unbounded on that side; the tail guard is not activated from an unbounded estimate.

---

## 5. ProbePermit lifecycle

```text
Reserved
   │ mark_dispatched(now) → sets first_send_us, latches full retry policy
   ▼
Dispatched (attempt 0..max)
   ├─ record_outcome          → Completed
   ├─ cancel_probe            → Cancelled
   ├─ retries_exhausted       → TimedOut
   └─ reservation-lease expiry → Reclaimed   (only while Reserved)
```

- Unique `ProbeId` per permit.
- Full policy (base RTO, \(\gamma_{\text{backoff}}\), max retries) is latched at `mark_dispatched`.
- Reservation lease ≠ network retry schedule. Network timers start only after successful `mark_dispatched`.
- `retries_exhausted` (latched max retries reached with no response) transitions to `TimedOut`, releases all in-flight / concurrency accounting, and does **not** automatically assert packet loss or path failure. Classification remains a separate step.
- After a probe is Dispatched, if the Tokio cancellation queue is full on `Drop`, the permit is marked for forced reclamation at the next engine maintenance tick (same accounting release as `TimedOut`, diagnostic counter incremented). No active permit is left permanently held.
- Tokio adapter: cancellation-safe guard using bounded non-blocking send; the above recovery is the fallback.

---

## 6. Generational arenas

- Fixed-capacity destination and group arenas.
- Generation-checked private handles (distinct types for destination vs group).
- Pin while any in-flight, pending retry, outstanding reservation, or queued dispatch exists. A group remains pinned while any child is pinned.
- Bounded key-to-slot lookup.
- Explicit `ArenaCapacityPinned` result; caller may retry after capacity frees.
- `u64` generation or permanent retirement on wrap.

---

## 7. Independent pacing

Global, group, and destination pacing (token/leaky bucket or equivalent) are separate from concurrency ceilings.

Every network transmission, including retransmissions, must acquire an atomic pacing reservation across all applicable pacing scopes.

All applicable pacing buckets are checked and consumed atomically. If any bucket fails, none of the tokens are consumed and `mark_dispatched` returns `PaceUntil { wake_at_us }`. The Reserved permit remains held; its reservation lease continues to run.

A retry does not acquire a new concurrency permit but must still pass the pacers.

While a permit is Reserved and waiting for pacing, only the reservation lease applies.

---

## 8. Histogram

- 64 buckets total: bucket 0 = underflow (< 100 µs), bucket 63 = overflow (≥ 10 s), buckets 1–62 = ordinary log-spaced range.
- Exact boundaries: geometric progression from 100 µs to 10 s across the 62 ordinary buckets. The 63 boundary values are published as a constant table in the implementation.
- Representative value for an ordinary bucket = geometric mean of its lower and upper edges.
- Underflow / overflow buckets report the boundary value (100 µs / 10 s) and are flagged.
- Rolling window of fixed capacity \(W\). Store exact bucket IDs in a ring; eviction decrements the exact bucket.
- Counters are `u32`. Window size \(W\) is chosen so that no bucket can exceed `u32::MAX` under normal operation. No scaling is performed.
- Quantization bound: maximum relative error ≤ 1.5× within the covered range. Implementation must test monotonicity, boundary behavior, inverse mapping, and actual quantization error.

---

## 9. RTO equations

**Base RTO (both profiles):**

\[
\text{RTO}_0 = \operatorname{clamp}\bigl(\text{SRTT} + \max(G,\,4\cdot\text{RTTVAR}),\,\text{RTO}_{\min},\,\text{RTO}_{\max}\bigr)
\]

**Adaptive profile only:**

Variance multiplier remains 4.

Backoff multiplier \(\gamma_{\text{backoff}}\in[1.4,1.6]\) applies exclusively to retries:

\[
\text{RTO}_n = \min\bigl(\text{RTO}_{\max},\,\text{RTO}_0\cdot\gamma_{\text{backoff}}^n\bigr)
\]

Complete retry policy is latched at first dispatch. Later configuration changes do not affect in-flight probes.

Reference profile uses RFC 6298 (\(2\times\) backoff, 1 s minimum).

---

## 10. EWMA time semantics (loss / rate-limit monitoring)

- Eligible successful observations contribute \(c=0\).
- Suspected path-loss observations contribute their calibrated \(c_{\text{loss}}\).
- Suspected rate-limit observations update the separate rate-limit EWMA.
- Unknown and ExpectedSilence do not update either signal.
- Between eligible observations the signal is held (no automatic decay toward zero). A long gap with zero eligible observations is not evidence of zero loss.
- \(\lambda(\Delta t)=1-e^{-\Delta t/\tau}\) applies only at the moment an eligible observation arrives.
- Confidence values are a validated type. Statistical calibration is a separate documented requirement with labeled-trace tests.

---

## Implementation notes

All state transitions, numerical calculations, and scheduling invariants must have independent tests.

P6851 measures network behavior, expresses uncertainty about what it observes, and makes bounded control decisions. It never turns an ambiguous timeout into a supposedly certain network diagnosis.
