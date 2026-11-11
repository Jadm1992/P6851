---
name: 6298-optimization
description: The unified **Final Mathematical Contract (v3 — Frozen for Implementation)** below consolidates the original specification and the final draft while restoring all five omitted requirements (RTT initialization, Karn's algorithm, explicit backoff formulas, AIMD equations, and time-normalized classified EWMAs) alongside deterministic rolling-window P95 tracking and epoch-median change detection.
---

## Final Mathematical Contract (v3 — Frozen)

### 1. Outlier-Resistant RTT Estimator & Tail Tracker

#### 1.1 Eligibility (Karn's Algorithm)

Let $n \in \{0, 1, \dots, N_{\max}\}$ denote the retransmission index of a probe ($n = 0$ is the initial transmission; $n \ge 1$ is a retry).

* **Rule:** Only responses to unambiguous, non-retransmitted probes ($n = 0$) produce an eligible RTT sample $R$.
* Ambiguous responses from retransmitted probes ($n \ge 1$) **must not** update $\text{SRTT}$, $\text{RTTVAR}$, the rolling tail tracker $T_{95}$, or the change detector.

#### 1.2 Initialization (First Eligible Sample)

Prior to the first eligible sample, the estimator is in state `Uninitialized` and uses the profile's initial timeout $\text{RTO}_{\text{init}}$. Upon receiving the first eligible sample $R_0$:


$$\text{SRTT} = R_0$$

$$\text{RTTVAR} = \max\left(\frac{R_0}{2},\, \text{RTTVAR}_{\min}\right)$$

* **Reference Profile:** $\text{RTTVAR}_{\min} = 0$ (strict RFC 6298).
* **Adaptive Profile:** $\text{RTTVAR}_{\min} \gt 0$ (default $1\text{ ms}$).

#### 1.3 Steady-State Update (Subsequent Eligible Samples)

For each subsequent eligible sample $R$:


$$e = R - \text{SRTT}_{\text{old}}$$

$$e_c = \operatorname{clamp}\bigl(e,\, -k \cdot \text{RTTVAR}_{\text{old}},\, +k \cdot \text{RTTVAR}_{\text{old}}\bigr)$$

$$\text{SRTT}_{\text{new}} = \text{SRTT}_{\text{old}} + \alpha \, e_c$$

$$\text{RTTVAR}_{\text{new}} = \max\bigl(\text{RTTVAR}_{\min},\, (1 - \beta)\,\text{RTTVAR}_{\text{old}} + \beta \, \vert{}e\vert{}\bigr)$$

* **Parameters:** $\alpha = \frac{1}{8}$, $\beta = \frac{1}{4}$.
* **Reference Profile:** $k = \infty$ (unclamped residual, $e_c = e$), $\text{RTTVAR}_{\min} = 0$.
* **Adaptive Profile:** $k \in [2.0, 3.0]$ (default $k = 2.5$), $\text{RTTVAR}_{\min} = 1\text{ ms}$. Note that $\text{RTTVAR}$ always updates on the **unclamped** absolute residual $\vert{}e\vert{}$ so rising variance expands the clamp envelope naturally.

#### 1.4 Deterministic Bounded Rolling-Window P95 ($T_{95}$)

To respond immediately when latency improves without historical drift, tail latency is tracked via a fixed-capacity ring buffer of the last $W$ eligible RTT samples ($W \in [64, 128]$, default $W = 64$):


$$T_{95} = R_{(\lceil 0.95 \cdot n_W \rceil)}$$


where $n_W = \min(n_{\text{samples}}, W)$ is the number of valid samples currently in the window and $R_{(1)} \le R_{(2)} \le \dots \le R_{(n_W)}$ are the ascending order statistics of the active buffer.

#### 1.5 Reproducible Regime-Change Detector

To prevent permanent clamping lock-out after a sustained step-increase in baseline path latency:

1. **Epoch Representative Sample ($M_e$):** At the close of each observation epoch $e$ containing at least $n_{\min}$ eligible samples (default $n_{\min} = 3$), compute the epoch median:

$$M_e = \operatorname{median}\bigl(\{R_i \in \mathcal{S}_e\}\bigr)$$



(Epochs with fewer than $n_{\min}$ eligible samples preserve the current violation counter without incrementing or resetting it.)
2. **Violation Condition:** Epoch $e$ violates the current baseline if:

$$M_e \gt \text{SRTT} + c \cdot \text{RTTVAR}$$



where $c \in [2.5, 4.0]$ is the deviation threshold multiplier (default $c = 3.0$).
3. **Forced Unclamped Re-Anchoring:** Maintain a consecutive violation counter $v$. Increment $v \leftarrow v + 1$ on each violating epoch and reset $v \leftarrow 0$ on any non-violating epoch. If $v = m$ consecutive violating epochs occur (default $m = 3$, configurable in $[2, 5]$), force an unclamped baseline transition using the violating epoch medians $\{M_{e-m+1}, \dots, M_e\}$:

$$\text{SRTT} \leftarrow M_e, \qquad \text{RTTVAR} \leftarrow \max\left(\text{RTTVAR}_{\min},\, \frac{\vert{}M_e - \text{SRTT}_{\text{old}}\vert{}}{2}\right), \qquad v \leftarrow 0$$



(Alternatively, replay the samples of those $m$ epochs through the unclamped RFC 6298 update equations and set $v \leftarrow 0$.)

---

### 2. Bounded RTO & Retry Backoff

#### 2.1 Base RTO Computation

With clock granularity $G$:


$$\text{RTO}_0 = \operatorname{clamp}\bigl(\text{SRTT} + \max(G,\, 4 \cdot \text{RTTVAR}),\, \text{RTO}_{\min},\, \text{RTO}_{\max}\bigr)$$

* **Reference Profile (RFC 6298):** $\text{RTO}_{\text{init}} = 1.0\text{ s}$, $\text{RTO}_{\min} = 1.0\text{ s}$, $\text{RTO}_{\max} \ge 60.0\text{ s}$ (default $60.0\text{ s}$).
* **Adaptive Profile:** Scanner-specific bounds, e.g., $\text{RTO}_{\text{init}} = 500\text{ ms}$, $\text{RTO}_{\min} \in [50\text{ ms}, 250\text{ ms}]$, $\text{RTO}_{\max} \in [2.0\text{ s}, 4.0\text{ s}]$.

#### 2.2 Retry Sequence Binding & Backoff Equations

When a probe is first transmitted ($n = 0$), it latches the current $\text{RTO}_0$ as its probe-local base timeout $\text{RTO}_{0,\text{probe}}$. Subsequent updates to global $\text{SRTT}$ and $\text{RTTVAR}$ affect only newly dispatched probes.

For retry attempt $n \in \{0, 1, \dots, N_{\max}\}$:

* **Reference Backoff ($2\times$):**

$$\text{RTO}_n = \min\bigl(\text{RTO}_{\max},\, \text{RTO}_{0,\text{probe}} \cdot 2^n\bigr)$$


* **Adaptive Backoff ($\gamma\times$):**

$$\text{RTO}_n = \min\bigl(\text{RTO}_{\max},\, \text{RTO}_{0,\text{probe}} \cdot \gamma^n\bigr), \quad \gamma \in [1.4, 1.6] \text{ (default } 1.5\text{)}$$



#### 2.3 Karn Reset Semantics

If destination-level backoff state is maintained, the backed-off timeout multiplier is cleared and reset to fresh $\text{RTO}_0$ **only** upon receiving an eligible ($n = 0$) acknowledgment.

---

### 3. Epoch-Based AIMD Concurrency Control

#### 3.1 Observation Epoch Definition

Concurrency updates occur at most once per observation epoch of duration:


$$\Delta t_{\text{epoch}} = \operatorname{clamp}\bigl(\text{SRTT},\, E_{\min},\, E_{\max}\bigr)$$


(When $\text{SRTT}$ is uninitialized, $\Delta t_{\text{epoch}} = E_{\text{init}}$, e.g., $100\text{ ms}$, with $E_{\min} = 10\text{ ms}$, $E_{\max} = 1.0\text{ s}$.)

#### 3.2 Per-Epoch State Transitions

Let $C_{\text{win}}$ denote the concurrency window for a controlled scope (per-destination $C_d$ or global $C_{\text{global}}$):

* **Lossless Epoch** (at least one probe completed and zero inferred loss or rate-limit events occurred in the epoch):

$$C_{\text{win,new}} = \min\bigl(C_{\max},\, C_{\text{win,old}} + 1\bigr)$$


* **Inferred-Loss Epoch** (one or more inferred packet-loss or rate-limit events occurred in the epoch; **at most one multiplicative decrease per epoch**):

$$C_{\text{win,new}} = \max\bigl(C_{\min},\, \lfloor C_{\text{win,old}} \cdot \delta \rfloor\bigr), \quad \delta \in [0.4, 0.5] \text{ (default } 0.5\text{)}$$


* **Idle Epoch** (no probes completed and no inferred losses):

$$C_{\text{win,new}} = C_{\text{win,old}}$$



#### 3.3 Dual Ceiling Enforcement

At all times, probe dispatch for destination $d$ is gated by both the per-destination window and the global ceiling:


$$\text{in\_flight}_d \le C_d \le C_{\max,\text{destination}}, \qquad \sum_{d} \text{in\_flight}_d \le C_{\text{global}} \le C_{\max,\text{global}}$$


with $C_{\min,\text{destination}} \ge 1$ and $C_{\min,\text{global}} \ge 1$.

---

### 4. Time-Normalized Classified Drop-Rate EWMA

#### 4.1 Event Classification

Every completed or expired probe is classified into one of four mutually exclusive outcomes:

1. `Success`: Valid response received.
2. `ExpectedSilence`: Unclassified timeout on an unconfirmed/filtered target where silence is normal (excluded from loss indicators).
3. `InferredLoss`: Timeout on a previously responsive host/port, or isolated drop amidst successful probes ($P_{\text{loss}} = 1, P_{\text{rate-limit}} = 0$).
4. `InferredRateLimit`: Explicit rate-limit signal (e.g., ICMP Type 3 Code 13 / 429-equivalent / RST burst) or correlated multi-probe drop spike across an epoch ($P_{\text{loss}} = 0, P_{\text{rate-limit}} = 1$).

#### 4.2 Time-Normalized EWMA Update

To guarantee that the estimator's adaptation rate is invariant to offered load and concurrency, updates are normalized by elapsed time $\Delta t$ since the previous update using characteristic time constant $\tau$ (default $\tau = 1.0\text{ s}$):


$$\lambda(\Delta t) = 1 - \exp\left(-\frac{\Delta t}{\tau}\right)$$

When evaluated once per fixed tick or epoch of duration $\Delta t$ with empirical epoch event rates $\hat{p}_{\text{loss}}, \hat{p}_{\text{rate-limit}} \in [0, 1]$ (where $\hat{p}_{\text{type}} = N_{\text{type}} / N_{\text{eligible\_probes}}$ in that interval):


$$D_{\text{loss,new}} = \bigl(1 - \lambda(\Delta t)\bigr)\, D_{\text{loss,old}} + \lambda(\Delta t)\, \hat{p}_{\text{loss}}$$

$$D_{\text{rate-limit,new}} = \bigl(1 - \lambda(\Delta t)\bigr)\, D_{\text{rate-limit,old}} + \lambda(\Delta t)\, \hat{p}_{\text{rate-limit}}$$

*(Note: For a fixed discrete evaluation tick $\Delta t_0 = \tau \ln(10/9) \approx 0.105\,\tau$, this reduces directly to the constant-weight form $D_{\text{new}} = 0.9\,D_{\text{old}} + 0.1\,\hat{p}$.)*

Both $D_{\text{loss}}$ and $D_{\text{rate-limit}}$ are exported for telemetry and regime diagnostics; neither is multiplied directly into $C_{\text{win}}$ in the base model.

---

### 5. Profile Matrix & Evaluation Contract

| Component | Reference Profile | Adaptive Profile |
| --- | --- | --- |
| **RTT Initialization** | $\text{SRTT}=R_0,\; \text{RTTVAR}=R_0/2$ | $\text{SRTT}=R_0,\; \text{RTTVAR}=\max(R_0/2,\, \text{RTTVAR}_{\min})$ |
| **Karn's Algorithm** | Enforced ($n=0$ only) | Enforced ($n=0$ only) |
| **Steady-State RTT** | Unclamped RFC 6298 ($\alpha=1/8, \beta=1/4$) | Clamped $\text{SRTT}$ ($k \in [2,3]$), unclamped $\text{RTTVAR}$ + floor |
| **Tail Estimator ($T_{95}$)** | Diagnostic rolling P95 ($W=64$) | Active rolling-window P95 ($W=64$) |
| **Change Detection** | Disabled | $m$ consecutive epochs with $\text{median}(R) \gt \text{SRTT} + c\cdot\text{RTTVAR}$ |
| **RTO Bounds** | $\text{RTO}_{\text{init}}=1\text{ s},\; [1\text{ s}, 60\text{ s}]$ | Configurable (e.g., $\text{RTO}_{\text{init}}=500\text{ ms},\; [100\text{ ms}, 3\text{ s}]$) |
| **Retry Backoff** | $\min(\text{RTO}_{\max},\, \text{RTO}_0 \cdot 2^n)$ | $\min(\text{RTO}_{\max},\, \text{RTO}_0 \cdot \gamma^n),\; \gamma \in [1.4, 1.6]$ |
| **Probe RTO Binding** | Latched at $n=0$ transmission | Latched at $n=0$ transmission |
| **Concurrency** | Fixed / caller-supplied ceiling | Epoch-based AIMD ($\delta \in [0.4, 0.5]$) + dual ceilings |
| **Loss Monitoring** | Passive telemetry only | Time-normalized $D_{\text{loss}}$ and $D_{\text{rate-limit}}$ |

**Benchmark Deliverables (Identical Replay Traces & Controlled Synthetic Conditions):**

1. **Estimation Accuracy:** $\text{SRTT}$ RMSE/MAE vs. ground-truth baseline RTT; $\text{RTTVAR}$ calibration error; $T_{95}$ tracking lag after step-up and step-down latency shifts.
2. **Timeout Pathology:** Total timeout frequency, **false-timeout rate** (retransmissions where original $n=0$ reply arrives after $\text{RTO}_0$), and recovery time after transient spikes vs. permanent step shifts.
3. **Throughput & Pacing:** Concurrency utilization ($C_{\text{active}} / C_{\text{optimal}}$), scan completion time, and P95/P99 end-to-end probe completion latency.
4. **Classifier Fidelity:** Precision/recall of $D_{\text{loss}}$ and $D_{\text{rate-limit}}$ against injected network drops, silent filtered ports, and token-bucket rate limiters.

---

### 6. Rust Implementation Contract (Types & Signatures)

The specification maps cleanly onto deterministic, allocation-free Rust structures parameterized by an explicit clock (`Instant` / `Duration`) for trace-driven benchmarking:

```rust
use core::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileMode {
    Reference,
    Adaptive,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    pub mode: ProfileMode,
    pub alpha: f64,                 // 0.125 (1/8)
    pub beta: f64,                  // 0.25  (1/4)
    pub clamp_k: f64,               // [2.0, 3.0] for Adaptive; f64::INFINITY for Reference
    pub rttvar_min: Duration,       // 0ms for Reference; e.g. 1ms for Adaptive
    pub clock_granularity: Duration,// G
    pub rto_init: Duration,         // 1000ms Reference; e.g. 500ms Adaptive
    pub rto_min: Duration,          // 1000ms Reference; e.g. 100ms Adaptive
    pub rto_max: Duration,          // 60s Reference; e.g. 3s Adaptive
    pub backoff_gamma: f64,         // 2.0 Reference; [1.4, 1.6] Adaptive
    pub change_c: f64,              // e.g. 3.0
    pub change_m_epochs: u32,       // e.g. 3
    pub min_epoch_samples: usize,   // e.g. 3
    pub aimd_delta: f64,            // [0.4, 0.5]
    pub c_min_dest: usize,
    pub c_max_dest: usize,
    pub c_min_global: usize,
    pub c_max_global: usize,
    pub ewma_tau: Duration,         // e.g. 1s time constant for classified loss
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// Successful response on initial transmission (n == 0). Eligible for RTT update.
    SuccessInitial { rtt: Duration },
    /// Successful response on retransmitted probe (n >= 1). Karn's rule rejects RTT sample.
    SuccessRetransmitted { attempt: u32 },
    /// Expected silence (e.g., filtered port). Excluded from loss & AIMD backoff.
    ExpectedSilence,
    /// Inferred network packet drop. Triggers AIMD decrease (once per epoch) & D_loss update.
    InferredLoss,
    /// Inferred target/middlebox rate limit. Triggers AIMD decrease & D_rate_limit update.
    InferredRateLimit,
}

/// Latched per-probe timeout schedule created at initial transmission (n == 0).
#[derive(Clone, Copy, Debug)]
pub struct ProbeRetrySchedule {
    pub base_rto: Duration,
    pub rto_max: Duration,
    pub gamma: f64,
}

impl ProbeRetrySchedule {
    #[inline]
    pub fn rto_for_attempt(&self, n: u32) -> Duration {
        let factor = self.gamma.powi(n as i32);
        let scaled = self.base_rto.mul_f64(factor);
        scaled.min(self.rto_max)
    }
}

```

