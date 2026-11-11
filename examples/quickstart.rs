//! Quickstart Example: Testing and Understanding the Adaptive RFC 6298 Engine
//!
//! Run with:
//!     cargo run --example quickstart
//!
//! Demonstrates:
//! 1. Initializing the engine with Adaptive vs Reference profiles.
//! 2. Multi-destination RTT isolation (fast local server vs slow remote server).
//! 3. Outlier resistance (clamping a transient 500ms latency spike).
//! 4. Dual-ceiling AIMD concurrency control and independent backoff on packet loss.
//! 5. Inspecting real-time telemetry (SRTT, RTTVAR, RTO, Concurrency Window, Loss EWMA).

use core::time::Duration;
use rfc6298_optimization::{
    AdaptiveConfig, AdaptiveEngine, ProbeOutcome, Timestamp,
};

fn main() {
    println!("============================================================");
    println!(" RFC 6298 Optimization Engine — Quickstart Walkthrough");
    println!("============================================================\n");

    // ------------------------------------------------------------------------
    // Step 1: Initialize Engine with Adaptive Profile
    // ------------------------------------------------------------------------
    let cfg = AdaptiveConfig::adaptive();
    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);
    let mut clock = Duration::ZERO;

    println!("1. Initialized Engine:");
    println!("   - Profile:          Adaptive (Scanner-Optimized)");
    println!("   - RTO Bounds:       {:?} .. {:?}", cfg.rto_min, cfg.rto_max);
    println!("   - Initial RTO:      {:?}", engine.current_rto());
    println!("   - Backoff Factor γ: {}\n", cfg.backoff_gamma);

    // ------------------------------------------------------------------------
    // Step 2: Multi-Destination Isolation
    // ------------------------------------------------------------------------
    println!("2. Simulating Probing across Two Different Destinations:");
    let local_dest = 101;   // Fast LAN server (~20ms RTT)
    let remote_dest = 102;  // Slow WAN server (~200ms RTT)

    // Feed 5 samples to Local Destination (~20ms)
    println!("   Sending 5 probes to Local Server (20ms)...");
    for _ in 0..5 {
        if engine.can_dispatch(local_dest) {
            engine.on_dispatch(local_dest);
            engine.record_dest_outcome(
                local_dest,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                },
            );
        }
    }

    // Feed 5 samples to Remote Destination (~200ms)
    println!("   Sending 5 probes to Remote Server (200ms)...");
    for _ in 0..5 {
        if engine.can_dispatch(remote_dest) {
            engine.on_dispatch(remote_dest);
            engine.record_dest_outcome(
                remote_dest,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(200),
                },
            );
        }
    }

    // Latch retry schedules for each destination
    let sched_local = engine.latch_dest_retry_schedule(local_dest);
    let sched_remote = engine.latch_dest_retry_schedule(remote_dest);

    let local_srtt = engine.dest_rtt(local_dest).unwrap().srtt();
    let remote_srtt = engine.dest_rtt(remote_dest).unwrap().srtt();

    println!("   -> Local Server:  SRTT = {:?}, Base Timeout (RTO) = {:?}", local_srtt, sched_local.rto_for_attempt(0));
    println!("   -> Remote Server: SRTT = {:?}, Base Timeout (RTO) = {:?}", remote_srtt, sched_remote.rto_for_attempt(0));
    println!("   [Notice: The slow server did NOT inflate the local server's timeout!]\n");

    // ------------------------------------------------------------------------
    // Step 3: Outlier Resistance (Clamped Residual)
    // ------------------------------------------------------------------------
    println!("3. Injecting an Outlier Spike on Local Server:");
    println!("   Local server baseline is ~20ms. Suddenly, a transient spike of 600ms occurs.");
    engine.record_dest_outcome(
        local_dest,
        ProbeOutcome::SuccessInitial {
            rtt: Duration::from_millis(600),
        },
    );

    let srtt_after_spike = engine.dest_rtt(local_dest).unwrap().srtt();
    let rto_after_spike = engine.latch_dest_retry_schedule(local_dest).rto_for_attempt(0);

    println!("   -> Local SRTT after spike: {:?} (Clamped by 2.5 * RTTVAR)", srtt_after_spike);
    println!("   -> Local RTO after spike:  {:?}", rto_after_spike);
    println!("   [In standard RFC 6298, this spike would have heavily polluted SRTT to ~92ms!]\n");

    // ------------------------------------------------------------------------
    // Step 4: Concurrency Pacing & Dual-Ceiling AIMD
    // ------------------------------------------------------------------------
    println!("4. Demonstrating AIMD Concurrency Adaptation over Epochs:");
    println!("   - Local Server continues with 100% success (additive increase).");
    println!("   - Remote Server begins dropping packets (multiplicative decrease).");

    for epoch in 1..=5 {
        // Local: 5 successes per epoch
        for _ in 0..5 {
            engine.record_dest_outcome(
                local_dest,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                },
            );
        }

        // Remote: 1 success and 3 packet losses per epoch
        engine.record_dest_outcome(
            remote_dest,
            ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(200),
            },
        );
        for _ in 0..3 {
            engine.record_dest_outcome(remote_dest, ProbeOutcome::InferredLoss);
        }

        // Advance time to close the epoch
        clock += Duration::from_millis(100);
        engine.tick(Timestamp::from_duration(clock));

        let win_local = engine.concurrency().get_dest(local_dest).unwrap().window.window();
        let win_remote = engine.concurrency().get_dest(remote_dest).unwrap().window.window();

        println!(
            "   Epoch {}: Local Window = {} (growing), Remote Window = {} (backed off to floor)",
            epoch, win_local, win_remote
        );
    }

    println!("\n5. Telemetry Snapshot:");
    let tel = engine.telemetry();
    println!("   - Global In-Flight:    {}/{}", engine.concurrency().global_in_flight(), cfg.c_max_global);
    println!("   - Loss Rate EWMA (D):  {:.2}%", tel.d_loss * 100.0);
    println!("   - Rate Limit EWMA:     {:.2}%", tel.d_rate_limit * 100.0);
    println!("   - Rolling P95 Tail:    {:?}", engine.p95_tracker().p95());
    println!("\n============================================================");
    println!(" Walkthrough Completed Successfully!");
    println!("============================================================");
}
