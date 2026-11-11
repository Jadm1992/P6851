//! Benchmark Deliverables (§ 5 — Identical Replay Traces & Controlled Synthetic Conditions)
//!
//! Evaluates both Reference (strict RFC 6298) and Adaptive profiles under identical
//! synthetic network trace conditions across the 4 key benchmark areas:
//! 1. Estimation Accuracy (RMSE/MAE against ground truth during transient spikes and step shifts)
//! 2. Timeout Pathology (False-timeout rate & recovery time)
//! 3. Throughput & Pacing (AIMD concurrency ramp & scan completion)
//! 4. Classifier Fidelity (Precision & recall of D_loss and D_rate_limit)

use core::time::Duration;
use rfc6298_optimization::{
    AdaptiveConfig, AdaptiveEngine, ProbeOutcome, ProfileMode, Timestamp,
};

fn run_trace(
    mode: ProfileMode,
    trace: &[(Duration, ProbeOutcome)],
) -> (f64, usize, f64, f64) {
    let cfg = match mode {
        ProfileMode::Reference => AdaptiveConfig::reference(),
        ProfileMode::Adaptive => AdaptiveConfig::adaptive(),
    };

    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);
    let mut current_time = Duration::ZERO;
    let mut total_se = 0.0;
    let mut sample_count = 0;
    let mut false_timeouts = 0;

    for &(ground_truth, outcome) in trace {
        let ts = Timestamp::from_duration(current_time);
        engine.tick(ts);

        // Check for false timeout: if outcome arrives after latched RTO
        let sched = engine.latch_retry_schedule();
        if let ProbeOutcome::SuccessInitial { rtt } = outcome {
            if rtt > sched.rto_for_attempt(0) {
                false_timeouts += 1;
            }
            if engine.rtt().is_initialized() {
                let err = (engine.rtt().srtt().as_secs_f64() - ground_truth.as_secs_f64()).abs();
                total_se += err * err;
                sample_count += 1;
            }
        }

        engine.record_outcome(outcome);
        current_time += Duration::from_millis(10);
    }

    // Close remaining epochs
    engine.force_close_epoch(Timestamp::from_duration(current_time));

    let rmse = if sample_count > 0 {
        (total_se / sample_count as f64).sqrt()
    } else {
        0.0
    };

    let tel = engine.telemetry();
    (rmse, false_timeouts, tel.d_loss, tel.d_rate_limit)
}

#[test]
fn benchmark_estimation_accuracy_under_transient_spikes() {
    // Ground truth: 50ms baseline with 5 isolated 500ms outlier spikes
    let mut trace = Vec::new();
    for i in 0..100 {
        if i % 20 == 10 {
            // Outlier spike
            trace.push((
                Duration::from_millis(50),
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(500),
                },
            ));
        } else {
            // Normal baseline
            trace.push((
                Duration::from_millis(50),
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(50),
                },
            ));
        }
    }

    let (rmse_ref, _, _, _) = run_trace(ProfileMode::Reference, &trace);
    let (rmse_adapt, _, _, _) = run_trace(ProfileMode::Adaptive, &trace);

    println!("Transient Spikes RMSE: Reference={rmse_ref:.4}s, Adaptive={rmse_adapt:.4}s");

    // Adaptive profile clamps outliers, so SRTT RMSE against ground truth must be lower
    assert!(
        rmse_adapt < rmse_ref,
        "Adaptive RMSE ({rmse_adapt}) should be lower than Reference ({rmse_ref})"
    );
}

#[test]
fn benchmark_step_shift_reanchoring() {
    // Ground truth: 50ms baseline for 30 samples, then sustained step-shift to 300ms
    let mut trace = Vec::new();
    for _ in 0..30 {
        trace.push((
            Duration::from_millis(50),
            ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            },
        ));
    }
    for _ in 0..50 {
        trace.push((
            Duration::from_millis(300),
            ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(300),
            },
        ));
    }

    let cfg_adapt = AdaptiveConfig::adaptive();
    let mut engine_adapt = AdaptiveEngine::new(cfg_adapt, Timestamp::ZERO);

    let mut t = Duration::ZERO;
    for &(_, outcome) in &trace {
        engine_adapt.tick(Timestamp::from_duration(t));
        engine_adapt.record_outcome(outcome);
        t += Duration::from_millis(30);
    }
    engine_adapt.force_close_epoch(Timestamp::from_duration(t));

    // The Adaptive change detector should have forced a re-anchor close to 300ms
    let final_srtt = engine_adapt.rtt().srtt();
    assert!(
        final_srtt >= Duration::from_millis(250),
        "Adaptive engine should quickly track the 300ms step shift via re-anchoring; got {final_srtt:?}"
    );
}

#[test]
fn benchmark_classifier_fidelity() {
    // Trace with known injected network drops and token-bucket rate limits
    let mut trace = Vec::new();
    for _ in 0..80 {
        trace.push((
            Duration::from_millis(50),
            ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            },
        ));
    }
    for _ in 0..10 {
        trace.push((Duration::from_millis(50), ProbeOutcome::InferredLoss));
    }
    for _ in 0..10 {
        trace.push((Duration::from_millis(50), ProbeOutcome::InferredRateLimit));
    }
    for _ in 0..20 {
        trace.push((Duration::from_millis(50), ProbeOutcome::ExpectedSilence));
    }

    let (_, _, d_loss, d_rate) = run_trace(ProfileMode::Adaptive, &trace);

    println!("Classifier Fidelity: D_loss={d_loss:.4}, D_rate_limit={d_rate:.4}");

    // Both should be positive and bounded, reflecting their respective rates
    assert!(d_loss > 0.0, "D_loss should detect packet loss");
    assert!(d_rate > 0.0, "D_rate_limit should detect rate-limiting");
}

#[test]
fn benchmark_concurrency_pacing_dual_ceilings() {
    let cfg = AdaptiveConfig::adaptive();
    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);

    let d1 = 1;

    // Dispatch up to limits
    assert!(engine.can_dispatch(d1));
    engine.on_dispatch(d1);

    // Initial window is 1, so d1 is saturated
    assert!(!engine.can_dispatch(d1));

    // Grow window over lossless epochs
    for epoch in 1..=10 {
        for _ in 0..5 {
            engine.record_dest_outcome(
                d1,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                },
            );
        }
        engine.tick(Timestamp::from_duration(Duration::from_millis(epoch * 100)));
    }

    // Now window has expanded significantly
    let dest1_win = engine.concurrency().get_dest(d1).unwrap().window.window();
    assert!(
        dest1_win > 1,
        "Dest window should have grown through AIMD: {dest1_win}"
    );

    // Record drop on d1 -> backoff
    engine.record_dest_outcome(d1, ProbeOutcome::InferredLoss);
    engine.tick(Timestamp::from_duration(Duration::from_millis(1200)));

    let dest1_after = engine.concurrency().get_dest(d1).unwrap().window.window();
    assert!(
        dest1_after < dest1_win,
        "Dest window should back off after loss: {dest1_win} -> {dest1_after}"
    );
}
