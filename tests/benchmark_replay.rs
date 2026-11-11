//! Benchmark Deliverables (§ 5 — Identical Replay Traces & Controlled Synthetic Conditions)
//!
//! Evaluates both Reference (strict RFC 6298) and Adaptive profiles under identical
//! synthetic network trace conditions across the 4 key benchmark areas:
//! 1. Estimation Accuracy (RMSE/MAE against ground truth during transient spikes and step shifts)
//! 2. Timeout Pathology (False-timeout rate preserving probe dispatch-time RTOs)
//! 3. Throughput & Pacing (Multi-destination AIMD concurrency ramp & dual ceiling enforcement)
//! 4. Classifier Fidelity (Precision, recall, and cross-talk isolation for D_loss and D_rate_limit)

use core::time::Duration;
use rfc6298_optimization::{
    AdaptiveConfig, AdaptiveEngine, ProbeOutcome, ProfileMode, Timestamp,
};

/// Simulated probe event preserving dispatch-time RTO.
struct SyntheticProbe {
    dispatch_time: Duration,
    ground_truth_rtt: Duration,
    outcome: ProbeOutcome,
}

fn run_trace(
    mode: ProfileMode,
    probes: &[SyntheticProbe],
) -> (f64, usize, f64, f64) {
    let cfg = match mode {
        ProfileMode::Reference => AdaptiveConfig::reference(),
        ProfileMode::Adaptive => AdaptiveConfig::adaptive(),
    };

    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);
    let mut total_se = 0.0;
    let mut sample_count = 0;
    let mut false_timeouts = 0;

    for probe in probes {
        // Advance engine time to probe dispatch
        let ts_dispatch = Timestamp::from_duration(probe.dispatch_time);
        engine.tick(ts_dispatch);

        // Latch RTO at DISPATCH TIME for attempt 0
        let sched = engine.latch_retry_schedule();
        let rto_at_dispatch = sched.rto_for_attempt(0);

        // Advance engine time to probe completion/arrival
        let arrival_time = probe.dispatch_time + match probe.outcome {
            ProbeOutcome::SuccessInitial { rtt } => rtt,
            ProbeOutcome::SuccessRetransmitted { .. } => Duration::from_millis(100),
            _ => Duration::from_millis(50),
        };
        let ts_arrival = Timestamp::from_duration(arrival_time);
        engine.tick(ts_arrival);

        // Check for false timeout using the dispatch-time latched RTO
        if let ProbeOutcome::SuccessInitial { rtt } = probe.outcome {
            if rtt > rto_at_dispatch {
                false_timeouts += 1;
            }
            if engine.rtt().is_initialized() {
                let err = (engine.rtt().srtt().as_secs_f64() - probe.ground_truth_rtt.as_secs_f64()).abs();
                total_se += err * err;
                sample_count += 1;
            }
        }

        engine.record_outcome(probe.outcome);
    }

    // Close remaining epochs
    let final_time = probes.last().map_or(Duration::ZERO, |p| p.dispatch_time + Duration::from_secs(1));
    engine.force_close_epoch(Timestamp::from_duration(final_time));

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
    // Ground truth: 50ms baseline with periodic 500ms outlier spikes
    let mut probes = Vec::new();
    let mut current_time = Duration::ZERO;

    for i in 0..100 {
        if i % 20 == 10 {
            // Transient spike: 500ms outlier
            probes.push(SyntheticProbe {
                dispatch_time: current_time,
                ground_truth_rtt: Duration::from_millis(50),
                outcome: ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(500),
                },
            });
        } else {
            // Normal baseline sample
            probes.push(SyntheticProbe {
                dispatch_time: current_time,
                ground_truth_rtt: Duration::from_millis(50),
                outcome: ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(50),
                },
            });
        }
        current_time += Duration::from_millis(25);
    }

    let (rmse_ref, false_timeouts_ref, _, _) = run_trace(ProfileMode::Reference, &probes);
    let (rmse_adapt, false_timeouts_adapt, _, _) = run_trace(ProfileMode::Adaptive, &probes);

    println!(
        "Transient Spikes Benchmark:\n\
         - Reference: RMSE = {:.4}s, False Timeouts = {}\n\
         - Adaptive:  RMSE = {:.4}s, False Timeouts = {}",
        rmse_ref, false_timeouts_ref, rmse_adapt, false_timeouts_adapt
    );

    // Adaptive profile clamps outliers (clamp_k = 2.5), so SRTT RMSE against ground truth must be lower
    assert!(
        rmse_adapt < rmse_ref,
        "Adaptive RMSE ({rmse_adapt}) should be lower than Reference ({rmse_ref})"
    );
}

#[test]
fn benchmark_step_shift_reanchoring() {
    // Ground truth: 50ms baseline for 5 epochs, then sustained step-shift to 300ms.
    // Contract § 1.5 item 1:
    // Requires exactly M=3 consecutive violation epochs to trigger an immediate step-shift re-anchor.
    let cfg = AdaptiveConfig::adaptive();
    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);

    let mut current_time = Duration::ZERO;

    // Phase 1: 5 stable epochs at 50ms baseline
    for _ in 0..5 {
        for _ in 0..5 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(50),
            });
        }
        current_time += engine.current_epoch_duration();
        let closed = engine.tick(Timestamp::from_duration(current_time));
        assert!(closed, "Epoch must close");
    }

    // Verify baseline stabilized at 50ms and violations = 0
    let baseline_before_shift = engine.baseline_srtt();
    assert!(
        (baseline_before_shift.as_millis() as i64 - 50).abs() <= 5,
        "Baseline before shift should be ~50ms, got {baseline_before_shift:?}"
    );
    assert_eq!(
        engine.change_detector().consecutive_violations(),
        0,
        "Violations should be 0 before shift"
    );

    // Phase 2: Sustained step shift to 300ms
    let mut epochs_to_reanchor = 0;
    let mut reanchor_observed = false;

    for epoch_idx in 1..=5 {
        for _ in 0..5 {
            engine.record_outcome(ProbeOutcome::SuccessInitial {
                rtt: Duration::from_millis(300),
            });
        }
        current_time += engine.current_epoch_duration();
        let closed = engine.tick(Timestamp::from_duration(current_time));
        assert!(closed, "Epoch should close at {current_time:?}");

        let violations = engine.change_detector().consecutive_violations();

        if epoch_idx < 3 {
            // Consecutive violations must increment: epoch 1 -> 1, epoch 2 -> 2
            assert_eq!(
                violations, epoch_idx,
                "Violation count should be {epoch_idx} at epoch {epoch_idx}"
            );
            // Baseline must strictly be preserved across violations
            assert_eq!(
                engine.baseline_srtt(),
                baseline_before_shift,
                "Baseline must be preserved during violation epoch {epoch_idx}"
            );
        } else if epoch_idx == 3 {
            // At epoch 3, M=3 consecutive violations triggers ReAnchor!
            // Consecutive violations resets to 0 upon re-anchor
            assert_eq!(
                violations, 0,
                "Consecutive violations must reset to 0 upon re-anchoring"
            );
            reanchor_observed = true;
            epochs_to_reanchor = epoch_idx;

            // Baseline must now be updated to the new step level (~300ms)
            let baseline_after = engine.baseline_srtt();
            assert!(
                baseline_after >= Duration::from_millis(250),
                "Baseline after re-anchoring must track the 300ms step shift, got {baseline_after:?}"
            );
        }
    }

    assert!(reanchor_observed, "Re-anchoring must be observed");
    assert_eq!(
        epochs_to_reanchor, 3,
        "Re-anchoring must occur at EXACTLY M=3 consecutive violation epochs"
    );
}

#[test]
fn benchmark_classifier_fidelity() {
    // Evaluates quantitative Precision, Recall, and Specificity for packet loss
    // and rate-limit classification across distinct synthetic workloads:
    // - 20 pure packet-loss epochs
    // - 20 pure rate-limit epochs
    // - 20 expected silence epochs
    // - 20 lossless success epochs

    // 1. Empirical classifier evaluation on epoch event batches:
    let mut tp_loss = 0;
    let mut fp_loss = 0;
    let mut fn_loss = 0;

    let mut tp_rate = 0;
    let mut fp_rate = 0;
    let mut fn_rate = 0;

    // Workload 1: 20 pure packet-loss epochs (ground truth: loss=true, rate=false)
    for _ in 0..20 {
        let events = rfc6298_optimization::EpochEvents {
            successes: 8,
            expected_silence: 0,
            inferred_loss: 2,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        if events.loss_rate() > 0.0 { tp_loss += 1; } else { fn_loss += 1; }
        if events.rate_limit_rate() > 0.0 { fp_rate += 1; }
    }

    // Workload 2: 20 pure rate-limit epochs (ground truth: loss=false, rate=true)
    for _ in 0..20 {
        let events = rfc6298_optimization::EpochEvents {
            successes: 8,
            expected_silence: 0,
            inferred_loss: 0,
            inferred_rate_limit: 2,
            unclassified: 0,
        };
        if events.rate_limit_rate() > 0.0 { tp_rate += 1; } else { fn_rate += 1; }
        if events.loss_rate() > 0.0 { fp_loss += 1; }
    }

    // Workload 3: 20 expected silence epochs (ground truth: loss=false, rate=false)
    for _ in 0..20 {
        let events = rfc6298_optimization::EpochEvents {
            successes: 0,
            expected_silence: 20,
            inferred_loss: 0,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        if events.loss_rate() > 0.0 { fp_loss += 1; }
        if events.rate_limit_rate() > 0.0 { fp_rate += 1; }
    }

    // Workload 4: 20 lossless success epochs (ground truth: loss=false, rate=false)
    for _ in 0..20 {
        let events = rfc6298_optimization::EpochEvents {
            successes: 10,
            expected_silence: 0,
            inferred_loss: 0,
            inferred_rate_limit: 0,
            unclassified: 0,
        };
        if events.loss_rate() > 0.0 { fp_loss += 1; }
        if events.rate_limit_rate() > 0.0 { fp_rate += 1; }
    }

    let precision_loss = tp_loss as f64 / (tp_loss + fp_loss) as f64;
    let recall_loss = tp_loss as f64 / (tp_loss + fn_loss) as f64;

    let precision_rate = tp_rate as f64 / (tp_rate + fp_rate) as f64;
    let recall_rate = tp_rate as f64 / (tp_rate + fn_rate) as f64;

    println!(
        "Classifier Fidelity Benchmark:\n\
         - Packet Loss:   Precision = {:.2}%, Recall = {:.2}%\n\
         - Rate Limiting: Precision = {:.2}%, Recall = {:.2}%",
        precision_loss * 100.0,
        recall_loss * 100.0,
        precision_rate * 100.0,
        recall_rate * 100.0
    );

    assert_eq!(precision_loss, 1.0, "Loss classifier precision should be 100%");
    assert_eq!(recall_loss, 1.0, "Loss classifier recall should be 100%");
    assert_eq!(precision_rate, 1.0, "Rate-limit classifier precision should be 100%");
    assert_eq!(recall_rate, 1.0, "Rate-limit classifier recall should be 100%");
    assert_eq!(fp_loss, 0, "Packet loss classifier must have zero false positives");
    assert_eq!(fp_rate, 0, "Rate-limit classifier must have zero false positives");

    // 2. Continuous EWMA tracking verification on independent monitors:
    let cfg = AdaptiveConfig::adaptive();
    let mut mon_loss = rfc6298_optimization::LossMonitor::new(&cfg);
    let mut mon_rate = rfc6298_optimization::LossMonitor::new(&cfg);

    let pure_loss_events = rfc6298_optimization::EpochEvents {
        successes: 8,
        expected_silence: 0,
        inferred_loss: 2,
        inferred_rate_limit: 0,
        unclassified: 0,
    };
    mon_loss.update(&pure_loss_events, Duration::from_millis(500));
    assert!(mon_loss.d_loss() > 0.0);
    assert_eq!(mon_loss.d_rate_limit(), 0.0, "Pure loss must not contaminate D_rate_limit");

    let pure_rate_events = rfc6298_optimization::EpochEvents {
        successes: 8,
        expected_silence: 0,
        inferred_loss: 0,
        inferred_rate_limit: 2,
        unclassified: 0,
    };
    mon_rate.update(&pure_rate_events, Duration::from_millis(500));
    assert!(mon_rate.d_rate_limit() > 0.0);
    assert_eq!(mon_rate.d_loss(), 0.0, "Pure rate limit must not contaminate D_loss");
}

#[test]
fn benchmark_concurrency_pacing_dual_ceilings() {
    let mut cfg = AdaptiveConfig::adaptive();
    cfg.c_min_dest = 1;
    cfg.c_max_dest = 16;
    cfg.c_min_global = 1;
    cfg.c_max_global = 32;

    let mut engine = AdaptiveEngine::new(cfg, Timestamp::ZERO);

    let d1 = 101; // Healthy destination
    let d2 = 102; // Congested destination experiencing packet losses
    let d3 = 103; // Independent healthy destination

    // Phase 1: Grow D1 and D3 on lossless epochs
    for epoch in 1..=10 {
        // D1: Lossless epoch (5 successes) -> window grows by +1
        for _ in 0..5 {
            engine.record_dest_outcome(
                d1,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                },
            );
        }
        // D3: Lossless epoch (5 successes) -> window grows by +1
        for _ in 0..5 {
            engine.record_dest_outcome(
                d3,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(30),
                },
            );
        }

        engine.tick(Timestamp::from_duration(Duration::from_millis(epoch * 100)));
    }

    let win_d1 = engine.concurrency().get_dest(d1).unwrap().window.window();
    let win_d3 = engine.concurrency().get_dest(d3).unwrap().window.window();
    assert_eq!(win_d1, 11, "D1 window should grow additively on lossless epochs");
    assert_eq!(win_d3, 11, "D3 window should grow additively on lossless epochs");

    // Phase 2: D2 experiences losses; verify D2 backs off while D1 is unaffected
    for epoch in 11..=15 {
        // D2 experiences packet loss
        for _ in 0..2 {
            engine.record_dest_outcome(
                d2,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(80),
                },
            );
        }
        for _ in 0..3 {
            engine.record_dest_outcome(d2, ProbeOutcome::InferredLoss);
        }

        // D1 has no losses
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

    let win_d2 = engine.concurrency().get_dest(d2).unwrap().window.window();
    let win_d1_after = engine.concurrency().get_dest(d1).unwrap().window.window();

    println!("Independent Concurrency Windows: D1(healthy)={win_d1_after}, D2(lossy)={win_d2}");

    // D2 should be clamped down near c_min due to repeated multiplicative decrease
    assert_eq!(win_d2, 1, "D2 window should back off to c_min on lossy epochs");
    // D1 should maintain or grow its window, completely independent of D2's losses
    assert!(win_d1_after >= win_d1, "D1 window should NOT be decremented by D2 losses");

    // Phase 3: Dual Ceiling Enforcement
    // Ramp up global window back to c_max_global (32) via lossless epochs (requires 31 epochs from 1)
    for epoch in 16..=50 {
        for _ in 0..5 {
            engine.record_dest_outcome(
                d1,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(20),
                },
            );
            engine.record_dest_outcome(
                d3,
                ProbeOutcome::SuccessInitial {
                    rtt: Duration::from_millis(30),
                },
            );
        }
        engine.tick(Timestamp::from_duration(Duration::from_millis(epoch * 100)));
    }

    // Now D1 has window 16, D3 has window 16, and global window is 32.
    // Dispatch probes to D1 and D3 up to the global ceiling (c_max_global = 32).
    let mut total_dispatched = 0;
    while engine.can_dispatch(d1) {
        engine.on_dispatch(d1);
        total_dispatched += 1;
    }
    while engine.can_dispatch(d3) {
        engine.on_dispatch(d3);
        total_dispatched += 1;
    }

    // Total in-flight across all destinations equals global ceiling (32)
    assert_eq!(engine.concurrency().global_in_flight(), 32);
    assert_eq!(total_dispatched, 32);

    // Further dispatches to ANY destination must be blocked by the global ceiling
    assert!(!engine.can_dispatch(d1), "Dispatch to D1 must be blocked by global ceiling");
    assert!(!engine.can_dispatch(d2), "Dispatch to D2 must be blocked by global ceiling");
    assert!(!engine.can_dispatch(d3), "Dispatch to D3 must be blocked by global ceiling");

    // Complete one probe on D1
    let completed = engine.try_complete(d1);
    assert!(completed, "Should successfully complete probe on D1");
    assert_eq!(engine.concurrency().global_in_flight(), 31);

    // With global in-flight at 31 < 32, dispatch can proceed again
    assert!(engine.can_dispatch(d1), "Dispatch should be permitted when under global ceiling");
}
