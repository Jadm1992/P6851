//! RFC 6298 Optimization — Full Implementation
//!
//! Implements the Final Mathematical Contract (v3 — Frozen):
//! - Outlier-resistant RTT estimator with Karn's algorithm
//! - Deterministic rolling-window P95 tail tracker
//! - Epoch-median regime-change detector
//! - Bounded RTO with configurable retry backoff
//! - Epoch-based AIMD concurrency control (dual ceiling)
//! - Time-normalized classified drop-rate EWMA

#![cfg_attr(not(feature = "std"), no_std)]

pub mod config;
pub mod rtt;
pub mod p95;
pub mod change_detector;
pub mod rto;
pub mod concurrency;
pub mod loss;
pub mod engine;

pub use config::*;
pub use rtt::RttEstimator;
pub use p95::RollingP95;
pub use change_detector::ChangeDetector;
pub use rto::ProbeRetrySchedule;
pub use concurrency::{AimdWindow, ConcurrencyController};
pub use loss::LossMonitor;
pub use engine::AdaptiveEngine;
