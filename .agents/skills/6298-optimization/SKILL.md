---
name: 6298-optimization
description: Execute the complete P6851 Rust implementation against CONTRACT.md, including production code, tests, benchmarks, and Git delivery.
---

# P6851 — Execute Full Implementation

**Repository:** `Jadm1992/P6851`  
**Starting branch:** `master`  
**Authoritative specification:** `CONTRACT.md`  
**Project instructions:** `AGENTS.md`

## Objective

Execute the complete P6851 implementation against the existing repository and frozen contract.

This is an implementation task. Do not write another contract, proposal, architecture document, or implementation plan as a substitute for coding.

## Requirements

1. Inspect the current repository, branch, working tree, source code, tests, dependencies, examples, and CI. Identify completed and missing functionality. Preserve existing work.

2. Implement all remaining requirements in `CONTRACT.md`, maintaining the four-plane architecture:
   - Measurement
   - Classification
   - Control
   - Execution

3. Implement and integrate:
   - `no_std` Sans-I/O core
   - Confidence-based classification
   - P95 and censored-evidence separation
   - Adaptive change detection
   - Hierarchical global/group/destination AIMD
   - Atomic hierarchical pacing
   - ProbePermit lifecycle and retry scheduling
   - Bounded generational arenas
   - Tokio execution adapter

4. Modify actual production code. No disconnected modules, placeholders, fake implementations, or documentation-only changes.

5. Add deterministic unit, integration, regression, property, and chronological trace-replay benchmarks. Compare Reference and Adaptive profiles against the specified baselines.

6. Work incrementally. Test each subsystem, fix failures before proceeding, preserve unrelated work, and follow `AGENTS.md`.

7. Run formatting, full workspace tests, Clippy with warnings denied, `no_std` checks, examples, and benchmarks. Report only checks actually executed.

## Git Delivery

- Work on a dedicated implementation branch.
- Commit and push the actual implementation, tests, and required workspace changes.
- Do not overwrite unrelated work or force-push `master`.
- Do not treat documentation or contract changes as implementation completion.

## Definition of Done

Every requirement in `CONTRACT.md` must be implemented, integrated, and independently tested.

Provide:
- Implementation branch and final commit SHA.
- Files changed.
- Functionality implemented.
- Actual test and verification results.
- Benchmark measurements and baseline comparisons.
- Remaining blockers, if any.

Treat CONTRACT.md as the specification to implement, not as the deliverable. Implement the requirements in the existing source code, add tests, run the required checks, and push the completed implementation. Documentation or contract changes alone do not constitute completion.
