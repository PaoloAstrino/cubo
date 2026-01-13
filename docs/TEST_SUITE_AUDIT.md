# Test suite audit (happy-path vs stress coverage)

This document summarizes a quick quality pass over the test suite, focusing on:
- tests that can pass while checking very little ("happy-path" / "doesn't crash" tests)
- stress/performance coverage

## What already looks strong
- There is meaningful concurrency coverage beyond `tests/stress/`, e.g. index publish/reader churn stress in `tests/indexing/test_publish_stress_concurrency.py`.
- Many unit/integration tests assert concrete invariants (counts, failure rates, retry behavior, etc.).

## Patterns that can hide regressions
These are not always wrong, but they are common sources of false confidence:
- **"No crash" tests with no assertions**: they can pass even if the system silently returns the wrong result.
- **Metrics checks gated behind `if metric > 0`**: if the metric is missing/zero, the test becomes a no-op.
- **Assertions that are always true in practice** (e.g. `>= 0` for a bounded metric).

## Tightened tests (done)
- `tests/stress/test_corrupted_data.py::test_huge_file_limit`
  - Now asserts the oversized file is skipped (no chunks returned) rather than just "no crash".
- `tests/performance/smoke/test_smoke_thresholds.py`
  - Now requires latency and recall metrics to be present and non-trivial, so it fails when retrieval silently breaks.

## Stress/perf coverage inventory (high level)
- `tests/stress/`
  - SQLite concurrent writes to vector store
  - repeated ingestion memory growth (slow, psutil)
  - corrupted input handling
- Other stress-like tests exist outside that folder:
  - `tests/indexing/test_publish_stress_concurrency.py` (concurrent readers while publishing new index versions)

## Suggested next stress tests (highest value gaps)
If you want broader protection against real-world failures, the next best additions are:
1) **API load + backpressure**: concurrent `/api/query` with realistic payloads, asserting latency bounds and no 5xx.
2) **Ingestion of mixed corpus at scale**: hundreds/thousands of mixed filetypes, asserting stable memory and deterministic chunk counts.
3) **Failure-mode stress**: force transient embedding failures/timeouts and assert retries/backoff + clean partial-state rollback.

## Optional tooling
- `tools/audit_tests.py` is a heuristic scanner that flags tests with suspiciously low assertion signal; it is intended to generate a review shortlist.
