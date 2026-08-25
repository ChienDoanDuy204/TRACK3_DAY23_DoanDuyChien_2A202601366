# Day 08 Lab Report

## 1. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.4 |
| Total retries | 3 |
| Total interrupts | 2 |

## 2. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |

## 3. Architecture

The graph is a `StateGraph(AgentState)` with 11 nodes:
intake → classify → (conditional) → {answer, tool, clarify, risky_action, retry}.
Conditional edges are driven by pure routing functions over state:

- `route_after_classify` maps the LLM classification to a branch.
- `route_after_evaluate` closes the retry loop when the judge returns `needs_retry`.
- `route_after_retry` bounds retries at `max_attempts`, escalating to dead_letter.
- `route_after_approval` proceeds on approval or falls back to clarify on rejection.

Every terminal branch funnels through finalize so each run emits a complete audit trail.

## 4. State schema

| Field | Reducer | Why |
|---|---|---|
| messages / events / errors / tool_results | append | append-only audit trail |
| route / attempt / evaluation_result / approval | overwrite | latest control value |

## 5. Failure analysis

1. Retry or tool failure: transient backend failures are simulated by the mock tool for
   error-route queries. The evaluate node gates the loop; retry increments the attempt
   counter and route_after_retry enforces the bound, so repeated failure lands in
   dead_letter instead of hanging the run.
2. Risky action without approval: destructive/financial requests are classified as risky,
   routed through risky_action → approval before any execution. The approval decision is
   recorded in state and audited via events; rejection reroutes to clarify rather than
   executing.

## 6. Persistence / recovery evidence

Each scenario runs under its own thread_id with a checkpointer attached at compile time, so full state history is recoverable per thread and runs can be resumed after a crash.

## 7. Improvement plan

With one more day I would: replace keyword fallbacks with an evaluation set of adversarial
queries, add real SQLite persistence plus crash-resume tests, emit per-node latency spans,
and add streaming of intermediate node updates for observability.
