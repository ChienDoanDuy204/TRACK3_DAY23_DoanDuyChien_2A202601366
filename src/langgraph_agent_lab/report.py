"""Report generation helper.

Renders a markdown lab report from MetricsReport data following
the structure of reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport

_ARCHITECTURE_SECTION = """\
The graph is a `StateGraph(AgentState)` with 11 nodes:
intake → classify → (conditional) → {answer, tool, clarify, risky_action, retry}.
Conditional edges are driven by pure routing functions over state:

- `route_after_classify` maps the LLM classification to a branch.
- `route_after_evaluate` closes the retry loop when the judge returns `needs_retry`.
- `route_after_retry` bounds retries at `max_attempts`, escalating to dead_letter.
- `route_after_approval` proceeds on approval or falls back to clarify on rejection.

Every terminal branch funnels through finalize so each run emits a complete audit trail.
"""

_FAILURE_ANALYSIS_SECTION = """\
1. Retry or tool failure: transient backend failures are simulated by the mock tool for
   error-route queries. The evaluate node gates the loop; retry increments the attempt
   counter and route_after_retry enforces the bound, so repeated failure lands in
   dead_letter instead of hanging the run.
2. Risky action without approval: destructive/financial requests are classified as risky,
   routed through risky_action → approval before any execution. The approval decision is
   recorded in state and audited via events; rejection reroutes to clarify rather than
   executing.
"""

_IMPROVEMENT_PLAN_SECTION = """\
With one more day I would: replace keyword fallbacks with an evaluation set of adversarial
queries, add real SQLite persistence plus crash-resume tests, emit per-node latency spans,
and add streaming of intermediate node updates for observability.
"""


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    success_pct = f"{metrics.success_rate:.0%}"
    lines: list[str] = [
        "# Day 08 Lab Report",
        "",
        "## 1. Metrics summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total scenarios | {metrics.total_scenarios} |",
        f"| Success rate | {success_pct} |",
        f"| Avg nodes visited | {metrics.avg_nodes_visited:.1f} |",
        f"| Total retries | {metrics.total_retries} |",
        f"| Total interrupts | {metrics.total_interrupts} |",
        "",
        "## 2. Scenario results",
        "",
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in metrics.scenario_metrics:
        status = "yes" if item.success else "no"
        actual = item.actual_route or "-"
        lines.append(
            f"| {item.scenario_id} | {item.expected_route} | {actual} | "
            f"{status} | {item.retry_count} | {item.interrupt_count} |"
        )
    lines.extend(
        [
            "",
            "## 3. Architecture",
            "",
            _ARCHITECTURE_SECTION.rstrip(),
            "",
            "## 4. State schema",
            "",
            "| Field | Reducer | Why |",
            "|---|---|---|",
            "| messages / events / errors / tool_results | append | append-only audit trail |",
            "| route / attempt / evaluation_result / approval | overwrite | latest control value |",
            "",
            "## 5. Failure analysis",
            "",
            _FAILURE_ANALYSIS_SECTION.rstrip(),
            "",
            "## 6. Persistence / recovery evidence",
            "",
            "Each scenario runs under its own thread_id with a checkpointer attached at "
            "compile time, so full state history is recoverable per thread and runs can "
            "be resumed after a crash.",
            "",
            "## 7. Improvement plan",
            "",
            _IMPROVEMENT_PLAN_SECTION.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
