"""Graph construction.

This module is intentionally import-safe. It imports LangGraph only inside the builder so unit tests
that check schema/metrics can run even if students are still debugging graph wiring.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState


def build_graph(checkpointer: Any | None = None) -> Any:  # noqa: ANN401
    """Build and compile the LangGraph workflow.

    Architecture:

    START → intake → classify → [conditional: route_after_classify]
      simple       → answer → finalize → END
      tool         → tool → evaluate → [conditional: route_after_evaluate]
                                           success → answer → finalize → END
                                           needs_retry → retry → [conditional: route_after_retry]
                                                                   tool (retry)
                                                                   dead_letter → finalize → END
      missing_info → clarify → finalize → END
      risky        → risky_action → approval → [conditional: route_after_approval]
                                                  approved → tool → evaluate → ...
                                                  rejected → clarify → finalize → END
      error        → retry → [conditional: route_after_retry] → ...
    """
    from langgraph.graph import END, START, StateGraph

    from . import nodes
    from .routing import (
        route_after_approval,
        route_after_classify,
        route_after_evaluate,
        route_after_retry,
    )

    builder = StateGraph(AgentState)

    builder.add_node("intake", nodes.intake_node)
    builder.add_node("classify", nodes.classify_node)
    builder.add_node("tool", nodes.tool_node)
    builder.add_node("evaluate", nodes.evaluate_node)
    builder.add_node("answer", nodes.answer_node)
    builder.add_node("clarify", nodes.ask_clarification_node)
    builder.add_node("risky_action", nodes.risky_action_node)
    builder.add_node("approval", nodes.approval_node)
    builder.add_node("retry", nodes.retry_or_fallback_node)
    builder.add_node("dead_letter", nodes.dead_letter_node)
    builder.add_node("finalize", nodes.finalize_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("tool", "evaluate")
    builder.add_conditional_edges("evaluate", route_after_evaluate)
    builder.add_edge("risky_action", "approval")
    builder.add_conditional_edges("approval", route_after_approval)
    builder.add_conditional_edges("retry", route_after_retry)
    for node in ("answer", "clarify", "dead_letter"):
        builder.add_edge(node, "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
