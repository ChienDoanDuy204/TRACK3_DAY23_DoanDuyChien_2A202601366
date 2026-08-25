"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event


class Classification(BaseModel):
    """Structured LLM output for intent classification."""

    route: Route
    reasoning: str = Field(default="short justification for the classification")


class Evaluation(BaseModel):
    """Structured LLM output for tool-result evaluation (LLM-as-judge)."""

    verdict: str  # "success" | "needs_retry"
    reasoning: str = ""


def _response_text(response: object) -> str:
    """Extract plain text from an LLM response across provider message formats."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        ).strip()
    return str(content).strip()


def _llm_text(prompt: str) -> str | None:
    """Invoke the configured LLM and return its text, or None when unavailable."""
    try:
        return _response_text(get_llm().invoke(prompt))
    except Exception:
        return None


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Student implementations ─────────────────────────────────────────


_FALLBACK_KEYWORDS: dict[str, tuple[str, ...]] = {
    Route.RISKY.value: ("delete", "refund", "cancel account", "drop ", "transfer money", "payout"),
    Route.TOOL.value: ("lookup", "order status", "search", "find order", "check inventory"),
    Route.MISSING_INFO.value: ("fix it", "can you fix", "that thing", "handle it"),
    Route.ERROR.value: ("timeout", "error", "failure", "exception", "crash", "unavailable"),
}


def _heuristic_route(query: str) -> str:
    """Keyword fallback used only when the LLM call itself fails."""
    lowered = query.lower()
    for route, keywords in _FALLBACK_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return route
    return Route.SIMPLE.value


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "")
    prompt = (
        "You are an intent classifier for a customer-support agent.\n"
        "Classify the user request into exactly one category:\n"
        "- simple: general question answerable directly, no tools or state changes needed\n"
        "- tool: requires looking up or acting on data through a tool/system\n"
        "- missing_info: too vague or incomplete to act on; needs clarification first\n"
        "- risky: destructive, financial, security-sensitive, or irreversible action "
        "requiring human approval\n"
        "- error: describes a system failure/timeout that needs retry handling\n"
        f"Priority guide when ambiguous: risky > tool > missing_info > error > simple.\n\n"
        f"User request: {query}"
    )
    route_value: str | None = None
    reasoning = ""
    try:
        structured_llm = get_llm().with_structured_output(Classification)
        result: Classification | None = structured_llm.invoke(prompt)
    except Exception:
        result = None
    valid_routes = {r.value for r in Route}
    if result is not None and str(result.route) in valid_routes:
        route_value = str(result.route)
        reasoning = result.reasoning
    else:
        route_value = _heuristic_route(query)

    risk_level = "high" if route_value == Route.RISKY.value else "low"
    return {
        "route": route_value,
        "risk_level": risk_level,
        "messages": [f"classify:{route_value}"],
        "events": [
            make_event(
                "classify", "completed", f"classified route={route_value}", reasoning=reasoning
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, simulating transient failures for error routes."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient backend failure while processing (attempt {attempt + 1})"
        event_type = "transient_failure"
    else:
        result = "SUCCESS: tool executed — order status=shipped, carrier=UPS, eta=2026-08-28"
        event_type = "completed"
    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result; heuristic gate with optional LLM-as-judge."""
    tool_results = state.get("tool_results") or []
    latest = tool_results[-1] if tool_results else ""

    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"

    prompt = (
        "You are an evaluator judging whether a tool result satisfies the user's request.\n"
        f"User request: {state.get('query', '')}\n"
        f"Tool result: {latest}\n"
        'Reply with JSON: {"verdict": "success"} if usable, else {"verdict": "needs_retry"}.'
    )
    try:
        judge = get_llm().with_structured_output(Evaluation).invoke(prompt)
    except Exception:
        judge = None
    if judge is not None and judge.verdict in ("success", "needs_retry"):
        evaluation_result = judge.verdict

    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [make_event("evaluate", "completed", f"evaluation={evaluation_result}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate the final grounded response with a real LLM call."""
    tool_results = state.get("tool_results") or []
    approval = state.get("approval")
    context_lines = [
        f"- User request: {state.get('query', '')}",
        f"- Tool results: {tool_results if tool_results else 'none'}",
        f"- Approval decision: {approval if approval else 'not applicable'}",
    ]
    prompt = (
        "You are a helpful, precise customer-support agent.\n"
        "Write a concise final answer to the user grounded ONLY in the context below.\n"
        "Never invent facts that are absent from the context.\n\n"
        + "\n".join(context_lines)
    )
    answer = _llm_text(prompt)
    if not answer:
        # Graceful degradation: the LLM call above is the required path; if the provider
        # is unreachable we still answer strictly from recorded context, never invented.
        answer = _grounded_fallback_answer(
            query=state.get("query", ""), tool_results=tool_results, approval=approval
        )
    return {
        "final_answer": answer,
        "pending_question": None,
        "messages": ["answer:generated"],
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def _grounded_fallback_answer(
    query: str,
    tool_results: list[str],
    approval: dict[str, Any] | None,
) -> str:
    """Compose an answer strictly from recorded state when the LLM is unavailable."""
    parts = [f"Regarding '{query}':"]
    if tool_results:
        parts.append(f"Tool result: {tool_results[-1]}")
    if approval is not None:
        verdict = "approved" if approval.get("approved") else "rejected"
        reviewer = approval.get("reviewer", "reviewer")
        parts.append(f"The risky action was reviewed ({verdict} by {reviewer}).")
    parts.append("(LLM generation unavailable — answered from recorded workflow context.)")
    return " ".join(parts)


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    prompt = (
        "The following support request is too vague to act on safely.\n"
        f"Request: {state.get('query', '')}\n"
        "Ask ONE specific clarifying question naming exactly what detail you need."
    )
    question = _llm_text(prompt) or (
        f"Could you provide more details about '{state.get('query', '')}'? "
        "For example: which system/order/account does this concern, and what outcome do you expect?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:question_sent"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    proposed_action = (
        f"Proposed action: execute '{state.get('query', '')}'. "
        "This action is destructive/financial/security-sensitive "
        "and irreversible once executed, so it requires explicit human approval."
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "awaiting_approval", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step (mock by default, interrupt() when enabled)."""
    proposed_action = state.get("proposed_action", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": proposed_action, "requires": "human approval"})
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        comment = decision.get("comment", "") if isinstance(decision, dict) else str(decision)
        approval = ApprovalDecision(approved=approved, reviewer="human", comment=str(comment))
    else:
        approval = ApprovalDecision(approved=True, comment="auto-approved (mock reviewer)")
    payload = approval.model_dump()
    return {
        "approval": payload,
        "messages": [f"approval:{'approved' if payload['approved'] else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval decided: approved={payload['approved']}",
                reviewer=payload["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a bounded retry attempt after a transient failure."""
    attempt = state.get("attempt", 0) + 1
    error_message = f"transient failure recorded; scheduling retry {attempt}"
    return {
        "attempt": attempt,
        "errors": [error_message],
        "messages": [f"retry:{attempt}"],
        "events": [
            make_event("retry", "retrying", error_message, attempt=attempt),
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    errors = state.get("errors") or []
    final_answer = (
        f"This request could not be completed after {max_attempts} attempt(s) "
        f"(last attempt: {attempt}). Logged to the dead-letter queue for operator review. "
        f"Recorded failures: {'; '.join(errors[-3:]) if errors else 'unknown failure'}."
    )
    return {
        "final_answer": final_answer,
        "messages": ["dead_letter:escalated"],
        "events": [make_event("dead_letter", "escalated", "max retries exceeded; escalated")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    route = state.get("route", "")
    return {
        "messages": [f"finalize:{route}"],
        "events": [make_event("finalize", "completed", "workflow finished", route=route)],
    }
