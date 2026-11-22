"""LangGraph supervisor graph: supervisor -> worker -> supervisor ... -> finish.

Routing between nodes reads only the inspectable state field `next_agent`, which the supervisor
sets from `plan_queue` (its plan) and from worker failures (`failed_agents`). A worker that
raises is converted into a failed AgentOutput, so one worker's failure never crashes the graph.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.common import AgentOutput, report_agent_result
from src.agents.data_agent import run_data
from src.agents.research import run_research
from src.agents.summary import run_summary
from src.agents.supervisor import make_plan, recovery_agent_for
from src.trace_events import TraceEvent, emit_token, emit_trace, hold_prefix, pending_prefix

# Hard stop on supervisor visits per turn: plan of <=3 workers plus recovery attempts.
MAX_SUPERVISOR_VISITS = 6

# Between the streamed answers of two workers in one turn. Emitted under its own node name so a
# provider fallback that discards one worker's partial tokens never discards the separator.
ANSWER_SEPARATOR = "\n\n---\n\n"
SEPARATOR_NODE = "separator"

WORKER_RUNNERS = {
    "research": run_research,
    "summary": run_summary,
    "data": run_data,
}


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    plan: list[str]
    plan_queue: list[str]
    plan_reason: str
    plan_source: str
    needs_web: bool
    agent_tasks: dict[str, str]
    next_agent: str
    last_agent: str
    agent_outputs: dict[str, dict]
    tried_agents: list[str]
    failed_agents: list[str]
    streamed_agents: list[str]
    citations: list[dict]
    supervisor_visits: int
    final_answer: str
    final_agent: str


def emit_route(next_agent: str, reason: str, state_fields: dict[str, Any]) -> None:
    emit_trace(
        TraceEvent(
            kind="agent_route",
            node="supervisor",
            summary=f"supervisor -> {next_agent}",
            detail={"next_agent": next_agent, "reason": reason, **state_fields},
        )
    )


async def supervisor_node(state: AgentState) -> dict:
    visits = state.get("supervisor_visits", 0) + 1
    plan_queue = list(state.get("plan_queue", []))
    tried_agents = list(state.get("tried_agents", []))
    failed_agents = list(state.get("failed_agents", []))
    update: dict[str, Any] = {"supervisor_visits": visits}
    reason = ""
    planner_detail: dict[str, Any] = {}

    if visits == 1:
        supervisor_plan = await make_plan(state["query"])
        plan_queue = list(supervisor_plan.plan)
        update.update(
            {
                "plan": list(supervisor_plan.plan),
                "plan_reason": supervisor_plan.reason,
                "plan_source": supervisor_plan.source,
                "needs_web": supervisor_plan.needs_web,
                "agent_tasks": dict(supervisor_plan.tasks),
            }
        )
        reason = f"plan {supervisor_plan.plan} ({supervisor_plan.source}): {supervisor_plan.reason}"
        planner_detail = {
            **supervisor_plan.detail,
            "tasks": dict(supervisor_plan.tasks),
            "needs_web": supervisor_plan.needs_web,
            "plan_source": supervisor_plan.source,
        }
    else:
        last_agent = state.get("last_agent", "")
        last_failed = last_agent in failed_agents
        any_success = any(output.get("ok") for output in state.get("agent_outputs", {}).values())
        recovery_agent = recovery_agent_for(last_agent, set(tried_agents)) if last_failed else None
        # Recover only when nothing useful exists yet and no remaining plan step can answer.
        if recovery_agent and not any_success and not plan_queue:
            plan_queue.insert(0, recovery_agent)
            reason = f"{last_agent} failed -> recovery via {recovery_agent}"
        elif last_failed:
            reason = f"{last_agent} failed; continuing plan"
        else:
            reason = f"{last_agent} done"

    if visits > MAX_SUPERVISOR_VISITS or not plan_queue:
        next_agent = "finish"
        reason = f"{reason}; plan complete" if reason else "plan complete"
    else:
        next_agent = plan_queue.pop(0)

    emit_route(
        next_agent,
        reason,
        {
            "plan_queue": plan_queue,
            "tried_agents": tried_agents,
            "failed_agents": failed_agents,
            "visit": visits,
            **planner_detail,
        },
    )
    update.update({"plan_queue": plan_queue, "next_agent": next_agent})
    return update


def route_from_supervisor(state: AgentState) -> str:
    return state.get("next_agent", "finish")


def make_worker_node(agent_name: str):
    runner = WORKER_RUNNERS[agent_name]

    async def worker_node(state: AgentState) -> dict:
        # A worker streams its answer to the user unless the next step is summary, which
        # condenses the earlier findings into the answer. So in ["research", "data"] both
        # stream, one after the other, and neither answer is lost.
        remaining_steps = state.get("plan_queue") or []
        # Any later summary step condenses this worker's findings, not only the very next one.
        is_final = "summary" not in remaining_steps
        separator_token = None
        if is_final and state.get("streamed_agents"):
            # Sent with this worker's first token, so a worker that fails silently adds nothing.
            separator_token = hold_prefix(ANSWER_SEPARATOR, SEPARATOR_NODE)
        try:
            output = await runner(state, is_final)
        except Exception as error:  # noqa: BLE001 - a worker crash must not crash the graph
            output = AgentOutput(agent=agent_name, ok=False, error=f"{type(error).__name__}: {error}")
            emit_trace(TraceEvent(kind="error", node=agent_name, summary=f"{agent_name} raised", detail={"error": str(error)}))
        finally:
            if separator_token is not None:
                separator_token["text"] = None
                pending_prefix.set(None)
        report_agent_result(output)

        agent_outputs = dict(state.get("agent_outputs", {}))
        agent_outputs[agent_name] = asdict(output)
        tried_agents = list(state.get("tried_agents", [])) + [agent_name]
        failed_agents = list(state.get("failed_agents", []))
        if not output.ok:
            failed_agents.append(agent_name)
        streamed_agents = list(state.get("streamed_agents", []))
        if is_final and output.ok:
            streamed_agents.append(agent_name)
        citations = list(state.get("citations", []))
        if output.ok:
            citations.extend(output.citations)
        return {
            "agent_outputs": agent_outputs,
            "tried_agents": tried_agents,
            "failed_agents": failed_agents,
            "streamed_agents": streamed_agents,
            "citations": citations,
            "last_agent": agent_name,
        }

    worker_node.__name__ = f"{agent_name}_node"
    return worker_node


# A worker error shown to the user is cut to this length.
FAILURE_NOTE_ERROR_CHARS = 160


def compose_failed_steps_note(state: AgentState) -> str:
    """One line per planned worker that failed, for an answer that other workers still gave."""
    agent_outputs = state.get("agent_outputs", {})
    note_lines = []
    for agent_name in state.get("failed_agents", []):
        error_text = (agent_outputs.get(agent_name, {}).get("error") or "no output")[:FAILURE_NOTE_ERROR_CHARS]
        note_lines.append(f"_Note: the {agent_name} step failed ({error_text}), so this answer does not cover its part._")
    return "\n\n".join(note_lines)


def compose_failure_message(state: AgentState) -> str:
    error_lines = []
    for agent_name, output in state.get("agent_outputs", {}).items():
        error_lines.append(f"- **{agent_name}**: {output.get('error') or 'no output'}")
    errors_block = "\n".join(error_lines) or "- no worker ran"
    return f"Sorry, I couldn't produce a grounded answer this time. What went wrong:\n{errors_block}\n\nPlease try again in a moment."


async def finish_node(state: AgentState) -> dict:
    tried_agents = state.get("tried_agents", [])
    agent_outputs = state.get("agent_outputs", {})
    streamed_agents = state.get("streamed_agents", [])
    successful_agents = []
    for agent_name in tried_agents:
        output = agent_outputs.get(agent_name, {})
        if output.get("ok") and output.get("text") and agent_name not in successful_agents:
            successful_agents.append(agent_name)

    final_agent = successful_agents[-1] if successful_agents else ""
    # Every streamed answer is part of the reply, in the order the user saw it.
    answer_parts = [agent_outputs[agent_name]["text"] for agent_name in successful_agents if agent_name in streamed_agents]
    if not final_agent:
        final_answer = compose_failure_message(state)
        emit_token(final_answer, "finish")
    else:
        if final_agent not in streamed_agents:
            # The answering worker didn't stream (the summary step after it failed); send it now.
            if answer_parts:
                emit_token(ANSWER_SEPARATOR, SEPARATOR_NODE)
            final_text = agent_outputs[final_agent]["text"]
            emit_token(final_text, final_agent)
            answer_parts.append(final_text)
        failed_steps_note = compose_failed_steps_note(state)
        if failed_steps_note:
            # The user should see that part of the request went unanswered, and why.
            emit_token(ANSWER_SEPARATOR, SEPARATOR_NODE)
            emit_token(failed_steps_note, "finish")
            answer_parts.append(failed_steps_note)
        final_answer = ANSWER_SEPARATOR.join(answer_parts)

    emit_trace(
        TraceEvent(
            kind="agent_route",
            node="finish",
            summary=f"answer from {final_agent or 'none'}",
            detail={"final_agent": final_agent, "tried_agents": tried_agents, "failed_agents": state.get("failed_agents", [])},
        )
    )
    return {"final_answer": final_answer, "final_agent": final_agent}


def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("supervisor", supervisor_node)
    for agent_name in WORKER_RUNNERS:
        graph_builder.add_node(agent_name, make_worker_node(agent_name))
        graph_builder.add_edge(agent_name, "supervisor")
    graph_builder.add_node("finish", finish_node)

    route_targets = {agent_name: agent_name for agent_name in WORKER_RUNNERS}
    route_targets["finish"] = "finish"
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges("supervisor", route_from_supervisor, route_targets)
    graph_builder.add_edge("finish", END)
    return graph_builder.compile()


compiled_graph = None


def get_graph():
    global compiled_graph
    if compiled_graph is None:
        compiled_graph = build_graph()
    return compiled_graph


async def stream_query(query: str, session_id: str) -> AsyncIterator[dict]:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    final_state: dict = {}
    initial_state: AgentState = {"query": query, "session_id": session_id, "supervisor_visits": 0}

    try:
        graph = get_graph()
        stream = graph.astream(initial_state, stream_mode=["custom", "values"])
        async for stream_mode, chunk in stream:
            if stream_mode == "values":
                final_state = chunk
                continue
            if chunk.get("type") == "token" and first_token_at is None:
                first_token_at = time.perf_counter()
            yield chunk
    except Exception as error:  # noqa: BLE001 - surface any orchestration failure to the client
        yield {"type": "error", "data": {"message": f"{type(error).__name__}: {error}"}}

    total_ms = (time.perf_counter() - started_at) * 1000
    ttft_ms = (first_token_at - started_at) * 1000 if first_token_at else None
    final_agent = final_state.get("final_agent", "")
    final_output = final_state.get("agent_outputs", {}).get(final_agent, {})
    yield {
        "type": "done",
        "data": {
            "answer": final_state.get("final_answer", ""),
            "final_agent": final_agent,
            # The model that wrote the last answer, so a fallback is visible without the trace.
            "answered_by": {"provider": final_output.get("provider", ""), "model": final_output.get("model", "")},
            "plan": final_state.get("plan", []),
            # What actually ran, recovery workers included; `plan` is what the supervisor planned.
            "agents_run": final_state.get("tried_agents", []),
            "failed_agents": final_state.get("failed_agents", []),
            "latency_ms": total_ms,
            "ttft_ms": ttft_ms,
        },
    }
