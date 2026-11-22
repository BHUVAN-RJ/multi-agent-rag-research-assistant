"""Orchestration tests: supervisor routing, failure isolation, recovery, streaming contract.

Workers and the planner are replaced with fakes so these run without keys or network; what is
under test is the graph's routing and failure handling, not the LLMs.
"""

import json

import pytest

import src.graph as graph_module
from src.agents import supervisor as supervisor_module
from src.agents.common import AgentOutput
from src.agents.supervisor import SupervisorPlan, keyword_plan, parse_plan_json, sanitize_plan
from src.trace_events import emit_token


def make_fake_worker(agent_name: str, ok: bool = True, raises: bool = False, call_log: list | None = None):
    async def fake_worker(state: dict, is_final: bool) -> AgentOutput:
        if call_log is not None:
            call_log.append((agent_name, is_final))
        if raises:
            raise RuntimeError(f"{agent_name} exploded")
        if not ok:
            return AgentOutput(agent=agent_name, ok=False, error=f"{agent_name} failed")
        answer_text = f"answer from {agent_name}"
        if is_final:
            emit_token(answer_text, agent_name)
        return AgentOutput(agent=agent_name, ok=True, text=answer_text, provider="fake", model="fake")

    return fake_worker


def install_plan(monkeypatch, plan: list[str]):
    async def fake_make_plan(query: str) -> SupervisorPlan:
        return SupervisorPlan(plan=plan, needs_web=False, reason="test plan", source="llm")

    monkeypatch.setattr(graph_module, "make_plan", fake_make_plan)


def install_workers(monkeypatch, workers: dict):
    monkeypatch.setattr(graph_module, "WORKER_RUNNERS", workers)
    monkeypatch.setattr(graph_module, "compiled_graph", None)


async def collect_events(query: str = "q") -> list[dict]:
    collected = []
    async for payload in graph_module.stream_query(query, "session-1"):
        collected.append(payload)
    return collected


def route_targets(events: list[dict]) -> list[str]:
    targets = []
    for payload in events:
        if payload["type"] == "trace" and payload["data"]["kind"] == "agent_route" and payload["data"]["node"] == "supervisor":
            targets.append(payload["data"]["detail"]["next_agent"])
    return targets


async def test_single_worker_plan_routes_and_streams(monkeypatch):
    call_log = []
    install_plan(monkeypatch, ["data"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", call_log=call_log),
            "summary": make_fake_worker("summary", call_log=call_log),
            "data": make_fake_worker("data", call_log=call_log),
        },
    )
    events = await collect_events()
    assert route_targets(events) == ["data", "finish"]
    assert call_log == [("data", True)]
    token_texts = [payload["data"]["text"] for payload in events if payload["type"] == "token"]
    assert token_texts == ["answer from data"]
    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["answer"] == "answer from data"
    assert events[-1]["data"]["ttft_ms"] is not None


async def test_two_step_plan_only_last_worker_streams(monkeypatch):
    call_log = []
    install_plan(monkeypatch, ["research", "summary"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", call_log=call_log),
            "summary": make_fake_worker("summary", call_log=call_log),
            "data": make_fake_worker("data", call_log=call_log),
        },
    )
    events = await collect_events()
    assert route_targets(events) == ["research", "summary", "finish"]
    assert call_log == [("research", False), ("summary", True)]
    assert events[-1]["data"]["answer"] == "answer from summary"


async def test_raising_worker_does_not_crash_graph_and_recovers(monkeypatch):
    call_log = []
    install_plan(monkeypatch, ["data"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", call_log=call_log),
            "summary": make_fake_worker("summary", call_log=call_log),
            "data": make_fake_worker("data", raises=True, call_log=call_log),
        },
    )
    events = await collect_events()
    # data raised -> supervisor recovers via research, which answers.
    assert route_targets(events) == ["data", "research", "finish"]
    error_traces = [p for p in events if p["type"] == "trace" and p["data"]["kind"] == "error"]
    assert error_traces and "exploded" in error_traces[0]["data"]["detail"]["error"]
    final_answer = events[-1]["data"]["answer"]
    assert final_answer.startswith("answer from research")
    # The user is told which part failed and why, instead of losing it silently.
    assert "the data step failed (RuntimeError: data exploded)" in final_answer
    assert events[-1]["data"]["agents_run"] == ["data", "research"]
    assert events[-1]["data"]["plan"] == ["data"]


async def test_all_workers_failing_yields_explanation_not_exception(monkeypatch):
    install_plan(monkeypatch, ["research"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", ok=False),
            "summary": make_fake_worker("summary", ok=False),
            "data": make_fake_worker("data", ok=False),
        },
    )
    events = await collect_events()
    assert route_targets(events) == ["research", "summary", "finish"]
    final_answer = events[-1]["data"]["answer"]
    assert "couldn't produce" in final_answer
    assert "research failed" in final_answer
    assert not any(payload["type"] == "error" for payload in events)


async def test_failed_middle_step_falls_back_to_earlier_output(monkeypatch):
    install_plan(monkeypatch, ["research", "summary"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research"),
            "summary": make_fake_worker("summary", ok=False),
            "data": make_fake_worker("data"),
        },
    )
    events = await collect_events()
    # research succeeded (not streamed), summary failed -> finish streams research's text.
    token_texts = [payload["data"]["text"] for payload in events if payload["type"] == "token"]
    assert token_texts[0] == "answer from research"
    assert "summary step failed" in token_texts[-1]
    assert events[-1]["data"]["final_agent"] == "research"


def test_parse_plan_json_tolerates_prose_and_fences():
    parsed = parse_plan_json('Sure!\n```json\n{"plan": ["data"], "needs_web": false, "reason": "counts"}\n```')
    assert parsed == {"plan": ["data"], "needs_web": False, "reason": "counts"}
    assert parse_plan_json("no json here") is None


def test_sanitize_plan_drops_unknown_and_duplicates():
    assert sanitize_plan(["Research", "coder", "research", "summary"]) == ["research", "summary"]
    assert sanitize_plan("research") == []


@pytest.mark.parametrize(
    ("query", "expected_plan", "expected_web"),
    [
        ("How many RAG papers were published per year since 2020?", ["data"], False),
        ("Summarize the key ideas of Self-RAG", ["summary"], False),
        ("How does ReAct interleave reasoning and acting?", ["research"], False),
        ("What are the latest developments in AI agents this week?", ["research"], True),
    ],
)
def test_keyword_fallback_classifier(query, expected_plan, expected_web):
    plan = keyword_plan(query, "test")
    assert plan.plan == expected_plan
    assert plan.needs_web is expected_web
    assert plan.source == "keyword_fallback"


async def test_make_plan_falls_back_when_llm_errors(monkeypatch):
    class FailedResult:
        error = "all providers failed"
        text = ""
        provider = ""
        model = ""
        latency_ms = 5.0
        input_tokens = None
        output_tokens = None

    async def failing_complete(*args, **kwargs):
        return FailedResult()

    monkeypatch.setattr(supervisor_module, "complete", failing_complete)
    plan = await supervisor_module.make_plan("Summarize the key ideas of Self-RAG")
    assert plan.source == "keyword_fallback"
    assert plan.plan == ["summary"]


async def test_failed_worker_citations_do_not_enter_state(monkeypatch):
    async def failing_with_citations(state: dict, is_final: bool) -> AgentOutput:
        return AgentOutput(agent="research", ok=False, error="llm down", citations=[{"id": "[1]"}])

    async def summary_checks_numbering(state: dict, is_final: bool) -> AgentOutput:
        # Numbering for the recovery worker must start at 1 again, not after the orphaned [1].
        assert state.get("citations", []) == []
        return AgentOutput(agent="summary", ok=True, text="ok", citations=[{"id": "[1]"}])

    install_plan(monkeypatch, ["research"])
    install_workers(
        monkeypatch,
        {"research": failing_with_citations, "summary": summary_checks_numbering, "data": make_fake_worker("data")},
    )
    final_state = await graph_module.get_graph().ainvoke({"query": "q", "session_id": "s", "supervisor_visits": 0})
    assert final_state["citations"] == [{"id": "[1]"}]
    assert final_state["final_agent"] == "summary"


async def test_planner_usage_reaches_first_route_trace(monkeypatch):
    async def fake_make_plan(query: str) -> SupervisorPlan:
        return SupervisorPlan(
            plan=["data"],
            needs_web=False,
            reason="r",
            source="llm",
            detail={"planner_latency_ms": 12.5, "planner_input_tokens": 40, "planner_output_tokens": 9},
        )

    monkeypatch.setattr(graph_module, "make_plan", fake_make_plan)
    install_workers(
        monkeypatch,
        {"research": make_fake_worker("research"), "summary": make_fake_worker("summary"), "data": make_fake_worker("data")},
    )
    events = await collect_events()
    route_details = [
        p["data"]["detail"] for p in events
        if p["type"] == "trace" and p["data"]["kind"] == "agent_route" and p["data"]["node"] == "supervisor"
    ]
    assert route_details[0]["planner_latency_ms"] == 12.5
    assert route_details[0]["planner_input_tokens"] == 40
    assert "planner_latency_ms" not in route_details[1]


async def test_make_plan_records_planner_usage(monkeypatch):
    class PlanResult:
        error = None
        text = '{"plan": ["research"], "needs_web": false, "reason": "concept question"}'
        provider = "nvidia"
        model = "m"
        latency_ms = 33.0
        input_tokens = 100
        output_tokens = 20

    async def fake_complete(*args, **kwargs):
        return PlanResult()

    monkeypatch.setattr(supervisor_module, "complete", fake_complete)
    plan = await supervisor_module.make_plan("How does ReAct work?")
    assert plan.source == "llm"
    assert plan.detail == {
        "planner_provider": "nvidia",
        "planner_model": "m",
        "planner_latency_ms": 33.0,
        "planner_input_tokens": 100,
        "planner_output_tokens": 20,
    }


@pytest.mark.live
@pytest.mark.parametrize(
    ("query", "expected_first_agent"),
    [
        ("Summarize the key ideas of Self-RAG", "summary"),
        ("How many papers per year are in the database?", "data"),
        ("How does ReAct interleave reasoning and acting?", "research"),
    ],
)
async def test_live_planner_routes_to_the_right_worker(query, expected_first_agent):
    from src.llm.providers import has_api_key

    if not has_api_key("nvidia"):
        pytest.skip("no nvidia key")
    plan = await supervisor_module.make_plan(query)
    if plan.source != "llm":
        pytest.skip(f"planner LLM unavailable: {plan.reason}")
    assert plan.plan[0] == expected_first_agent


async def test_two_independent_workers_both_reach_the_answer(monkeypatch):
    # Live bug: for ["research", "data"] only data's text was returned and research's cited
    # answer was dropped. Both must stream, in order, with a separator.
    call_log = []
    install_plan(monkeypatch, ["research", "data"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", call_log=call_log),
            "summary": make_fake_worker("summary", call_log=call_log),
            "data": make_fake_worker("data", call_log=call_log),
        },
    )
    events = await collect_events()
    assert call_log == [("research", True), ("data", True)]
    token_texts = [payload["data"]["text"] for payload in events if payload["type"] == "token"]
    assert token_texts == ["answer from research", graph_module.ANSWER_SEPARATOR, "answer from data"]
    final_answer = events[-1]["data"]["answer"]
    assert final_answer == f"answer from research{graph_module.ANSWER_SEPARATOR}answer from data"
    assert events[-1]["data"]["final_agent"] == "data"


async def test_worker_before_summary_does_not_stream(monkeypatch):
    call_log = []
    install_plan(monkeypatch, ["data", "summary"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research", call_log=call_log),
            "summary": make_fake_worker("summary", call_log=call_log),
            "data": make_fake_worker("data", call_log=call_log),
        },
    )
    events = await collect_events()
    assert call_log == [("data", False), ("summary", True)]
    assert events[-1]["data"]["answer"] == "answer from summary"


async def test_planner_sub_tasks_reach_each_worker(monkeypatch):
    from src.agents.common import task_for

    seen_tasks = {}

    def make_task_recorder(agent_name):
        async def record_task(state, is_final):
            seen_tasks[agent_name] = task_for(state, agent_name)
            return AgentOutput(agent=agent_name, ok=True, text=agent_name)

        return record_task

    async def fake_make_plan(query):
        return SupervisorPlan(
            plan=["research", "data"], needs_web=False, reason="r", source="llm",
            tasks={"research": "What is RAG-Token?", "data": "How many RAG papers per year?"},
        )

    monkeypatch.setattr(graph_module, "make_plan", fake_make_plan)
    install_workers(
        monkeypatch,
        {"research": make_task_recorder("research"), "summary": make_task_recorder("summary"), "data": make_task_recorder("data")},
    )
    await collect_events("How many RAG papers per year, and what is RAG-Token?")
    assert seen_tasks == {"research": "What is RAG-Token?", "data": "How many RAG papers per year?"}


def test_task_for_falls_back_to_query():
    from src.agents.common import task_for

    two_step = ["research", "data"]
    assert task_for({"query": "q", "plan": two_step, "agent_tasks": {"data": "d"}}, "research") == "q"
    assert task_for({"query": "q", "plan": two_step, "agent_tasks": {"data": "d"}}, "data") == "d"
    assert task_for({"query": "q"}, "data") == "q"


def test_task_for_ignores_the_planner_rewrite_in_a_one_worker_plan():
    # Live bug: "Summarize the Reflexion paper" became "... preserving citations from the arXiv
    # source", and the rewrite no longer matched the paper.
    from src.agents.common import task_for

    state = {"query": "Summarize the Reflexion paper", "plan": ["summary"], "agent_tasks": {"summary": "rewritten"}}
    assert task_for(state, "summary") == "Summarize the Reflexion paper"


@pytest.mark.parametrize(
    ("query", "llm_plan", "expected_plan"),
    [
        ("Summarize the Self-RAG paper", ["research", "summary"], ["summary"]),
        ("Summarize this week's news about agent releases", ["research"], ["summary"]),
        ("TL;DR of https://example.com/post", ["research", "summary"], ["summary"]),
        # A data step is kept: summary cannot run SQL.
        ("Summarize how many papers per year we have", ["data", "summary"], ["data", "summary"]),
        # Not a summary request: untouched.
        ("Explain ReAct", ["research"], ["research"]),
    ],
)
def test_summary_rule(query, llm_plan, expected_plan):
    plan, _ = supervisor_module.enforce_summary_rule(query, llm_plan)
    assert plan == expected_plan


async def test_make_plan_keeps_tasks_only_for_planned_workers(monkeypatch):
    class PlanResult:
        error = None
        text = '{"plan": ["research", "data"], "tasks": {"research": "explain X", "data": "count X", "summary": "stray"}, "needs_web": false, "reason": "both"}'
        provider = "nvidia"
        model = "m"
        latency_ms = 1.0
        input_tokens = 1
        output_tokens = 1

    async def fake_complete(*args, **kwargs):
        return PlanResult()

    monkeypatch.setattr(supervisor_module, "complete", fake_complete)
    plan = await supervisor_module.make_plan("count papers about X and explain X")
    assert plan.plan == ["research", "data"]
    assert plan.tasks == {"research": "explain X", "data": "count X"}


async def test_worker_before_a_later_summary_step_does_not_stream(monkeypatch):
    # Live bug: plan [data, research, summary] printed data's table and then summary's restatement.
    call_log = []
    install_plan(monkeypatch, ["data", "research", "summary"])
    install_workers(
        monkeypatch,
        {name: make_fake_worker(name, call_log=call_log) for name in ("research", "summary", "data")},
    )
    events = await collect_events()
    assert call_log == [("data", False), ("research", False), ("summary", True)]
    token_texts = [payload["data"]["text"] for payload in events if payload["type"] == "token"]
    assert token_texts == ["answer from summary"]


async def test_second_worker_failing_leaves_no_dangling_separator(monkeypatch):
    # The separator used to be sent before the second worker ran, so its failure left a bare "---".
    install_plan(monkeypatch, ["research", "data"])
    install_workers(
        monkeypatch,
        {
            "research": make_fake_worker("research"),
            "summary": make_fake_worker("summary"),
            "data": make_fake_worker("data", ok=False),
        },
    )
    events = await collect_events()
    tokens = [payload["data"] for payload in events if payload["type"] == "token"]
    separator_tokens = [token for token in tokens if token["node"] == graph_module.SEPARATOR_NODE]
    # Exactly one separator: the one in front of the failure note, none from the silent worker.
    assert len(separator_tokens) == 1
    assert tokens[-1]["node"] == "finish" and "data step failed" in tokens[-1]["text"]
    assert events[-1]["data"]["failed_agents"] == ["data"]


async def test_done_frame_names_the_model_that_answered(monkeypatch):
    install_plan(monkeypatch, ["research"])
    install_workers(monkeypatch, {name: make_fake_worker(name) for name in ("research", "summary", "data")})
    events = await collect_events()
    assert events[-1]["data"]["answered_by"] == {"provider": "fake", "model": "fake"}


async def test_first_route_trace_shows_needs_web_and_plan_source(monkeypatch):
    install_plan(monkeypatch, ["research"])
    install_workers(monkeypatch, {name: make_fake_worker(name) for name in ("research", "summary", "data")})
    events = await collect_events()
    first_route = next(p for p in events if p["type"] == "trace" and p["data"]["kind"] == "agent_route")
    assert first_route["data"]["detail"]["needs_web"] is False
    assert first_route["data"]["detail"]["plan_source"] == "llm"


@pytest.mark.parametrize(
    ("query", "llm_plan", "expected_plan"),
    [
        # Live: a web question got a data step that described agent papers as LLM releases.
        ("Which open-weight LLMs were released last month?", ["research", "data"], ["research"]),
        ("Explain ReAct and count the RAG papers per year", ["research", "data"], ["research", "data"]),
        # A data-only plan is kept even without English metadata words (other languages).
        ("¿Cuántos artículos hay por año?", ["data"], ["data"]),
        # A TL;DR asked for anywhere in a multi-part request ends with summary.
        ("Explain ReAct, count RAG papers per year, then give a short TL;DR", ["research", "data"], ["research", "data", "summary"]),
        # Nobody asked for condensing: research answers directly.
        ("How do open-weight LLMs compare on agent benchmarks?", ["research", "summary"], ["research"]),
        ("Find what papers say about tool use and summarize it", ["research", "summary"], ["research", "summary"]),
    ],
)
async def test_plan_rules_drop_unasked_data_and_add_requested_summary(monkeypatch, query, llm_plan, expected_plan):
    plan_json = json.dumps({"plan": llm_plan, "needs_web": False, "reason": "r"})

    class PlanResult:
        error = None
        text = plan_json
        provider = "nvidia"
        model = "m"
        latency_ms = 1.0
        input_tokens = 1
        output_tokens = 1

    async def fake_complete(*args, **kwargs):
        return PlanResult()

    monkeypatch.setattr(supervisor_module, "complete", fake_complete)
    plan = await supervisor_module.make_plan(query)
    assert plan.plan == expected_plan


def test_keyword_fallback_needs_a_metadata_word_for_data():
    assert supervisor_module.keyword_plan("What is the most important idea in ReAct?", "t").plan == ["research"]
    assert supervisor_module.keyword_plan("Which topic has the most papers?", "t").plan == ["data"]


def test_planner_prompt_pins_english_worker_names_and_web_for_off_topic():
    prompt = supervisor_module.SUPERVISOR_SYSTEM_PROMPT
    assert "in English, whatever language the user writes in" in prompt
    assert "General technical questions do not need the web" not in prompt
