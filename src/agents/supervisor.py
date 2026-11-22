"""Supervisor: classifies the query into an ordered worker plan, then routes on state fields.

The first visit makes one cheap LLM call that returns a JSON plan. Every later visit is
deterministic routing over inspectable state (plan_queue, agent_outputs, failed_agents), so the
trace shows exactly why each worker ran. If the LLM plan is unusable, a keyword classifier takes
over and the trace says so.
"""

import json
import re
from dataclasses import dataclass, field

from src.agents.common import AGENT_NAMES
from src.llm.router import TaskProfile, complete

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent research assistant about LLM agents, multi-agent systems and retrieval-augmented generation (RAG).
Workers:
- research: answers questions from a corpus of ~40 arXiv papers on agents/RAG, plus live web search for recent events.
- summary: summarizes a paper, a URL, or the findings of an earlier worker, preserving citations.
- data: runs SQL over a database of arXiv paper METADATA (titles, authors, dates, categories, topics) for counts, trends, rankings, lists.
Routing rules:
- A request to summarize, give a TL;DR, overview or key ideas of a paper, URL or topic goes to summary (summary retrieves its own sources). Use ["research", "summary"] only when the request first needs new facts found and then condensed.
- Counts, trends, rankings or lists over paper metadata go to data.
- Explanations and questions about how methods work go to research.
Choose the shortest ordered plan of workers that answers the user. Use two workers only when the request clearly needs both (e.g. numbers plus an explanation).
For each worker in the plan, write its own sub-task: the part of the request that worker should answer, as a standalone question. With one worker, the sub-task is the whole request.
Use data only when the user asks for counts, lists, rankings or trends over the paper database; never add it to a question about the outside world.
Set needs_web=true when the question involves recent/current events, news, releases, anything after 2024, a URL, or anything that is not about LLM agents, multi-agent systems or RAG (the corpus covers only those). Questions about methods in those papers do not need the web.
Worker names in "plan" and "tasks" are always exactly research, summary or data, in English, whatever language the user writes in. Write each sub-task in the user's language.
Respond with ONLY a JSON object: {"plan": ["research", "data"], "tasks": {"research": "<sub-task>", "data": "<sub-task>"}, "needs_web": false, "reason": "<one sentence on why>"}"""

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

DATA_KEYWORDS = ("how many", "count", "number of", "per year", "trend", "most ", "top ", "list of", "which authors", "statistics", "rank")
# Words that tie a request to the paper-metadata database. "The most important idea in ReAct"
# has a data keyword but no metadata word, so it is a research question.
METADATA_WORDS = ("paper", "author", "categor", "topic", "database", "arxiv", "publication", "published", "dataset of")
SUMMARY_KEYWORDS = ("summarize", "summarise", "summary", "tl;dr", "tldr", "key ideas", "overview of")
# A request that opens with one of these is a summary request, whatever the planner says.
SUMMARY_REQUEST_PREFIXES = ("summarize", "summarise", "tl;dr", "tldr", "give me a summary", "give me a tl;dr", "summary of")
MAX_TASK_CHARS = 500
WEB_KEYWORDS = ("latest", "recent", "news", "this week", "today", "2025", "2026", "current", "new release", "http://", "https://")


@dataclass
class SupervisorPlan:
    plan: list[str]
    needs_web: bool
    reason: str
    source: str  # "llm" or "keyword_fallback"
    provider: str = ""
    model: str = ""
    detail: dict = field(default_factory=dict)
    # Per-worker sub-question, e.g. {"research": "...", "data": "..."}; empty means use the query.
    tasks: dict = field(default_factory=dict)


def keyword_plan(query: str, why_fallback: str) -> SupervisorPlan:
    lowered = query.lower()
    needs_web = any(keyword in lowered for keyword in WEB_KEYWORDS)
    wants_data = any(keyword in lowered for keyword in DATA_KEYWORDS) and mentions_metadata(query)
    wants_summary = any(keyword in lowered for keyword in SUMMARY_KEYWORDS)
    if wants_data:
        plan = ["data"]
    elif wants_summary:
        plan = ["summary"]
    else:
        plan = ["research"]
    reason = f"keyword classifier ({why_fallback}): data={wants_data}, summary={wants_summary}, web={needs_web}"
    return SupervisorPlan(plan=plan, needs_web=needs_web, reason=reason, source="keyword_fallback")


def parse_plan_json(llm_text: str) -> dict | None:
    match = JSON_OBJECT_PATTERN.search(llm_text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def sanitize_plan(raw_plan: object) -> list[str]:
    if not isinstance(raw_plan, list):
        return []
    valid_steps: list[str] = []
    for step in raw_plan:
        step_name = str(step).strip().lower()
        # Drop unknown names and duplicates; cap at 3 workers per turn.
        if step_name in AGENT_NAMES and step_name not in valid_steps:
            valid_steps.append(step_name)
    return valid_steps[:3]


def sanitize_tasks(raw_tasks: object, plan: list[str]) -> dict[str, str]:
    if not isinstance(raw_tasks, dict):
        return {}
    clean_tasks: dict[str, str] = {}
    for agent_name in plan:
        task_text = str(raw_tasks.get(agent_name, "") or "").strip()
        if task_text:
            clean_tasks[agent_name] = task_text[:MAX_TASK_CHARS]
    return clean_tasks


def mentions_metadata(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in METADATA_WORDS)


def drop_unwanted_data_step(query: str, plan: list[str]) -> tuple[list[str], bool]:
    """Remove a data step the request never asked for.

    Live run: "open-weight LLMs released last month" got ["research", "data"], and data then
    described arXiv agent papers as LLM releases. Only a step that has another worker beside it
    is dropped, so a data-only plan (any language) still reaches the data agent.
    """
    if "data" not in plan or len(plan) == 1 or mentions_metadata(query):
        return plan, False
    return [step for step in plan if step != "data"], True


def add_requested_summary(query: str, plan: list[str]) -> tuple[list[str], bool]:
    """A multi-part request that asks for a TL;DR anywhere ends with the summary agent."""
    lowered = query.lower()
    asks_for_summary = any(keyword in lowered for keyword in SUMMARY_KEYWORDS)
    if not asks_for_summary or "summary" in plan or len(plan) < 2 or len(plan) >= 3:
        return plan, False
    return plan + ["summary"], True


def drop_unrequested_summary(query: str, plan: list[str]) -> tuple[list[str], bool]:
    """Remove a trailing condensation step nobody asked for.

    Live run: "which open-weight LLMs were released last month and how do they compare" got
    ["research", "summary"]; summary re-condensed research's answer, added a model call, and
    lost detail. Without a summary keyword the earlier workers answer directly.
    """
    lowered = query.lower()
    asks_for_summary = any(keyword in lowered for keyword in SUMMARY_KEYWORDS)
    if asks_for_summary or "summary" not in plan or len(plan) == 1:
        return plan, False
    return [step for step in plan if step != "summary"], True


def is_summary_request(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered.startswith(SUMMARY_REQUEST_PREFIXES)


def enforce_summary_rule(query: str, plan: list[str]) -> tuple[list[str], bool]:
    """A request that opens with "Summarize..." goes to summary alone.

    The planner LLM sometimes prepends research ("research then summarize"), which skips the
    summary agent's paper-scoped retrieval and its own web search. Summary gathers its own
    sources, so research adds nothing. Plans that also need data are left alone.
    """
    if not is_summary_request(query) or "data" in plan or plan == ["summary"]:
        return plan, False
    return ["summary"], True


async def make_plan(query: str) -> SupervisorPlan:
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    profile = TaskProfile(complexity="simple", context_chars=len(query))
    llm_result = await complete("supervisor", messages, profile, stream_tokens=False, node="supervisor")
    planner_usage = {
        "planner_provider": llm_result.provider,
        "planner_model": llm_result.model,
        "planner_latency_ms": llm_result.latency_ms,
        "planner_input_tokens": llm_result.input_tokens,
        "planner_output_tokens": llm_result.output_tokens,
    }
    if llm_result.error:
        fallback_plan = keyword_plan(query, f"LLM unavailable: {llm_result.error[:80]}")
        fallback_plan.detail = planner_usage
        return fallback_plan

    parsed = parse_plan_json(llm_result.text)
    if parsed is None:
        fallback_plan = keyword_plan(query, "LLM returned non-JSON plan")
        fallback_plan.detail = planner_usage
        return fallback_plan
    plan = sanitize_plan(parsed.get("plan"))
    if not plan:
        fallback_plan = keyword_plan(query, "LLM plan had no valid workers")
        # The raw plan shows why it was rejected (e.g. worker names translated).
        fallback_plan.detail = {**planner_usage, "rejected_plan": str(parsed.get("plan"))[:200]}
        return fallback_plan

    needs_web_flag = bool(parsed.get("needs_web", False))
    reason_text = str(parsed.get("reason", "")).strip() or "no reason given"
    plan, summary_rule_applied = enforce_summary_rule(query, plan)
    if summary_rule_applied:
        reason_text = f"{reason_text} [summary request -> summary agent only]"
    plan, data_dropped = drop_unwanted_data_step(query, plan)
    if data_dropped:
        reason_text = f"{reason_text} [data dropped: request names no paper metadata]"
    plan, summary_added = add_requested_summary(query, plan)
    if summary_added:
        reason_text = f"{reason_text} [TL;DR requested -> summary added last]"
    plan, summary_dropped = drop_unrequested_summary(query, plan)
    if summary_dropped:
        reason_text = f"{reason_text} [summary dropped: no summary asked for]"
    tasks = sanitize_tasks(parsed.get("tasks"), plan)
    return SupervisorPlan(
        plan=plan,
        needs_web=needs_web_flag,
        reason=reason_text,
        source="llm",
        provider=llm_result.provider,
        model=llm_result.model,
        detail=planner_usage,
        tasks=tasks,
    )


def recovery_agent_for(failed_agent: str, tried_agents: set[str]) -> str | None:
    """When a worker fails, pick a different worker that can still produce a grounded answer."""
    recovery_order = {
        "research": ["summary"],
        "summary": ["research"],
        "data": ["research"],
    }
    for candidate in recovery_order.get(failed_agent, []):
        if candidate not in tried_agents:
            return candidate
    return None
