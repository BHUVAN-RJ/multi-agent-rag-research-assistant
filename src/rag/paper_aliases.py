"""Names people use for the corpus papers, mapped to arXiv ids.

The MiniLM embedder does not know most paper names: "What is ColBERT late interaction?" returns
no ColBERT chunk at all, and many titles carry no short name ("Corrective Retrieval Augmented
Generation" is CRAG). Retrieval and the summary agent look names up here instead.
"""

import re

# Names people use for corpus papers, mapped to the paper's arXiv id. Many titles carry no short
# name ("Corrective Retrieval Augmented Generation" is CRAG), so a title match alone misses them,
# and a match against retrieved chunks misses any paper the embedding search did not surface.
PAPER_ALIASES: dict[str, str] = {
    "chain-of-thought prompting": "2201.11903",
    "chain of thought prompting": "2201.11903",
    "self-consistency": "2203.11171",
    "tree of thoughts": "2305.10601",
    "react": "2210.03629",
    "reflexion": "2303.11366",
    "self-ask": "2210.03350",
    "compositionality gap": "2210.03350",
    "toolformer": "2302.04761",
    "hugginggpt": "2303.17580",
    "gorilla": "2305.15334",
    "toolllm": "2307.16789",
    "tool learning with foundation models": "2304.08354",
    "generative agents": "2304.03442",
    "voyager": "2305.16291",
    "memgpt": "2310.08560",
    "swe-agent": "2405.15793",
    "autogen": "2308.08155",
    "metagpt": "2308.00352",
    "camel": "2303.17760",
    "chatdev": "2307.07924",
    "agentverse": "2308.10848",
    "multiagent debate": "2305.14325",
    "multi-agent debate": "2305.14325",
    "agentbench": "2308.03688",
    "webarena": "2307.13854",
    "swe-bench": "2310.06770",
    "mt-bench": "2306.05685",
    "llm-as-a-judge": "2306.05685",
    "dpr": "2004.04906",
    "dense passage retrieval": "2004.04906",
    "colbert": "2004.12832",
    "realm": "2002.08909",
    "retro": "2112.04426",
    "atlas": "2208.03299",
    "rag survey": "2312.10997",
    "self-rag": "2310.11511",
    "raptor": "2401.18059",
    "hyde": "2212.10496",
    "crag": "2401.15884",
    "corrective rag": "2401.15884",
    "graphrag": "2404.16130",
    "graph rag": "2404.16130",
    "flare": "2305.06983",
    "ircot": "2212.10509",
    "ragas": "2309.15217",
    "lost in the middle": "2307.03172",
}

# Longest alias first, so "rag survey" wins over shorter names inside the same request.
ALIAS_PATTERNS = [
    (re.compile(rf"(?<![\w-]){re.escape(alias)}(?![\w-])"), source_id)
    for alias, source_id in sorted(PAPER_ALIASES.items(), key=lambda item: -len(item[0]))
]


# A request rarely names more papers than this; more would crowd the re-rank pool.
MAX_NAMED_PAPERS = 3


def find_aliased_papers(query: str) -> list[str]:
    """arXiv ids of every corpus paper the query names, longest name first, without repeats."""
    lowered_query = query.lower()
    named_ids: list[str] = []
    for alias_pattern, source_id in ALIAS_PATTERNS:
        if source_id not in named_ids and alias_pattern.search(lowered_query):
            named_ids.append(source_id)
    return named_ids[:MAX_NAMED_PAPERS]


def find_aliased_paper(query: str) -> str | None:
    named_ids = find_aliased_papers(query)
    return named_ids[0] if named_ids else None
