r"""LangGraph workflow:

recall -> plan -> search -> extract -> cross_check -> assess --(gaps & budget left)--> plan
                                                        \--(sufficient)--> decide -> learn -> END
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .memory import MemoryStore
from .schemas import (Approach, Assessment, CrossCheck, Finding, Lessons, Plan, Source, SourceClaims)
from .tools import SearchBackend, ddgs_backend, deep_web_search

MAX_ITERATIONS = 3


class AgentState(TypedDict, total=False):
    question: str
    domain: str
    memory_hints: list[dict]
    gaps: list[str]
    queries: list[str]
    sources: Annotated[list[Source], operator.add]
    claims: list[dict]  # {"statement", "url"}
    findings: list[Finding]
    iteration: int
    approach: Approach
    learned: int


def _fmt_sources(sources: list[Source]) -> str:
    return "\n".join(f"[{i}] {s.url} (credibility prior {s.credibility:.2f})" for i, s in enumerate(sources))


def build_graph(
    llm: Any,
    memory: MemoryStore,
    search_backend: SearchBackend = ddgs_backend,
    fetcher: Callable[[str], str] | None = None,
    max_iterations: int = MAX_ITERATIONS,
):
    """`llm` is any LangChain chat model supporting `.with_structured_output`."""

    def ask(schema, system: str, user: str):
        return llm.with_structured_output(schema).invoke([("system", system), ("human", user)])

    def recall(state: AgentState):
        return {"memory_hints": memory.recall(state["question"]), "iteration": 0, "gaps": []}

    def plan(state: AgentState):
        hints = "\n".join(f"- ({h['kind']}, conf {h['confidence']:.2f}) {h['text']}" for h in state.get("memory_hints", []))
        gaps = "\n".join(f"- {g}" for g in state.get("gaps", []))
        p = ask(
            Plan,
            "You plan web research for ANY domain. Prior memory is only a hint to be re-verified, never truth. "
            "Always include at least one query seeking counter-evidence or criticism.",
            f"Question: {state['question']}\nMemory hints:\n{hints or '(none)'}\nOpen gaps to resolve:\n{gaps or '(first pass)'}",
        )
        return {"domain": p.domain, "queries": p.queries}

    def search(state: AgentState):
        seen = {s.url for s in state.get("sources", [])}
        new: list[Source] = []
        for q in state["queries"]:
            kwargs = {"fetcher": fetcher} if fetcher else {}
            for s in deep_web_search(q, backend=search_backend, **kwargs):
                if s.url not in seen:
                    seen.add(s.url)
                    new.append(s)
        return {"sources": new}

    def extract(state: AgentState):
        claims = []
        for s in state["sources"]:
            out = ask(
                SourceClaims,
                "Extract atomic factual claims from the page that bear on the question. Only what the text says.",
                f"Question: {state['question']}\nURL: {s.url}\nTEXT:\n{s.text}",
            )
            claims += [{"statement": c.statement, "url": s.url} for c in out.claims]
        return {"claims": claims}

    def cross_check(state: AgentState):
        claim_txt = "\n".join(f"- {c['statement']}  <{c['url']}>" for c in state["claims"])
        out = ask(
            CrossCheck,
            "Cluster equivalent claims across sources and decide which are correct. Rules: 'corroborated' needs >=2 "
            "INDEPENDENT sources agreeing; if sources disagree mark 'contested' and weigh credibility priors and "
            "evidence quality; one source only => 'single_source' with low confidence; clearly wrong => 'refuted'. "
            "Never invent URLs; use only those given.",
            f"Question: {state['question']}\nSources:\n{_fmt_sources(state['sources'])}\nClaims:\n{claim_txt}",
        )
        return {"findings": out.findings}

    def assess(state: AgentState):
        findings = "\n".join(f"- [{f.status} {f.confidence:.2f}] {f.statement}" for f in state["findings"])
        a = ask(
            Assessment,
            "Decide if verified findings suffice to give a well-grounded approach. If key points are contested or "
            "single-sourced, list specific gaps to search next.",
            f"Question: {state['question']}\nFindings:\n{findings}",
        )
        it = state.get("iteration", 0) + 1
        return {"gaps": [] if a.sufficient else a.gaps, "iteration": it}

    def route(state: AgentState):
        if state["gaps"] and state["iteration"] < max_iterations:
            return "plan"
        return "decide"

    def decide(state: AgentState):
        usable = [f for f in state["findings"] if f.status in ("corroborated", "contested", "single_source")]
        txt = "\n".join(
            f"- [{f.status} {f.confidence:.2f}] {f.statement} (for: {f.supporting}; against: {f.conflicting}) {f.reasoning}"
            for f in usable
        )
        ap = ask(
            Approach,
            "Produce the best approach grounded ONLY in the findings. Prefer corroborated facts, explicitly flag "
            "contested/single-source ones as caveats, and keep confidence consistent with the evidence.",
            f"Question: {state['question']}\nFindings:\n{txt}",
        )
        return {"approach": ap}

    def learn(state: AgentState):
        findings = "\n".join(f"- [{f.status} {f.confidence:.2f}] {f.statement}" for f in state["findings"])
        out = ask(
            Lessons,
            "Distill durable lessons. kind='fact': only corroborated, high-confidence domain knowledge. "
            "kind='strategy': what search/verification approach worked or failed for this kind of question. "
            "Skip anything contested or single-sourced.",
            f"Domain: {state['domain']}\nQuestion: {state['question']}\nFindings:\n{findings}\n"
            f"Iterations used: {state['iteration']}",
        )
        for l in out.lessons:
            memory.add(l)
        return {"learned": len(out.lessons)}

    g = StateGraph(AgentState)
    for name, fn in [("recall", recall), ("plan", plan), ("search", search), ("extract", extract),
                     ("cross_check", cross_check), ("assess", assess), ("decide", decide), ("learn", learn)]:
        g.add_node(name, fn)
    g.add_edge(START, "recall")
    g.add_edge("recall", "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "extract")
    g.add_edge("extract", "cross_check")
    g.add_edge("cross_check", "assess")
    g.add_conditional_edges("assess", route, {"plan": "plan", "decide": "decide"})
    g.add_edge("decide", "learn")
    g.add_edge("learn", END)
    return g.compile()


def run_agent(question: str, llm: Any, memory: MemoryStore, **kw) -> AgentState:
    return build_graph(llm, memory, **kw).invoke({"question": question})
