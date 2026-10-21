# Self-learning research agent (LangGraph)

A domain-agnostic agent whose only tool is **deep web search**. It plans queries,
reads the actual pages, extracts claims, **cross-checks them across sources**
(corroborated / contested / single-source / refuted, weighted by source credibility),
loops on remaining gaps, then recommends an approach with confidence + caveats.

```
recall -> plan -> search -> extract -> cross_check -> assess -> (gaps? -> plan) -> decide -> learn
```

**Self-learning:** the `learn` node distills verified facts and research strategies into
`memory.json`; later runs recall them as *hints* (re-verified, never trusted blindly), and
repeated confirmation raises their confidence.

## Run
```
pip install -e .
export ANTHROPIC_API_KEY=...
selflearn-agent "What is the best way to ..."
pytest
```
Search backend is pluggable (`tools.SearchBackend`); default is DuckDuckGo via `ddgs` (no key).
