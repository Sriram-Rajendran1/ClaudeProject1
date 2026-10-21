"""The agent's only tool (for now): deep web search.

"Deep" = run a search, then actually fetch and read each result page, rather
than trusting snippets. Each source gets a heuristic credibility prior.
"""
from __future__ import annotations

from typing import Callable, Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .schemas import Source

MAX_CHARS = 6000


class SearchBackend(Protocol):
    def __call__(self, query: str, max_results: int) -> list[dict]: ...  # {"title","href","body"}


def ddgs_backend(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    return list(DDGS().text(query, max_results=max_results))


def score_credibility(url: str) -> float:
    """Cheap prior from the domain; the LLM still weighs content & agreement."""
    host = urlparse(url).netloc.lower()
    if host.endswith((".gov", ".edu")) or ".gov." in host or ".ac." in host:
        return 0.9
    if any(h in host for h in ("wikipedia.org", "arxiv.org", "nature.com", "who.int", "docs.", "github.com")):
        return 0.8
    if any(h in host for h in ("reddit.com", "quora.com", "medium.com", "pinterest.", "facebook.", "tiktok.")):
        return 0.35
    return 0.5


def fetch_text(url: str, timeout: float = 10.0) -> str:
    r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "selflearn-agent/0.1"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())[:MAX_CHARS]


def deep_web_search(
    query: str,
    max_results: int = 5,
    backend: SearchBackend = ddgs_backend,
    fetcher: Callable[[str], str] = fetch_text,
) -> list[Source]:
    sources: list[Source] = []
    for hit in backend(query, max_results):
        url = hit.get("href") or hit.get("url") or ""
        if not url:
            continue
        try:
            text = fetcher(url)
        except Exception:
            text = ""
        if not text:  # fall back to the snippet if the page can't be read
            text = hit.get("body", "")
        if text:
            sources.append(Source(url=url, title=hit.get("title", ""), text=text, credibility=score_credibility(url)))
    return sources
