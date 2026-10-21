"""Persistent lesson store: the agent's self-learning substrate.

Facts are stored as *hints*: they are surfaced to the planner, but the agent
still re-verifies them via search, and confidence is reinforced or decayed
based on later outcomes.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .schemas import Lesson

_WORD = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class MemoryStore:
    def __init__(self, path: str | Path = "memory.json"):
        self.path = Path(path)
        self.items: list[dict] = []
        if self.path.exists():
            self.items = json.loads(self.path.read_text())

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.items, indent=2))

    def add(self, lesson: Lesson) -> None:
        """Insert, or reinforce an existing near-identical lesson."""
        new_tok = _tokens(lesson.text)
        for it in self.items:
            old_tok = _tokens(it["text"])
            overlap = len(new_tok & old_tok) / max(1, len(new_tok | old_tok))
            if it["domain"] == lesson.domain and it["kind"] == lesson.kind and overlap > 0.7:
                it["confidence"] = min(1.0, (it["confidence"] + lesson.confidence) / 2 + 0.05)
                it["seen"] += 1
                it["updated"] = time.time()
                self._save()
                return
        self.items.append({**lesson.model_dump(), "seen": 1, "updated": time.time()})
        self._save()

    def recall(self, query: str, k: int = 6) -> list[dict]:
        q = _tokens(query)
        scored = []
        for it in self.items:
            overlap = len(q & (_tokens(it["text"]) | _tokens(it["domain"])))
            if overlap:
                scored.append((overlap * it["confidence"], it))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [it for _, it in scored[:k]]
