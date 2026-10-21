from selflearn_agent import run_agent
from selflearn_agent.memory import MemoryStore
from selflearn_agent.schemas import *


class FakeLLM:
    """Scripted structured-output LLM; first assess call reports a gap to test the loop."""

    def __init__(self):
        self.assess_calls = 0

    def with_structured_output(self, schema):
        outer = self

        class R:
            def invoke(self, msgs):
                if schema is Plan:
                    return Plan(domain="python", queries=["q1", "q2"])
                if schema is SourceClaims:
                    return SourceClaims(claims=[RawClaim(statement="Use venv")])
                if schema is CrossCheck:
                    return CrossCheck(findings=[Finding(statement="Use venv", status="corroborated",
                                                        supporting=["https://a.edu/x"], confidence=0.9)])
                if schema is Assessment:
                    outer.assess_calls += 1
                    return Assessment(sufficient=outer.assess_calls > 1, gaps=["need counter-evidence"])
                if schema is Approach:
                    return Approach(recommendation="Use venv", steps=["python -m venv"], confidence=0.9)
                if schema is Lessons:
                    return Lessons(lessons=[Lesson(kind="fact", domain="python", text="Use venv for isolation", confidence=0.9)])
        return R()


def fake_search(query, max_results):
    return [{"title": "t", "href": f"https://{query}.example.com/p", "body": "Use venv for isolation."}]


def fail_fetch(url):
    raise RuntimeError("offline")


def test_loop_and_learning(tmp_path):
    mem = MemoryStore(tmp_path / "m.json")
    llm = FakeLLM()
    st = run_agent("how to isolate python deps", llm, mem, search_backend=fake_search, fetcher=fail_fetch)
    assert st["iteration"] == 2  # looped once on gap
    assert st["approach"].recommendation == "Use venv"
    assert len(st["sources"]) == 2  # q1,q2 deduped across rounds
    assert st["learned"] == 1
    # persisted and recalled in a fresh run
    assert MemoryStore(tmp_path / "m.json").recall("python dependency isolation venv")


def test_memory_reinforces(tmp_path):
    mem = MemoryStore(tmp_path / "m.json")
    l = Lesson(kind="fact", domain="x", text="alpha beta gamma delta", confidence=0.6)
    mem.add(l); mem.add(l)
    assert len(mem.items) == 1 and mem.items[0]["seen"] == 2 and mem.items[0]["confidence"] > 0.6
