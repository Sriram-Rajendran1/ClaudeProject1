import argparse
import os
import sys

from langchain_anthropic import ChatAnthropic

from .graph import run_agent
from .memory import MemoryStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-learning web-research agent (LangGraph)")
    ap.add_argument("question")
    ap.add_argument("--memory", default="memory.json")
    ap.add_argument("--model", default=os.getenv("AGENT_MODEL", "claude-sonnet-5-5"))
    args = ap.parse_args()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    state = run_agent(args.question, ChatAnthropic(model=args.model), MemoryStore(args.memory))
    a = state["approach"]
    print(f"\n# Domain: {state['domain']}  |  rounds: {state['iteration']}  |  lessons learned: {state['learned']}")
    print(f"\n## Recommendation (confidence {a.confidence:.0%})\n{a.recommendation}\n\n## Steps")
    for i, s in enumerate(a.steps, 1):
        print(f"{i}. {s}")
    if a.caveats:
        print("\n## Caveats")
        for c in a.caveats:
            print(f"- {c}")
    print("\n## Sources")
    for u in a.sources_used:
        print(f"- {u}")


if __name__ == "__main__":
    main()
