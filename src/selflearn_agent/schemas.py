"""Pydantic schemas shared by graph state and structured LLM outputs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    credibility: float = 0.5  # 0..1 heuristic prior, see tools.score_credibility


class Plan(BaseModel):
    domain: str = Field(description="Short label for the problem domain, e.g. 'cooking', 'kubernetes'")
    queries: list[str] = Field(description="2-5 diverse web search queries, incl. one to find counter-evidence")


class RawClaim(BaseModel):
    statement: str = Field(description="One atomic, checkable factual claim relevant to the question")


class SourceClaims(BaseModel):
    claims: list[RawClaim]


class Finding(BaseModel):
    statement: str
    status: Literal["corroborated", "contested", "single_source", "refuted"]
    supporting: list[str] = Field(default_factory=list, description="source URLs that support it")
    conflicting: list[str] = Field(default_factory=list, description="source URLs that contradict it")
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class CrossCheck(BaseModel):
    findings: list[Finding]


class Assessment(BaseModel):
    sufficient: bool = Field(description="True if verified findings are enough to answer well")
    gaps: list[str] = Field(default_factory=list, description="What is still unknown/contested and needs searching")


class Approach(BaseModel):
    recommendation: str
    steps: list[str]
    confidence: float = Field(ge=0, le=1)
    caveats: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)


class Lesson(BaseModel):
    kind: Literal["fact", "strategy"] = Field(description="'fact' = verified domain knowledge; 'strategy' = how to research/solve better")
    domain: str
    text: str
    confidence: float = Field(ge=0, le=1)


class Lessons(BaseModel):
    lessons: list[Lesson]
