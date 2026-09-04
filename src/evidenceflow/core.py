"""Deterministic lexical evidence-flow analysis."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

_KNOWN_TYPES = {"user", "assistant", "tool_call", "tool_result", "retrieval", "system", "reasoning"}
_EVIDENCE_TYPES = {"tool_result", "retrieval"}
_TEXT_TYPES = {"user", "assistant", "tool_result", "retrieval", "system", "reasoning"}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with",
}
_TOKEN_RE = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*", re.UNICODE)
_BOUNDARY_RE = re.compile(r"(?:\r?\n\s*){2,}|(?<=[.!?])\s+(?=[^\W_])", re.UNICODE)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(_TOKEN_RE.findall(text))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in _normalize_text(text).split() if token not in _STOP_WORDS)


def _split_units(text: str) -> list[dict[str, Any]]:
    units = []
    for segment in _BOUNDARY_RE.split(text.strip()):
        normalized = _normalize_text(segment)
        tokens = _tokens(segment)
        if len(tokens) < 2:
            continue
        units.append({"text": segment.strip(), "normalized": normalized, "tokens": tokens})
    return units


def _matches(unit: Mapping[str, Any], later_text: str) -> bool:
    later_normalized = _normalize_text(later_text)
    if not later_normalized:
        return False
    normalized = unit["normalized"]
    if normalized and normalized in later_normalized:
        return True
    source_tokens = set(unit["tokens"])
    later_tokens = set(_tokens(later_text))
    overlap = len(source_tokens & later_tokens)
    return overlap >= 2 and overlap / len(source_tokens) >= 0.5


def _validate_trace(trace: Any) -> list[dict[str, Any]]:
    if isinstance(trace, (str, bytes)) or not isinstance(trace, Iterable):
        raise TypeError("trace must be a non-string iterable of mappings")
    steps = []
    seen_ids = set()
    for index, raw_step in enumerate(trace):
        if not isinstance(raw_step, Mapping):
            raise TypeError(f"step {index} must be a mapping")
        step = dict(raw_step)
        step_type = step.get("type")
        if not isinstance(step_type, str) or step_type not in _KNOWN_TYPES:
            raise ValueError(f"step {index} has unsupported type {step_type!r}")
        step_id = step.get("id", f"step-{index}")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"step {index} id must be a non-empty string")
        if step_id in seen_ids:
            raise ValueError(f"duplicate step id: {step_id!r}")
        seen_ids.add(step_id)
        if step_type in _TEXT_TYPES:
            content = step.get("content")
            if not isinstance(content, str):
                raise TypeError(f"step {index} content must be a string")
        if step_type == "tool_call" and "tool" in step and not isinstance(step["tool"], str):
            raise TypeError(f"step {index} tool must be a string")
        if step_type in _EVIDENCE_TYPES:
            for key in ("tool", "source", "evidence_id"):
                if key in step and (not isinstance(step[key], str) or not step[key]):
                    raise ValueError(f"step {index} {key} must be a non-empty string")
        step["id"] = step_id
        steps.append(step)
    return steps


def _extract_evidence(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    seen_ids = set()
    for step_index, step in enumerate(steps):
        if step["type"] not in _EVIDENCE_TYPES:
            continue
        evidence_id = step.get("evidence_id", f"evidence-{len(evidence)}")
        if evidence_id in seen_ids:
            raise ValueError(f"duplicate evidence id: {evidence_id!r}")
        seen_ids.add(evidence_id)
        evidence.append({
            "evidence_id": evidence_id,
            "step_id": step["id"],
            "step_index": step_index,
            "source_type": step["type"],
            "tool": step.get("tool"),
            "source": step.get("source"),
            "units": _split_units(step["content"]),
        })
    return evidence


def analyze(trace: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze apparent lexical evidence reuse in a normalized trace."""
    steps = _validate_trace(trace)
    evidence = _extract_evidence(steps)
    return {"steps": len(steps), "evidence_sources": len(evidence)}
