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
    if normalized and f" {normalized} " in f" {later_normalized} ":
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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record[key]
        if name is None:
            continue
        item = result.setdefault(name, {
            "evidence_sources": 0,
            "evidence_units": 0,
            "reused_units": 0,
            "final_reused_units": 0,
        })
        item["evidence_sources"] += 1
        item["evidence_units"] += record["evidence_units"]
        item["reused_units"] += record["reused_units"]
        item["final_reused_units"] += record["final_reused_units"]
    for item in result.values():
        item["survival_ratio"] = _ratio(item["reused_units"], item["evidence_units"])
        item["final_answer_survival_ratio"] = _ratio(
            item["final_reused_units"], item["evidence_units"]
        )
    return result


def _repeated_groups(evidence: list[dict[str, Any]]) -> list[list[str]]:
    adjacency = {item["evidence_id"]: set() for item in evidence}
    for index, left in enumerate(evidence):
        for right in evidence[index + 1:]:
            repeated = any(
                _matches(left_unit, right_unit["text"]) or _matches(right_unit, left_unit["text"])
                for left_unit in left["units"]
                for right_unit in right["units"]
            )
            if repeated:
                adjacency[left["evidence_id"]].add(right["evidence_id"])
                adjacency[right["evidence_id"]].add(left["evidence_id"])
    groups = []
    visited = set()
    for item in evidence:
        evidence_id = item["evidence_id"]
        if evidence_id in visited or not adjacency[evidence_id]:
            continue
        stack = [evidence_id]
        group = []
        visited.add(evidence_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        order = {value["evidence_id"]: i for i, value in enumerate(evidence)}
        groups.append(sorted(group, key=order.get))
    return groups


def analyze(trace: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze apparent lexical evidence reuse in a normalized trace."""
    steps = _validate_trace(trace)
    evidence = _extract_evidence(steps)
    final_index = next(
        (index for index in range(len(steps) - 1, -1, -1) if steps[index]["type"] == "assistant"),
        None,
    )
    records = []
    for item in evidence:
        unit_reuse_steps = []
        for unit in item["units"]:
            reused_at = [
                step["id"]
                for step in steps[item["step_index"] + 1:]
                if step["type"] in {"assistant", "reasoning"} and _matches(unit, step["content"])
            ]
            unit_reuse_steps.append(reused_at)
        reused_units = sum(bool(value) for value in unit_reuse_steps)
        final_reused_units = (
            sum(steps[final_index]["id"] in value for value in unit_reuse_steps)
            if final_index is not None and final_index > item["step_index"] else 0
        )
        reused_at = [
            step["id"]
            for step in steps[item["step_index"] + 1:]
            if any(step["id"] in value for value in unit_reuse_steps)
        ]
        last_index = max((i for i, step in enumerate(steps) if step["id"] in reused_at), default=None)
        records.append({
            "evidence_id": item["evidence_id"],
            "step_id": item["step_id"],
            "source_type": item["source_type"],
            "tool": item["tool"],
            "source": item["source"],
            "evidence_units": len(item["units"]),
            "reused_units": reused_units,
            "final_reused_units": final_reused_units,
            "survival_ratio": _ratio(reused_units, len(item["units"])),
            "final_answer_survival_ratio": _ratio(final_reused_units, len(item["units"])),
            "reused_at": reused_at,
            "reuse_count": len(reused_at),
            "last_reuse_step": steps[last_index]["id"] if last_index is not None else None,
            "steps_until_last_reuse": last_index - item["step_index"] if last_index is not None else None,
        })

    evidence_units = sum(item["evidence_units"] for item in records)
    reused_units = sum(item["reused_units"] for item in records)
    final_reused_units = sum(item["final_reused_units"] for item in records)
    repeated_groups = _repeated_groups(evidence)
    unused = [
        {
            "evidence_id": item["evidence_id"], "step_id": item["step_id"],
            "tool": item["tool"], "source": item["source"], "evidence_units": item["evidence_units"],
        }
        for item in records if item["reused_units"] == 0
    ]
    return {
        "steps": len(steps),
        "evidence_sources": len(records),
        "tool_results": sum(step["type"] == "tool_result" for step in steps),
        "retrieval_steps": sum(step["type"] == "retrieval" for step in steps),
        "evidence_units": evidence_units,
        "reused_units": reused_units,
        "final_reused_units": final_reused_units,
        "overall_survival_ratio": _ratio(reused_units, evidence_units),
        "final_answer_survival_ratio": _ratio(final_reused_units, evidence_units),
        "apparently_unused_sources": len(unused),
        "evidence": records,
        "tools": _aggregate(records, "tool"),
        "sources": _aggregate(records, "source"),
        "unused_evidence": unused,
        "repeated_evidence_groups": repeated_groups,
    }
