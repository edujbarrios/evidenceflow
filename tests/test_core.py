from copy import deepcopy

import pytest

from evidenceflow import analyze
from evidenceflow.core import _extract_evidence, _matches, _normalize_text, _split_units, _validate_trace


def test_normalization_preserves_values_and_ignores_case_and_punctuation():
    assert _normalize_text("  GPT-5: 3.14, BARCELONA!  ") == "gpt-5 3.14 barcelona"
    unit = _split_units("The Olympic host was Barcelona, Spain.")[0]
    assert _matches(unit, "THE OLYMPIC HOST WAS BARCELONA SPAIN")


def test_units_split_and_short_evidence_is_filtered():
    units = _split_units("Yes.\n\nBarcelona hosted the games. Madrid was not selected!")
    assert [item["text"] for item in units] == ["Barcelona hosted the games.", "Madrid was not selected!"]


def test_containment_and_partial_reuse():
    unit = _split_units("The 1992 Summer Olympics were held in Barcelona, Spain.")[0]
    assert _matches(unit, "The 1992 Summer Olympics were held in Barcelona, Spain.")
    assert _matches(unit, "They were held in Barcelona, Spain.")
    assert not _matches(unit, "Athens hosted another sporting event.")


def test_deterministic_ids_and_input_not_mutated():
    trace = [
        {"type": "tool_result", "tool": "search", "content": "Barcelona hosted the games."},
        {"id": "answer", "type": "assistant", "content": "Barcelona hosted the games."},
    ]
    original = deepcopy(trace)
    evidence = _extract_evidence(_validate_trace(trace))
    assert evidence[0]["evidence_id"] == "evidence-0"
    assert _validate_trace(trace)[0]["id"] == "step-0"
    assert trace == original


@pytest.mark.parametrize("trace,error", [
    (None, TypeError), ("bad", TypeError), ([1], TypeError), ([{}], ValueError),
    ([{"type": "unknown"}], ValueError), ([{"type": "assistant"}], TypeError),
    ([{"type": "tool_result", "content": None}], TypeError),
    ([{"id": "x", "type": "user", "content": "a"}, {"id": "x", "type": "assistant", "content": "b"}], ValueError),
])
def test_invalid_trace_structures(trace, error):
    with pytest.raises(error):
        analyze(trace)


def test_duplicate_supplied_evidence_ids():
    trace = [
        {"type": "retrieval", "evidence_id": "same", "content": "First useful sentence."},
        {"type": "retrieval", "evidence_id": "same", "content": "Second useful sentence."},
    ]
    with pytest.raises(ValueError, match="duplicate evidence id"):
        analyze(trace)
