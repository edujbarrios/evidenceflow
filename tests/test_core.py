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


def test_empty_trace_and_trace_without_evidence():
    empty = analyze([])
    assert empty["steps"] == empty["evidence_units"] == 0
    assert empty["overall_survival_ratio"] == empty["final_answer_survival_ratio"] == 0.0
    report = analyze([{"type": "assistant", "content": "No evidence was retrieved."}])
    assert report["evidence"] == [] and report["tools"] == {}


def test_hand_computed_flow_tool_source_and_final_metrics():
    trace = [
        {"id": "q", "type": "user", "content": "Where were the games held?"},
        {"id": "r1", "type": "tool_result", "tool": "search", "content":
         "Barcelona hosted the 1992 Olympics.\n\nThe mascot was Cobi."},
        {"id": "thought", "type": "reasoning", "content": "Barcelona hosted the 1992 Olympics."},
        {"id": "r2", "type": "retrieval", "source": "manual.pdf", "content":
         "The games took place in Barcelona.\n\nSailing happened on the coast."},
        {"id": "answer", "type": "assistant", "content": "The games took place in Barcelona."},
    ]
    report = analyze(trace)
    assert report["steps"] == 5 and report["evidence_sources"] == 2
    assert report["evidence_units"] == 4
    assert report["reused_units"] == 2
    assert report["final_reused_units"] == 1
    assert report["overall_survival_ratio"] == 0.5
    assert report["final_answer_survival_ratio"] == 0.25
    assert report["tools"]["search"]["survival_ratio"] == 0.5
    assert report["sources"]["manual.pdf"]["final_answer_survival_ratio"] == 0.5
    assert report["evidence"][0]["reused_at"] == ["thought"]
    assert report["evidence"][0]["steps_until_last_reuse"] == 1


def test_unused_evidence_and_intermediate_reuse_without_final_reuse():
    report = analyze([
        {"id": "tool", "type": "tool_result", "tool": "lookup", "content": "Mercury is closest to the Sun."},
        {"id": "thought", "type": "reasoning", "content": "Mercury is closest to the Sun."},
        {"id": "unused", "type": "retrieval", "source": "notes", "content": "Neptune has strong winds."},
        {"id": "final", "type": "assistant", "content": "Mercury is the answer."},
    ])
    assert report["reused_units"] == 1
    assert report["final_reused_units"] == 0
    assert report["apparently_unused_sources"] == 1
    assert report["unused_evidence"][0]["evidence_id"] == "evidence-1"


def test_repeated_groups_are_transitive_and_deterministic():
    trace = [
        {"type": "tool_result", "content": "Barcelona hosted the Olympic games."},
        {"type": "retrieval", "content": "The Olympic games took place in Barcelona."},
        {"type": "retrieval", "content": "Games took place in Barcelona, Spain."},
    ]
    first = analyze(trace)
    second = analyze(trace)
    assert first == second
    assert first["repeated_evidence_groups"] == [["evidence-0", "evidence-1", "evidence-2"]]
