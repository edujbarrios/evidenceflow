# EvidenceFlow

[![PyPI version](https://img.shields.io/pypi/v/evidenceflow?label=PyPI&logo=pypi&cacheSeconds=60)](https://pypi.org/project/evidenceflow/)
[![License: MPL 2.0](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](https://github.com/edujbarrios/evidenceflow/blob/main/LICENSE)

Track how retrieved and tool-generated evidence propagates through AI-agent traces.

EvidenceFlow is a lightweight deterministic Python library for measuring apparent evidence reuse across retrieval, tool results, intermediate reasoning, and final outputs.

## Installation

```bash
pip install evidenceflow
```

## Usage

```python
from evidenceflow import analyze

trace = [
    {"id": "1", "type": "user", "content": "Where were the 1992 Summer Olympics held?"},
    {
        "id": "2",
        "type": "tool_result",
        "tool": "search",
        "content": (
            "The 1992 Summer Olympics were held in Barcelona, Spain.\n\n"
            "The official mascot was Cobi."
        ),
    },
    {"id": "3", "type": "assistant", "content": "They were held in Barcelona, Spain."},
]

report = analyze(trace)

summary_keys = (
    "evidence_units",
    "reused_units",
    "final_reused_units",
    "overall_survival_ratio",
    "final_answer_survival_ratio",
)

print({key: report[key] for key in summary_keys})
print(report["tools"]["search"])
```

Output:

```python
{'evidence_units': 2, 'reused_units': 1, 'final_reused_units': 1, 'overall_survival_ratio': 0.5, 'final_answer_survival_ratio': 0.5}
{'evidence_sources': 1, 'evidence_units': 2, 'reused_units': 1, 'final_reused_units': 1, 'survival_ratio': 0.5, 'final_answer_survival_ratio': 0.5}
```

The search result introduced two evidence units. The final response has detected overlap with the Barcelona unit, while the mascot unit does not appear later, so both overall and final-answer lexical survival are `0.5`.

## Why EvidenceFlow?

Agent traces can contain much more retrieved or tool-generated text than appears in later reasoning or the final answer. EvidenceFlow provides a small, offline diagnostic for inspecting that flow. It is not an observability platform or model evaluator.

* No LLM calls or embeddings
* No external services, server, database, or dashboard
* No framework or provider coupling
* No runtime dependencies

## Trace format

Pass any non-string iterable of dictionaries. Every step needs a supported `type`; textual steps also need string `content`. Supported types are `user`, `assistant`, `tool_call`, `tool_result`, `retrieval`, `system`, and `reasoning`. Unknown types raise `ValueError` so silently omitted evidence cannot skew a report.

Step IDs are optional and deterministically default to `step-0`, `step-1`, and so on. Supplied IDs must be unique. Evidence is introduced only by `tool_result` and `retrieval` steps in v0.1.0. Their optional `evidence_id` values must be unique; otherwise IDs default to `evidence-0`, `evidence-1`, and so on.

## Analyze evidence flow

`analyze(trace)` returns a JSON-serializable dictionary. Its top-level counts include `evidence_units`, `reused_units`, `final_reused_units`, `overall_survival_ratio`, and `final_answer_survival_ratio`. Each item in `evidence` identifies its introduction step and later `assistant` or `reasoning` steps where overlap was detected.

The final answer is the last `assistant` step. Evidence introduced after it cannot survive into it. Input dictionaries are never mutated.

## Tool-level analysis

`report["tools"]` groups tool results by their `tool` value and reports source and unit counts plus overall and final-answer survival ratios. These are neutral diagnostics: low overlap does not establish that a tool was unnecessary.

## Retrieval/source analysis

A retrieval step may provide a `source`, such as `manual.pdf`. `report["sources"]` aggregates the same metrics by that value. Retrieval steps without a source remain in overall metrics but are not placed in a named source group.

## Evidence survival

An evidence source's survival ratio is:

$$
\text{survival ratio}
=
\frac{\text{evidence units with detected overlap in a later assistant or reasoning step}}
{\text{evidence units introduced}}
$$

Final-answer survival uses the same denominator but counts only units detected in the last assistant response. Empty denominators produce `0.0`.

Per-source fields also include `reused_at`, `reuse_count`, `last_reuse_step`, and `steps_until_last_reuse`.

## Repeated and unused evidence

`unused_evidence` lists sources with no units apparently reused later. `repeated_evidence_groups` lists connected groups of evidence sources whose units substantially overlap. Groups are transitive and ordered by trace position. Neither result proves semantic redundancy or non-use.

## How it works

EvidenceFlow normalizes Unicode and case, handles punctuation while preserving values such as `3.14`, `2026`, and `GPT-5`, and splits evidence on simple sentence or paragraph boundaries. Segments with fewer than two informative words are ignored. A small fixed stop-word list reduces noise.

A unit matches when its normalized text is contained in a later step, or when at least two informative words overlap and they cover at least half of the unit's unique informative words. This deliberately simple rule is deterministic and readable in `src/evidenceflow/core.py`.

## Use cases

* Inspect which tool outputs have high or low detected downstream coverage
* Find retrieval sources whose text appears in a final response
* Surface large evidence acquisitions with no detected later overlap
* Track apparent propagation through intermediate reasoning steps
* Add deterministic information-flow checks to CI

## Features

* Plain dictionary input and JSON-serializable output
* Tool-level and retrieval-source aggregation
* Per-evidence propagation and simple decay fields
* Apparently unused and repeated evidence diagnostics
* Deterministic IDs and output
* Standard-library-only runtime

## Limitations

EvidenceFlow measures lexical overlap. It does **not** prove causal influence, model attention, hidden-state influence, semantic equivalence, factual correctness, or answer quality. Evidence may be paraphrased, summarized, transformed, inferred, or omitted from visible text while still affecting model behavior. Conversely, lexical overlap may be coincidental. Low detected survival therefore does not mean evidence was useless.

The sentence splitter is intentionally simple, the fixed stop-word list is English-oriented, and the lexical threshold can produce false positives and false negatives. v0.1.0 does not include embeddings, custom scorers, trace adapters, visual evidence, graph export, HTML reports, or streaming traces.

## Issues

Report issues in the [GitHub issue tracker](https://github.com/edujbarrios/evidenceflow/issues).

## Author

Eduardo J. Barrios — [edujbarrios@outlook.com](mailto:edujbarrios@outlook.com)

## License

[Mozilla Public License 2.0](https://github.com/edujbarrios/evidenceflow/blob/main/LICENSE)
