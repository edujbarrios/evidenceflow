"""Core implementation for EvidenceFlow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def analyze(trace: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze apparent lexical evidence reuse in a normalized trace."""
    raise NotImplementedError("evidence analysis is not implemented yet")
