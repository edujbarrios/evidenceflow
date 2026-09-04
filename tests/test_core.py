import pytest

from evidenceflow import analyze


def test_analyze_is_exposed():
    with pytest.raises(NotImplementedError):
        analyze([])
