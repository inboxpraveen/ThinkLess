from __future__ import annotations

import pytest

from thinkless.confidence import from_distribution, from_yes_probability, normalize_distribution


def test_uniform_distribution_has_zero_confidence() -> None:
    assert from_distribution({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(0.0)


def test_point_mass_has_full_confidence() -> None:
    assert from_distribution({"a": 1.0, "b": 0.0, "c": 0.0}) == pytest.approx(1.0)


def test_matches_the_typesafe_formula() -> None:
    # (k * p_max - 1) / (k - 1) with k = 3, p_max = 0.8
    assert from_distribution({"a": 0.8, "b": 0.1, "c": 0.1}) == pytest.approx(0.7)


def test_laya_example_from_the_smoke_test() -> None:
    # Laya reported 0.9782 for this five-way distribution; the normalized
    # max-probability confidence is 0.9941.
    probs = {"a": 0.9953, "b": 0.0006, "c": 0.0027, "d": 0.0007, "e": 0.0007}
    assert from_distribution(probs) == pytest.approx(0.9941, abs=1e-4)


def test_yes_no_confidence() -> None:
    assert from_yes_probability(0.5) == pytest.approx(0.0)
    assert from_yes_probability(0.9) == pytest.approx(0.8)
    assert from_yes_probability(0.1) == pytest.approx(0.8)
    assert from_yes_probability(0.4296) == pytest.approx(0.1408, abs=1e-4)


def test_edge_cases() -> None:
    assert from_distribution({}) == 0.0
    assert from_distribution({"only": 0.3}) == 1.0
    assert from_yes_probability(1.7) == 1.0


def test_normalize_distribution() -> None:
    assert normalize_distribution({"a": 3.0, "b": 1.0}) == {"a": 0.75, "b": 0.25}
    assert normalize_distribution({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}
    assert normalize_distribution({"a": -1.0, "b": 1.0}) == {"a": 0.0, "b": 1.0}
