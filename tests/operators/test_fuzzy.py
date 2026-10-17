import sys

import pytest

from superleaf.operators.comparison import F
from superleaf.operators.fuzzy import FuzzyMatcher


def test_fuzzy_matcher():
    fruits = ["apple", "banana", "grape"]
    partial_matcher = FuzzyMatcher(targets=fruits, substring=True)
    full_matcher = FuzzyMatcher(targets=fruits, substring=False)

    apple_sentence = "I like to eat an apple pie."
    result = partial_matcher(apple_sentence)
    best_match = result.best_match()
    assert best_match.target == "apple"
    assert best_match.score > 80

    result = full_matcher(apple_sentence)
    best_match = result.best_match()
    assert best_match.target == "apple"
    assert best_match.score < 80

    result = partial_matcher("I enjoy banna smoothies.")
    best_match = result.best_match()
    assert best_match.target == "banana"
    assert best_match.score > 80


def test_fuzzy_matcher_scores_normalized_target():
    result = FuzzyMatcher("Apple Pie!")("apple pie")
    assert result.score == 100
    assert result.target == "apple pie"
    assert result.target_raw == "Apple Pie!"


def test_fuzzy_match_defaults_match_fuzzy_matcher():
    assert F.fuzzy_match("APPLE")("I like apples.") is True
    assert FuzzyMatcher("APPLE")("I like apples.").score >= 80
    assert F.fuzzy_match("APPLE", normalizer=None)("I like apples.") is False
    assert F.fuzzy_match("apple", substring=False)("I like apples.") is False


def test_fuzzy_match_non_strings():
    assert F.fuzzy_match("a")(None) is False
    with pytest.raises(TypeError):
        F.fuzzy_match("a", raise_type_error=True)(1)


def test_missing_rapidfuzz_raises_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)
    monkeypatch.delitem(sys.modules, "superleaf.operators.fuzzy")
    with pytest.raises(ImportError, match=r"superleaf\[fuzzy\]"):
        F.fuzzy_match("a")
