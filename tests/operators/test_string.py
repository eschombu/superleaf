from superleaf.operators.string import str_op, FuzzyMatcher
from superleaf.sequences import flat_map, mapped


def test_str_op():
    original = ["abc  ", "  12, 3", "  you,   and me!!  "]
    stripped = [s.strip() for s in original]
    assert mapped(str_op("strip"), original) == stripped

    squeezed = "abc.12.3.you.and me!!"
    assert ".".join(
        mapped(str_op("strip"), flat_map(str_op("strip") >> str_op("split", ","), original))
    ) == squeezed


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
