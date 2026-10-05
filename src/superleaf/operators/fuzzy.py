from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional

try:
    from rapidfuzz import fuzz, utils
except ImportError as e:
    raise ImportError(
        "Fuzzy matching requires rapidfuzz. Install it with `pip install superleaf[fuzzy]`."
    ) from e

from superleaf.operators.base import bool_operator, BooleanOperator, Operator


@dataclass
class FuzzyMatchResult:
    """Score of one string against one target.

    ``string`` and ``target`` are the normalized values that were scored; ``string_raw`` and ``target_raw``
    hold the originals when normalization changed them.
    """

    string: str
    target: str
    score: float
    string_raw: str | None = None
    target_raw: str | None = None

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in asdict(self).items() if v is not None)})"


@dataclass
class FuzzyMatchResults:
    """Scores of one string against several targets."""

    results: list[FuzzyMatchResult]

    def best_match(self) -> FuzzyMatchResult:
        """Return the highest-scoring result.

        Returns
        -------
        FuzzyMatchResult
            The result with the highest score.
        """
        return max(self.results, key=lambda r: r.score)


class FuzzyMatcher(Operator):
    """Score strings against one or more targets with rapidfuzz.

    Calling the matcher on a string returns a :class:`FuzzyMatchResult` for a single target, or
    :class:`FuzzyMatchResults` for several.

    Parameters
    ----------
    targets : str or iterable of str
        Strings to match against.
    normalizer : callable or None, optional
        Applied to inputs and targets before scoring. Default ``rapidfuzz.utils.default_process``
        (lowercase, non-alphanumerics replaced by spaces, trimmed). None disables normalization.
    substring : bool, optional
        If True (default), score with ``fuzz.partial_ratio`` so a target can match part of the input.
        Otherwise use ``fuzz.ratio``.

    Raises
    ------
    ValueError
        If ``targets`` is empty.
    """

    def __init__(
            self,
            targets: str | Iterable[str],
            normalizer: Optional[Callable[[str], str]] = utils.default_process,
            substring: bool = True,
    ):
        if isinstance(targets, str):
            targets = [targets]
        else:
            targets = list(targets)
        if len(targets) == 0:
            raise ValueError("At least one target string must be provided.")
        self.single_target = len(targets) == 1
        self.targets = targets
        self.normalizer = normalizer
        if normalizer:
            self.targets_norm = [normalizer(t) for t in targets]
        else:
            self.targets_norm = targets
        self.substring = substring

    def __call__(self, s: str) -> FuzzyMatchResult | FuzzyMatchResults:
        s_norm = self.normalizer(s) if self.normalizer else s
        scorer = fuzz.partial_ratio if self.substring else fuzz.ratio

        results = []
        for t, t_norm in zip(self.targets, self.targets_norm):
            result_kws = dict(string=s_norm, target=t_norm, score=scorer(t_norm, s_norm))
            if s_norm != s:
                result_kws['string_raw'] = s
            if t_norm != t:
                result_kws['target_raw'] = t
            results.append(FuzzyMatchResult(**result_kws))

        if self.single_target:
            return results[0]
        return FuzzyMatchResults(results=results)

    def _predicate(self, threshold: float, raise_type_error: bool = False) -> Callable[[object], bool]:
        def f(s) -> bool:
            if not isinstance(s, str):
                if raise_type_error:
                    raise TypeError(f"Input must be a string, but got {type(s)}")
                return False
            result = self(s)
            if isinstance(result, FuzzyMatchResults):
                result = result.best_match()
            return result.score >= threshold

        return f

    def to_bool_operator(
            self, threshold: float = 80.0, raise_type_error: bool = False, exceptions=None, fallback=False
    ) -> BooleanOperator:
        """Predicate testing that the best score is at least ``threshold``.

        Parameters
        ----------
        threshold : float, optional
            Minimum score (0-100). Default 80.
        raise_type_error : bool, optional
            If True, raise ``TypeError`` for non-``str`` inputs; otherwise they give False.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback`` (see :func:`bool_operator`).
        fallback : bool, optional
            Result when one of ``exceptions`` is raised.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return bool_operator(self._predicate(threshold, raise_type_error), exceptions=exceptions, fallback=fallback)
