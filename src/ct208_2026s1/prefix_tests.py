from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

PrefixTest = Callable[[Sequence[int], int], bool]


def always_true_prefix_test(current_values: Sequence[int], level: int) -> bool:
    return True


def build_all_true_tests(n_size: int) -> List[PrefixTest]:
    """
    100% case: all permutations are valid.
    """
    return [always_true_prefix_test for _ in range(n_size)]


def build_first_value_allowed_tests(
    n_size: int,
    allowed_first_values: Iterable[int],
) -> Tuple[List[PrefixTest], float]:
    """
    Exact ratio = len(allowed_first_values) / n_size.

    For n=10, this gives exact tenths:
      {1}       -> 10%
      {1,2}     -> 20%
      ...
      {1..10}   -> 100%
    """
    allowed = set(allowed_first_values)
    if not allowed:
        raise ValueError("allowed_first_values must not be empty.")
    if any(v < 1 or v > n_size for v in allowed):
        raise ValueError("allowed_first_values must be within 1..n_size.")

    def t1(current_values: Sequence[int], level: int) -> bool:
        return current_values[1] in allowed

    tests: List[PrefixTest] = [t1]
    tests.extend(always_true_prefix_test for _ in range(n_size - 1))

    actual_ratio = len(allowed) / n_size
    return tests, actual_ratio


def build_pair_prefix_tests_for_target_ratio(
    n_size: int,
    target_ratio: float,
) -> Tuple[List[PrefixTest], float]:
    """
    Approximate a target acceptance ratio using only prefix tests of size 1 and 2.

    Each allowed ordered pair (a1, a2) contributes exactly (n-2)! leaves.
    Therefore:

      accepted_ratio = allowed_pairs / (n * (n - 1))

    This is much closer to 10% than restricting only a1 when n != 10.
    """
    if n_size < 2:
        raise ValueError("n_size must be at least 2.")
    if not (0 < target_ratio <= 1):
        raise ValueError("target_ratio must be in (0, 1].")

    all_pairs = [
        (a1, a2)
        for a1 in range(1, n_size + 1)
        for a2 in range(1, n_size + 1)
        if a1 != a2
    ]

    allowed_pair_count = max(1, round(target_ratio * len(all_pairs)))
    allowed_pair_count = min(allowed_pair_count, len(all_pairs))

    allowed_pairs = set(all_pairs[:allowed_pair_count])
    allowed_first_values = {a1 for (a1, _) in allowed_pairs}

    def t1(current_values: Sequence[int], level: int) -> bool:
        return current_values[1] in allowed_first_values

    def t2(current_values: Sequence[int], level: int) -> bool:
        return (current_values[1], current_values[2]) in allowed_pairs

    tests: List[PrefixTest] = [t1, t2]
    tests.extend(always_true_prefix_test for _ in range(n_size - 2))

    actual_ratio = allowed_pair_count / len(all_pairs)
    return tests, actual_ratio
