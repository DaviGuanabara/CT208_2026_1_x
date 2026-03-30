from typing import List, Sequence

from algorithm_x import AlgorithmX, PrefixTest


def always_true_prefix_test(current_values: Sequence[int], level: int) -> bool:
    """
    Accept every prefix.

    This makes Algorithm X behave as a lexicographic generator without pruning.
    """
    return True


def build_all_true_tests(n_size: int) -> List[PrefixTest]:
    return [always_true_prefix_test for _ in range(n_size)]


def build_first_value_limit_tests(
    n_size: int, max_first_value: int
) -> List[PrefixTest]:
    """
    A simple didactic pruning family.

    At level 1, only prefixes whose first value is <= max_first_value are accepted.
    All deeper levels are accepted.
    """

    def test(current_values: Sequence[int], level: int) -> bool:
        if level == 1:
            return current_values[1] <= max_first_value
        return True

    return [test for _ in range(n_size)]


if __name__ == "__main__":
    values = [1, 2, 3]
    tests = build_all_true_tests(len(values))

    algorithm = AlgorithmX(values, tests)
    permutations = algorithm.generate(generation_ratio=1.0, store_permutations=True)

    print("Generated permutations:")
    for permutation in permutations:
        print(permutation)

    print("\nMetrics:")
    print(algorithm.get_metrics_dict())
    print("Raw row:", algorithm.get_raw_row())
    print("Overhead ratio:", algorithm.get_overhead_ratio())
