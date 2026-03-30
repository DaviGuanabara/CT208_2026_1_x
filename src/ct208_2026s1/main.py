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

    def run_single_experiment(n_size: int, generation_ratio: float) -> dict:
        input_values = list(range(1, n_size + 1))
        prefix_tests = build_all_true_tests(n_size)

        algorithm = AlgorithmX(input_values, prefix_tests)
        algorithm.generate(
            generation_ratio=generation_ratio,
            store_permutations=False,
        )

        metrics = algorithm.get_metrics_dict()

        return {
            "n": n_size,
            "comparisons": metrics["comparisons"],
            "local_assignments": metrics["local_assignments"],
            "vector_assignments": metrics["vector_assignments"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "generated_permutations": metrics["generated_permutations"],
            "prefix_tests": metrics["prefix_tests"],
            "overhead_ratio": algorithm.get_overhead_ratio(),
        }

    print("Algorithm X raw rows")
    print(
        "n, "
        "C_100, A_local_100, A_vector_100, T_100, "
        "C_10, A_local_10, A_vector_10, T_10, "
        "generated_10"
    )

    for n_size in range(3, 11):
        result_100 = run_single_experiment(n_size, 1.0)
        result_10 = run_single_experiment(n_size, 0.1)

        print(
            f"{n_size}, "
            f"{result_100['comparisons']}, "
            f"{result_100['local_assignments']}, "
            f"{result_100['vector_assignments']}, "
            f"{result_100['elapsed_seconds']}, "
            f"{result_10['comparisons']}, "
            f"{result_10['local_assignments']}, "
            f"{result_10['vector_assignments']}, "
            f"{result_10['elapsed_seconds']}, "
            f"{result_10['generated_permutations']}"
        )
