from __future__ import annotations

from typing import Callable, List, Optional, Sequence

try:
    from .instrumented_algorithm import InstrumentedAlgorithm
except ImportError:
    from instrumented_algorithm import InstrumentedAlgorithm


PrefixTest = Callable[[Sequence[int], int], bool]


"""
It is a Finite State Machine!
"""


class AlgorithmX(InstrumentedAlgorithm):
    """
    Algorithm X: lexicographic generation with restricted prefixes.

    This follows the TAOCP X2..X6 structure:
      X2 enter level
      X3 test prefix
      X4 increase k
      X5 increase a_k
      X6 decrease k

    Important:
    - The set of generated permutations is defined by the prefix tests.
    - We do NOT use generation_ratio/early-stop for the professor's experiments.
    - The cost of the checks t_i themselves should be disregarded from the
      counted comparisons/assignments, so the tests must not touch the counters.
    """

    def __init__(
        self,
        sorted_values: Sequence[int],
        prefix_tests: Sequence[PrefixTest],
    ) -> None:
        super().__init__()

        if not sorted_values:
            raise ValueError("sorted_values must not be empty.")
        if len(prefix_tests) != len(sorted_values):
            raise ValueError(
                "prefix_tests must contain exactly one test for each prefix size."
            )

        self.sorted_values = list(sorted_values)
        self.prefix_tests = list(prefix_tests)
        self.n_size = len(sorted_values)

        self._initialize_runtime_state()

    def _initialize_runtime_state(self) -> None:
        """
        1-based indexing to stay close to TAOCP notation.

        candidate_index_by_level[k] = q chosen at level k
        current_value_by_level[k]   = actual value chosen at level k
        next_available_index        = l array (cyclic linked list of unused items)
        previous_link_index_by_level[k] = u_k
        """
        self.candidate_index_by_level = [0] * (self.n_size + 1)
        self.current_value_by_level = [0] * (self.n_size + 1)
        self.next_available_index = list(range(1, self.n_size + 1)) + [0]
        self.previous_link_index_by_level = [0] * (self.n_size + 1)

        self.k_current_level = 1
        self.p_previous_index = 0
        self.q_current_candidate_index = 0
        self.current_state = "X2"

    def _reset_for_run(self) -> None:
        self.reset_metrics()
        self._initialize_runtime_state()

    def _run_current_prefix_test(self) -> bool:
        """
        Run t_k(a1,...,ak). The body of the test is NOT instrumented.
        We only count the fact that a prefix test was invoked.
        """
        self.record_prefix_test()
        current_test = self.prefix_tests[self.k_current_level - 1]
        return current_test(self.current_value_by_level, self.k_current_level)

    def _current_output_permutation(self) -> List[int]:
        return self.current_value_by_level[1 : self.n_size + 1]

    def generate(
        self,
        store_permutations: bool = False,
        max_outputs: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Generate all permutations accepted by the prefix tests.

        max_outputs is kept only as an optional debug convenience.
        It must NOT be used for the professor's 100% / 10% experiments.
        """
        if max_outputs is not None and max_outputs <= 0:
            raise ValueError("max_outputs must be positive when provided.")

        self._reset_for_run()
        generated_permutations: List[List[int]] = []

        self.start_timer()

        # FINITE STATE MACHINE EXECUTION
        while True:
            # X2. [Enter level k] Set p <- 0, q <- l0.
            if self.compare_equal(self.current_state, "X2"):
                self.p_previous_index = self.assign_local(0)
                self.q_current_candidate_index = self.assign_local(
                    self.next_available_index[0]
                )
                self.current_state = self.assign_local("X3")

            # X3. [Test a1...ak]
            elif self.compare_equal(self.current_state, "X3"):
                self.write_vector(
                    self.candidate_index_by_level,
                    self.k_current_level,
                    self.q_current_candidate_index,
                )

                current_value = self.sorted_values[self.q_current_candidate_index - 1]
                self.write_vector(
                    self.current_value_by_level,
                    self.k_current_level,
                    current_value,
                )

                # Se prefixo for inválido, rejeita e volta para X3. Se for válido, testa se é folha.
                #
                if self.compare_true(not self._run_current_prefix_test()):
                    self.record_rejected_prefix()
                    self.current_state = self.assign_local("X5")

                elif self.compare_equal(self.k_current_level, self.n_size):
                    self.record_generated_permutation()

                    if store_permutations:
                        generated_permutations.append(
                            self._current_output_permutation()
                        )

                    if max_outputs is not None and self.compare_greater_equal(
                        self.metrics.generated_permutations, max_outputs
                    ):
                        self.stop_timer()
                        return generated_permutations

                    self.current_state = self.assign_local("X6")

                else:
                    self.current_state = self.assign_local("X4")

            # X4. [Increase k]
            elif self.compare_equal(self.current_state, "X4"):
                self.write_vector(
                    self.previous_link_index_by_level,
                    self.k_current_level,
                    self.p_previous_index,
                )

                next_after_q = self.next_available_index[self.q_current_candidate_index]
                self.write_vector(
                    self.next_available_index,
                    self.p_previous_index,
                    next_after_q,
                )

                self.k_current_level = self.assign_local(self.k_current_level + 1)
                self.current_state = self.assign_local("X2")

            # X5. [Increase a_k]
            elif self.compare_equal(self.current_state, "X5"):
                self.p_previous_index = self.assign_local(
                    self.q_current_candidate_index
                )
                self.q_current_candidate_index = self.assign_local(
                    self.next_available_index[self.p_previous_index]
                )

                if self.compare_not_equal(self.q_current_candidate_index, 0):
                    self.current_state = self.assign_local("X3")
                else:
                    self.current_state = self.assign_local("X6")

            # X6. [Decrease k]
            elif self.compare_equal(self.current_state, "X6"):
                self.k_current_level = self.assign_local(self.k_current_level - 1)

                if self.compare_equal(self.k_current_level, 0):
                    self.stop_timer()
                    return generated_permutations

                self.p_previous_index = self.assign_local(
                    self.previous_link_index_by_level[self.k_current_level]
                )
                self.q_current_candidate_index = self.assign_local(
                    self.candidate_index_by_level[self.k_current_level]
                )

                self.write_vector(
                    self.next_available_index,
                    self.p_previous_index,
                    self.q_current_candidate_index,
                )
                self.current_state = self.assign_local("X5")

            else:
                self.stop_timer()
                raise RuntimeError(f"Unknown state: {self.current_state}")

    def get_raw_row(self) -> List[float]:
        return [
            self.n_size,
            self.metrics.comparisons,
            self.metrics.local_assignments,
            self.metrics.vector_assignments,
            self.metrics.elapsed_seconds,
        ]

    def get_overhead_ratio(self) -> float:
        """
        Suggested internal indicator:
        prefix tests performed per generated permutation.
        """
        if self.metrics.generated_permutations == 0:
            return 0.0
        return self.metrics.prefix_tests / self.metrics.generated_permutations
