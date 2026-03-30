"""
Didactic class-based implementation of Algorithm X.

This version follows the PDF structure:
X2 -> X3 -> X4 / X5 / X6

It inherits from InstrumentedAlgorithm so that the counting logic is
centralized and the algorithm body remains easier to read.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

from instrumented_algorithm import InstrumentedAlgorithm


PrefixTest = Callable[[Sequence[int], int], bool]


class AlgorithmX(InstrumentedAlgorithm):
    """
    Algorithm X: lexicographic generation with prefix pruning.

    Main idea:
    - Build the permutation one level at a time.
    - At level k, choose a candidate.
    - Test the prefix.
    - If valid, go deeper.
    - If invalid, skip that branch.
    - If no candidate remains, backtrack.
    """

    def __init__(
        self,
        sorted_values: Sequence[int],
        prefix_tests: Sequence[PrefixTest],
    ) -> None:
        super().__init__()

        if len(sorted_values) == 0:
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
        Initialize all runtime vectors and control variables.

        We use 1-based indexing to stay close to the PDF.
        """
        # A in the PDF: store the chosen candidate index at each level.
        self.candidate_index_by_level = [0] * (self.n_size + 1)

        # Extra helper for didactic clarity:
        # store the actual chosen values at each level.
        self.current_value_by_level = [0] * (self.n_size + 1)

        # L in the PDF: linked list of available candidate indexes.
        # Initial form:
        # 0 -> 1 -> 2 -> ... -> n -> 0
        self.next_available_index = list(range(1, self.n_size + 1)) + [0]

        # U in the PDF: stores the previous link for each level.
        self.previous_link_index_by_level = [0] * (self.n_size + 1)

        self.k_current_level = 1
        self.p_previous_index = 0
        self.q_current_candidate_index = 0
        self.current_state = "X2"

    def _reset_for_run(self) -> None:
        """
        Reset counters and runtime state before each execution.
        """
        self.reset_metrics()
        self._initialize_runtime_state()

    def _run_current_prefix_test(self) -> bool:
        """
        Run the prefix test corresponding to the current level.
        """
        self.record_prefix_test()
        current_test = self.prefix_tests[self.k_current_level - 1]
        return current_test(self.current_value_by_level, self.k_current_level)

    def _current_output_permutation(self) -> List[int]:
        """
        Return the current complete permutation as a standard Python list.
        """
        return self.current_value_by_level[1 : self.n_size + 1]

    def generate(
        self,
        generation_ratio: float = 1.0,
        store_permutations: bool = True,
    ) -> List[List[int]]:
        """
        Execute Algorithm X.

        Parameters
        ----------
        generation_ratio:
            Fraction of the total output to generate.
            Example:
            - 1.0 means 100%
            - 0.1 means 10%

        store_permutations:
            If True, return the generated permutations.
            If False, only count them.
        """
        if generation_ratio <= 0 or generation_ratio > 1:
            raise ValueError("generation_ratio must be in the interval (0, 1].")

        self._reset_for_run()

        total_permutations = math.factorial(self.n_size)
        target_output_count = max(1, math.ceil(total_permutations * generation_ratio))

        generated_permutations: List[List[int]] = []

        self.start_timer()

        while True:
            # ----------------------------------------------------------
            # X2: [enter level k]
            #
            # p <- 0
            # q <- l_0
            # ----------------------------------------------------------
            if self.compare_equal(self.current_state, "X2"):
                self.p_previous_index = self.assign_local(0)
                self.q_current_candidate_index = self.assign_local(
                    self.next_available_index[0]
                )
                self.current_state = self.assign_local("X3")

            # ----------------------------------------------------------
            # X3: [test a1, ..., a_k]
            #
            # a_k <- q
            # if !t_k(...) go X5
            # if k = n visit and go X6
            # else go X4
            # ----------------------------------------------------------
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

                if self.compare_true(not self._run_current_prefix_test()):
                    self.record_rejected_prefix()
                    self.current_state = self.assign_local("X5")

                elif self.compare_equal(self.k_current_level, self.n_size):
                    self.record_generated_permutation()

                    if store_permutations:
                        generated_permutations.append(
                            self._current_output_permutation()
                        )

                    if self.compare_greater_equal(
                        self.metrics.generated_permutations,
                        target_output_count,
                    ):
                        self.stop_timer()
                        return generated_permutations

                    self.current_state = self.assign_local("X6")

                else:
                    self.current_state = self.assign_local("X4")

            # ----------------------------------------------------------
            # X4: descend one level
            #
            # u_k <- p
            # l_p <- l_q
            # k++
            # go X2
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # X5: try next candidate at the same level
            #
            # p <- q
            # q <- l_p
            # if q != 0 go X3
            # else go X6
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # X6: backtrack
            #
            # k--
            # if k = 0 halt
            # p <- u_k
            # q <- a_k
            # l_p <- q
            # go X5
            # ----------------------------------------------------------
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
        """
        Return a spreadsheet-friendly row:
        [n, comparisons, local_assignments, vector_assignments, elapsed_seconds]
        """
        return [
            self.n_size,
            self.metrics.comparisons,
            self.metrics.local_assignments,
            self.metrics.vector_assignments,
            self.metrics.elapsed_seconds,
        ]

    def get_overhead_ratio(self) -> float:
        """
        A simple overhead indicator.

        This is just a suggested first metric:
        how many prefix tests were performed per generated permutation.
        """
        if self.metrics.generated_permutations == 0:
            return 0.0

        return self.metrics.prefix_tests / self.metrics.generated_permutations
