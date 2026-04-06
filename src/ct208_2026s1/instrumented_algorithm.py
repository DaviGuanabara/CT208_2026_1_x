"""
Intermediate instrumentation layer.

This class encapsulates counted operations so that the final algorithm
is much cleaner and closer to the PDF pseudo-code.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import List, Sequence, TypeVar

try:
    from .auto_counter import AutoCounter
except ImportError:
    from auto_counter import AutoCounter



T = TypeVar("T")


class InstrumentedAlgorithm(AutoCounter):
    """
    Intermediate layer between raw counting and concrete algorithms.

    It offers counted primitives such as:
    - local assignment
    - vector assignment
    - comparisons
    - timing
    """

    def __init__(self) -> None:
        super().__init__()
        self._start_time: float | None = None

    def assign_local(self, value: T) -> T:
        """
        Count one local assignment and return the assigned value.
        """
        self.inc_local_assignments()
        return value

    def write_vector(self, vector: List[T], index: int, value: T) -> None:
        """
        Count one vector assignment and write the value.
        """
        self.inc_vector_assignments()
        vector[index] = value

    def compare_equal(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left == right

    def compare_not_equal(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left != right

    def compare_less_than(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left < right  # type: ignore

    def compare_less_equal(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left <= right  # type: ignore

    def compare_greater_than(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left > right  # type: ignore

    def compare_greater_equal(self, left: T, right: T) -> bool:
        self.inc_comparisons()
        return left >= right  # type: ignore

    def compare_true(self, condition: bool) -> bool:
        """
        Count a comparison that is already expressed as a boolean condition.
        """
        self.inc_comparisons()
        return condition

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()

    def stop_timer(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was not started.")

        elapsed_seconds = time.perf_counter() - self._start_time
        self.set_elapsed_seconds(elapsed_seconds)
        self._start_time = None
        return elapsed_seconds

    def record_generated_permutation(self) -> None:
        self.inc_generated_permutations()

    def record_rejected_prefix(self) -> None:
        self.inc_rejected_prefixes()

    def record_prefix_test(self) -> None:
        self.inc_prefix_tests()
