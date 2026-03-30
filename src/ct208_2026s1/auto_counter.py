"""
Low-level counter infrastructure.

This class should know nothing about Algorithm X itself.
It only stores raw metrics and offers generic increment helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CounterMetrics:
    comparisons: int = 0
    local_assignments: int = 0
    vector_assignments: int = 0
    generated_permutations: int = 0
    rejected_prefixes: int = 0
    prefix_tests: int = 0
    elapsed_seconds: float = 0.0


class AutoCounter:
    """
    Base class for raw metric storage.

    This class is intentionally simple.
    It does not try to "magically" detect all operations in Python.
    Instead, higher layers call these methods explicitly.
    """

    def __init__(self) -> None:
        self.metrics = CounterMetrics()

    def reset_metrics(self) -> None:
        self.metrics = CounterMetrics()

    def inc_comparisons(self, amount: int = 1) -> None:
        self.metrics.comparisons += amount

    def inc_local_assignments(self, amount: int = 1) -> None:
        self.metrics.local_assignments += amount

    def inc_vector_assignments(self, amount: int = 1) -> None:
        self.metrics.vector_assignments += amount

    def inc_generated_permutations(self, amount: int = 1) -> None:
        self.metrics.generated_permutations += amount

    def inc_rejected_prefixes(self, amount: int = 1) -> None:
        self.metrics.rejected_prefixes += amount

    def inc_prefix_tests(self, amount: int = 1) -> None:
        self.metrics.prefix_tests += amount

    def set_elapsed_seconds(self, elapsed_seconds: float) -> None:
        self.metrics.elapsed_seconds = elapsed_seconds

    def get_metrics_dict(self) -> dict:
        return {
            "comparisons": self.metrics.comparisons,
            "local_assignments": self.metrics.local_assignments,
            "vector_assignments": self.metrics.vector_assignments,
            "generated_permutations": self.metrics.generated_permutations,
            "rejected_prefixes": self.metrics.rejected_prefixes,
            "prefix_tests": self.metrics.prefix_tests,
            "elapsed_seconds": self.metrics.elapsed_seconds,
        }
