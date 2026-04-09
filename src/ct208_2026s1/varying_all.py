from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

try:
    from .algorithm_x import AlgorithmX
    from .prefix_tests import (
        build_all_true_tests,
        build_first_value_allowed_tests,
    )
except ImportError:
    from algorithm_x import AlgorithmX
    from prefix_tests import (
        build_all_true_tests,
        build_first_value_allowed_tests,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "out" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "varying_all.csv"

METRIC_KEYS = [
    "comparisons",
    "local_assignments",
    "vector_assignments",
    "elapsed_seconds",
]


def run_case(
    n_size: int,
    prefix_tests,
    requested_ratio: float,
    scenario_name: str,
) -> Dict[str, float]:
    """
    Run one Algorithm X experiment and return all relevant metrics.
    """
    input_values = list(range(1, n_size + 1))
    algorithm = AlgorithmX(input_values, prefix_tests)
    algorithm.generate(store_permutations=False)

    metrics = algorithm.get_metrics_dict()
    generated = metrics["generated_permutations"]

    # Since the filters are constructed using allowed first values,
    # the exact total accepted ratio is generated / n!.
    #
    # We avoid factorial here because the algorithm already tells us
    # how many complete permutations were generated, and for this script
    # we only need the measured ratio relative to the unrestricted run.
    return {
        "n": n_size,
        "scenario": scenario_name,
        "requested_ratio": requested_ratio,
        "generated_permutations": generated,
        "prefix_tests": metrics["prefix_tests"],
        "rejected_prefixes": metrics["rejected_prefixes"],
        "comparisons": metrics["comparisons"],
        "local_assignments": metrics["local_assignments"],
        "vector_assignments": metrics["vector_assignments"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "internal_overhead_ratio": algorithm.get_overhead_ratio(),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    """
    Write rows to a CSV file.
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_varying_all_rows() -> List[Dict[str, float]]:
    """
    Build one unified dataset varying:
    - n from 3 to 10
    - requested ratio from 10% to 100%

    For each n, we use first-element filters:
        allowed first values = {1..m}

    This gives an exact ratio m/n.

    Therefore:
    - for n = 10, we get exactly 10%, 20%, ..., 100%
    - for other n, we get the available exact fractions induced by m/n

    Example:
    - n = 5 gives 20%, 40%, 60%, 80%, 100%
    - n = 8 gives 12.5%, 25%, ..., 100%

    To keep the output aligned with the user's request, we keep only rows
    whose actual ratio is at least 10%.
    """
    all_rows: List[Dict[str, float]] = []

    for n_size in range(3, 11):
        baseline_tests = build_all_true_tests(n_size)
        baseline_result = run_case(
            n_size=n_size,
            prefix_tests=baseline_tests,
            requested_ratio=1.0,
            scenario_name=f"X_n{n_size}_ratio_1.0",
        )

        baseline_generated = baseline_result["generated_permutations"]

        for allowed_count in range(1, n_size + 1):
            prefix_tests, exact_ratio = build_first_value_allowed_tests(
                n_size=n_size,
                allowed_first_values=range(1, allowed_count + 1),
            )

            if exact_ratio < 0.1:
                continue

            result = run_case(
                n_size=n_size,
                prefix_tests=prefix_tests,
                requested_ratio=exact_ratio,
                scenario_name=f"X_n{n_size}_ratio_{exact_ratio:.4f}",
            )

            row = dict(result)

            if baseline_generated == 0:
                row["actual_ratio"] = 0.0
            else:
                row["actual_ratio"] = (
                    result["generated_permutations"] / baseline_generated
                )

            for metric_key in METRIC_KEYS:
                baseline_metric_value = baseline_result[metric_key]

                if baseline_metric_value == 0:
                    row[f"{metric_key}_normalized"] = 0.0
                    row[f"overhead_{metric_key}_pct"] = 0.0
                else:
                    normalized_value = result[metric_key] / baseline_metric_value
                    row[f"{metric_key}_normalized"] = normalized_value
                    row[f"overhead_{metric_key}_pct"] = (
                        normalized_value - row["actual_ratio"]
                    ) * 100.0

            all_rows.append(row)

    return all_rows


def print_summary(rows: List[Dict[str, float]]) -> None:
    """
    Print a small summary of the generated dataset.
    """
    print("Unified dataset generated.")
    print(f"Number of rows: {len(rows)}")
    print(f"Output file: {OUTPUT_CSV}")


def main() -> None:
    total_start = time.perf_counter()

    rows = build_varying_all_rows()
    write_csv(OUTPUT_CSV, rows)

    print_summary(rows)

    total_end = time.perf_counter()
    total_time = total_end - total_start

    print(f"TOTAL EXECUTION TIME (FULL SCRIPT): {total_time:.6f} seconds")


if __name__ == "__main__":
    main()
