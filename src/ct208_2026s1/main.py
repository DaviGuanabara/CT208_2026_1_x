from __future__ import annotations

import csv
from math import factorial
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

try:
    from .algorithm_x import AlgorithmX
    from .prefix_tests import (
        build_all_true_tests,
        build_first_value_allowed_tests,
        build_pair_prefix_tests_for_target_ratio,
    )
except ImportError:
    from algorithm_x import AlgorithmX
    from prefix_tests import (
        build_all_true_tests,
        build_first_value_allowed_tests,
        build_pair_prefix_tests_for_target_ratio,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

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
    input_values = list(range(1, n_size + 1))
    algorithm = AlgorithmX(input_values, prefix_tests)
    algorithm.generate(store_permutations=False)

    metrics = algorithm.get_metrics_dict()
    generated = metrics["generated_permutations"]
    actual_ratio = generated / factorial(n_size)

    return {
        "n": n_size,
        "scenario": scenario_name,
        "requested_ratio": requested_ratio,
        "actual_ratio": actual_ratio,
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
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_raw_rows_for_spreadsheet() -> List[Dict[str, float]]:
    """
    Produces rows for n = 3..10:
      - X 100%: all_true tests
      - X ~10%: pair-prefix filter aiming at ~10%
    """
    rows: List[Dict[str, float]] = []

    for n_size in range(3, 11):
        result_100 = run_case(
            n_size=n_size,
            prefix_tests=build_all_true_tests(n_size),
            requested_ratio=1.0,
            scenario_name="X_100",
        )

        prefix_tests_10, _ = build_pair_prefix_tests_for_target_ratio(
            n_size=n_size,
            target_ratio=0.1,
        )
        result_10 = run_case(
            n_size=n_size,
            prefix_tests=prefix_tests_10,
            requested_ratio=0.1,
            scenario_name="X_10_approx",
        )

        rows.append(
            {
                "n": n_size,
                "C_100": result_100["comparisons"],
                "A_local_100": result_100["local_assignments"],
                "A_vector_100": result_100["vector_assignments"],
                "T_100": result_100["elapsed_seconds"],
                "C_10": result_10["comparisons"],
                "A_local_10": result_10["local_assignments"],
                "A_vector_10": result_10["vector_assignments"],
                "T_10": result_10["elapsed_seconds"],
                "generated_10": result_10["generated_permutations"],
                "actual_ratio_10": result_10["actual_ratio"],
                "prefix_tests_100": result_100["prefix_tests"],
                "prefix_tests_10": result_10["prefix_tests"],
                "rejected_prefixes_10": result_10["rejected_prefixes"],
            }
        )

    return rows


def build_ratio_sweep_fixed_n10() -> List[Dict[str, float]]:
    """
    Fixed n=10, vary the generated percentage using real filters.

    We use first-element filters:
      allowed first values = {1..m}
    so the ratio is exactly m/10.
    """
    n_size = 10
    baseline = run_case(
        n_size=n_size,
        prefix_tests=build_all_true_tests(n_size),
        requested_ratio=1.0,
        scenario_name="X_ratio_sweep_baseline",
    )

    rows: List[Dict[str, float]] = []

    for allowed_count in range(1, 11):
        tests, exact_ratio = build_first_value_allowed_tests(
            n_size=n_size,
            allowed_first_values=range(1, allowed_count + 1),
        )
        result = run_case(
            n_size=n_size,
            prefix_tests=tests,
            requested_ratio=exact_ratio,
            scenario_name=f"X_ratio_{exact_ratio:.1f}",
        )

        row = dict(result)
        for metric_key in METRIC_KEYS:
            base_value = baseline[metric_key]
            if base_value == 0:
                row[f"overhead_{metric_key}_pct"] = 0.0
            else:
                row[f"overhead_{metric_key}_pct"] = (
                    (result[metric_key] / base_value) - exact_ratio
                ) * 100.0

        rows.append(row)

    return rows


def plot_n_sweep(raw_rows: List[Dict[str, float]]) -> None:
    ns = [int(row["n"]) for row in raw_rows]

    metric_map = {
        "C": ("C_100", "C_10", "comparisons"),
        "A_local": ("A_local_100", "A_local_10", "local assignments"),
        "A_vector": ("A_vector_100", "A_vector_10", "vector assignments"),
        "T": ("T_100", "T_10", "time (seconds)"),
    }

    for stem, (col_100, col_10, ylabel) in metric_map.items():
        plt.figure(figsize=(8, 5))
        plt.plot(ns, [row[col_100] for row in raw_rows], marker="o", label="100%")
        plt.plot(ns, [row[col_10] for row in raw_rows], marker="o", label="~10%")
        plt.yscale("log")
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.title(f"Algorithm X: {ylabel} vs n")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"x_{stem.lower()}_vs_n.png", dpi=160)
        plt.close()


def plot_ratio_sweep(ratio_rows: List[Dict[str, float]]) -> None:
    ratios = [row["actual_ratio"] * 100.0 for row in ratio_rows]

    metric_map = {
        "comparisons": "comparisons",
        "local_assignments": "local assignments",
        "vector_assignments": "vector assignments",
        "elapsed_seconds": "time (seconds)",
    }

    for metric_key, ylabel in metric_map.items():
        plt.figure(figsize=(8, 5))
        plt.plot(
            ratios,
            [row[metric_key] for row in ratio_rows],
            marker="o",
            label="measured",
        )
        plt.xlabel("accepted percentage (%)")
        plt.ylabel(ylabel)
        plt.title(f"Algorithm X, n=10: {ylabel} vs accepted percentage")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"x_ratio_sweep_{metric_key}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            ratios,
            [row[f"overhead_{metric_key}_pct"] for row in ratio_rows],
            marker="o",
            label="overhead %",
        )
        plt.xlabel("accepted percentage (%)")
        plt.ylabel("overhead (%)")
        plt.title(f"Algorithm X, n=10: overhead of {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"x_ratio_sweep_overhead_{metric_key}.png", dpi=160)
        plt.close()


def print_raw_rows_for_copy_paste(raw_rows: List[Dict[str, float]]) -> None:
    print("Algorithm X rows for spreadsheet")
    print(
        "n, C_100, A_local_100, A_vector_100, T_100, "
        "C_10, A_local_10, A_vector_10, T_10, generated_10, actual_ratio_10"
    )
    for row in raw_rows:
        print(
            f"{int(row['n'])}, "
            f"{int(row['C_100'])}, "
            f"{int(row['A_local_100'])}, "
            f"{int(row['A_vector_100'])}, "
            f"{row['T_100']}, "
            f"{int(row['C_10'])}, "
            f"{int(row['A_local_10'])}, "
            f"{int(row['A_vector_10'])}, "
            f"{row['T_10']}, "
            f"{int(row['generated_10'])}, "
            f"{row['actual_ratio_10']:.6f}"
        )


def main() -> None:
    raw_rows = build_raw_rows_for_spreadsheet()
    ratio_rows = build_ratio_sweep_fixed_n10()

    write_csv(OUT_DIR / "x_raw_rows.csv", raw_rows)
    write_csv(OUT_DIR / "x_ratio_sweep_n10.csv", ratio_rows)

    plot_n_sweep(raw_rows)
    plot_ratio_sweep(ratio_rows)

    print_raw_rows_for_copy_paste(raw_rows)
    print()
    print(f"Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
