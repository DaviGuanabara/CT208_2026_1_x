from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent

SCRIPT_NAME = Path(__file__).stem
PLOT_DIR = OUT_DIR / SCRIPT_NAME

RAW_ROWS_CSV = OUT_DIR / "data" / "x_raw_rows.csv"
RATIO_SWEEP_CSV = OUT_DIR / "data" / "x_ratio_sweep_n10.csv"


def ensure_output_directory() -> None:
    """
    Create the output directory for this script if it does not exist.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the two CSV files required for plotting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - raw_rows_df: data for size sweep (n from 3 to 10)
        - ratio_sweep_df: data for acceptance ratio sweep at fixed n
    """
    if not RAW_ROWS_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {RAW_ROWS_CSV}")

    if not RATIO_SWEEP_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {RATIO_SWEEP_CSV}")

    raw_rows_df = pd.read_csv(RAW_ROWS_CSV)
    ratio_sweep_df = pd.read_csv(RATIO_SWEEP_CSV)

    return raw_rows_df, ratio_sweep_df


def save_plot(filename: str) -> None:
    """
    Save the current matplotlib figure inside out/<script_name>/.
    """
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_cost_vs_n_log(
    raw_rows_df: pd.DataFrame,
    metric_100_column: str,
    metric_10_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot cost vs n using logarithmic scale on the y-axis.

    This is useful when the curves only separate clearly for larger n.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        raw_rows_df["n"],
        raw_rows_df[metric_100_column],
        marker="o",
        linewidth=2,
        label="X 100%",
    )
    plt.plot(
        raw_rows_df["n"],
        raw_rows_df[metric_10_column],
        marker="s",
        linewidth=2,
        label="X 10%",
    )

    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    save_plot(filename)


def build_normalized_ratio_dataframe(ratio_sweep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized metrics relative to the 100% run.

    For each metric, this computes:
        metric_normalized = metric / metric_at_100%

    This allows direct visual comparison with the identity line y = x.
    """
    full_row = ratio_sweep_df.loc[ratio_sweep_df["requested_ratio"] == 1.0]

    if full_row.empty:
        raise ValueError("The ratio sweep CSV does not contain the 100% reference row.")

    full_row = full_row.iloc[0]

    normalized_df = ratio_sweep_df.copy()

    normalized_df["accepted_percent"] = normalized_df["actual_ratio"] * 100.0
    normalized_df["identity_ratio"] = normalized_df["actual_ratio"]

    normalized_df["comparisons_normalized"] = (
        normalized_df["comparisons"] / full_row["comparisons"]
    )
    normalized_df["local_assignments_normalized"] = (
        normalized_df["local_assignments"] / full_row["local_assignments"]
    )
    normalized_df["vector_assignments_normalized"] = (
        normalized_df["vector_assignments"] / full_row["vector_assignments"]
    )
    normalized_df["elapsed_seconds_normalized"] = (
        normalized_df["elapsed_seconds"] / full_row["elapsed_seconds"]
    )

    return normalized_df


def plot_normalized_cost_vs_acceptance(
    normalized_df: pd.DataFrame,
    normalized_metric_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot normalized cost against accepted percentage.

    The identity line y = x is plotted as a reference.
    If the measured curve is above the identity line, then the cost is
    decreasing more slowly than the accepted percentage.
    """
    plt.figure(figsize=(8, 5))

    x_percent = normalized_df["accepted_percent"]
    y_identity = normalized_df["identity_ratio"]
    y_metric = normalized_df[normalized_metric_column]

    plt.plot(
        x_percent,
        y_identity,
        linestyle="--",
        linewidth=2,
        label="Identity line (ideal proportional reduction)",
    )

    plt.plot(
        x_percent,
        y_metric,
        marker="o",
        linewidth=2,
        label=y_label,
    )

    plt.xlabel("Accepted permutations (%)")
    plt.ylabel("Normalized cost")
    plt.title(title)
    plt.ylim(0, max(1.15, float(y_metric.max()) * 1.05))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    save_plot(filename)


def plot_comparison_with_secondary_percentage_axis(
    normalized_df: pd.DataFrame,
    normalized_metric_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot normalized cost with a secondary y-axis in percentage.

    This is useful when you want the audience to immediately read values
    such as 0.8 as 80%.
    """
    fig, ax_left = plt.subplots(figsize=(8, 5))

    x_percent = normalized_df["accepted_percent"]
    y_identity = normalized_df["identity_ratio"]
    y_metric = normalized_df[normalized_metric_column]

    line_identity = ax_left.plot(
        x_percent,
        y_identity,
        linestyle="--",
        linewidth=2,
        label="Identity line",
    )[0]

    line_metric = ax_left.plot(
        x_percent,
        y_metric,
        marker="o",
        linewidth=2,
        label=y_label,
    )[0]

    ax_left.set_xlabel("Accepted permutations (%)")
    ax_left.set_ylabel("Normalized cost (0 to 1)")
    ax_left.set_title(title)
    ax_left.set_ylim(0, max(1.15, float(y_metric.max()) * 1.05))
    ax_left.grid(True, linestyle="--", alpha=0.4)

    ax_right = ax_left.twinx()
    ax_right.set_ylim(ax_left.get_ylim()[0] * 100, ax_left.get_ylim()[1] * 100)
    ax_right.set_ylabel("Normalized cost (%)")

    lines = [line_identity, line_metric]
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="best")

    save_plot(filename)


def save_normalized_ratio_csv(normalized_df: pd.DataFrame) -> None:
    """
    Save the normalized ratio sweep data used to generate the plots.
    """
    normalized_df.to_csv(PLOT_DIR / "x_ratio_sweep_n10_normalized.csv", index=False)


def main() -> None:
    """
    Generate all plots for Algorithm X from the CSV data already stored in out/.
    """
    ensure_output_directory()

    raw_rows_df, ratio_sweep_df = load_csv_data()
    normalized_df = build_normalized_ratio_dataframe(ratio_sweep_df)

    save_normalized_ratio_csv(normalized_df)

    # ------------------------------------------------------------------
    # 1. Cost vs n, using log scale on y
    # ------------------------------------------------------------------
    plot_cost_vs_n_log(
        raw_rows_df=raw_rows_df,
        metric_100_column="C_100",
        metric_10_column="C_10",
        y_label="Comparisons",
        title="Algorithm X: Comparisons vs n",
        filename="comparisons_vs_n_log.png",
    )

    plot_cost_vs_n_log(
        raw_rows_df=raw_rows_df,
        metric_100_column="A_local_100",
        metric_10_column="A_local_10",
        y_label="Local assignments",
        title="Algorithm X: Local assignments vs n",
        filename="local_assignments_vs_n_log.png",
    )

    plot_cost_vs_n_log(
        raw_rows_df=raw_rows_df,
        metric_100_column="A_vector_100",
        metric_10_column="A_vector_10",
        y_label="Vector assignments",
        title="Algorithm X: Vector assignments vs n",
        filename="vector_assignments_vs_n_log.png",
    )

    plot_cost_vs_n_log(
        raw_rows_df=raw_rows_df,
        metric_100_column="T_100",
        metric_10_column="T_10",
        y_label="Elapsed time (seconds)",
        title="Algorithm X: Time vs n",
        filename="time_vs_n_log.png",
    )

    # ------------------------------------------------------------------
    # 2. Normalized cost vs accepted percentage
    # ------------------------------------------------------------------
    plot_normalized_cost_vs_acceptance(
        normalized_df=normalized_df,
        normalized_metric_column="comparisons_normalized",
        y_label="Comparisons / comparisons(100%)",
        title="Algorithm X: Normalized comparisons vs accepted percentage",
        filename="normalized_comparisons_vs_acceptance.png",
    )

    plot_normalized_cost_vs_acceptance(
        normalized_df=normalized_df,
        normalized_metric_column="local_assignments_normalized",
        y_label="Local assignments / local assignments(100%)",
        title="Algorithm X: Normalized local assignments vs accepted percentage",
        filename="normalized_local_assignments_vs_acceptance.png",
    )

    plot_normalized_cost_vs_acceptance(
        normalized_df=normalized_df,
        normalized_metric_column="vector_assignments_normalized",
        y_label="Vector assignments / vector assignments(100%)",
        title="Algorithm X: Normalized vector assignments vs accepted percentage",
        filename="normalized_vector_assignments_vs_acceptance.png",
    )

    plot_normalized_cost_vs_acceptance(
        normalized_df=normalized_df,
        normalized_metric_column="elapsed_seconds_normalized",
        y_label="Time / time(100%)",
        title="Algorithm X: Normalized time vs accepted percentage",
        filename="normalized_time_vs_acceptance.png",
    )

    # ------------------------------------------------------------------
    # 3. One extra version with secondary y-axis in percentage
    # ------------------------------------------------------------------
    plot_comparison_with_secondary_percentage_axis(
        normalized_df=normalized_df,
        normalized_metric_column="comparisons_normalized",
        y_label="Comparisons / comparisons(100%)",
        title="Algorithm X: Normalized comparisons vs accepted percentage",
        filename="normalized_comparisons_vs_acceptance_dual_axis.png",
    )

    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
