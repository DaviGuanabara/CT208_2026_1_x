from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent

SCRIPT_NAME = Path(__file__).stem
PLOT_DIR = OUT_DIR / SCRIPT_NAME

VARYING_ALL_CSV = OUT_DIR / "data" / "varying_all.csv"


def ensure_output_directory() -> None:
    """
    Create the output directory for this script if it does not exist.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_data() -> pd.DataFrame:
    """
    Load the unified varying_all.csv dataset.

    Returns
    -------
    pd.DataFrame
        Unified dataset varying both n and ratio.
    """
    if not VARYING_ALL_CSV.exists():
        raise FileNotFoundError(f"Missing CSV file: {VARYING_ALL_CSV}")

    return pd.read_csv(VARYING_ALL_CSV)


def save_plot(filename: str) -> None:
    """
    Save the current matplotlib figure inside out/<script_name>/.
    """
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def build_normalized_dataframe(varying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalized metrics relative to the 100% row for each fixed n.

    For each n, the row with actual_ratio == 1.0 is used as baseline.
    """
    normalized_frames = []

    for n_value, group_df in varying_df.groupby("n"):
        baseline_df = group_df.loc[group_df["actual_ratio"] == 1.0]

        if baseline_df.empty:
            raise ValueError(f"Missing 100% baseline row for n = {n_value}")

        baseline_row = baseline_df.iloc[0]
        normalized_group = group_df.copy()

        normalized_group["accepted_percent"] = normalized_group["actual_ratio"] * 100.0
        normalized_group["identity_ratio"] = normalized_group["actual_ratio"]

        normalized_group["comparisons_normalized"] = (
            normalized_group["comparisons"] / baseline_row["comparisons"]
        )
        normalized_group["local_assignments_normalized"] = (
            normalized_group["local_assignments"] / baseline_row["local_assignments"]
        )
        normalized_group["vector_assignments_normalized"] = (
            normalized_group["vector_assignments"] / baseline_row["vector_assignments"]
        )
        normalized_group["elapsed_seconds_normalized"] = (
            normalized_group["elapsed_seconds"] / baseline_row["elapsed_seconds"]
        )

        normalized_frames.append(normalized_group)

    normalized_df = pd.concat(normalized_frames, ignore_index=True)
    return normalized_df.sort_values(["n", "accepted_percent"])


def save_normalized_csv(normalized_df: pd.DataFrame) -> None:
    """
    Save the normalized dataset used for plotting.
    """
    normalized_df.to_csv(PLOT_DIR / "varying_all_normalized.csv", index=False)


def plot_metric_vs_ratio_by_n(
    varying_df: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    filename: str,
    use_log_scale: bool = False,
) -> None:
    """
    Plot a raw metric against accepted percentage, with one curve per n.
    """
    plt.figure(figsize=(9, 6))

    for n_value, group_df in varying_df.groupby("n"):
        sorted_group = group_df.sort_values("actual_ratio")
        plt.plot(
            sorted_group["actual_ratio"] * 100.0,
            sorted_group[metric_column],
            marker="o",
            linewidth=2,
            label=f"n={n_value}",
        )

    if use_log_scale:
        plt.yscale("log")

    plt.xlabel("Accepted permutations (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(title="Input size")

    save_plot(filename)


def plot_normalized_metric_vs_ratio_by_n(
    normalized_df: pd.DataFrame,
    normalized_metric_column: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot a normalized metric against accepted percentage, with one curve per n.
    The identity line is included as a reference.
    """
    plt.figure(figsize=(9, 6))

    all_x_values = sorted(normalized_df["accepted_percent"].unique())
    identity_line = [x / 100.0 for x in all_x_values]

    plt.plot(
        all_x_values,
        identity_line,
        linestyle="--",
        linewidth=2,
        label="Identity line",
        color="black",
    )

    for n_value, group_df in normalized_df.groupby("n"):
        sorted_group = group_df.sort_values("accepted_percent")
        plt.plot(
            sorted_group["accepted_percent"],
            sorted_group[normalized_metric_column],
            marker="o",
            linewidth=2,
            label=f"n={n_value}",
        )

    plt.xlabel("Accepted permutations (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.ylim(0, max(1.15, float(normalized_df[normalized_metric_column].max()) * 1.05))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Input size")

    save_plot(filename)


def plot_metric_vs_n_for_selected_ratios(
    varying_df: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    filename: str,
    selected_ratios: list[float],
    use_log_scale: bool = True,
) -> None:
    """
    Plot metric vs n for selected accepted ratios.

    Because not every n has exactly the same possible ratios, this function
    selects rows whose actual_ratio is close to the requested ratio.
    """
    plt.figure(figsize=(9, 6))

    tolerance = 0.03

    for selected_ratio in selected_ratios:
        subset_rows = []

        for n_value, group_df in varying_df.groupby("n"):
            group_df = group_df.copy()
            group_df["ratio_distance"] = (
                group_df["actual_ratio"] - selected_ratio
            ).abs()

            best_row = group_df.sort_values("ratio_distance").iloc[0]

            if best_row["ratio_distance"] <= tolerance:
                subset_rows.append(best_row)

        if subset_rows:
            subset_df = pd.DataFrame(subset_rows).sort_values("n")
            plt.plot(
                subset_df["n"],
                subset_df[metric_column],
                marker="o",
                linewidth=2,
                label=f"{int(selected_ratio * 100)}%",
            )

    if use_log_scale:
        plt.yscale("log")

    plt.xlabel("n")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(title="Accepted %")

    save_plot(filename)


def main() -> None:
    """
    Generate plots from the unified varying_all.csv dataset.
    """
    ensure_output_directory()

    varying_df = load_csv_data()
    normalized_df = build_normalized_dataframe(varying_df)

    save_normalized_csv(normalized_df)

    # ------------------------------------------------------------------
    # 1. Raw metric vs accepted percentage, one curve per n
    # ------------------------------------------------------------------
    plot_metric_vs_ratio_by_n(
        varying_df=varying_df,
        metric_column="comparisons",
        y_label="Comparisons",
        title="Algorithm X: Comparisons vs accepted percentage",
        filename="comparisons_vs_ratio_by_n.png",
        use_log_scale=True,
    )

    plot_metric_vs_ratio_by_n(
        varying_df=varying_df,
        metric_column="local_assignments",
        y_label="Local assignments",
        title="Algorithm X: Local assignments vs accepted percentage",
        filename="local_assignments_vs_ratio_by_n.png",
        use_log_scale=True,
    )

    plot_metric_vs_ratio_by_n(
        varying_df=varying_df,
        metric_column="vector_assignments",
        y_label="Vector assignments",
        title="Algorithm X: Vector assignments vs accepted percentage",
        filename="vector_assignments_vs_ratio_by_n.png",
        use_log_scale=True,
    )

    plot_metric_vs_ratio_by_n(
        varying_df=varying_df,
        metric_column="elapsed_seconds",
        y_label="Elapsed time (seconds)",
        title="Algorithm X: Time vs accepted percentage",
        filename="time_vs_ratio_by_n.png",
        use_log_scale=True,
    )

    # ------------------------------------------------------------------
    # 2. Normalized metric vs accepted percentage, one curve per n
    # ------------------------------------------------------------------
    plot_normalized_metric_vs_ratio_by_n(
        normalized_df=normalized_df,
        normalized_metric_column="comparisons_normalized",
        y_label="Normalized comparisons",
        title="Algorithm X: Normalized comparisons vs accepted percentage",
        filename="normalized_comparisons_vs_ratio_by_n.png",
    )

    plot_normalized_metric_vs_ratio_by_n(
        normalized_df=normalized_df,
        normalized_metric_column="local_assignments_normalized",
        y_label="Normalized local assignments",
        title="Algorithm X: Normalized local assignments vs accepted percentage",
        filename="normalized_local_assignments_vs_ratio_by_n.png",
    )

    plot_normalized_metric_vs_ratio_by_n(
        normalized_df=normalized_df,
        normalized_metric_column="vector_assignments_normalized",
        y_label="Normalized vector assignments",
        title="Algorithm X: Normalized vector assignments vs accepted percentage",
        filename="normalized_vector_assignments_vs_ratio_by_n.png",
    )

    plot_normalized_metric_vs_ratio_by_n(
        normalized_df=normalized_df,
        normalized_metric_column="elapsed_seconds_normalized",
        y_label="Normalized time",
        title="Algorithm X: Normalized time vs accepted percentage",
        filename="normalized_time_vs_ratio_by_n.png",
    )

    # ------------------------------------------------------------------
    # 3. Metric vs n for selected percentages
    # ------------------------------------------------------------------
    selected_ratios = [0.25, 0.5, 0.75, 1.0]

    plot_metric_vs_n_for_selected_ratios(
        varying_df=varying_df,
        metric_column="comparisons",
        y_label="Comparisons",
        title="Algorithm X: Comparisons vs n for selected accepted percentages",
        filename="comparisons_vs_n_selected_ratios.png",
        selected_ratios=selected_ratios,
        use_log_scale=True,
    )

    plot_metric_vs_n_for_selected_ratios(
        varying_df=varying_df,
        metric_column="elapsed_seconds",
        y_label="Elapsed time (seconds)",
        title="Algorithm X: Time vs n for selected accepted percentages",
        filename="time_vs_n_selected_ratios.png",
        selected_ratios=selected_ratios,
        use_log_scale=True,
    )

    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
