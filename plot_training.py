import argparse
import csv
from collections import defaultdict


def plot_metrics(path: str, output: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    period_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    cumulative_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            name = row.get("weapon_name") or f"weapon {row['weapon_id']}"
            period_series[name].append((int(row["period_index"]), float(row["period_points"])))
            cumulative_series[name].append((int(row["period_index"]), float(row["cumulative_points"])))

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for label, points in sorted(period_series.items()):
        points.sort()
        x_values, y_values = zip(*points)
        axes[0].plot(x_values, y_values, marker=".", label=label)
    for label, points in sorted(cumulative_series.items()):
        points.sort()
        x_values, y_values = zip(*points)
        axes[1].plot(x_values, y_values, marker=".", label=label)
    axes[0].set_title("Points earned per weapon period")
    axes[0].set_ylabel("Period points")
    axes[1].set_title("Cumulative progress by weapon")
    axes[1].set_xlabel("Weapon-specific ten-minute period")
    axes[1].set_ylabel("Cumulative points")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        if period_series:
            axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_metrics.csv")
    parser.add_argument("--output", default="training_progress.png")
    args = parser.parse_args()
    plot_metrics(args.input, args.output)