import argparse
import csv
from collections import defaultdict


def plot_metrics(path: str, output: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = "overall" if row["scope"] == "overall" else f"weapon {row['weapon_id']}"
            series[label].append((int(row["period_index"]), float(row["period_points"])))

    figure, axes = plt.subplots()
    for label, points in sorted(series.items()):
        points.sort()
        x_values, y_values = zip(*points)
        axes.plot(x_values, y_values, marker=".", label=label)
    axes.set_xlabel("Ten-minute period")
    axes.set_ylabel("Points earned in period")
    axes.set_title("Terraria combat training progress")
    axes.grid(True, alpha=0.25)
    if series:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_metrics.csv")
    parser.add_argument("--output", default="training_progress.png")
    args = parser.parse_args()
    plot_metrics(args.input, args.output)