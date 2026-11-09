import argparse
import csv
from collections import defaultdict


def plot_metrics(path: str, output: str) -> None:
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = "overall" if row["scope"] == "overall" else f"weapon {row['weapon_id']}"
            series[label].append((int(row["period_index"]), float(row["period_points"])))

    for label, points in sorted(series.items()):
        points.sort()
        x_values, y_values = zip(*points)
        plt.plot(x_values, y_values, marker=".", label=label)
    plt.xlabel("Ten-minute period")
    plt.ylabel("Points earned in period")
    plt.title("Terraria combat training progress")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_metrics.csv")
    parser.add_argument("--output", default="training_progress.png")
    args = parser.parse_args()
    plot_metrics(args.input, args.output)