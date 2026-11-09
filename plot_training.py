import argparse
import csv
from collections import defaultdict
import re
from pathlib import Path


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


def plot_weapon_metrics(path: str, output_directory: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("scope") == "weapon":
                grouped[int(row["weapon_id"])].append(row)

    output_folder = Path(output_directory)
    output_folder.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for weapon_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["period_index"]))
        weapon_name = rows[-1].get("weapon_name") or f"weapon_{weapon_id}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", weapon_name).strip("._") or "unknown"
        output_path = output_folder / f"weapon_{weapon_id}_{safe_name}.png"
        periods = [int(row["period_index"]) + 1 for row in rows]
        period_points = [float(row["period_points"]) for row in rows]
        cumulative_points = [float(row["cumulative_points"]) for row in rows]

        figure, axes = plt.subplots(2, 1, figsize=(9, 7))
        axes[0].plot(periods, period_points, marker=".")
        axes[0].set_title(f"{weapon_name}: points per 10-minute period")
        axes[0].set_ylabel("Period points")
        axes[1].plot(periods, cumulative_points, marker=".", color="tab:green")
        axes[1].set_title(f"{weapon_name}: cumulative training points")
        axes[1].set_xlabel("Training period for this weapon")
        axes[1].set_ylabel("Cumulative points")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        outputs.append(output_path)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training_metrics.csv")
    parser.add_argument("--output", default="training_progress.png")
    args = parser.parse_args()
    plot_metrics(args.input, args.output)