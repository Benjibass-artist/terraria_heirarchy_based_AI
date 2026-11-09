import csv
from pathlib import Path


class TrainingMetricsLogger:
    """Append-only ten-minute point history for weapon and overall progress."""

    fields = (
        "scope",
        "weapon_id",
        "period_index",
        "global_step",
        "cumulative_points",
        "period_points",
    )

    def __init__(self, path: str | Path = "training_metrics.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.fields).writeheader()
        self.weapon_cumulative: dict[int, float] = {}
        self.weapon_periods: dict[int, int] = {}
        self.overall_cumulative = 0.0
        self.overall_period = 0
        self._restore_state()

    def _restore_state(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                period = int(row["period_index"])
                cumulative = float(row["cumulative_points"])
                if row["scope"] == "overall":
                    self.overall_cumulative = cumulative
                    self.overall_period = max(self.overall_period, period + 1)
                else:
                    weapon_id = int(row["weapon_id"])
                    if period >= self.weapon_periods.get(weapon_id, 0):
                        self.weapon_cumulative[weapon_id] = cumulative
                        self.weapon_periods[weapon_id] = period + 1

    def record_period(
        self,
        weapon_id: int,
        global_step: int,
        cumulative_points: float,
    ) -> None:
        weapon_id = int(weapon_id)
        previous = self.weapon_cumulative.get(weapon_id, 0.0)
        period_points = cumulative_points - previous
        period_index = self.weapon_periods.get(weapon_id, 0)
        self._append(
            "weapon",
            weapon_id,
            period_index,
            global_step,
            cumulative_points,
            period_points,
        )
        self.weapon_cumulative[weapon_id] = cumulative_points
        self.weapon_periods[weapon_id] = period_index + 1

    def record_overall_period(self, global_step: int, cumulative_points: float) -> None:
        period_points = cumulative_points - self.overall_cumulative
        self._append(
            "overall",
            -1,
            self.overall_period,
            global_step,
            cumulative_points,
            period_points,
        )
        self.overall_cumulative = cumulative_points
        self.overall_period += 1

    def _append(
        self,
        scope: str,
        weapon_id: int,
        period_index: int,
        global_step: int,
        cumulative_points: float,
        period_points: float,
    ) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=self.fields).writerow(
                {
                    "scope": scope,
                    "weapon_id": weapon_id,
                    "period_index": period_index,
                    "global_step": global_step,
                    "cumulative_points": cumulative_points,
                    "period_points": period_points,
                }
            )