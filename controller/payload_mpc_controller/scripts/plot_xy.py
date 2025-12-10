#!/usr/bin/env python3
"""Render analytic XY trajectory comparisons as PNG images."""

import csv
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

try:
    from .plot_force_comparison import matplotlib, plt  # type: ignore
except ImportError:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from plot_force_comparison import matplotlib, plt  # type: ignore

if plt is None or matplotlib is None:
    raise RuntimeError(
        "matplotlib is not available. Install matplotlib or ensure plot_force_comparison.py dependencies are satisfied."
    )

from matplotlib import patches  # type: ignore

WIDTH = 10.0
HEIGHT = 7.2
LINE_WIDTH = 2.5
COLORS = {
    "quad_ref": "#FFD43B",
    "quad_actual": "#1F77B4",
    "payload_ref": "#FF6F61",
    "payload_actual": "#2CA02C",
}
STYLES = {
    "quad_ref": (8, 5),
    "quad_actual": None,
    "payload_ref": (8, 5),
    "payload_actual": None,
}
LEGEND_ITEMS = [
    ("Quad Ref", "quad_ref"),
    ("Quad Actual", "quad_actual"),
    ("Payload Ref", "payload_ref"),
    ("Payload Actual", "payload_actual"),
]
SERIES_NAMES: Sequence[Tuple[str, str, str]] = (
    ("quad_ref_x", "quad_ref_y", "quad_ref"),
    ("quad_actual_x", "quad_actual_y", "quad_actual"),
    ("payload_ref_x", "payload_ref_y", "payload_ref"),
    ("payload_actual_x", "payload_actual_y", "payload_actual"),
)


def _ensure_args() -> Tuple[str, float, float, str]:
    if len(sys.argv) < 5:
        print("Usage: plot_xy.py <csv> <quad_rmse> <payload_rmse> <output_png>")
        sys.exit(1)
    csv_path = sys.argv[1]
    quad_rmse = float(sys.argv[2])
    payload_rmse = float(sys.argv[3])
    output_path = sys.argv[4]
    return csv_path, quad_rmse, payload_rmse, output_path


def _read_rows(csv_path: str) -> List[Dict[str, float]]:
    if not os.path.isfile(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(row[k]) if row[k] else float("nan") for k in row})
    return rows


def _build_series(rows: Sequence[Dict[str, float]]) -> Tuple[Dict[str, List[Tuple[float, float]]], float, float, float, float]:
    series: Dict[str, List[Tuple[float, float]]] = {name: [] for _, _, name in SERIES_NAMES}
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    for x_key, y_key, name in SERIES_NAMES:
        for row in rows:
            x = row.get(x_key, float("nan"))
            y = row.get(y_key, float("nan"))
            if math.isfinite(x) and math.isfinite(y):
                series[name].append((x, y))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    return series, min_x, max_x, min_y, max_y


def _configure_main_axis(ax_main, bounds):
    min_x, max_x, min_y, max_y = bounds
    if not math.isfinite(min_x) or not math.isfinite(max_x) or not math.isfinite(min_y) or not math.isfinite(max_y):
        print("No valid data to plot")
        sys.exit(0)

    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    pad_x = span_x * 0.05
    pad_y = span_y * 0.05

    ax_main.set_xlim(min_x - pad_x, max_x + pad_x)
    ax_main.set_ylim(min_y - pad_y, max_y + pad_y)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_xlabel("X [m]", fontsize=14)
    ax_main.set_ylabel("Y [m]", fontsize=14)
    ax_main.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    rect = patches.Rectangle(
        (min_x - pad_x, min_y - pad_y),
        span_x + 2 * pad_x,
        span_y + 2 * pad_y,
        linewidth=1.2,
        edgecolor="#D0D0D0",
        facecolor="none",
    )
    ax_main.add_patch(rect)


def _draw_series(ax_main, series: Dict[str, List[Tuple[float, float]]]):
    line_map: Dict[str, object] = {}
    for name, points in series.items():
        if len(points) < 2:
            continue
        xs, ys = zip(*points)
        line, = ax_main.plot(xs, ys, color=COLORS.get(name, "#000000"), linewidth=LINE_WIDTH)
        dash = STYLES.get(name)
        if dash:
            line.set_dashes(dash)
        line_map[name] = line
    return line_map


def main() -> None:
    csv_path, quad_rmse, payload_rmse, output_path = _ensure_args()
    rows = _read_rows(csv_path)
    series, min_x, max_x, min_y, max_y = _build_series(rows)
    if not any(len(points) > 1 for points in series.values()):
        print("No valid data to plot")
        sys.exit(0)

    fig, ax_main = plt.subplots(figsize=(WIDTH, HEIGHT), facecolor="#FFFFFF")
    _configure_main_axis(ax_main, (min_x, max_x, min_y, max_y))
    line_map = _draw_series(ax_main, series)
    ax_main.set_title("Analytic XY Trajectory Comparison", fontsize=16)

    handles = []
    labels = []
    for label, name in LEGEND_ITEMS:
        line = line_map.get(name)
        if line is not None:
            handles.append(line)
            labels.append(label)
    if handles:
        legend = ax_main.legend(handles, labels, loc="lower right", framealpha=0.9)
        legend.get_frame().set_edgecolor("#B0B0B0")

    rmse_text = f"Quad RMSE: {quad_rmse:.3f} m\nPayload RMSE: {payload_rmse:.3f} m"
    ax_main.text(
        0.02,
        0.02,
        rmse_text,
        transform=ax_main.transAxes,
        fontsize=11,
        color="#333333",
        va="bottom",
    )

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved XY plot to {output_path}")


if __name__ == "__main__":
    main()
