#!/usr/bin/env python3
"""Render XYZ position and velocity tracking comparisons from analytic CSV exports."""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .plot_force_comparison import matplotlib, plt  # type: ignore
except ImportError:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from plot_force_comparison import matplotlib, plt  # type: ignore

if plt is None or matplotlib is None:
    raise RuntimeError(
        "matplotlib is not available. Ensure plot_force_comparison.py dependencies are satisfied."
    )

LINE_COLORS = {
    "reference": "#1f77b4",
    "actual": "#ff7f0e",
}
ERROR_COLOR = "#d62728"
SUBPLOT_W = 5.0
SUBPLOT_H = 4.0
TIME_KEYS = ("time_sec", "time", "timestamp")

POSITION_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("X Position", "quad_ref_x", "quad_actual_x"),
    ("Y Position", "quad_ref_y", "quad_actual_y"),
    ("Z Position", "quad_ref_z", "quad_actual_z"),
)

VELOCITY_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("X Velocity", "quad_ref_vx", "quad_actual_vx"),
    ("Y Velocity", "quad_ref_vy", "quad_actual_vy"),
    ("Z Velocity", "quad_ref_vz", "quad_actual_vz"),
)

PAYLOAD_POSITION_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("Load X Position", "payload_ref_x", "payload_actual_x"),
    ("Load Y Position", "payload_ref_y", "payload_actual_y"),
    ("Load Z Position", "payload_ref_z", "payload_actual_z"),
)

PAYLOAD_VELOCITY_FIELDS: Sequence[Tuple[str, str, str]] = (
    ("Load X Velocity", "payload_ref_vx", "payload_actual_vx"),
    ("Load Y Velocity", "payload_ref_vy", "payload_actual_vy"),
    ("Load Z Velocity", "payload_ref_vz", "payload_actual_vz"),
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Input CSV exported by MPC analytic logger.")
    parser.add_argument("quad_position_png", help="Output PNG for quad XYZ position tracking.")
    parser.add_argument("quad_velocity_png", help="Output PNG for quad XYZ velocity tracking.")
    parser.add_argument("payload_position_png", help="Output PNG for payload XYZ position tracking.")
    parser.add_argument("payload_velocity_png", help="Output PNG for payload XYZ velocity tracking.")
    parser.add_argument("quad_error_png", help="Output PNG for quad position/velocity errors.")
    parser.add_argument("payload_error_png", help="Output PNG for payload position/velocity errors.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (primarily for debugging).",
    )
    return parser.parse_args(argv)


def read_rows(csv_path: str) -> Tuple[List[Dict[str, float]], List[str]]:
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"CSV file not found: {csv_path}")

    rows: List[Dict[str, float]] = []
    fieldnames: List[str] = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        for row in reader:
            converted: Dict[str, float] = {}
            for key in row:
                value = row.get(key, "")
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = float("nan")
            rows.append(converted)
    if not rows:
        raise RuntimeError(f"No data rows found in {csv_path}")
    return rows, fieldnames


def extract_time_series(rows: Sequence[Dict[str, float]]) -> List[float]:
    """Return time stamps normalized to start at zero."""
    raw_times: List[float] = []
    for row in rows:
        time_value = math.nan
        for key in TIME_KEYS:
            candidate = row.get(key, math.nan)
            if math.isfinite(candidate):
                time_value = candidate
                break
        raw_times.append(time_value)
    base = next((value for value in raw_times if math.isfinite(value)), 0.0)
    series: List[float] = []
    for value in raw_times:
        if math.isfinite(value):
            series.append(value - base)
        else:
            series.append(math.nan)
    return series


def ensure_fields_available(fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise RuntimeError(f"CSV file is missing columns required for plotting: {', '.join(missing)}")


def build_axis_data(
    rows: Sequence[Dict[str, float]],
    mapping: Sequence[Tuple[str, str, str]],
) -> Dict[str, Dict[str, List[float]]]:
    data: Dict[str, Dict[str, List[float]]] = {}
    for label, ref_key, actual_key in mapping:
        axis_entry = {"reference": [], "actual": []}
        for row in rows:
            axis_entry["reference"].append(row.get(ref_key, math.nan))
            axis_entry["actual"].append(row.get(actual_key, math.nan))
        data[label] = axis_entry
    return data


def build_error_series(
    rows: Sequence[Dict[str, float]],
    mapping: Sequence[Tuple[str, str, str]],
) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    for label, ref_key, actual_key in mapping:
        values: List[float] = []
        for row in rows:
            ref_val = row.get(ref_key, math.nan)
            act_val = row.get(actual_key, math.nan)
            if math.isfinite(ref_val) and math.isfinite(act_val):
                values.append(act_val - ref_val)
            else:
                values.append(math.nan)
        series[label] = values
    return series


def render_tracking_plot(
    times: Sequence[float],
    series: Dict[str, Dict[str, List[float]]],
    output_path: str,
    title: str,
    ylabel: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 8), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for axis, (axis_label, samples) in zip(axes, series.items()):
        axis.plot(times, samples["reference"], label="Reference", color=LINE_COLORS["reference"], linewidth=1.5)
        axis.plot(
            times,
            samples["actual"],
            label="Actual",
            color=LINE_COLORS["actual"],
            linewidth=1.2,
            linestyle="--",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(axis_label)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_error_panels(
    times: Sequence[float],
    label_value_pairs: Sequence[Tuple[str, List[float]]],
    output_path: str,
    title: str,
    ylabel: str,
    show: bool,
) -> None:
    if len(label_value_pairs) != 6:
        raise ValueError("Expected six error series for rendering.")

    fig, axes = plt.subplots(3, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H * 3), sharex=True)
    axes = axes.flatten()
    for idx, (axis, (label, values)) in enumerate(zip(axes, label_value_pairs)):
        axis.plot(times, values, color=ERROR_COLOR, linewidth=1.4)
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_title(label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if idx % 2 == 0:
            axis.set_ylabel(f"{ylabel}\n(Position)")
        else:
            axis.set_ylabel(f"{ylabel}\n(Velocity)")
    axes[-1].set_xlabel("Time [s]")
    axes[-2].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.96])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows, fieldnames = read_rows(args.csv)
    times = extract_time_series(rows)

    ensure_fields_available(
        fieldnames,
        [ref for _, ref, _ in POSITION_FIELDS]
        + [act for _, _, act in POSITION_FIELDS]
        + [ref for _, ref, _ in VELOCITY_FIELDS]
        + [act for _, _, act in VELOCITY_FIELDS]
        + [ref for _, ref, _ in PAYLOAD_POSITION_FIELDS]
        + [act for _, _, act in PAYLOAD_POSITION_FIELDS]
        + [ref for _, ref, _ in PAYLOAD_VELOCITY_FIELDS]
        + [act for _, _, act in PAYLOAD_VELOCITY_FIELDS],
    )

    quad_position_series = build_axis_data(rows, POSITION_FIELDS)
    quad_velocity_series = build_axis_data(rows, VELOCITY_FIELDS)
    payload_position_series = build_axis_data(rows, PAYLOAD_POSITION_FIELDS)
    payload_velocity_series = build_axis_data(rows, PAYLOAD_VELOCITY_FIELDS)

    render_tracking_plot(
        times,
        quad_position_series,
        args.quad_position_png,
        title="Quad Position Tracking",
        ylabel="Position [m]",
        show=args.show,
    )
    render_tracking_plot(
        times,
        quad_velocity_series,
        args.quad_velocity_png,
        title="Quad Velocity Tracking",
        ylabel="Velocity [m/s]",
        show=args.show,
    )
    render_tracking_plot(
        times,
        payload_position_series,
        args.payload_position_png,
        title="Load Position Tracking",
        ylabel="Position [m]",
        show=args.show,
    )
    render_tracking_plot(
        times,
        payload_velocity_series,
        args.payload_velocity_png,
        title="Load Velocity Tracking",
        ylabel="Velocity [m/s]",
        show=args.show,
    )

    quad_pos_errors = build_error_series(rows, POSITION_FIELDS)
    quad_vel_errors = build_error_series(rows, VELOCITY_FIELDS)
    payload_pos_errors = build_error_series(rows, PAYLOAD_POSITION_FIELDS)
    payload_vel_errors = build_error_series(rows, PAYLOAD_VELOCITY_FIELDS)

    quad_error_pairs: List[Tuple[str, List[float]]] = [
        ("Quad Position Error X", quad_pos_errors["X Position"]),
        ("Quad Velocity Error X", quad_vel_errors["X Velocity"]),
        ("Quad Position Error Y", quad_pos_errors["Y Position"]),
        ("Quad Velocity Error Y", quad_vel_errors["Y Velocity"]),
        ("Quad Position Error Z", quad_pos_errors["Z Position"]),
        ("Quad Velocity Error Z", quad_vel_errors["Z Velocity"]),
    ]
    payload_error_pairs: List[Tuple[str, List[float]]] = [
        ("Load Position Error X", payload_pos_errors["Load X Position"]),
        ("Load Velocity Error X", payload_vel_errors["Load X Velocity"]),
        ("Load Position Error Y", payload_pos_errors["Load Y Position"]),
        ("Load Velocity Error Y", payload_vel_errors["Load Y Velocity"]),
        ("Load Position Error Z", payload_pos_errors["Load Z Position"]),
        ("Load Velocity Error Z", payload_vel_errors["Load Z Velocity"]),
    ]

    render_error_panels(
        times,
        quad_error_pairs,
        args.quad_error_png,
        title="Quad Position / Velocity Errors",
        ylabel="Error",
        show=args.show,
    )
    render_error_panels(
        times,
        payload_error_pairs,
        args.payload_error_png,
        title="Load Position / Velocity Errors",
        ylabel="Error",
        show=args.show,
    )


if __name__ == "__main__":
    main()
