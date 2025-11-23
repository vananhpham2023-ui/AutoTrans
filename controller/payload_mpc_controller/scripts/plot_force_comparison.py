#!/usr/bin/env python3
"""Utility to render force comparison plots from recorded CSV data."""

import argparse
import csv
import math
import os
import sys
import time
import types
from datetime import timedelta, tzinfo
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from matplotlib.ticker import AutoMinorLocator


def _install_dateutil_stub() -> None:
    """Create a lightweight stub for python-dateutil to satisfy matplotlib imports."""
    if "dateutil" in sys.modules:
        return

    dateutil_module = types.ModuleType("dateutil")
    dateutil_module.__all__ = ["tz", "parser", "relativedelta", "rrule"]
    dateutil_module.__version__ = "9.9-stub"

    tz_module = types.ModuleType("dateutil.tz")

    class _FixedOffset(tzinfo):
        def __init__(self, offset_seconds: int = 0, name: str = "UTC") -> None:
            self._offset = timedelta(seconds=offset_seconds)
            self._name = name

        def utcoffset(self, dt):  # type: ignore[override]
            return self._offset

        def tzname(self, dt):  # type: ignore[override]
            return self._name

        def dst(self, dt):  # type: ignore[override]
            return timedelta(0)

    def _coerce_seconds(value) -> int:
        if isinstance(value, timedelta):
            return int(value.total_seconds())
        return int(value)

    def tzutc():
        return _FixedOffset(0, "UTC")

    def tzoffset(name, offset):
        seconds = _coerce_seconds(offset)
        return _FixedOffset(seconds, name or f"UTC{seconds // 3600:+03d}")

    def tzlocal():
        try:
            if time.daylight and time.localtime().tm_isdst:
                seconds = -time.altzone
            else:
                seconds = -time.timezone
        except Exception:
            seconds = 0
        return _FixedOffset(seconds, "local")

    def gettz(name=None):
        if name in (None, "local"):
            return tzlocal()
        return tzutc()

    tz_module.tzutc = tzutc
    tz_module.tzoffset = tzoffset
    tz_module.tzlocal = tzlocal
    tz_module.gettz = gettz
    tz_module.UTC = tzutc()
    tz_module.__all__ = ["tzutc", "tzoffset", "tzlocal", "gettz", "UTC"]

    parser_module = types.ModuleType("dateutil.parser")

    def parse(*args, **kwargs):
        raise ValueError("dateutil parser is unavailable in the lightweight stub.")

    parser_module.parse = parse
    parser_module.__all__ = ["parse"]

    relativedelta_module = types.ModuleType("dateutil.relativedelta")

    class relativedelta:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    relativedelta_module.relativedelta = relativedelta
    relativedelta_module.__all__ = ["relativedelta"]

    rrule_module = types.ModuleType("dateutil.rrule")

    class rrule:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __iter__(self):
            return iter(())

    class rruleset(list):
        pass

    class weekday:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __call__(self, *args, **kwargs):
            return self.__class__(*args, **kwargs)

    def rrulewrapper(*args, **kwargs):
        return rrule(*args, **kwargs)

    weekdays = []
    for idx, name in enumerate(("MO", "TU", "WE", "TH", "FR", "SA", "SU")):
        day = weekday(idx)
        setattr(rrule_module, name, day)
        weekdays.append(day)

    freq_names = (
        ("YEARLY", 0),
        ("MONTHLY", 1),
        ("WEEKLY", 2),
        ("DAILY", 3),
        ("HOURLY", 4),
        ("MINUTELY", 5),
        ("SECONDLY", 6),
    )
    for name, value in freq_names:
        setattr(rrule_module, name, value)

    rrule_module.rrule = rrule
    rrule_module.rruleset = rruleset
    rrule_module.weekday = weekday
    rrule_module.rrulewrapper = rrulewrapper
    rrule_module.__all__ = ["rrule", "rruleset", "weekday", "rrulewrapper"] + [
        name for name in ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
    ] + [name for name, _ in freq_names]

    dateutil_module.tz = tz_module
    dateutil_module.parser = parser_module
    dateutil_module.relativedelta = relativedelta_module
    dateutil_module.rrule = rrule_module

    sys.modules["dateutil"] = dateutil_module
    sys.modules["dateutil.tz"] = tz_module
    sys.modules["dateutil.parser"] = parser_module
    sys.modules["dateutil.relativedelta"] = relativedelta_module
    sys.modules["dateutil.rrule"] = rrule_module


def _load_matplotlib():
    try:
        import matplotlib  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "dateutil":  # pragma: no cover - stub path
            _install_dateutil_stub()
            import matplotlib  # type: ignore
        else:
            raise

    # 强制使用非交互式后端Agg，确保在非主线程中也能正常工作
    # 即使有DISPLAY环境变量，也使用Agg后端以避免GUI线程问题
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

    return matplotlib, plt


_MATPLOTLIB_ERROR: Optional[BaseException] = None

try:
    matplotlib, plt = _load_matplotlib()
except Exception as exc:  # pragma: no cover - matplotlib import guard
    matplotlib = None
    plt = None
    _MATPLOTLIB_ERROR = exc


CSV_FIELDS = [
    "time",
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
    "fl_est_x",
    "fl_est_y",
    "fl_est_z",
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
    "fq_est_x",
    "fq_est_y",
    "fq_est_z",
]
WIND_FIELDS = ["wind_x", "wind_y", "wind_z"]
ERROR_COLOR = "#d62728"
SUBPLOT_W = 5.0
SUBPLOT_H = 4.0

PLOT_LAYOUT = [
    ("Load Force X", ("fl_true", 0), ("fl_est", 0)),
    ("Load Force Y", ("fl_true", 1), ("fl_est", 1)),
    ("Load Force Z", ("fl_true", 2), ("fl_est", 2)),
    ("Quad Force X", ("fq_true", 0), ("fq_est", 0)),
    ("Quad Force Y", ("fq_true", 1), ("fq_est", 1)),
    ("Quad Force Z", ("fq_true", 2), ("fq_est", 2)),
]


def load_samples(csv_path: str) -> List[Dict[str, Sequence[float]]]:
    """Load samples from CSV into a structured list."""
    samples: List[Dict[str, Sequence[float]]] = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = set(CSV_FIELDS) - set(fieldnames)
        if missing:
            raise RuntimeError(f"CSV file {csv_path} is missing columns: {sorted(missing)}")
        has_wind = all(field in fieldnames for field in WIND_FIELDS)
        for row in reader:
            sample = {
                "time": float(row["time"]),
                "fl_true": [
                    float(row["fl_true_x"]),
                    float(row["fl_true_y"]),
                    float(row["fl_true_z"]),
                ],
                "fl_est": [
                    float(row["fl_est_x"]),
                    float(row["fl_est_y"]),
                    float(row["fl_est_z"]),
                ],
                "fq_true": [
                    float(row["fq_true_x"]),
                    float(row["fq_true_y"]),
                    float(row["fq_true_z"]),
                ],
                "fq_est": [
                    float(row["fq_est_x"]),
                    float(row["fq_est_y"]),
                    float(row["fq_est_z"]),
                ],
            }
            if has_wind:
                sample["wind"] = [
                    float(row["wind_x"]),
                    float(row["wind_y"]),
                    float(row["wind_z"]),
                ]
            samples.append(sample)
    return samples


def compute_rmse(samples: Sequence[Dict[str, Sequence[float]]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute RMSE values for load (fl) and quad (fq) force components."""
    if not samples:
        return {
            "fl": {"components": np.zeros(3), "total": np.array([0.0])},
            "fq": {"components": np.zeros(3), "total": np.array([0.0])},
        }

    fl_true = np.array([s["fl_true"] for s in samples], dtype=float)
    fl_est = np.array([s["fl_est"] for s in samples], dtype=float)
    fq_true = np.array([s["fq_true"] for s in samples], dtype=float)
    fq_est = np.array([s["fq_est"] for s in samples], dtype=float)

    fl_error = fl_est - fl_true
    fq_error = fq_est - fq_true

    fl_rmse_components = np.sqrt(np.mean(np.square(fl_error), axis=0))
    fq_rmse_components = np.sqrt(np.mean(np.square(fq_error), axis=0))

    fl_rmse_total = np.sqrt(np.mean(np.sum(np.square(fl_error), axis=1)))
    fq_rmse_total = np.sqrt(np.mean(np.sum(np.square(fq_error), axis=1)))

    return {
        "fl": {"components": fl_rmse_components, "total": np.array([fl_rmse_total])},
        "fq": {"components": fq_rmse_components, "total": np.array([fq_rmse_total])},
    }


def _ensure_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is not available. Install matplotlib or run in an environment with GUI support."
        )


def render_force_plot(
    samples: Sequence[Dict[str, Sequence[float]]],
    output_path: Optional[str] = None,
    total_output_path: Optional[str] = None,
    show: bool = False,
    rmse: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    title: str = "External Force Comparison",
) -> None:
    """Render a multi-axis plot comparing true and estimated forces."""
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")

    base_time = samples[0]["time"]
    times = np.array([s["time"] - base_time for s in samples], dtype=float)

    data_cache: Dict[str, np.ndarray] = {}
    for key in ("fl_true", "fl_est", "fq_true", "fq_est"):
        data_cache[key] = np.array([s[key] for s in samples], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for axis, (label, true_key, est_key) in zip(axes, PLOT_LAYOUT):
        true_data = data_cache[true_key[0]][:, true_key[1]]
        est_data = data_cache[est_key[0]][:, est_key[1]]

        axis.plot(times, true_data, label="True", color="#1f77b4", linewidth=1.5)
        axis.plot(times, est_data, label="Estimated", color="#ff7f0e", linewidth=1.2, linestyle="--")
        axis.set_ylabel("Force [N]")
        axis.set_title(label)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Time [s]")
    axes[-2].set_xlabel("Time [s]")
    axes[-3].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right")
    fig.suptitle(title, fontsize=16)

    if rmse is not None:
        fl_rmse = rmse["fl"]["components"]
        fq_rmse = rmse["fq"]["components"]
        fl_total = rmse["fl"]["total"][0]
        fq_total = rmse["fq"]["total"][0]
        fig.text(
            0.5,
            0.02,
            (
                f"Load RMSE [N]: total={fl_total:.3f}, "
                f"x={fl_rmse[0]:.3f}, y={fl_rmse[1]:.3f}, z={fl_rmse[2]:.3f} | "
                f"Quad RMSE [N]: total={fq_total:.3f}, "
                f"x={fq_rmse[0]:.3f}, y={fq_rmse[1]:.3f}, z={fq_rmse[2]:.3f}"
            ),
            ha="center",
            fontsize=12,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    total_fig = None
    if total_output_path:
        total_fig, total_axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        total_axes = np.atleast_1d(total_axes).flatten()
        total_pairs = [
            ("Total Load Force", "fl_true", "fl_est"),
            ("Total Quad Force", "fq_true", "fq_est"),
        ]
        for axis, (label, true_key, est_key) in zip(total_axes, total_pairs):
            true_total = np.linalg.norm(data_cache[true_key], axis=1)
            est_total = np.linalg.norm(data_cache[est_key], axis=1)
            axis.plot(times, true_total, label="True", color="#1f77b4", linewidth=1.5)
            axis.plot(
                times,
                est_total,
                label="Estimated",
                color="#ff7f0e",
                linewidth=1.2,
                linestyle="--",
            )
            axis.set_title(label)
            axis.set_ylabel("Force [N]")
            axis.set_xlabel("Time [s]")
            axis.legend(loc="upper right")
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        total_fig.suptitle(title, fontsize=16)
        total_fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if total_output_path and total_fig is not None:
        os.makedirs(os.path.dirname(total_output_path) or ".", exist_ok=True)
        total_fig.savefig(total_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        if total_fig is not None:
            plt.close(total_fig)


def render_force_error_plot(
    samples: Sequence[Dict[str, Sequence[float]]],
    output_path: Optional[str] = None,
    total_output_path: Optional[str] = None,
    show: bool = False,
    title: str = "External Force Estimation Errors",
) -> None:
    """Render force estimation error plots for load and quad components."""
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")

    base_time = samples[0]["time"]
    times = np.array([s["time"] - base_time for s in samples], dtype=float)

    data_cache: Dict[str, np.ndarray] = {}
    for key in ("fl_true", "fl_est", "fq_true", "fq_est"):
        data_cache[key] = np.array([s[key] for s in samples], dtype=float)

    error_cache = {
        "fl": data_cache["fl_est"] - data_cache["fl_true"],
        "fq": data_cache["fq_est"] - data_cache["fq_true"],
    }

    error_layout = [
        ("Load Force Error X", ("fl", 0)),
        ("Load Force Error Y", ("fl", 1)),
        ("Load Force Error Z", ("fl", 2)),
        ("Quad Force Error X", ("fq", 0)),
        ("Quad Force Error Y", ("fq", 1)),
        ("Quad Force Error Z", ("fq", 2)),
    ]

    fig_width = SUBPLOT_W * 3
    fig_height = SUBPLOT_H * 2
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height), sharex=True)
    axes = axes.flatten()
    max_time = times[-1] if len(times) else 1.0
    ticks = np.linspace(0.0, max_time, num=5)
    for axis, (label, (err_key, idx)) in zip(axes, error_layout):
        axis.plot(times, error_cache[err_key][:, idx], color=ERROR_COLOR, linewidth=1.4)
        axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        axis.set_ylabel("Force Error [N]")
        axis.set_title(label)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.set_xlim(0.0, max_time)
        axis.set_xticks(ticks)
        axis.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    axes[-1].set_xlabel("Time [s]")
    axes[-2].set_xlabel("Time [s]")
    axes[-3].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    total_fig = None
    if total_output_path:
        total_fig, total_axes = plt.subplots(1, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H), sharex=True)
        total_axes = np.atleast_1d(total_axes).flatten()
        load_total_error = np.linalg.norm(error_cache["fl"], axis=1)
        quad_total_error = np.linalg.norm(error_cache["fq"], axis=1)
        total_pairs = [
            ("Load Total Force Error", load_total_error),
            ("Quad Total Force Error", quad_total_error),
        ]
        for axis, (label, values) in zip(total_axes, total_pairs):
            axis.plot(times, values, color=ERROR_COLOR, linewidth=1.4)
            axis.set_ylabel("Force Error [N]")
            axis.set_xlabel("Time [s]")
            axis.set_title(label)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.set_xlim(0.0, max_time)
            axis.set_xticks(ticks)
            axis.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        total_fig.suptitle(title, fontsize=16)
        total_fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
        if total_output_path:
            os.makedirs(os.path.dirname(total_output_path) or ".", exist_ok=True)
            total_fig.savefig(total_output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        if total_fig is not None:
            plt.close(total_fig)


def render_wind_velocity_plot(
    samples: Sequence[Dict[str, Sequence[float]]],
    output_path: str,
    show: bool = False,
    title: str = "Wind Field Velocity Components",
) -> None:
    """Render a single-axis plot for XYZ wind velocity components."""
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")
    if not all("wind" in sample for sample in samples):
        raise ValueError("Wind data missing in samples; cannot render plot.")

    base_time = samples[0]["time"]
    times = np.array([s["time"] - base_time for s in samples], dtype=float)
    wind_data = np.array([s["wind"] for s in samples], dtype=float)

    fig, axis = plt.subplots(figsize=(12, 5))
    labels = [("X Axis", "#1f77b4"), ("Y Axis", "#2ca02c"), ("Z Axis", "#d62728")]
    for idx, (label, color) in enumerate(labels):
        axis.plot(times, wind_data[:, idx], label=label, color=color, linewidth=1.5)
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Velocity [m/s]")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axis.legend(loc="upper right")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a force comparison plot from CSV data.")
    parser.add_argument("csv", help="Input CSV file generated by the force recorder.")
    parser.add_argument(
        "--output",
        "-o",
        help="Output PNG path (default: same directory as CSV with .png extension).",
        default=None,
    )
    parser.add_argument(
        "--total-output",
        help="Optional PNG path for total force comparison (default: base name with _totals suffix).",
    )
    parser.add_argument(
        "--error-output",
        help="Optional PNG path for per-axis force error plot (default: base name with _errors suffix).",
    )
    parser.add_argument(
        "--error-total-output",
        help="Optional PNG path for total force error plot (default: base name with _errors_totals suffix).",
    )
    parser.add_argument(
        "--wind-output",
        help="Optional PNG path for wind velocity components (auto-detected if CSV includes wind data).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    args = parser.parse_args()

    samples = load_samples(args.csv)
    if not samples:
        raise SystemExit(f"No samples available in {args.csv}")

    rmse = compute_rmse(samples)

    output_path = args.output
    base, _ = os.path.splitext(args.csv)
    if output_path is None:
        output_path = base + ".png"
    total_output = args.total_output
    if total_output is None:
        total_output = base + "_totals.png"
    error_output = args.error_output
    if error_output is None:
        error_output = base + "_errors.png"
    error_total_output = args.error_total_output
    if error_total_output is None:
        error_total_output = base + "_errors_totals.png"

    render_force_plot(
        samples,
        output_path=output_path,
        total_output_path=total_output,
        show=args.show,
        rmse=rmse,
    )
    print(f"Saved plot to {output_path}")
    print(f"Saved totals plot to {total_output}")

    render_force_error_plot(
        samples,
        output_path=error_output,
        total_output_path=error_total_output,
        show=args.show,
    )
    print(f"Saved error plot to {error_output}")
    print(f"Saved total error plot to {error_total_output}")

    wind_samples = [s for s in samples if "wind" in s]
    have_wind = len(wind_samples) > 0
    wind_output = args.wind_output
    if have_wind:
        if wind_output is None:
            wind_output = base + "_wind.png"
        render_wind_velocity_plot(
            wind_samples,
            output_path=wind_output,
            show=args.show,
        )
        print(f"Saved wind plot to {wind_output}")
    elif wind_output:
        print("Wind output path provided but no wind data is available in the CSV.")
    print(
        "RMSE summary: "
        f"Load total={rmse['fl']['total'][0]:.3f} "
        f"(x={rmse['fl']['components'][0]:.3f}, "
        f"y={rmse['fl']['components'][1]:.3f}, "
        f"z={rmse['fl']['components'][2]:.3f}); "
        f"Quad total={rmse['fq']['total'][0]:.3f} "
        f"(x={rmse['fq']['components'][0]:.3f}, "
        f"y={rmse['fq']['components'][1]:.3f}, "
        f"z={rmse['fq']['components'][2]:.3f})"
    )


if __name__ == "__main__":
    main()
