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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    # 统一采用更适合论文排版的轻量级样式：较细线条、较小字体。
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.0,
        }
    )
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
GRID_LINEWIDTH = 0.4
GRID_ALPHA = 0.3
# Match deal force_total2.png line widths so wind plots share the same visual weight.
WIND_LINEWIDTH = 1.3

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


def _downsample_and_smooth(
    times: np.ndarray,
    value_series: Sequence[np.ndarray],
    target_rate_hz: float = 50.0,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Downsample and lightly smooth multiple series sharing the same timestamps."""
    times_np = np.asarray(times, dtype=float)
    if times_np.size < 2 or target_rate_hz <= 0.0:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    finite = np.isfinite(times_np)
    if np.count_nonzero(finite) < 2:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    dt_est = float(np.median(np.diff(times_np[finite])))
    if not math.isfinite(dt_est) or dt_est <= 0.0:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    target_dt = 1.0 / float(target_rate_hz)
    stride = max(1, int(round(target_dt / dt_est)))
    times_ds = times_np[::stride]

    processed: List[np.ndarray] = []
    for values in value_series:
        arr = np.asarray(values, dtype=float)
        arr_ds = arr[::stride]
        if smooth_window > 1 and arr_ds.size > smooth_window:
            k = int(smooth_window)
            kernel = np.ones(k, dtype=float) / float(k)
            padded = np.pad(arr_ds, (k - 1, 0), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            processed.append(smoothed)
        else:
            processed.append(arr_ds)
    return times_ds, processed


def render_force_plot(
    samples: Sequence[Dict[str, Sequence[float]]],
    output_path: Optional[str] = None,
    total_output_path: Optional[str] = None,
    show: bool = False,
    rmse: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    title: str = "External Force Comparison",
    target_rate_hz: float = 50.0,
    smooth_window: int = 5,
    time_xlim: Optional[Tuple[float, float]] = (0.0, 40.0),
    force_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Render a multi-axis plot comparing true and estimated forces."""
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")

    base_time = samples[0]["time"]
    times_full = np.array([s["time"] - base_time for s in samples], dtype=float)

    data_cache: Dict[str, np.ndarray] = {}
    for key in ("fl_true", "fl_est", "fq_true", "fq_est"):
        data_cache[key] = np.array([s[key] for s in samples], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.flatten()

    for axis, (label, true_key, est_key) in zip(axes, PLOT_LAYOUT):
        true_data = data_cache[true_key[0]][:, true_key[1]]
        est_data = data_cache[est_key[0]][:, est_key[1]]

        times_ds, (true_ds, est_ds) = _downsample_and_smooth(
            times_full, [true_data, est_data], target_rate_hz=target_rate_hz, smooth_window=smooth_window
        )

        axis.plot(times_ds, true_ds, label="True", color="#1f77b4", linewidth=1.0)
        axis.plot(times_ds, est_ds, label="Estimated", color="#ff7f0e", linewidth=0.9, linestyle="--")
        axis.set_ylabel("Force [N]")
        axis.set_title(label)
        axis.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

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

    if time_xlim is not None:
        for axis in axes:
            axis.set_xlim(time_xlim[0], time_xlim[1])
    if force_ylim is not None:
        for axis in axes:
            axis.set_ylim(force_ylim[0], force_ylim[1])

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
            times_ds, (true_ds, est_ds) = _downsample_and_smooth(
                times_full, [true_total, est_total], target_rate_hz=target_rate_hz, smooth_window=smooth_window
            )
            axis.plot(times_ds, true_ds, label="True", color="#1f77b4", linewidth=1.0)
            axis.plot(
                times_ds,
                est_ds,
                label="Estimated",
                color="#ff7f0e",
                linewidth=0.9,
                linestyle="--",
            )
            axis.set_title(label)
            axis.set_ylabel("Force [N]")
            axis.set_xlabel("Time [s]")
            axis.legend(loc="upper right")
            axis.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
            if time_xlim is not None:
                axis.set_xlim(time_xlim[0], time_xlim[1])
            if force_ylim is not None:
                axis.set_ylim(force_ylim[0], force_ylim[1])
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
    target_rate_hz: float = 50.0,
    smooth_window: int = 5,
    time_xlim: Optional[Tuple[float, float]] = (0.0, 40.0),
    force_error_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Render force estimation error plots for load and quad components."""
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")

    base_time = samples[0]["time"]
    times_full = np.array([s["time"] - base_time for s in samples], dtype=float)

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

    for axis, (label, (err_key, idx)) in zip(axes, error_layout):
        err_series = error_cache[err_key][:, idx]
        times_ds, (err_ds,) = _downsample_and_smooth(
            times_full, [err_series], target_rate_hz=target_rate_hz, smooth_window=smooth_window
        )
        axis.plot(times_ds, err_ds, color=ERROR_COLOR, linewidth=1.0)
        axis.axhline(0.0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.6)
        axis.set_ylabel("Force Error [N]")
        axis.set_title(label)
        axis.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

    max_time = times_full[-1] if times_full.size else 1.0
    ticks = np.linspace(0.0, max_time, num=5)

    for axis in axes:
        axis.set_xlim(0.0 if time_xlim is None else time_xlim[0], max_time if time_xlim is None else time_xlim[1])
        axis.set_xticks(ticks)
        axis.xaxis.set_minor_locator(AutoMinorLocator(n=2))
        if force_error_ylim is not None:
            axis.set_ylim(force_error_ylim[0], force_error_ylim[1])
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
            times_ds, (err_ds,) = _downsample_and_smooth(
                times_full, [values], target_rate_hz=target_rate_hz, smooth_window=smooth_window
            )
            axis.plot(times_ds, err_ds, color=ERROR_COLOR, linewidth=1.0)
            axis.set_ylabel("Force Error [N]")
            axis.set_xlabel("Time [s]")
            axis.set_title(label)
            axis.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
            axis.set_xlim(0.0 if time_xlim is None else time_xlim[0], max_time if time_xlim is None else time_xlim[1])
            axis.set_xticks(ticks)
            axis.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            if force_error_ylim is not None:
                axis.set_ylim(force_error_ylim[0], force_error_ylim[1])
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
    target_rate_hz: float = 50.0,
    smooth_window: int = 5,
    time_xlim: Optional[Tuple[float, float]] = None,
    velocity_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Render a single-axis plot for XYZ wind velocity components.

    This version performs light downsampling and optional moving-average
    smoothing to produce publication-style figures similar to Dryden
    wind plots commonly used in the literature.
    """
    _ensure_matplotlib()
    if not samples:
        raise ValueError("No samples provided for plotting.")
    if not all("wind" in sample for sample in samples):
        raise ValueError("Wind data missing in samples; cannot render plot.")

    base_time = samples[0]["time"]
    times_full = np.array([s["time"] - base_time for s in samples], dtype=float)
    wind_full = np.array([s["wind"] for s in samples], dtype=float)

    # Detect when the wind field effectively ends (last non-zero sample).
    wind_norm = np.linalg.norm(wind_full, axis=1)
    active_indices = np.nonzero(wind_norm > 1e-3)[0]
    if active_indices.size > 0:
        last_active_time = float(times_full[active_indices[-1]])
    else:
        last_active_time = float(times_full[-1])
    default_end_time = last_active_time + 0.5

    series_list = [wind_full[:, 0], wind_full[:, 1], wind_full[:, 2]]
    times_ds, processed = _downsample_and_smooth(
        times_full,
        series_list,
        target_rate_hz=target_rate_hz,
        smooth_window=smooth_window,
    )
    wind_plot = np.stack(processed, axis=1)

    # Use a rectangular aspect ratio that is visually balanced.
    fig, axis = plt.subplots(figsize=(SUBPLOT_W * 1.2, SUBPLOT_H * 1.2))
    labels = [("X Axis", "#1f77b4"), ("Y Axis", "#2ca02c"), ("Z Axis", "#d62728")]
    for idx, (label, color) in enumerate(labels):
        axis.plot(
            times_ds,
            wind_plot[:, idx],
            label=label,
            color=color,
            linewidth=WIND_LINEWIDTH,
        )

    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Velocity [m/s]")
    axis.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    axis.legend(loc="upper right")

    if time_xlim is not None:
        axis.set_xlim(time_xlim[0], time_xlim[1])
    else:
        # Default horizontal range: from start of data to 0.5 s
        # after the wind field effectively ends.
        x_start = float(times_ds[0])
        x_end = float(default_end_time)
        if x_end <= x_start:
            x_end = float(times_ds[-1])
        axis.set_xlim(x_start, x_end)

    if velocity_ylim is not None:
        axis.set_ylim(velocity_ylim[0], velocity_ylim[1])
    else:
        # Choose a y-range with a small margin below the data and a moderate
        # blank region above (~1/5 of the axis height) so curves remain near
        # the center without excessive empty space.
        y_min = float(np.min(wind_plot))
        y_max = float(np.max(wind_plot))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        data_span = y_max - y_min
        if data_span <= 0.0:
            axis.set_ylim(y_min - 1.0, y_max + 1.0)
        else:
            bottom_ratio = 0.05  # ~5% of axis height below data
            top_ratio = 0.20     # ~20% blank space above data
            visible_ratio = max(1.0 - bottom_ratio - top_ratio, 0.1)
            axis_span = data_span / visible_ratio
            bottom_margin = bottom_ratio * axis_span
            ymin_axis = y_min - bottom_margin
            ymax_axis = ymin_axis + axis_span
            axis.set_ylim(ymin_axis, ymax_axis)

    # No figure-level title for cleaner exported figures.
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
