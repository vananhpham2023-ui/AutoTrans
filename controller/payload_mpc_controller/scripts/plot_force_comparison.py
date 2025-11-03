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

    if not os.environ.get("DISPLAY"):  # pragma: no cover - headless path
        matplotlib.use("Agg")
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
        missing = set(CSV_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV file {csv_path} is missing columns: {sorted(missing)}")
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

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    samples = load_samples(args.csv)
    if not samples:
        raise SystemExit(f"No samples available in {args.csv}")

    rmse = compute_rmse(samples)

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.csv)
        output_path = base + ".png"

    render_force_plot(samples, output_path=output_path, show=args.show, rmse=rmse)
    print(f"Saved plot to {output_path}")
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
