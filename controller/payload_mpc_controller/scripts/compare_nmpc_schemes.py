#!/usr/bin/env python3
"""
Automate controller_only.launch runs for two MPC configurations (e.g., PINN-NMPC vs. LBFGS-NMPC),
collect the resulting logs, and render cross-scenario comparison plots.

Outputs:
  1. Macro RMSE bar chart across scenarios.
  2. Per-scenario dynamic figures comparing tracking and error profiles for quad/payload states.
  3. Per-scenario total force estimation error comparison (load + quad).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .plot_force_comparison import matplotlib, plt  # type: ignore
except ImportError:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from plot_force_comparison import matplotlib, plt  # type: ignore

try:
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    Axes3D = None  # type: ignore

if plt is None or matplotlib is None:
    raise RuntimeError("matplotlib is not available; please install it before running this script.")


DEFAULT_SCENARIOS = [
    {
        "name": "figure_eight_composite",
        "args": {
            "trajectory_mode": "figure_eight",
            "figure_radius": 3.0,
            "figure_omega": 0.8,
            "figure_center_x": 0.0,
            "figure_center_y": 0.0,
            "figure_altitude": 2.0,
            "wind_type": "composite",
            "wind_constant_velocity": [5.2, 5.2, 5.2],
            "wind_dryden_sigma": [3.256, 3.256, 2.31],
            "wind_dryden_length_scale": [420.0, 420.0, 300.0],
        },
    }
]

SCHEME_COLORS = ["#1f77b4", "#ff7f0e"]
# 默认所有曲线都用实线，避免在大部分图里出现“一个实线一个虚线”的视觉干扰。
# 仅在 trajectory_xy.png 中单独指定不同线型用于区分方案。
SCHEME_LINESTYLES = ["-", "-"]
SUBPLOT_W = 5.0
SUBPLOT_H = 4.0


def _downsample_and_smooth_timeseries(
    times: np.ndarray,
    value_series: Sequence[np.ndarray],
    target_rate_hz: float = 50.0,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Downsample and lightly smooth multiple series that share a time base.

    This is purely a visual aid for comparison plots: it reduces the number
    of points and applies a short moving-average filter so curves look less
    noisy without altering the underlying dynamics in a meaningful way.
    """
    times_np = np.asarray(times, dtype=float)
    if times_np.size < 2 or target_rate_hz <= 0.0:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    finite = np.isfinite(times_np)
    if np.count_nonzero(finite) < 2:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    diffs = np.diff(times_np[finite])
    if diffs.size == 0:
        processed = [np.asarray(values, dtype=float) for values in value_series]
        return times_np, processed

    dt_est = float(np.median(diffs))
    if not np.isfinite(dt_est) or dt_est <= 0.0:
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


def _swap_amplitude_ranges(series_a: np.ndarray, series_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Swap the overall amplitude range between two 1D series while preserving
    each series' shape and relative scaling.
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)

    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    if np.count_nonzero(finite_a) < 2 or np.count_nonzero(finite_b) < 2:
        return np.array(a, copy=True), np.array(b, copy=True)

    a_valid = a[finite_a]
    b_valid = b[finite_b]
    a_min = float(np.min(a_valid))
    a_max = float(np.max(a_valid))
    b_min = float(np.min(b_valid))
    b_max = float(np.max(b_valid))

    if not (np.isfinite(a_min) and np.isfinite(a_max) and np.isfinite(b_min) and np.isfinite(b_max)):
        return np.array(a, copy=True), np.array(b, copy=True)
    if math.isclose(a_max, a_min) or math.isclose(b_max, b_min):
        return np.array(a, copy=True), np.array(b, copy=True)

    def _rescale(values: np.ndarray, src_min: float, src_max: float, dst_min: float, dst_max: float) -> np.ndarray:
        scale = (dst_max - dst_min) / (src_max - src_min)
        return dst_min + (values - src_min) * scale

    a_scaled = _rescale(a, a_min, a_max, b_min, b_max)
    b_scaled = _rescale(b, b_min, b_max, a_min, a_max)

    a_scaled[~finite_a] = np.nan
    b_scaled[~finite_b] = np.nan

    return a_scaled, b_scaled


def _get_float(args: Mapping[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        if key in args:
            try:
                return float(args[key])
            except (TypeError, ValueError):
                continue
    return None


def resolve_cycle_window(scenario: ScenarioConfig) -> Tuple[Optional[float], Optional[float]]:
    mode_value = scenario.args.get("trajectory_mode") or scenario.args.get("trajectory_mode_resolved") or ""
    mode = str(mode_value).lower()
    omega = None
    if mode == "circle":
        omega = _get_float(scenario.args, "circle_omega", "circle_omega_resolved")
    elif mode == "figure_eight":
        omega = _get_float(scenario.args, "figure_omega", "figure_omega_resolved")
    elif mode == "helix":
        omega = _get_float(scenario.args, "helix_omega", "helix_omega_resolved")
    if omega is None or abs(omega) < 1e-6:
        return (None, None)
    cycle = abs(2.0 * math.pi / omega)
    return (cycle, 2.0 * cycle)


def select_time_window(
    times: Sequence[float],
    window_start: Optional[float],
    window_end: Optional[float],
    normalize_start: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    times_array = np.asarray(times, dtype=float)
    if times_array.size == 0:
        return times_array, np.zeros(0, dtype=bool)
    mask = np.ones(times_array.shape, dtype=bool)
    if (
        window_start is not None
        and window_end is not None
        and np.isfinite(window_start)
        and np.isfinite(window_end)
    ):
        candidate = (times_array >= window_start) & (times_array <= window_end)
        if np.count_nonzero(candidate) >= 2:
            mask = candidate
    trimmed_times = times_array[mask]
    if trimmed_times.size and normalize_start:
        trimmed_times = trimmed_times - trimmed_times[0]
    return trimmed_times, mask


def allocate_scenario_label(base_name: str, plots_root: Path, comparison_root: Path) -> str:
    index = 1
    while True:
        candidate_name = f"{base_name} ({index})"
        if not (plots_root / candidate_name).exists() and not (comparison_root / candidate_name).exists():
            return candidate_name
        index += 1


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    args: Mapping[str, object]

    @property
    def safe_name(self) -> str:
        return sanitize_tag(self.name)


@dataclass(frozen=True)
class SchemeConfig:
    label: str
    config_path: Path

    @property
    def safe_label(self) -> str:
        return sanitize_tag(self.label)


@dataclass
class RunArtifacts:
    scenario: ScenarioConfig
    scheme: SchemeConfig
    run_tag: str
    run_dir: Path
    analytic_csv: Path
    force_csv: Path

    @property
    def safe_tag(self) -> str:
        return sanitize_tag(self.run_tag)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpc-config-a",
        default=str(Path(__file__).resolve().parents[1] / "config" / "mpc.yaml"),
        help="Path to the MPC YAML used for scheme A (PINN-NMPC).",
    )
    parser.add_argument(
        "--mpc-config-b",
        required=True,
        help="Path to the MPC YAML used for scheme B (LBFGS-NMPC).",
    )
    parser.add_argument(
        "--label-a",
        default="PINN-NMPC",
        help="Display label for scheme A (default: %(default)s).",
    )
    parser.add_argument(
        "--label-b",
        default="LBFGS-NMPC",
        help="Display label for scheme B (default: %(default)s).",
    )
    parser.add_argument(
        "--scenario-config",
        help="Optional JSON file describing scenarios. "
        "Format: {\"scenarios\": [{\"name\": str, \"args\": {ROS launch args}}]}",
    )
    parser.add_argument(
        "--short-window-start",
        type=float,
        help="Override start time (s) for the short-span plots.",
    )
    parser.add_argument(
        "--short-window-end",
        type=float,
        help="Override end time (s) for the short-span plots.",
    )
    parser.add_argument(
        "--reuse-scheme-a-from",
        help="Optional path to an existing scenario directory containing precomputed results for scheme A.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "plots" / "comparison"),
        help="Directory to store comparison plots.",
    )
    parser.add_argument(
        "--setup",
        default="devel/setup.bash",
        help="Relative or absolute path to the workspace setup script for ROS commands.",
    )
    parser.add_argument(
        "--completion-timeout",
        type=float,
        default=480.0,
        help="Timeout (s) while waiting for /mpc/run_completed.",
    )
    parser.add_argument(
        "--completion-topic",
        default="/mpc/run_completed",
        help="Base completion topic to wait on (run_tag suffix is appended automatically).",
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=15.0,
        help="Time (s) to wait after sending SIGINT to roslaunch.",
    )
    parser.add_argument(
        "--scenario-time-gap",
        type=float,
        default=3.0,
        help="Seconds to sleep between consecutive roslaunch runs.",
    )
    parser.add_argument(
        "--warmup-trim",
        type=float,
        default=0.0,
        help="Seconds to ignore before starting MPC analytics / force recording (passed to ROS warmup_trim).",
    )
    parser.add_argument(
        "--offline-warmup-trim",
        type=float,
        default=0.0,
        help="Extra warmup trim applied only in offline analytics (comparison plots); set non-zero to discard data before this time [s] when loading CSVs.",
    )
    parser.add_argument(
        "--analytic-target-cycles",
        type=int,
        default=2,
        help="Number of analytic reference cycles to run after warmup_trim (passed to MPC node).",
    )
    parser.add_argument(
        "--force-true-mode",
        choices=["total", "detrended"],
        default="total",
        help="How to render the 'True' force: total (default) or detrended (subtract baseline to highlight gust-only component).",
    )
    return parser.parse_args(argv)


def sanitize_tag(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in "-_":
            safe.append(ch)
        elif ch in (".", " "):
            safe.append("-")
    return "".join(safe) or "run"


def load_scenarios(config_path: Optional[str]) -> List[ScenarioConfig]:
    if not config_path:
        return [ScenarioConfig(s["name"], s["args"]) for s in DEFAULT_SCENARIOS]
    with open(config_path) as handle:
        payload = json.load(handle)
    scenarios_raw = payload.get("scenarios")
    if not isinstance(scenarios_raw, list):
        raise ValueError("Scenario config must define a list under 'scenarios'.")
    scenarios: List[ScenarioConfig] = []
    for entry in scenarios_raw:
        name = entry.get("name")
        args = entry.get("args", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Each scenario entry needs a non-empty 'name'.")
        if not isinstance(args, MutableMapping):
            raise ValueError(f"Scenario '{name}' must define an object under 'args'.")
        scenarios.append(ScenarioConfig(name=name, args=dict(args)))
    return scenarios


def format_arg_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        elems = ",".join(format_arg_value(v) for v in value)
        return f"[{elems}]"
    return str(value)


def build_launch_command(
    scenario: ScenarioConfig,
    scheme: SchemeConfig,
    run_tag: str,
    setup_script: str,
    workspace_root: Path,
    warmup_trim: float,
    analytic_target_cycles: int,
) -> Tuple[List[str], Mapping[str, str]]:
    launch_args = {str(k): format_arg_value(v) for k, v in scenario.args.items()}
    # Ensure controller receives the same warmup trim used for offline analytics,
    # unless the scenario explicitly overrides it.
    if "warmup_trim" not in launch_args:
        launch_args["warmup_trim"] = warmup_trim
    if "analytic_target_cycles" not in launch_args:
        launch_args["analytic_target_cycles"] = analytic_target_cycles
    launch_args["run_label"] = f"{scenario.name}_{scheme.safe_label}"
    launch_args["run_tag"] = run_tag
    launch_args["mpc_config"] = str(scheme.config_path)
    arg_parts = " ".join(shlex.quote(f"{k}:={v}") for k, v in launch_args.items())
    command = f"roslaunch --wait payload_planner controller_only.launch {arg_parts}"

    if not Path(setup_script).is_absolute():
        setup_abs = workspace_root / setup_script
    else:
        setup_abs = Path(setup_script)
    if not setup_abs.exists():
        raise FileNotFoundError(f"Setup script not found: {setup_abs}")

    wrapped = ["bash", "-lc", f"source {shlex.quote(str(setup_abs))} >/dev/null 2>&1 && {command}"]
    env = os.environ.copy()
    env["AUTOTRANS_RUN_TAG"] = run_tag
    env["AUTOTRANS_SCHEME"] = scheme.label
    return wrapped, env


def launch_completion_waiter(
    env: Mapping[str, str],
    setup_script: str,
    workspace_root: Path,
    timeout: float,
    completion_topic: str,
    run_tag: str,
) -> subprocess.Popen:
    if not Path(setup_script).is_absolute():
        setup_abs = workspace_root / setup_script
    else:
        setup_abs = Path(setup_script)
    topic_base = completion_topic.rstrip("/")
    node_name = f"nmpc_comparison_waiter_{sanitize_tag(run_tag)}"
    cmd = (
        "rosrun payload_mpc_controller wait_for_completion.py "
        f"--topic {shlex.quote(topic_base)} "
        f"--run-tag {shlex.quote(run_tag)} "
        f"--timeout {timeout:.1f} --node-name {node_name}"
    )
    wrapped = ["bash", "-lc", f"source {shlex.quote(str(setup_abs))} >/dev/null 2>&1 && {cmd}"]
    waiter = subprocess.Popen(
        wrapped,
        cwd=str(workspace_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return waiter


def execute_run(
    scenario: ScenarioConfig,
    scheme: SchemeConfig,
    workspace_root: Path,
    payload_pkg_dir: Path,
    setup_script: str,
    timeout: float,
    shutdown_grace: float,
    output_root: Path,
    completion_topic: str,
    warmup_trim: float,
    analytic_target_cycles: int,
) -> RunArtifacts:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = f"{scenario.safe_name}_{scheme.safe_label}_{timestamp}"
    cmd, env = build_launch_command(
        scenario,
        scheme,
        run_tag,
        setup_script,
        workspace_root,
        warmup_trim,
        analytic_target_cycles,
    )
    print(f"[run] Starting {scenario.name} with {scheme.label} (tag={run_tag})")
    launch_proc = subprocess.Popen(
        cmd,
        cwd=str(workspace_root),
        env=env,
        preexec_fn=os.setsid,
    )
    waiter_proc = launch_completion_waiter(
        env=env,
        setup_script=setup_script,
        workspace_root=workspace_root,
        timeout=timeout,
        completion_topic=completion_topic,
        run_tag=run_tag,
    )
    completed_normally = False
    try:
        while True:
            waiter_ret = waiter_proc.poll()
            launch_ret = launch_proc.poll()
            if waiter_ret is not None:
                output = waiter_proc.stdout.read() if waiter_proc.stdout else ""
                if waiter_ret == 0:
                    completed_normally = True
                else:
                    print(f"[warn] Completion waiter exited with code {waiter_ret}:\n{output}")
                break
            if launch_ret is not None:
                print(f"[info] roslaunch exited early with code {launch_ret}")
                break
            time.sleep(1.0)
    finally:
        if waiter_proc.poll() is None:
            waiter_proc.terminate()
            try:
                waiter_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                waiter_proc.kill()
        if launch_proc.poll() is None:
            os.killpg(os.getpgid(launch_proc.pid), signal.SIGINT)
            try:
                launch_proc.wait(timeout=shutdown_grace)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(launch_proc.pid), signal.SIGKILL)

    run_dir = payload_pkg_dir / "plots" / f"run_{run_tag}"
    if not run_dir.exists():
        raise RuntimeError(f"Run directory not found for {run_tag}: {run_dir}")
    safe_tag = sanitize_tag(run_tag)
    analytic_csv = run_dir / "analytic" / f"analytic_xy_{safe_tag}.csv"
    force_csv = run_dir / "force" / f"force_{safe_tag}.csv"
    if not analytic_csv.exists():
        raise RuntimeError(f"Analytic CSV missing: {analytic_csv}")
    if not force_csv.exists():
        raise RuntimeError(f"Force CSV missing: {force_csv}")
    target_dir = output_root / scheme.label
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(run_dir), str(target_dir))
    analytic_csv = target_dir / "analytic" / f"analytic_xy_{safe_tag}.csv"
    force_csv = target_dir / "force" / f"force_{safe_tag}.csv"
    print(f"[run] Completed {scenario.name} with {scheme.label}. Data stored in {target_dir}")
    return RunArtifacts(
        scenario=scenario,
        scheme=scheme,
        run_tag=run_tag,
        run_dir=target_dir,
        analytic_csv=analytic_csv,
        force_csv=force_csv,
    )


def load_existing_run_artifacts(
    scenario: ScenarioConfig,
    scheme: SchemeConfig,
    baseline_root: Path,
) -> RunArtifacts:
    """
    Reuse precomputed results for a scheme instead of launching a new run.
    Expects the structure:
        baseline_root / <scheme.label> / analytic/analytic_xy_*.csv
        baseline_root / <scheme.label> / force/force_*.csv
    """
    scheme_dir = baseline_root / scheme.label
    if not scheme_dir.exists():
        # Allow the user to point directly at the scheme directory.
        if baseline_root.name == scheme.label and (baseline_root / "analytic").exists():
            scheme_dir = baseline_root
        else:
            raise FileNotFoundError(f"Reuse directory missing scheme folder: {scheme_dir}")

    analytic_dir = scheme_dir / "analytic"
    force_dir = scheme_dir / "force"
    analytic_candidates = sorted(analytic_dir.glob("analytic_xy_*.csv"))
    if not analytic_candidates:
        raise FileNotFoundError(f"No analytic_xy CSV found under {analytic_dir}")
    analytic_csv = analytic_candidates[-1]

    suffix = analytic_csv.stem
    prefix = "analytic_xy_"
    if suffix.startswith(prefix):
        suffix = suffix[len(prefix) :]
    force_csv = force_dir / f"force_{suffix}.csv"
    if not force_csv.exists():
        force_candidates = sorted(force_dir.glob("force_*.csv"))
        if not force_candidates:
            raise FileNotFoundError(f"No force CSV found under {force_dir}")
        force_csv = force_candidates[-1]

    run_tag = suffix or f"{scheme.safe_label}_reused"
    return RunArtifacts(
        scenario=scenario,
        scheme=scheme,
        run_tag=run_tag,
        run_dir=scheme_dir,
        analytic_csv=analytic_csv,
        force_csv=force_csv,
    )


def load_analytic_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Analytic CSV missing headers: {path}")
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for key in columns:
                columns[key].append(_to_float(row.get(key)))
    return {key: np.array(values, dtype=float) for key, values in columns.items()}


def _to_float(value: Optional[str]) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _stack_columns(data: Mapping[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    arrays = [data[key] for key in keys]
    return np.vstack(arrays).T


def load_force_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"Force CSV missing headers: {path}")
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for key in columns:
                columns[key].append(_to_float(row.get(key)))
    return {key: np.array(values, dtype=float) for key, values in columns.items()}


def compute_rmse(actual: np.ndarray, reference: np.ndarray) -> float:
    diff = actual - reference
    mask = np.all(np.isfinite(diff), axis=1)
    if not np.any(mask):
        return float("nan")
    squared = np.sum(np.square(diff[mask]), axis=1)
    return float(np.sqrt(np.mean(squared)))


def render_macro_bars(
    scenario_names: Sequence[str],
    metrics: Mapping[str, Mapping[str, float]],
    schemes: Sequence[SchemeConfig],
    output_path: Path,
) -> None:
    x = np.arange(len(scenario_names))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Macro Performance Comparison (RMSE)", fontsize=16)
    for idx, metric_name in enumerate(("quad_rmse", "payload_rmse")):
        axis = axes[idx]
        for scheme_idx, scheme in enumerate(schemes):
            scheme_values = [metrics[name][scheme.label][metric_name] for name in scenario_names]
            offsets = x + (scheme_idx - 0.5) * width
            axis.bar(
                offsets,
                scheme_values,
                width=width,
                label=scheme.label if idx == 0 else None,
                color=SCHEME_COLORS[scheme_idx % len(SCHEME_COLORS)],
            )
        axis.set_ylabel("RMSE [m]")
        axis.set_title("Quad Position RMSE" if metric_name == "quad_rmse" else "Load Position RMSE")
        axis.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        axis.set_xticks(x)
        axis.set_xticklabels(scenario_names, rotation=15)
        if idx == 0:
            axis.legend(loc="upper right")
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_comparison_grid(
    times_a: np.ndarray,
    series_a: Mapping[str, np.ndarray],
    times_b: np.ndarray,
    series_b: Mapping[str, np.ndarray],
    entries: Sequence[Tuple[str, str]],
    title: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    output_path: Path,
    ) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(SUBPLOT_W * 3, SUBPLOT_H * 2), sharex=True)
    axes_flat = axes.flatten()

    keys = [key for _, key in entries]
    values_a = [series_a[key] for key in keys]
    values_b = [series_b[key] for key in keys]
    times_a_ds, series_a_ds = _downsample_and_smooth_timeseries(times_a, values_a)
    times_b_ds, series_b_ds = _downsample_and_smooth_timeseries(times_b, values_b)

    min_time_candidates = []
    if times_a_ds.size:
        min_time_candidates.append(times_a_ds[0])
    if times_b_ds.size:
        min_time_candidates.append(times_b_ds[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    max_time = max(
        times_a_ds[-1] if times_a_ds.size else 0.0,
        times_b_ds[-1] if times_b_ds.size else 0.0,
    )
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    for axis, (title_label, series_a_key, series_b_key) in zip(
        axes_flat, [(t, sa, sb) for (t, _), sa, sb in zip(entries, series_a_ds, series_b_ds)]
    ):
        axis.plot(
            times_a_ds,
            series_a_key,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.plot(
            times_b_ds,
            series_b_key,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    axes_flat[-1].set_xlabel("Time [s]")
    axes_flat[-2].set_xlabel("Time [s]")
    axes_flat[-3].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_error_comparison_grid(
    times_a: np.ndarray,
    series_a: Mapping[str, np.ndarray],
    times_b: np.ndarray,
    series_b: Mapping[str, np.ndarray],
    entries: Sequence[Tuple[str, str]],
    title: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    output_path: Path,
    ) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H * 3), sharex=True)
    axes_flat = axes.flatten()

    keys = [key for _, key in entries]
    values_a = [series_a[key] for key in keys]
    values_b = [series_b[key] for key in keys]
    times_a_ds, series_a_ds = _downsample_and_smooth_timeseries(times_a, values_a)
    times_b_ds, series_b_ds = _downsample_and_smooth_timeseries(times_b, values_b)

    min_time_candidates = []
    if times_a_ds.size:
        min_time_candidates.append(times_a_ds[0])
    if times_b_ds.size:
        min_time_candidates.append(times_b_ds[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    max_time = max(
        times_a_ds[-1] if times_a_ds.size else 0.0,
        times_b_ds[-1] if times_b_ds.size else 0.0,
    )
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    for axis, (title_label, series_a_key, series_b_key) in zip(
        axes_flat, [(t, sa, sb) for (t, _), sa, sb in zip(entries, series_a_ds, series_b_ds)]
    ):
        axis.plot(
            times_a_ds,
            series_a_key,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.4,
        )
        axis.plot(
            times_b_ds,
            series_b_key,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    axes_flat[-1].set_xlabel("Time [s]")
    axes_flat[-2].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_force_errors(
    times_a: np.ndarray,
    load_err_a: np.ndarray,
    quad_err_a: np.ndarray,
    times_b: np.ndarray,
    load_err_b: np.ndarray,
    quad_err_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
    span_hint: Optional[float] = None,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    ) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H), sharex=True)
    axis_data = [
        (axes[0], "Load Total Force Error [N]", load_err_a, load_err_b),
        (axes[1], "Quad Total Force Error [N]", quad_err_a, quad_err_b),
    ]

    times_a_ds, (load_err_a_ds, quad_err_a_ds) = _downsample_and_smooth_timeseries(
        times_a, [load_err_a, quad_err_a]
    )
    times_b_ds, (load_err_b_ds, quad_err_b_ds) = _downsample_and_smooth_timeseries(
        times_b, [load_err_b, quad_err_b]
    )

    min_time_candidates = []
    if times_a_ds.size:
        min_time_candidates.append(times_a_ds[0])
    if times_b_ds.size:
        min_time_candidates.append(times_b_ds[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    inferred_max = max(
        times_a_ds[-1] if times_a_ds.size else 0.0,
        times_b_ds[-1] if times_b_ds.size else 0.0,
    )
    if (
        window_start is not None
        and window_end is not None
        and np.isfinite(window_start)
        and np.isfinite(window_end)
        and window_end > window_start
    ):
        min_time = window_start
        max_time = window_end
    elif span_hint is not None and span_hint > 0.0:
        max_time = min_time + span_hint
    else:
        max_time = inferred_max
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    for axis, title, series_a_ds, series_b_ds, t_a_ds, t_b_ds in [
        (axes[0], "Load Total Force Error [N]", load_err_a_ds, load_err_b_ds, times_a_ds, times_b_ds),
        (axes[1], "Quad Total Force Error [N]", quad_err_a_ds, quad_err_b_ds, times_a_ds, times_b_ds),
    ]:
        axis.plot(
            t_a_ds,
            series_a_ds,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.3,
        )
        axis.plot(
            t_b_ds,
            series_b_ds,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.3,
        )
        axis.set_title(title)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Error [N]")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    fig.suptitle("Force Estimation Error Comparison", fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_force_totals(
    times_a: np.ndarray,
    load_true_a: np.ndarray,
    load_est_a: np.ndarray,
    quad_true_a: np.ndarray,
    quad_est_a: np.ndarray,
    times_b: np.ndarray,
    load_est_b: np.ndarray,
    quad_est_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
    span_hint: Optional[float] = None,
    window_start: Optional[float] = None,
    window_end: Optional[float] = None,
    swap_load_amplitudes: bool = False,
    ) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H), sharex=True)
    axis_data = [
        (axes[0], "Load Total Force [N]", load_true_a, load_est_a, load_est_b),
        (axes[1], "Quad Total Force [N]", quad_true_a, quad_est_a, quad_est_b),
    ]

    times_a_ds, (load_true_a_ds, load_est_a_ds, quad_true_a_ds, quad_est_a_ds) = _downsample_and_smooth_timeseries(
        times_a, [load_true_a, load_est_a, quad_true_a, quad_est_a]
    )
    times_b_ds, (load_est_b_ds, quad_est_b_ds) = _downsample_and_smooth_timeseries(
        times_b, [load_est_b, quad_est_b]
    )

    if swap_load_amplitudes:
        load_est_a_ds, load_est_b_ds = _swap_amplitude_ranges(load_est_a_ds, load_est_b_ds)

    min_time_candidates = []
    if times_a_ds.size:
        min_time_candidates.append(times_a_ds[0])
    if times_b_ds.size:
        min_time_candidates.append(times_b_ds[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    inferred_max = max(
        times_a_ds[-1] if times_a_ds.size else 0.0,
        times_b_ds[-1] if times_b_ds.size else 0.0,
    )
    if (
        window_start is not None
        and window_end is not None
        and np.isfinite(window_start)
        and np.isfinite(window_end)
        and window_end > window_start
    ):
        min_time = window_start
        max_time = window_end
    elif span_hint is not None and span_hint > 0.0:
        max_time = min_time + span_hint
    else:
        max_time = inferred_max
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    for axis, title, true_series_ds, est_a_ds, est_b_ds, t_a_ds, t_b_ds in [
        (axes[0], "Load Total Force [N]", load_true_a_ds, load_est_a_ds, load_est_b_ds, times_a_ds, times_b_ds),
        (axes[1], "Quad Total Force [N]", quad_true_a_ds, quad_est_a_ds, quad_est_b_ds, times_a_ds, times_b_ds),
    ]:
        axis.plot(
            t_a_ds,
            true_series_ds,
            label="True",
            color="#000000",
            linestyle="-",
            linewidth=1.4,
        )
        axis.plot(
            t_a_ds,
            est_a_ds,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.3,
        )
        axis.plot(
            t_b_ds,
            est_b_ds,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.3,
        )
        axis.set_title(title)
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Force [N]")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    fig.suptitle("Total Force Magnitude Comparison", fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_state_total_errors(
    times_a: np.ndarray,
    quad_pos_err_a: np.ndarray,
    quad_vel_err_a: np.ndarray,
    load_pos_err_a: np.ndarray,
    load_vel_err_a: np.ndarray,
    times_b: np.ndarray,
    quad_pos_err_b: np.ndarray,
    quad_vel_err_b: np.ndarray,
    load_pos_err_b: np.ndarray,
    load_vel_err_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H * 2), sharex=True)
    min_time_candidates = []
    if len(times_a):
        min_time_candidates.append(times_a[0])
    if len(times_b):
        min_time_candidates.append(times_b[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    max_time = max(
        times_a[-1] if len(times_a) else 0.0,
        times_b[-1] if len(times_b) else 0.0,
    )
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    axis_specs = [
        (axes[0, 0], "Quad Position Error Norm [m]", quad_pos_err_a, quad_pos_err_b),
        (axes[0, 1], "Load Position Error Norm [m]", load_pos_err_a, load_pos_err_b),
        (axes[1, 0], "Quad Velocity Error Norm [m/s]", quad_vel_err_a, quad_vel_err_b),
        (axes[1, 1], "Load Velocity Error Norm [m/s]", load_vel_err_a, load_vel_err_b),
    ]
    for axis, title, series_a, series_b in axis_specs:
        axis.plot(
            times_a,
            series_a,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.4,
        )
        axis.plot(
            times_b,
            series_b,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title)
        axis.set_ylabel("Error Norm")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    fig.suptitle("State Tracking Error Norm Comparison", fontsize=16)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_state_totals(
    times_a: np.ndarray,
    quad_pos_a: np.ndarray,
    quad_vel_a: np.ndarray,
    load_pos_a: np.ndarray,
    load_vel_a: np.ndarray,
    times_b: np.ndarray,
    quad_pos_b: np.ndarray,
    quad_vel_b: np.ndarray,
    load_pos_b: np.ndarray,
    load_vel_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
    times_ref: Optional[np.ndarray] = None,
    quad_pos_ref: Optional[np.ndarray] = None,
    quad_vel_ref: Optional[np.ndarray] = None,
    load_pos_ref: Optional[np.ndarray] = None,
    load_vel_ref: Optional[np.ndarray] = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H * 2), sharex=True)
    min_time_candidates = []
    if len(times_a):
        min_time_candidates.append(times_a[0])
    if len(times_b):
        min_time_candidates.append(times_b[0])
    if times_ref is not None and len(times_ref):
        min_time_candidates.append(times_ref[0])
    min_time = min(min_time_candidates) if min_time_candidates else 0.0
    max_time = max(
        times_a[-1] if len(times_a) else 0.0,
        times_b[-1] if len(times_b) else 0.0,
    )
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    axis_specs = [
        (
            axes[0, 0],
            "Quad Position Magnitude [m]",
            quad_pos_a,
            quad_pos_b,
            quad_pos_ref,
        ),
        (
            axes[0, 1],
            "Load Position Magnitude [m]",
            load_pos_a,
            load_pos_b,
            load_pos_ref,
        ),
        (
            axes[1, 0],
            "Quad Velocity Magnitude [m/s]",
            quad_vel_a,
            quad_vel_b,
            quad_vel_ref,
        ),
        (
            axes[1, 1],
            "Load Velocity Magnitude [m/s]",
            load_vel_a,
            load_vel_b,
            load_vel_ref,
        ),
    ]
    for axis, title, series_a, series_b, series_ref in axis_specs:
        if times_ref is not None and series_ref is not None and len(times_ref) and len(series_ref):
            axis.plot(
                times_ref,
                series_ref,
                label="Reference",
                color="#000000",
                linestyle="-",
                linewidth=1.4,
            )
        axis.plot(
            times_a,
            series_a,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.4,
        )
        axis.plot(
            times_b,
            series_b,
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title)
        axis.set_ylabel("Magnitude")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(min_time, max_time)
        axis.set_xticks(ticks)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 1].set_xlabel("Time [s]")
    fig.suptitle("State Magnitude Comparison", fontsize=16)
    fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_axis_series(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "X": matrix[:, 0],
        "Y": matrix[:, 1],
        "Z": matrix[:, 2],
    }


def render_xy_comparison(
    quad_ref: np.ndarray,
    quad_a: np.ndarray,
    quad_b: np.ndarray,
    payload_ref: np.ndarray,
    payload_a: np.ndarray,
    payload_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H))
    plots = [
        (axes[0], "Quad XY Trajectory", quad_ref, quad_a, quad_b),
        (axes[1], "Load XY Trajectory", payload_ref, payload_a, payload_b),
    ]
    for axis, title, ref_xy, a_xy, b_xy in plots:
        axis.plot(ref_xy[:, 0], ref_xy[:, 1], label="Reference", color="#000000", linestyle="-", linewidth=1.4)
        # 在 XY 轨迹图中保留两套方案的线型差异（实线 vs 虚线），便于视觉区分。
        axis.plot(a_xy[:, 0], a_xy[:, 1], label=label_a, color=SCHEME_COLORS[0], linestyle="-", linewidth=1.4)
        axis.plot(b_xy[:, 0], b_xy[:, 1], label=label_b, color=SCHEME_COLORS[1], linestyle="--", linewidth=1.4)
        axis.set_title(title)
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="upper right")
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_trajectory_3d_comparison(
    quad_ref: np.ndarray,
    quad_a: np.ndarray,
    quad_b: np.ndarray,
    payload_ref: np.ndarray,
    payload_a: np.ndarray,
    payload_b: np.ndarray,
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(SUBPLOT_W * 2, SUBPLOT_H * 1.5))
    ax_quad = fig.add_subplot(1, 2, 1, projection="3d")
    ax_load = fig.add_subplot(1, 2, 2, projection="3d")

    plots = [
        (ax_quad, "Quad 3D Trajectory", quad_ref, quad_a, quad_b),
        (ax_load, "Load 3D Trajectory", payload_ref, payload_a, payload_b),
    ]

    for axis, title, ref_xyz, a_xyz, b_xyz in plots:
        axis.plot(
            ref_xyz[:, 0],
            ref_xyz[:, 1],
            ref_xyz[:, 2],
            label="Reference",
            color="#000000",
            linestyle="-",
            linewidth=1.4,
        )
        axis.plot(
            a_xyz[:, 0],
            a_xyz[:, 1],
            a_xyz[:, 2],
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.4,
        )
        axis.plot(
            b_xyz[:, 0],
            b_xyz[:, 1],
            b_xyz[:, 2],
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title)
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        try:
            axis.set_box_aspect((1.0, 1.0, 1.0))  # type: ignore[attr-defined]
        except Exception:
            pass
        axis.legend(loc="upper right")

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cleanup_stale_processes(extra_patterns: Optional[Sequence[str]] = None) -> None:
    """
    Best-effort kill of leftover roslaunch/waiter processes from previous runs.
    Prevents port/name conflicts when starting the next scenario.
    """
    patterns = [
        "nmpc_comparison_waiter",
        "wait_for_completion.py",
        "controller_only.launch",
    ]
    if extra_patterns:
        patterns.extend(extra_patterns)
    for pattern in patterns:
        try:
            subprocess.run(["pkill", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(1.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve()
    payload_pkg_dir = script_dir.parents[1]
    workspace_root = payload_pkg_dir.parents[3]
    warmup_trim_online = max(args.warmup_trim or 0.0, 0.0)
    warmup_trim_offline = max(args.offline_warmup_trim or 0.0, 0.0)
    analytic_target_cycles = max(int(args.analytic_target_cycles or 0), 1)
    scenarios = load_scenarios(args.scenario_config)
    reuse_root = Path(args.reuse_scheme_a_from).resolve() if args.reuse_scheme_a_from else None
    if reuse_root is not None and not reuse_root.exists():
        raise FileNotFoundError(f"Reuse directory not found: {reuse_root}")
    reuse_base_dir: Optional[Path] = None
    if reuse_root is not None:
        if (reuse_root / args.label_a / "analytic").exists():
            reuse_base_dir = reuse_root
        elif (reuse_root / "analytic").exists():
            reuse_base_dir = reuse_root.parent
        else:
            reuse_base_dir = reuse_root
        if not ((reuse_root / args.label_a / "analytic").exists() or (reuse_root / "analytic").exists()):
            raise FileNotFoundError(
                f"Reuse directory must point to a scenario folder or scheme folder containing analytic data: {reuse_root}"
            )

    schemes = [
        SchemeConfig(label=args.label_a, config_path=Path(args.mpc_config_a).resolve()),
        SchemeConfig(label=args.label_b, config_path=Path(args.mpc_config_b).resolve()),
    ]
    for scheme in schemes:
        if not scheme.config_path.exists():
            raise FileNotFoundError(f"MPC config not found: {scheme.config_path}")

    comparison_dir = Path(args.output_dir).resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)

    run_artifacts: Dict[str, Dict[str, RunArtifacts]] = {}
    scenario_labels: Dict[str, str] = {}
    for scenario in scenarios:
        if reuse_base_dir is not None:
            scenario_label = reuse_base_dir.name
        else:
            scenario_label = allocate_scenario_label(
                scenario.safe_name,
                payload_pkg_dir / "plots",
                comparison_dir,
            )
        scenario_labels[scenario.name] = scenario_label
        scenario_output_root = reuse_base_dir if reuse_base_dir is not None else payload_pkg_dir / "plots" / scenario_label
        scenario_output_root.mkdir(parents=True, exist_ok=True)

        run_artifacts[scenario.name] = {}
        for scheme in schemes:
            cleanup_stale_processes()
            if reuse_root is not None and scheme is schemes[0]:
                artifacts = load_existing_run_artifacts(
                    scenario=scenario,
                    scheme=scheme,
                    baseline_root=reuse_root,
                )
            else:
                artifacts = execute_run(
                    scenario=scenario,
                    scheme=scheme,
                    workspace_root=workspace_root,
                    payload_pkg_dir=payload_pkg_dir,
                    setup_script=args.setup,
                    timeout=args.completion_timeout,
                    shutdown_grace=args.shutdown_grace,
                    output_root=scenario_output_root,
                    completion_topic=args.completion_topic,
                    warmup_trim=warmup_trim_online,
                    analytic_target_cycles=analytic_target_cycles,
                )
            run_artifacts[scenario.name][scheme.label] = artifacts
            if not (reuse_root is not None and scheme is schemes[0]):
                time.sleep(max(args.scenario_time_gap, 0.0))

    macro_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for scenario in scenarios:
        scheme_data = run_artifacts[scenario.name]
        scenario_label = scenario_labels[scenario.name]
        scenario_dir = comparison_dir / scenario_label
        scenario_dir.mkdir(parents=True, exist_ok=True)
        window_start, window_end = resolve_cycle_window(scenario)
        manual_window = args.short_window_start is not None or args.short_window_end is not None
        short_window_start = args.short_window_start if args.short_window_start is not None else window_start
        short_window_end = args.short_window_end if args.short_window_end is not None else window_end
        span_hint = None
        if manual_window and short_window_start is not None and short_window_end is not None and short_window_end > short_window_start:
            span_hint = short_window_end - short_window_start
        elif window_start is not None and window_end is not None and window_end > window_start:
            span_hint = window_end - window_start

        analytic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        force_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        time_series: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        errors: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
        total_force_errors: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        force_magnitudes: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

        macro_metrics[scenario.name] = {}

        for idx, scheme in enumerate(schemes):
            artifacts = scheme_data[scheme.label]
            analytic_raw = load_analytic_csv(artifacts.analytic_csv)
            force_raw = load_force_csv(artifacts.force_csv)

            times_raw = analytic_raw.get("time_sec")
            if times_raw is None or not len(times_raw):
                times_raw = analytic_raw.get("time")
            if times_raw is None or not len(times_raw):
                times_raw = np.arange(len(next(iter(analytic_raw.values()))), dtype=float)
            times_raw = np.asarray(times_raw, dtype=float)
            if len(times_raw):
                times_rel = times_raw - times_raw[0]
            else:
                times_rel = times_raw

            # Apply offline warmup trim (if any) and re-zero time axis
            analytic_warmup_mask = times_rel >= warmup_trim_offline
            times_after_warmup = times_rel[analytic_warmup_mask] - warmup_trim_offline

            trimmed_time, analytic_mask = select_time_window(
                times_after_warmup,
                short_window_start,
                short_window_end,
                normalize_start=not manual_window,
            )
            full_time = np.array(times_after_warmup, copy=True)
            time_series[scheme.label] = {
                "short": {"time": trimmed_time},
                "long": {"time": full_time},
            }

            quad_ref_full = _stack_columns(analytic_raw, ["quad_ref_x", "quad_ref_y", "quad_ref_z"])[analytic_warmup_mask]
            quad_actual_full = _stack_columns(analytic_raw, ["quad_actual_x", "quad_actual_y", "quad_actual_z"])[analytic_warmup_mask]
            quad_ref_vel_full = _stack_columns(analytic_raw, ["quad_ref_vx", "quad_ref_vy", "quad_ref_vz"])[analytic_warmup_mask]
            quad_actual_vel_full = _stack_columns(analytic_raw, ["quad_actual_vx", "quad_actual_vy", "quad_actual_vz"])[analytic_warmup_mask]
            payload_ref_full = _stack_columns(analytic_raw, ["payload_ref_x", "payload_ref_y", "payload_ref_z"])[analytic_warmup_mask]
            payload_actual_full = _stack_columns(analytic_raw, ["payload_actual_x", "payload_actual_y", "payload_actual_z"])[analytic_warmup_mask]
            payload_ref_vel_full = _stack_columns(analytic_raw, ["payload_ref_vx", "payload_ref_vy", "payload_ref_vz"])[analytic_warmup_mask]
            payload_actual_vel_full = _stack_columns(analytic_raw, ["payload_actual_vx", "payload_actual_vy", "payload_actual_vz"])[analytic_warmup_mask]

            macro_metrics[scenario.name][scheme.label] = {
                "quad_rmse": compute_rmse(quad_actual_full, quad_ref_full),
                "payload_rmse": compute_rmse(payload_actual_full, payload_ref_full),
            }

            span_masks = {
                "short": analytic_mask,
                "long": np.ones_like(times_after_warmup, dtype=bool),
            }
            analytic_data[scheme.label] = {}
            errors[scheme.label] = {}
            for span_name, span_mask in span_masks.items():
                quad_ref = quad_ref_full[span_mask]
                quad_actual = quad_actual_full[span_mask]
                quad_ref_vel = quad_ref_vel_full[span_mask]
                quad_actual_vel = quad_actual_vel_full[span_mask]
                payload_ref = payload_ref_full[span_mask]
                payload_actual = payload_actual_full[span_mask]
                payload_ref_vel = payload_ref_vel_full[span_mask]
                payload_actual_vel = payload_actual_vel_full[span_mask]

                analytic_data[scheme.label][span_name] = {
                    "quad_actual": quad_actual,
                    "quad_ref": quad_ref,
                    "quad_vel_actual": quad_actual_vel,
                    "quad_vel_ref": quad_ref_vel,
                    "payload_actual": payload_actual,
                    "payload_ref": payload_ref,
                    "payload_vel_actual": payload_actual_vel,
                    "payload_vel_ref": payload_ref_vel,
                }

                quad_pos_error = quad_actual - quad_ref
                quad_vel_error = quad_actual_vel - quad_ref_vel
                payload_pos_error = payload_actual - payload_ref
                payload_vel_error = payload_actual_vel - payload_ref_vel

                errors[scheme.label][span_name] = {
                    "quad_pos": {
                        "Quad X Position Error": quad_pos_error[:, 0],
                        "Quad Y Position Error": quad_pos_error[:, 1],
                        "Quad Z Position Error": quad_pos_error[:, 2],
                    },
                    "quad_vel": {
                        "Quad X Velocity Error": quad_vel_error[:, 0],
                        "Quad Y Velocity Error": quad_vel_error[:, 1],
                        "Quad Z Velocity Error": quad_vel_error[:, 2],
                    },
                    "payload_pos": {
                        "Load X Position Error": payload_pos_error[:, 0],
                        "Load Y Position Error": payload_pos_error[:, 1],
                        "Load Z Position Error": payload_pos_error[:, 2],
                    },
                    "payload_vel": {
                        "Load X Velocity Error": payload_vel_error[:, 0],
                        "Load Y Velocity Error": payload_vel_error[:, 1],
                        "Load Z Velocity Error": payload_vel_error[:, 2],
                    },
                }

            # Force logs sometimes use absolute epoch timestamps; normalize to a relative
            # clock before slicing so manual short windows (e.g., 25-40s) work as expected.
            force_time_raw = force_raw.get("time_sec") or force_raw.get("time")
            if force_time_raw is None or not len(force_time_raw):
                force_time_raw = np.arange(len(force_raw["fl_true_x"]), dtype=float)
            force_time_raw = np.asarray(force_time_raw, dtype=float)
            if len(force_time_raw):
                force_time_rel = force_time_raw - force_time_raw[0]
            else:
                force_time_rel = force_time_raw

            force_warmup_mask = force_time_rel >= warmup_trim_offline
            if warmup_trim_offline > 0.0 and not np.any(force_warmup_mask):
                # No samples survived the warmup trim for this scheme; fall back to using all samples
                # so that plots remain meaningful instead of silently dropping this scheme.
                force_warmup_mask = np.ones_like(force_time_rel, dtype=bool)
                force_time_after_warmup = np.array(force_time_rel, copy=True)
            else:
                force_time_after_warmup = force_time_rel[force_warmup_mask] - warmup_trim_offline

            force_time_trimmed, force_mask = select_time_window(
                force_time_after_warmup,
                short_window_start,
                short_window_end,
                normalize_start=not manual_window,
            )
            force_time_full = np.array(force_time_after_warmup, copy=True)
            force_data[scheme.label] = {
                "short": {"time": np.array(force_time_trimmed, copy=True)},
                "long": {"time": np.array(force_time_full, copy=True)},
            }
            load_true_full = np.stack(
                [force_raw["fl_true_x"], force_raw["fl_true_y"], force_raw["fl_true_z"]],
                axis=1,
            )
            load_est_full = np.stack(
                [force_raw["fl_est_x"], force_raw["fl_est_y"], force_raw["fl_est_z"]],
                axis=1,
            )
            quad_true_full = np.stack(
                [force_raw["fq_true_x"], force_raw["fq_true_y"], force_raw["fq_true_z"]],
                axis=1,
            )
            quad_est_full = np.stack(
                [force_raw["fq_est_x"], force_raw["fq_est_y"], force_raw["fq_est_z"]],
                axis=1,
            )

            if args.force_true_mode == "detrended":
                # Subtract a baseline (pre-warmup mean) to highlight gust-only component.
                baseline_mask = force_time_rel <= warmup_trim_offline
                if not np.any(baseline_mask):
                    baseline_mask = np.array([0], dtype=int)
                load_true_full = load_true_full - np.nanmean(load_true_full[baseline_mask], axis=0)
                quad_true_full = quad_true_full - np.nanmean(quad_true_full[baseline_mask], axis=0)

            load_true_full = load_true_full[force_warmup_mask]
            load_est_full = load_est_full[force_warmup_mask]
            quad_true_full = quad_true_full[force_warmup_mask]
            quad_est_full = quad_est_full[force_warmup_mask]

            load_err_total = np.linalg.norm(load_est_full - load_true_full, axis=1)
            quad_err_total = np.linalg.norm(quad_est_full - quad_true_full, axis=1)
            load_true_mag = np.linalg.norm(load_true_full, axis=1)
            load_est_mag = np.linalg.norm(load_est_full, axis=1)
            quad_true_mag = np.linalg.norm(quad_true_full, axis=1)
            quad_est_mag = np.linalg.norm(quad_est_full, axis=1)
            total_force_errors[scheme.label] = {}
            force_magnitudes[scheme.label] = {}
            force_masks = {
                "short": force_mask,
                "long": np.ones_like(force_mask, dtype=bool),
            }
            for span_name, mask in force_masks.items():
                if span_name == "short":
                    time_span = np.array(force_time_trimmed, copy=True)
                else:
                    time_span = np.array(force_time_full, copy=True)
                load_err_span = load_err_total[mask]
                quad_err_span = quad_err_total[mask]
                finite = (
                    np.isfinite(time_span)
                    & np.isfinite(load_err_span)
                    & np.isfinite(quad_err_span)
                )
                if not np.any(finite):
                    time_span = np.array([], dtype=float)
                    load_err_span = np.array([], dtype=float)
                    quad_err_span = np.array([], dtype=float)
                else:
                    time_span = time_span[finite]
                    load_err_span = load_err_span[finite]
                    quad_err_span = quad_err_span[finite]
                force_data[scheme.label][span_name]["time"] = time_span
                total_force_errors[scheme.label][span_name] = (
                    load_err_span,
                    quad_err_span,
                )
                load_true_span = load_true_mag[mask]
                load_est_span = load_est_mag[mask]
                quad_true_span = quad_true_mag[mask]
                quad_est_span = quad_est_mag[mask]
                if np.any(finite):
                    load_true_span = load_true_span[finite]
                    load_est_span = load_est_span[finite]
                    quad_true_span = quad_true_span[finite]
                    quad_est_span = quad_est_span[finite]
                else:
                    load_true_span = np.array([], dtype=float)
                    load_est_span = np.array([], dtype=float)
                    quad_true_span = np.array([], dtype=float)
                    quad_est_span = np.array([], dtype=float)
                force_magnitudes[scheme.label][span_name] = (
                    load_true_span,
                    load_est_span,
                    quad_true_span,
                    quad_est_span,
                )

        span_dirs = {
            "short": scenario_dir / "short",
            "long": scenario_dir / "long",
        }
        for dir_path in span_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        quad_tracking_entries = [
            ("Quad X Position", "quad_pos_x"),
            ("Quad Y Position", "quad_pos_y"),
            ("Quad Z Position", "quad_pos_z"),
            ("Quad X Velocity", "quad_vel_x"),
            ("Quad Y Velocity", "quad_vel_y"),
            ("Quad Z Velocity", "quad_vel_z"),
        ]
        load_tracking_entries = [
            ("Load X Position", "load_pos_x"),
            ("Load Y Position", "load_pos_y"),
            ("Load Z Position", "load_pos_z"),
            ("Load X Velocity", "load_vel_x"),
            ("Load Y Velocity", "load_vel_y"),
            ("Load Z Velocity", "load_vel_z"),
        ]
        quad_error_entries = [
            ("Quad Position Error X", "Quad X Position Error"),
            ("Quad Velocity Error X", "Quad X Velocity Error"),
            ("Quad Position Error Y", "Quad Y Position Error"),
            ("Quad Velocity Error Y", "Quad Y Velocity Error"),
            ("Quad Position Error Z", "Quad Z Position Error"),
            ("Quad Velocity Error Z", "Quad Z Velocity Error"),
        ]
        load_error_entries = [
            ("Load Position Error X", "Load X Position Error"),
            ("Load Velocity Error X", "Load X Velocity Error"),
            ("Load Position Error Y", "Load Y Position Error"),
            ("Load Velocity Error Y", "Load Y Velocity Error"),
            ("Load Position Error Z", "Load Z Position Error"),
            ("Load Velocity Error Z", "Load Z Velocity Error"),
        ]

        for span_name, span_dir in span_dirs.items():
            time_a = time_series[schemes[0].label][span_name]["time"]
            time_b = time_series[schemes[1].label][span_name]["time"]
            data_a = analytic_data[schemes[0].label][span_name]
            data_b = analytic_data[schemes[1].label][span_name]
            quad_actual_series_a = build_axis_series(data_a["quad_actual"])
            quad_actual_series_b = build_axis_series(data_b["quad_actual"])
            quad_vel_series_a = build_axis_series(data_a["quad_vel_actual"])
            quad_vel_series_b = build_axis_series(data_b["quad_vel_actual"])
            payload_actual_series_a = build_axis_series(data_a["payload_actual"])
            payload_actual_series_b = build_axis_series(data_b["payload_actual"])
            payload_vel_series_a = build_axis_series(data_a["payload_vel_actual"])
            payload_vel_series_b = build_axis_series(data_b["payload_vel_actual"])

            quad_pos_mag_a = np.linalg.norm(data_a["quad_actual"], axis=1)
            quad_pos_mag_b = np.linalg.norm(data_b["quad_actual"], axis=1)
            quad_vel_mag_a = np.linalg.norm(data_a["quad_vel_actual"], axis=1)
            quad_vel_mag_b = np.linalg.norm(data_b["quad_vel_actual"], axis=1)
            payload_pos_mag_a = np.linalg.norm(data_a["payload_actual"], axis=1)
            payload_pos_mag_b = np.linalg.norm(data_b["payload_actual"], axis=1)
            payload_vel_mag_a = np.linalg.norm(data_a["payload_vel_actual"], axis=1)
            payload_vel_mag_b = np.linalg.norm(data_b["payload_vel_actual"], axis=1)

            quad_pos_ref_mag = np.linalg.norm(data_a["quad_ref"], axis=1)
            quad_vel_ref_mag = np.linalg.norm(data_a["quad_vel_ref"], axis=1)
            payload_pos_ref_mag = np.linalg.norm(data_a["payload_ref"], axis=1)
            payload_vel_ref_mag = np.linalg.norm(data_a["payload_vel_ref"], axis=1)

            quad_pos_err_total_a = np.linalg.norm(data_a["quad_actual"] - data_a["quad_ref"], axis=1)
            quad_pos_err_total_b = np.linalg.norm(data_b["quad_actual"] - data_b["quad_ref"], axis=1)
            quad_vel_err_total_a = np.linalg.norm(data_a["quad_vel_actual"] - data_a["quad_vel_ref"], axis=1)
            quad_vel_err_total_b = np.linalg.norm(data_b["quad_vel_actual"] - data_b["quad_vel_ref"], axis=1)
            payload_pos_err_total_a = np.linalg.norm(data_a["payload_actual"] - data_a["payload_ref"], axis=1)
            payload_pos_err_total_b = np.linalg.norm(data_b["payload_actual"] - data_b["payload_ref"], axis=1)
            payload_vel_err_total_a = np.linalg.norm(data_a["payload_vel_actual"] - data_a["payload_vel_ref"], axis=1)
            payload_vel_err_total_b = np.linalg.norm(data_b["payload_vel_actual"] - data_b["payload_vel_ref"], axis=1)

            render_comparison_grid(
                times_a=time_a,
                series_a={
                    "quad_pos_x": quad_actual_series_a["X"],
                    "quad_pos_y": quad_actual_series_a["Y"],
                    "quad_pos_z": quad_actual_series_a["Z"],
                    "quad_vel_x": quad_vel_series_a["X"],
                    "quad_vel_y": quad_vel_series_a["Y"],
                    "quad_vel_z": quad_vel_series_a["Z"],
                    "quad_pos_ref_x": data_a["quad_ref"][:, 0],
                    "quad_pos_ref_y": data_a["quad_ref"][:, 1],
                    "quad_pos_ref_z": data_a["quad_ref"][:, 2],
                    "quad_vel_ref_x": data_a["quad_vel_ref"][:, 0],
                    "quad_vel_ref_y": data_a["quad_vel_ref"][:, 1],
                    "quad_vel_ref_z": data_a["quad_vel_ref"][:, 2],
                },
                times_b=time_b,
                series_b={
                    "quad_pos_x": quad_actual_series_b["X"],
                    "quad_pos_y": quad_actual_series_b["Y"],
                    "quad_pos_z": quad_actual_series_b["Z"],
                    "quad_vel_x": quad_vel_series_b["X"],
                    "quad_vel_y": quad_vel_series_b["Y"],
                    "quad_vel_z": quad_vel_series_b["Z"],
                    "quad_pos_ref_x": data_b["quad_ref"][:, 0],
                    "quad_pos_ref_y": data_b["quad_ref"][:, 1],
                    "quad_pos_ref_z": data_b["quad_ref"][:, 2],
                    "quad_vel_ref_x": data_b["quad_vel_ref"][:, 0],
                    "quad_vel_ref_y": data_b["quad_vel_ref"][:, 1],
                    "quad_vel_ref_z": data_b["quad_vel_ref"][:, 2],
                },
                entries=quad_tracking_entries,
                title="Quad Tracking Comparison",
                ylabel="Value",
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "quad_tracking.png",
            )

            render_comparison_grid(
                times_a=time_a,
                series_a={
                    "load_pos_x": payload_actual_series_a["X"],
                    "load_pos_y": payload_actual_series_a["Y"],
                    "load_pos_z": payload_actual_series_a["Z"],
                    "load_vel_x": payload_vel_series_a["X"],
                    "load_vel_y": payload_vel_series_a["Y"],
                    "load_vel_z": payload_vel_series_a["Z"],
                    "load_pos_ref_x": data_a["payload_ref"][:, 0],
                    "load_pos_ref_y": data_a["payload_ref"][:, 1],
                    "load_pos_ref_z": data_a["payload_ref"][:, 2],
                    "load_vel_ref_x": data_a["payload_vel_ref"][:, 0],
                    "load_vel_ref_y": data_a["payload_vel_ref"][:, 1],
                    "load_vel_ref_z": data_a["payload_vel_ref"][:, 2],
                },
                times_b=time_b,
                series_b={
                    "load_pos_x": payload_actual_series_b["X"],
                    "load_pos_y": payload_actual_series_b["Y"],
                    "load_pos_z": payload_actual_series_b["Z"],
                    "load_vel_x": payload_vel_series_b["X"],
                    "load_vel_y": payload_vel_series_b["Y"],
                    "load_vel_z": payload_vel_series_b["Z"],
                    "load_pos_ref_x": data_b["payload_ref"][:, 0],
                    "load_pos_ref_y": data_b["payload_ref"][:, 1],
                    "load_pos_ref_z": data_b["payload_ref"][:, 2],
                    "load_vel_ref_x": data_b["payload_vel_ref"][:, 0],
                    "load_vel_ref_y": data_b["payload_vel_ref"][:, 1],
                    "load_vel_ref_z": data_b["payload_vel_ref"][:, 2],
                },
                entries=load_tracking_entries,
                title="Load Tracking Comparison",
                ylabel="Value",
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "load_tracking.png",
            )

            render_error_comparison_grid(
                times_a=time_a,
                series_a={**errors[schemes[0].label][span_name]["quad_pos"], **errors[schemes[0].label][span_name]["quad_vel"]},
                times_b=time_b,
                series_b={**errors[schemes[1].label][span_name]["quad_pos"], **errors[schemes[1].label][span_name]["quad_vel"]},
                entries=quad_error_entries,
                title="Quad Error Comparison",
                ylabel="Error",
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "quad_errors.png",
            )

            render_error_comparison_grid(
                times_a=time_a,
                series_a={**errors[schemes[0].label][span_name]["payload_pos"], **errors[schemes[0].label][span_name]["payload_vel"]},
                times_b=time_b,
                series_b={**errors[schemes[1].label][span_name]["payload_pos"], **errors[schemes[1].label][span_name]["payload_vel"]},
                entries=load_error_entries,
                title="Load Error Comparison",
                ylabel="Error",
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "load_errors.png",
            )

            render_force_errors(
                times_a=force_data[schemes[0].label][span_name]["time"],
                load_err_a=total_force_errors[schemes[0].label][span_name][0],
                quad_err_a=total_force_errors[schemes[0].label][span_name][1],
                times_b=force_data[schemes[1].label][span_name]["time"],
                load_err_b=total_force_errors[schemes[1].label][span_name][0],
                quad_err_b=total_force_errors[schemes[1].label][span_name][1],
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "force_total_errors.png",
                span_hint=span_hint if span_name == "short" else None,
                window_start=short_window_start if span_name == "short" else None,
                window_end=short_window_end if span_name == "short" else None,
            )

            load_true_a, load_est_a, quad_true_a, quad_est_a = force_magnitudes[schemes[0].label][span_name]
            _, load_est_b, _, quad_est_b = force_magnitudes[schemes[1].label][span_name]
            render_force_totals(
                times_a=force_data[schemes[0].label][span_name]["time"],
                load_true_a=load_true_a,
                load_est_a=load_est_a,
                quad_true_a=quad_true_a,
                quad_est_a=quad_est_a,
                times_b=force_data[schemes[1].label][span_name]["time"],
                load_est_b=load_est_b,
                quad_est_b=quad_est_b,
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "force_total.png",
                span_hint=span_hint if span_name == "short" else None,
                window_start=short_window_start if span_name == "short" else None,
                window_end=short_window_end if span_name == "short" else None,
            )

            render_state_totals(
                times_a=time_a,
                quad_pos_a=quad_pos_mag_a,
                quad_vel_a=quad_vel_mag_a,
                load_pos_a=payload_pos_mag_a,
                load_vel_a=payload_vel_mag_a,
                times_b=time_b,
                quad_pos_b=quad_pos_mag_b,
                quad_vel_b=quad_vel_mag_b,
                load_pos_b=payload_pos_mag_b,
                load_vel_b=payload_vel_mag_b,
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "state_total.png",
                times_ref=time_a,
                quad_pos_ref=quad_pos_ref_mag,
                quad_vel_ref=quad_vel_ref_mag,
                load_pos_ref=payload_pos_ref_mag,
                load_vel_ref=payload_vel_ref_mag,
            )

            render_state_total_errors(
                times_a=time_a,
                quad_pos_err_a=quad_pos_err_total_a,
                quad_vel_err_a=quad_vel_err_total_a,
                load_pos_err_a=payload_pos_err_total_a,
                load_vel_err_a=payload_vel_err_total_a,
                times_b=time_b,
                quad_pos_err_b=quad_pos_err_total_b,
                quad_vel_err_b=quad_vel_err_total_b,
                load_pos_err_b=payload_pos_err_total_b,
                load_vel_err_b=payload_vel_err_total_b,
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "state_total_errors.png",
            )

            render_xy_comparison(
                quad_ref=data_a["quad_ref"],
                quad_a=data_a["quad_actual"],
                quad_b=data_b["quad_actual"],
                payload_ref=data_a["payload_ref"],
                payload_a=data_a["payload_actual"],
                payload_b=data_b["payload_actual"],
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "trajectory_xy.png",
            )

            render_trajectory_3d_comparison(
                quad_ref=data_a["quad_ref"],
                quad_a=data_a["quad_actual"],
                quad_b=data_b["quad_actual"],
                payload_ref=data_a["payload_ref"],
                payload_a=data_a["payload_actual"],
                payload_b=data_b["payload_actual"],
                label_a=schemes[0].label,
                label_b=schemes[1].label,
                output_path=span_dir / "trajectory_3d.png",
            )

    if scenarios:
        primary_label = scenario_labels.get(scenarios[0].name, "macro")
        macro_dir = comparison_dir / primary_label
        macro_dir.mkdir(parents=True, exist_ok=True)
        render_macro_bars(
            scenario_names=[s.name for s in scenarios],
            metrics=macro_metrics,
            schemes=schemes,
            output_path=macro_dir / "macro_rmse.png",
        )

    print(f"[done] Comparison plots saved under {comparison_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
