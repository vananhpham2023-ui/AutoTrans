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
SCHEME_LINESTYLES = ["-", "--"]
SUBPLOT_W = 5.0
SUBPLOT_H = 4.0


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


def select_time_window(times: Sequence[float], window_start: Optional[float], window_end: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
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
    if trimmed_times.size:
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
) -> Tuple[List[str], Mapping[str, str]]:
    launch_args = {str(k): format_arg_value(v) for k, v in scenario.args.items()}
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
) -> RunArtifacts:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag = f"{scenario.safe_name}_{scheme.safe_label}_{timestamp}"
    cmd, env = build_launch_command(scenario, scheme, run_tag, setup_script, workspace_root)
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
    max_time = max(
        times_a[-1] if len(times_a) else 0.0,
        times_b[-1] if len(times_b) else 0.0,
    )
    if max_time <= 0:
        max_time = 1.0
    ticks = np.linspace(0.0, max_time, num=5)
    for axis, (title_label, key) in zip(axes_flat, entries):
        axis.plot(
            times_a,
            series_a[key],
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.plot(
            times_b,
            series_b[key],
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(0.0, max_time)
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
    max_time = max(
        times_a[-1] if len(times_a) else 0.0,
        times_b[-1] if len(times_b) else 0.0,
    )
    if max_time <= 0:
        max_time = 1.0
    ticks = np.linspace(0.0, max_time, num=5)

    for axis, (title_label, key) in zip(axes_flat, entries):
        axis.plot(
            times_a,
            series_a[key],
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.4,
        )
        axis.plot(
            times_b,
            series_b[key],
            label=label_b,
            color=SCHEME_COLORS[1],
            linestyle=SCHEME_LINESTYLES[1],
            linewidth=1.4,
        )
        axis.set_title(title_label)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.legend(loc="upper right")
        axis.set_xlim(0.0, max_time)
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
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(SUBPLOT_W * 2, SUBPLOT_H), sharex=True)
    axis_data = [
        (axes[0], "Load Total Force Error [N]", load_err_a, load_err_b),
        (axes[1], "Quad Total Force Error [N]", quad_err_a, quad_err_b),
    ]
    inferred_max = max(
        times_a[-1] if len(times_a) else 0.0,
        times_b[-1] if len(times_b) else 0.0,
    )
    max_time = span_hint if (span_hint is not None and span_hint > 0.0) else inferred_max
    if max_time <= 0:
        max_time = 1.0
    ticks = np.linspace(0.0, max_time, num=5)
    for axis, title, series_a, series_b in axis_data:
        axis.plot(
            times_a,
            series_a,
            label=label_a,
            color=SCHEME_COLORS[0],
            linestyle=SCHEME_LINESTYLES[0],
            linewidth=1.3,
        )
        axis.plot(
            times_b,
            series_b,
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
        axis.set_xlim(0.0, max_time)
        axis.set_xticks(ticks)
    fig.suptitle("Force Estimation Error Comparison", fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
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
        axis.plot(a_xy[:, 0], a_xy[:, 1], label=label_a, color=SCHEME_COLORS[0], linestyle=SCHEME_LINESTYLES[1], linewidth=1.4)
        axis.plot(b_xy[:, 0], b_xy[:, 1], label=label_b, color=SCHEME_COLORS[1], linestyle=SCHEME_LINESTYLES[1], linewidth=1.4)
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
    scenarios = load_scenarios(args.scenario_config)

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
        scenario_label = allocate_scenario_label(
            scenario.safe_name,
            payload_pkg_dir / "plots",
            comparison_dir,
        )
        scenario_labels[scenario.name] = scenario_label
        scenario_output_root = payload_pkg_dir / "plots" / scenario_label
        scenario_output_root.mkdir(parents=True, exist_ok=True)

        run_artifacts[scenario.name] = {}
        for scheme in schemes:
            cleanup_stale_processes()
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
            )
            run_artifacts[scenario.name][scheme.label] = artifacts
            time.sleep(max(args.scenario_time_gap, 0.0))

    macro_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for scenario in scenarios:
        scheme_data = run_artifacts[scenario.name]
        scenario_label = scenario_labels[scenario.name]
        scenario_dir = comparison_dir / scenario_label
        scenario_dir.mkdir(parents=True, exist_ok=True)
        window_start, window_end = resolve_cycle_window(scenario)
        span_hint = None
        if window_start is not None and window_end is not None and window_end > window_start:
            span_hint = window_end - window_start

        analytic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        force_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        time_series: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        errors: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {}
        total_force_errors: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

        macro_metrics[scenario.name] = {}

        for idx, scheme in enumerate(schemes):
            artifacts = scheme_data[scheme.label]
            analytic_raw = load_analytic_csv(artifacts.analytic_csv)
            force_raw = load_force_csv(artifacts.force_csv)

            times = analytic_raw.get("time_sec")
            if times is None or not len(times):
                times = analytic_raw.get("time")
            if times is None or not len(times):
                times = np.arange(len(next(iter(analytic_raw.values()))), dtype=float)
            trimmed_time, analytic_mask = select_time_window(times, window_start, window_end)
            full_time = np.array(times, copy=True)
            if len(full_time):
                full_time = full_time - full_time[0]
            time_series[scheme.label] = {
                "short": {"time": trimmed_time},
                "long": {"time": full_time},
            }

            quad_ref_full = _stack_columns(analytic_raw, ["quad_ref_x", "quad_ref_y", "quad_ref_z"])
            quad_actual_full = _stack_columns(analytic_raw, ["quad_actual_x", "quad_actual_y", "quad_actual_z"])
            quad_ref_vel_full = _stack_columns(analytic_raw, ["quad_ref_vx", "quad_ref_vy", "quad_ref_vz"])
            quad_actual_vel_full = _stack_columns(analytic_raw, ["quad_actual_vx", "quad_actual_vy", "quad_actual_vz"])
            payload_ref_full = _stack_columns(analytic_raw, ["payload_ref_x", "payload_ref_y", "payload_ref_z"])
            payload_actual_full = _stack_columns(analytic_raw, ["payload_actual_x", "payload_actual_y", "payload_actual_z"])
            payload_ref_vel_full = _stack_columns(analytic_raw, ["payload_ref_vx", "payload_ref_vy", "payload_ref_vz"])
            payload_actual_vel_full = _stack_columns(analytic_raw, ["payload_actual_vx", "payload_actual_vy", "payload_actual_vz"])

            macro_metrics[scenario.name][scheme.label] = {
                "quad_rmse": compute_rmse(quad_actual_full, quad_ref_full),
                "payload_rmse": compute_rmse(payload_actual_full, payload_ref_full),
            }

            span_masks = {
                "short": analytic_mask,
                "long": np.ones_like(analytic_mask, dtype=bool),
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

            force_time = force_raw.get("time_sec") or force_raw.get("time")
            if force_time is None or not len(force_time):
                force_time = np.arange(len(force_raw["fl_true_x"]), dtype=float)
            force_time_trimmed, force_mask = select_time_window(force_time, window_start, window_end)
            force_time_full = np.array(force_time, copy=True)
            if len(force_time_full):
                force_time_full = force_time_full - force_time_full[0]
            force_data[scheme.label] = {
                "short": {"time": np.array(force_time_trimmed, copy=True)},
                "long": {"time": np.array(force_time_full, copy=True)},
            }
            load_true = np.stack(
                [force_raw["fl_true_x"], force_raw["fl_true_y"], force_raw["fl_true_z"]],
                axis=1,
            )
            load_est = np.stack(
                [force_raw["fl_est_x"], force_raw["fl_est_y"], force_raw["fl_est_z"]],
                axis=1,
            )
            quad_true = np.stack(
                [force_raw["fq_true_x"], force_raw["fq_true_y"], force_raw["fq_true_z"]],
                axis=1,
            )
            quad_est = np.stack(
                [force_raw["fq_est_x"], force_raw["fq_est_y"], force_raw["fq_est_z"]],
                axis=1,
            )
            load_err_total = np.linalg.norm(load_est - load_true, axis=1)
            quad_err_total = np.linalg.norm(quad_est - quad_true, axis=1)
            total_force_errors[scheme.label] = {}
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
