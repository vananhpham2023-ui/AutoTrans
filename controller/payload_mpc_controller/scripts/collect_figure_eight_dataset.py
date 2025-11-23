#!/usr/bin/env python3
"""
Automate data collection and cleaning for the three figure-eight turbulence scenarios
described in TASK3. The script sequentially launches each scenario, records a fixed
number of samples via `pinn_data_recorder.py`, and finally merges/cleans all CSVs
into a single dataset suitable for PINN training.

Usage (from the catkin workspace root):
    rosrun payload_mpc_controller collect_figure_eight_dataset.py \
        --samples 150000 --warmup 8.0 --record-duration 45.0 \
        --output-root src/AutoTrans/controller/payload_mpc_controller/plots/figure_eight_batch
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SCENARIOS = [
    {
        "name": "figure_eight_turbulence_light",
        "wind_sigma": [1.086, 1.086, 0.77],
        "wind_length": [420.0, 420.0, 300.0],
    },
    {
        "name": "figure_eight_turbulence_medium",
        "wind_sigma": [2.171, 2.171, 1.54],
        "wind_length": [420.0, 420.0, 300.0],
    },
    {
        "name": "figure_eight_turbulence_heavy",
        "wind_sigma": [3.256, 3.256, 2.31],
        "wind_length": [420.0, 420.0, 300.0],
    },
]

REQUIRED_FIELDS = [
    "timestamp",
    "quad_acc_x",
    "quad_acc_y",
    "quad_acc_z",
    "load_acc_x",
    "load_acc_y",
    "load_acc_z",
    "cable_dir_x",
    "cable_dir_y",
    "cable_dir_z",
    "quad_rot_r13",
    "quad_rot_r23",
    "quad_rot_r33",
    "thrust_total",
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=150000,
        help="Maximum samples to record per scenario (default: %(default)s)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=6.0,
        help="Warmup time before starting recorder (seconds).",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=60.0,
        help="Extra duration to wait after recorder starts (seconds).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="plots/figure_eight_batch",
        help="Root directory to store raw runs and cleaned dataset.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="devel/setup.bash",
        help="Setup file to source before launching ROS commands.",
    )
    return parser.parse_args(argv)


def resolve_workspace_root(script_path: Path) -> Path:
    current = script_path
    while current != current.parent:
        if (current / "src").exists() and ((current / "devel").exists() or (current / "build").exists()):
            return current
        current = current.parent
    return script_path


def run_command(cmd: str, setup_file: str | None, env: Dict[str, str]) -> subprocess.Popen:
    if setup_file:
        wrapper = ["bash", "-lc", f"source {setup_file} >/dev/null 2>&1 && {cmd}"]
    else:
        wrapper = ["bash", "-lc", cmd]
    return subprocess.Popen(wrapper, env=env, preexec_fn=os.setsid)


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=5.0)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def collect_for_scenario(
    scenario: Dict[str, object],
    args: argparse.Namespace,
    workspace_root: Path,
    output_root: Path,
) -> Path:
    setup_file: str | None = None
    if args.setup:
        setup_path = Path(args.setup)
        if not setup_path.is_absolute():
            setup_path = workspace_root / setup_path
        setup_file = str(setup_path.resolve())

    scenario_dir = output_root / scenario["name"]
    pinn_dir = scenario_dir / "pinn"
    pinn_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pinn_dir / f"pinn_dataset_{scenario['name']}.csv"

    base_launch = (
        "roslaunch payload_planner controller_only.launch "
        "trajectory_mode:=figure_eight figure_radius:=3.0 figure_omega:=0.8 "
        "figure_center_x:=0.0 figure_center_y:=0.0 figure_altitude:=2.0 "
        "wind_type:=composite wind_constant_velocity:='[5.2,5.2,5.2]' "
        f"wind_dryden_sigma:='{scenario['wind_sigma']}' "
        f"wind_dryden_length_scale:='{scenario['wind_length']}' "
        f"run_label:={scenario['name']}"
    )

    env = os.environ.copy()
    env["AUTOTRANS_RUN_TAG"] = scenario["name"]
    env["PINN_DATA_OUTPUT"] = str(pinn_dir)
    env["PINN_SCENARIO_ID"] = scenario["name"]

    launch_proc = run_command(base_launch, setup_file, env)
    time.sleep(args.warmup)

    recorder_cmd = (
        "rosrun payload_mpc_controller pinn_data_recorder.py "
        f"_scenario_id:={scenario['name']} "
        f"_output_dir:={pinn_dir} "
        f"_output_filename:={csv_path.name} "
        f"_max_samples:={args.samples}"
    )
    recorder_proc = run_command(recorder_cmd, setup_file, env)

    time.sleep(args.record_duration)

    stop_process(recorder_proc)
    stop_process(launch_proc)

    if not csv_path.exists():
        raise RuntimeError(f"Scenario {scenario['name']} failed: missing {csv_path}")
    return csv_path


def is_valid_row(row: Dict[str, str]) -> bool:
    try:
        for key in REQUIRED_FIELDS:
            if key not in row or row[key].strip() == "":
                return False
        values = [float(row[key]) for key in REQUIRED_FIELDS if key != "timestamp"]
        if not all(math.isfinite(v) for v in values):
            return False
        thrust = abs(float(row["thrust_total"]))
        if thrust <= 0.0 or thrust > 150.0:
            return False
        return True
    except ValueError:
        return False


def merge_and_clean(csv_paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as out_file:
        writer: csv.writer = csv.writer(out_file)
        header_written = False
        kept = 0
        for path in csv_paths:
            with path.open() as in_file:
                reader = csv.DictReader(in_file)
                if not header_written:
                    writer.writerow(reader.fieldnames or [])
                    header_written = True
                for row in reader:
                    if not row:
                        continue
                    if is_valid_row(row):
                        writer.writerow([row.get(col, "") for col in reader.fieldnames])
                        kept += 1
        print(f"[clean] kept {kept} samples written to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    workspace_root = resolve_workspace_root(script_dir)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = workspace_root / output_root
    output_root = output_root.resolve()

    raw_csvs: List[Path] = []
    for scenario in SCENARIOS:
        print(f"[collect] Running {scenario['name']} ...")
        csv_path = collect_for_scenario(scenario, args, workspace_root, output_root)
        raw_csvs.append(csv_path)
        print(f"[collect] Scenario {scenario['name']} completed: {csv_path}")

    cleaned_path = output_root / "clean_dataset.csv"
    merge_and_clean(raw_csvs, cleaned_path)
    print(f"[done] Clean dataset ready at {cleaned_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
