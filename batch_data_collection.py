#!/usr/bin/env python3
"""Automate batch PINN data collection across trajectory × wind scenarios."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EXPECTED_PINN_FIELDS = [
    "timestamp",
    "scenario_id",
    "trajectory_mode",
    "wind_type",
    "quad_pos_x",
    "quad_pos_y",
    "quad_pos_z",
    "quad_vel_x",
    "quad_vel_y",
    "quad_vel_z",
    "quad_rot_r13",
    "quad_rot_r23",
    "quad_rot_r33",
    "quad_omega_x",
    "quad_omega_y",
    "quad_omega_z",
    "quad_acc_x",
    "quad_acc_y",
    "quad_acc_z",
    "load_pos_x",
    "load_pos_y",
    "load_pos_z",
    "load_vel_x",
    "load_vel_y",
    "load_vel_z",
    "load_acc_x",
    "load_acc_y",
    "load_acc_z",
    "cable_dir_x",
    "cable_dir_y",
    "cable_dir_z",
    "motor_rpm_1",
    "motor_rpm_2",
    "motor_rpm_3",
    "motor_rpm_4",
    "thrust_total",
    "mass_quad",
    "mass_load",
    "l_length",
    "sqrt_kf",
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
    "fl_est_x",
    "fl_est_y",
    "fl_est_z",
    "fq_est_x",
    "fq_est_y",
    "fq_est_z",
    "wind_x",
    "wind_y",
    "wind_z",
]

TRAJECTORIES = ("circle", "figure_eight")
PURE_TURBULENCE = {
    "c1_low": {"sigma": [0.6, 0.6, 0.3], "length": [12.0, 12.0, 6.0]},
    "c2_medium": {"sigma": [1.5, 1.5, 0.5], "length": [18.0, 18.0, 9.0]},
    "c3_high": {"sigma": [2.5, 2.5, 0.8], "length": [25.0, 25.0, 12.0]},
}

WIND_DIRECTIONS = {
    "w1": {"label": "horizontal_x", "vec": lambda v: [v, 0.0, 0.0]},
    "w2": {"label": "vertical", "vec": lambda v: [0.0, 0.0, v]},
    "w3": {
        "label": "horizontal_diagonal",
        "vec": lambda v: [round(v / math.sqrt(2), 3), round(v / math.sqrt(2), 3), 0.0],
    },
    "w4": {
        "label": "ascending_oblique",
        "vec": lambda v: [round(v / math.sqrt(3), 3)] * 3,
    },
    "w5": {"label": "biaxial_xy", "vec": lambda v: [v, v, 0.0]},
    "w6": {
        "label": "descending_oblique",
        "vec": lambda v: [round(v / math.sqrt(3), 3), round(-v / math.sqrt(3), 3), round(-v / math.sqrt(3), 3)],
    },
}

SPEEDS = {"weak": 2.0, "medium": 5.0, "strong": 9.0}
TURBULENCE_LEVELS = {
    "low": {"sigma": [0.6, 0.6, 0.3], "length": [12.0, 12.0, 6.0]},
    "medium": {"sigma": [1.5, 1.5, 0.5], "length": [18.0, 18.0, 9.0]},
    "high": {"sigma": [2.5, 2.5, 0.8], "length": [25.0, 25.0, 12.0]},
}

DEFAULT_MIN_SAMPLES = 1000
DEFAULT_WARMUP = 5.0
DEFAULT_SCENARIO_TIMEOUT = 420.0
DEFAULT_RECORDER_MAX_SAMPLES = 0
DEFAULT_RECORDER_GRACE = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 5.0
DEFAULT_CHECKPOINT = "batch_checkpoint.json"
DEFAULT_REPORT_FILE = "batch_run_report.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_BASE_PORT = 11311
DEFAULT_CYCLES_TARGET = 10.0
DEFAULT_TIMEOUT_MARGIN = 20.0
DEFAULT_COMPLETION_TOPIC = "/mpc/run_completed"


@dataclass
class ScenarioSpec:
    name: str
    trajectory: str
    wind: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def safe_name(self) -> str:
        return "".join(c if (c.isalnum() or c in "-_") else "-" for c in self.name)


@dataclass
class ScenarioResult:
    scenario: ScenarioSpec
    status: str
    samples: int = 0
    pinn_csv: Optional[Path] = None
    force_csv: Optional[Path] = None
    analytic_csv: Optional[Path] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    attempts: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario.name,
            "status": self.status,
            "samples": self.samples,
            "pinn_csv": str(self.pinn_csv) if self.pinn_csv else None,
            "force_csv": str(self.force_csv) if self.force_csv else None,
            "analytic_csv": str(self.analytic_csv) if self.analytic_csv else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "notes": self.notes,
            "metadata": self.scenario.metadata,
        }


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, object]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                self._data = {}

    def mark(self, result: ScenarioResult) -> None:
        with self._lock:
            self._data[result.scenario.name] = result.to_dict()
            if self.path.parent and self.path.parent != Path(""):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    def get(self, scenario_name: str) -> Optional[Dict[str, object]]:
        return self._data.get(scenario_name)

    def completed(self) -> Dict[str, Dict[str, object]]:
        return {k: v for k, v in self._data.items() if v.get("status") == "success"}


class ScenarioRegistry:
    def __init__(self) -> None:
        self.scenarios: Dict[str, ScenarioSpec] = {}
        self._build_all()

    def _build_all(self) -> None:
        for traj in TRAJECTORIES:
            for base_name, meta in PURE_TURBULENCE.items():
                name = f"{traj}.{base_name}"
                metadata = {"type": "dryden", **meta}
                self.scenarios[name] = ScenarioSpec(name, traj, base_name, metadata)

            for direction_key, direction_meta in WIND_DIRECTIONS.items():
                for speed_key, speed_value in SPEEDS.items():
                    for turb_key, turb_meta in TURBULENCE_LEVELS.items():
                        wind_name = f"{direction_key}_{speed_key}_{turb_key}"
                        full_name = f"{traj}.{wind_name}"
                        vec = direction_meta["vec"](speed_value)
                        metadata = {
                            "type": "composite",
                            "direction": direction_meta["label"],
                            "speed": speed_value,
                            "speed_label": speed_key,
                            "sigma": turb_meta["sigma"],
                            "length": turb_meta["length"],
                            "wind_vector": vec,
                        }
                        self.scenarios[full_name] = ScenarioSpec(full_name, traj, wind_name, metadata)

    def resolve(self, names: Iterable[str]) -> List[ScenarioSpec]:
        resolved: List[ScenarioSpec] = []
        for name in names:
            spec = self.scenarios.get(name)
            if not spec:
                raise ValueError(f"Unknown scenario '{name}'")
            resolved.append(self._clone(spec))
        return resolved

    def profile(self, profile_name: str) -> List[ScenarioSpec]:
        if profile_name == "full":
            names = sorted(self.scenarios)
        elif profile_name == "lite":
            names = sorted(self._lite_names())
        else:
            raise ValueError(f"Unsupported profile '{profile_name}'")
        return [self._clone(self.scenarios[n]) for n in names]

    def _lite_names(self) -> List[str]:
        keep = set()
        for traj in TRAJECTORIES:
            for base in PURE_TURBULENCE:
                keep.add(f"{traj}.{base}")
            for direction in WIND_DIRECTIONS:
                for speed in ("weak", "strong"):
                    for turb in ("low", "high"):
                        keep.add(f"{traj}.{direction}_{speed}_{turb}")
        return sorted(keep)

    def _clone(self, spec: ScenarioSpec) -> ScenarioSpec:
        return ScenarioSpec(spec.name, spec.trajectory, spec.wind, dict(spec.metadata))


class GracefulTerminator:
    def __init__(self) -> None:
        self._stop = threading.Event()
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:  # type: ignore[override]
        print(f"\n[batch] Received signal {signum}, stopping after current tasks...", file=sys.stderr)
        self._stop.set()

    def requested(self) -> bool:
        return self._stop.is_set()


class PortAllocator:
    def __init__(self, base: int = 11311) -> None:
        self._next = base
        self._lock = threading.Lock()

    def acquire(self) -> int:
        with self._lock:
            port = self._next
            self._next += 1
            return port


class ProcessGroup:
    def __init__(self, popen: subprocess.Popen, log_handle=None):
        self.popen = popen
        self._log_handle = log_handle

    def wait(self, timeout: Optional[float] = None) -> int:
        code = self.popen.wait(timeout=timeout)
        self._close_log()
        return code

    def poll(self) -> Optional[int]:
        return self.popen.poll()

    def terminate(self, sig: int = signal.SIGINT, wait: float = 5.0) -> None:
        if self.popen.poll() is None:
            try:
                os.killpg(os.getpgid(self.popen.pid), sig)
            except ProcessLookupError:
                pass
            try:
                self.popen.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.popen.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle and not self._log_handle.closed:
            self._log_handle.close()
            self._log_handle = None


class ScenarioRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        workspace_root: Path,
        payload_pkg: Path,
        registry: ScenarioRegistry,
    ) -> None:
        self.args = args
        self.workspace_root = workspace_root
        self.payload_pkg = payload_pkg
        self.registry = registry
        self.port_allocator = PortAllocator(args.base_port)
        log_dir = Path(args.log_dir)
        if not log_dir.is_absolute():
            log_dir = self.payload_pkg / "plots" / log_dir
        self.stdout_dir = log_dir
        self.stdout_dir.mkdir(parents=True, exist_ok=True)
        if args.setup_file:
            self.setup_file: Optional[str] = args.setup_file
        else:
            candidate = self.workspace_root / "devel" / "setup.bash"
            self.setup_file = str(candidate) if candidate.exists() else None

    def run(self, scenario: ScenarioSpec, attempt: int) -> ScenarioResult:
        start_time = dt.datetime.utcnow()
        result = ScenarioResult(
            scenario=scenario,
            status="running",
            started_at=start_time.isoformat(),
            attempts=attempt,
        )
        warmup = float(scenario.metadata.get("warmup", self.args.warmup))
        timeout = self._scenario_timeout(scenario)
        recorder_grace = float(scenario.metadata.get("recorder_grace", self.args.recorder_grace))
        min_samples = int(scenario.metadata.get("min_samples", self.args.min_samples))
        max_samples = int(scenario.metadata.get("recorder_max_samples", self.args.recorder_max_samples))
        run_seed = self._run_seed(attempt)
        run_tag = f"{scenario.safe_name}_{run_seed}"
        output_root = self.payload_pkg / "plots" / f"run_{run_tag}"
        pinn_dir = output_root / "pinn"
        force_dir = output_root / "force"
        analytic_dir = output_root / "analytic"
        for directory in (pinn_dir, force_dir, analytic_dir):
            directory.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["PINN_SCENARIO_ID"] = scenario.name
        env["PINN_DATA_OUTPUT"] = str(pinn_dir)
        env["FORCE_DATA_OUTPUT"] = str(force_dir)
        env["AUTOTRANS_ANALYTIC_DIR"] = str(analytic_dir)
        env["AUTOTRANS_RUN_TAG"] = run_tag
        ros_port = self.port_allocator.acquire()
        attempts = 0
        while self._port_in_use(ros_port):
            result.notes.append(f"Port {ros_port} busy, retrying")
            ros_port = self.port_allocator.acquire()
            attempts += 1
            if attempts > 20:
                raise RuntimeError("Unable to allocate free ROS master port")
        env["ROS_MASTER_URI"] = f"http://127.0.0.1:{ros_port}"
        env.setdefault("ROS_IP", "127.0.0.1")
        env.setdefault("ROS_HOSTNAME", "127.0.0.1")

        setup_file = self.setup_file

        master: Optional[ProcessGroup] = None
        roslaunch_pg: Optional[ProcessGroup] = None
        recorder_pg: Optional[ProcessGroup] = None
        completion_pg: Optional[ProcessGroup] = None
        completion_triggered = False
        try:
            master = self._start_roscore(ros_port, env, setup_file)
            roslaunch_pg = self._start_roslaunch(scenario, run_seed, env, setup_file)
            time.sleep(max(0.0, warmup))
            recorder_pg, _ = self._start_recorder(
                scenario,
                pinn_dir,
                run_tag,
                env,
                setup_file,
                max_samples,
            )
            topic = (getattr(self.args, "completion_topic", DEFAULT_COMPLETION_TOPIC) or "").strip()
            if topic:
                completion_pg = self._start_completion_waiter(
                    run_tag=run_tag,
                    env=env,
                    setup_file=setup_file,
                    topic=topic,
                )
                time.sleep(0.2)
                if completion_pg.poll() is not None:
                    result.notes.append("completion waiter failed to start")
                    completion_pg = None
            # Wait for main simulation up to timeout. If it exceeds, treat as normal cutoff.
            cutoff = False
            try:
                completion_triggered = self._wait_for_completion(roslaunch_pg, timeout, completion_pg)
            except RuntimeError as exc:
                if "Process exceeded timeout" in str(exc):
                    cutoff = True
                    result.notes.append("roslaunch reached timeout; terminating and proceeding to finalize")
                else:
                    raise
            finally:
                if roslaunch_pg:
                    roslaunch_pg.terminate()
                    roslaunch_pg = None

            # Give recorder time to flush even if timeout/cutoff happened
            try:
                grace = max(0.0, recorder_grace)
                if cutoff:
                    grace = max(grace, self.args.timeout_margin)
                self._wait_for_completion(recorder_pg, grace)
            except RuntimeError as exc:
                if "Process exceeded timeout" in str(exc):
                    result.notes.append("recorder grace window reached; finalizing")
                else:
                    raise
            finally:
                recorder_pg = None
            if completion_triggered:
                result.notes.append("completion topic signaled run end")
        except Exception as exc:  # pylint: disable=broad-except
            result.status = "failed"
            result.notes.append(str(exc))
        finally:
            if completion_pg:
                completion_pg.terminate()
            if recorder_pg:
                recorder_pg.terminate()
            if roslaunch_pg:
                roslaunch_pg.terminate()
            if master:
                master.terminate()

        end_time = dt.datetime.utcnow()
        result.finished_at = end_time.isoformat()
        if result.status != "failed":
            validation = self._validate_output(output_root, min_samples)
            result.status = "success" if validation["ok"] else "failed"
            result.samples = validation.get("samples", 0)
            result.pinn_csv = validation.get("pinn_csv")
            result.force_csv = validation.get("force_csv")
            result.analytic_csv = validation.get("analytic_csv")
            result.notes.extend(validation.get("notes", []))
        return result

    def _start_roscore(self, port: int, env: Dict[str, str], setup_file: Optional[str]) -> ProcessGroup:
        cmd = f"roscore -p {port}"
        log_handle = self._open_log(f"roscore_{port}")
        popen = subprocess.Popen(
            self._wrap_command(cmd, setup_file),
            cwd=str(self.workspace_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self._wait_for_master(env["ROS_MASTER_URI"], timeout=10.0)
        return ProcessGroup(popen, log_handle)

    def _start_roslaunch(
        self,
        scenario: ScenarioSpec,
        run_seed: str,
        env: Dict[str, str],
        setup_file: Optional[str],
    ) -> ProcessGroup:
        scenario_arg = scenario.name
        cmd = (
            "roslaunch --wait payload_planner controller_only.launch "
            f"scenario:={scenario_arg} run_seed:={run_seed} "
            f"run_label:={scenario_arg}"
        )
        log_handle = self._open_log(f"launch_{scenario.safe_name}_{run_seed}")
        popen = subprocess.Popen(
            self._wrap_command(cmd, setup_file),
            cwd=str(self.workspace_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        return ProcessGroup(popen, log_handle)

    def _start_recorder(
        self,
        scenario: ScenarioSpec,
        output_dir: Path,
        run_tag: str,
        env: Dict[str, str],
        setup_file: Optional[str],
        max_samples: int,
    ) -> Tuple[ProcessGroup, Path]:
        filename = f"pinn_dataset_{run_tag}.csv"
        node_name = f"pinn_data_recorder_{self._safe_name(run_tag)}"
        cmd = (
            "rosrun payload_mpc_controller pinn_data_recorder.py "
            f"_scenario_id:={scenario.name} _output_dir:={output_dir} "
            f"_output_filename:={filename} _max_samples:={max_samples} "
            f"__name:={node_name}"
        )
        log_handle = self._open_log(f"recorder_{scenario.safe_name}_{run_tag}")
        popen = subprocess.Popen(
            self._wrap_command(cmd, setup_file),
            cwd=str(self.workspace_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        return ProcessGroup(popen, log_handle), output_dir / filename

    def _start_completion_waiter(
        self,
        run_tag: str,
        env: Dict[str, str],
        setup_file: Optional[str],
        topic: str,
    ) -> ProcessGroup:
        node_name = f"completion_waiter_{self._safe_name(run_tag)}"
        cmd = (
            "rosrun payload_mpc_controller wait_for_completion.py "
            f"--topic {shlex.quote(topic)} --node-name {shlex.quote(node_name)}"
        )
        log_handle = self._open_log(f"waiter_{self._safe_name(run_tag)}")
        popen = subprocess.Popen(
            self._wrap_command(cmd, setup_file),
            cwd=str(self.workspace_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        return ProcessGroup(popen, log_handle)

    def _wait_for_completion(
        self,
        proc: Optional[ProcessGroup],
        timeout: float,
        completion_listener: Optional[ProcessGroup] = None,
    ) -> bool:
        if proc is None:
            return False
        triggered = False
        deadline = time.time() + timeout if timeout > 0 else None
        while True:
            code = proc.poll()
            if code is not None:
                proc.wait()
                break
            if completion_listener:
                listener = completion_listener
                completion_code = listener.poll()
                if completion_code is not None:
                    try:
                        listener.wait()
                    except subprocess.TimeoutExpired:
                        pass
                    if completion_code == 0:
                        triggered = True
                        completion_listener = None
                        proc.terminate()
                        continue
                    completion_listener = None
            if deadline is not None and time.time() >= deadline:
                raise RuntimeError("Process exceeded timeout")
            time.sleep(0.5)
        return triggered

    def _validate_output(self, output_root: Path, min_samples: int) -> Dict[str, object]:
        notes: List[str] = []
        pinn_csv = self._pick_latest_csv(output_root / "pinn")
        force_csv = self._pick_latest_csv(output_root / "force")
        analytic_csv = self._pick_latest_csv(output_root / "analytic")
        result = {"ok": True, "notes": notes}
        if not pinn_csv:
            notes.append("Missing PINN CSV")
            result["ok"] = False
        else:
            samples = self._count_samples(pinn_csv)
            missing_fields = self._missing_fields(pinn_csv)
            result["samples"] = samples
            result["pinn_csv"] = pinn_csv
            if samples < min_samples:
                notes.append(f"Only {samples} samples (<{min_samples})")
                result["ok"] = False
            if missing_fields:
                notes.append(f"Missing fields: {', '.join(sorted(missing_fields))}")
                result["ok"] = False
        if force_csv:
            result["force_csv"] = force_csv
        else:
            notes.append("Missing force CSV")
            result["ok"] = False
        # Analytic CSV is helpful but optional; do not fail run if absent.
        if analytic_csv:
            result["analytic_csv"] = analytic_csv
        else:
            notes.append("Missing analytic CSV (optional)")
        return result

    def _pick_latest_csv(self, directory: Path) -> Optional[Path]:
        csvs = sorted(directory.glob("*.csv"))
        return csvs[-1] if csvs else None

    def _count_samples(self, csv_path: Path) -> int:
        with csv_path.open() as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _ in reader)

    def _missing_fields(self, csv_path: Path) -> List[str]:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
        return [field for field in EXPECTED_PINN_FIELDS if field not in header]

    def _run_seed(self, attempt: int) -> str:
        base = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        suffix = random.randint(1000, 9999)
        return f"{base}_{attempt}_{suffix}"

    def _wrap_command(self, cmd: str, setup_file: Optional[str]) -> List[str]:
        if setup_file:
            return ["bash", "-lc", f"source {setup_file} >/dev/null 2>&1 && {cmd}"]
        return ["bash", "-lc", cmd]

    def _open_log(self, tag: str):
        log_path = self.stdout_dir / f"{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path.open("w")

    def _safe_name(self, value: str) -> str:
        """Return a ROS-safe base name derived from value."""
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "node"
        if not cleaned[0].isalpha():
            cleaned = f"node_{cleaned}"
        return cleaned

    def _wait_for_master(self, uri: str, timeout: float = 10.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
            if self._master_up(uri):
                return
            time.sleep(0.5)
        raise RuntimeError(f"ROS master {uri} did not come up in time")

    def _master_up(self, uri: str) -> bool:
        try:
            subprocess.check_call(
                self._wrap_command("rosnode list >/dev/null 2>&1", self.setup_file),
                env={**os.environ, "ROS_MASTER_URI": uri},
                cwd=str(self.workspace_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _scenario_timeout(self, scenario: ScenarioSpec) -> float:
        if not getattr(self.args, "auto_timeout", False):
            return float(scenario.metadata.get("timeout", self.args.timeout))
        period = self._trajectory_period(scenario.trajectory)
        if period <= 0.0:
            return float(scenario.metadata.get("timeout", self.args.timeout))
        cycles = float(scenario.metadata.get("cycles_to_report", self.args.cycles_target))
        margin = float(getattr(self.args, "timeout_margin", DEFAULT_TIMEOUT_MARGIN))
        return cycles * period + margin + float(self.args.warmup)

    @staticmethod
    def _trajectory_period(trajectory: str) -> float:
        omega_map = {"circle": 0.5, "figure_eight": 0.3, "helix": 0.4}
        omega = omega_map.get(trajectory, 0.0)
        if abs(omega) < 1e-6:
            return 0.0
        return abs(2.0 * math.pi / omega)


class ProgressMonitor:
    """Periodic stderr progress reporter for long batch runs."""

    def __init__(self, total: int, interval: float = 1.0) -> None:
        self.total = max(0, total)
        self.interval = interval
        self._completed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start_time = time.time()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.total == 0 or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def increment(self) -> None:
        with self._lock:
            self._completed += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._thread = None
        if self.total > 0:
            self._print_line(final=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._print_line()

    def _print_line(self, final: bool = False) -> None:
        done, total = self._snapshot()
        percent = (done / total * 100.0) if total else 0.0
        elapsed = time.time() - self._start_time
        eta = self._eta(done, elapsed, total)
        msg = (
            f"\r[batch] Progress: {done}/{total} ({percent:5.1f}%) "
            f"elapsed={int(elapsed):4d}s eta={eta:>5}"
        )
        end = "\n" if final else ""
        print(msg, file=sys.stderr, end=end, flush=True)

    def _snapshot(self) -> Tuple[int, int]:
        with self._lock:
            return min(self._completed, self.total), self.total

    @staticmethod
    def _eta(done: int, elapsed: float, total: int) -> str:
        remaining = max(0, total - done)
        if done == 0 or elapsed <= 0.0:
            return "--"
        rate = done / elapsed
        if rate <= 0.0:
            return "--"
        seconds = int(remaining / rate)
        return f"{seconds}s"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["lite", "full"], default="lite", help="Scenario profile")
    parser.add_argument("--scenarios", nargs="*", help="Explicit scenario names")
    parser.add_argument("--include", nargs="*", default=[], help="Substring filters to include")
    parser.add_argument("--exclude", nargs="*", default=[], help="Substring filters to drop")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--timeout", type=float, default=DEFAULT_SCENARIO_TIMEOUT)
    parser.add_argument(
        "--recorder-grace",
        type=float,
        default=DEFAULT_RECORDER_GRACE,
        help="Seconds to wait for recorder shutdown",
    )
    parser.add_argument("--recorder-max-samples", type=int, default=DEFAULT_RECORDER_MAX_SAMPLES)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--setup-file", type=str, default=None)
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume (enabled by default)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=str, help="Optional JSON/YAML config file")
    parser.add_argument("--min-index", type=int, default=0, help="Start from this scenario index")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of scenarios to run")
    parser.add_argument("--parallel", type=int, default=None, help="Deprecated alias for --max-concurrency")
    parser.add_argument("--trajectories", nargs="*", help="Filter trajectories (circle, figure_eight)")
    parser.add_argument("--winds", nargs="*", help="Filter wind presets (e.g., w4_strong_high)")
    parser.add_argument("--report-format", choices=["json", "csv"], default="json")
    parser.add_argument("--dry-output", action="store_true", help="Dry-run plus summary report only")
    parser.add_argument("--resume-failed", action="store_true", help="Only rerun failed scenarios from checkpoint")
    parser.add_argument("--status-only", action="store_true", help="List checkpoint status and exit")
    parser.add_argument("--no-progress", action="store_true", help="Disable periodic progress indicator")
    parser.add_argument(
        "--auto-timeout",
        action="store_true",
        default=True,
        help="Automatically stop each scenario after target cycles (overrides --timeout).",
    )
    parser.add_argument(
        "--cycles-target",
        type=float,
        default=DEFAULT_CYCLES_TARGET,
        help="Target trajectory cycles when auto-timeout is enabled.",
    )
    parser.add_argument(
        "--timeout-margin",
        type=float,
        default=DEFAULT_TIMEOUT_MARGIN,
        help="Extra seconds after cycles to allow recorder flush when auto-timeout is enabled.",
    )
    parser.add_argument(
        "--completion-topic",
        type=str,
        default=DEFAULT_COMPLETION_TOPIC,
        help="Topic (std_msgs/Empty) that signals a run has completed.",
    )
    args = parser.parse_args(argv)
    if args.parallel is not None and args.max_concurrency == 1:
        args.max_concurrency = args.parallel
    args.resume = not args.no_resume
    args.progress = not args.no_progress
    return args


def load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {path} not found")
    text = config_path.read_text()
    if path.endswith(('.yaml', '.yml')):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pylint: disable=import-outside-toplevel
            raise RuntimeError("PyYAML not installed") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def apply_config_defaults(args: argparse.Namespace, config: Dict[str, object]) -> None:
    defaults = config.get("defaults")
    if isinstance(defaults, dict):
        baseline_map = {
            "min_samples": DEFAULT_MIN_SAMPLES,
            "warmup": DEFAULT_WARMUP,
            "timeout": DEFAULT_SCENARIO_TIMEOUT,
            "recorder_grace": DEFAULT_RECORDER_GRACE,
            "recorder_max_samples": DEFAULT_RECORDER_MAX_SAMPLES,
            "max_retries": DEFAULT_MAX_RETRIES,
            "log_dir": DEFAULT_LOG_DIR,
            "base_port": DEFAULT_BASE_PORT,
            "retry_backoff": DEFAULT_RETRY_BACKOFF,
        }
        for key, baseline in baseline_map.items():
            if key in defaults and getattr(args, key, baseline) == baseline:
                setattr(args, key, defaults[key])
        for key in ("checkpoint", "report"):
            if key in defaults:
                setattr(args, key, defaults[key])

    recorder_cfg = config.get("recorder")
    if isinstance(recorder_cfg, dict):
        if "max_samples" in recorder_cfg and args.recorder_max_samples == DEFAULT_RECORDER_MAX_SAMPLES:
            args.recorder_max_samples = recorder_cfg["max_samples"]
        if "warmup" in recorder_cfg:
            if args.warmup == DEFAULT_WARMUP:
                args.warmup = recorder_cfg["warmup"]



def apply_filters(
    candidates: List[ScenarioSpec],
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
    trajectories: Optional[Sequence[str]],
    winds: Optional[Sequence[str]],
) -> List[ScenarioSpec]:
    def match_filters(name: str, filters: Sequence[str]) -> bool:
        return any(f in name for f in filters)

    filtered = []
    for spec in candidates:
        if trajectories and spec.trajectory not in trajectories:
            continue
        if winds and all(filter not in spec.wind for filter in winds):
            continue
        if include_substrings and not match_filters(spec.name, include_substrings):
            continue
        if exclude_substrings and match_filters(spec.name, exclude_substrings):
            continue
        filtered.append(spec)
    return filtered


def scenario_list_from_config(config: Dict[str, object], registry: ScenarioRegistry) -> Optional[List[ScenarioSpec]]:
    scenarios = config.get("scenarios")
    if not scenarios:
        return None
    if not isinstance(scenarios, list):
        raise ValueError("'scenarios' must be a list")

    resolved: List[ScenarioSpec] = []
    for entry in scenarios:
        if isinstance(entry, str):
            resolved.extend(registry.resolve([entry]))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise ValueError("Scenario dict requires 'name'")
            spec = registry.resolve([name])[0]
            metadata_overrides = entry.get("metadata", {})
            if isinstance(metadata_overrides, dict):
                spec.metadata.update(metadata_overrides)
            for key, value in entry.items():
                if key not in {"name", "metadata"}:
                    spec.metadata[key] = value
            resolved.append(spec)
        else:
            raise ValueError("Scenario entries must be strings or dicts")
    return resolved


def write_report(path: Path, results: List[ScenarioResult], fmt: str = "json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = [res.to_dict() for res in results]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        import csv as csv_module

        with path.open("w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(
                [
                    "scenario",
                    "status",
                    "samples",
                    "pinn_csv",
                    "force_csv",
                    "analytic_csv",
                    "attempts",
                    "notes",
                    "started_at",
                    "finished_at",
                ]
            )
            for res in results:
                writer.writerow(
                    [
                        res.scenario.name,
                        res.status,
                        res.samples,
                        res.pinn_csv or "",
                        res.force_csv or "",
                        res.analytic_csv or "",
                        res.attempts,
                        " | ".join(res.notes),
                        res.started_at or "",
                        res.finished_at or "",
                    ]
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent
    workspace_root = repo_root.parent.parent
    payload_pkg = repo_root / "controller" / "payload_mpc_controller"

    registry = ScenarioRegistry()
    config = load_config(args.config)
    apply_config_defaults(args, config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = payload_pkg / "plots" / checkpoint_path
    checkpoint = CheckpointStore(checkpoint_path)

    if args.status_only:
        print(json.dumps(checkpoint._data, indent=2, ensure_ascii=False))
        return 0

    if args.resume_failed:
        failed = [name for name, data in checkpoint._data.items() if data.get("status") == "failed"]
        if not failed:
            print("No failed scenarios to rerun.")
            return 0
        base_list = registry.resolve(failed)
    else:
        config_scenarios = scenario_list_from_config(config, registry)
        if args.scenarios:
            base_list = registry.resolve(args.scenarios)
        elif config_scenarios:
            base_list = config_scenarios
        else:
            base_list = registry.profile(args.profile)

    base_list = apply_filters(
        base_list,
        args.include,
        args.exclude,
        args.trajectories,
        args.winds,
    )

    if args.min_index:
        base_list = base_list[args.min_index :]
    if args.limit is not None:
        base_list = base_list[: args.limit]

    if not base_list:
        print("No scenarios to run.")
        return 0

    if args.dry_run or args.dry_output:
        for idx, scenario in enumerate(base_list, start=1):
            print(f"[{idx}/{len(base_list)}] {scenario.name}")
        if args.dry_output:
            report_path = Path(args.report)
            if not report_path.is_absolute():
                report_path = payload_pkg / "plots" / report_path
            write_report(report_path, [], args.report_format)
        return 0

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = payload_pkg / "plots" / report_path

    progress = ProgressMonitor(len(base_list)) if args.progress else None
    if progress:
        progress.start()

    def make_skip_result(spec: ScenarioSpec, entry: Dict[str, object], reason: str) -> ScenarioResult:
        skip = ScenarioResult(spec, "skipped", samples=int(entry.get("samples", 0) or 0))
        skip.started_at = entry.get("started_at")
        skip.finished_at = entry.get("finished_at")
        for field_name in ("pinn_csv", "force_csv", "analytic_csv"):
            value = entry.get(field_name)
            if value:
                setattr(skip, field_name, Path(value))
        prior_notes = entry.get("notes")
        if isinstance(prior_notes, list):
            skip.notes.extend(prior_notes)
        elif prior_notes:
            skip.notes.append(str(prior_notes))
        skip.notes.append(reason)
        return skip

    skipped_results: List[ScenarioResult] = []
    if args.resume:
        pending: List[ScenarioSpec] = []
        for spec in base_list:
            entry = checkpoint.get(spec.name)
            if entry and entry.get("status") == "success":
                skipped_results.append(make_skip_result(spec, entry, "Already recorded (checkpoint)"))
            else:
                pending.append(spec)
        base_list = pending

    if not base_list:
        if skipped_results:
            write_report(report_path, skipped_results, args.report_format)
            print(f"All scenarios already completed earlier. Report → {report_path}")
            return 0
        print("No scenarios to run.")
        return 0

    runner = ScenarioRunner(args, workspace_root, payload_pkg, registry)
    terminator = GracefulTerminator()
    results: List[ScenarioResult] = list(skipped_results)
    total = len(base_list)

    def run_once(spec: ScenarioSpec) -> ScenarioResult:
        retry_budget = int(spec.metadata.get("max_retries", args.max_retries))
        backoff = float(spec.metadata.get("retry_backoff", args.retry_backoff))
        attempt = 1
        while attempt <= max(1, retry_budget):
            if terminator.requested():
                aborted = ScenarioResult(spec, "aborted")
                aborted.notes.append("Termination requested")
                return aborted
            checkpoint_entry = checkpoint.get(spec.name) if args.resume else None
            if checkpoint_entry and checkpoint_entry.get("status") == "success":
                return make_skip_result(spec, checkpoint_entry, "Already completed; resume enabled")
            result = runner.run(spec, attempt)
            result.attempts = attempt
            checkpoint.mark(result)
            if result.status == "success":
                return result
            attempt += 1
            time.sleep(max(0.0, backoff))
        failure = ScenarioResult(spec, "failed")
        failure.notes.append("Exceeded retry budget")
        return failure

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        future_map = {executor.submit(run_once, spec): spec for spec in base_list}
        completed = 0
        for future in concurrent.futures.as_completed(future_map):
            spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                result = ScenarioResult(spec, "failed", notes=[str(exc)])
                checkpoint.mark(result)
            else:
                if result.status not in {"skipped"}:
                    checkpoint.mark(result)
            results.append(result)
            completed += 1
            print(f"[{completed}/{total}] {spec.name} → {result.status} ({result.samples} samples)")
            if progress:
                progress.increment()
            if terminator.requested():
                break

    if progress:
        progress.stop()

    write_report(report_path, results, args.report_format)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    aborted = sum(1 for r in results if r.status == "aborted")
    print(
        f"Completed {success} success, {failed} failed, {skipped} skipped, {aborted} aborted. "
        f"Report → {report_path}"
    )
    return 0 if failed == 0 and aborted == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
