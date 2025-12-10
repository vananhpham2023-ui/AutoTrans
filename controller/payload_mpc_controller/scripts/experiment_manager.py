#!/usr/bin/env python3
"""
Automate one experiment loop:
1) Apply parameter overrides to lbfgs_nmpc.yaml.
2) Run compare_nmpc_schemes.py via rosrun.
3) Parse metrics_summary.txt for LBFGS (and optionally PINN).
4) Detect "DYNAMIC ERROR: Length is too long" in rosrun output.
5) Append a JSONL record for traceability.

Threshold logic:
- LBFGS is marked inferior if (LBFGS - PINN) >= 3% of PINN (or abs epsilon).
- Experiment is marked failed if any metric gap >= 9% of PINN (or abs epsilon).
Both cases still continue the parameter search; flags are logged.

Dynamic error handling:
- When "DYNAMIC ERROR: Length is too long" appears, by default the script sends SIGINT
  (then SIGKILL after a short grace) to stop the run. Disable with --stop-on-dynamic-error.

This script does not make any intelligent parameter choices; it is a thin
automation shell so you can drive it manually or from an MCP workflow.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run this script.") from exc


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
DEFAULT_INFERIOR_REL = 0.03  # 3% gap => LBFGS is considered inferior (but continue search)
DEFAULT_FAILURE_REL = 0.09   # 9% gap => mark experiment as failed (still continue search)
DEFAULT_ABS_EPS = 1e-3


def parse_overrides(entries: Sequence[str]) -> Dict[str, Any]:
    """Parse key=value strings into a dict, with ast.literal_eval where possible."""
    updates: Dict[str, Any] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Override must be key=value, got: {raw}")
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            parsed: Any = ast.literal_eval(val)
        except Exception:
            parsed = val
        updates[key] = parsed
    return updates


def load_yaml(path: Path) -> MutableMapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def set_deep(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        node = cursor.get(part)
        if not isinstance(node, MutableMapping):
            node = {}
            cursor[part] = node
        cursor = node
    cursor[parts[-1]] = value


def apply_overrides(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    result = json.loads(json.dumps(base))  # deep copy via JSON
    for key, value in updates.items():
        set_deep(result, key, value)
    return result


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def run_compare_command(
    cmd: Sequence[str],
    cwd: Path,
    stream_output: bool = True,
    stop_on_dynamic_error: bool = True,
    shutdown_grace: float = 5.0,
) -> Tuple[int, List[str], bool, bool]:
    """Run the rosrun command, capture combined output, and detect dynamic errors.

    If stop_on_dynamic_error is True, send SIGINT (and SIGKILL after shutdown_grace)
    to the process group once the dynamic error string is observed.
    """
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    lines: List[str] = []
    dynamic_error = False
    stopped_due_to_dynamic = False
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        lines.append(raw_line)
        if stream_output:
            sys.stdout.write(raw_line)
        plain = strip_ansi(raw_line)
        if "DYNAMIC ERROR: Length is too long" in plain:
            dynamic_error = True
            if stop_on_dynamic_error and proc.poll() is None:
                stopped_due_to_dynamic = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=max(shutdown_grace, 0.5))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                break
    if proc.poll() is None:
        proc.wait()
    return proc.returncode, lines, dynamic_error, stopped_due_to_dynamic


def list_dirs(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


def metrics_from_file(path: Path) -> Dict[str, Any]:
    """
    Parse metrics_summary.txt into a nested dict:
    {section: {metric_name: value}}
    """
    if not path.exists():
        return {}
    section: Optional[str] = None
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            section_match = re.match(r"^\[(.+)\]$", line)
            if section_match:
                section = section_match.group(1)
                data.setdefault(section, {})
                continue
            if ":" not in line:
                continue
            key_raw, value_raw = line.split(":", 1)
            key = key_raw.strip()
            value_str = value_raw.strip()
            try:
                value: Any = float(value_str)
            except ValueError:
                value = value_str
            if section is None:
                data[key] = value
            else:
                if not isinstance(data.get(section), MutableMapping):
                    data[section] = {}
                data[section][key] = value
    return data


def flatten_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for section, payload in metrics.items():
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    flat[f"{section}.{key}"] = float(value)
        elif isinstance(payload, (int, float)):
            flat[str(section)] = float(payload)
    return flat


def compare_metrics(
    pinn: Mapping[str, float],
    lbfgs: Mapping[str, float],
    inferior_rel: float = DEFAULT_INFERIOR_REL,
    failure_rel: float = DEFAULT_FAILURE_REL,
    abs_eps: float = DEFAULT_ABS_EPS,
) -> Tuple[bool, bool, List[Mapping[str, Any]]]:
    results: List[Mapping[str, Any]] = []
    all_inferior = True
    failure_any = False
    for name, lbfgs_val in lbfgs.items():
        pinn_val = pinn.get(name)
        if pinn_val is None:
            continue
        diff = lbfgs_val - pinn_val
        base = abs(pinn_val) if abs(pinn_val) > 1e-9 else 1.0
        worse = diff > max(abs_eps, 0.0)  # only care if LBFGS is larger
        inferior = worse and diff >= max(abs_eps, inferior_rel * base)
        failure = worse and diff >= max(abs_eps, failure_rel * base)
        results.append(
            {
                "metric": name,
                "pinn": pinn_val,
                "lbfgs": lbfgs_val,
                "diff": diff,
                "lbfgs_inferior": inferior,
                "lbfgs_failure_gap": failure,
            }
        )
        if not inferior:
            all_inferior = False
        if failure:
            failure_any = True
    return all_inferior, failure_any, results


def find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for candidate in paths:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def build_rosrun_command(args: argparse.Namespace, workspace_root: Path) -> List[str]:
    setup_script = Path(args.setup_script)
    if not setup_script.is_absolute():
        setup_script = workspace_root / setup_script
    import shlex

    cmd = (
        f"source {shlex.quote(str(setup_script))} >/dev/null 2>&1 && "
        f"rosrun payload_mpc_controller compare_nmpc_schemes.py "
        f"--mpc-config-a {shlex.quote(str(Path(args.mpc_config_a).resolve()))} "
        f"--mpc-config-b {shlex.quote(str(Path(args.mpc_config_b).resolve()))} "
        f"--scenario-config {shlex.quote(str(Path(args.scenario_config).resolve()))} "
        f"--completion-timeout {args.completion_timeout} "
        f"--reuse-scheme-a-from {shlex.quote(str(args.reuse_scheme_a_from))} "
        f"--short-window-start {args.short_window_start} "
        f"--short-window-end {args.short_window_end}"
    )
    return ["bash", "-lc", cmd]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    # Locate catkin workspace root by looking for a parent that contains both src and devel/build.
    workspace_root = None
    for candidate in script_path.parents:
        if (candidate / "src").exists() and ((candidate / "devel").exists() or (candidate / "build").exists()):
            workspace_root = candidate
            break
    if workspace_root is None:
        workspace_root = script_path.parents[4]
    default_reuse = payload_pkg_dir / "plots" / "figure_eight_gust_event (2)"
    default_log = payload_pkg_dir / "experiments" / "lbfgs_vs_pinn_log.jsonl"
    parser = argparse.ArgumentParser(description="Automate LBFGS NMPC experiments.")
    parser.add_argument("--lbfgs-config", default=str(payload_pkg_dir / "config" / "lbfgs_nmpc.yaml"))
    parser.add_argument("--pinn-config", default=str(payload_pkg_dir / "config" / "pinn_nmpc.yaml"))
    parser.add_argument("--scenario-config", default=str(payload_pkg_dir / "config" / "my_scenarios.json"))
    parser.add_argument("--mpc-config-a", default=None, help="Optional override for PINN config (scheme A).")
    parser.add_argument("--mpc-config-b", default=None, help="Optional override for LBFGS config (scheme B).")
    parser.add_argument("--reuse-scheme-a-from", default=str(default_reuse))
    parser.add_argument("--short-window-start", type=float, default=25.0)
    parser.add_argument("--short-window-end", type=float, default=40.0)
    parser.add_argument("--completion-timeout", type=float, default=150.0)
    parser.add_argument("--setup-script", default="devel/setup.bash")
    parser.add_argument("--stop-on-dynamic-error", action="store_true", default=True, help="Stop the run when 'DYNAMIC ERROR: Length is too long' appears.")
    parser.add_argument("--no-stop-on-dynamic-error", action="store_false", dest="stop_on_dynamic_error", help="Do not stop the run when the dynamic error appears.")
    parser.add_argument("--dynamic-error-shutdown-grace", type=float, default=5.0, help="Grace period (s) after SIGINT before SIGKILL when stopping due to dynamic error.")
    parser.add_argument("--log-file", default=str(default_log))
    parser.add_argument("--persist-edits", action="store_true", help="Keep lbfgs_nmpc.yaml changes after the run.")
    parser.add_argument("--param", action="append", default=[], help="Override parameter, format key=value. Supports dotted paths.")
    parser.add_argument("--workspace-root", default=str(workspace_root))
    parser.add_argument("--pinn-metrics", default=None, help="Optional explicit path to PINN metrics_summary.txt.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    workspace_root = Path(args.workspace_root).resolve()

    lbfgs_config_path = Path(args.lbfgs_config).resolve()
    pinn_config_path = Path(args.pinn_config).resolve()
    mpc_config_a = Path(args.mpc_config_a).resolve() if args.mpc_config_a else pinn_config_path
    mpc_config_b = Path(args.mpc_config_b).resolve() if args.mpc_config_b else lbfgs_config_path

    scenario_config_path = Path(args.scenario_config).resolve()
    reuse_scheme_a_from = Path(args.reuse_scheme_a_from).resolve()
    log_path = Path(args.log_file).resolve()

    overrides = parse_overrides(args.param)
    original_bytes = lbfgs_config_path.read_bytes()
    updated_config = None

    before_runs = set(list_dirs(payload_pkg_dir / "plots"))
    before_comparisons = set(list_dirs(payload_pkg_dir / "plots" / "comparison"))

    cmd: List[str] = []
    exit_code = -1
    dynamic_error = False
    stopped_due_to_dynamic = False
    error_message: Optional[str] = None

    try:
        if overrides:
            base_yaml = load_yaml(lbfgs_config_path)
            updated_config = apply_overrides(base_yaml, overrides)
            write_yaml(lbfgs_config_path, updated_config)
            print(f"[info] Applied overrides to {lbfgs_config_path}")
        else:
            updated_config = load_yaml(lbfgs_config_path)

        cmd = build_rosrun_command(
            argparse.Namespace(
                mpc_config_a=mpc_config_a,
                mpc_config_b=mpc_config_b,
                scenario_config=scenario_config_path,
                completion_timeout=args.completion_timeout,
                reuse_scheme_a_from=reuse_scheme_a_from,
                short_window_start=args.short_window_start,
                short_window_end=args.short_window_end,
                setup_script=args.setup_script,
            ),
            workspace_root=workspace_root,
        )
        print(f"[info] Running command: {' '.join(cmd)}")
        exit_code, _output_lines, dynamic_error, stopped_due_to_dynamic = run_compare_command(
            cmd,
            cwd=workspace_root,
            stop_on_dynamic_error=args.stop_on_dynamic_error,
            shutdown_grace=args.dynamic_error_shutdown_grace,
        )
        print(f"[info] Command finished with code {exit_code} (dynamic_error={dynamic_error}, stopped={stopped_due_to_dynamic})")
        if dynamic_error and not stopped_due_to_dynamic and args.stop_on_dynamic_error:
            print("[warn] Dynamic error detected but process did not stop; check downstream nodes.")
        if dynamic_error and stopped_due_to_dynamic:
            print("[info] Run interrupted due to dynamic error.")
    except Exception as exc:
        error_message = str(exc)
        print(f"[error] {error_message}", file=sys.stderr)
    finally:
        if not args.persist_edits:
            lbfgs_config_path.write_bytes(original_bytes)
            print(f"[info] Restored original {lbfgs_config_path}")

    after_runs = set(list_dirs(payload_pkg_dir / "plots"))
    after_comparisons = set(list_dirs(payload_pkg_dir / "plots" / "comparison"))

    new_runs = sorted(after_runs - before_runs)
    new_comparisons = sorted(after_comparisons - before_comparisons)

    if error_message and exit_code == -1:
        exit_code = 1

    lbfgs_metrics_candidates: List[Path] = []
    for run_name in reversed(new_runs):
        if "LBFGS" in run_name.upper():
            lbfgs_metrics_candidates.append(payload_pkg_dir / "plots" / run_name / "metrics_summary.txt")
        lbfgs_metrics_candidates.append(
            payload_pkg_dir / "plots" / run_name / "LBFGS-NMPC" / "metrics_summary.txt"
        )
    for comp_name in reversed(new_comparisons):
        lbfgs_metrics_candidates.append(
            payload_pkg_dir / "plots" / "comparison" / comp_name / "LBFGS-NMPC" / "metrics_summary.txt"
        )
    lbfgs_metrics_path = find_first_existing(lbfgs_metrics_candidates)
    lbfgs_metrics = metrics_from_file(lbfgs_metrics_path) if lbfgs_metrics_path else {}

    pinn_metrics_path = None
    if args.pinn_metrics:
        pinn_metrics_path = Path(args.pinn_metrics).resolve()
    else:
        pinn_metrics_path = reuse_scheme_a_from / "PINN-NMPC" / "metrics_summary.txt"
    pinn_metrics = metrics_from_file(pinn_metrics_path) if pinn_metrics_path and pinn_metrics_path.exists() else {}

    flat_lbfgs = flatten_metrics(lbfgs_metrics)
    flat_pinn = flatten_metrics(pinn_metrics)
    all_inferior = None
    metric_failure = None
    metric_comparison: List[Mapping[str, Any]] = []
    if flat_lbfgs and flat_pinn:
        all_inferior, metric_failure, metric_comparison = compare_metrics(flat_pinn, flat_lbfgs)
    if metric_failure:
        print("[warn] At least one metric gap exceeded the 9% failure threshold.")

    record = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "lbfgs_config_path": str(lbfgs_config_path),
        "pinn_config_path": str(pinn_config_path),
        "scenario_config": str(scenario_config_path),
        "reuse_scheme_a_from": str(reuse_scheme_a_from),
        "command": " ".join(cmd),
        "exit_code": exit_code,
        "dynamic_error": dynamic_error,
        "stopped_due_to_dynamic_error": stopped_due_to_dynamic,
        "stop_on_dynamic_error_setting": args.stop_on_dynamic_error,
        "dynamic_error_shutdown_grace": args.dynamic_error_shutdown_grace,
        "overrides": overrides,
        "persist_edits": args.persist_edits,
        "new_run_dirs": new_runs,
        "new_comparison_dirs": new_comparisons,
        "lbfgs_metrics_path": str(lbfgs_metrics_path) if lbfgs_metrics_path else None,
        "pinn_metrics_path": str(pinn_metrics_path) if pinn_metrics_path else None,
        "lbfgs_metrics": lbfgs_metrics,
        "pinn_metrics": pinn_metrics,
        "flat_lbfgs_metrics": flat_lbfgs,
        "flat_pinn_metrics": flat_pinn,
        "metric_comparison": metric_comparison,
        "lbfgs_inferior_all_metrics": all_inferior,
        "metrics_failure_gap": metric_failure,
        "experiment_failure_due_to_metrics": bool(metric_failure) if metric_failure is not None else None,
        "error": error_message,
    }
    append_jsonl(log_path, record)
    print(f"[done] Logged run to {log_path}")

    if dynamic_error:
        print("[warn] Detected 'DYNAMIC ERROR: Length is too long' in rosrun output.")
    if lbfgs_metrics_path is None:
        print("[warn] Could not locate LBFGS metrics_summary.txt; check run output.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
