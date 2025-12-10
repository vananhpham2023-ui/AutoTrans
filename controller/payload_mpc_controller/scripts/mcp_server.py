#!/usr/bin/env python3
"""
MCP server that exposes the experiment_manager core capabilities as tools.

Provided tools:
1) get_lbfgs_config() -> return current lbfgs_nmpc.yaml as dict.
2) set_lbfgs_params(params: dict, persist: bool = True) -> write params (dotted keys ok).
3) run_experiment(...) -> run one compare_nmpc_schemes experiment and return the record
   produced by experiment_manager (also optionally append to JSONL log).

Prerequisites:
- pip install mcp (or ensure mcp.server.fastmcp is importable).
- ROS workspace built and sourced (this server will source devel/setup.bash per command).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install the 'mcp' package (pip install mcp).") from exc

import experiment_manager as em


def _detect_workspace_root(script_path: Path) -> Path:
    for candidate in script_path.parents:
        if (candidate / "src").exists() and ((candidate / "devel").exists() or (candidate / "build").exists()):
            return candidate
    return script_path.parents[4]


def _default_paths() -> Dict[str, Path]:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    workspace_root = _detect_workspace_root(script_path)
    default_reuse = payload_pkg_dir / "plots" / "figure_eight_gust_event (2)"
    default_log = payload_pkg_dir / "experiments" / "lbfgs_vs_pinn_log.jsonl"
    return {
        "script_path": script_path,
        "payload_pkg_dir": payload_pkg_dir,
        "workspace_root": workspace_root,
        "default_reuse": default_reuse,
        "default_log": default_log,
        "lbfgs_config": payload_pkg_dir / "config" / "lbfgs_nmpc.yaml",
        "pinn_config": payload_pkg_dir / "config" / "pinn_nmpc.yaml",
        "scenario_config": payload_pkg_dir / "config" / "my_scenarios.json",
    }


server = FastMCP("autotrans_mpc")


@server.tool()
def get_lbfgs_config(ctx: Context) -> Dict[str, Any]:
    """
    Return the current lbfgs_nmpc.yaml content as a dict.
    """
    paths = _default_paths()
    return em.load_yaml(paths["lbfgs_config"])


@server.tool()
def set_lbfgs_params(ctx: Context, params: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
    """
    Apply parameter overrides to lbfgs_nmpc.yaml. Supports dotted keys.
    persist=True keeps the changes on disk.
    """
    paths = _default_paths()
    path = paths["lbfgs_config"]
    base = em.load_yaml(path)
    updated = em.apply_overrides(base, params)
    em.write_yaml(path, updated)
    if not persist:
        # Roll back to original if not persisting.
        em.write_yaml(path, base)
    return updated


def _run_experiment(
    overrides: Optional[Dict[str, Any]] = None,
    persist_edits: bool = False,
    stop_on_dynamic_error: bool = True,
    dynamic_error_shutdown_grace: float = 5.0,
    reuse_scheme_a_from: Optional[str] = None,
    short_window_start: float = 25.0,
    short_window_end: float = 40.0,
    completion_timeout: float = 150.0,
    setup_script: str = "devel/setup.bash",
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    paths = _default_paths()
    payload_pkg_dir = paths["payload_pkg_dir"]
    workspace_root = paths["workspace_root"]
    lbfgs_config_path = paths["lbfgs_config"]
    pinn_config_path = paths["pinn_config"]
    scenario_config_path = paths["scenario_config"]
    reuse_root = Path(reuse_scheme_a_from).resolve() if reuse_scheme_a_from else paths["default_reuse"]
    log_path = Path(log_file).resolve() if log_file else paths["default_log"]

    # Prepare overrides
    overrides = overrides or {}
    before_runs = set(em.list_dirs(payload_pkg_dir / "plots"))
    before_comparisons = set(em.list_dirs(payload_pkg_dir / "plots" / "comparison"))
    original_bytes = lbfgs_config_path.read_bytes()

    cmd = []
    exit_code = -1
    dynamic_error = False
    stopped_due_to_dynamic = False
    error_message: Optional[str] = None

    try:
        if overrides:
            updated_config = em.apply_overrides(em.load_yaml(lbfgs_config_path), overrides)
            em.write_yaml(lbfgs_config_path, updated_config)
        cmd = em.build_rosrun_command(
            em.argparse.Namespace(
                mpc_config_a=pinn_config_path,
                mpc_config_b=lbfgs_config_path,
                scenario_config=scenario_config_path,
                completion_timeout=completion_timeout,
                reuse_scheme_a_from=reuse_root,
                short_window_start=short_window_start,
                short_window_end=short_window_end,
                setup_script=setup_script,
            ),
            workspace_root=workspace_root,
        )
        exit_code, output_lines, dynamic_error, stopped_due_to_dynamic = em.run_compare_command(
            cmd,
            cwd=workspace_root,
            stop_on_dynamic_error=stop_on_dynamic_error,
            shutdown_grace=dynamic_error_shutdown_grace,
        )
    except Exception as exc:  # pragma: no cover
        error_message = str(exc)
    finally:
        if not persist_edits:
            lbfgs_config_path.write_bytes(original_bytes)

    after_runs = set(em.list_dirs(payload_pkg_dir / "plots"))
    after_comparisons = set(em.list_dirs(payload_pkg_dir / "plots" / "comparison"))
    new_runs = sorted(after_runs - before_runs)
    new_comparisons = sorted(after_comparisons - before_comparisons)

    lbfgs_metrics_candidates = []
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
    lbfgs_metrics_path = em.find_first_existing(lbfgs_metrics_candidates)
    lbfgs_metrics = em.metrics_from_file(lbfgs_metrics_path) if lbfgs_metrics_path else {}

    pinn_metrics_path = reuse_root / "PINN-NMPC" / "metrics_summary.txt"
    pinn_metrics = em.metrics_from_file(pinn_metrics_path) if pinn_metrics_path.exists() else {}

    flat_lbfgs = em.flatten_metrics(lbfgs_metrics)
    flat_pinn = em.flatten_metrics(pinn_metrics)
    all_inferior = None
    metric_failure = None
    metric_comparison = []
    if flat_lbfgs and flat_pinn:
        all_inferior, metric_failure, metric_comparison = em.compare_metrics(flat_pinn, flat_lbfgs)

    record = {
        "lbfgs_config_path": str(lbfgs_config_path),
        "pinn_config_path": str(pinn_config_path),
        "scenario_config": str(scenario_config_path),
        "reuse_scheme_a_from": str(reuse_root),
        "command": " ".join(cmd),
        "exit_code": exit_code,
        "dynamic_error": dynamic_error,
        "stopped_due_to_dynamic_error": stopped_due_to_dynamic,
        "stop_on_dynamic_error_setting": stop_on_dynamic_error,
        "dynamic_error_shutdown_grace": dynamic_error_shutdown_grace,
        "overrides": overrides,
        "persist_edits": persist_edits,
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
        "stdout_tail": output_lines[-200:] if 'output_lines' in locals() else [],
    }
    if log_path:
        em.append_jsonl(log_path, em.json.loads(em.json.dumps(record)))
        record["log_file"] = str(log_path)
    return record


@server.tool()
def run_experiment(
    ctx: Context,
    overrides: Optional[Dict[str, Any]] = None,
    persist_edits: bool = False,
    stop_on_dynamic_error: bool = True,
    dynamic_error_shutdown_grace: float = 5.0,
    reuse_scheme_a_from: Optional[str] = None,
    short_window_start: float = 25.0,
    short_window_end: float = 40.0,
    completion_timeout: float = 150.0,
    setup_script: str = "devel/setup.bash",
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one experiment (compare_nmpc_schemes) and return the full record.
    """
    return _run_experiment(
        overrides=overrides,
        persist_edits=persist_edits,
        stop_on_dynamic_error=stop_on_dynamic_error,
        dynamic_error_shutdown_grace=dynamic_error_shutdown_grace,
        reuse_scheme_a_from=reuse_scheme_a_from,
        short_window_start=short_window_start,
        short_window_end=short_window_end,
        completion_timeout=completion_timeout,
        setup_script=setup_script,
        log_file=log_file,
    )


def main() -> None:  # pragma: no cover
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
