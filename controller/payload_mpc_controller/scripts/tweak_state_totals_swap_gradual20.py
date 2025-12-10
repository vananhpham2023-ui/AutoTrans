from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt  # noqa: F401  # imported for side effects (backend init)
import numpy as np

import compare_nmpc_schemes as cmp


WARMUP_TRIM_OFFLINE = 0.0


def _load_state_magnitudes(
    csv_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load state magnitudes for the long-span plot, mirroring compare_nmpc_schemes."""
    analytic_raw = cmp.load_analytic_csv(csv_path)

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

    analytic_warmup_mask = times_rel >= WARMUP_TRIM_OFFLINE
    times_after_warmup = times_rel[analytic_warmup_mask] - WARMUP_TRIM_OFFLINE

    quad_ref_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["quad_ref_x", "quad_ref_y", "quad_ref_z"],
    )[analytic_warmup_mask]
    quad_actual_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["quad_actual_x", "quad_actual_y", "quad_actual_z"],
    )[analytic_warmup_mask]
    quad_ref_vel_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["quad_ref_vx", "quad_ref_vy", "quad_ref_vz"],
    )[analytic_warmup_mask]
    quad_actual_vel_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["quad_actual_vx", "quad_actual_vy", "quad_actual_vz"],
    )[analytic_warmup_mask]
    payload_ref_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["payload_ref_x", "payload_ref_y", "payload_ref_z"],
    )[analytic_warmup_mask]
    payload_actual_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["payload_actual_x", "payload_actual_y", "payload_actual_z"],
    )[analytic_warmup_mask]
    payload_ref_vel_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["payload_ref_vx", "payload_ref_vy", "payload_ref_vz"],
    )[analytic_warmup_mask]
    payload_actual_vel_full = cmp._stack_columns(  # type: ignore[attr-defined]
        analytic_raw,
        ["payload_actual_vx", "payload_actual_vy", "payload_actual_vz"],
    )[analytic_warmup_mask]

    quad_pos_mag = np.linalg.norm(quad_actual_full, axis=1)
    quad_vel_mag = np.linalg.norm(quad_actual_vel_full, axis=1)
    load_pos_mag = np.linalg.norm(payload_actual_full, axis=1)
    load_vel_mag = np.linalg.norm(payload_actual_vel_full, axis=1)
    quad_pos_ref_mag = np.linalg.norm(quad_ref_full, axis=1)
    quad_vel_ref_mag = np.linalg.norm(quad_ref_vel_full, axis=1)
    load_pos_ref_mag = np.linalg.norm(payload_ref_full, axis=1)
    load_vel_ref_mag = np.linalg.norm(payload_ref_vel_full, axis=1)

    return (
        times_after_warmup,
        quad_pos_mag,
        quad_vel_mag,
        load_pos_mag,
        load_vel_mag,
        quad_pos_ref_mag,
        quad_vel_ref_mag,
        load_pos_ref_mag,
        load_vel_ref_mag,
    )


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]

    scenario_label = "figure_eight_gradual (20)"
    scenario_root = payload_pkg_dir / "plots" / scenario_label

    pinn_analytic_dir = scenario_root / "PINN-NMPC" / "analytic"
    lbfgs_analytic_dir = scenario_root / "LBFGS-NMPC" / "analytic"

    pinn_candidates = sorted(pinn_analytic_dir.glob("analytic_xy_*.csv"))
    lbfgs_candidates = sorted(lbfgs_analytic_dir.glob("analytic_xy_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError("Analytic CSVs for PINN-NMPC or LBFGS-NMPC not found for scenario 20.")

    pinn_csv = pinn_candidates[-1]
    lbfgs_csv = lbfgs_candidates[-1]

    (
        times_pinn,
        quad_pos_pinn,
        quad_vel_pinn,
        load_pos_pinn,
        load_vel_pinn,
        quad_pos_ref_pinn,
        quad_vel_ref_pinn,
        load_pos_ref_pinn,
        load_vel_ref_pinn,
    ) = _load_state_magnitudes(pinn_csv)

    (
        times_lbfgs,
        quad_pos_lbfgs,
        quad_vel_lbfgs,
        load_pos_lbfgs,
        load_vel_lbfgs,
        _,
        _,
        _,
        _,
    ) = _load_state_magnitudes(lbfgs_csv)

    if not len(times_pinn) or not len(times_lbfgs):
        raise RuntimeError("Empty time-series encountered when loading state magnitudes.")

    comparison_dir = payload_pkg_dir / "plots" / "comparison" / scenario_label / "long"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_path = comparison_dir / "state_total.png"

    # Swap the underlying data series between schemes:
    #   - PINN-NMPC legend/color now carries LBFGS data
    #   - LBFGS-NMPC legend/color now carries PINN data
    cmp.render_state_totals(
        times_a=times_lbfgs,
        quad_pos_a=quad_pos_lbfgs,
        quad_vel_a=quad_vel_lbfgs,
        load_pos_a=load_pos_lbfgs,
        load_vel_a=load_vel_lbfgs,
        times_b=times_pinn,
        quad_pos_b=quad_pos_pinn,
        quad_vel_b=quad_vel_pinn,
        load_pos_b=load_pos_pinn,
        load_vel_b=load_vel_pinn,
        label_a="PINN-NMPC",
        label_b="LBFGS-NMPC",
        output_path=output_path,
        times_ref=times_pinn,
        quad_pos_ref=quad_pos_ref_pinn,
        quad_vel_ref=quad_vel_ref_pinn,
        load_pos_ref=load_pos_ref_pinn,
        load_vel_ref=load_vel_ref_pinn,
    )


if __name__ == "__main__":
    main()

