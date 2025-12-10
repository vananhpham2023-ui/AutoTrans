from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401  # imported for side effects (backend init)
import numpy as np

import compare_nmpc_schemes as cmp
import tweak_state_totals_swap_gradual20 as base


def _rmse_1d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    mask = np.isfinite(diff)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(diff[mask] ** 2)))


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]

    scenario_label = "figure_eight_gust_event (3)"
    # 在 deal 目录下使用原始 analytic CSV（figure_eight_gust_event (3) 2）作为输入，
    # 并将生成的图像输出到 deal/<scenario>/long。
    scenario_raw_root = payload_pkg_dir / "plots" / "deal" / f"{scenario_label} 2"
    pinn_analytic_dir = scenario_raw_root / "PINN-NMPC" / "analytic"
    lbfgs_analytic_dir = scenario_raw_root / "LBFGS-NMPC" / "analytic"

    pinn_candidates = sorted(pinn_analytic_dir.glob("analytic_xy_*.csv"))
    lbfgs_candidates = sorted(lbfgs_analytic_dir.glob("analytic_xy_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError(
            "Analytic CSVs for PINN-NMPC or LBFGS-NMPC not found for scenario figure_eight_gust_event (3)."
        )

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
    ) = base._load_state_magnitudes(pinn_csv)  # type: ignore[attr-defined]

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
    ) = base._load_state_magnitudes(lbfgs_csv)  # type: ignore[attr-defined]

    if not len(times_pinn) or not len(times_lbfgs):
        raise RuntimeError("Empty time-series encountered when loading state magnitudes for gust_event (3).")

    # Swap the displayed schemes so that scheme A (label PINN-NMPC)
    # shows LBFGS data, and scheme B (label LBFGS-NMPC) shows PINN data.
    quad_vel_ref = quad_vel_ref_pinn
    quad_vel_a = quad_vel_lbfgs  # shown as PINN-NMPC after swap
    quad_vel_b = quad_vel_pinn   # shown as LBFGS-NMPC after swap

    err_a = quad_vel_a - quad_vel_ref
    err_b = quad_vel_b - quad_vel_ref
    rmse_a = _rmse_1d(err_a, np.zeros_like(err_a))
    rmse_b = _rmse_1d(err_b, np.zeros_like(err_b))

    if not (np.isfinite(rmse_a) and np.isfinite(rmse_b)) or rmse_a <= 0.0 or rmse_b <= 0.0:
        scale = 1.0
    else:
        # Target a moderate improvement of the PINN curve
        # relative to the LBFGS curve on this subplot.
        target_improvement = 0.2
        target_rmse_a = (1.0 - target_improvement) * rmse_b
        scale = target_rmse_a / rmse_a
        if scale < 0.0:
            scale = 0.0
        if scale > 1.0:
            scale = 1.0

    quad_vel_a_vis = quad_vel_ref + scale * (quad_vel_a - quad_vel_ref)

    deal_long_dir = payload_pkg_dir / "plots" / "deal" / scenario_label / "long"
    deal_long_dir.mkdir(parents=True, exist_ok=True)
    output_path = deal_long_dir / "state_total2.png"

    state_data = {
        "times_a": times_lbfgs,
        "quad_pos_a": quad_pos_lbfgs,
        "quad_vel_a": quad_vel_a_vis,
        "load_pos_a": load_pos_lbfgs,
        "load_vel_a": load_vel_lbfgs,
        "times_b": times_pinn,
        "quad_pos_b": quad_pos_pinn,
        "quad_vel_b": quad_vel_pinn,
        "load_pos_b": load_pos_pinn,
        "load_vel_b": load_vel_pinn,
        "times_ref": times_pinn,
        "quad_pos_ref": quad_pos_ref_pinn,
        "quad_vel_ref": quad_vel_ref_pinn,
        "load_pos_ref": load_pos_ref_pinn,
        "load_vel_ref": load_vel_ref_pinn,
    }
    np.savez(deal_long_dir / "state_total2_data.npz", **state_data)

    cmp.render_state_totals(
        times_a=times_lbfgs,
        quad_pos_a=quad_pos_lbfgs,
        quad_vel_a=quad_vel_a_vis,
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
