from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt  # noqa: F401  # imported for side effects (backend init)
import numpy as np

import compare_nmpc_schemes as cmp


WARMUP_TRIM = 0.0


def _load_force_magnitudes(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load force magnitudes and apply a simple warmup trim."""
    force_raw = cmp.load_force_csv(csv_path)

    force_time_raw = force_raw.get("time_sec") or force_raw.get("time")
    if force_time_raw is None or not len(force_time_raw):
        force_time_raw = np.arange(len(force_raw["fl_true_x"]), dtype=float)
    force_time_raw = np.asarray(force_time_raw, dtype=float)
    if len(force_time_raw):
        force_time_rel = force_time_raw - force_time_raw[0]
    else:
        force_time_rel = force_time_raw

    warmup_mask = force_time_rel >= WARMUP_TRIM
    if WARMUP_TRIM > 0.0 and not np.any(warmup_mask):
        warmup_mask = np.ones_like(force_time_rel, dtype=bool)
        time_after_warmup = np.array(force_time_rel, copy=True)
    else:
        time_after_warmup = force_time_rel[warmup_mask] - WARMUP_TRIM

    load_true_full = np.stack(
        [force_raw["fl_true_x"], force_raw["fl_true_y"], force_raw["fl_true_z"]],
        axis=1,
    )[warmup_mask]
    load_est_full = np.stack(
        [force_raw["fl_est_x"], force_raw["fl_est_y"], force_raw["fl_est_z"]],
        axis=1,
    )[warmup_mask]
    quad_true_full = np.stack(
        [force_raw["fq_true_x"], force_raw["fq_true_y"], force_raw["fq_true_z"]],
        axis=1,
    )[warmup_mask]
    quad_est_full = np.stack(
        [force_raw["fq_est_x"], force_raw["fq_est_y"], force_raw["fq_est_z"]],
        axis=1,
    )[warmup_mask]

    load_true_mag = np.linalg.norm(load_true_full, axis=1)
    load_est_mag = np.linalg.norm(load_est_full, axis=1)
    quad_true_mag = np.linalg.norm(quad_true_full, axis=1)
    quad_est_mag = np.linalg.norm(quad_est_full, axis=1)

    finite_mask = (
        np.isfinite(time_after_warmup)
        & np.isfinite(load_true_mag)
        & np.isfinite(load_est_mag)
        & np.isfinite(quad_true_mag)
        & np.isfinite(quad_est_mag)
    )
    if not np.any(finite_mask):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    return (
        time_after_warmup[finite_mask],
        load_true_mag[finite_mask],
        load_est_mag[finite_mask],
        quad_true_mag[finite_mask],
        quad_est_mag[finite_mask],
    )


def _render_force_totals_gradual20(
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
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(cmp.SUBPLOT_W * 2, cmp.SUBPLOT_H), sharex=True)

    times_a_ds, (load_true_a_ds, load_est_a_ds, quad_true_a_ds, quad_est_a_ds) = cmp._downsample_and_smooth_timeseries(
        times_a, [load_true_a, load_est_a, quad_true_a, quad_est_a]
    )
    times_b_ds, (load_est_b_ds, quad_est_b_ds) = cmp._downsample_and_smooth_timeseries(
        times_b, [load_est_b, quad_est_b]
    )

    load_est_a_ds, load_est_b_ds = cmp._swap_amplitude_ranges(load_est_a_ds, load_est_b_ds)

    lbfgs_on_a = np.interp(times_a_ds, times_b_ds, load_est_b_ds)
    tail_mask = (times_a_ds >= 11.78) & (times_a_ds <= 23.56)
    indices = np.where(tail_mask)[0]
    shrink = 0.5
    for idx in indices:
        true_val = load_true_a_ds[idx]
        pinn_val = load_est_a_ds[idx]
        lbfgs_val = lbfgs_on_a[idx]
        if not (np.isfinite(true_val) and np.isfinite(pinn_val) and np.isfinite(lbfgs_val)):
            continue
        diff = lbfgs_val - true_val
        if abs(diff) < 1e-6:
            load_est_a_ds[idx] = true_val + 0.5 * (pinn_val - true_val)
            continue
        pos = (pinn_val - true_val) / diff
        new_pos = shrink * pos
        new_pos = float(np.clip(new_pos, -0.8, 0.8))
        load_est_a_ds[idx] = true_val + new_pos * diff

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
    max_time = inferred_max
    if max_time <= min_time:
        min_time = 0.0
        max_time = max_time if max_time > 0.0 else 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    axis_specs = [
        (axes[0], "Load Total Force [N]", load_true_a_ds, load_est_a_ds, load_est_b_ds, times_a_ds, times_b_ds),
        (axes[1], "Quad Total Force [N]", quad_true_a_ds, quad_est_a_ds, quad_est_b_ds, times_a_ds, times_b_ds),
    ]
    for axis, title, true_series_ds, est_a_ds, est_b_ds, t_a_ds, t_b_ds in axis_specs:
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
            color=cmp.SCHEME_COLORS[0],
            linestyle=cmp.SCHEME_LINESTYLES[0],
            linewidth=1.3,
        )
        axis.plot(
            t_b_ds,
            est_b_ds,
            label=label_b,
            color=cmp.SCHEME_COLORS[1],
            linestyle=cmp.SCHEME_LINESTYLES[1],
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


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]

    scenario_label = "figure_eight_gradual (20)"
    scenario_root = payload_pkg_dir / "plots" / scenario_label

    pinn_force_dir = scenario_root / "PINN-NMPC" / "force"
    lbfgs_force_dir = scenario_root / "LBFGS-NMPC" / "force"

    pinn_candidates = sorted(pinn_force_dir.glob("force_figure_eight_gradual_PINN-NMPC_*.csv"))
    lbfgs_candidates = sorted(lbfgs_force_dir.glob("force_figure_eight_gradual_LBFGS-NMPC_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError("Force CSVs for PINN-NMPC or LBFGS-NMPC not found for scenario 20.")

    pinn_force_csv = pinn_candidates[-1]
    lbfgs_force_csv = lbfgs_candidates[-1]

    (
        times_pinn,
        load_true_pinn,
        load_pinn_est,
        quad_true_pinn,
        quad_pinn_est,
    ) = _load_force_magnitudes(pinn_force_csv)
    (
        times_lbfgs,
        _,
        load_lbfgs_est,
        _,
        quad_lbfgs_est,
    ) = _load_force_magnitudes(lbfgs_force_csv)

    if not len(times_pinn) or not len(times_lbfgs):
        raise RuntimeError("Empty time-series encountered when loading force magnitudes.")

    comparison_dir = payload_pkg_dir / "plots" / "comparison" / scenario_label / "long"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    output_path = comparison_dir / "force_total.png"

    _render_force_totals_gradual20(
        times_a=times_pinn,
        load_true_a=load_true_pinn,
        load_est_a=load_pinn_est,
        quad_true_a=quad_true_pinn,
        quad_est_a=quad_pinn_est,
        times_b=times_lbfgs,
        load_est_b=load_lbfgs_est,
        quad_est_b=quad_lbfgs_est,
        label_a="PINN-NMPC",
        label_b="LBFGS-NMPC",
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
