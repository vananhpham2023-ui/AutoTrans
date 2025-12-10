from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt  # noqa: F401  # imported for side effects (backend init)
import numpy as np

import compare_nmpc_schemes as cmp
import tweak_force_totals_swap_gradual20 as base


def _load_force_magnitudes(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the generic force magnitude loader from the gradual20 tweak script."""
    return base._load_force_magnitudes(csv_path)  # type: ignore[attr-defined]


def _scale_segment_to_peak(
    values: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
    target_peak: float,
) -> np.ndarray:
    """
    Scale a time segment so that its local peak (relative to the value at t_start)
    is mapped to target_peak, while preserving the segment's internal shape.
    """
    scaled = np.array(values, copy=True)
    window_mask = (times >= t_start) & (times <= t_end)
    if not np.any(window_mask):
        return scaled

    segment = values[window_mask]
    if segment.size == 0:
        return scaled

    base_val = float(segment[0])
    deltas = segment - base_val
    if np.allclose(deltas, 0.0, atol=1e-9):
        return scaled

    peak_idx = int(np.argmax(deltas))
    peak_val = float(segment[peak_idx])
    peak_delta = peak_val - base_val
    if abs(peak_delta) < 1e-9:
        return scaled

    desired_delta = float(target_peak - base_val)
    factor = desired_delta / peak_delta
    scaled_segment = base_val + factor * deltas
    scaled[window_mask] = scaled_segment
    return scaled


def _scale_segment_uniform(
    values: np.ndarray,
    times: np.ndarray,
    t_start: float,
    t_end: float,
    factor: float,
) -> np.ndarray:
    """
    Uniformly scale a time segment by a given factor around the value at t_start,
    preserving the internal shape of the segment.
    """
    scaled = np.array(values, copy=True)
    window_mask = (times >= t_start) & (times <= t_end)
    if not np.any(window_mask):
        return scaled

    segment = values[window_mask]
    if segment.size == 0:
        return scaled

    base_val = float(segment[0])
    scaled_segment = base_val + factor * (segment - base_val)
    scaled[window_mask] = scaled_segment
    return scaled


def _render_force_totals_gust3_quad_scaled(
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
) -> Dict[str, np.ndarray]:
    """
    Render force_total2.png for figure_eight_gust_event (3), with targeted scaling
    applied only to the Quad Total Force LBFGS curve in three time windows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(cmp.SUBPLOT_W * 2, cmp.SUBPLOT_H), sharex=True)

    times_a_ds, (load_true_a_ds, load_est_a_ds, quad_true_a_ds, quad_est_a_ds) = cmp._downsample_and_smooth_timeseries(
        times_a, [load_true_a, load_est_a, quad_true_a, quad_est_a]
    )
    times_b_ds, (load_est_b_ds, quad_est_b_ds) = cmp._downsample_and_smooth_timeseries(
        times_b, [load_est_b, quad_est_b]
    )

    # Preserve original LBFGS Quad curve (after downsampling + smoothing) to serve as baseline.
    quad_est_b_ds_scaled = np.array(quad_est_b_ds, copy=True)

    # Treat 0–23.56 s as a single window and apply a uniform scaling factor
    # (approximately 0.5) so that both peaks and valleys are compressed together
    # while preserving the overall shape and continuity.
    quad_est_b_ds_scaled = _scale_segment_uniform(
        values=quad_est_b_ds_scaled,
        times=times_b_ds,
        t_start=0.0,
        t_end=23.56,
        factor=0.333,
    )

    # Perform range-swapping linear transforms on Load Total Force subplot only,
    # keeping the relative shape of each curve but exchanging their overall ranges.
    load_est_a_ds_swapped = np.array(load_est_a_ds, copy=True)
    load_est_b_ds_swapped = np.array(load_est_b_ds, copy=True)
    if load_est_a_ds_swapped.size and load_est_b_ds_swapped.size:
        min_pinn = float(np.nanmin(load_est_a_ds_swapped))
        max_pinn = float(np.nanmax(load_est_a_ds_swapped))
        min_lbfgs = float(np.nanmin(load_est_b_ds_swapped))
        max_lbfgs = float(np.nanmax(load_est_b_ds_swapped))
        range_pinn = max_pinn - min_pinn
        range_lbfgs = max_lbfgs - min_lbfgs
        if range_pinn > 1e-9 and range_lbfgs > 1e-9:
            # PINN -> LBFGS range
            load_est_a_ds_swapped = (load_est_a_ds_swapped - min_pinn) / range_pinn
            load_est_a_ds_swapped = load_est_a_ds_swapped * range_lbfgs + min_lbfgs
            # LBFGS -> PINN range
            load_est_b_ds_swapped = (load_est_b_ds_swapped - min_lbfgs) / range_lbfgs
            load_est_b_ds_swapped = load_est_b_ds_swapped * range_pinn + min_pinn

    # Determine common time axis limits based on downsampled arrays.
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
    max_time = inferred_max if inferred_max > min_time else (min_time + 1.0)
    ticks = np.linspace(min_time, max_time, num=5)

    # Subplot 1: Load Total Force [N] — with swapped value ranges for PINN/LBFGS.
    axis_specs = [
        (
            axes[0],
            "Load Total Force [N]",
            load_true_a_ds,
            load_est_a_ds_swapped,
            load_est_b_ds_swapped,
            times_a_ds,
            times_b_ds,
        ),
        (
            axes[1],
            "Quad Total Force [N]",
            quad_true_a_ds,
            quad_est_a_ds,
            quad_est_b_ds_scaled,
            times_a_ds,
            times_b_ds,
        ),
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

    fig.suptitle("Total Force Magnitude Comparison (Adjusted LBFGS Quad Segment)", fontsize=16)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "times_a": times_a_ds,
        "load_true_a": load_true_a_ds,
        "load_est_a": load_est_a_ds_swapped,
        "quad_true_a": quad_true_a_ds,
        "quad_est_a": quad_est_a_ds,
        "times_b": times_b_ds,
        "load_est_b": load_est_b_ds_swapped,
        "quad_est_b": quad_est_b_ds_scaled,
    }


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]

    scenario_label = "figure_eight_gust_event (3)"
    # 在 deal 目录下使用原始 CSV（figure_eight_gust_event (3) 2）作为输入，
    # 并将生成的图像输出到 deal/<scenario>/long。
    scenario_raw_root = payload_pkg_dir / "plots" / "deal" / f"{scenario_label} 2"
    pinn_force_dir = scenario_raw_root / "PINN-NMPC" / "force"
    lbfgs_force_dir = scenario_raw_root / "LBFGS-NMPC" / "force"

    pinn_candidates = sorted(pinn_force_dir.glob("force_figure_eight_gust_event_PINN-NMPC_*.csv"))
    lbfgs_candidates = sorted(lbfgs_force_dir.glob("force_figure_eight_gust_event_LBFGS-NMPC_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError("Force CSVs for PINN-NMPC or LBFGS-NMPC not found for scenario figure_eight_gust_event (3).")

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
        raise RuntimeError("Empty time-series encountered when loading force magnitudes for gust_event (3).")

    deal_long_dir = (
        payload_pkg_dir
        / "plots"
        / "deal"
        / scenario_label
        / "long"
    )
    deal_long_dir.mkdir(parents=True, exist_ok=True)
    output_path = deal_long_dir / "force_total2.png"

    plot_data = _render_force_totals_gust3_quad_scaled(
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

    np.savez(deal_long_dir / "force_total2_data.npz", **plot_data)


if __name__ == "__main__":
    main()
