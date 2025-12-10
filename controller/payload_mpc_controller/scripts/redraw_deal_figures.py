import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from matplotlib import ticker

import compare_nmpc_schemes as cmp

STATE_DATA_KEYS: Sequence[str] = (
    "times_a",
    "quad_pos_a",
    "quad_vel_a",
    "load_pos_a",
    "load_vel_a",
    "times_b",
    "quad_pos_b",
    "quad_vel_b",
    "load_pos_b",
    "load_vel_b",
    "times_ref",
    "quad_pos_ref",
    "quad_vel_ref",
    "load_pos_ref",
    "load_vel_ref",
)

FORCE_DATA_KEYS: Sequence[str] = (
    "times_a",
    "load_true_a",
    "load_est_a",
    "quad_true_a",
    "quad_est_a",
    "times_b",
    "load_est_b",
    "quad_est_b",
)


def _nice_tick_step(value_range: float, target_ticks: int = 6) -> float:
    if not math.isfinite(value_range) or value_range <= 0.0:
        return 1.0
    raw_step = value_range / float(max(1, target_ticks))
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10**exponent)


def _uniform_axis_limits(
    axes: Sequence,
    data_groups: Sequence[Sequence[np.ndarray]],
    min_value: float = 0.0,
    target_ticks: int = 6,
    lower_override: Optional[float] = None,
    upper_override: Optional[float] = None,
    lower_margin: float = 0.0,
    upper_margin: float = 0.0,
) -> None:
    finite_min: Optional[float] = None
    finite_max: Optional[float] = None
    for group in data_groups:
        for series in group:
            arr = np.asarray(series, dtype=float)
            if arr.size == 0:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            min_val = float(finite.min())
            max_val = float(finite.max())
            finite_min = min_val if finite_min is None else min(finite_min, min_val)
            finite_max = max_val if finite_max is None else max(finite_max, max_val)
    if finite_max is None and upper_override is None:
        return
    base_min = lower_override if lower_override is not None else (
        min_value if finite_min is None else min(min_value, finite_min)
    )
    base_max = upper_override if upper_override is not None else (
        max(min_value, finite_max if finite_max is not None else min_value)
    )
    if base_max <= base_min:
        base_max = base_min + 1.0
    span = base_max - base_min
    lower = base_min - span * max(lower_margin, 0.0)
    upper = base_max + span * max(upper_margin, 0.0)
    full_span = max(upper - lower, span)
    step = _nice_tick_step(full_span, target_ticks=target_ticks)
    locator = ticker.MultipleLocator(step)
    for axis in axes:
        axis.set_ylim(lower, upper)
        axis.yaxis.set_major_locator(locator)


def _resolve_npz_path(scenario_long: Path, base_name: str, fallback_base: Optional[str]) -> Optional[Path]:
    candidates = [scenario_long / f"{base_name}_data.npz"]
    if fallback_base and fallback_base != base_name:
        candidates.append(scenario_long / f"{fallback_base}_data.npz")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_npz_arrays(npz_path: Path, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as loaded:
        missing = [key for key in keys if key not in loaded]
        if missing:
            raise KeyError(f"{npz_path} is missing keys: {', '.join(missing)}")
        return {key: np.asarray(loaded[key]) for key in keys}


def _load_state_magnitudes(analytic_csv: Path):
    data = cmp.load_analytic_csv(analytic_csv)

    times_raw = data.get("time_sec")
    if times_raw is None or not len(times_raw):
        times_raw = data.get("time")
    if times_raw is None or not len(times_raw):
        times_raw = np.arange(len(next(iter(data.values()))), dtype=float)
    times_raw = np.asarray(times_raw, dtype=float)
    times_rel = times_raw - times_raw[0] if len(times_raw) else times_raw

    def _stack(prefix: str):
        return np.vstack(
            [data[f"{prefix}_x"], data[f"{prefix}_y"], data[f"{prefix}_z"]]
        ).T

    quad_ref = _stack("quad_ref")
    quad_act = _stack("quad_actual")
    quad_ref_v = np.vstack(
        [data["quad_ref_vx"], data["quad_ref_vy"], data["quad_ref_vz"]]
    ).T
    quad_act_v = np.vstack(
        [data["quad_actual_vx"], data["quad_actual_vy"], data["quad_actual_vz"]]
    ).T
    load_ref = _stack("payload_ref")
    load_act = _stack("payload_actual")
    load_ref_v = np.vstack(
        [data["payload_ref_vx"], data["payload_ref_vy"], data["payload_ref_vz"]]
    ).T
    load_act_v = np.vstack(
        [data["payload_actual_vx"], data["payload_actual_vy"], data["payload_actual_vz"]]
    ).T

    quad_pos_mag = np.linalg.norm(quad_act, axis=1)
    quad_vel_mag = np.linalg.norm(quad_act_v, axis=1)
    load_pos_mag = np.linalg.norm(load_act, axis=1)
    load_vel_mag = np.linalg.norm(load_act_v, axis=1)

    quad_pos_ref_mag = np.linalg.norm(quad_ref, axis=1)
    quad_vel_ref_mag = np.linalg.norm(quad_ref_v, axis=1)
    load_pos_ref_mag = np.linalg.norm(load_ref, axis=1)
    load_vel_ref_mag = np.linalg.norm(load_ref_v, axis=1)

    return (
        times_rel,
        quad_pos_mag,
        quad_vel_mag,
        load_pos_mag,
        load_vel_mag,
        quad_pos_ref_mag,
        quad_vel_ref_mag,
        load_pos_ref_mag,
        load_vel_ref_mag,
    )


def _load_force_magnitudes(force_csv: Path):
    force_raw = cmp.load_force_csv(force_csv)

    t_raw = force_raw.get("time_sec")
    if t_raw is None or not len(t_raw):
        t_raw = force_raw.get("time")
    if t_raw is None or not len(t_raw):
        t_raw = np.arange(len(force_raw["fl_true_x"]), dtype=float)
    t_raw = np.asarray(t_raw, dtype=float)
    t_rel = t_raw - t_raw[0] if len(t_raw) else t_raw

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

    load_true_mag = np.linalg.norm(load_true, axis=1)
    load_est_mag = np.linalg.norm(load_est, axis=1)
    quad_true_mag = np.linalg.norm(quad_true, axis=1)
    quad_est_mag = np.linalg.norm(quad_est, axis=1)

    return t_rel, load_true_mag, load_est_mag, quad_true_mag, quad_est_mag


def _load_state_plot_data(scenario_orig: Path, scenario_long: Path, base_name: str) -> Dict[str, np.ndarray]:
    fallback_base = f"{base_name[:-1]}2" if base_name.endswith("3") else None
    data_npz = _resolve_npz_path(scenario_long, base_name, fallback_base)
    if data_npz is not None:
        return _load_npz_arrays(data_npz, STATE_DATA_KEYS)

    pinn_dir = scenario_orig / "PINN-NMPC" / "analytic"
    lbfgs_dir = scenario_orig / "LBFGS-NMPC" / "analytic"
    pinn_candidates = sorted(pinn_dir.glob("analytic_xy_*.csv"))
    lbfgs_candidates = sorted(lbfgs_dir.glob("analytic_xy_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError(f"Analytic CSVs not found under {scenario_orig}")

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

    return {
        "times_a": times_pinn,
        "quad_pos_a": quad_pos_pinn,
        "quad_vel_a": quad_vel_pinn,
        "load_pos_a": load_pos_pinn,
        "load_vel_a": load_vel_pinn,
        "times_b": times_lbfgs,
        "quad_pos_b": quad_pos_lbfgs,
        "quad_vel_b": quad_vel_lbfgs,
        "load_pos_b": load_pos_lbfgs,
        "load_vel_b": load_vel_lbfgs,
        "times_ref": times_pinn,
        "quad_pos_ref": quad_pos_ref_pinn,
        "quad_vel_ref": quad_vel_ref_pinn,
        "load_pos_ref": load_pos_ref_pinn,
        "load_vel_ref": load_vel_ref_pinn,
    }


def _load_force_plot_data(scenario_orig: Path, scenario_long: Path, base_name: str) -> Dict[str, np.ndarray]:
    fallback_base = f"{base_name[:-1]}2" if base_name.endswith("3") else None
    data_npz = _resolve_npz_path(scenario_long, base_name, fallback_base)
    if data_npz is not None:
        return _load_npz_arrays(data_npz, FORCE_DATA_KEYS)

    pinn_force_dir = scenario_orig / "PINN-NMPC" / "force"
    lbfgs_force_dir = scenario_orig / "LBFGS-NMPC" / "force"
    pinn_candidates = sorted(pinn_force_dir.glob("force_*.csv"))
    lbfgs_candidates = sorted(lbfgs_force_dir.glob("force_*.csv"))
    if not pinn_candidates or not lbfgs_candidates:
        raise FileNotFoundError(f"Force CSVs not found under {scenario_orig}")
    pinn_csv = pinn_candidates[-1]
    lbfgs_csv = lbfgs_candidates[-1]

    (
        times_pinn,
        load_true_pinn,
        load_est_pinn,
        quad_true_pinn,
        quad_est_pinn,
    ) = _load_force_magnitudes(pinn_csv)
    (
        times_lbfgs,
        _,
        load_est_lbfgs,
        _,
        quad_est_lbfgs,
    ) = _load_force_magnitudes(lbfgs_csv)

    return {
        "times_a": times_pinn,
        "load_true_a": load_true_pinn,
        "load_est_a": load_est_pinn,
        "quad_true_a": quad_true_pinn,
        "quad_est_a": quad_est_pinn,
        "times_b": times_lbfgs,
        "load_est_b": load_est_lbfgs,
        "quad_est_b": quad_est_lbfgs,
    }


def _redraw_state_total_deal(scenario_label: str, base_name: str):
    """重画 deal 目录下的 state_totalX.png + 四幅子图（使用已优化数据）。"""
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    deal_root = payload_pkg_dir / "plots" / "deal"

    scenario_orig = deal_root / f"{scenario_label} 2"
    scenario_long = deal_root / scenario_label / "long"

    state_data = _load_state_plot_data(scenario_orig, scenario_long, base_name)
    times_a = np.asarray(state_data["times_a"], dtype=float)
    times_b = np.asarray(state_data["times_b"], dtype=float)
    times_ref = np.asarray(state_data["times_ref"], dtype=float)

    def _safe_first(arr: np.ndarray) -> Optional[float]:
        return float(arr[0]) if arr.size else None

    def _safe_last(arr: np.ndarray) -> Optional[float]:
        return float(arr[-1]) if arr.size else None

    min_candidates = [
        val for val in (_safe_first(times_ref), _safe_first(times_a), _safe_first(times_b)) if val is not None
    ]
    max_candidates = [
        val for val in (_safe_last(times_ref), _safe_last(times_a), _safe_last(times_b)) if val is not None
    ]
    min_time = min(min_candidates) if min_candidates else 0.0
    max_time = max(max_candidates) if max_candidates else (min_time + 1.0)
    if max_time <= min_time:
        max_time = min_time + 1.0
    ticks = np.linspace(min_time, max_time, num=5)

    fig, axes = plt.subplots(2, 2, figsize=(cmp.SUBPLOT_W * 2, cmp.SUBPLOT_H * 2), sharex=True)
    axes = axes.ravel()

    def _plot_subplot(ax, y_ref, y_a, y_b, ylabel: str):
        if times_ref.size and y_ref.size:
            ax.plot(times_ref, y_ref, label="Reference", color="#000000", linewidth=1.4)
        ax.plot(times_a, y_a, label="PINN-NMPC", color=cmp.SCHEME_COLORS[0], linewidth=1.3)
        ax.plot(times_b, y_b, label="LBFGS-NMPC", color=cmp.SCHEME_COLORS[1], linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_xlim(min_time, max_time)
        ax.set_xticks(ticks)

    _plot_subplot(
        axes[0],
        state_data["quad_pos_ref"],
        state_data["quad_pos_a"],
        state_data["quad_pos_b"],
        "Quad Position [m]",
    )
    _plot_subplot(
        axes[1],
        state_data["load_pos_ref"],
        state_data["load_pos_a"],
        state_data["load_pos_b"],
        "Load Position [m]",
    )
    _plot_subplot(
        axes[2],
        state_data["quad_vel_ref"],
        state_data["quad_vel_a"],
        state_data["quad_vel_b"],
        "Quad Velocity [m/s]",
    )
    _plot_subplot(
        axes[3],
        state_data["load_vel_ref"],
        state_data["load_vel_a"],
        state_data["load_vel_b"],
        "Load Velocity [m/s]",
    )

    for ax in axes[:2]:
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("Time [s]")

    # 在主 state_total 图中为四个子图添加 (a)–(d) 标注。
    sub_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes, sub_labels):
        ax.text(
            0.5,
            -0.17,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    axes[2].set_xlabel("Time [s]")
    axes[3].set_xlabel("Time [s]")

    if base_name.endswith("3"):
        pos_groups = [
            (state_data["quad_pos_ref"], state_data["quad_pos_a"], state_data["quad_pos_b"]),
            (state_data["load_pos_ref"], state_data["load_pos_a"], state_data["load_pos_b"]),
        ]
        vel_groups = [
            (state_data["quad_vel_ref"], state_data["quad_vel_a"], state_data["quad_vel_b"]),
            (state_data["load_vel_ref"], state_data["load_vel_a"], state_data["load_vel_b"]),
        ]
        _uniform_axis_limits(
            axes[:2],
            pos_groups,
            lower_override=2.0,
            upper_override=4.0,
            lower_margin=0.03,
            upper_margin=0.03,
            target_ticks=5,
        )
        _uniform_axis_limits(
            axes[2:],
            vel_groups,
            lower_override=1.25,
            upper_override=3.5,
            lower_margin=0.03,
            upper_margin=0.03,
            target_ticks=5,
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.09, wspace=0.25, hspace=0.25)
    else:
        fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.97])
    scenario_long.mkdir(parents=True, exist_ok=True)
    (scenario_long / f"{base_name}.png").unlink(missing_ok=True)
    fig.savefig(scenario_long / f"{base_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 单独四张子图
    parts = [
        ("quad_pos_ref", "quad_pos_a", "quad_pos_b", "Quad Position [m]", "quad_pos"),
        ("load_pos_ref", "load_pos_a", "load_pos_b", "Load Position [m]", "load_pos"),
        ("quad_vel_ref", "quad_vel_a", "quad_vel_b", "Quad Velocity [m/s]", "quad_vel"),
        ("load_vel_ref", "load_vel_a", "load_vel_b", "Load Velocity [m/s]", "load_vel"),
    ]
    for ref_key, a_key, b_key, ylabel, name in parts:
        y_ref = state_data[ref_key]
        y_a = state_data[a_key]
        y_b = state_data[b_key]
        fig_s, ax_s = plt.subplots(1, 1, figsize=(cmp.SUBPLOT_W, cmp.SUBPLOT_H))
        _plot_subplot(ax_s, y_ref, y_a, y_b, ylabel)
        ax_s.set_xlabel("Time [s]")
        fig_s.tight_layout(rect=[0.05, 0.08, 0.97, 0.97])
        out = scenario_long / f"{base_name}_{name}.png"
        out.unlink(missing_ok=True)
        fig_s.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig_s)


def _redraw_force_total_deal(scenario_label: str, base_name: str):
    """重画 deal 目录下的 force_totalX.png + 两幅子图（使用已优化数据）。"""
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    deal_root = payload_pkg_dir / "plots" / "deal"

    scenario_orig = deal_root / f"{scenario_label} 2"
    scenario_long = deal_root / scenario_label / "long"

    force_data = _load_force_plot_data(scenario_orig, scenario_long, base_name)
    times_a = np.asarray(force_data["times_a"], dtype=float)
    times_b = np.asarray(force_data["times_b"], dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cmp.SUBPLOT_W * 2, cmp.SUBPLOT_H),
        sharex=True,
        sharey=base_name.endswith("3"),
    )

    def _plot_force(ax, t_true, y_true, t_a, y_a, t_b, y_b, ylabel: str):
        ax.plot(t_true, y_true, label="True", color="#000000", linewidth=1.4)
        ax.plot(t_a, y_a, label="PINN-NMPC", color=cmp.SCHEME_COLORS[0], linewidth=1.3)
        ax.plot(t_b, y_b, label="LBFGS-NMPC", color=cmp.SCHEME_COLORS[1], linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [s]")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="upper right")

    _plot_force(
        axes[0],
        times_a,
        force_data["load_true_a"],
        times_a,
        force_data["load_est_a"],
        times_b,
        force_data["load_est_b"],
        "Load Force [N]",
    )
    _plot_force(
        axes[1],
        times_a,
        force_data["quad_true_a"],
        times_a,
        force_data["quad_est_a"],
        times_b,
        force_data["quad_est_b"],
        "Quad Force [N]",
    )

    if base_name.endswith("3"):
        force_groups = [
            (force_data["load_true_a"], force_data["load_est_a"], force_data["load_est_b"]),
            (force_data["quad_true_a"], force_data["quad_est_a"], force_data["quad_est_b"]),
        ]
        _uniform_axis_limits(
            axes,
            force_groups,
            min_value=0.0,
            target_ticks=5,
            lower_margin=0.03,
            upper_margin=0.03,
        )
        for ax in axes:
            ax.tick_params(axis="y", labelleft=True)

    # 为 force_total 图中的左右子图添加 (a)、(b) 标注。
    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(
            0.5,
            -0.20,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
    scenario_long.mkdir(parents=True, exist_ok=True)
    (scenario_long / f"{base_name}.png").unlink(missing_ok=True)
    fig.savefig(scenario_long / f"{base_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 单独两张子图
    pairs = [
        ("load_true_a", "load_est_a", "load_est_b", "Load Force [N]", "load"),
        ("quad_true_a", "quad_est_a", "quad_est_b", "Quad Force [N]", "quad"),
    ]
    for true_key, pinn_key, lbfgs_key, ylabel, name in pairs:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(cmp.SUBPLOT_W, cmp.SUBPLOT_H))
        _plot_force(
            ax_s,
            times_a,
            force_data[true_key],
            times_a,
            force_data[pinn_key],
            times_b,
            force_data[lbfgs_key],
            ylabel,
        )
        fig_s.tight_layout(rect=[0.05, 0.08, 0.97, 0.97])
        out = scenario_long / f"{base_name}_{name}.png"
        out.unlink(missing_ok=True)
        fig_s.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig_s)


def main() -> None:
    scenarios = [
        "figure_eight_gradual (20)",
        "figure_eight_gust_event (3)",
    ]
    for label in scenarios:
        # 先对 2 版本做标题/拆子图处理
        _redraw_state_total_deal(label, "state_total2")
        _redraw_force_total_deal(label, "force_total2")
        # 再对 3 版本做同样处理（如果原始 PNG 已复制到该名称，将被覆盖为统一风格）
        _redraw_state_total_deal(label, "state_total3")
        _redraw_force_total_deal(label, "force_total3")


if __name__ == "__main__":
    main()
