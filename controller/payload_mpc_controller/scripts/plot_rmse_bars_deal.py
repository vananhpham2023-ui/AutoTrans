import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

import compare_nmpc_schemes as cmp

STATE_SUFFIXES: Tuple[str, ...] = ("_total", "_x", "_y", "_z")
FORCE_SUFFIXES: Tuple[str, ...] = ("_total", "_fx", "_fy", "_fz")
POSITION_LABELS = ("P", "Px", "Py", "Pz")
VELOCITY_LABELS = ("V", "Vx", "Vy", "Vz")
FORCE_LABELS = ("F", "Fx", "Fy", "Fz")
SCHEME_LABELS = ("LBFGS-NMPC", "PINN-NMPC")
AX_WIDTH = cmp.SUBPLOT_W * 0.9
AX_HEIGHT = cmp.SUBPLOT_H * 0.95


def _parse_state_summary(path: Path) -> Tuple[str, Dict[str, Tuple[float, float]]]:
    """Parse state RMSE TXT exported under deal/.../rmse_state_lbfgs_vs_pinn_optimized.txt."""
    with path.open("r", encoding="utf-8") as handle:
        raw_lines = handle.readlines()

    lines = [line.strip() for line in raw_lines]
    scenario = path.parent.parent.name
    for line in lines:
        if line.startswith("Scenario:"):
            scenario = line.split(":", 1)[1].strip()
            break

    metrics: Dict[str, Tuple[float, float]] = {}
    header_seen = False
    for line in lines:
        if not line:
            continue
        if line.startswith("---"):
            break
        if not header_seen:
            if line.startswith("Metric"):
                header_seen = True
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            lbfgs_val = float(parts[1])
            pinn_val = float(parts[2])
        except ValueError:
            continue
        metrics[name] = (lbfgs_val, pinn_val)

    return scenario, metrics


def _parse_force_summary(path: Path) -> Tuple[str, Dict[str, Tuple[float, float]]]:
    """Parse force RMSE TXT exported under deal/.../rmse_force_force_total2_axes_and_total.txt."""
    with path.open("r", encoding="utf-8") as handle:
        raw_lines = handle.readlines()

    lines = [line.rstrip("\n") for line in raw_lines]
    scenario = path.parent.parent.name
    for line in lines:
        if line.startswith("Scenario:"):
            scenario = line.split(":", 1)[1].strip()
            break

    metrics = _parse_force_balanced(lines)
    if not metrics:
        metrics = _parse_force_csv(lines)
    if not metrics:
        raise ValueError(f"Unable to parse force RMSE summary from {path}")
    return scenario, metrics


def _parse_force_balanced(lines: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("== Balanced RMSE"):
            start_idx = idx + 1
            break
    if start_idx is None:
        return {}

    scheme_indices = {"lbfgs-nmpc": 0, "pinn-nmpc": 1}
    current_scheme: Optional[int] = None
    raw_metrics: Dict[str, List[Optional[float]]] = {}

    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("==") or stripped.lower().startswith("notes"):
            break
        if stripped.startswith("[") and stripped.endswith("]"):
            scheme_name = stripped[1:-1].strip()
            normalized = scheme_name.split("(", 1)[0].strip().lower()
            current_scheme = scheme_indices.get(normalized)
            continue
        if ":" not in stripped or current_scheme is None:
            continue
        key, value_str = stripped.split(":", 1)
        key = key.strip()
        cleaned = value_str.strip()
        if not cleaned:
            continue
        try:
            value = float(cleaned.split()[0])
        except ValueError:
            continue
        slots = raw_metrics.setdefault(key, [None, None])
        slots[current_scheme] = value

    metrics: Dict[str, Tuple[float, float]] = {}
    for name, pair in raw_metrics.items():
        if pair[0] is None or pair[1] is None:
            continue
        metrics[name] = (float(pair[0]), float(pair[1]))
    return metrics


def _parse_force_csv(lines: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    metrics: Dict[str, Tuple[float, float]] = {}
    header_seen = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("---"):
            break
        if not header_seen:
            if line.startswith("Metric"):
                header_seen = True
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            lbfgs_val = float(parts[1])
            pinn_val = float(parts[2])
        except ValueError:
            continue
        metrics[name] = (lbfgs_val, pinn_val)
    return metrics


def _extract_component_values(
    metrics: Dict[str, Tuple[float, float]],
    prefix: str,
    suffixes: Sequence[str] = STATE_SUFFIXES,
) -> np.ndarray:
    values: List[Tuple[float, float]] = []
    for suffix in suffixes:
        key = f"{prefix}{suffix}"
        if key not in metrics:
            raise KeyError(f"Missing metric '{key}'")
        values.append(metrics[key])
    return np.asarray(values, dtype=float)


def _nice_tick_step(max_value: float, target_ticks: int = 6) -> float:
    if not math.isfinite(max_value) or max_value <= 0.0:
        return 1.0
    raw_step = max_value / float(max(1, target_ticks))
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1.0
    elif fraction <= 2:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10**exponent)


def _apply_uniform_ylim(
    axes: Sequence,
    data_arrays: Sequence[np.ndarray],
    min_value: float = 0.0,
) -> None:
    finite_max = None
    for arr in data_arrays:
        if arr.size == 0:
            continue
        val = float(np.nanmax(arr))
        if not math.isfinite(val):
            continue
        finite_max = val if finite_max is None else max(finite_max, val)
    if finite_max is None:
        return
    step = _nice_tick_step(finite_max)
    upper = max(min_value + step, math.ceil(finite_max / step) * step)
    locator = ticker.MultipleLocator(step)
    for axis in axes:
        axis.set_ylim(min_value, upper)
        axis.yaxis.set_major_locator(locator)


def _plot_grouped_bars(
    axis,
    values: np.ndarray,
    component_labels: Sequence[str],
    ylabel: str,
) -> None:
    axis.set_axisbelow(True)
    num_groups, num_schemes = values.shape
    group_spacing = 0.82
    x = np.arange(num_groups, dtype=float) * group_spacing
    bar_width = 0.25
    gap = 0.02
    total_span = num_schemes * bar_width + (num_schemes - 1) * gap

    bar_containers = []
    for scheme_idx in range(num_schemes):
        offset = -0.5 * total_span + scheme_idx * (bar_width + gap) + 0.5 * bar_width
        container = axis.bar(
            x + offset,
            values[:, scheme_idx],
            width=bar_width,
            color=cmp.SCHEME_COLORS[scheme_idx % len(cmp.SCHEME_COLORS)],
        )
        bar_containers.append(container)

    axis.set_xticks(x)
    axis.set_xticklabels(component_labels)
    axis.tick_params(axis="x", rotation=0)
    if num_groups > 0:
        span = group_spacing * (num_groups - 1)
        axis.set_xlim(-group_spacing * 0.6, span + group_spacing * 0.6)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    legend_handles = [container for container in bar_containers]
    axis.legend(
        legend_handles,
        SCHEME_LABELS,
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=0.9,
        handletextpad=0.4,
        borderaxespad=0.35,
    )


def _add_panel_label(axis, label: str) -> None:
    axis.text(
        0.5,
        -0.14,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=12,
    )


def render_state_rmse_bars(path: Path) -> None:
    _, metrics = _parse_state_summary(path)

    prefixes = [
        "quad_pos_rmse",
        "load_pos_rmse",
        "quad_vel_rmse",
        "load_vel_rmse",
    ]
    required_keys = [f"{prefix}{suffix}" for prefix in prefixes for suffix in STATE_SUFFIXES]
    for key in required_keys:
        if key not in metrics:
            raise KeyError(f"Missing metric '{key}' in {path}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(AX_WIDTH * 2, AX_HEIGHT * 2),
    )
    axes_flat = axes.ravel()

    panel_values: List[np.ndarray] = []
    panel_specs = [
        ("quad_pos_rmse", "Quad Position RMSE [m]", POSITION_LABELS, "(a)"),
        ("load_pos_rmse", "Load Position RMSE [m]", POSITION_LABELS, "(b)"),
        ("quad_vel_rmse", "Quad Velocity RMSE [m/s]", VELOCITY_LABELS, "(c)"),
        ("load_vel_rmse", "Load Velocity RMSE [m/s]", VELOCITY_LABELS, "(d)"),
    ]
    for axis, (prefix, ylabel, component_labels, panel_label) in zip(axes_flat, panel_specs):
        values = _extract_component_values(metrics, prefix)
        panel_values.append(values)
        _plot_grouped_bars(axis, values, component_labels, ylabel)
        _add_panel_label(axis, panel_label)

    _apply_uniform_ylim(axes_flat[:2], panel_values[:2])
    _apply_uniform_ylim(axes_flat[2:], panel_values[2:])

    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.2, wspace=0.2, hspace=0.25)
    output_path = path.with_name("rmse_state_bars.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_force_rmse_bars(path: Path) -> None:
    _, metrics = _parse_force_summary(path)

    prefixes = [
        "load_force_rmse",
        "quad_force_rmse",
    ]
    required_keys = [f"{prefix}{suffix}" for prefix in prefixes for suffix in FORCE_SUFFIXES]
    for key in required_keys:
        if key not in metrics:
            raise KeyError(f"Missing metric '{key}' in {path}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(AX_WIDTH * 2, AX_HEIGHT),
        sharey=True,
    )

    panel_specs = [
        ("load_force_rmse", "Load Force RMSE [N]", "(a)"),
        ("quad_force_rmse", "Quad Force RMSE [N]", "(b)"),
    ]
    for axis, (prefix, ylabel, panel_label) in zip(axes, panel_specs):
        values = _extract_component_values(metrics, prefix, suffixes=FORCE_SUFFIXES)
        _plot_grouped_bars(axis, values, FORCE_LABELS, ylabel)
        _add_panel_label(axis, panel_label)

    axes[1].tick_params(labelleft=True, labelright=False, left=True, right=False)
    axes[1].set_yticks(axes[0].get_yticks())
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.28, wspace=0.25)
    output_path = path.with_name("rmse_force_bars.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    deal_root = payload_pkg_dir / "plots" / "deal"
    autotrans_root = payload_pkg_dir.parents[1]
    paper1_deal_root = autotrans_root / "paper1" / "deal"

    deal_roots = [deal_root]
    if paper1_deal_root.is_dir():
        deal_roots.append(paper1_deal_root)

    scenarios = [
        "figure_eight_gradual (20)",
        "figure_eight_gust_event (3)",
    ]

    for root in deal_roots:
        for scenario in scenarios:
            long_dir = root / scenario / "long"
            state_txt = long_dir / "rmse_state_lbfgs_vs_pinn_optimized.txt"
            force_txt = long_dir / "rmse_force_force_total2_axes_and_total.txt"
            if state_txt.is_file():
                render_state_rmse_bars(state_txt)
            if force_txt.is_file():
                render_force_rmse_bars(force_txt)


if __name__ == "__main__":
    main()
