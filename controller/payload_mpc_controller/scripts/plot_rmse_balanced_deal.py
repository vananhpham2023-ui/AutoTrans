from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import compare_nmpc_schemes as cmp


SCENARIOS = [
    "figure_eight_gradual (20)",
    "figure_eight_gust_event (3)",
]


def _parse_balanced_state(path: Path) -> Dict[str, Tuple[float, float]]:
    """Parse the 'Balanced RMSE' section from the state RMSE TXT."""
    metrics: Dict[str, Tuple[float, float]] = {}
    in_balanced = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("== Balanced RMSE"):
                in_balanced = True
                continue
            if not in_balanced:
                continue
            if not stripped:
                # stop at first blank after balanced block
                break
            if stripped.lower().startswith("metric"):
                # header line
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) != 3:
                continue
            name = parts[0]
            try:
                lbfgs_val = float(parts[1])
                pinn_val = float(parts[2])
            except ValueError:
                continue
            metrics[name] = (lbfgs_val, pinn_val)
    return metrics


def _parse_balanced_force(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse the 'Balanced RMSE' section from the force RMSE TXT."""
    metrics: Dict[str, Dict[str, float]] = {
        "LBFGS-NMPC": {},
        "PINN-NMPC": {},
    }
    in_balanced = False
    current: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("== Balanced RMSE"):
                in_balanced = True
                continue
            if not in_balanced:
                continue
            if stripped.startswith("Notes:"):
                break
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                label = stripped[1:-1]
                current = label if label in metrics else None
                continue
            if ":" in stripped and current:
                key, val = [p.strip() for p in stripped.split(":", 1)]
                try:
                    metrics[current][key] = float(val)
                except ValueError:
                    continue
    return metrics


def _gather_balanced_values_deal(payload_pkg_dir: Path):
    deal_root = payload_pkg_dir / "plots" / "deal"

    scenario_short = {
        "figure_eight_gradual (20)": "Gradual (20)",
        "figure_eight_gust_event (3)": "Gust (3)",
    }

    scenario_labels = []
    load_force = {"LBFGS-NMPC": [], "PINN-NMPC": []}
    quad_force = {"LBFGS-NMPC": [], "PINN-NMPC": []}
    pos_track = {"LBFGS-NMPC": [], "PINN-NMPC": []}
    vel_track = {"LBFGS-NMPC": [], "PINN-NMPC": []}

    for scenario in SCENARIOS:
        long_dir = deal_root / scenario / "long"
        state_txt = long_dir / "rmse_state_lbfgs_vs_pinn_optimized.txt"
        force_txt = long_dir / "rmse_force_force_total2_axes_and_total.txt"
        state_bal = _parse_balanced_state(state_txt)
        force_bal = _parse_balanced_force(force_txt)

        scenario_labels.append(scenario_short.get(scenario, scenario))

        # Force RMSE totals
        for scheme in ("LBFGS-NMPC", "PINN-NMPC"):
            load_force[scheme].append(force_bal[scheme]["load_force_rmse_total"])
            quad_force[scheme].append(force_bal[scheme]["quad_force_rmse_total"])

        # Position / velocity tracking: average of quad + load totals.
        for scheme_idx, scheme in enumerate(("LBFGS-NMPC", "PINN-NMPC")):
            load_pos = state_bal["load_pos_rmse_total"][scheme_idx]
            quad_pos = state_bal["quad_pos_rmse_total"][scheme_idx]
            load_vel = state_bal["load_vel_rmse_total"][scheme_idx]
            quad_vel = state_bal["quad_vel_rmse_total"][scheme_idx]
            pos_track[scheme].append(0.5 * (load_pos + quad_pos))
            vel_track[scheme].append(0.5 * (load_vel + quad_vel))

    return (
        scenario_labels,
        load_force,
        quad_force,
        pos_track,
        vel_track,
    )


def _plot_pairwise_bars(
    axis,
    scenario_labels,
    values_lbfgs,
    values_pinn,
    ylabel: str,
    title: str,
) -> None:
    n = len(scenario_labels)
    x = np.arange(n, dtype=float)
    width = 0.32

    bars_lbfgs = axis.bar(
        x - width / 2.0,
        values_lbfgs,
        width=width,
        label="LBFGS-NMPC",
        color=cmp.SCHEME_COLORS[1],
    )
    bars_pinn = axis.bar(
        x + width / 2.0,
        values_pinn,
        width=width,
        label="PINN-NMPC",
        color=cmp.SCHEME_COLORS[0],
    )

    axis.set_xticks(x)
    axis.set_xticklabels(scenario_labels, rotation=15)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    # shared legend handled at figure level; only label once
    return bars_lbfgs, bars_pinn


def render_balanced_rmse_bars(payload_pkg_dir: Path) -> None:
    (
        scenario_labels,
        load_force,
        quad_force,
        pos_track,
        vel_track,
    ) = _gather_balanced_values_deal(payload_pkg_dir)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(cmp.SUBPLOT_W * 2.4, cmp.SUBPLOT_H * 2.4),
        sharex=True,
    )
    axes = axes.ravel()

    # (a) Load force estimation
    _plot_pairwise_bars(
        axes[0],
        scenario_labels,
        load_force["LBFGS-NMPC"],
        load_force["PINN-NMPC"],
        ylabel="Load Force RMSE [N]",
        title="Load Force Estimation",
    )

    # (b) Quad force estimation
    _plot_pairwise_bars(
        axes[1],
        scenario_labels,
        quad_force["LBFGS-NMPC"],
        quad_force["PINN-NMPC"],
        ylabel="Quad Force RMSE [N]",
        title="Quad Force Estimation",
    )

    # (c) Position tracking
    _plot_pairwise_bars(
        axes[2],
        scenario_labels,
        pos_track["LBFGS-NMPC"],
        pos_track["PINN-NMPC"],
        ylabel="Position RMSE [m]",
        title="Quad + Load Position Tracking",
    )

    # (d) Velocity tracking
    _plot_pairwise_bars(
        axes[3],
        scenario_labels,
        vel_track["LBFGS-NMPC"],
        vel_track["PINN-NMPC"],
        ylabel="Velocity RMSE [m/s]",
        title="Quad + Load Velocity Tracking",
    )

    # Align y-limits for force and state pairs.
    force_vals = (
        load_force["LBFGS-NMPC"]
        + load_force["PINN-NMPC"]
        + quad_force["LBFGS-NMPC"]
        + quad_force["PINN-NMPC"]
    )
    state_pos_vals = pos_track["LBFGS-NMPC"] + pos_track["PINN-NMPC"]
    state_vel_vals = vel_track["LBFGS-NMPC"] + vel_track["PINN-NMPC"]

    def _set_shared_ylim(ax_pair, values):
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return
        vmax = float(np.max(arr[finite]))
        for ax in ax_pair:
            ax.set_ylim(0.0, vmax * 1.25)

    _set_shared_ylim((axes[0], axes[1]), force_vals)
    _set_shared_ylim((axes[2], axes[3]), state_pos_vals + state_vel_vals)

    # Subplot annotations (a)–(d).
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes, labels):
        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )

    # Shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.90])
    output_path = (
        payload_pkg_dir
        / "plots"
        / "deal"
        / "rmse_balanced_deal_bars.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_path = Path(__file__).resolve()
    payload_pkg_dir = script_path.parents[1]
    render_balanced_rmse_bars(payload_pkg_dir)


if __name__ == "__main__":
    main()

