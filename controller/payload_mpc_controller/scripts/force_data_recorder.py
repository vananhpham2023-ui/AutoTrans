#!/usr/bin/env python3
"""ROS node that records true vs. estimated external forces and generates comparison reports."""

import csv
import math
import os
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import AccelStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber

try:
    # When executed as a package module (e.g. via rosrun with catkin install)
    from .plot_force_comparison import (
        CSV_FIELDS,
        compute_rmse,
        render_force_plot,
        render_force_error_plot,
        render_wind_velocity_plot,
        plt as plotter_plt,
    )
except ImportError:
    # Fallback for direct script execution where relative import is unavailable
    import os
    import sys

    script_dir = os.path.dirname(os.path.realpath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from plot_force_comparison import (  # type: ignore
        CSV_FIELDS,
        compute_rmse,
        render_force_plot,
        render_force_error_plot,
        render_wind_velocity_plot,
        plt as plotter_plt,
    )


class ForceDataRecorder:
    """Record synchronized true/estimated forces, provide live visualization, and generate reports."""

    def __init__(self) -> None:
        self.true_topic = rospy.get_param("~true_force_topic", "/so3_quadrotor/true_force")
        self.est_topic = rospy.get_param("~estimated_force_topic", "/mpc_controller_node/mpc/force")
        self.cycles_to_report = rospy.get_param("~cycles_to_report", 10)
        self.extra_wait = rospy.get_param("~extra_wait_seconds", 0.0)
        self.wind_topic = rospy.get_param("~wind_topic", "/so3_quadrotor/wind")
        self._wind_enabled = bool(self.wind_topic)

        self._samples: List[Dict[str, np.ndarray]] = []
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None
        self._report_generated = False
        self._last_plot_update = 0.0

        pkg_path = rospkg.RosPack().get_path("payload_mpc_controller")
        default_dir = os.path.join(pkg_path, "plots")
        run_tag = rospy.get_param(
            "~run_tag",
            os.environ.get("AUTOTRANS_RUN_TAG", time.strftime("run_%Y%m%d_%H%M%S")),
        )
        output_dir = rospy.get_param(
            "~output_dir",
            os.environ.get("FORCE_DATA_OUTPUT", default_dir),
        )
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._run_root_dir = os.path.dirname(os.path.abspath(output_dir)) if output_dir else ""
        safe_tag = ''.join(c if c.isalnum() or c in "-_" else "-" for c in run_tag)
        prefix = f"force_{safe_tag}"
        self.csv_path = os.path.join(output_dir, f"{prefix}.csv")
        self.png_path = os.path.join(output_dir, f"{prefix}.png")
        self.totals_png_path = os.path.join(output_dir, f"{prefix}_totals.png")
        self.wind_png_path = os.path.join(output_dir, f"{prefix}_wind.png")
        self.errors_png_path = os.path.join(output_dir, f"{prefix}_errors.png")
        self.error_totals_png_path = os.path.join(output_dir, f"{prefix}_errors_totals.png")
        rospy.loginfo("[force_data_recorder] Writing outputs under %s (run_tag=%s)", output_dir, safe_tag)

        self._cycle_seconds = self._cycle_duration()
        self.report_duration = self._resolve_report_duration()
        rospy.loginfo(
            "[force_data_recorder] Waiting %.1f s (%.1f cycles) before generating report.",
            self.report_duration,
            0.0 if self.report_duration == 0 or self._cycle_seconds == 0 else self.report_duration / self._cycle_seconds,
        )

        self._init_plot()

        true_sub = Subscriber(self.true_topic, AccelStamped)
        est_sub = Subscriber(self.est_topic, AccelStamped)
        subs = [true_sub, est_sub]
        if self._wind_enabled:
            self._wind_sub = Subscriber(self.wind_topic, Vector3Stamped)
            subs.append(self._wind_sub)
        else:
            self._wind_sub = None
        slop = rospy.get_param("~sync_slop", 0.05)
        queue_size = rospy.get_param("~sync_queue", 200)
        self._sync = ApproximateTimeSynchronizer(subs, queue_size=queue_size, slop=slop)
        if self._wind_enabled:
            self._sync.registerCallback(self._on_force_pair_with_wind)
        else:
            self._sync.registerCallback(self._on_force_pair)

        rospy.on_shutdown(self._on_shutdown)

    def _metrics_summary_path(self) -> Optional[str]:
        if not self._run_root_dir:
            return None
        return os.path.join(self._run_root_dir, "metrics_summary.txt")

    def _append_metrics_block(self, header: str, lines: Sequence[str]) -> None:
        if not lines:
            return
        path = self._metrics_summary_path()
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"[{header}]\n")
                for line in lines:
                    handle.write(line + "\n")
                handle.write("\n")
        except OSError as exc:
            rospy.logwarn("[force_data_recorder] Failed to append metrics summary (%s): %s", header, exc)

    def _record_force_rmse(self, rmse: Dict[str, Dict[str, np.ndarray]]) -> None:
        if not rmse:
            return
        fl = rmse.get("fl")
        fq = rmse.get("fq")
        if not fl or not fq:
            return
        fmt = lambda label, value, unit: f"{label} [{unit}]: {float(value):.4f}"
        load_lines = [
            fmt("X", fl["components"][0], "N"),
            fmt("Y", fl["components"][1], "N"),
            fmt("Z", fl["components"][2], "N"),
        ]
        quad_lines = [
            fmt("X", fq["components"][0], "N"),
            fmt("Y", fq["components"][1], "N"),
            fmt("Z", fq["components"][2], "N"),
        ]
        total_lines = [
            fmt("Load total", fl["total"][0], "N"),
            fmt("Quad total", fq["total"][0], "N"),
        ]
        self._append_metrics_block("Load Force RMSE", load_lines)
        self._append_metrics_block("Quad Force RMSE", quad_lines)
        self._append_metrics_block("Total Force RMSE", total_lines)

    def _cycle_duration(self) -> float:
        """Infer the reference trajectory cycle duration from MPC parameters."""
        try:
            use_planner = bool(rospy.get_param("/mpc_controller_node/reference/use_planner"))
            if use_planner:
                return 0.0
        except KeyError:
            pass

        mode = rospy.get_param("/mpc_controller_node/reference/mode", "circle")
        if mode in ("circle", "figure_eight", "helix"):
            omega = rospy.get_param(
                f"/mpc_controller_node/reference/{mode}/angular_velocity",
                rospy.get_param("/mpc_controller_node/reference/circle/angular_velocity", 0.0),
            )
            if abs(omega) > 1e-6:
                return abs(2.0 * math.pi / omega)
        rospy.logwarn_throttle(
            10.0,
            "[force_data_recorder] Unable to determine cycle duration for mode '%s', falling back to fixed window.",
            mode,
        )
        return 0.0

    def _resolve_report_duration(self) -> float:
        cycle = self._cycle_seconds
        manual_duration = rospy.get_param("~report_duration_override", 0.0)
        if manual_duration > 0.0:
            return manual_duration + max(self.extra_wait, 0.0)
        if cycle > 0.0 and self.cycles_to_report > 0:
            return self.cycles_to_report * cycle + max(self.extra_wait, 0.0)
        default_window = rospy.get_param("~fallback_window_seconds", 60.0)
        return default_window + max(self.extra_wait, 0.0)

    def _init_plot(self) -> None:
        """Prepare live plot if matplotlib backend allows it."""
        self._plt = plotter_plt
        if self._plt is None:
            rospy.loginfo("[force_data_recorder] matplotlib not available, live plot disabled.")
            self._live_plot = False
            return

        backend = self._plt.get_backend().lower()
        non_interactive = backend.endswith("agg")
        self._live_plot = rospy.get_param("~enable_plot", True) and not non_interactive
        if not self._live_plot:
            rospy.loginfo(
                "[force_data_recorder] Live plot disabled (backend=%s, enable_plot=%s).",
                backend,
                rospy.get_param("~enable_plot", True),
            )
            return

        self._plt.ion()
        self._fig, axes = self._plt.subplots(2, 3, figsize=(16, 8), sharex=True)
        self._axes = axes.flatten()
        self._lines_true = []
        self._lines_est = []

        labels = [
            "Load Force X",
            "Load Force Y",
            "Load Force Z",
            "Quad Force X",
            "Quad Force Y",
            "Quad Force Z",
        ]
        for axis, label in zip(self._axes, labels):
            line_true, = axis.plot([], [], label="True", color="#1f77b4", linewidth=1.4)
            line_est, = axis.plot([], [], label="Estimated", color="#ff7f0e", linestyle="--", linewidth=1.2)
            axis.set_ylabel("Force [N]")
            axis.set_title(label)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            self._lines_true.append(line_true)
            self._lines_est.append(line_est)
        for axis in self._axes[-3:]:
            axis.set_xlabel("Time [s]")
        self._axes[0].legend(loc="upper right")
        self._fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        self._rmse_text = self._fig.text(0.5, 0.02, "", ha="center")

    def _on_force_pair(self, true_msg: AccelStamped, est_msg: AccelStamped) -> None:
        timestamp = est_msg.header.stamp.to_sec()
        true_fl = np.array(
            [true_msg.accel.linear.x, true_msg.accel.linear.y, true_msg.accel.linear.z],
            dtype=float,
        )
        true_fq = np.array(
            [true_msg.accel.angular.x, true_msg.accel.angular.y, true_msg.accel.angular.z],
            dtype=float,
        )
        est_fl = np.array(
            [est_msg.accel.linear.x, est_msg.accel.linear.y, est_msg.accel.linear.z],
            dtype=float,
        )
        est_fq = np.array(
            [est_msg.accel.angular.x, est_msg.accel.angular.y, est_msg.accel.angular.z],
            dtype=float,
        )

        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            self._samples.append(
                {
                    "time": timestamp,
                    "fl_true": true_fl,
                    "fl_est": est_fl,
                    "fq_true": true_fq,
                    "fq_est": est_fq,
                }
            )
            elapsed = timestamp - self._start_time

        if self._live_plot:
            self._maybe_update_plot()

        if not self._report_generated and elapsed >= self.report_duration and len(self._samples) > 10:
            rospy.loginfo("[force_data_recorder] Report window reached (%.2f s).", elapsed)
            self._generate_report()

    def _on_force_pair_with_wind(self, true_msg: AccelStamped, est_msg: AccelStamped, wind_msg: Vector3Stamped) -> None:
        timestamp = est_msg.header.stamp.to_sec()
        true_fl = np.array(
            [true_msg.accel.linear.x, true_msg.accel.linear.y, true_msg.accel.linear.z],
            dtype=float,
        )
        true_fq = np.array(
            [true_msg.accel.angular.x, true_msg.accel.angular.y, true_msg.accel.angular.z],
            dtype=float,
        )
        est_fl = np.array(
            [est_msg.accel.linear.x, est_msg.accel.linear.y, est_msg.accel.linear.z],
            dtype=float,
        )
        est_fq = np.array(
            [est_msg.accel.angular.x, est_msg.accel.angular.y, est_msg.accel.angular.z],
            dtype=float,
        )
        wind = np.array([wind_msg.vector.x, wind_msg.vector.y, wind_msg.vector.z], dtype=float)

        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            self._samples.append(
                {
                    "time": timestamp,
                    "fl_true": true_fl,
                    "fl_est": est_fl,
                    "fq_true": true_fq,
                    "fq_est": est_fq,
                    "wind": wind,
                }
            )
            elapsed = timestamp - self._start_time

        if self._live_plot:
            self._maybe_update_plot()

        if not self._report_generated and elapsed >= self.report_duration and len(self._samples) > 10:
            rospy.loginfo("[force_data_recorder] Report window reached (%.2f s).", elapsed)
            self._generate_report()

    def _maybe_update_plot(self) -> None:
        now = rospy.get_time()
        if now - self._last_plot_update < 0.1:
            return
        self._last_plot_update = now

        with self._lock:
            samples = list(self._samples)
        if not samples:
            return
        base_time = samples[0]["time"]
        times = np.array([s["time"] - base_time for s in samples], dtype=float)

        data_cache = {
            "fl_true": np.array([s["fl_true"] for s in samples], dtype=float),
            "fl_est": np.array([s["fl_est"] for s in samples], dtype=float),
            "fq_true": np.array([s["fq_true"] for s in samples], dtype=float),
            "fq_est": np.array([s["fq_est"] for s in samples], dtype=float),
        }

        pairs = [
            ("fl_true", "fl_est", 0),
            ("fl_true", "fl_est", 1),
            ("fl_true", "fl_est", 2),
            ("fq_true", "fq_est", 0),
            ("fq_true", "fq_est", 1),
            ("fq_true", "fq_est", 2),
        ]
        for axis, line_true, line_est, (true_key, est_key, idx) in zip(
            self._axes, self._lines_true, self._lines_est, pairs
        ):
            line_true.set_data(times, data_cache[true_key][:, idx])
            line_est.set_data(times, data_cache[est_key][:, idx])
            axis.relim()
            axis.autoscale_view()

        rmse = compute_rmse(samples)
        rmse_text = (
            f"Load RMSE [N]: total={rmse['fl']['total'][0]:.3f}, "
            f"x={rmse['fl']['components'][0]:.3f}, y={rmse['fl']['components'][1]:.3f}, z={rmse['fl']['components'][2]:.3f} | "
            f"Quad RMSE [N]: total={rmse['fq']['total'][0]:.3f}, "
            f"x={rmse['fq']['components'][0]:.3f}, y={rmse['fq']['components'][1]:.3f}, z={rmse['fq']['components'][2]:.3f}"
        )
        self._rmse_text.set_text(rmse_text)
        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def _generate_report(self) -> None:
        with self._lock:
            if self._report_generated:
                return
            samples = list(self._samples)
            self._report_generated = True

        if not samples:
            rospy.logwarn("[force_data_recorder] No samples captured; skipping report.")
            return

        rospy.loginfo("[force_data_recorder] Writing CSV to %s", self.csv_path)
        with open(self.csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = list(CSV_FIELDS)
            if self._wind_enabled:
                header.extend(["wind_x", "wind_y", "wind_z"])
            writer.writerow(header)
            for sample in samples:
                row = [
                    sample["time"],
                    sample["fl_true"][0],
                    sample["fl_true"][1],
                    sample["fl_true"][2],
                    sample["fl_est"][0],
                    sample["fl_est"][1],
                    sample["fl_est"][2],
                    sample["fq_true"][0],
                    sample["fq_true"][1],
                    sample["fq_true"][2],
                    sample["fq_est"][0],
                    sample["fq_est"][1],
                    sample["fq_est"][2],
                ]
                if self._wind_enabled:
                    row.extend(sample.get("wind", np.zeros(3)))
                writer.writerow(row)

        rmse = compute_rmse(samples)
        rospy.loginfo(
            "[force_data_recorder] RMSE summary -> Load total: %.3f N (x=%.3f, y=%.3f, z=%.3f); "
            "Quad total: %.3f N (x=%.3f, y=%.3f, z=%.3f)",
            rmse["fl"]["total"][0],
            rmse["fl"]["components"][0],
            rmse["fl"]["components"][1],
            rmse["fl"]["components"][2],
            rmse["fq"]["total"][0],
            rmse["fq"]["components"][0],
            rmse["fq"]["components"][1],
            rmse["fq"]["components"][2],
        )
        self._record_force_rmse(rmse)

        try:
            render_force_plot(
                samples,
                output_path=self.png_path,
                total_output_path=self.totals_png_path,
                show=False,
                rmse=rmse,
            )
            rospy.loginfo(
                "[force_data_recorder] Saved force comparison plots to %s and %s",
                self.png_path,
                self.totals_png_path,
            )
        except RuntimeError as exc:
            rospy.logwarn("Failed to render force comparison plot: %s", exc)

        try:
            render_force_error_plot(
                samples,
                output_path=self.errors_png_path,
                total_output_path=self.error_totals_png_path,
                show=False,
            )
            rospy.loginfo(
                "[force_data_recorder] Saved force error plots to %s and %s",
                self.errors_png_path,
                self.error_totals_png_path,
            )
        except RuntimeError as exc:
            rospy.logwarn("Failed to render force error plot: %s", exc)

        if self._wind_enabled:
            wind_samples = [s for s in samples if "wind" in s]
            if wind_samples:
                try:
                    render_wind_velocity_plot(
                        wind_samples,
                        output_path=self.wind_png_path,
                        show=False,
                    )
                    rospy.loginfo(
                        "[force_data_recorder] Saved wind velocity plot to %s",
                        self.wind_png_path,
                    )
                except RuntimeError as exc:
                    rospy.logwarn("Failed to render wind velocity plot: %s", exc)
            else:
                rospy.logwarn(
                    "[force_data_recorder] Wind topic configured but no wind data recorded; skipping wind plot."
                )

    def _on_shutdown(self) -> None:
        if not self._report_generated:
            rospy.loginfo("[force_data_recorder] Node shutting down, generating final report.")
            self._generate_report()


def main() -> None:
    rospy.init_node("force_data_recorder", anonymous=False)
    recorder = ForceDataRecorder()
    rospy.loginfo(
        "[force_data_recorder] Node started. Subscribing to %s and %s.",
        recorder.true_topic,
        recorder.est_topic,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
