#!/usr/bin/env python3
"""ROS node that records synchronized PINN training samples from the AutoTrans stack."""

import csv
import math
import os
import threading
import time
from typing import Dict, Optional, Tuple

import rospy
import rospkg
from geometry_msgs.msg import AccelStamped, Vector3Stamped
from mavros_msgs.msg import ESCTelemetry
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


LatestMsg = Tuple[object, float]


FIELDNAMES = [
    "timestamp",
    "scenario_id",
    "trajectory_mode",
    "wind_type",
    "quad_pos_x",
    "quad_pos_y",
    "quad_pos_z",
    "quad_vel_x",
    "quad_vel_y",
    "quad_vel_z",
    "quad_rot_r13",
    "quad_rot_r23",
    "quad_rot_r33",
    "quad_omega_x",
    "quad_omega_y",
    "quad_omega_z",
    "quad_acc_x",
    "quad_acc_y",
    "quad_acc_z",
    "load_pos_x",
    "load_pos_y",
    "load_pos_z",
    "load_vel_x",
    "load_vel_y",
    "load_vel_z",
    "load_acc_x",
    "load_acc_y",
    "load_acc_z",
    "cable_dir_x",
    "cable_dir_y",
    "cable_dir_z",
    "motor_rpm_1",
    "motor_rpm_2",
    "motor_rpm_3",
    "motor_rpm_4",
    "thrust_total",
    "mass_quad",
    "mass_load",
    "l_length",
    "sqrt_kf",
    "fl_true_x",
    "fl_true_y",
    "fl_true_z",
    "fq_true_x",
    "fq_true_y",
    "fq_true_z",
    "fl_est_x",
    "fl_est_y",
    "fl_est_z",
    "fq_est_x",
    "fq_est_y",
    "fq_est_z",
    "wind_x",
    "wind_y",
    "wind_z",
]


class PinnDataRecorder:
    """Collect multi-topic telemetry and serialize aligned samples for PINN training."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest: Dict[str, LatestMsg] = {}
        self._missing_counts: Dict[str, int] = {}
        self._stale_counts: Dict[str, int] = {}

        self.anchor_topic = rospy.get_param("~true_force_topic", "/so3_quadrotor/true_force")
        self.estimated_force_topic = rospy.get_param("~estimated_force_topic", "/mpc_controller_node/mpc/force")
        self.odom_topic = rospy.get_param("~odom_topic", "/visual_slam/odom")
        self.quad_imu_topic = rospy.get_param("~quad_imu_topic", "/imu")
        self.payload_odom_topic = rospy.get_param("~payload_odom_topic", "/payload_odom")
        self.payload_imu_topic = rospy.get_param("~payload_imu_topic", "/load_imu")
        self.cable_topic = rospy.get_param("~cable_topic", "/cable_info")
        self.wind_topic = rospy.get_param("~wind_topic", "/so3_quadrotor/wind")
        self.rpm_topic = rospy.get_param("~rpm_topic", "/rpm")

        self.max_delay = rospy.get_param("~max_topic_delay", 0.05)
        self.optional_max_delay = rospy.get_param("~optional_max_delay", 0.2)
        self.max_samples = rospy.get_param("~max_samples", 0)

        self.scenario_id = rospy.get_param("~scenario_id", os.environ.get("PINN_SCENARIO_ID", ""))
        self.trajectory_mode = rospy.get_param(
            "~trajectory_mode_hint",
            rospy.get_param("/mpc_controller_node/reference/mode", "unknown"),
        )
        self.wind_type = rospy.get_param(
            "~wind_type_hint",
            rospy.get_param("/payload_planner/simulator/wind/type", ""),
        )
        self.run_tag = rospy.get_param(
            "~run_tag",
            os.environ.get("AUTOTRANS_RUN_TAG", time.strftime("run_%Y%m%d_%H%M%S")),
        )

        self.mass_quad = self._resolve_param("/mpc_controller_node/mass_q", 1.150)
        self.mass_load = self._resolve_param("/mpc_controller_node/mass_l", 0.285)
        self.l_length = self._resolve_param("/mpc_controller_node/l_length", 0.66844507)
        self.kf = self._resolve_param("/mpc_controller_node/force_estimator/kf", 8.98132e-9)
        self.sqrt_kf = math.sqrt(self.kf) if self.kf > 0.0 else float("nan")

        self._setup_output()
        self.samples_written = 0

        self._register_topic("odom")
        self._register_topic("quad_imu")
        self._register_topic("payload_odom")
        self._register_topic("payload_imu")
        self._register_topic("cable")
        self._register_topic("est_force")
        self._register_topic("wind")
        self._register_topic("rpm")

        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._make_store_cb("odom"), queue_size=200)
        self._quad_imu_sub = rospy.Subscriber(
            self.quad_imu_topic, Imu, self._make_store_cb("quad_imu"), queue_size=400
        )
        self._payload_odom_sub = rospy.Subscriber(
            self.payload_odom_topic, Odometry, self._make_store_cb("payload_odom"), queue_size=200
        )
        self._payload_imu_sub = rospy.Subscriber(
            self.payload_imu_topic, Imu, self._make_store_cb("payload_imu"), queue_size=200
        )
        self._cable_sub = rospy.Subscriber(
            self.cable_topic, Imu, self._make_store_cb("cable"), queue_size=200
        )
        self._est_force_sub = rospy.Subscriber(
            self.estimated_force_topic,
            AccelStamped,
            self._make_store_cb("est_force"),
            queue_size=200,
        )
        if self.wind_topic:
            self._wind_sub = rospy.Subscriber(
                self.wind_topic, Vector3Stamped, self._make_store_cb("wind"), queue_size=200
            )
        else:
            self._wind_sub = None
        self._rpm_sub = rospy.Subscriber(self.rpm_topic, ESCTelemetry, self._make_store_cb("rpm"), queue_size=200)

        anchor_queue = rospy.get_param("~anchor_queue", 200)
        self._anchor_sub = rospy.Subscriber(
            self.anchor_topic, AccelStamped, self._on_anchor_force, queue_size=anchor_queue
        )

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "[pinn_data_recorder] Recording to %s (scenario_id='%s'). Required topics: %s",
            self.csv_path,
            self.scenario_id,
            [
                self.anchor_topic,
                self.estimated_force_topic,
                self.odom_topic,
                self.quad_imu_topic,
                self.payload_odom_topic,
                self.payload_imu_topic,
                self.cable_topic,
            ],
        )

    def _setup_output(self) -> None:
        pkg_path = rospkg.RosPack().get_path("payload_mpc_controller")
        default_dir = os.path.join(pkg_path, "plots", "pinn_dataset")
        output_dir = rospy.get_param(
            "~output_dir",
            os.environ.get("PINN_DATA_OUTPUT", default_dir),
        )
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        safe_tag = "".join([c if (c.isalnum() or c in "-_") else "-" for c in self.run_tag])
        filename = rospy.get_param("~output_filename", f"pinn_dataset_{safe_tag}_{timestamp_str}.csv")
        self.csv_path = os.path.join(output_dir, filename)
        rospy.loginfo("[pinn_data_recorder] Output CSV: %s", self.csv_path)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._csv_file.flush()

    def _register_topic(self, key: str) -> None:
        self._missing_counts[key] = 0
        self._stale_counts[key] = 0

    def _resolve_param(self, name: str, default: float) -> float:
        try:
            return float(rospy.get_param(name))
        except KeyError:
            rospy.logwarn_once("[pinn_data_recorder] Parameter %s not found; falling back to %.4f.", name, default)
            return default
        except (TypeError, ValueError):
            rospy.logwarn_once("[pinn_data_recorder] Parameter %s is not numeric; falling back to %.4f.", name, default)
            return default

    def _make_store_cb(self, key: str):
        def _callback(msg):
            stamp = self._extract_time(msg)
            with self._lock:
                self._latest[key] = (msg, stamp)

        return _callback

    @staticmethod
    def _extract_time(msg) -> float:
        if hasattr(msg, "header"):
            header = getattr(msg, "header")
            if header is not None and hasattr(header, "stamp"):
                return header.stamp.to_sec()
        return rospy.get_rostime().to_sec()

    def _on_anchor_force(self, true_msg: AccelStamped) -> None:
        stamp = true_msg.header.stamp.to_sec()
        sample = self._build_sample(true_msg, stamp)
        if sample is None:
            return

        if self._csv_file.closed:
            rospy.logwarn_once(
                "[pinn_data_recorder] CSV file already closed; dropping subsequent samples."
            )
        else:
            self._writer.writerow(sample)
            self._csv_file.flush()
            self.samples_written += 1

        if self._csv_file.closed:
            return

        if self.samples_written % 500 == 0:
            rospy.loginfo("[pinn_data_recorder] Recorded %d samples.", self.samples_written)

        if self.max_samples > 0 and self.samples_written >= self.max_samples:
            rospy.loginfo(
                "[pinn_data_recorder] Reached max_samples=%d, shutting down.", self.max_samples
            )
            rospy.signal_shutdown("pinn_data_recorder reached max_samples")

    def _build_sample(self, true_msg: AccelStamped, stamp: float) -> Optional[Dict[str, float]]:
        required_keys = ("odom", "quad_imu", "payload_odom", "payload_imu", "cable", "est_force")
        with self._lock:
            cache = self._latest.copy()

        for key in required_keys:
            entry = cache.get(key)
            if entry is None:
                self._missing_counts[key] += 1
                return None
            msg, msg_stamp = entry
            if abs(msg_stamp - stamp) > self.max_delay:
                self._stale_counts[key] += 1
                return None

        odom: Odometry = cache["odom"][0]
        quad_imu: Imu = cache["quad_imu"][0]
        payload_odom: Odometry = cache["payload_odom"][0]
        payload_imu: Imu = cache["payload_imu"][0]
        cable: Imu = cache["cable"][0]
        est_force: AccelStamped = cache["est_force"][0]

        quad_pos = odom.pose.pose.position
        quad_vel = odom.twist.twist.linear
        r13, r23, r33 = self._quat_third_column(odom.pose.pose.orientation)
        quad_omega = quad_imu.angular_velocity
        quad_acc = quad_imu.linear_acceleration

        load_pos = payload_odom.pose.pose.position
        load_vel = payload_odom.twist.twist.linear
        load_acc = payload_imu.linear_acceleration

        cable_dir = self._normalize_vector(
            cable.angular_velocity.x, cable.angular_velocity.y, cable.angular_velocity.z
        )

        rpms, thrust_total = self._resolve_rpm(cache.get("rpm"), stamp)
        wind = self._resolve_wind(cache.get("wind"), stamp)

        sample = {
            "timestamp": stamp,
            "scenario_id": self.scenario_id,
            "trajectory_mode": self.trajectory_mode,
            "wind_type": self.wind_type,
            "quad_pos_x": quad_pos.x,
            "quad_pos_y": quad_pos.y,
            "quad_pos_z": quad_pos.z,
            "quad_vel_x": quad_vel.x,
            "quad_vel_y": quad_vel.y,
            "quad_vel_z": quad_vel.z,
            "quad_rot_r13": r13,
            "quad_rot_r23": r23,
            "quad_rot_r33": r33,
            "quad_omega_x": quad_omega.x,
            "quad_omega_y": quad_omega.y,
            "quad_omega_z": quad_omega.z,
            "quad_acc_x": quad_acc.x,
            "quad_acc_y": quad_acc.y,
            "quad_acc_z": quad_acc.z,
            "load_pos_x": load_pos.x,
            "load_pos_y": load_pos.y,
            "load_pos_z": load_pos.z,
            "load_vel_x": load_vel.x,
            "load_vel_y": load_vel.y,
            "load_vel_z": load_vel.z,
            "load_acc_x": load_acc.x,
            "load_acc_y": load_acc.y,
            "load_acc_z": load_acc.z,
            "cable_dir_x": cable_dir[0],
            "cable_dir_y": cable_dir[1],
            "cable_dir_z": cable_dir[2],
            "motor_rpm_1": rpms[0],
            "motor_rpm_2": rpms[1],
            "motor_rpm_3": rpms[2],
            "motor_rpm_4": rpms[3],
            "thrust_total": thrust_total,
            "mass_quad": self.mass_quad,
            "mass_load": self.mass_load,
            "l_length": self.l_length,
            "sqrt_kf": self.sqrt_kf,
            "fl_true_x": true_msg.accel.linear.x,
            "fl_true_y": true_msg.accel.linear.y,
            "fl_true_z": true_msg.accel.linear.z,
            "fq_true_x": true_msg.accel.angular.x,
            "fq_true_y": true_msg.accel.angular.y,
            "fq_true_z": true_msg.accel.angular.z,
            "fl_est_x": est_force.accel.linear.x,
            "fl_est_y": est_force.accel.linear.y,
            "fl_est_z": est_force.accel.linear.z,
            "fq_est_x": est_force.accel.angular.x,
            "fq_est_y": est_force.accel.angular.y,
            "fq_est_z": est_force.accel.angular.z,
            "wind_x": wind[0],
            "wind_y": wind[1],
            "wind_z": wind[2],
        }
        return sample

    def _resolve_rpm(self, entry: Optional[LatestMsg], stamp: float) -> Tuple[Tuple[float, float, float, float], float]:
        nan = float("nan")
        if entry is None:
            return (nan, nan, nan, nan), nan

        msg: ESCTelemetry
        msg, msg_stamp = entry
        if abs(msg_stamp - stamp) > self.optional_max_delay or not msg.esc_telemetry:
            return (nan, nan, nan, nan), nan

        rpms = [item.rpm for item in msg.esc_telemetry]
        while len(rpms) < 4:
            rpms.append(nan)
        rpms = tuple(rpms[:4])

        if self.kf <= 0.0 or any(math.isnan(rpm) for rpm in rpms):
            return rpms, nan

        thrust = self.kf * sum(rpm * rpm for rpm in rpms)
        return rpms, thrust

    def _resolve_wind(self, entry: Optional[LatestMsg], stamp: float) -> Tuple[float, float, float]:
        nan = float("nan")
        if entry is None:
            return (nan, nan, nan)

        msg: Vector3Stamped
        msg, msg_stamp = entry
        if abs(msg_stamp - stamp) > self.optional_max_delay:
            return (nan, nan, nan)
        vec = msg.vector
        return (vec.x, vec.y, vec.z)

    @staticmethod
    def _normalize_vector(x: float, y: float, z: float) -> Tuple[float, float, float]:
        norm = math.sqrt(x * x + y * y + z * z)
        if norm < 1e-6:
            return (float("nan"), float("nan"), float("nan"))
        inv = 1.0 / norm
        return (x * inv, y * inv, z * inv)

    @staticmethod
    def _quat_third_column(orientation) -> Tuple[float, float, float]:
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        r13 = 2.0 * (x * z + w * y)
        r23 = 2.0 * (y * z - w * x)
        r33 = 1.0 - 2.0 * (x * x + y * y)
        return (r13, r23, r33)

    def _on_shutdown(self) -> None:
        try:
            self._csv_file.flush()
            self._csv_file.close()
        except Exception:
            pass

        summary = ", ".join(
            f"{key}: missing={self._missing_counts.get(key, 0)}, stale={self._stale_counts.get(key, 0)}"
            for key in sorted(self._missing_counts.keys())
        )
        rospy.loginfo("[pinn_data_recorder] Shutdown after writing %d samples. %s", self.samples_written, summary)


def main() -> None:
    rospy.init_node("pinn_data_recorder", anonymous=False)
    PinnDataRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
