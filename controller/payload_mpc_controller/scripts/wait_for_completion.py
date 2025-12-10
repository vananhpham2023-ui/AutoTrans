#!/usr/bin/env python3
"""Block until a completion topic publishes a std_msgs/Empty message."""

import argparse
import os
import sys

import rospy
from std_msgs.msg import Empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/mpc/run_completed",
        help="Base topic to wait on (std_msgs/Empty). If --run-tag is provided, it will be appended.",
    )
    parser.add_argument(
        "--run-tag",
        default=os.environ.get("AUTOTRANS_RUN_TAG", ""),
        help="Run tag suffix for the completion topic (default: $AUTOTRANS_RUN_TAG).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Optional timeout in seconds; 0 means wait indefinitely.",
    )
    parser.add_argument(
        "--node-name",
        default="completion_waiter",
        help="ROS node name to use while waiting.",
    )
    return parser.parse_args()


def _sanitize_tag(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        elif ch in (".", " ", "-"):
            safe.append("_")
    return "".join(safe) or "run"


def main() -> int:
    args = parse_args()
    topic = args.topic.rstrip("/")
    if args.run_tag:
        topic = f"{topic}/{_sanitize_tag(args.run_tag)}"
    rospy.init_node(args.node_name, anonymous=True, disable_signals=True)
    start_time = rospy.Time.now()
    rospy.loginfo("Waiting for completion on %s (started at %.3f)", topic, start_time.to_sec())
    try:
        if args.timeout > 0.0:
            rospy.wait_for_message(topic, Empty, timeout=args.timeout)
        else:
            rospy.wait_for_message(topic, Empty)
    except rospy.ROSException as exc:
        rospy.logerr("Failed to wait for completion: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
