#!/usr/bin/env python3
"""Block until a completion topic publishes a std_msgs/Empty message."""

import argparse
import sys

import rospy
from std_msgs.msg import Empty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/mpc/run_completed",
        help="Topic to wait on (std_msgs/Empty).",
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


def main() -> int:
    args = parse_args()
    rospy.init_node(args.node_name, anonymous=True, disable_signals=True)
    try:
        if args.timeout > 0.0:
            rospy.wait_for_message(args.topic, Empty, timeout=args.timeout)
        else:
            rospy.wait_for_message(args.topic, Empty)
    except rospy.ROSException as exc:
        rospy.logerr("Failed to wait for completion: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
