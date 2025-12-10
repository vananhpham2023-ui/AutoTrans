#!/bin/bash
# 诊断脚本：检查ROS话题和节点状态

echo "================================"
echo "检查ROS节点状态"
echo "================================"
rosnode list

echo ""
echo "================================"
echo "检查所有话题"
echo "================================"
rostopic list

echo ""
echo "================================"
echo "检查里程计话题发布频率"
echo "================================"
timeout 3 rostopic hz /visual_slam/odom || echo "话题 /visual_slam/odom 没有数据"

echo ""
echo "================================"
echo "检查负载里程计话题"
echo "================================"
timeout 3 rostopic hz /payload_odom || echo "话题 /payload_odom 没有数据"

echo ""
echo "================================"
echo "检查MPC控制器节点信息"
echo "================================"
rosnode info /mpc_controller_node 2>&1 | grep -A 5 "Subscriptions:"

echo ""
echo "================================"
echo "检查仿真器节点信息"
echo "================================"
rosnode info /so3_quadrotor 2>&1 | grep -A 5 "Publications:"

echo ""
echo "================================"
echo "检查里程计话题内容（最近1条消息）"
echo "================================"
timeout 2 rostopic echo /visual_slam/odom -n 1 || echo "无法获取里程计数据"
