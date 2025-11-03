#!/bin/bash
# 完整的诊断和修复脚本

echo "========================================"
echo "AutoTrans 仿真诊断脚本"
echo "========================================"

# 步骤1：检查workspace是否正确source
echo ""
echo "[步骤1] 检查workspace设置..."
if [ -z "$ROS_PACKAGE_PATH" ]; then
    echo "❌ 错误: ROS环境未设置"
    echo "   请运行: source /opt/ros/noetic/setup.bash"
    exit 1
fi

if echo "$ROS_PACKAGE_PATH" | grep -q "autotrans_ws"; then
    echo "✓ Workspace已正确source"
else
    echo "❌ 警告: autotrans_ws未在ROS_PACKAGE_PATH中"
    echo "   请运行: cd ~/autotrans_ws && source devel/setup.bash"
    exit 1
fi

# 步骤2：检查编译状态
echo ""
echo "[步骤2] 检查包是否已编译..."
if [ -f "$HOME/autotrans_ws/devel/lib/libso3_quadrotor_nodelet.so" ]; then
    echo "✓ so3_quadrotor nodelet已编译"
else
    echo "❌ 错误: so3_quadrotor nodelet未找到"
    echo "   请运行: cd ~/autotrans_ws && catkin build so3_quadrotor"
    exit 1
fi

if [ -f "$HOME/autotrans_ws/devel/lib/payload_mpc_controller/mpc_controller_node" ]; then
    echo "✓ mpc_controller_node已编译"
else
    echo "❌ 错误: mpc_controller_node未找到"
    echo "   请运行: cd ~/autotrans_ws && catkin build payload_mpc_controller"
    exit 1
fi

# 步骤3：测试nodelet是否可以加载
echo ""
echo "[步骤3] 测试nodelet加载..."
timeout 3 rosrun nodelet nodelet standalone so3_quadrotor/Nodelet __name:=test_nodelet >/tmp/nodelet_test.log 2>&1 &
NODELET_PID=$!
sleep 2

if ps -p $NODELET_PID > /dev/null; then
    echo "✓ Nodelet可以加载"
    kill $NODELET_PID 2>/dev/null
else
    echo "❌ 错误: Nodelet加载失败"
    echo "   日志:"
    cat /tmp/nodelet_test.log
    exit 1
fi

# 步骤4：提供运行建议
echo ""
echo "========================================"
echo "诊断完成！建议按以下步骤运行："
echo "========================================"
echo ""
echo "1. 打开一个新终端，运行:"
echo "   cd ~/autotrans_ws"
echo "   source devel/setup.bash"
echo "   roslaunch payload_planner controller_only.launch trajectory_mode:=circle 2>&1 | tee /tmp/sim_log.txt"
echo ""
echo "2. 如果看到警告，打开另一个终端运行诊断:"
echo "   bash ~/autotrans_ws/src/AutoTrans/check_topics.sh"
echo ""
echo "3. 检查关键话题:"
echo "   rostopic hz /visual_slam/odom"
echo "   rostopic hz /payload_odom"
echo ""
echo "4. 检查节点状态:"
echo "   rosnode list | grep -E 'so3_quadrotor|mpc_controller'"
echo ""
echo "5. 如果仍有问题，查看完整日志:"
echo "   cat /tmp/sim_log.txt | grep -E 'ERROR|WARN|so3_quadrotor|wind'"
echo ""
