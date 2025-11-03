#!/bin/bash
# 快速测试脚本

echo "========================================="
echo "测试修复后的仿真器"
echo "========================================="

cd ~/autotrans_ws
source devel/setup.bash

echo ""
echo "正在启动测试..."
echo "如果看到里程计数据发布，则修复成功！"
echo ""

timeout 10 roslaunch ~/autotrans_ws/src/AutoTrans/test_simulator.launch 2>&1 | tee /tmp/test_output.log &
TEST_PID=$!

sleep 5

echo ""
echo "========================================="
echo "检查结果："
echo "========================================="

# 检查nodelet是否崩溃
if grep -q "Failed to load nodelet" /tmp/test_output.log; then
    echo "❌ Nodelet加载失败"
    cat /tmp/test_output.log | grep -i "error\|fatal"
    exit 1
fi

if grep -q "exit code -11" /tmp/test_output.log; then
    echo "❌ Segmentation fault仍然存在"
    exit 1
fi

# 检查风场配置是否加载
if grep -q "Loading wind field configuration" /tmp/test_output.log; then
    echo "✓ 风场配置开始加载"
fi

if grep -q "Loaded configuration type" /tmp/test_output.log; then
    echo "✓ 风场配置加载成功"
fi

# 检查里程计话题
sleep 2
ODOM_HZ=$(timeout 2 rostopic hz /test/odom 2>&1 | grep "average rate" || echo "no data")

if echo "$ODOM_HZ" | grep -q "average rate"; then
    echo "✓ 里程计数据正常发布: $ODOM_HZ"
    echo ""
    echo "========================================="
    echo "✅ 修复成功！仿真器正常运行"
    echo "========================================="
else
    echo "⚠ 里程计数据未发布"
    echo "   请检查日志: cat /tmp/test_output.log"
fi

# 清理
kill $TEST_PID 2>/dev/null
sleep 1
pkill -9 -f "test_simulator" 2>/dev/null

exit 0
