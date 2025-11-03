# AutoTrans: Control-Oriented Trajectory Workflow

[![AutoTrans](imgs/cover.png)](https://youtu.be/X9g-ivBqy5g)

**Video**: [YouTube](https://youtu.be/X9g-ivBqy5g) | [Bilibili](https://www.bilibili.com/video/BV1aM4y1a7c9)

---

## Overview
- 全流程支持控制端直接生成圆形、八字和螺旋解析轨迹，可与原有规划器多项式轨迹一键切换。
- MPC 控制器通过 `reference_generator` 封装的 PVAJSC 采样接口获取高阶导数，保持 yaw 规划与负载平坦化逻辑一致。
- `replan.launch` / `controller_only.launch` 与双套 RViz 配置对应双模式，便于对比仿真与控制-only 演示。
- 阶段 6 新增单元测试，逐项校验解析轨迹的导数公式、reset 行为以及有限长度螺旋的收尾状态。

---

## 依赖与环境准备

1. **系统要求**：Ubuntu 18.04/20.04 + ROS（推荐 Noetic）。
2. **安装依赖包**
   ```bash
   sudo apt update
   sudo apt install ros-"${ROS_DISTRO}"-mavros-msgs
   sudo apt install python3-setuptools   # 提供 catkin 工具所需的 pkg_resources
   ```
   可选：若编译时看到 PCL 关于 `pcap/png/libusb` 的 WARNING，可以安装 `libpcap-dev libpng-dev libusb-1.0-0-dev` 后重新编译。
3. **创建工作空间**
   ```bash
   mkdir -p ~/autotrans_ws/src
   cd ~/autotrans_ws/src
   git clone https://github.com/HKUST-Aerial-Robotics/AutoTrans.git
   cd ~/autotrans_ws
   ```

---

## 编译与测试流程

1. **首次全量编译**
   ```bash
   catkin build
   source devel/setup.bash
   ```
   为方便使用，可将 `source ~/autotrans_ws/devel/setup.bash` 添加进 `~/.bashrc`。

2. **Stage 6 单元测试（解析轨迹）**
   ```bash
   catkin build payload_mpc_controller --no-deps --catkin-make-args reference_generator_test
   devel/.private/payload_mpc_controller/lib/payload_mpc_controller/reference_generator_test
   ```
   - 第一步生成并编译测试目标
   - 第二步直接执行可执行文件获取结果（3/3 PASS 即为通过）
   测试覆盖解析轨迹的 PVAJSC 精确度、螺旋轨迹收尾导数归零以及 `reset()` 函数行为。

3. **增量编译建议**
   - 调整控制器相关代码：`catkin build payload_mpc_controller`
   - 调整仿真器：`catkin build so3_quadrotor`
   - 调整可视化工具：`catkin build odom_visualization`

---

## 运行流程

### Step 1：控制-only 解析轨迹演示
```bash
roslaunch payload_planner controller_only.launch trajectory_mode:=helix
```
- 支持 `trajectory_mode:=circle|figure_eight|helix`；半径、角速度、中心等均可通过等名参数覆盖，若需自定义解析轨迹起点，可附加 `init_x/init_y/init_z`。
- 运行时观测：
  - `rostopic echo /mpc_controller_node/mpc/reference_trajectory`
  - `rostopic echo /mpc_controller_node/mpc/trajectory_predicted`
  - RViz 自动加载 `controller_only.rviz`，仅保留控制相关显示；`Geometry` MarkerArray 渲染当前（实际/预测）无人机、载荷及缆绳模型，参考轨迹保留为 Path 方便对比。
- 解析模式运行 10 个周期后，控制节点会在日志中输出无人机/载荷的 RMSE，并在 `payload_mpc_controller/plots/` 下生成 `analytic_xy.csv` 与 `analytic_xy.svg`（图像会通过 `xdg-open` 自动弹出）。
- 若启用外力估计器（默认开启），仿真器会额外发布 `/true_force` 真值并与 `/mpc/force` 估计值对齐；`force_data_recorder` 会在 `plots/` 目录下写出 `force_comparison.csv/.png`，终端同步打印 RMSE。
- 想要在线切换轨迹，可执行：
  ```bash
  rosparam set /mpc_controller_node/reference/mode circle
  rosparam set /mpc_controller_node/reference/use_planner false
  ```

### Step 2：规划器 + 控制器协同
```bash
roslaunch payload_planner replan.launch use_planner:=true
```
- 若置 `use_planner:=false` 可回到解析轨迹模式，并自动切换到精简 RViz 视图。
- 通过 RViz “2D Nav Goal” 下发目标；`/planning/trajectory` 与 `/mpc/...` 话题可用于验证规划器输出与 MPC 跟踪情况。

### Step 3：可视化配置
- `roslaunch payload_planner rviz.launch use_planner:=true`  
  → 全量仿真视图（含地图、障碍物等）。
- `roslaunch payload_planner rviz.launch use_planner:=false`  
  → 控制-only 视图（聚焦无人机-负载状态）。

### Step 4：外力估计 & 风场可视化

- `controller_only.launch` / `replan.launch` 会自动启动 `force_data_recorder` 节点，基于 `message_filters::Synchronizer` 对齐 `/true_force`、`/mpc/force`（`geometry_msgs::AccelStamped`）以及可选 `/so3_quadrotor/wind`（`geometry_msgs::Vector3Stamped`）。

- 运行约 10 个轨迹周期后，节点会写出 `payload_mpc_controller/plots/force_comparison.csv` 与 `force_comparison.png`，终端汇报负载/无人机的总体及分轴 RMSE；启用风场时 CSV 会追加 `wind_x/y/z` 列。

- 关键话题：`/so3_quadrotor/true_force`（线速度=负载 `fl`，角速度=无人机 `fq`）、`/mpc_controller_node/mpc/force`（估计值）、`/so3_quadrotor/wind`（当前风速）。

- 风场示例：
  ```bash
  # 无风
  roslaunch payload_planner controller_only.launch wind_type:=none
  # 恒定风（幅值/方向可覆盖）
  roslaunch payload_planner controller_only.launch \
    wind_type:=constant \
    wind_constant_velocity:=[5.0,0.0,0.0]
  # 周期阵风：正弦 / 方波（使用 wind_override_yaml 覆盖具体参数）
  roslaunch payload_planner controller_only.launch \
    wind_type:=gust \
    wind_gust_axis:=[0.0,1.0,0.0] \
    wind_override_yaml:="{gust: {mode: sine, amplitude: 0.4, frequency: 0.3}}"
  roslaunch payload_planner controller_only.launch \
    wind_type:=gust \
    wind_gust_axis:=[0.0,1.0,0.0] \
    wind_override_yaml:="{gust: {mode: square, amplitude: 0.4, frequency: 0.3, duty_cycle: 0.4}}"
  # Dryden 湍流（最小配置：直接指定 sigma）
  roslaunch payload_planner controller_only.launch \
    wind_type:=dryden \
    wind_dryden_sigma:=[0.5,0.5,0.2]
  ```

- 轨迹/风场组合速查（控制-only 模式）：
  ```bash
  # 圆轨迹 + 无风
  roslaunch payload_planner controller_only.launch trajectory_mode:=circle      wind_type:=none
  # 八字轨迹 + 恒定 1 m/s 东风（假设配置文件已给出速度/方向）
  roslaunch payload_planner controller_only.launch \
    trajectory_mode:=figure_eight \
    wind_type:=constant \
    wind_constant_velocity:=[1.0,0.0,0.0]
  # 螺旋轨迹 + 正弦阵风（自定义振幅/频率）
  roslaunch payload_planner controller_only.launch \
    trajectory_mode:=helix \
    wind_type:=gust \
    wind_gust_axis:=[0.0,1.0,0.0] \
    wind_override_yaml:="{gust: {mode: sine, amplitude: 0.6, frequency: 0.25}}"
  # 圆轨迹 + 方波阵风（占空比 40%）
  roslaunch payload_planner controller_only.launch \
    trajectory_mode:=circle \
    wind_type:=gust \
    wind_gust_axis:=[0.0,1.0,0.0] \
    wind_override_yaml:="{gust: {mode: square, amplitude: 0.4, frequency: 0.3, duty_cycle: 0.4}}"
  # 解析轨迹 + Dryden 湍流（最小配置）
  roslaunch payload_planner controller_only.launch \
    trajectory_mode:=helix \
    wind_type:=dryden \
    wind_dryden_sigma:=[0.5,0.5,0.2]
  ```

- 若希望在规划器模式下指定风场，可直接在 `replan.launch` 中附加同样的风场参数；例如：
  ```bash
  # 阵风（正弦）
  roslaunch payload_planner replan.launch \
    use_planner:=true \
    wind_type:=gust \
    wind_gust_axis:=[0.0,1.0,0.0] \
    wind_override_yaml:="{gust: {mode: sine, amplitude: 0.5, frequency: 0.2}}"

  # Dryden（最小配置）
  roslaunch payload_planner replan.launch \
    use_planner:=true \
    wind_type:=dryden \
    wind_dryden_sigma:=[0.5,0.5,0.2]
  ```

- 启动后也可通过 rosparam 在线切换轨迹与风场：
  ```bash
  rosparam set /mpc_controller_node/reference/mode figure_eight
  rosparam set /payload_planner/simulator/wind/type gust
  rosparam set /payload_planner/simulator/wind/gust/amplitude 0.3
  ```
  其中 `wind_override_yaml` 支持任意 YAML 片段，需使用双引号包裹并注意在 shell 中转义；示例：
  ```bash
  roslaunch payload_planner controller_only.launch \
    wind_type:=gust \
    wind_override_yaml:="{gust: {amplitude: 0.6, frequency: 0.3, axis: [0,1,0]}}"
  ```

- 若环境缺少 `python-dateutil` 或 GUI，脚本会自动启用轻量级兼容层，在 headless 场景仍可生成 PNG；也可使用 `rosrun payload_mpc_controller plot_force_comparison.py` 离线重绘。



---

## 验证清单
- `catkin build` 与 `reference_generator_test` 均通过。
- `controller_only.launch` 下 `/mpc/reference_trajectory` 与 `/mpc/trajectory_predicted` 在 RViz 中对齐，日志提示 `ANALYTIC trajectory tracking`。
- 运行约 10 个周期后，日志出现 `ANALYTIC RMSE (...)` 且自动生成/弹出 `analytic_xy.svg`。
- 切回 `use_planner:=true` 后日志出现 `ANALYTIC --> HOVER`，规划器话题恢复订阅。
- `roswtf` 与 `rostopic hz` 未报频率/连接异常。

---

## 常见问题 & Tips
- **解析轨迹不动**：确认 `rosparam get /mpc_controller_node/reference/use_planner` 为 `false`。
- **缺失话题**：使用 `rqt_graph` 或 `rostopic list | grep mpc` 排查发布者是否存在。
- **控制有抖动**：执行 `sudo apt install cpufrequtils` 后运行 `sudo cpufreq-set -g performance` 提升 MPC 求解性能。
- **参数恢复默认**：`rosparam load controller/payload_mpc_controller/config/mpc.yaml /mpc_controller_node`。

---

## Credits & Contact
- 代码基于 HKUST Aerial Robotics Group 原始 AutoTrans 框架，解析轨迹模块与 Stage 0-6 迭代由当前项目扩展。
- 技术交流：Haojia Li (hlied@connect.ust.hk)。欢迎提交 Issue/PR 共同完善。
