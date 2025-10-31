# AGENT.md

Guidance for Codex when working inside the AutoTrans ROS workspace.

## Project Snapshot
- **AutoTrans** delivers real-time planning and NMPC control for a quadrotor carrying a suspended payload. The current focus (stages 0-6 in `task`) is enabling analytic (circle / figure-eight / helix) reference trajectories alongside the original planner.
- The workspace root is `~/autotrans_ws`; the ROS package lives in `src/AutoTrans/`.
- Key runtime artifacts: analytic runs export plots to `controller/payload_mpc_controller/plots/`, notably `analytic_xy.csv|svg`.

## Environment Setup
1. Source ROS: `source /opt/ros/${ROS_DISTRO}/setup.bash`.
2. Enter workspace: `cd ~/autotrans_ws`.
3. Build (first time or after dependency changes): `catkin build`  
   - Add `source ~/autotrans_ws/devel/setup.bash` to your shell if you work here often.
4. Before running anything, source the overlay: `source devel/setup.bash`.

## Build & Test Commands
- Full rebuild: `catkin build`.
- Controller-only rebuild: `catkin build payload_mpc_controller`.
- Analytic trajectory unit tests:  
  ```bash
  catkin build payload_mpc_controller --no-deps --catkin-make-args reference_generator_test
  devel/.private/payload_mpc_controller/lib/payload_mpc_controller/reference_generator_test
  ```
  Expect a `3/3 tests passed` summary.
- When touching ACADO-generated code, regenerate from `controller/payload_mpc_controller/model/` before rebuilding.

## Simulation & Demos
- Analytic reference showcase (controller-only RViz scene):  
  `roslaunch payload_planner controller_only.launch trajectory_mode:=helix`
  - Modes: `circle | figure_eight | helix`; override parameters via launch args (e.g., `radius`, `omega`, `init_x`).
  - After ~10 cycles you should see RMSE logs and the analytic plot saved/opened.
- Planner + controller pipeline:  
  `roslaunch payload_planner replan.launch use_planner:=true`
  - Toggle analytic mode at runtime:  
    ```
    rosparam set /mpc_controller_node/reference/use_planner false
    rosparam set /mpc_controller_node/reference/mode circle
    ```
- RViz shortcuts:  
  `roslaunch payload_planner rviz.launch use_planner:=false` → `controller_only.rviz`,  
  `use_planner:=true` → full simulation view.

## Repository Tour
- `controller/payload_mpc_controller/`
  - `src/` — MPC node, finite state machine, analytic trajectory wiring.
  - `reference_generator/` — `AnalyticTrajectory` base + circle/figure-eight/helix implementations.
  - `config/mpc.yaml` — primary runtime parameters (reference switch, MPC weights, constraints).
  - `test/reference_generator_test.cpp` — unit coverage for analytic PVAJSC sampling and reset logic.
- `planner/plan_manage/`
  - `launch/replan.launch` — planner/controller integration; conditionally includes planner stack.
  - `launch/controller_only.launch` — demo entry point for analytic trajectories.
  - `launch/rviz.launch` & `controller_only.rviz` — visualization presets.
- `uav_simulator/so3_quadrotor` — dynamics simulator that consumes SO3 commands.
- `Utils/` — shared messages, visualization helpers, RViz plugins.
- `task` — staged roadmap; check here before prioritizing work.

## Development Notes
- Preserve dual-mode behavior: analytic references must coexist with planner-fed trajectories. Any controller change should keep both `use_planner true/false` paths compiling and running.
- When editing `mpc.yaml`, `planning_params.yaml`, or analytic trajectory config, document default values and validate ROS param names against launch files.
- If you touch plotting or logging code, keep the CSV/SVG outputs stable (`analytic_xy.csv/svg`) to avoid breaking downstream analyses.
- For high-impact controller edits, capture RMSE numbers via a `controller_only.launch` run and mention them in PR/commit notes.
- Follow repo conventions: modern C++ (C++14), Eigen for math, ROS logging (`ROS_INFO_THROTTLE`, etc.), catkin-style formatting.

## Troubleshooting Cheatsheet
- Analytic trajectory not advancing: `rosparam get /mpc_controller_node/reference/use_planner` should be `false`.
- Missing MPC topics: verify `payload_mpc_controller` node is active (`rosnode info /mpc_controller_node`) and that the reference generator reset did not throw assertions.
- MPC jitter: install `cpufrequtils` and set governor to performance (`sudo cpufreq-set -g performance`).
- Planner mode regressed: relaunch `replan.launch use_planner:=true` and confirm remaps for `/planning/trajectory` and `/planning/cmd`.

Keep this file in sync with the staged goals recorded in `task` and the top-level `README.md` when workflows change.

