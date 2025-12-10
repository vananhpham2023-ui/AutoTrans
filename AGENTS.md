# Repository Guidelines

## Project Structure & Module Organization
- Root: `src/AutoTrans` (catkin multi-package). Key packages:
  - `controller/payload_mpc_controller` — MPC controller, test target and plots.
  - `planner/{plan_manage, plan_env, path_searching, traj_opt, traj_utils}` — planning stack; launch files in `plan_manage/launch/`.
  - `uav_simulator/so3_quadrotor` — quadrotor + payload simulator (nodelet).
  - `Utils/{uav_utils, odom_visualization}` — shared utilities and RViz helpers.
- Tooling/scripts: `batch_data_collection.py`, `quick_test.sh`, `diagnose.sh`, `check_topics.sh`.
- Example launch: `test_simulator.launch` (smoke test).

## Build, Test, and Development Commands
- First build: `catkin build` → `source devel/setup.bash` (add to `~/.bashrc` for convenience).
- Incremental builds (examples):
  - `catkin build payload_mpc_controller`
  - `catkin build so3_quadrotor`
- Run locally:
  - Control-only: `roslaunch payload_planner controller_only.launch trajectory_mode:=circle`
  - Planner + control: `roslaunch payload_planner replan.launch use_planner:=true`
- Unit test (Stage 6):
  - `catkin build payload_mpc_controller --no-deps --catkin-make-args reference_generator_test`
  - `devel/.private/payload_mpc_controller/lib/payload_mpc_controller/reference_generator_test` (expect 3/3 PASS)
- Diagnostics:
  - Smoke test: `bash src/AutoTrans/quick_test.sh`
  - Topic/node checks: `bash src/AutoTrans/diagnose.sh` or `bash src/AutoTrans/check_topics.sh`

## Coding Style & Naming Conventions
- C++ (catkin): follow existing style; prefer Eigen; keep headers under `include/`. Use `clang-format` if configured; otherwise match nearby files.
- Python: 4-space indentation, PEP 8, snake_case filenames (e.g., `data_tools.py`).
- ROS: topics/params in `snake_case`; launch files in `snake_case.launch`.
- Do not commit generated artifacts under `plots/` or large binaries.

## Testing Guidelines
- Keep `reference_generator_test` green; validate trajectories and reset behavior before PRs.
- When adding tests, prefer `catkin_add_gtest`/rostest and colocate under the package.
- Provide minimal `roslaunch` commands used to verify changes.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject; scope prefix helpful (e.g., `planner: fix yaw sampling`). English or Chinese ok; keep body to explain what/why.
- PRs must include:
  - Summary, rationale, and affected packages
  - Repro steps (exact `roslaunch`/params) and logs/screenshots (RViz, plots)
  - Build status (`catkin build` OK) and test results

## Environment Tips
- Target Ubuntu 18.04/20.04 + ROS Noetic. Ensure `mavros-msgs`, OpenCV/PCL/Eigen are installed. Always `source devel/setup.bash` before running.
