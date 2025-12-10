# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoTrans is a real-time planning and control framework for quadrotor UAVs with suspended payloads. It consists of three main components:
1. **Motion Planner** - Generates dynamic feasible trajectories considering time-varying payload shape
2. **NMPC Controller** - Disturbance-aware Nonlinear Model Predictive Control using ACADO
3. **UAV Simulator** - Simulates quadrotor dynamics with suspended payload

## Build System

This is a ROS (Robot Operating System) Catkin workspace.

### Build Commands
```bash
# Build the entire workspace (first time)
cd ~/autotrans_ws
catkin build
source devel/setup.bash

# Incremental builds for specific packages
catkin build payload_mpc_controller      # Controller changes
catkin build so3_quadrotor                # Simulator changes
catkin build payload_planner              # Planner changes
```

### Prerequisites
- ROS Noetic (or compatible distribution)
- `sudo apt install ros-"${ROS_DISTRO}"-mavros-msgs python3-setuptools`

### Running Tests
```bash
# Build and run reference generator unit tests
catkin build payload_mpc_controller --no-deps --catkin-make-args reference_generator_test
devel/.private/payload_mpc_controller/lib/payload_mpc_controller/reference_generator_test
```

### Running the Simulation

**Controller-only mode (analytic trajectories):**
```bash
roslaunch payload_planner controller_only.launch trajectory_mode:=helix
# Options: trajectory_mode:=circle|figure_eight|helix
```

**Full simulation mode (planner + controller):**
```bash
roslaunch payload_planner replan.launch use_planner:=true
# Use "2D Nav Goal" in RViz to set goal positions
```

**Simple run (legacy):**
```bash
roslaunch payload_planner simple_run.launch
```

## Architecture

### Directory Structure

- `planner/` - Motion planning modules
  - `plan_manage/` - High-level planning management, FSM, and main planning node
  - `traj_opt/` - Trajectory optimization using MINCO (Minimum Control Effort)
  - `path_searching/` - Kinodynamic A* for initial path search
  - `plan_env/` - Environment representation (grid map, raycast)
  - `traj_utils/` - Trajectory utilities and data structures

- `controller/` - NMPC controller
  - `payload_mpc_controller/` - MPC wrapper around ACADO-generated code
    - `model/` - ACADO model generation code (requires separate compilation)
    - `externals/qpoases/` - QP solver library used by ACADO
    - `test/` - Unit tests for reference generator
    - `scripts/` - Plotting scripts (plot_xy.py)
    - `plots/` - Output directory for trajectory visualizations

- `uav_simulator/` - Simulation components
  - `so3_quadrotor/` - SO(3) based quadrotor dynamics simulator
  - `mockamap/` - Mock environment generator
  - `map_generator/` - Random forest map generator
  - `local_sensing/` - Sensor simulation (depth camera, point cloud)

- `Utils/` - Shared utilities
  - `quadrotor_msgs/` - ROS message definitions for quadrotor control
  - `uav_utils/` - General UAV utility functions
  - `odom_visualization/` - Visualization tools for odometry
  - `rviz_plugins/` - Custom RViz plugins

### Key Components

#### Planning Pipeline
1. **ReFSM** (planner_manager.h:30): Main replanning interface `reboundReplan()`
2. **KinodynamicAstar** (kinodynamic_astar.h): Initial path search considering dynamics
3. **PolyTrajOptimizer** (poly_traj_optimizer.h): MINCO-based trajectory optimization
   - Handles payload geometry constraints
   - Enforces quadrotor dynamic feasibility
   - Uses L-BFGS optimization
4. **TrajContainer** (plan_container.hpp): Stores trajectory state

#### Control Pipeline
1. **MpcWrapper** (mpc_wrapper.h): Interface to ACADO-generated MPC
   - State size: `kStateSize` (ACADO_NX) - quadrotor + payload states
   - Input size: `kInputSize` (ACADO_NU) - thrust + body rates
   - Samples: `kSamples` (ACADO_N = 20 by default)
2. **ReferenceGenerator** (reference_generator.h): Generates analytic reference trajectories
   - CircleTrajectory: Circular motion with configurable radius and angular velocity
   - FigureEightTrajectory: Lissajous figure-eight pattern
   - HelixTrajectory: Spiral motion with vertical climb
3. **ForceEstimator** (force_estimator.hpp): Online disturbance estimation
4. Outputs SO3Command to quadrotor simulator

### Configuration Files

- `planner/plan_manage/config/planning_params.yaml` - Planning weights, system parameters (masses, cable length), search parameters
- `controller/payload_mpc_controller/config/mpc.yaml` - MPC cost matrices, input limits, force estimator settings, reference trajectory parameters
- `controller/payload_mpc_controller/config/model.yaml` - Dynamic model parameters (masses, cable length, gravity, MPC horizon)

**IMPORTANT**: If you modify `model.yaml`, you must regenerate the ACADO MPC code (see below).

## Development Workflows

### Modifying MPC Model Parameters

The MPC uses auto-generated code from ACADO. If you change model parameters in `config/model.yaml`:

1. Install ACADO: http://acado.github.io/install_linux.html
2. Source ACADO environment: `source ACADO_ROOT/build/acado_env.sh`
3. Navigate to model directory: `cd controller/payload_mpc_controller/model`
4. Build code generator: `cmake . && make`
5. Generate MPC code: `./quadrotor_payload_mpc_with_ext_force_codegen ../config/model.yaml`
6. Rebuild the workspace: `catkin build`

### Tuning Planning Parameters

Key parameters in `planning_params.yaml`:
- `weight_obstacle`, `weight_feasibility`, `weight_quad_feasibility` - Trajectory optimization weights
- `payload_size`, `drone_size`, `L_length` - Physical dimensions
- `mass_payload`, `mass_quad` - System masses (must match `model.yaml`)
- `max_theta` - Maximum cable angle constraint (degrees)

### Tuning Controller Parameters

Key parameters in `mpc.yaml`:
- `Q_pos_xy/z`, `Q_velocity`, `Q_payload_xy/z` - State tracking costs
- `R_thrust`, `R_pitchroll`, `R_yaw` - Input costs
- `max_bodyrate_xy/z`, `min_thrust`, `max_thrust` - Input constraints
- `force_estimator.use_force_estimator` - Enable/disable disturbance estimation
- `reference.use_planner` - Switch between planner and analytic trajectories
- `reference.mode` - Select analytic trajectory type (circle, figure_eight, helix)

### Switching Between Planner and Analytic Trajectories

**At launch time:**
```bash
roslaunch payload_planner replan.launch use_planner:=false trajectory_mode:=circle
```

**At runtime:**
```bash
rosparam set /mpc_controller_node/reference/use_planner false
rosparam set /mpc_controller_node/reference/mode helix
```

### Analytic Trajectory Outputs

When running in analytic trajectory mode, the controller:
- Publishes reference and predicted trajectories to `/mpc_controller_node/mpc/reference_trajectory` and `/mpc_controller_node/mpc/trajectory_predicted`
- Computes and logs RMSE after ~10 cycles
- Generates CSV and PNG plots in `controller/payload_mpc_controller/plots/analytic_xy.*`
- Auto-opens SVG visualization with `xdg-open`

## Important Notes

- **Trajectory Representation**: Uses MINCO (Minimum Control Effort) trajectories for planning, not standard polynomials
- **Coordinate Frames**: World frame (planning) -> Body frame (control). Payload state is tracked separately.
- **MPC Horizon**: Default 20 steps × 0.041s = 0.82s prediction horizon
- **External Forces**: MPC compensates for estimated disturbances (e.g., rotor drag) via force_estimator
- **Flatness Map** (flatness.hpp): Converts quadrotor-payload flat outputs to full system states
- **Performance Optimization**: Use `sudo cpufreq-set -g performance` to boost MPC solve speed
- **Dual RViz Configs**: `sim_vis.rviz` for full simulation, `controller_only.rviz` for analytic trajectories

## ROS Topics

Key topics:
- `/visual_slam/odom` - Quadrotor odometry input
- `/payload_odom` - Payload odometry input
- `/planning/trajectory` - Published trajectory for controller (planner mode)
- `/mpc_controller_node/mpc/reference_trajectory` - Reference trajectory (analytic mode)
- `/mpc_controller_node/mpc/trajectory_predicted` - MPC predicted trajectory
- `/so3cmd` or `/mavros/setpoint_raw/attitude` - SO(3) control command output
- `/move_base_simple/goal` - Goal position trigger (use RViz "2D Nav Goal")

## Testing

The simulation environment includes:
- Random forest obstacle generation
- Depth camera simulation
- Point cloud rendering
- Full quadrotor-payload dynamics
- Unit tests for analytic trajectory derivatives and behavior
