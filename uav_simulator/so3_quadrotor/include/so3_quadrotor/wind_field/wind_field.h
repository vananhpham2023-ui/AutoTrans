#pragma once

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

constexpr double kTwoPi = 2.0 * M_PI;

namespace ros
{
class NodeHandle;
} // namespace ros

namespace so3_quadrotor
{
namespace wind
{

enum class WindFieldType
{
  None = 0,
  Constant,
  Gust,
  GustEvent, // deterministic, time-bounded gust
  Gradual,   // time-bounded ramp of mean wind
  Dryden,
  Composite
};

enum class GustMode
{
  Sine = 0,
  Square
};

struct ConstantWindConfig
{
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
};

struct GustWindConfig
{
  GustMode mode{GustMode::Sine};
  Eigen::Vector3d axis{Eigen::Vector3d::UnitX()};
  double amplitude{0.0};
  double frequency{0.1};
  double bias{0.0};
  double duty_cycle{0.5};
  double phase{0.0};
};

struct DrydenWindConfig
{
  double reference_height{10.0};
  Eigen::Vector3d sigma{Eigen::Vector3d::Zero()};
  Eigen::Vector3d length_scale{Eigen::Vector3d::Constant(50.0)};
  double bandwidth{1.0};
  unsigned int seed{0u};
};

struct GustEventConfig
{
  // Only horizontal axes are used; z is ignored for safety.
  Eigen::Vector3d axis{Eigen::Vector3d::UnitX()};
  double magnitude{0.0};      // m/s peak increment (stage 1 & 3 plateau)
  double start_time{0.0};     // seconds, relative to analytic timeline (stage 1 start)
  double start_time_offset{0.0}; // seconds, global offset applied in factory (e.g., warmup_trim)
  double hold_duration{0.0};  // seconds at full magnitude for stage 1 & 3
  double rise_time{0.0};      // seconds to ramp from 0 -> magnitude (stage 1)
  double fall_time{0.0};      // seconds to ramp from magnitude -> 0 (stage 4)

  // Optional multi-stage extension (stage 2 & 3 interpolation target).
  // When enabled, the gust vector is:
  //   A1 = axis * magnitude
  //   A2 = axis2 * magnitude2
  // and evolves through four phases:
  //   1) 0 -> A1  (rise_time, then hold_duration)
  //   2) A1 -> A2 (rise_time2, then hold_duration2)
  //   3) A2 -> A1 (fall_time2, then hold_duration)
  //   4) A1 -> 0  (fall_time)
  // If magnitude2 equals magnitude and axis2 aligns with axis, the
  // behaviour reduces to the classic single-stage gust.
  Eigen::Vector3d axis2{Eigen::Vector3d::UnitX()};
  double magnitude2{0.0};
  double hold_duration2{0.0};
  double rise_time2{0.0};
  double fall_time2{0.0};
  bool multi_stage{false};
};

struct GradualWindConfig
{
  bool enabled{false};     // gate the gradual envelope on/off
  double start_time{0.0};  // seconds, absolute sim time (relative if offset at ctor)
  double stop_time{0.0};   // seconds, absolute sim time
  // Optional cycle-aware ramp profile (e.g., ramp up in cycle 2, hold in 3-8, ramp down in 9).
  // When enabled, ramp windows derive from the wind window start/stop (or gradual start/stop if no window).
  // Ramp durations should be very small (e.g., < controller step_T/10) to approximate an instant switch.
  bool cycle_profile_enabled{false};
  double ramp_up_time{0.0};    // seconds
  double ramp_down_time{0.0};  // seconds
};

struct WindWindowConfig
{
  bool enabled{false};      // gate the entire wind field on/off
  bool auto_from_cycle{false};
  int skip_cycles{0};       // number of initial cycles with wind off
  int active_cycles{0};     // number of cycles with wind on
  double cycle_time{0.0};   // seconds per cycle (if auto_from_cycle)
  double start_time{0.0};   // manual start time (s)
  double stop_time{0.0};    // manual stop time (s)
};

struct WindFieldConfig
{
  WindFieldType type{WindFieldType::None};
  ConstantWindConfig constant{};
  GustWindConfig gust{};
  DrydenWindConfig dryden{};
  GustEventConfig gust_event{};
  GradualWindConfig gradual{};
  WindWindowConfig window{};
};

class WindFieldBase
{
public:
  virtual ~WindFieldBase() = default;
  virtual Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const = 0;
};

WindFieldType typeFromString(const std::string &type);
std::string typeToString(WindFieldType type);
GustMode gustModeFromString(const std::string &mode);
std::string gustModeToString(GustMode mode);

WindFieldConfig loadWindFieldConfig(const ros::NodeHandle &nh);

std::unique_ptr<WindFieldBase> createWindField(const ros::NodeHandle &nh);
std::unique_ptr<WindFieldBase> createWindField(const WindFieldConfig &config);

} // namespace wind
} // namespace so3_quadrotor
