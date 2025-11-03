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
  Dryden
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

struct WindFieldConfig
{
  WindFieldType type{WindFieldType::None};
  ConstantWindConfig constant{};
  GustWindConfig gust{};
  DrydenWindConfig dryden{};
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
