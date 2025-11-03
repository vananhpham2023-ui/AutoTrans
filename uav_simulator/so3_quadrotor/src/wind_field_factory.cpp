#include "so3_quadrotor/wind_field/wind_field.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <ros/console.h>
#include <ros/node_handle.h>

namespace so3_quadrotor
{
namespace wind
{

namespace
{

Eigen::Vector3d readVector3(const ros::NodeHandle &nh, const std::string &key, const Eigen::Vector3d &default_value)
{
  Eigen::Vector3d result = default_value;
  std::vector<double> data;
  if (nh.getParam(key, data))
  {
    if (data.size() == 3u)
    {
      result = Eigen::Vector3d(data[0], data[1], data[2]);
    }
    else
    {
      ROS_WARN_STREAM("Parameter '" << nh.resolveName(key) << "' requires 3 elements, found " << data.size()
                                    << ". Using default value.");
    }
  }
  return result;
}

std::string normalizeString(const std::string &input)
{
  std::string lower = input;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lower;
}

class NoneWindField : public WindFieldBase
{
public:
  Eigen::Vector3d sample(const Eigen::Vector3d &, double) const override
  {
    return Eigen::Vector3d::Zero();
  }
};

class ConstantWindField : public WindFieldBase
{
public:
  explicit ConstantWindField(const ConstantWindConfig &config) : velocity_(config.velocity) {}

  Eigen::Vector3d sample(const Eigen::Vector3d &, double) const override
  {
    return velocity_;
  }

private:
  Eigen::Vector3d velocity_;
};

class PeriodicGustWindField : public WindFieldBase
{
public:
  explicit PeriodicGustWindField(const GustWindConfig &config)
      : axis_(config.axis.normalized()),
        amplitude_(std::fabs(config.amplitude)),
        frequency_(std::fabs(config.frequency)),
        bias_(config.bias),
        duty_cycle_(config.duty_cycle),
        phase_(config.phase),
        mode_(config.mode)
  {
    if (!std::isfinite(frequency_) || frequency_ < 1e-6)
    {
      frequency_ = 0.0;
    }
    if (axis_.norm() < 1e-6)
    {
      axis_ = Eigen::Vector3d::UnitX();
    }
    if (duty_cycle_ < 0.0)
    {
      duty_cycle_ = 0.0;
    }
    else if (duty_cycle_ > 1.0)
    {
      duty_cycle_ = 1.0;
    }
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &, double time_sec) const override
  {
    const double wave = bias_ + waveform(time_sec);
    return axis_ * wave;
  }

private:
  double waveform(double time_sec) const
  {
    if (amplitude_ < 1e-6 || frequency_ <= 0.0)
    {
      return 0.0;
    }

    if (mode_ == GustMode::Square)
    {
      const double period = 1.0 / frequency_;
      const double offset = phase_ / (kTwoPi * frequency_);
      double t = std::fmod(time_sec + offset, period);
      if (t < 0.0)
      {
        t += period;
      }
      const double high_duration = duty_cycle_ * period;
      const bool high = t < high_duration;
      return high ? amplitude_ : -amplitude_;
    }
    else
    {
      return amplitude_ * std::sin(kTwoPi * frequency_ * time_sec + phase_);
    }
  }

  Eigen::Vector3d axis_;
  double amplitude_{0.0};
  double frequency_{0.0};
  double bias_{0.0};
  double duty_cycle_{0.5};
  double phase_{0.0};
  GustMode mode_{GustMode::Sine};
};

class DrydenWindField : public WindFieldBase
{
public:
  explicit DrydenWindField(const DrydenWindConfig &config)
      : config_(config),
        state_(Eigen::Vector3d::Zero()),
        last_time_(std::numeric_limits<double>::quiet_NaN()),
        generator_(config.seed ? config.seed : std::random_device{}()),
        normal_dist_(0.0, 1.0),
        base_bandwidth_(std::max(config.bandwidth, 1e-3)),
        length_scale_{1.0, 1.0, 1.0}
  {
    for (int i = 0; i < 3; ++i)
    {
      length_scale_[i] = std::max(config_.length_scale[i], 1e-3);
    }
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const override
  {
    double dt = 0.0;
    if (std::isfinite(last_time_))
    {
      dt = std::max(0.0, time_sec - last_time_);
    }
    last_time_ = time_sec;

    if (dt <= 0.0)
    {
      return state_;
    }

    Eigen::Vector3d sigma = config_.sigma;
    if (config_.reference_height > 1e-3)
    {
      const double height = std::max(0.0, position_world.z());
      const double attenuation = std::exp(-height / config_.reference_height);
      sigma *= attenuation;
    }

    for (int i = 0; i < 3; ++i)
    {
      const double tau = length_scale_[i] / base_bandwidth_;
      const double alpha = std::exp(-dt / tau);
      const double sigma_abs = std::fabs(sigma[i]);
      if (sigma_abs < 1e-6)
      {
        state_[i] = alpha * state_[i];
        continue;
      }
      const double gain = sigma_abs * std::sqrt(std::max(0.0, 1.0 - alpha * alpha));
      state_[i] = alpha * state_[i] + gain * normal_dist_(generator_);
    }

    return state_;
  }

private:
  DrydenWindConfig config_;
  mutable Eigen::Vector3d state_{Eigen::Vector3d::Zero()};
  mutable double last_time_;
  mutable std::mt19937 generator_;
  mutable std::normal_distribution<double> normal_dist_;
  double base_bandwidth_{1.0};
  double length_scale_[3];
};

} // namespace

WindFieldType typeFromString(const std::string &type)
{
  std::string lower = normalizeString(type);
  if (lower == "constant")
  {
    return WindFieldType::Constant;
  }
  if (lower == "gust")
  {
    return WindFieldType::Gust;
  }
  if (lower == "dryden")
  {
    return WindFieldType::Dryden;
  }
  return WindFieldType::None;
}

std::string typeToString(WindFieldType type)
{
  switch (type)
  {
  case WindFieldType::Constant:
    return "constant";
  case WindFieldType::Gust:
    return "gust";
  case WindFieldType::Dryden:
    return "dryden";
  default:
    return "none";
  }
}

GustMode gustModeFromString(const std::string &mode)
{
  std::string lower = normalizeString(mode);
  if (lower == "square")
  {
    return GustMode::Square;
  }
  return GustMode::Sine;
}

std::string gustModeToString(GustMode mode)
{
  switch (mode)
  {
  case GustMode::Square:
    return "square";
  default:
    return "sine";
  }
}

WindFieldConfig loadWindFieldConfig(const ros::NodeHandle &nh)
{
  WindFieldConfig config;

  std::string type_str;
  nh.param<std::string>("type", type_str, "none");
  config.type = typeFromString(type_str);

  // constant parameters
  config.constant.velocity = readVector3(nh, "constant/velocity", Eigen::Vector3d::Zero());
  if (config.constant.velocity.isZero())
  {
    double speed = nh.param("constant/speed", 0.0);
    Eigen::Vector3d direction = readVector3(nh, "constant/direction", Eigen::Vector3d::UnitX());
    if (direction.norm() > 1e-6)
    {
      config.constant.velocity = direction.normalized() * speed;
    }
  }

  // gust parameters
  std::string gust_mode;
  nh.param<std::string>("gust/mode", gust_mode, "sine");
  config.gust.mode = gustModeFromString(gust_mode);
  config.gust.axis = readVector3(nh, "gust/axis", Eigen::Vector3d::UnitX());
  if (config.gust.axis.norm() < 1e-6)
  {
    config.gust.axis = Eigen::Vector3d::UnitX();
  }
  else
  {
    config.gust.axis.normalize();
  }
  config.gust.amplitude = nh.param("gust/amplitude", 0.0);
  config.gust.frequency = nh.param("gust/frequency", 0.1);
  config.gust.bias = nh.param("gust/bias", 0.0);
  config.gust.duty_cycle = nh.param("gust/duty_cycle", 0.5);
  config.gust.phase = nh.param("gust/phase", 0.0);

  // dryden parameters
  config.dryden.reference_height = nh.param("dryden/reference_height", 10.0);
  config.dryden.sigma = readVector3(nh, "dryden/sigma", Eigen::Vector3d::Zero());
  config.dryden.length_scale = readVector3(nh, "dryden/length_scale", Eigen::Vector3d::Constant(50.0));
  config.dryden.bandwidth = nh.param("dryden/bandwidth", 1.0);
  int seed = nh.param("dryden/seed", 0);
  config.dryden.seed = static_cast<unsigned int>(seed);

  return config;
}

std::unique_ptr<WindFieldBase> createWindField(const WindFieldConfig &config)
{
  switch (config.type)
  {
  case WindFieldType::Constant:
    return std::make_unique<ConstantWindField>(config.constant);
  case WindFieldType::Gust:
    return std::make_unique<PeriodicGustWindField>(config.gust);
  case WindFieldType::Dryden:
    return std::make_unique<DrydenWindField>(config.dryden);
  default:
    return std::make_unique<NoneWindField>();
  }
}

std::unique_ptr<WindFieldBase> createWindField(const ros::NodeHandle &nh)
{
  WindFieldConfig config = loadWindFieldConfig(nh);
  ROS_INFO_STREAM("[wind_field] Loaded configuration type=" << typeToString(config.type));
  return createWindField(config);
}

} // namespace wind
} // namespace so3_quadrotor
