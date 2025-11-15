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
        last_time_(std::numeric_limits<double>::quiet_NaN()),
        generator_(config.seed ? config.seed : std::random_device{}()),
        normal_dist_(0.0, 1.0),
        airspeed_(std::max(config.bandwidth, 1e-3)),
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
      // 对于过大的时间步长，进行裁剪以保持数值稳定
      if (dt > 0.5)
      {
        dt = 0.5;
      }
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

    const double V = std::max(airspeed_, 1e-3);
    const double sqrt_dt = std::sqrt(dt);

    Eigen::Vector3d gust = Eigen::Vector3d::Zero();

    // MIL-F-8785C Dryden 模型：
    // u 轴（一阶）： G_u(s) = σ_u * sqrt(2 V / L_u) / (s + V / L_u)
    {
      const double Lu = length_scale_[0];
      const double sigma_u = std::fabs(sigma[0]);

      if (sigma_u > 1e-6 && Lu > 1e-6)
      {
        const double a_u = V / Lu;
        const double b0_u = sigma_u * std::sqrt(2.0 * V / Lu);

        // 一阶 SDE：dx = -a_u x dt + dW,  y = b0_u x
        u_state_ += (-a_u * u_state_) * dt + sqrt_dt * normal_dist_(generator_);
        gust.x() = b0_u * u_state_;
      }
      else
      {
        // 无有效湍流时仅进行指数衰减
        if (Lu > 1e-6)
        {
          const double a_u = V / Lu;
          u_state_ += (-a_u * u_state_) * dt;
        }
        else
        {
          u_state_ = 0.0;
        }
        gust.x() = 0.0;
      }
    }

    // v 轴（二阶）： G_v(s) = σ_v * sqrt(3 V / L_v) * (s + V/(√3 L_v)) / (s + V/L_v)^2
    {
      const double Lv = length_scale_[1];
      const double sigma_v = std::fabs(sigma[1]);

      double &x1 = v_state_[0];
      double &x2 = v_state_[1];

      if (sigma_v > 1e-6 && Lv > 1e-6)
      {
        const double a_v = V / Lv;
        const double z_v = V / (std::sqrt(3.0) * Lv);
        const double b0_v = sigma_v * std::sqrt(3.0 * V / Lv);

        const double dx1 = x2;
        const double dx2 = -a_v * a_v * x1 - 2.0 * a_v * x2;

        x1 += dx1 * dt;
        x2 += dx2 * dt + sqrt_dt * normal_dist_(generator_);

        gust.y() = b0_v * (z_v * x1 + x2);
      }
      else
      {
        if (Lv > 1e-6)
        {
          const double a_v = V / Lv;
          const double dx1 = x2;
          const double dx2 = -a_v * a_v * x1 - 2.0 * a_v * x2;
          x1 += dx1 * dt;
          x2 += dx2 * dt;
        }
        else
        {
          x1 = 0.0;
          x2 = 0.0;
        }
        gust.y() = 0.0;
      }
    }

    // w 轴（二阶）： G_w(s) = σ_w * sqrt(3 V / L_w) * (s + V/(√3 L_w)) / (s + V/L_w)^2
    {
      const double Lw = length_scale_[2];
      const double sigma_w = std::fabs(sigma[2]);

      double &x1 = w_state_[0];
      double &x2 = w_state_[1];

      if (sigma_w > 1e-6 && Lw > 1e-6)
      {
        const double a_w = V / Lw;
        const double z_w = V / (std::sqrt(3.0) * Lw);
        const double b0_w = sigma_w * std::sqrt(3.0 * V / Lw);

        const double dx1 = x2;
        const double dx2 = -a_w * a_w * x1 - 2.0 * a_w * x2;

        x1 += dx1 * dt;
        x2 += dx2 * dt + sqrt_dt * normal_dist_(generator_);

        gust.z() = b0_w * (z_w * x1 + x2);
      }
      else
      {
        if (Lw > 1e-6)
        {
          const double a_w = V / Lw;
          const double dx1 = x2;
          const double dx2 = -a_w * a_w * x1 - 2.0 * a_w * x2;
          x1 += dx1 * dt;
          x2 += dx2 * dt;
        }
        else
        {
          x1 = 0.0;
          x2 = 0.0;
        }
        gust.z() = 0.0;
      }
    }

    // 保存最近一次输出，便于 dt<=0 时复用
    state_ = gust;
    return gust;
  }

private:
  DrydenWindConfig config_;
  mutable double last_time_;
  mutable std::mt19937 generator_;
  mutable std::normal_distribution<double> normal_dist_;
  double airspeed_{1.0};
  double length_scale_[3];
  // Dryden 滤波器状态：u 轴一阶，v/w 轴二阶
  mutable Eigen::Vector3d state_{Eigen::Vector3d::Zero()};
  mutable double u_state_{0.0};
  mutable Eigen::Vector2d v_state_{Eigen::Vector2d::Zero()};
  mutable Eigen::Vector2d w_state_{Eigen::Vector2d::Zero()};
};

class CompositeWindField : public WindFieldBase
{
public:
  CompositeWindField(const ConstantWindConfig &constant_config, const DrydenWindConfig &dryden_config)
      : constant_velocity_(constant_config.velocity),
        dryden_field_(dryden_config)
  {
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const override
  {
    // Dryden component produces turbulence around the steady bias.
    return constant_velocity_ + dryden_field_.sample(position_world, time_sec);
  }

private:
  Eigen::Vector3d constant_velocity_{Eigen::Vector3d::Zero()};
  DrydenWindField dryden_field_;
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
  if (lower == "composite" || lower == "dryden_composite")
  {
    return WindFieldType::Composite;
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
  case WindFieldType::Composite:
    return "composite";
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
  case WindFieldType::Composite:
    return std::make_unique<CompositeWindField>(config.constant, config.dryden);
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
