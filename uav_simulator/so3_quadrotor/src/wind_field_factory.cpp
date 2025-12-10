#include "so3_quadrotor/wind_field/wind_field.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <ros/console.h>
#include <ros/node_handle.h>
#include <ros/time.h>

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

double inferCycleTime(const ros::NodeHandle &nh)
{
  // Try to stay consistent with plotting logic that derives cycle from angular velocity.
  const std::vector<std::string> omega_keys = {
      "figure_omega",
      "circle_omega",
      "helix_omega",
      "reference/figure_eight/angular_velocity",
      "reference/circle/angular_velocity",
      "reference/helix/angular_velocity",
      "/figure_omega",
      "/circle_omega",
      "/helix_omega",
      "/mpc_controller_node/reference/figure_eight/angular_velocity",
      "/mpc_controller_node/reference/circle/angular_velocity",
      "/mpc_controller_node/reference/helix/angular_velocity"};

  ros::NodeHandle nh_global;
  double omega = 0.0;
  for (const auto &key : omega_keys)
  {
    if (nh.getParam(key, omega) || nh_global.getParam(key, omega))
    {
      if (std::fabs(omega) > 1e-6)
      {
        return kTwoPi / std::fabs(omega);
      }
    }
  }
  return 0.0;
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

class GustEventWindField : public WindFieldBase
{
public:
  GustEventWindField(const GustEventConfig &config,
                     const DrydenWindConfig &dryden_config,
                     const WindWindowConfig &window_config)
      : dryden_field_(dryden_config),
        window_(window_config)
  {
    time_offset_ = ros::Time::now().toSec();
    axis_ = config.axis;
    if (axis_.norm() < 1e-6)
    {
      axis_ = Eigen::Vector3d::UnitX();
    }
    axis_.z() = 0.0;
    axis_.normalize();
    magnitude_ = std::isfinite(config.magnitude) ? config.magnitude : 0.0;
    axis2_ = config.axis2;
    if (axis2_.norm() < 1e-6)
    {
      axis2_ = axis_;
    }
    axis2_.z() = 0.0;
    axis2_.normalize();
    magnitude2_ = std::isfinite(config.magnitude2) ? config.magnitude2 : magnitude_;
    const double analytic_offset = std::max(0.0, config.start_time_offset);
    start_ = config.start_time + analytic_offset + time_offset_;
    hold_ = config.hold_duration;
    rise_ = config.rise_time;
    fall_ = config.fall_time;
    hold2_ = config.hold_duration2;
    rise2_ = config.rise_time2;
    fall2_ = config.fall_time2;
    multi_stage_ = config.multi_stage;

    // Resolve window times (auto or manual) to align gust_event with the same
    // start/stop semantics as the Gradual wind field.
    if (window_.enabled && window_.auto_from_cycle && window_.active_cycles > 0 && window_.cycle_time > 1e-6)
    {
      window_.start_time = static_cast<double>(window_.skip_cycles) * window_.cycle_time;
      window_.stop_time = static_cast<double>(window_.skip_cycles + window_.active_cycles) * window_.cycle_time;
    }
    if (window_.enabled)
    {
      window_.start_time += time_offset_ + analytic_offset;
      window_.stop_time += time_offset_ + analytic_offset;
    }
    if (window_.enabled && window_.stop_time <= window_.start_time)
    {
      ROS_WARN("[wind_field] GustEvent window enabled but stop_time <= start_time (start=%.3f, stop=%.3f). Disabling window.",
               window_.start_time, window_.stop_time);
      window_.enabled = false;
    }
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const override
  {
    if (multi_stage_)
    {
      const double gate = windowGate(time_sec);
      if (gate <= 0.0 || magnitude_ <= 1e-9)
      {
        return Eigen::Vector3d::Zero();
      }

      const double e1 = primaryEnvelope(time_sec);
      const double e2 = secondaryEnvelope(time_sec);

      Eigen::Vector3d wind = Eigen::Vector3d::Zero();
      if (e1 > 1e-9)
      {
        wind = dryden_field_.sample(position_world, time_sec) * e1;
      }

      const Eigen::Vector3d A1 = axis_ * magnitude_;
      const Eigen::Vector3d A2 = axis2_ * magnitude2_;
      const Eigen::Vector3d gust = A1 * e1 + (A2 - A1) * e2;
      return gate * (wind + gust);
    }
    else
    {
      const double scale = envelope(time_sec);
      Eigen::Vector3d wind = Eigen::Vector3d::Zero();
      if (scale > 1e-9)
      {
        wind = dryden_field_.sample(position_world, time_sec) * scale;
        wind += axis_ * magnitude_ * scale; // Dryden + deterministic gust share the same envelope
      }
      return wind;
    }
  }

private:
  // Primary envelope e1(t): controls overall on/off of the gust and Dryden.
  //  - Stage 1: 0 -> 1 (rise_)
  //  - Stage 1 hold: 1 for hold_
  //  - Stage 2 & 3: stays at 1 while transitioning between A1/A2
  //  - Stage 4: 1 -> 0 (fall_)
  double primaryEnvelope(double time_sec) const
  {
    if (rise_ <= 1e-6 && fall_ <= 1e-6)
    {
      // Degenerate: treat as ideal step between start_ and the end of stage 3 hold.
      const double t_on = start_;
      const double t_off = start_ + hold_ + rise2_ + hold2_ + fall2_ + hold_;
      if (time_sec < t_on || time_sec >= t_off)
      {
        return 0.0;
      }
      return 1.0;
    }

    const double t0 = start_;
    const double t1 = t0 + rise_;
    const double t3 = t0 + rise_ + hold_ + rise2_ + hold2_ + fall2_ + hold_;
    const double t4 = t3 + fall_;

    if (time_sec < t0)
    {
      return 0.0;
    }
    if (time_sec < t1)
    {
      const double denom = std::max(rise_, 1e-6);
      return std::min(1.0, std::max(0.0, (time_sec - t0) / denom));
    }
    if (time_sec < t3)
    {
      return 1.0;
    }
    if (time_sec < t4)
    {
      const double denom = std::max(fall_, 1e-6);
      return std::min(1.0, std::max(0.0, 1.0 - (time_sec - t3) / denom));
    }
    return 0.0;
  }

  // Secondary envelope e2(t): drives the interpolation A1 -> A2 -> A1.
  double secondaryEnvelope(double time_sec) const
  {
    if (rise2_ <= 1e-6 && fall2_ <= 1e-6 && hold2_ <= 0.0)
    {
      return 0.0;
    }

    const double t2_start = start_ + rise_ + hold_;
    const double t2_rise_end = t2_start + rise2_;
    const double t2_hold_end = t2_rise_end + hold2_;
    const double t2_fall_end = t2_hold_end + fall2_;

    if (time_sec < t2_start || time_sec >= t2_fall_end)
    {
      return 0.0;
    }
    if (time_sec < t2_rise_end)
    {
      const double denom = std::max(rise2_, 1e-6);
      return std::min(1.0, std::max(0.0, (time_sec - t2_start) / denom));
    }
    if (time_sec < t2_hold_end)
    {
      return 1.0;
    }
    // fall back from 1 -> 0
    const double denom = std::max(fall2_, 1e-6);
    return std::min(1.0, std::max(0.0, 1.0 - (time_sec - t2_hold_end) / denom));
  }

  double envelope(double time_sec) const
  {
    // First apply optional window gating (shared semantics with Gradual wind).
    const double gate = windowGate(time_sec);
    if (gate <= 0.0)
    {
      return 0.0;
    }

    // Then apply the intrinsic gust_event timing (rise/hold/fall).
    if (magnitude_ <= 1e-9)
    {
      return 0.0;
    }
    const double t = time_sec;
    const double up_end = start_ + rise_;
    const double hold_end = up_end + hold_;
    const double down_end = hold_end + fall_;

    if (t < start_)
    {
      return 0.0;
    }
    if (t < up_end)
    {
      const double denom = std::max(rise_, 1e-6);
      return gate * std::min(1.0, std::max(0.0, (t - start_) / denom));
    }
    if (t < hold_end)
    {
      return gate * 1.0;
    }
    if (t < down_end)
    {
      const double denom = std::max(fall_, 1e-6);
      return gate * std::min(1.0, std::max(0.0, 1.0 - (t - hold_end) / denom));
    }
    return 0.0;
  }

  double windowGate(double time_sec) const
  {
    if (!window_.enabled)
    {
      return 1.0;
    }
    if (window_.stop_time > window_.start_time && (time_sec < window_.start_time || time_sec >= window_.stop_time))
    {
      return 0.0;
    }
    return 1.0;
  }

  DrydenWindField dryden_field_;
  Eigen::Vector3d axis_{Eigen::Vector3d::UnitX()};
  double magnitude_{0.0};
  Eigen::Vector3d axis2_{Eigen::Vector3d::UnitX()};
  double magnitude2_{0.0};
  double start_{0.0};
  double hold_{0.0};
  double rise_{0.0};
  double fall_{0.0};
  double time_offset_{0.0};
  double hold2_{0.0};
  double rise2_{0.0};
  double fall2_{0.0};
  bool multi_stage_{false};
  WindWindowConfig window_;
};

class GradualWindField : public WindFieldBase
{
public:
  GradualWindField(const ConstantWindConfig &constant_config,
                   const DrydenWindConfig &dryden_config,
                   const GradualWindConfig &gradual_config,
                   const WindWindowConfig &window_config)
      : target_velocity_(constant_config.velocity),
        dryden_field_(dryden_config),
        window_(window_config)
  {
    time_offset_ = ros::Time::now().toSec();
    gradual_enabled_ = true; // this wind type always produces output when configured
    gradual_start_ = gradual_config.start_time + time_offset_;
    gradual_stop_ = gradual_config.stop_time + time_offset_;
    cycle_profile_enabled_ = gradual_config.cycle_profile_enabled;
    ramp_up_time_ = std::max(0.0, gradual_config.ramp_up_time);
    ramp_down_time_ = std::max(0.0, gradual_config.ramp_down_time);

    // Resolve window times (auto or manual)
    if (window_.enabled && window_.auto_from_cycle && window_.active_cycles > 0 && window_.cycle_time > 1e-6)
    {
      window_.start_time = static_cast<double>(window_.skip_cycles) * window_.cycle_time;
      window_.stop_time = static_cast<double>(window_.skip_cycles + window_.active_cycles) * window_.cycle_time;
    }
    if (window_.enabled && window_.auto_from_cycle)
    {
      window_.start_time += time_offset_;
      window_.stop_time += time_offset_;
    }
    else if (window_.enabled && !window_.auto_from_cycle)
    {
      window_.start_time += time_offset_;
      window_.stop_time += time_offset_;
    }
    if (window_.enabled && window_.stop_time <= window_.start_time)
    {
      ROS_WARN("[wind_field] Window enabled but stop_time <= start_time (start=%.3f, stop=%.3f). Disabling window.",
               window_.start_time, window_.stop_time);
      window_.enabled = false;
    }

    // Precompute ramp edges for cycle-aware profile
    if (cycle_profile_enabled_)
    {
      const double default_ramp = 0.004; // seconds
      const double on_start = window_.enabled ? window_.start_time : gradual_start_;
      const double on_stop = (window_.enabled && window_.stop_time > window_.start_time) ? window_.stop_time : gradual_stop_;
      if (on_stop > on_start)
      {
        const double up_dt = (ramp_up_time_ > 1e-6) ? ramp_up_time_ : default_ramp;
        const double down_dt = (ramp_down_time_ > 1e-6) ? ramp_down_time_ : default_ramp;
        ramp_up_start_ = on_start;
        ramp_up_end_ = on_start + up_dt;
        ramp_down_end_ = on_stop;
        ramp_down_start_ = on_stop - down_dt;
        if (ramp_down_start_ < ramp_up_end_)
        {
          const double mid = 0.5 * (ramp_up_end_ + ramp_down_start_);
          ramp_up_end_ = mid;
          ramp_down_start_ = mid;
        }
      }
      else
      {
        cycle_profile_enabled_ = false;
      }
    }
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const override
  {
    double envelope = 0.0;
    if (cycle_profile_enabled_)
    {
      envelope = cycleEnvelope(time_sec);
    }
    else
    {
      envelope = gradualEnvelope(time_sec);
      if (window_.enabled)
      {
        envelope *= windowGate(time_sec);
      }
    }

    // Only emit Dryden turbulence while the gradual envelope is active.
    // Still step the filter every call, but gate the output so the wind topic
    // stays zero before wind_gradual_start_time.
    const double turbulence_gate = (envelope > 1e-6) ? 1.0 : 0.0;
    Eigen::Vector3d wind = dryden_field_.sample(position_world, time_sec) * turbulence_gate;

    // Add mean gradual wind (already shaped by the smooth envelope).
    wind += target_velocity_ * envelope;
    return wind;
  }

private:
  double windowGate(double time_sec) const
  {
    if (!window_.enabled)
    {
      return 1.0;
    }
    if (window_.stop_time > window_.start_time && (time_sec < window_.start_time || time_sec >= window_.stop_time))
    {
      return 0.0;
    }
    return 1.0;
  }

  double smoothStep(double u) const
  {
    const double u3 = u * u * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;
    return 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  }

  double gradualEnvelope(double time_sec) const
  {
    if (!gradual_enabled_)
    {
      return 0.0;
    }
    if (time_sec <= gradual_start_ || time_sec >= gradual_stop_)
    {
      return 0.0;
    }
    const double duration = gradual_stop_ - gradual_start_;
    if (duration <= 0.0)
    {
      return 0.0;
    }
    double u = (time_sec - gradual_start_) / duration;
    if (u < 0.0)
    {
      u = 0.0;
    }
    else if (u > 1.0)
    {
      u = 1.0;
    }
    const double s = smoothStep(u);
    return 4.0 * s * (1.0 - s);
  }

  double cycleEnvelope(double time_sec) const
  {
    if (!cycle_profile_enabled_)
    {
      return 0.0;
    }
    if (time_sec < ramp_up_start_)
    {
      return 0.0;
    }
    if (time_sec < ramp_up_end_)
    {
      const double denom = std::max(ramp_up_end_ - ramp_up_start_, 1e-6);
      return smoothStep((time_sec - ramp_up_start_) / denom);
    }
    if (time_sec < ramp_down_start_)
    {
      return 1.0;
    }
    if (time_sec < ramp_down_end_)
    {
      const double denom = std::max(ramp_down_end_ - ramp_down_start_, 1e-6);
      return 1.0 - smoothStep((time_sec - ramp_down_start_) / denom);
    }
    return window_.enabled ? windowGate(time_sec) : 0.0;
  }

  Eigen::Vector3d target_velocity_{Eigen::Vector3d::Zero()};
  DrydenWindField dryden_field_;
  bool gradual_enabled_{false};
  double gradual_start_{0.0};
  double gradual_stop_{0.0};
  bool cycle_profile_enabled_{false};
  double ramp_up_start_{0.0};
  double ramp_up_end_{0.0};
  double ramp_down_start_{0.0};
  double ramp_down_end_{0.0};
  double ramp_up_time_{0.0};
  double ramp_down_time_{0.0};
  double time_offset_{0.0};
  WindWindowConfig window_;
};

class CompositeWindField : public WindFieldBase
{
public:
  CompositeWindField(const ConstantWindConfig &constant_config,
                     const DrydenWindConfig &dryden_config,
                     const GustEventConfig &gust_event_config,
                     const GradualWindConfig &gradual_config,
                     const WindWindowConfig &window_config)
      : constant_velocity_(constant_config.velocity),
        dryden_field_(dryden_config),
        window_(window_config)
  {
    time_offset_ = ros::Time::now().toSec();
    gust_axis_ = gust_event_config.axis;
    gust_magnitude_ = gust_event_config.magnitude;
    gust_start_ = gust_event_config.start_time + time_offset_;
    gust_hold_ = gust_event_config.hold_duration;
    gust_rise_ = gust_event_config.rise_time;
    gust_fall_ = gust_event_config.fall_time;
    if (!std::isfinite(gust_magnitude_))
    {
      gust_magnitude_ = 0.0;
    }
    if (gust_axis_.norm() < 1e-6)
    {
      gust_axis_ = Eigen::Vector3d::UnitX();
    }
    gust_axis_.z() = 0.0; // only horizontal gusts
    gust_axis_.normalize();

    gradual_enabled_ = gradual_config.enabled;
    gradual_start_ = gradual_config.start_time + time_offset_;
    gradual_stop_ = gradual_config.stop_time + time_offset_;
    gradual_cycle_profile_enabled_ = gradual_config.cycle_profile_enabled;
    ramp_up_time_ = std::max(0.0, gradual_config.ramp_up_time);
    ramp_down_time_ = std::max(0.0, gradual_config.ramp_down_time);

    // Resolve window times (auto or manual)
    if (window_.enabled && window_.auto_from_cycle && window_.active_cycles > 0 && window_.cycle_time > 1e-6)
    {
      window_.start_time = static_cast<double>(window_.skip_cycles) * window_.cycle_time;
      window_.stop_time = static_cast<double>(window_.skip_cycles + window_.active_cycles) * window_.cycle_time;
    }
    // Treat cycle-derived times as offsets from now so gating aligns with the current sim time base.
    if (window_.enabled && window_.auto_from_cycle)
    {
      window_.start_time += time_offset_;
      window_.stop_time += time_offset_;
    }
    else if (window_.enabled && !window_.auto_from_cycle)
    {
      // Manual start/stop are also treated as offsets from launch for consistency with gust/gradual.
      window_.start_time += time_offset_;
      window_.stop_time += time_offset_;
    }
    if (window_.enabled && window_.stop_time <= window_.start_time)
    {
      ROS_WARN("[wind_field] Window enabled but stop_time <= start_time (start=%.3f, stop=%.3f). Disabling window.",
               window_.start_time, window_.stop_time);
      window_.enabled = false;
    }
    if (gradual_enabled_ && !gradual_cycle_profile_enabled_ && gradual_stop_ <= gradual_start_)
    {
      ROS_WARN("[wind_field] Gradual wind enabled but stop_time <= start_time (start=%.3f, stop=%.3f). Disabling gradual.",
               gradual_start_, gradual_stop_);
      gradual_enabled_ = false;
    }

    // Precompute ramp edges for cycle-aware profile. Default to a very short ramp (4 ms) if not provided,
    // so the effective step response is within ~step_T/10 when step_T≈0.041 s.
    if (gradual_cycle_profile_enabled_)
    {
      const double default_ramp = 0.004; // seconds
      const double on_start = window_.enabled ? window_.start_time : gradual_start_;
      const double on_stop = (window_.enabled && window_.stop_time > window_.start_time) ? window_.stop_time : gradual_stop_;
      if (on_stop > on_start)
      {
        const double up_dt = (ramp_up_time_ > 1e-6) ? ramp_up_time_ : default_ramp;
        const double down_dt = (ramp_down_time_ > 1e-6) ? ramp_down_time_ : default_ramp;
        ramp_up_start_ = on_start;
        ramp_up_end_ = on_start + up_dt;
        ramp_down_end_ = on_stop;
        ramp_down_start_ = on_stop - down_dt;
        // Prevent overlap; split the remaining interval if needed.
        if (ramp_down_start_ < ramp_up_end_)
        {
          const double mid = 0.5 * (ramp_up_end_ + ramp_down_start_);
          ramp_up_end_ = mid;
          ramp_down_start_ = mid;
        }
      }
      else
      {
        gradual_cycle_profile_enabled_ = false;
      }
    }
  }

  Eigen::Vector3d sample(const Eigen::Vector3d &position_world, double time_sec) const override
  {
    const double envelope = windowEnvelope(time_sec);

    // Dryden component produces turbulence around the steady bias.
    // Only apply constant bias on the horizontal axes; leave vertical axis to Dryden.
    Eigen::Vector3d wind = dryden_field_.sample(position_world, time_sec);
    wind.x() += constant_velocity_.x();
    wind.y() += constant_velocity_.y();

    const double gust_scale = gustEnvelope(time_sec);
    if (gust_scale > 1e-9)
    {
      const Eigen::Vector3d gust = gust_axis_ * gust_magnitude_ * gust_scale;
      wind.x() += gust.x();
      wind.y() += gust.y();
      // z intentionally untouched
    }
    if (gradual_cycle_profile_enabled_)
    {
      // Scale entire wind (constant + Dryden + gust) to follow the cycle-aware on/off ramp.
      if (envelope < 1.0)
      {
        wind *= envelope;
      }
    }
    else
    {
      const double gradual_scale = gradualEnvelope(time_sec);
      if (gradual_scale > 1e-9)
      {
        wind.x() += constant_velocity_.x() * gradual_scale;
        wind.y() += constant_velocity_.y() * gradual_scale;
        // z intentionally untouched
      }
      const double gate = windowGate(time_sec);
      if (gate < 1.0)
      {
        wind *= gate; // hard on/off; gate is 0 or 1
      }
    }
    return wind;
  }

private:
  double windowEnvelope(double time_sec) const
  {
    if (!gradual_cycle_profile_enabled_)
    {
      return windowGate(time_sec);
    }
    // Cycle-aware profile: short ramp up/down around the window edges.
    // Use precomputed ramp markers.
    if (time_sec < ramp_up_start_)
    {
      return 0.0;
    }
    if (time_sec < ramp_up_end_)
    {
      const double denom = std::max(ramp_up_end_ - ramp_up_start_, 1e-6);
      return smoothStep((time_sec - ramp_up_start_) / denom);
    }
    if (time_sec < ramp_down_start_)
    {
      return 1.0;
    }
    if (time_sec < ramp_down_end_)
    {
      const double denom = std::max(ramp_down_end_ - ramp_down_start_, 1e-6);
      return 1.0 - smoothStep((time_sec - ramp_down_start_) / denom);
    }
    return window_.enabled ? windowGate(time_sec) : 0.0;
  }

  double windowGate(double time_sec) const
  {
    if (!window_.enabled)
    {
      return 1.0;
    }
    if (window_.stop_time > window_.start_time && (time_sec < window_.start_time || time_sec >= window_.stop_time))
    {
      return 0.0;
    }
    return 1.0;
  }

  double smoothStep(double u) const
  {
    // quintic smoothstep: C2 continuous, s(0)=0, s(1)=1
    const double u3 = u * u * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;
    return 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
  }

  double gradualEnvelope(double time_sec) const
  {
    if (!gradual_enabled_)
    {
      return 0.0;
    }
    if (time_sec <= gradual_start_ || time_sec >= gradual_stop_)
    {
      return 0.0;
    }
    const double duration = gradual_stop_ - gradual_start_;
    if (duration <= 0.0)
    {
      return 0.0;
    }
    double u = (time_sec - gradual_start_) / duration;
    if (u < 0.0)
    {
      u = 0.0;
    }
    else if (u > 1.0)
    {
      u = 1.0;
    }
    const double s = smoothStep(u);
    // Bell-shaped, zero slope at boundaries, peak at midpoint.
    return 4.0 * s * (1.0 - s);
  }

  double gustEnvelope(double time_sec) const
  {
    if (gust_magnitude_ <= 1e-9)
    {
      return 0.0;
    }
    const double t = time_sec;
    const double up_end = gust_start_ + gust_rise_;
    const double hold_end = up_end + gust_hold_;
    const double down_end = hold_end + gust_fall_;

    if (t < gust_start_)
    {
      return 0.0;
    }
    if (t < up_end)
    {
      const double denom = std::max(gust_rise_, 1e-6);
      return std::min(1.0, std::max(0.0, (t - gust_start_) / denom));
    }
    if (t < hold_end)
    {
      return 1.0;
    }
    if (t < down_end)
    {
      const double denom = std::max(gust_fall_, 1e-6);
      return std::min(1.0, std::max(0.0, 1.0 - (t - hold_end) / denom));
    }
    return 0.0;
  }

private:
  Eigen::Vector3d constant_velocity_{Eigen::Vector3d::Zero()};
  DrydenWindField dryden_field_;
  Eigen::Vector3d gust_axis_{Eigen::Vector3d::UnitX()};
  double gust_magnitude_{0.0};
  double gust_start_{0.0};
  double gust_hold_{0.0};
  double gust_rise_{0.0};
  double gust_fall_{0.0};
  bool gradual_enabled_{false};
  double gradual_start_{0.0};
  double gradual_stop_{0.0};
  bool gradual_cycle_profile_enabled_{false};
  double ramp_up_start_{0.0};
  double ramp_up_end_{0.0};
  double ramp_down_start_{0.0};
  double ramp_down_end_{0.0};
  double ramp_up_time_{0.0};
  double ramp_down_time_{0.0};
  double time_offset_{0.0};
  WindWindowConfig window_;
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
  if (lower == "gust_event" || lower == "gustevent")
  {
    return WindFieldType::GustEvent;
  }
  if (lower == "gradual")
  {
    return WindFieldType::Gradual;
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
  case WindFieldType::GustEvent:
    return "gust_event";
  case WindFieldType::Gradual:
    return "gradual";
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

  // deterministic gust event (horizontal only)
  config.gust_event.axis = readVector3(nh, "gust_event/axis", Eigen::Vector3d::UnitX());
  if (config.gust_event.axis.head<2>().norm() < 1e-6)
  {
    config.gust_event.axis = Eigen::Vector3d::UnitX();
  }
  config.gust_event.axis.z() = 0.0;
  config.gust_event.axis.normalize();
  config.gust_event.magnitude = nh.param("gust_event/magnitude", 0.0);
  config.gust_event.start_time = nh.param("gust_event/start_time", 0.0);
  config.gust_event.start_time_offset = nh.param("gust_event/start_time_offset", 0.0);
  config.gust_event.hold_duration = nh.param("gust_event/hold_duration", 0.0);
  config.gust_event.rise_time = nh.param("gust_event/rise_time", 0.0);
  config.gust_event.fall_time = nh.param("gust_event/fall_time", 0.0);

  // Optional multi-stage gust parameters (stage 2 & 3 target).
  config.gust_event.axis2 = readVector3(nh, "gust_event/axis2", config.gust_event.axis);
  if (config.gust_event.axis2.head<2>().norm() < 1e-6)
  {
    config.gust_event.axis2 = config.gust_event.axis;
  }
  config.gust_event.axis2.z() = 0.0;
  config.gust_event.axis2.normalize();
  config.gust_event.magnitude2 = nh.param("gust_event/magnitude2", config.gust_event.magnitude);
  config.gust_event.hold_duration2 = nh.param("gust_event/hold_duration2", 0.0);
  config.gust_event.rise_time2 = nh.param("gust_event/rise_time2", 0.0);
  config.gust_event.fall_time2 = nh.param("gust_event/fall_time2", 0.0);
  config.gust_event.multi_stage = nh.param("gust_event/multi_stage", false);
  if (!config.gust_event.multi_stage)
  {
    // Auto-enable multi-stage if any secondary duration or magnitude is non-trivial.
    const bool has_second_stage = (config.gust_event.hold_duration2 > 1e-6) ||
                                  (config.gust_event.rise_time2 > 1e-6) ||
                                  (config.gust_event.fall_time2 > 1e-6) ||
                                  (std::fabs(config.gust_event.magnitude2 - config.gust_event.magnitude) > 1e-6) ||
                                  ((config.gust_event.axis2.head<2>() - config.gust_event.axis.head<2>()).norm() > 1e-6);
    if (has_second_stage)
    {
      config.gust_event.multi_stage = true;
    }
  }

  // gradual wind (smooth ramp of mean wind on XY)
  config.gradual.enabled = nh.param("gradual/enabled", false);
  config.gradual.start_time = nh.param("gradual/start_time", 0.0);
  config.gradual.stop_time = nh.param("gradual/stop_time", 0.0);
  config.gradual.cycle_profile_enabled = nh.param("gradual/cycle_profile_enabled", false);
  config.gradual.ramp_up_time = nh.param("gradual/ramp_up_time", 0.0);
  config.gradual.ramp_down_time = nh.param("gradual/ramp_down_time", config.gradual.ramp_up_time);

  // wind window gating (hard on/off, no ramp)
  config.window.enabled = nh.param("window/enabled", false);
  config.window.auto_from_cycle = nh.param("window/auto_from_cycle", false);
  config.window.skip_cycles = nh.param("window/skip_cycles", 0);
  config.window.active_cycles = nh.param("window/active_cycles", 0);
  config.window.cycle_time = nh.param("window/cycle_time", 0.0);
  if (config.window.auto_from_cycle && config.window.cycle_time <= 0.0)
  {
    config.window.cycle_time = inferCycleTime(nh);
    if (config.window.cycle_time <= 0.0)
    {
      ROS_WARN("[wind_field] auto_from_cycle enabled but cycle_time unavailable; window gating will be disabled.");
      config.window.enabled = false;
      config.window.auto_from_cycle = false;
    }
  }
  config.window.start_time = nh.param("window/start_time", 0.0);
  config.window.stop_time = nh.param("window/stop_time", 0.0);

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
  case WindFieldType::GustEvent:
    return std::make_unique<GustEventWindField>(config.gust_event, config.dryden, config.window);
  case WindFieldType::Gradual:
  {
    ConstantWindConfig constant_full = config.constant;
    return std::make_unique<GradualWindField>(constant_full, config.dryden, config.gradual, config.window);
  }
  case WindFieldType::Dryden:
    return std::make_unique<DrydenWindField>(config.dryden);
  case WindFieldType::Composite:
  {
    ConstantWindConfig constant_xy = config.constant;
    if (std::fabs(constant_xy.velocity.z()) > 1e-9)
    {
      ROS_WARN_STREAM("[wind_field] Composite wind: ignoring constant vertical component "
                      << constant_xy.velocity.z() << " (using horizontal-only bias).");
      constant_xy.velocity.z() = 0.0;
    }
    return std::make_unique<CompositeWindField>(constant_xy, config.dryden, config.gust_event, config.gradual, config.window);
  }
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
