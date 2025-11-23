#include "pinn_force_estimator.hpp"
#include "mpc_params.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include <Eigen/Geometry>
#include <boost/property_tree/json_parser.hpp>
#include <ros/ros.h>

namespace PayloadMPC
{

  PinnForceEstimator::PinnForceEstimator()
  {
    quad_acc_body_.setZero();
    load_acc_body_.setZero();
    cable_dir_.setZero();
    body_z_world_.setZero();
    rpm_.setZero();
    last_fl_.setZero();
    last_fq_.setZero();
    input_buffer_.fill(0.0F);
    inputs_.reserve(1);
  }

  void PinnForceEstimator::init(MpcParams &params)
  {
    params_ = &params;
    const double cfg_max_force = params.force_estimator_param_.max_force;
    if (!std::isfinite(cfg_max_force) || cfg_max_force <= 0.0)
    {
      ROS_WARN("[PinnForceEstimator] Invalid max_force %.3f, fallback to 10 N", cfg_max_force);
      max_force_ = 10.0;
    }
    else
    {
      max_force_ = cfg_max_force;
    }
    fl_filter_.set_cutoff_frequency(params.force_estimator_param_.sample_freq_fl,
                                    params.force_estimator_param_.cutoff_freq_fl);
    fq_filter_.set_cutoff_frequency(params.force_estimator_param_.sample_freq_fq,
                                    params.force_estimator_param_.cutoff_freq_fq);
    fl_filter_.reset(Eigen::Vector3d::Zero());
    fq_filter_.reset(Eigen::Vector3d::Zero());
    filters_configured_ =
        (params.force_estimator_param_.cutoff_freq_fl > 0.0 && params.force_estimator_param_.sample_freq_fl > 0.0) ||
        (params.force_estimator_param_.cutoff_freq_fq > 0.0 && params.force_estimator_param_.sample_freq_fq > 0.0);
    sqrt_kf_ = params.force_estimator_param_.sqrt_kf;
    use_estimator_ = params.force_estimator_param_.use_force_estimator;
    consecutive_failures_ = 0;
    saturation_guard_counter_ = 0;
    last_fl_.setZero();
    last_fq_.setZero();

    model_ready_ = false;
    normalizer_ready_ = false;
    state_ready_ = false;

    model_path_.clear();
    normalizer_path_.clear();

    if (!use_estimator_)
    {
      ROS_WARN("[PinnForceEstimator] Force estimator disabled via configuration.");
      return;
    }

    if (!params.pinn_force_estimator_param_.enabled)
    {
      ROS_WARN("[PinnForceEstimator] PINN estimator disabled in pinn config.");
      use_estimator_ = false;
      return;
    }

    model_path_ = params.pinn_force_estimator_param_.model_path;
    normalizer_path_ = params.pinn_force_estimator_param_.normalizer_path;
    ema_alpha_ = std::clamp(params.pinn_force_estimator_param_.ema_alpha, 0.0, 1.0);
    failover_threshold_ = std::max(1, params.pinn_force_estimator_param_.failover_threshold);
    saturation_guard_limit_ = std::max(failover_threshold_, 5);

    if (model_path_.empty() || normalizer_path_.empty())
    {
      ROS_ERROR("[PinnForceEstimator] Model or normalizer path is empty. Disable PINN estimator.");
      use_estimator_ = false;
      return;
    }

    if (!loadNormalizer(normalizer_path_))
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] Failed to load normalizer file: " << normalizer_path_);
      use_estimator_ = false;
      return;
    }

    if (!loadModel(model_path_))
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] Failed to load TorchScript model: " << model_path_);
      use_estimator_ = false;
      return;
    }

    if (!model_ready_ || !normalizer_ready_)
    {
      ROS_ERROR("[PinnForceEstimator] Model or normalizer not ready. PINN disabled.");
      use_estimator_ = false;
    }
    else
    {
      ROS_INFO_STREAM("[PinnForceEstimator] PINN estimator initialized with model " << model_path_);
    }
  }

  void PinnForceEstimator::enableForceEstimator()
  {
    use_estimator_ = true;
    consecutive_failures_ = 0;
    last_fl_.setZero();
    last_fq_.setZero();
  }

  void PinnForceEstimator::disableForceEstimator()
  {
    use_estimator_ = false;
    consecutive_failures_ = 0;
    last_fl_.setZero();
    last_fq_.setZero();
  }

  void PinnForceEstimator::setSystemState(const Eigen::Vector3d &quad_acc_body,
                                          const Eigen::Quaterniond &Rotwb,
                                          const Eigen::Vector3d &load_acc_body,
                                          const Eigen::Quaterniond &load_Rotwb,
                                          const Eigen::Vector3d cable,
                                          const Eigen::Vector4d &Rpm)
  {
    (void)load_Rotwb;
    quad_acc_body_ = quad_acc_body;
    if (!quad_acc_body_.allFinite())
    {
      quad_acc_body_.setZero();
    }
    load_acc_body_ = load_acc_body;
    if (!load_acc_body_.allFinite())
    {
      load_acc_body_.setZero();
    }
    rpm_ = Rpm;
    for (Eigen::Index i = 0; i < rpm_.size(); ++i)
    {
      if (!std::isfinite(rpm_(i)))
      {
        rpm_(i) = 0.0;
      }
    }

    Eigen::Quaterniond quat = Rotwb;
    if (!std::isfinite(quat.w()) || !std::isfinite(quat.x()) || !std::isfinite(quat.y()) || !std::isfinite(quat.z()))
    {
      quat = Eigen::Quaterniond::Identity();
    }
    else
    {
      quat.normalize();
    }
    body_z_world_ = quat.toRotationMatrix().col(2);
    if (!body_z_world_.allFinite())
    {
      body_z_world_.setZero();
    }

    double cable_norm = cable.norm();
    if (std::isfinite(cable_norm) && cable_norm > 1e-6)
    {
      cable_dir_ = cable / cable_norm;
    }
    else
    {
      cable_dir_.setZero();
    }

    if (sqrt_kf_ > 0.0)
    {
      Eigen::Array4d rpm_squared = (rpm_.array()).square();
      thrust_ = sqrt_kf_ * sqrt_kf_ * rpm_squared.sum();
    }
    else
    {
      thrust_ = 0.0;
    }

    if (!std::isfinite(thrust_) || thrust_ < 0.0)
    {
      thrust_ = 0.0;
    }

    state_ready_ = true;
  }

  void PinnForceEstimator::caculate_force(Eigen::Vector3d &fl, Eigen::Vector3d &fq)
  {
    fl.setZero();
    fq.setZero();

    if (!use_estimator_ || !model_ready_ || !normalizer_ready_)
    {
      return;
    }

    if (!state_ready_)
    {
      return;
    }

    std::string reason;
    if (!prepareInputBuffer(reason))
    {
      handleFailure(reason, fl, fq);
      return;
    }

    Eigen::Vector3d fl_raw, fq_raw;
    if (!runInference(fl_raw, fq_raw, reason))
    {
      handleFailure(reason, fl, fq);
      return;
    }

    if (!ensureFinite(fl_raw) || !ensureFinite(fq_raw))
    {
      handleFailure("Non-finite inference output", fl, fq);
      return;
    }

    applyPostProcessing(fl_raw, fq_raw);
    fl = fl_raw;
    fq = fq_raw;
    if (!enforceSafety(fl, fq))
    {
      saturation_guard_counter_ = 0;
      handleFailure("Force saturation guard triggered", fl, fq);
      return;
    }
    consecutive_failures_ = 0;
  }

  bool PinnForceEstimator::loadNormalizer(const std::string &path)
  {
    try
    {
      boost::property_tree::ptree tree;
      boost::property_tree::read_json(path, tree);
      if (!fillAxis(tree, "input", normalizer_.input))
      {
        ROS_ERROR_STREAM("[PinnForceEstimator] Invalid input normalizer data in " << path);
        return false;
      }
      if (!fillAxis(tree, "output", normalizer_.output))
      {
        ROS_ERROR_STREAM("[PinnForceEstimator] Invalid output normalizer data in " << path);
        return false;
      }
      normalizer_ready_ = true;
      normalizer_path_ = path;
      return true;
    }
    catch (const boost::property_tree::json_parser::json_parser_error &e)
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] JSON parser error: " << e.what());
    }
    catch (const std::exception &e)
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] Failed to read normalizer: " << e.what());
    }
    return false;
  }

  bool PinnForceEstimator::loadModel(const std::string &path)
  {
    try
    {
      module_ = torch::jit::load(path);
      module_.to(torch::kCPU);
      module_.to(torch::kFloat32);
      module_.eval();

      auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
      input_tensor_ = torch::from_blob(
          input_buffer_.data(),
          {1LL, static_cast<long long>(kInputDim)},
          [](void *) {},
          tensor_options);
      inputs_.clear();
      inputs_.emplace_back(input_tensor_);
      no_grad_guard_ = std::make_unique<torch::NoGradGuard>();
      model_ready_ = true;
      model_path_ = path;
      return true;
    }
    catch (const c10::Error &e)
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] TorchScript load error: " << e.what());
    }
    catch (const std::exception &e)
    {
      ROS_ERROR_STREAM("[PinnForceEstimator] Unexpected load exception: " << e.what());
    }
    model_ready_ = false;
    return false;
  }

  bool PinnForceEstimator::prepareInputBuffer(std::string &reason)
  {
    std::array<double, kInputDim> features{};
    std::size_t idx = 0;

    auto write_vec = [&](const Eigen::Vector3d &vec) {
      for (int i = 0; i < 3; ++i)
      {
        if (idx >= kInputDim)
        {
          reason = "Feature overflow";
          return false;
        }
        features[idx++] = vec(i);
      }
      return true;
    };

    if (!write_vec(quad_acc_body_))
      return false;
    if (!write_vec(load_acc_body_))
      return false;
    if (!write_vec(cable_dir_))
      return false;
    if (!write_vec(body_z_world_))
      return false;

    if (idx >= kInputDim)
    {
      reason = "Feature index overflow";
      return false;
    }
    features[idx++] = thrust_;

    if (idx != kInputDim)
    {
      reason = "Feature dimension mismatch";
      return false;
    }

    for (std::size_t i = 0; i < kInputDim; ++i)
    {
      if (!ensureFinite(features[i]))
      {
        std::ostringstream oss;
        oss << "Non-finite input feature @" << i;
        reason = oss.str();
        return false;
      }
      input_buffer_[i] = static_cast<float>(normalizeInput(features[i], i));
    }
    return true;
  }

  bool PinnForceEstimator::runInference(Eigen::Vector3d &fl, Eigen::Vector3d &fq, std::string &reason)
  {
    try
    {
      torch::Tensor output_tensor = module_.forward(inputs_).toTensor();
      torch::Tensor flattened = output_tensor.reshape({-1}).to(torch::kCPU).contiguous();
      if (flattened.numel() < static_cast<long long>(kOutputDim))
      {
        reason = "PINN output dimension mismatch";
        return false;
      }
      const float *ptr = flattened.data_ptr<float>();
      fq << denormalizeOutput(ptr[0], 0),
          denormalizeOutput(ptr[1], 1),
          denormalizeOutput(ptr[2], 2);
      fl << denormalizeOutput(ptr[3], 3),
          denormalizeOutput(ptr[4], 4),
          denormalizeOutput(ptr[5], 5);
      return true;
    }
    catch (const c10::Error &e)
    {
      reason = e.what();
    }
    catch (const std::exception &e)
    {
      reason = e.what();
    }
    return false;
  }

  void PinnForceEstimator::applyPostProcessing(Eigen::Vector3d &fl, Eigen::Vector3d &fq)
  {
    saturate(fl);
    saturate(fq);

    if (ema_alpha_ < 1.0)
    {
      fl = ema_alpha_ * fl + (1.0 - ema_alpha_) * last_fl_;
      fq = ema_alpha_ * fq + (1.0 - ema_alpha_) * last_fq_;
    }

    if (filters_configured_)
    {
      fl = fl_filter_.apply(fl);
      fq = fq_filter_.apply(fq);
      saturate(fl);
      saturate(fq);
    }

    last_fl_ = fl;
    last_fq_ = fq;
  }

  void PinnForceEstimator::saturate(Eigen::Vector3d &value) const
  {
    double norm = value.norm();
    if (norm > max_force_ && max_force_ > 0.0)
    {
      value = value * (max_force_ / norm);
    }
  }

  bool PinnForceEstimator::enforceSafety(const Eigen::Vector3d &fl, const Eigen::Vector3d &fq)
  {
    if (max_force_ <= 0.0)
    {
      saturation_guard_counter_ = 0;
      return true;
    }
    const double guard_limit = max_force_ * kSaturationGuardRatio;
    const double fl_norm = fl.norm();
    const double fq_norm = fq.norm();
    if (fl_norm < guard_limit && fq_norm < guard_limit)
    {
      saturation_guard_counter_ = 0;
      return true;
    }
    ++saturation_guard_counter_;
    if (saturation_guard_counter_ < saturation_guard_limit_)
    {
      ROS_WARN_THROTTLE(1.0, "[PinnForceEstimator] Force output near saturation (|fl|=%.2f, |fq|=%.2f, limit=%.2f).",
                        fl_norm, fq_norm, max_force_);
      return true;
    }
    return false;
  }

  void PinnForceEstimator::handleFailure(const std::string &reason, Eigen::Vector3d &fl, Eigen::Vector3d &fq)
  {
    ++consecutive_failures_;
    fl.setZero();
    fq.setZero();
    last_fl_.setZero();
    last_fq_.setZero();
    if (consecutive_failures_ >= failover_threshold_)
    {
      use_estimator_ = false;
      ROS_ERROR_STREAM("[PinnForceEstimator] Consecutive failures reached limit (" << reason
                                                                                   << "). Disabling PINN force estimator.");
    }
    else
    {
      ROS_WARN_STREAM_THROTTLE(1.0, "[PinnForceEstimator] Inference failed: " << reason);
    }
  }

  double PinnForceEstimator::normalizeInput(double value, std::size_t idx) const
  {
    const double min_v = normalizer_.input.min[idx];
    const double max_v = normalizer_.input.max[idx];
    const double span = normalizer_.input.span[idx];
    if (span <= kMinSpanEpsilon)
    {
      return 0.0;
    }
    const double clamped = std::clamp(value, min_v, max_v);
    return 2.0 * (clamped - min_v) * normalizer_.input.inv_span[idx] - 1.0;
  }

  double PinnForceEstimator::denormalizeOutput(double value, std::size_t idx) const
  {
    const double span = normalizer_.output.span[idx];
    if (span <= kMinSpanEpsilon)
    {
      return normalizer_.output.min[idx];
    }
    const double min_v = normalizer_.output.min[idx];
    const double max_v = normalizer_.output.max[idx];
    const double scaled = 0.5 * (value + 1.0) * span + min_v;
    const double lo = std::min(min_v, max_v);
    const double hi = std::max(min_v, max_v);
    return std::clamp(scaled, lo, hi);
  }

  bool PinnForceEstimator::ensureFinite(const Eigen::Vector3d &value) const
  {
    return value.allFinite();
  }

  bool PinnForceEstimator::ensureFinite(double value) const
  {
    return std::isfinite(value);
  }

  bool PinnForceEstimator::isOperational() const
  {
    return use_estimator_ && model_ready_ && normalizer_ready_;
  }

} // namespace PayloadMPC
