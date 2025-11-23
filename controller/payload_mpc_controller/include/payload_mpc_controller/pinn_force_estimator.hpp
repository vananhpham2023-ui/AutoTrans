#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "base_force_estimator.hpp"
#include "lowpassfilter2p.h"

#ifdef AUTOTRANS_WITH_TORCH
#include <boost/property_tree/ptree.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace PayloadMPC
{

  class PinnForceEstimator : public BaseForceEstimator
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PinnForceEstimator();
    ~PinnForceEstimator() override = default;

    void init(MpcParams &params) override;
    void enableForceEstimator() override;
    void disableForceEstimator() override;
    void setSystemState(const Eigen::Vector3d &quad_acc_body,
                        const Eigen::Quaterniond &Rotwb,
                        const Eigen::Vector3d &load_acc_body,
                        const Eigen::Quaterniond &load_Rotwb,
                        const Eigen::Vector3d cable,
                        const Eigen::Vector4d &Rpm) override;
    void caculate_force(Eigen::Vector3d &fl, Eigen::Vector3d &fq) override;
    bool isOperational() const override;

  private:
#ifdef AUTOTRANS_WITH_TORCH
    static constexpr std::size_t kInputDim = 13;
    static constexpr std::size_t kOutputDim = 6;
    static constexpr double kMinSpanEpsilon = 1e-9;
    static constexpr double kSaturationGuardRatio = 0.95;

    template <std::size_t Dimension>
    struct NormalizerAxis
    {
      std::array<double, Dimension> min{};
      std::array<double, Dimension> max{};
      std::array<double, Dimension> span{};
      std::array<double, Dimension> inv_span{};
    };

    struct NormalizerSet
    {
      NormalizerAxis<kInputDim> input;
      NormalizerAxis<kOutputDim> output;
    };

    bool loadNormalizer(const std::string &path);
    bool loadModel(const std::string &path);
    template <std::size_t Dimension>
    bool fillAxis(const boost::property_tree::ptree &parent,
                  const std::string &section,
                  NormalizerAxis<Dimension> &axis);
    template <std::size_t Dimension>
    static bool readArray(const boost::property_tree::ptree &node,
                          std::array<double, Dimension> &target);
    bool prepareInputBuffer(std::string &reason);
    bool runInference(Eigen::Vector3d &fl, Eigen::Vector3d &fq, std::string &reason);
    void applyPostProcessing(Eigen::Vector3d &fl, Eigen::Vector3d &fq);
    void saturate(Eigen::Vector3d &value) const;
    bool enforceSafety(const Eigen::Vector3d &fl, const Eigen::Vector3d &fq);
    void handleFailure(const std::string &reason, Eigen::Vector3d &fl, Eigen::Vector3d &fq);
    double normalizeInput(double value, std::size_t idx) const;
    double denormalizeOutput(double value, std::size_t idx) const;
    bool ensureFinite(const Eigen::Vector3d &value) const;
    bool ensureFinite(double value) const;

    MpcParams *params_{nullptr};
    NormalizerSet normalizer_;
    bool normalizer_ready_{false};
    bool model_ready_{false};
    bool state_ready_{false};
    bool use_estimator_{false};
    double max_force_{0.0};
    double sqrt_kf_{0.0};
    double ema_alpha_{1.0};
    int failover_threshold_{5};
    int consecutive_failures_{0};
    int saturation_guard_counter_{0};
    int saturation_guard_limit_{5};
    bool filters_configured_{false};

    Eigen::Vector3d quad_acc_body_;
    Eigen::Vector3d load_acc_body_;
    Eigen::Vector3d cable_dir_;
    Eigen::Vector3d body_z_world_;
    Eigen::Vector4d rpm_;
    double thrust_{0.0};

    Eigen::Vector3d last_fl_;
    Eigen::Vector3d last_fq_;
    LowPassFilter2p<Eigen::Vector3d> fl_filter_;
    LowPassFilter2p<Eigen::Vector3d> fq_filter_;

    std::array<float, kInputDim> input_buffer_{};
    torch::Tensor input_tensor_;
    std::vector<torch::jit::IValue> inputs_;
    torch::jit::Module module_;
    std::unique_ptr<torch::NoGradGuard> no_grad_guard_;

    std::string model_path_;
    std::string normalizer_path_;
#endif
  };

} // namespace PayloadMPC

#ifdef AUTOTRANS_WITH_TORCH
namespace PayloadMPC
{

  template <std::size_t Dimension>
  bool PinnForceEstimator::readArray(const boost::property_tree::ptree &node,
                                     std::array<double, Dimension> &target)
  {
    std::size_t idx = 0;
    for (const auto &child : node)
    {
      if (idx >= Dimension)
      {
        return false;
      }
      target[idx++] = child.second.get_value<double>();
    }
    return idx == Dimension;
  }

  template <std::size_t Dimension>
  bool PinnForceEstimator::fillAxis(const boost::property_tree::ptree &parent,
                                    const std::string &section,
                                    NormalizerAxis<Dimension> &axis)
  {
    auto section_node = parent.get_child_optional(section);
    if (!section_node)
    {
      return false;
    }
    auto min_node = section_node->get_child_optional("min");
    auto max_node = section_node->get_child_optional("max");
    if (!min_node || !max_node)
    {
      return false;
    }
    if (!readArray(*min_node, axis.min) || !readArray(*max_node, axis.max))
    {
      return false;
    }
    for (std::size_t i = 0; i < Dimension; ++i)
    {
      if (!std::isfinite(axis.min[i]) || !std::isfinite(axis.max[i]))
      {
        return false;
      }
      if (axis.max[i] < axis.min[i])
      {
        return false;
      }
      axis.span[i] = axis.max[i] - axis.min[i];
      if (axis.span[i] <= kMinSpanEpsilon)
      {
        axis.span[i] = 0.0;
        axis.inv_span[i] = 0.0;
      }
      else
      {
        axis.inv_span[i] = 1.0 / axis.span[i];
      }
    }
    return true;
  }

} // namespace PayloadMPC
#endif
