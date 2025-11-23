#ifndef AUTOTRANS_WITH_TORCH

#include <new>

#include "pinn_force_estimator.hpp"

#include <ros/ros.h>

namespace PayloadMPC
{

  PinnForceEstimator::PinnForceEstimator() = default;

  void PinnForceEstimator::init(MpcParams &)
  {
    ROS_WARN_THROTTLE(5.0, "[PinnForceEstimator] LibTorch is unavailable in this build; only L-BFGS estimators are active.");
  }

  void PinnForceEstimator::enableForceEstimator() {}

  void PinnForceEstimator::disableForceEstimator() {}

  void PinnForceEstimator::setSystemState(const Eigen::Vector3d &,
                                          const Eigen::Quaterniond &,
                                          const Eigen::Vector3d &,
                                          const Eigen::Quaterniond &,
                                          const Eigen::Vector3d,
                                          const Eigen::Vector4d &)
  {
  }

  void PinnForceEstimator::caculate_force(Eigen::Vector3d &fl, Eigen::Vector3d &fq)
  {
    fl.setZero();
    fq.setZero();
  }

  bool PinnForceEstimator::isOperational() const
  {
    return false;
  }

} // namespace PayloadMPC

#endif // AUTOTRANS_WITH_TORCH
