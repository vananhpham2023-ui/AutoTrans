#pragma once

#include "Eigen/Eigen"
#include "Eigen/Geometry"

namespace PayloadMPC
{

  class MpcParams;

  class BaseForceEstimator
  {
  public:
    BaseForceEstimator() = default;
    virtual ~BaseForceEstimator() = default;

    BaseForceEstimator(const BaseForceEstimator &) = delete;
    BaseForceEstimator &operator=(const BaseForceEstimator &) = delete;
    BaseForceEstimator(BaseForceEstimator &&) = delete;
    BaseForceEstimator &operator=(BaseForceEstimator &&) = delete;

    virtual void init(MpcParams &params) = 0;
    virtual void enableForceEstimator() = 0;
    virtual void disableForceEstimator() = 0;
    virtual void setSystemState(const Eigen::Vector3d &quad_acc_body,
                                const Eigen::Quaterniond &Rotwb,
                                const Eigen::Vector3d &load_acc_body,
                                const Eigen::Quaterniond &load_Rotwb,
                                const Eigen::Vector3d cable,
                                const Eigen::Vector4d &Rpm) = 0;
    virtual void caculate_force(Eigen::Vector3d &fl, Eigen::Vector3d &fq) = 0;
    virtual bool isOperational() const = 0;
  };

} // namespace PayloadMPC
