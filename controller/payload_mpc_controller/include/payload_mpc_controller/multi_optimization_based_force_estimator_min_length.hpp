/**
 * @file multi_optimization_based_force_estimator_min_length.hpp
 * @author Haojia Li (hlied@connect.ust.hk)
 * @brief optimization based extern force estimator minimizing force change length
 * @version 1.0
 * @date 2022-10-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once
#include <float.h>
#include "Eigen/Eigen"
#include "mpc_params.h"
#include <math.h>
#include "deque"
#include "lbfgs.hpp"
#include <chrono>
namespace PayloadMPC
{

  class MultiOptForceEstimator
  {
    typedef struct
    {
      Eigen::Vector3d total_force;
      Eigen::Vector3d C;
      Eigen::Vector3d B;
      Eigen::Vector3d cable;
      int USE_CONSTANT_MOMENT;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } systemState;

    inline static double huber_loss(const double residual, double &gradient, const double delta = 1.0)
    {
      double huber_cost = 0.0;
      double abs_r = fabs(residual);

      if (abs_r <= delta)
      {
        huber_cost = 0.5 * residual * residual;
        gradient = residual;
      }
      else
      {
        huber_cost = delta * (abs_r - 0.5 * delta);
        gradient = delta * (residual > 0.0 ? 1.0 : -1.0);
      }
      return huber_cost;
    }

  inline static void distanceSqrVarianceWithGradCost2p(const double wei_sqrvar_, const Eigen::Matrix3Xd &ps,
                                                            Eigen::Matrix3Xd &gdp,
                                                            double &var)
  {
    int N = ps.cols() - 1;
    Eigen::Matrix3Xd dps = 
    ps.rightCols(N) - ps.leftCols(N);
    Eigen::VectorXd dsqrs = dps.colwise().squaredNorm().transpose();
    double dsqrsum = dsqrs.sum();
    // double dquarsum = dsqrs.squaredNorm();
    // double dsqrmean = dsqrsum / N;
    // double dquarmean = dquarsum / N;
    var = wei_sqrvar_ * dsqrsum;
    gdp.resize(3, N + 1);
    gdp.setZero();
    for (int i = 0; i <= N; i++)
    {
      if (i != 0)
      {
        gdp.col(i) += wei_sqrvar_ * 2.0 * dps.col(i - 1);
      }
      if (i != N)
      {
        gdp.col(i) += (-2.0)*wei_sqrvar_ * dps.col(i);
      }
    }
    return;
  }

    static double cost_function(void *instance, const Eigen::VectorXd &x, Eigen::VectorXd &g)
    {
      double cost = 0;
      MultiOptForceEstimator &estimator = *(MultiOptForceEstimator *)instance;
      Eigen::Map<const Eigen::Matrix3Xd> fq_array(x.data(), 3, estimator.param_num_);
      Eigen::Map<const Eigen::Matrix3Xd> fl_array(x.data() + 3 * estimator.param_num_, 3, estimator.param_num_);

      Eigen::Map<Eigen::Matrix3Xd> fq_gradient(g.data(), 3, estimator.param_num_);
      Eigen::Map<Eigen::Matrix3Xd> fl_gradient(g.data() + 3 * estimator.param_num_, 3, estimator.param_num_);
      fq_gradient.setZero();
      fl_gradient.setZero();

      Eigen::Matrix3Xd fq_gradient_var, fl_gradient_var;
      double cost_var = 0;

      distanceSqrVarianceWithGradCost2p(estimator.var_weight_, fq_array, fq_gradient_var, cost_var);
      cost += cost_var;
      fq_gradient += fq_gradient_var;
      
      cost_var=0;
      distanceSqrVarianceWithGradCost2p(estimator.var_weight_, fl_array, fl_gradient_var, cost_var);
      cost += cost_var;
      fl_gradient += fl_gradient_var;

      int i = 0;
      for (const systemState &state_ : estimator.state_buffer)
      {
        Eigen::Vector3d local_fq, local_fl;
        Eigen::Vector3d local_fq_grad(0, 0, 0), local_fl_grad(0, 0, 0);
        Eigen::Vector3d fl = fl_array.col(i);
        Eigen::Vector3d fq = fq_array.col(i);
        if (state_.USE_CONSTANT_MOMENT == 1)
        {
          local_fl[0] = fl[1] * (state_.cable[2]) - fl[2] * (state_.cable[1]);
          local_fl[1] = fl[2] * (state_.cable[0]) - fl[0] * (state_.cable[2]);
          local_fl[2] = fl[0] * (state_.cable[1]) - fl[1] * (state_.cable[0]);
          local_fq[0] = fq[0];
          local_fq[1] = fq[1];
          local_fq[2] = fq[2];
          double res = (fl[0] * (state_.cable[0]) + fl[1] * (state_.cable[1]) + fl[2] * (state_.cable[2]));
          double huber_gradient;
          cost += huber_loss(res, huber_gradient);
          fl_gradient.col(i) += huber_gradient * state_.cable;
        }
        else if (state_.USE_CONSTANT_MOMENT == 2)
        {
          local_fq[0] = fq[1] * (state_.cable[2]) - fq[2] * (state_.cable[1]);
          local_fq[1] = fq[2] * (state_.cable[0]) - fq[0] * (state_.cable[2]);
          local_fq[2] = fq[0] * (state_.cable[1]) - fq[1] * (state_.cable[0]);
          local_fl[0] = fl[0];
          local_fl[1] = fl[1];
          local_fl[2] = fl[2];
          double res = fq[0] * (state_.cable[0]) + fq[1] * (state_.cable[1]) + fq[2] * (state_.cable[2]);
          double huber_gradient;
          cost += huber_loss(res, huber_gradient);
          fq_gradient.col(i) += huber_gradient * state_.cable;
        }
        else
        {
          local_fq[0] = fq[0];
          local_fq[1] = fq[1];
          local_fq[2] = fq[2];
          local_fl[0] = fl[0];
          local_fl[1] = fl[1];
          local_fl[2] = fl[2];
          // residuals[9] = T(0);
          // residuals[6] = local_fq[0] * T(state_.cable[0]) + local_fq[1] * T(state_.cable[1]) + local_fq[2] * T(state_.cable[2]);
          double res = fl[0] * (state_.cable[0]) + fl[1] * (state_.cable[1]) + fl[2] * (state_.cable[2]);
          double huber_gradient;
          cost += huber_loss(res, huber_gradient);
          fl_gradient.col(i) += huber_gradient * state_.cable;
        }

        Eigen::Vector3d r1 = local_fl + local_fq - state_.total_force;
        double huber_gradient;
        cost += huber_loss(r1[0], huber_gradient);
        local_fl_grad(0) += huber_gradient;
        local_fq_grad(0) += huber_gradient;
        cost += huber_loss(r1[1], huber_gradient);
        local_fl_grad(1) += huber_gradient;
        local_fq_grad(1) += huber_gradient;
        cost += huber_loss(r1[2], huber_gradient);
        local_fl_grad(2) += huber_gradient;
        local_fq_grad(2) += huber_gradient;

        double r2[3];

        r2[0] = (local_fq[2] + (state_.C[2])) * (state_.cable[1]) - (local_fq[1] + (state_.C[1])) * (state_.cable[2]);
        cost += huber_loss(r2[0], huber_gradient);
        local_fq_grad(2) += huber_gradient * state_.cable[1];
        local_fq_grad(1) -= huber_gradient * state_.cable[2];

        r2[1] = (local_fq[0] + (state_.C[0])) * (state_.cable[2]) - (local_fq[2] + (state_.C[2])) * (state_.cable[0]);
        cost += huber_loss(r2[1], huber_gradient);
        local_fq_grad(0) += huber_gradient * state_.cable[2];
        local_fq_grad(2) -= huber_gradient * state_.cable[0];

        r2[2] = (local_fq[1] + (state_.C[1])) * (state_.cable[0]) - (local_fq[0] + (state_.C[0])) * (state_.cable[1]);
        cost += huber_loss(r2[2], huber_gradient);
        local_fq_grad(1) += huber_gradient * state_.cable[0];
        local_fq_grad(0) -= huber_gradient * state_.cable[1];
        if (state_.USE_CONSTANT_MOMENT == 1)
        {
          // local_fl[0] = fl[1] * (state_.cable[2]) - fl[2] * (state_.cable[1]);
          // local_fl[1] = fl[2] * (state_.cable[0]) - fl[0] * (state_.cable[2]);
          // local_fl[2] = fl[0] * (state_.cable[1]) - fl[1] * (state_.cable[0]);
          fq_gradient.col(i) += local_fq_grad;
          fl_gradient(1, i) += local_fl_grad(0) * state_.cable[2];
          fl_gradient(2, i) -= local_fl_grad(0) * state_.cable[1];
          fl_gradient(2, i) += local_fl_grad(1) * state_.cable[0];
          fl_gradient(0, i) -= local_fl_grad(1) * state_.cable[2];
          fl_gradient(0, i) += local_fl_grad(2) * state_.cable[1];
          fl_gradient(1, i) -= local_fl_grad(2) * state_.cable[0];
        }
        else if (state_.USE_CONSTANT_MOMENT == 2)
        {
          // local_fq[0] = fq[1] * (state_.cable[2]) - fq[2] * (state_.cable[1]);
          // local_fq[1] = fq[2] * (state_.cable[0]) - fq[0] * (state_.cable[2]);
          // local_fq[2] = fq[0] * (state_.cable[1]) - fq[1] * (state_.cable[0]);
          fq_gradient(1, i) += local_fq_grad(0) * state_.cable[2];
          fq_gradient(2, i) -= local_fq_grad(0) * state_.cable[1];
          fq_gradient(2, i) += local_fq_grad(1) * state_.cable[0];
          fq_gradient(0, i) -= local_fq_grad(1) * state_.cable[2];
          fq_gradient(0, i) += local_fq_grad(2) * state_.cable[1];
          fq_gradient(1, i) -= local_fq_grad(2) * state_.cable[0];
          fl_gradient.col(i) += local_fl_grad;
        }
        else
        {
          fq_gradient.col(i) += local_fq_grad;
          fl_gradient.col(i) += local_fl_grad;
        }

        i++;
      }
      return cost;
    }

    Eigen::Vector3d opt_fq_, opt_fl_;
    Eigen::VectorXd opt_variable;
    double var_weight_{50.0};

  public:
    Eigen::Vector3d quad_acc_;
    Eigen::Vector3d load_acc_;
    Eigen::Vector3d cable_;
    Eigen::Vector4d rpm_;
    Eigen::Quaterniond quad_q_;
    Eigen::Vector3d Thr_;
    double T_, sqrt_kf_;

    std::deque<systemState> state_buffer;

    LowPassFilter2p<Eigen::Vector3d> fq_filter_;
    LowPassFilter2p<Eigen::Vector3d> fl_filter_;

    double mass_quad_, mass_load_, g_, imu_body_length_, l_length_, imu_load_length_, Thrust_;
    double sample_freq_fq_{333.333}, sample_freq_fl_{333.333};
    double cutoff_freq_fq_{100.0}, cutoff_freq_fl_{100.0};

    double max_force_, max_force_sqr_;
    bool use_force_estimator_;
    int USE_CONSTANT_MOMENT_;
    int queue_size_;

    int param_num_;

  public:
    void init(MpcParams &params)
    {
      mass_quad_ = params.dyn_params_.mass_q;
      mass_load_ = params.dyn_params_.mass_l;
      l_length_ = params.dyn_params_.l_length;
      imu_body_length_ = params.force_estimator_param_.imu_body_length;
      imu_load_length_ = l_length_ - imu_body_length_;

      sqrt_kf_ = params.force_estimator_param_.sqrt_kf;
      use_force_estimator_ = params.force_estimator_param_.use_force_estimator;
      max_force_ = params.force_estimator_param_.max_force;
      max_force_sqr_ = max_force_ * max_force_;

      sample_freq_fq_ = params.force_estimator_param_.sample_freq_fq;
      sample_freq_fl_ = params.force_estimator_param_.sample_freq_fl;
      cutoff_freq_fq_ = params.force_estimator_param_.cutoff_freq_fq;
      cutoff_freq_fl_ = params.force_estimator_param_.cutoff_freq_fl;

      USE_CONSTANT_MOMENT_ = params.force_estimator_param_.USE_CONSTANT_MOMENT;
      var_weight_ = params.force_estimator_param_.var_weight;

      g_ = params.gravity_;

      fq_filter_.set_cutoff_frequency(sample_freq_fq_, cutoff_freq_fq_);
      fl_filter_.set_cutoff_frequency(sample_freq_fl_, cutoff_freq_fl_);
      fq_filter_.reset(Eigen::Vector3d::Zero());
      fl_filter_.reset(Eigen::Vector3d::Zero());

      queue_size_ = params.force_estimator_param_.max_queue;
      // all_opt_fl_.resize(queue_size_,3);
      // all_opt_fq_.resize(queue_size_,3);
      opt_variable.resize(2 * 3 * (queue_size_ + 1));
      opt_variable.setZero();
    }

    void enableForceEstimator()
    {
      use_force_estimator_ = true;
    }
    void disableForceEstimator()
    {
      use_force_estimator_ = false;
    }

    /**
     * @brief 设置系统状态并更新力估计缓冲区
     * 
     * @param[in] quad_acc_body 四旋翼在机体坐标系下的加速度(包含重力)
     * @param[in] Rotwb 四旋翼从机体到世界坐标系的旋转四元数
     * @param[in] load_acc_body 负载在机体坐标系下的加速度(包含重力)
     * @param[in] load_Rotwb 负载从机体到世界坐标系的旋转四元数
     * @param[in] cable 缆绳方向向量(世界坐标系)
     * @param[in] Rpm 四旋翼电机转速(RPM)
     * 
     * @note 功能流程:
     * 1. 将加速度转换到世界坐标系
     * 2. 计算负载加速度补偿(考虑IMU安装位置)
     * 3. 计算推力大小和方向
     * 4. 构建系统状态结构体:
     *    - C: 推力与四旋翼惯性力的差值
     *    - total_force: 系统总外力(四旋翼惯性力+负载惯性力-推力)
     *    - cable: 缆绳方向
     *    - USE_CONSTANT_MOMENT: 力矩约束模式
     * 5. 将状态加入缓冲区队列，保持队列长度不超过限制
     * 
     * @warning 调用频率应与传感器更新频率一致
     * @see caculate_force
     */
    void setSystemState(const Eigen::Vector3d &quad_acc_body, const Eigen::Quaterniond &Rotwb, const Eigen::Vector3d &load_acc_body, const Eigen::Quaterniond &load_Rotwb, const Eigen::Vector3d cable, const Eigen::Vector4d &Rpm)
    {
      quad_acc_ = Rotwb * quad_acc_body; //include gravity
      // quad_acc_(2) -= g_;
      quad_q_ = Rotwb;

      load_acc_ = load_Rotwb * load_acc_body; //include gravity
      // load_acc_(2) -= g_;

      load_acc_ = (load_acc_ - quad_acc_) / imu_body_length_ * l_length_ + quad_acc_;

      cable_ = cable;

      rpm_ = Rpm;
      T_ = (sqrt_kf_ * rpm_).squaredNorm();
      Thr_ = T_ * quad_q_.toRotationMatrix().col(2);

      systemState state;
      // state.C = Thr_ - mass_quad_ * (Eigen::Vector3d(0, 0, g_) + quad_acc_);
      // state.B = -mass_load_ * (Eigen::Vector3d(0, 0, g_) + load_acc_);
      // state.total_force = mass_quad_ * (Eigen::Vector3d(0, 0, g_) + quad_acc_) + mass_load_ * (Eigen::Vector3d(0, 0, g_) + load_acc_) - Thr_;

      state.C = Thr_ - mass_quad_ *  quad_acc_;
      state.total_force = mass_quad_ * quad_acc_ + mass_load_ * load_acc_ - Thr_;
      state.cable = cable;
      state.USE_CONSTANT_MOMENT = USE_CONSTANT_MOMENT_;
      // std::cout<<"state.cable: "<<state.cable.transpose()<<std::endl;

      state_buffer.emplace_back(state);
      while (state_buffer.size() > queue_size_)
      {
        state_buffer.pop_front();
      }
    }

    /**
    * @brief 多状态优化的力估计核心计算函数（最小化力变化方差）
    * 
    * @param[out] fl  负载外力估计值（三维向量，单位：牛顿）
    * @param[out] fq  四旋翼外力估计值（三维向量，单位：牛顿）
    * 
    * @note 算法流程：
    * 1. 状态缓冲区检查：当缓冲区状态数≤2时，直接返回零向量（需至少3个状态点计算方差）
    * 2. 优化变量调整：根据当前状态数(param_num_)动态调整优化变量维度（2*3*N）
    * 3. 边界条件预处理：当优化变量超出3倍max_force_时重置为零（异常值处理）
    * 4. L-BFGS优化计算：使用32记忆体的L-BFGS算法进行500次迭代优化
    * 5. 结果后处理：
    *    - 提取最新时刻的力估计值（param_num_-1索引）
    *    - 力矢量归一化处理（超过max_force_时限制幅值）
    *    - 应用低通滤波器（截止频率由cutoff_freq_*参数控制）
    * 
    * @note 核心优化目标：
    * - 最小化相邻状态点间的力变化方差（通过var_weight_调节优化强度）
    * - 满足力矩平衡方程：local_fl + local_fq = total_force
    * - 满足Cable方向约束（根据USE_CONSTANT_MOMENT_模式不同） 
    * 
    * @warning 重要说明：
    * - 需确保调用前已通过setSystemState维护状态缓冲区（建议队列长度≥3）
    * - 最大外力限制max_force_需与物理系统实际承受能力匹配
    * - 滤波器参数(sample_freq_* cutoff_freq_*)需与系统实际频率一致
    * 
    * @remark 典型调用场景：
    * 在控制器主循环中实时调用，建议调用频率与状态更新频率保持一致
    * 需配合MPC参数中的force_estimator_param配置使用
    */
    void caculate_force(Eigen::Vector3d &fl, Eigen::Vector3d &fq)
    {
      if (use_force_estimator_)
      {
        if (state_buffer.size() <= 2)       // 当状态缓冲区中的数据点少于等于2个时，不进行力估计计算。因为优化需要至少3个数据点才能计算力变化的方差
        {
          fl = Eigen::Vector3d::Zero();
          fq = Eigen::Vector3d::Zero();
        }
        else
        {
          // auto start = std::chrono::steady_clock::now();
          param_num_ = state_buffer.size();
          opt_variable.conservativeResize(2 * 3 * param_num_); // 调整优化变量的大小为2*3*param_num_，即每个状态点有6个变量(3个四旋翼力+3个负载力)
          if (opt_variable.maxCoeff() > 3 * max_force_ || opt_variable.minCoeff() < -3 * max_force_)  // 检查优化变量是否超出3倍最大力限制，如果超出则重置为零
          {
            opt_variable.setZero();
          }

          // debug code (gradient check)
          /*Eigen::VectorXd opt_variable_debug = opt_variable;
          Eigen::VectorXd grad;
          grad.resizeLike(opt_variable);

          Eigen::Map<Eigen::Matrix3Xd> opt_fq(opt_variable_debug.data(), 3, param_num_);
          Eigen::Map<Eigen::Matrix3Xd> opt_fl(opt_variable_debug.data() + 3 * param_num_, 3, param_num_);

          Eigen::Map<Eigen::Matrix3Xd> grad_fq(grad.data(), 3, param_num_);
          Eigen::Map<Eigen::Matrix3Xd> grad_fl(grad.data() + 3 * param_num_, 3, param_num_);
          double ori_cost = 0;
          ori_cost = MultiOptForceEstimator::cost_function(this,opt_variable_debug,grad);
          std::cout <<"opt_variable fq" <<std::endl << opt_fq << std::endl;
          std::cout <<"opt_variable fl" <<std::endl << opt_fl << std::endl;
          std::cout <<"grad fq"<<std::endl << grad_fq << std::endl;
          std::cout <<"grad fl"<<std::endl << grad_fl << std::endl;

          Eigen::VectorXd num_grad,temp_grad;
          num_grad.resizeLike(opt_variable);
          temp_grad.resizeLike(opt_variable);
          const double eps = 1e-6;
          for (int i = 0; i < num_grad.size(); i++)
          {
            opt_variable_debug = opt_variable;
            opt_variable_debug(i) += eps;
            double num_cost;
            num_cost = MultiOptForceEstimator::cost_function(this, opt_variable_debug,temp_grad);
            num_grad(i) = (num_cost - ori_cost) / eps;
          }
          Eigen::Map<Eigen::Matrix3Xd> num_grad_fq(num_grad.data(), 3, param_num_);
          Eigen::Map<Eigen::Matrix3Xd> num_grad_fl(num_grad.data() + 3 * param_num_, 3, param_num_);
          std::cout<<std::endl  <<"num_grad fq"<<std::endl << num_grad_fq << std::endl;
          std::cout<<std::endl  <<"num_grad fl"<<std::endl  << num_grad_fl << std::endl;*/

          // end debug code

          lbfgs::lbfgs_parameter_t opt_param;
          opt_param.mem_size = 32;
          opt_param.g_epsilon = 1.0e-5;
          opt_param.delta = 1.0e-6;
          opt_param.past = 0;
          opt_param.max_iterations = 500;
          double final_cost;

          int error_code = lbfgs::lbfgs_optimize(opt_variable, final_cost, MultiOptForceEstimator::cost_function, nullptr, nullptr, this, opt_param);
          // auto end = std::chrono::steady_clock::now();
          // std::cout << "Elapsed time in microseconds: "
          // << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
          // << " µs" << std::endl;
          if (error_code < 0)
          {
            if (error_code == lbfgs::LBFGSERR_MAXIMUMITERATION)
            {
              
            }
            else
            {
              std::cout << "[Force estimator]error code: " << std::string(lbfgs::lbfgs_strerror(error_code)) << std::endl;
            }
          
          }
          // else
          // {
          //   std::cout<<"LBFGS final cost: "<<final_cost<<std::endl;
          // }

          // 将优化变量opt_variable映射到3xN矩阵，分别表示quadrotor和load的力估计
          Eigen::Map<Eigen::Matrix3Xd> all_opt_fq_(opt_variable.data(), 3, param_num_);     // all_opt_fq_: 3xN矩阵，存储quadrotor所受外力的历史估计值
          Eigen::Map<Eigen::Matrix3Xd> all_opt_fl_(opt_variable.data() + 3 * param_num_, 3, param_num_);    // all_opt_fl_: 3xN矩阵，存储load所受外力的历史估计值

          opt_fl_ = all_opt_fl_.col(param_num_ - 1);  // load最新外力估计
          opt_fq_ = all_opt_fq_.col(param_num_ - 1);  // quadrotor最新外力估计


          if (opt_fl_.squaredNorm() > max_force_sqr_)    // 检查load外力是否超过最大限制，如果超过限制，则归一化并缩放到最大允许值
          {
            opt_fl_ = opt_fl_.normalized() * max_force_;
            all_opt_fl_.setZero();
          }
          if (opt_fq_.squaredNorm() > max_force_sqr_) // 检查quadrotor外力是否超过最大限制，如果超过限制，则归一化并缩放到最大允许值      //有点问题啊，这外力又不是听你话的，你说不超过最大限制就不超过啊。
          {
            opt_fq_ = opt_fq_.normalized() * max_force_;
            all_opt_fq_.setZero();
          }
        }
        opt_fl_ = fl_filter_.apply(opt_fl_);          // 对负载外力估计值进行低通滤波处理，平滑噪声
        opt_fq_ = fq_filter_.apply(opt_fq_);          // 对四旋翼外力估计值进行低通滤波处理，平滑噪声 
        fl = opt_fl_;
        fq = opt_fq_;
      }
      else
      {
        fl = Eigen::Vector3d::Zero();
        fq = Eigen::Vector3d::Zero();
      }
    }
  };

} // namespace PayloadMPC
