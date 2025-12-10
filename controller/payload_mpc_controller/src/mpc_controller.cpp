// ================================================================
// 文件：mpc_controller.cpp
// 职责：MPC算法核心实现
// 主要功能：（有待确认）
//   - 吊挂负载动力学建模 `setHoverReference`方法
//   - 参考轨迹生成与插值 `setTrajectoyReference`方法
//   - 推力-转速映射模型 `convertThrust`方法
//   - 偏航角规划与速率限制 `calculate_yaw`方法
//   - 实时求解状态预测 `execMPC`方法
// 通用特性说明
// 1. 矩阵维度规范：
//    - 状态矩阵：kStateSize=23 (位置/姿态/速度/负载状态等)
//    - 输入矩阵：kInputSize=4 (推力/角速度指令)
// 2. 时间参数：
//    - mpc_time_step_ 控制时间分辨率
//    - kSamples 预测时域步数
// ================================================================
#include "mpc_controller.h"
#include <nav_msgs/Odometry.h>
#include <ctime>

namespace PayloadMPC
{
  /**
  * @brief 构造函数：MPC控制器初始化
  * @param params MPC参数配置(权重/动力学参数/限制条件等)
  * 功能：构建MPC求解环境
  *   - 初始化状态/输入代价矩阵Q,R
  *   - 设置无人机-负载动力学参数
  *   - 构建悬停状态初始参考轨迹
  *   - 启动MPC求解准备线程
  **/
  MpcController::MpcController(MpcParams &params) : params_(params), mpc_time_step_(params_.step_T_)
  {
    auto & q_gain_ = params_.q_gain_;
    auto & r_gain_ = params_.r_gain_;

    // 设置动力学参数（无人机质量、负载质量、绳长）
    mpc_wrapper_.setDynamicParams(params_.dyn_params_.mass_q,params_.dyn_params_.mass_l,params_.dyn_params_.l_length);
    Eigen::Matrix<real_t, kCostSize, kCostSize> Q = (Eigen::Matrix<real_t, kCostSize, 1>() << q_gain_.Q_pos_xy, q_gain_.Q_pos_xy, q_gain_.Q_pos_z,
                                                     q_gain_.Q_attitude_rp, q_gain_.Q_attitude_rp, q_gain_.Q_attitude_rp, q_gain_.Q_attitude_yaw,
                                                     q_gain_.Q_velocity, q_gain_.Q_velocity, q_gain_.Q_velocity,
                                                     q_gain_.Q_payload_xy, q_gain_.Q_payload_xy, q_gain_.Q_payload_z,
                                                     q_gain_.Q_payload_velocity, q_gain_.Q_payload_velocity, q_gain_.Q_payload_velocity,
                                                     q_gain_.Q_cable, q_gain_.Q_cable, q_gain_.Q_cable,
                                                     q_gain_.Q_dcable, q_gain_.Q_dcable, q_gain_.Q_dcable)
                                                        .finished()
                                                        .asDiagonal();
    Eigen::Matrix<real_t, kInputSize, kInputSize> R = (Eigen::Matrix<real_t, kInputSize, 1>() << r_gain_.R_thrust, r_gain_.R_pitchroll, r_gain_.R_pitchroll, r_gain_.R_yaw).finished().asDiagonal();

    Eigen::Matrix<real_t, kStateSize, 1> initial_state = (Eigen::Matrix<real_t, kStateSize, 1>() << 0.0, 0.0, params_.dyn_params_.l_length,
                                                          1.0, 0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, 0.0,
                                                          0.0, 0.0, -1.0,
                                                          0.0, 0.0, 0.0)
                                                             .finished();
    reference_states_ = initial_state.replicate(1,kSamples+1);
    Eigen::Matrix<real_t, kInputSize, 1> initial_input = (Eigen::Matrix<real_t, kInputSize, 1>() << (params_.dyn_params_.mass_l + params_.dyn_params_.mass_q) * params_.gravity_, 0, 0, 0).finished();

    hover_input_ = initial_input.replicate(1, kSamples + 1);
    reference_inputs_ = hover_input_;

    mpc_wrapper_.initialize(Q, R, initial_state, initial_input, params_.state_cost_exponential_, params_.input_cost_exponential_);
    mpc_wrapper_.setExternalForce(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
    mpc_wrapper_.setLimits(
        params_.min_thrust_, params_.max_thrust_,
        params_.max_bodyrate_xy_, params_.max_bodyrate_z_);
    // 重置吊挂动力学模型
    payload_flat_.reset(params_.dyn_params_.mass_q, params_.dyn_params_.mass_l, params_.dyn_params_.l_length, params_.gravity_);

    // first_traj_received_ = false;
    solve_from_scratch_ = false;
    timing_feedback_ = 0;
    timing_preparation_ = 0;
    preparation_thread_ = std::thread(&MpcWrapper::prepare, mpc_wrapper_);
    
  }
  /**
  * @brief 主求解函数：执行MPC滚动优化
  * @param reference_states 参考状态轨迹矩阵[kStateSize x (kSamples+1)]
  * @param reference_inputs 参考输入轨迹矩阵[kInputSize x (kSamples+1)] 
  * @param estimated_state 当前状态估计值[kStateSize x 1]
  * @param[out] predicted_states 输出预测状态序列
  * @param[out] control_inputs 输出最优控制输入序列
  * 功能：实时求解MPC问题
  *   - 同步准备线程
  *   - 注入参考轨迹
  *   - 求解优化问题
  *   - 启动异步准备线程
  **/
  void MpcController::execMPC(const Eigen::Matrix<real_t, kStateSize, kSamples + 1> &reference_states,
                              const Eigen::Matrix<real_t, kInputSize, kSamples + 1> &reference_inputs,
                              const Eigen::Matrix<real_t, kStateSize, 1> &estimated_state,
                              Eigen::Matrix<real_t, kStateSize, kSamples + 1> &predicted_states,
                              Eigen::Matrix<real_t, kInputSize, kSamples> &control_inputs)
  {
    const clock_t start = clock();

    preparation_thread_.join(); // waiting the preparation_thread_ finished

    // Get the feedback from MPC.

    mpc_wrapper_.setTrajectory(reference_states, reference_inputs);

    if (solve_from_scratch_)
    {
      ROS_INFO("Solving MPC with hover as initial guess.");
      mpc_wrapper_.solve(estimated_state);
      solve_from_scratch_ = false;
    }
    else
    {
      constexpr bool do_preparation_step(false); // the preparation step has been done by another thread
      mpc_wrapper_.update(estimated_state, do_preparation_step);
    }

    mpc_wrapper_.getStates(predicted_states);
    mpc_wrapper_.getInputs(control_inputs);

    // Start a thread to prepare for the next execution.
    preparation_thread_ = std::thread(&MpcController::preparationThread, this);

    // Timing
    const clock_t end = clock();
    timing_feedback_ = 0.9*timing_feedback_ + 0.1* double(end - start) / CLOCKS_PER_SEC;
    if (params_.print_info_)
    ROS_INFO_THROTTLE(1.0, "MPC Timing: Latency: %1.2f ms  |  Total: %1.2f ms",
                      timing_feedback_ * 1000, (timing_feedback_ + timing_preparation_) * 1000);
  }

  void MpcController::execMPC(const Eigen::Matrix<real_t, kStateSize, 1> &estimated_state,
                              Eigen::Matrix<real_t, kStateSize, kSamples + 1> &predicted_states,
                              Eigen::Matrix<real_t, kInputSize, kSamples> &control_inputs)
  {
    execMPC(reference_states_, reference_inputs_, estimated_state, predicted_states, control_inputs);
  }

  /**
  * @brief 悬停参考：设置悬停状态参考轨迹
  * @param quad_position 无人机期望位置
  * @param yaw 期望偏航角
  * 功能：生成悬停状态参考
  *   - 通过负载平坦化模型计算悬停姿态
  *   - 构建全零速度/负载位置参考
  *   - 初始化推力输入为重力补偿
  **/
  void MpcController::setHoverReference(const Eigen::Ref<const Eigen::Vector3d> &quad_position, const double yaw)
  {
    real_t psi = yaw;
    Eigen::Matrix<real_t, 3, 1> quad_pos = quad_position.cast<real_t>();
    Eigen::Vector3d local_fq = fq_;
    Eigen::Vector3d local_fl = fl_;
    if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 2)
    {
      local_fq = fq_.cross(Eigen::Vector3d(0,0,-1));
    }
    else if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 1)
    {
      local_fl = fl_.cross(Eigen::Vector3d(0,0,-1));
    }
    Eigen::Quaterniond quad_q;
    double thr;
    Eigen::Vector3d omg_;
    // 通过负载平坦化模型计算无人机姿态
    payload_flat_.forward_rot(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),local_fq,local_fl,psi,0,quad_q,thr,omg_);
    Eigen::Quaternion<real_t> q = quad_q.cast<real_t>();
    
    // Eigen::Quaternion<real_t> q(Eigen::AngleAxis<real_t>(psi, Eigen::Matrix<real_t, 3, 1>::UnitZ()));

    real_t payload_z = quad_pos.z() - params_.dyn_params_.l_length;
    Eigen::Matrix<real_t, kStateSize, 1> reference_state;
    q.normalize();
    reference_state << quad_pos.x(), quad_pos.y(), quad_pos.z(),
        q.w(), q.x(), q.y(), q.z(),
        0.0, 0.0, 0.0,
        quad_pos.x(), quad_pos.y(), payload_z,
        0.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 0.0, 0.0;

    Eigen::Matrix<real_t, kInputSize, 1> hover_in = hover_input_.col(0);
    hover_in(kThrust,0)=thr;
    reference_states_ = reference_state.replicate(1, kSamples + 1);
    reference_inputs_ = hover_in.replicate(1, kSamples + 1);
  }
  /** 
  * @brief 轨迹跟踪：注入轨迹参考
  * @param traj 轨迹对象（需实现getPVAJSC接口）
  * @param tstart 轨迹起始时间
  * @param start_yaw 初始偏航角
  * 功能：轨迹离散化处理
  *   - 按MPC时间步长采样轨迹点
  *   - 通过负载平坦化模型前馈计算
  *   - 生成状态-输入参考序列
  *   - 处理偏航角速率约束
  **/
  void MpcController::setTrajectoyReference(Trajectory &traj, double tstart, double start_yaw)
  {
    const double t_step = mpc_time_step_;
    double t_all = traj.getTotalDuration() - 1.0e-3;
    double t = tstart;
    double yaw, yaw_dot;
    Eigen::Vector3d pos, vel, acc, jerk, snap, crackle;

    Eigen::Vector3d pos_quad, vel_quad, acc_quad, jerk_quad, p, pd;
    Eigen::Quaterniond quat;
    double thr;
    Eigen::Vector3d omg;

    // last_yaw_ = start_yaw;  // must reset the last_yaw_ 

    for (int i = 0; i < (kSamples + 1); i++)
    {
      Eigen::MatrixXd pvajs;
      if (t > t_all)
      { // if t is larger than the total time, use the last point
        // TODO : Consider 2 trajectories
        t = t_all;
        pvajs = traj.getPVAJSC(t);
        pos = pvajs.col(0);
        vel = Eigen::Vector3d::Zero();
        acc =  Eigen::Vector3d::Zero();
        jerk = Eigen::Vector3d::Zero();
        snap = Eigen::Vector3d::Zero();
        crackle =  Eigen::Vector3d::Zero();
      }
      else
      {
        pvajs = traj.getPVAJSC(t);
        pos = pvajs.col(0);
        vel = pvajs.col(1);
        acc = pvajs.col(2);
        jerk = pvajs.col(3);
        snap = pvajs.col(4);
        crackle = pvajs.col(5);
      }
      // 通过负载平坦化模型计算无人机位置
      payload_flat_.forward_p(pos, vel, acc, jerk, snap, crackle, pos_quad, vel_quad, acc_quad, jerk_quad, p, pd);
      if (params_.use_fix_yaw_)
      {
        yaw = start_yaw;
        yaw_dot = 0.0; 
      }
      else
      {
        if(i == 0) // reset last_yaw_ from the first point
        {
          if(t - t_step <= 0.0)
          {
            if (fabs(vel_quad(0))+fabs(vel_quad(1))>0.1)
            {
              last_yaw_ = atan2(vel_quad(1), vel_quad(0));
              last_yaw_dot_ = 0.0;
            }
            else
            {
              last_yaw_ = start_yaw;
              last_yaw_dot_ = 0.0;
            }
          }
          else // Get yaw in last step 
          {
            Eigen::MatrixXd last_pvajs = traj.getPVAJSC(t-t_step);
            Eigen::Vector3d last_pos= last_pvajs.col(0);
            Eigen::Vector3d last_vel = last_pvajs.col(1);
            Eigen::Vector3d last_acc = last_pvajs.col(2);
            Eigen::Vector3d last_jerk = last_pvajs.col(3);
            Eigen::Vector3d last_snap = last_pvajs.col(4);
            Eigen::Vector3d last_crackle = last_pvajs.col(5); 
            Eigen::Vector3d last_pos_quad, last_vel_quad, last_acc_quad, last_jerk_quad, last_p, last_pd;

            payload_flat_.forward_p(last_pos, last_vel, last_acc, last_jerk, last_snap, last_crackle, last_pos_quad, last_vel_quad, last_acc_quad, last_jerk_quad, last_p, last_pd);

            if (fabs(last_vel_quad(0))+fabs(last_vel_quad(1))>0.1)
            {
              last_yaw_ = atan2(last_vel_quad(1), last_vel_quad(0));
              last_yaw_dot_ = 0.0;
            }
            else
            {
              last_yaw_ = start_yaw;
              last_yaw_dot_ = 0.0;
            }
          }
        }
        calculate_yaw(vel_quad, t_step,yaw, yaw_dot);  
      }
      Eigen::Vector3d local_fq = fq_;
      Eigen::Vector3d local_fl = fl_;
      if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 2)
      {
        local_fq = fq_.cross(p);
      }
      else if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 1)
      {
        local_fl = fl_.cross(p);
      }
      
      payload_flat_.forward_rot(acc,jerk, acc_quad, jerk_quad,local_fq,local_fl,yaw,yaw_dot,quat,thr,omg);

      Eigen::Matrix<real_t, kStateSize, 1> reference_state;
      reference_state << pos_quad.x(), pos_quad.y(), pos_quad.z(),
          quat.w(), quat.x(), quat.y(), quat.z(),
          vel_quad.x(), vel_quad.y(), vel_quad.z(),
          pos.x(), pos.y(), pos.z(),
          vel.x(), vel.y(), vel.z(),
          p.x(), p.y(), p.z(),
          pd.x(), pd.y(), pd.z();
      reference_states_.col(i) = reference_state;

      Eigen::Matrix<real_t, kInputSize, 1> reference_input;
      // reference_input << thr, omg.x(), omg.y(), omg.z();
      reference_input << thr, omg.x(), omg.y(), 0.0;
      reference_inputs_.col(i) = reference_input;
      t += t_step;
    }
  }

  void MpcController::setAnalyticReference(const std::function<Eigen::Matrix<double, 3, 6>(double)> &trajectory_fn,
                                           double tstart,
                                           double start_yaw)
  {
    const double t_step = mpc_time_step_;
    double t = tstart;
    double yaw, yaw_dot;
    Eigen::Vector3d pos, vel, acc, jerk, snap, crackle;

    Eigen::Vector3d pos_quad, vel_quad, acc_quad, jerk_quad, p, pd;
    Eigen::Quaterniond quat;
    double thr;
    Eigen::Vector3d omg;

    for (int i = 0; i < (kSamples + 1); i++)
    {
      const double eval_t = t > 0.0 ? t : 0.0;
      Eigen::Matrix<double, 3, 6> pvajs = trajectory_fn(eval_t);

      pos = pvajs.col(0);
      vel = pvajs.col(1);
      acc = pvajs.col(2);
      jerk = pvajs.col(3);
      snap = pvajs.col(4);
      crackle = pvajs.col(5);

      payload_flat_.forward_p(pos, vel, acc, jerk, snap, crackle, pos_quad, vel_quad, acc_quad, jerk_quad, p, pd);
      if (params_.use_fix_yaw_)
      {
        yaw = start_yaw;
        yaw_dot = 0.0;
      }
      else
      {
        if (i == 0)
        {
          if (t - t_step <= 0.0)
          {
            if (fabs(vel_quad(0)) + fabs(vel_quad(1)) > 0.1)
            {
              last_yaw_ = atan2(vel_quad(1), vel_quad(0));
              last_yaw_dot_ = 0.0;
            }
            else
            {
              last_yaw_ = start_yaw;
              last_yaw_dot_ = 0.0;
            }
          }
          else
          {
            const double last_eval_t = (t - t_step) > 0.0 ? (t - t_step) : 0.0;
            Eigen::Matrix<double, 3, 6> last_pvajs = trajectory_fn(last_eval_t);
            Eigen::Vector3d last_pos = last_pvajs.col(0);
            Eigen::Vector3d last_vel = last_pvajs.col(1);
            Eigen::Vector3d last_acc = last_pvajs.col(2);
            Eigen::Vector3d last_jerk = last_pvajs.col(3);
            Eigen::Vector3d last_snap = last_pvajs.col(4);
            Eigen::Vector3d last_crackle = last_pvajs.col(5);
            Eigen::Vector3d last_pos_quad, last_vel_quad, last_acc_quad, last_jerk_quad, last_p, last_pd;

            payload_flat_.forward_p(last_pos, last_vel, last_acc, last_jerk, last_snap, last_crackle,
                                    last_pos_quad, last_vel_quad, last_acc_quad, last_jerk_quad, last_p, last_pd);

            if (fabs(last_vel_quad(0)) + fabs(last_vel_quad(1)) > 0.1)
            {
              last_yaw_ = atan2(last_vel_quad(1), last_vel_quad(0));
              last_yaw_dot_ = 0.0;
            }
            else
            {
              last_yaw_ = start_yaw;
              last_yaw_dot_ = 0.0;
            }
          }
        }
        calculate_yaw(vel_quad, t_step, yaw, yaw_dot);
      }

      Eigen::Vector3d local_fq = fq_;
      Eigen::Vector3d local_fl = fl_;
      if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 2)
      {
        local_fq = fq_.cross(p);
      }
      else if (params_.force_estimator_param_.USE_CONSTANT_MOMENT == 1)
      {
        local_fl = fl_.cross(p);
      }

      payload_flat_.forward_rot(acc, jerk, acc_quad, jerk_quad, local_fq, local_fl, yaw, yaw_dot, quat, thr, omg);

      Eigen::Matrix<real_t, kStateSize, 1> reference_state;
      reference_state << pos_quad.x(), pos_quad.y(), pos_quad.z(),
          quat.w(), quat.x(), quat.y(), quat.z(),
          vel_quad.x(), vel_quad.y(), vel_quad.z(),
          pos.x(), pos.y(), pos.z(),
          vel.x(), vel.y(), vel.z(),
          p.x(), p.y(), p.z(),
          pd.x(), pd.y(), pd.z();
      reference_states_.col(i) = reference_state;

      Eigen::Matrix<real_t, kInputSize, 1> reference_input;
      reference_input << thr, omg.x(), omg.y(), 0.0;
      reference_inputs_.col(i) = reference_input;
      t += t_step;
    }
  }

  /**
  * @brief MPC求解准备线程 
  * 功能：在独立线程中执行MPC求解器的预处理步骤，用于优化下一控制周期的计算效率
  * 计时：统计预处理耗时并更新平滑处理后的准备时间变量(timing_preparation_)
  **/
  void MpcController::preparationThread()
  {
    const clock_t start = clock();

    mpc_wrapper_.prepare();

    // Timing
    const clock_t end = clock();
    timing_preparation_ = timing_preparation_*0.9 + 0.1* double(end - start) / CLOCKS_PER_SEC;
  }

  /**
  * @brief 角度限幅：确保角度在[-π, π]范围内
  * @param ang 输入角度
  * @return 限幅后的角度
  * 原理：通过循环加减2π将角度映射到标准范围
  **/
  double MpcController::angle_limit(double ang)
  {
    while (ang > M_PI)
    {
      ang -= 2.0 * M_PI;
    }
    while (ang <= -M_PI)
    {
      ang += 2.0 * M_PI;
    }
    return ang;
  }
  /**
  * @brief 计算圆周角度最小差值
  * @param a 起始角度(rad)
  * @param b 终止角度(rad)
  * @return 沿单位圆的最短路径角度差[-π, π]
  * 特性：保证返回差值具有最小绝对值（考虑圆周特性）
  **/
  double MpcController::angle_diff(double a, double b)
  {
    double d1, d2;
    d1 = a-b;
    d2 = 2*M_PI - fabs(d1);
    if(d1 > 0)
      d2 *= -1.0;
    if(fabs(d1) < fabs(d2))
      return(d1);
    else
      return(d2);
  }
  /**
  * @brief 偏航角规划与速率限制
  * @param[in] vel 无人机速度向量（决定期望航向）
  * @param[in] dt 时间步长(s)
  * @param[out] yaw 计算后的目标偏航角(rad)
  * @param[out] yawdot 受速率限制的偏航角速度(rad/s)
  * 功能：根据速度方向生成平滑偏航角轨迹，应用角速度限制(YAW_DOT_MAX_PER_SEC)
  * 特性：当速度过小时保持上一时刻偏航角
  **/
  void MpcController::calculate_yaw(Eigen::Vector3d &vel, const double dt, double &yaw, double &yawdot)
  {
    const double YAW_DOT_MAX_PER_SEC = params_.max_bodyrate_z_;   // 偏航角速率限制

    double yaw_temp;
    double max_yaw_change = YAW_DOT_MAX_PER_SEC * dt;

    // tangent line
    if ((fabs(vel(1)) + fabs(vel(0))) < 0.1)
    {
      yaw_temp = last_yaw_;
    }
    else
    {
      yaw_temp = atan2(vel(1), vel(0));     
    }
    double yaw_diff = angle_diff(yaw_temp, last_yaw_);  //角度差计算与限幅
    
    if (yaw_diff > max_yaw_change )
    {
      yaw_diff = max_yaw_change;
      yawdot = YAW_DOT_MAX_PER_SEC;
    }
    else if(yaw_diff < -max_yaw_change)
    {
      yaw_diff = -max_yaw_change;
      yawdot = -YAW_DOT_MAX_PER_SEC;
    }
    else
    {
      yawdot = yaw_diff / dt;
    }
    
    yaw = last_yaw_ + yaw_diff;

    // std::cout<<"last_yaw: "<< last_yaw_ << "yaw: " << yaw << " yawdot: " << yawdot << std::endl;
    
    // std::cout<< "dt: " << dt << " max_yaw_change: " << max_yaw_change << " vel: " << vel << " last_yaw_: " << last_yaw_ << " yaw: " << yaw << " yawdot: " << yawdot << std::endl;

    last_yaw_ = yaw;
    last_yaw_dot_ = yawdot;
  }

  /**
  * @brief 推力-转速归一化转换
  * @param[in] thrust 实际推力值(N)
  * @param[in] voltage 当前电压（暂未使用）
  * @return 归一化后的推力指令[0,1]
  * 功能：将带时间戳的推力值压入队列(timed_thrust)，供推力模型估计器使用
  **/
  double MpcController::convertThrust(const double& thrust, const double voltage)
  {
    double normalized_thrust;
    // double des_acc_norm = thrust;
    // This compensates for an acceleration component in thrust direction due
    // to the square of the body-horizontal velocity.
    // des_acc_norm -= param.rt_drag.k_thrust_horz * (pow(est_v.x(), 2) + pow(est_v.y(), 2));

    normalized_thrust = thrust / thrustscale_;
    timed_thrust.emplace(std::make_pair(ros::Time::now(), normalized_thrust));

    return normalized_thrust;
  }
  
  /**
  * @brief 推力模型在线估计（带遗忘因子递推最小二乘法）
  * @param[in] est_a 当前加速度估计值[m/s²]
  * @param[in] quad_q 无人机当前姿态四元数
  * @param[in] est_loadp 负载位置估计值[m]
  * @param[in] est_loaddp 负载速度估计值[m/s]
  * @param[in] rpm 电机转速[RPM]
  * @param[in] voltage 当前电压[V]
  * @param[in] param MPC参数配置
  * @return bool 是否成功更新推力模型
  * 功能：
  *   - 根据35-45ms前的推力指令和实际反馈（转速/加速度）更新推力系数
  *   - 支持两种反馈模式：电机转速模型/加速度反推模型
  *   - 自动维护时间同步队列(timed_thrust)
  *   - 安全边界检查（悬停百分比限制）
  **/
  bool MpcController::estimateThrustModel(
      const Eigen::Vector3d &est_a,
      const Eigen::Quaterniond &quad_q,
      const Eigen::Vector3d &est_loadp,
      const Eigen::Vector3d &est_loaddp,
      const Eigen::Vector4d &rpm,
      const double voltage,
      const MpcParams &param)
  {

    ros::Time t_now = ros::Time::now();
    while (timed_thrust.size() >= 1)
    {
      // 选择35~45ms前的数据
      std::pair<ros::Time, double> t_t = timed_thrust.front();
      double time_passed = (t_now - t_t.first).toSec();
      if (time_passed > 0.045) // 45ms
      {
        // printf("continue, time_passed=%f\n", time_passed);
        timed_thrust.pop();
        continue;
      }
      if (time_passed < 0.035) // 35ms
      {
        // printf("skip, time_passed=%f\n", time_passed);
        return false;
      }

      /***********************************************************/
      /* 带遗忘因子的递推最小二乘算法 */
      /***********************************************************/
      double normlized_thr = t_t.second;
      timed_thrust.pop();
      if (param.thr_map_.accurate_thrust_model == 0)
      {
        resetThrustMapping(); // Or skip
      }
      else if(param.thr_map_.accurate_thrust_model == 1 )
      {
        
        double thr_fb = rotor2thrust(param.force_estimator_param_.sqrt_kf,rpm);

        /***********************************/
        /* 模型：thr = thrustscale_ * normlized_thr */
        /***********************************/
          double gamma = 1 / (rho2 + normlized_thr * P * normlized_thr);
          double K = gamma * P * normlized_thr;
          double curr_thrustscale = thrustscale_ + K * (thr_fb - normlized_thr * thrustscale_);
          thrustscale_ = param.thr_map_.filter_factor*curr_thrustscale +(1- param.thr_map_.filter_factor)*thrustscale_;
          P = (1 - K * normlized_thr) * P / rho2;
          // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc, gamma, K, P);
          // fflush(stdout);
        const double hover_percentage = (param.gravity_ * (param.dyn_params_.mass_q))/ thrustscale_;
        if (hover_percentage > 0.8 || hover_percentage < 0.1)
        {
          ROS_ERROR("估计的悬停百分比 >0.8 或 <0.1! 可能加速度振动过高!");
          thrustscale_ = hover_percentage > 0.8 ? param.gravity_ / 0.8 : thrustscale_;
          thrustscale_ = hover_percentage < 0.1 ? param.gravity_ / 0.1 : thrustscale_;
        }
        debug.hover_percentage = hover_percentage; // debug
        if (param.thr_map_.print_val)
        {
          ROS_WARN("悬停百分比 = %f 推力系数 = %f", debug.hover_percentage,thrustscale_);
        }
      }
      else if (param.thr_map_.accurate_thrust_model == 2)
      {
        double thr_fb = acc2thrust(param.dyn_params_.mass_l,param.dyn_params_.mass_q,param.dyn_params_.l_length,param.gravity_,
        est_a(2),quad_q,est_loadp,est_loaddp);

        /***********************************/
        /* 模型：thr = thrustscale_ * normlized_thr */
        /***********************************/
          double gamma = 1 / (rho2 + normlized_thr * P * normlized_thr);
          double K = gamma * P * normlized_thr;
          double curr_thrustscale = thrustscale_ + K * (thr_fb - normlized_thr * thrustscale_);
          thrustscale_ = param.thr_map_.filter_factor*curr_thrustscale +(1- param.thr_map_.filter_factor)*thrustscale_;
          P = (1 - K * normlized_thr) * P / rho2;
          // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc, gamma, K, P);
          // fflush(stdout);
        const double hover_percentage = (param.gravity_ * (param.dyn_params_.mass_q))/ thrustscale_;
        if (hover_percentage > 0.8 || hover_percentage < 0.1)
        {
          ROS_ERROR("估计的悬停百分比 >0.8 或 <0.1! 可能加速度振动过高!");
          thrustscale_ = hover_percentage > 0.8 ? param.gravity_ / 0.8 : thrustscale_;
          thrustscale_ = hover_percentage < 0.1 ? param.gravity_ / 0.1 : thrustscale_;
        }
        debug.hover_percentage = hover_percentage; // debug
        if (param.thr_map_.print_val)
        {
          ROS_WARN("悬停百分比 = %f 推力系数 = %f", debug.hover_percentage,thrustscale_);
        }
      }

      return true;
    }

    return false;
  }
/**
* @brief 电机转速-推力转换模型
* @param[in] sqrt_kf 推力系数平方根[N/(RPM²)]
* @param[in] rpm 电机转速向量[4x1 RPM]
* @return double 计算的实际推力值[N]
* 模型公式：T = kf * Σ(rpm_i²)
**/
  double inline MpcController::rotor2thrust(const double& sqrt_kf, const Eigen::Vector4d& rpm)
  {
    return (sqrt_kf * rpm).squaredNorm();
  }
  /**
  * @brief 加速度反推推力计算（基于吊挂系统动力学）
  * @param[in] Ml 负载质量[kg]
  * @param[in] Mq 无人机质量[kg]
  * @param[in] l_length 绳长[m]
  * @param[in] g 重力加速度[m/s²]
  * @param[in] acc_z 当前Z轴加速度[m/s²]
  * @param[in] quad 无人机姿态四元数
  * @param[in] cable 吊索方向向量
  * @param[in] dcable 吊索方向导数
  * @return double 所需推力值[N]
  * 功能：通过负载动力学方程反推体轴系Z向推力需求
  **/
  double inline MpcController::acc2thrust(const double& Ml, const double& Mq,const double& l_length, const double& g, const double& acc_z,
                              const Eigen::Quaterniond& quad, const Eigen::Vector3d& cable, const Eigen::Vector3d& dcable)

{
  const double q_w = quad.w();
  const double q_x = quad.x();
  const double q_y = quad.y();
  const double q_z = quad.z();
  const double cable_x = cable.x();
  const double cable_y = cable.y();
  const double cable_z = cable.z();
  
  const double q_sqr = q_w*q_w - q_x*q_x - q_y*q_y + q_z*q_z;
  const double dcable_sqr = dcable.squaredNorm();//dcable_x^2 + dcable_y^2 + dcable_z^2;
  const double qwqxminusqyqz = 2*q_w*q_x - 2*q_y*q_z;
  const double qwqyminusqxqz = 2*q_w*q_y + 2*q_x*q_z;
  // 基于加速度和姿态计算所需推力
  double des_T = -(Mq*(acc_z - g*(q_sqr) + (g*(Ml + Mq)*(q_sqr) - Ml*(g + (Mq*cable_z*l_length*(dcable_sqr))/(Ml + Mq))*(q_sqr)
                - (Ml*Mq*cable_x*l_length*(qwqyminusqxqz)*(dcable_sqr))/(Ml + Mq) 
                + (Ml*Mq*cable_y*l_length*(qwqxminusqyqz)*(dcable_sqr))/(Ml + Mq))/Mq))/((Ml*cable_x*(qwqyminusqxqz)*(cable_z*(q_sqr) 
                - cable_y*(qwqxminusqyqz) + cable_x*(qwqyminusqxqz)))/(Ml + Mq) - (Ml*cable_y*(qwqxminusqyqz)*(cable_z*(q_sqr) 
                - cable_y*(qwqxminusqyqz) + cable_x*(qwqyminusqxqz)))/(Ml + Mq) + (Ml*cable_z*(cable_z*(q_sqr) 
                - cable_y*(qwqxminusqyqz) + cable_x*(qwqyminusqxqz))*(q_sqr))/(Ml + Mq) - 1);

  return des_T; //accelration in body z axis(including g) 返回体轴系Z方向推力
}
  /**
  * @brief 推力映射重置
  * 功能：
  *   - 恢复推力系数到初始值（基于悬停百分比）
  *   - 重置递推算法协方差矩阵P
  **/
  void MpcController::resetThrustMapping(void)
  {
    thrustscale_ = (params_.gravity_ * (params_.dyn_params_.mass_q)) / params_.thr_map_.hover_percentage;
    thr_scale_compensate = 1.0;
    P = 1e6;
  }

}
