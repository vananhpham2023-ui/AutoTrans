#pragma once
#include "so3_quadrotor/geometry_utils.hpp"
#include <iostream>
#include "random"

const double kLengthError = 0.025; // 绳索长度误差阈值
namespace so3_quadrotor {
#define RED "\033[31m"    /* Red */
#define YELLOW "\033[33m" /* Yellow */
#define RESET "\033[0m"
// data types
struct Config {
  double          g;      // gravity - 重力加速度
  double          mass;   // 无人机质量
  Eigen::Matrix3d J;      // Inertia - 惯性矩阵
  double          kf;     // 升力系数
  double          km;     // 扭矩系数
  double          arm_length; // 机臂长度
  double          motor_time_constant; // unit: sec - 电机时间常数
  double          max_rpm; // 最大电机转速
  double          min_rpm; // 最小电机转速
  double          mass_l; // load mass - 负载质量
  double          l_length; // load length - 绳索长度
  double          kOmega[3]; // 角速度控制器比例增益
  double          kdOmega[3]; // 角速度控制器微分增益
};
struct Control {
  double rpm[4]; // 四个电机的目标转速
};
struct Cmd {
  double thrust; // 期望总推力
  Eigen::Vector3d body_rates; // 期望机体角速度
};

class Quadrotor {
 private:
  // parameters
  Config config_; // 系统配置参数
  // state
  struct State {
    Eigen::Vector3d x = Eigen::Vector3d::Zero(); // 无人机位置
    Eigen::Vector3d v = Eigen::Vector3d::Zero(); // 无人机速度
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); // 无人机姿态旋转矩阵
    Eigen::Vector3d omega = Eigen::Vector3d::Zero(); // 无人机角速度
    Eigen::Vector4d motor_rpm = Eigen::Vector4d::Zero(); // 电机转速

    Eigen::Vector3d xl = Eigen::Vector3d(0,0,-0.1); // position of payload - 负载位置
    Eigen::Vector3d vl = Eigen::Vector3d::Zero(); // velocity of payload - 负载速度
    Eigen::Vector3d ql = Eigen::Vector3d(0,0,-1); // line vector - 绳索方向向量
    Eigen::Vector3d dql = Eigen::Vector3d::Zero(); // line vector rate - 绳索方向向量的导数

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // 状态向量加法运算符重载
    inline State operator+(const State &t) const {
      State sum;
      sum.x = x + t.x;
      sum.v = v + t.v;
      sum.R = R + t.R;
      sum.omega = omega + t.omega;
      sum.motor_rpm = motor_rpm + t.motor_rpm;

      sum.xl = xl + t.xl;
      sum.vl = vl + t.vl;
      sum.ql = ql + t.ql;
      sum.dql = dql + t.dql;
      return sum;
    }
    // 状态向量标量乘法运算符重载
    inline State operator*(const double &t) const {
      State mul;
      mul.x = x * t;
      mul.v = v * t;
      mul.R = R * t;
      mul.omega = omega * t;
      mul.motor_rpm = motor_rpm * t;

      mul.xl = xl * t;
      mul.vl = vl * t;
      mul.ql = ql * t;
      mul.dql = dql * t;
      return mul;
    }
    // 状态向量标量除法运算符重载
    inline State operator/(const double &t) const {
      State mul;
      mul.x = x / t;
      mul.v = v / t;
      mul.R = R / t;
      mul.omega = omega / t;
      mul.motor_rpm = motor_rpm / t;

      mul.xl = xl / t;
      mul.vl = vl / t;
      mul.ql = ql / t;
      mul.dql = dql / t;
      return mul;
    }
  } state_; // 当前系统状态
  Eigen::Vector4d  input_ = Eigen::Vector4d::Zero(); // 电机输入命令
  Eigen::Vector3d wind_quad_world_ = Eigen::Vector3d::Zero(); // 世界系下无人机位置处风速
  Eigen::Vector3d wind_load_world_ = Eigen::Vector3d::Zero(); // 世界系下负载位置处风速
  Eigen::Vector3d fl_true_ = Eigen::Vector3d::Zero(); // 上一次积分后的真实负载外力
  Eigen::Vector3d fq_true_ = Eigen::Vector3d::Zero(); // 上一次积分后的真实无人机外力
 public:
  std::default_random_engine generator; // 随机数生成器
  std::normal_distribution<double> distribution{0.0,1.0}; // 正态分布
  const Config &config; // 配置参数引用（只读）
  const State &state; // 状态引用（只读）
  State last_state; // 上一时刻状态
  double last_dt; // 上一时间步长
  
  /**
   * @brief 四旋翼动力学模型构造函数
   * @param conf 系统配置参数
   */
  Quadrotor(const Config &conf) : config_(conf), 
    config(config_), state(state_) {};
  ~Quadrotor() {};

  /**
   * @brief 设置电机期望转速输入
   * @param u1 电机1期望转速
   * @param u2 电机2期望转速
   * @param u3 电机3期望转速
   * @param u4 电机4期望转速
   * 
   * 电机编号：
   *   *1*    前
   * 3     4
   *    2
   * 其中1和2顺时针旋转，3和4逆时针旋转（从上方看）
   */
  inline void setInput(double u1, double u2, double u3, double u4) {
    input_(0) = u1;
    input_(1) = u2;
    input_(2) = u3;
    input_(3) = u4;
    // 检查输入是否为NaN并限制转速范围
    for (size_t i=0; i<4; ++i) {
      if (std::isnan(input_(i))) {
        printf("so3_quadrotor: NAN input!\n");
      }
      input_(i) = input_(i) < config_.max_rpm ? input_(i) : config_.max_rpm;
      input_(i) = input_(i) > config_.min_rpm ? input_(i) : config_.min_rpm;
    }
  }

  inline void setWindVelocity(const Eigen::Vector3d &wind_quad_world, const Eigen::Vector3d &wind_load_world) {
    wind_quad_world_ = wind_quad_world;
    wind_load_world_ = wind_load_world;
  }

  /**
   * @brief 生成随机力向量
   * @param std 噪声标准差
   * @return 服从正态分布的随机力向量
   */
  Eigen::Vector3d random_force(double std)
  {
    Eigen::Vector3d noise(distribution(generator),distribution(generator),distribution(generator));
    return noise*std;
  }

  /**
   * @brief 计算状态导数（系统动力学方程）
   * @param state 当前系统状态
   * @return 状态导数（系统状态变化率）
   * 
   * 该函数实现了四旋翼无人机及其负载的完整动力学方程，包括：
   * 1. 姿态矩阵的重新正交化
   * 2. 总推力和力矩计算
   * 3. 空气阻力计算
   * 4. 基于绳索张力状态（松绳/紧绳）的不同动力学模型
   */
  inline State diff(const State &state, Eigen::Vector3d *fl_out = nullptr, Eigen::Vector3d *fq_out = nullptr) {
    State state_dot;
    // Re-orthonormalize R (polar decomposition)
    Eigen::LLT<Eigen::Matrix3d> llt(state.R.transpose() * state.R);
    Eigen::Matrix3d             P = llt.matrixL();
    Eigen::Matrix3d             R = state.R * P.inverse();

    Eigen::Matrix3d omega_vee(Eigen::Matrix3d::Zero());

    omega_vee(2, 1) = state.omega(0);
    omega_vee(1, 2) = -state.omega(0);
    omega_vee(0, 2) = state.omega(1);
    omega_vee(2, 0) = -state.omega(1);
    omega_vee(1, 0) = state.omega(2);
    omega_vee(0, 1) = -state.omega(2);

    Eigen::Vector4d motor_rpm_sq = state.motor_rpm.array().square();

    double thrust = config_.kf * motor_rpm_sq.sum();

    Eigen::Vector3d moments;
    moments(0) = config_.kf * (motor_rpm_sq(2) - motor_rpm_sq(3)) * config_.arm_length;
    moments(1) = config_.kf * (motor_rpm_sq(1) - motor_rpm_sq(0)) * config_.arm_length;
    moments(2) = config_.km * (motor_rpm_sq(0) + motor_rpm_sq(1) - motor_rpm_sq(2) -
                        motor_rpm_sq(3));

    const double kVelocityEps = 1e-6;

    Eigen::Vector3d vquad_rel = state.v - wind_quad_world_;
    double vquad_rel_norm = vquad_rel.norm();
    Eigen::Vector3d vquad_dir = Eigen::Vector3d::Zero();
    if (vquad_rel_norm > kVelocityEps) {
      vquad_dir = vquad_rel / vquad_rel_norm;
    }
    double resistancequad = 0.10 *                                        // C
                        3.14159265 * (config_.arm_length) * (config_.arm_length) * // S
                        vquad_rel_norm * vquad_rel_norm;

    Eigen::Vector3d vload_rel = state.vl - wind_load_world_;
    double vload_rel_norm = vload_rel.norm();
    Eigen::Vector3d vload_dir = Eigen::Vector3d::Zero();
    if (vload_rel_norm > kVelocityEps) {
      vload_dir = vload_rel / vload_rel_norm;
    }
    double resistanceload= 0.10 *                                        // C
                        3.14159265 * (config_.arm_length) * (config_.arm_length) * // S
                        vload_rel_norm * vload_rel_norm;

    Eigen::Vector3d fl = Eigen::Vector3d(0,0.0,0) + random_force(0.0)-resistanceload *vload_dir ;
    Eigen::Vector3d fq = Eigen::Vector3d(0.0,0,0) + random_force(0.0)-resistancequad *vquad_dir;

    if (fl_out) {
      *fl_out = fl;
    }
    if (fq_out) {
      *fq_out = fq;
    }


    double delta = (state.x - state.xl).norm() - config_.l_length;
    if (delta <= -kLengthError ) //Nontaut dynamics
    {
      
      state_dot.x = state.v;
      state_dot.v = -Eigen::Vector3d(0, 0, config_.g) + thrust * R.col(2) / config_.mass + fq/ config_.mass;
      state_dot.R = R * omega_vee;

      state_dot.omega = config_.J.inverse() * (moments - state.omega.cross(config_.J * state.omega)); //the same
      state_dot.motor_rpm = (input_ - state.motor_rpm) / config_.motor_time_constant;

      state_dot.xl = state.vl;
      state_dot.vl = -Eigen::Vector3d(0, 0, config_.g) + fl/config_.mass_l;
      state_dot.ql = (state.vl - state.v)/config_.l_length;
      state_dot.dql = (fl/config_.mass_l -fq/ config_.mass -thrust * R.col(2) / config_.mass)/config_.l_length;

      // state_dot.wl = -state.ql.cross(thrust * R.col(2))/(config_.mass * config_.l_length);
    }
    else if (delta > -kLengthError && delta < kLengthError) //taut dynamics
    { 
      
      Eigen::Vector3d norm_ql = state.ql.normalized();
      state_dot.x = state.v;
      state_dot.v = -Eigen::Vector3d(0, 0, config_.g) + (thrust * R.col(2)+fq) / config_.mass + (config_.mass_l*config_.l_length* state.dql.dot(state.dql) + norm_ql.dot(fl-config_.mass_l/config_.mass*(fq+thrust * R.col(2))))*norm_ql/(config_.mass+config_.mass_l);
      state_dot.R = R * omega_vee;

      state_dot.omega = config_.J.inverse() * (moments - state.omega.cross(config_.J * state.omega)); //the same
      state_dot.motor_rpm = (input_ - state.motor_rpm) / config_.motor_time_constant;

      state_dot.xl = state.vl;
      state_dot.vl = ( (norm_ql.dot(thrust * R.col(2)+ fq-fl*config_.mass/config_.mass_l) - config_.mass*config_.l_length* state.dql.dot(state.dql)) * norm_ql/(config_.mass+config_.mass_l) ) -Eigen::Vector3d(0, 0, config_.g)+fl/config_.mass_l;
      // state_dot.vl = ( (state.ql.dot(thrust * R.col(2)) - config_.mass*config_.l_length* state.dql.dot(state.dql)) * state.ql/(config_.mass+config_.mass_l) ) -Eigen::Vector3d(0, 0, config_.g) - resistance * vnorm / config_.mass_l + f_ext;
      state_dot.ql  = state.dql;
      state_dot.dql = (1.0/(config.mass*config_.l_length)) * (norm_ql.cross(norm_ql.cross(thrust * R.col(2)))) - state.dql.dot(state.dql)*norm_ql;
      
      // state_dot.x = state.v;
      // state_dot.v = -config_.l_length* state_dot.dql + state_dot.vl ;
    }
    else 
    {
      std::cout<<RED<<"DYNAMIC ERROR: Length is too long"<<RESET<<std::endl;
       Eigen::Vector3d norm_ql = state.ql.normalized();
      state_dot.x = state.v;
      state_dot.v = -Eigen::Vector3d(0, 0, config_.g) + (thrust * R.col(2)+fq) / config_.mass + (config_.mass_l*config_.l_length* state.dql.dot(state.dql) + norm_ql.dot(fl-config_.mass_l/config_.mass*(fq+thrust * R.col(2))))*norm_ql/(config_.mass+config_.mass_l);
      state_dot.R = R * omega_vee;

      state_dot.omega = config_.J.inverse() * (moments - state.omega.cross(config_.J * state.omega)); //the same
      state_dot.motor_rpm = (input_ - state.motor_rpm) / config_.motor_time_constant;

      state_dot.xl = state.vl;
      state_dot.vl = ( (norm_ql.dot(thrust * R.col(2)+ fq-fl*config_.mass/config_.mass_l) - config_.mass*config_.l_length* state.dql.dot(state.dql)) * norm_ql/(config_.mass+config_.mass_l) ) -Eigen::Vector3d(0, 0, config_.g)+fl/config_.mass_l;
      // state_dot.vl = ( (state.ql.dot(thrust * R.col(2)) - config_.mass*config_.l_length* state.dql.dot(state.dql)) * state.ql/(config_.mass+config_.mass_l) ) -Eigen::Vector3d(0, 0, config_.g) - resistance * vnorm / config_.mass_l + f_ext;
      state_dot.ql  = state.dql;
      state_dot.dql = (1.0/(config.mass*config_.l_length)) * (norm_ql.cross(norm_ql.cross(thrust * R.col(2)))) - state.dql.dot(state.dql)*norm_ql;
      
      // state_dot.x = state.v;
      // state_dot.v = -config_.l_length* state_dot.dql + state_dot.vl ;
    }
    
    return state_dot;
  }
  /**
   * @brief 执行动力学仿真的单步迭代
   * @param dt 时间步长
   * 
   * 该函数使用四阶龙格-库塔方法对系统动力学方程进行数值积分，并在紧绳状态下强制
   * 满足无人机与负载之间的位置约束。
   */
  inline void step(const double &dt) {
    // Runge–Kutta
    last_state = state_;
    last_dt = dt; //for numerical differentiation
    State k1 = diff(state_);
    State k2 = diff(state_+k1*dt/2);
    State k3 = diff(state_+k2*dt/2);
    Eigen::Vector3d fl_k4(Eigen::Vector3d::Zero()), fq_k4(Eigen::Vector3d::Zero());
    State k4 = diff(state_+k3*dt, &fl_k4, &fq_k4);
    state_ = state_ + (k1+k2*2+k3*2+k4) * dt/6;
    fl_true_ = fl_k4;
    fq_true_ = fq_k4;
    
    state_.ql.normalize();
    double delta = (state_.x - state_.xl).norm() - config_.l_length;
    if (delta > -kLengthError && delta < kLengthError) //taut dynamics
    {
      state_.x = state_.xl - config_.l_length* state_.ql;
      state_.v = state_.vl - config_.l_length* state_.dql;
    }
    else if (delta >= kLengthError)
    {
      std::cout<<RED<<"DYNAMIC ERROR: Length is too long"<<RESET<<std::endl;
      state_.dql = Eigen::Vector3d::Zero();
      state_.x = state_.xl - config_.l_length* state_.ql;
      state_.v = state_.vl - config_.l_length* state_.dql;
    }
  }
  /**
   * @brief 根据控制命令计算电机转速
   * @param cmd 控制命令（总推力和期望机体角速度）
   * @return 电机控制指令（四个电机的目标转速）
   * 
   * 该函数实现了从期望控制命令到电机转速的映射，包括：
   * 1. 角速度控制器（PD控制器）
   * 2. 内部动态抑制（INDI）
   * 3. 电机转速分配算法
   */
  inline Control getControl(const Cmd& cmd) {
    // 静态变量用于存储上一时刻的误差和状态
    static double LasteOm1 = 0;
    static double LasteOm2 = 0;
    static double LasteOm3 = 0;

    static float LastOm1 = 0;
    static float LastOm2 = 0;
    static float LastOm3 = 0;

    static double LastRpm1 = 0;
    static double LastRpm2 = 0;
    static double LastRpm3 = 0;
    static double LastRpm4 = 0;

    // 获取系统参数
    double         kf = config_.kf;
    double         km = config_.km / kf * kf;
    double          d = config_.arm_length;
    Eigen::Matrix3f J = config_.J.cast<float>();
    float     I[3][3] = { { J(0, 0), J(0, 1), J(0, 2) },
                          { J(1, 0), J(1, 1), J(1, 2) },
                          { J(2, 0), J(2, 1), J(2, 2) } };
    
    // 获取当前姿态（欧拉角）
    Eigen::Vector3d ypr = uav_utils::R_to_ypr(state_.R);
    Eigen::Matrix3d R; 
    R = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());
    
    // 获取当前角速度
    float Om1 = state_.omega(0);
    float Om2 = state_.omega(1);
    float Om3 = state_.omega(2);

    // 获取当前电机转速
    double rpm1 = state_.motor_rpm(0);
    double rpm2 = state_.motor_rpm(1);
    double rpm3 = state_.motor_rpm(2);
    double rpm4 = state_.motor_rpm(3);
    
    // 获取期望总推力
    float force = cmd.thrust;

    // 计算角速度误差
    float eOm1 = cmd.body_rates[0] - Om1;
    float eOm2 = cmd.body_rates[1] - Om2;
    float eOm3 = cmd.body_rates[2] - Om3;

    // 获取期望角速度
    float cOm1 = cmd.body_rates[0];
    float cOm2 = cmd.body_rates[1];
    float cOm3 = cmd.body_rates[2];

    // 计算角速度交叉项（由科里奥利力和向心力引起）
    float in1 = cOm2 * (I[2][0] * cOm1 + I[2][1] * cOm2 + I[2][2] * cOm3) -
                cOm3 * (I[1][0] * cOm1 + I[1][1] * cOm2 + I[1][2] * cOm3);
    float in2 = cOm3 * (I[0][0] * cOm1 + I[0][1] * cOm2 + I[0][2] * cOm3) -
                cOm1 * (I[2][0] * cOm1 + I[2][1] * cOm2 + I[2][2] * cOm3);
    float in3 = cOm1 * (I[1][0] * cOm1 + I[1][1] * cOm2 + I[1][2] * cOm3) -
                cOm2 * (I[0][0] * cOm1 + I[0][1] * cOm2 + I[0][2] * cOm3);

    // 计算角速度变化率
    float dOm1 = (Om1-LastOm1);
    float dOm2 = (Om2-LastOm2);
    float dOm3 = (Om3-LastOm3);

    // 计算电机转速变化率
    float drpm1 = (rpm1-LastRpm1);
    float drpm2 = (rpm2-LastRpm2);
    float drpm3 = (rpm3-LastRpm3);
    float drpm4 = (rpm4-LastRpm4);

    // 计算当前电机产生的力矩
    float M2_from_rpm = (rpm2*rpm2 - rpm1*rpm1)*d*kf;
    float M1_from_rpm = (rpm3*rpm3 - rpm4*rpm4)*d*kf;
    float M3_from_rpm = (rpm1*rpm1 + rpm2*rpm2 - rpm3*rpm3 - rpm4*rpm4)*km + 0*(drpm1 + drpm2  - drpm3  -drpm4); 

    // 计算INDI（内部动态抑制）补偿项
    float t_com1 = M1_from_rpm - (dOm1 * I[0][0] + dOm2 * I[0][1] + dOm3 * I[0][2]);
    float t_com2 = M2_from_rpm - (dOm1 * I[1][0] + dOm2 * I[1][1] + dOm3 * I[1][2]);
    float t_com3 = M3_from_rpm - (dOm1 * I[2][0] + dOm2 * I[2][1] + dOm3 * I[2][2]);
    
    // PD控制器计算期望力矩
    float M1 = config_.kOmega[0] * eOm1 + config_.kdOmega[0] * (eOm1-LasteOm1) + in1 + t_com1;
    float M2 = config_.kOmega[1] * eOm2 + config_.kdOmega[1] * (eOm2-LasteOm2) + in2 + t_com2;
    float M3 = config_.kOmega[2] * eOm3 + config_.kdOmega[2] * (eOm3-LasteOm3) + in3 + t_com3;

    // 更新历史状态
    LasteOm1 = eOm1;
    LasteOm2 = eOm2;
    LasteOm3 = eOm3;
    LastOm1 = Om1;
    LastOm2 = Om2;
    LastOm3 = Om3;
    LastRpm1 = rpm1;
    LastRpm2 = rpm2;
    LastRpm3 = rpm3;
    LastRpm4 = rpm4;

    // 计算每个电机转速的平方（控制分配）
    float w_sq[4];
    w_sq[0] = force / (4 * kf) - M2 / (2 * d * kf) + M3 / (4 * km);
    w_sq[1] = force / (4 * kf) + M2 / (2 * d * kf) + M3 / (4 * km);
    w_sq[2] = force / (4 * kf) + M1 / (2 * d * kf) - M3 / (4 * km);
    w_sq[3] = force / (4 * kf) - M1 / (2 * d * kf) - M3 / (4 * km);

    // 计算最终电机转速
    Control control;
    for (int i = 0; i < 4; i++) {
      if (w_sq[i] < 0) w_sq[i] = 0; // 防止负数开平方
      control.rpm[i] = sqrtf(w_sq[i]); // 开平方得到转速
    }
    
    return control;
  }

  // 设置初始状态的函数
  
  /**
   * @brief 设置负载位置
   * @param pos 负载位置向量
   */
  inline void setPos(const Eigen::Vector3d &pos) {
    state_.xl = pos;
  }
  
  /**
   * @brief 设置无人机位置
   * @param pos 无人机位置向量
   */
  inline void setQuadPos(const Eigen::Vector3d &pos) {
    state_.x = pos;
  }
  
  /**
   * @brief 设置无人机姿态（欧拉角）
   * @param ypr 欧拉角（yaw, pitch, roll）
   */
  inline void setYpr(const Eigen::Vector3d &ypr) {
    state_.R = uav_utils::ypr_to_R(ypr);
  }
  
  /**
   * @brief 设置电机转速
   * @param rpm 四个电机的转速向量
   */
  inline void setRpm(const Eigen::Vector4d &rpm) {
    state_.motor_rpm = rpm;
  }
  
  /**
   * @brief 设置绳索方向向量
   * @param q 绳索方向向量
   */
  inline void setq(const Eigen::Vector3d &q) {
    state_.ql = q;
  }
  
  // 获取状态值的函数
  
  /**
   * @brief 获取无人机重力
   * @return 无人机重力（质量×重力加速度）
   */
  inline double getGrav() const {
    return config_.g * config_.mass;
  }
  
  /**
   * @brief 获取绳索方向向量
   * @return 绳索方向向量
   */
  inline Eigen::Vector3d getq() const {
    return state_.ql;
  }
  
  /**
   * @brief 获取无人机位置
   * @return 无人机位置向量
   */
  inline Eigen::Vector3d getPos() const {
    return state_.x;
  }
  
  /**
   * @brief 获取无人机速度
   * @return 无人机速度向量
   */
  inline Eigen::Vector3d getVel() const {
    return state_.v;
  }
  
  /**
   * @brief 获取负载位置
   * @return 负载位置向量
   */
  inline Eigen::Vector3d getLoadPos() const {
    return state_.xl ;
  }
  
  /**
   * @brief 获取负载速度
   * @return 负载速度向量
   */
  inline Eigen::Vector3d getLoadVel() const {
    return state_.vl ;
  }
  
  /**
   * @brief 获取负载加速度（数值微分）
   * @return 负载加速度向量
   */
  inline Eigen::Vector3d getLoadAcc() const {
    return (state_.vl - last_state.vl)/last_dt;
  }
  
  /**
   * @brief 获取负载绳索方向向量
   * @return 负载绳索方向向量
   */
  inline Eigen::Vector3d getLoadq() const {
    return state_.ql ;
  }
  
  /**
   * @brief 获取绳索方向向量的导数
   * @return 绳索方向向量的导数
   */
  inline Eigen::Vector3d getLoaddq() const {
    return state_.dql ;
  }
  
  /**
   * @brief 获取电机转速
   * @return 四个电机的转速向量
   */
  inline Eigen::Vector4d getRpm() const {
    return state_.motor_rpm ;
  }
  
  /**
   * @brief 获取绳索方向向量的二阶导数（数值微分）
   * @return 绳索方向向量的二阶导数
   */
  inline Eigen::Vector3d getLoadddq() const {
    return (state_.dql - last_state.dql)/last_dt;
  }
  
  /**
   * @brief 获取无人机加速度（数值微分）
   * @return 无人机加速度向量
   */
  inline Eigen::Vector3d getQuadAcc() const {
    return (state_.v - last_state.v)/last_dt;
  }
  
  /**
   * @brief 获取无人机姿态四元数
   * @return 无人机姿态四元数
   */
  inline Eigen::Quaterniond getQuat() const {
    return Eigen::Quaterniond(state_.R);
  }
  
  /**
   * @brief 获取无人机角速度
   * @return 无人机角速度向量
   */
  inline Eigen::Vector3d getOmega() const {
    return state_.omega;
  }

  /**
   * @brief 获取上一积分周期末的真实负载外力
   * @return 负载外力（世界坐标系）
   */
  inline Eigen::Vector3d getTrueLoadForce() const {
    return fl_true_;
  }

  /**
   * @brief 获取上一积分周期末的真实无人机外力
   * @return 无人机外力（世界坐标系）
   */
  inline Eigen::Vector3d getTrueQuadForce() const {
    return fq_true_;
  }

};

} // namespace so3_quadrotor
