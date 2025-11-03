#include "so3_quadrotor/quadrotor_dynamics.hpp"
#include <quadrotor_msgs/SO3Command.h>
#include <mavros_msgs/ESCTelemetry.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>

#include <algorithm>
#include <geometry_msgs/AccelStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <random>

#include "so3_quadrotor/wind_field/wind_field.h"
namespace so3_quadrotor {

class Nodelet : public nodelet::Nodelet {
 private:
  std::shared_ptr<Quadrotor> quadrotorPtr_;
  Control control_;
  Cmd cmd_;
  ros::Publisher odom_pub_, imu_pub_, vis_pub_,odom_payload_pub_,cable_info_pub_;
  ros::Publisher rpm_pub_,load_imu_pub_, true_force_pub_, wind_pub_;
  ros::Subscriber cmd_sub_;
  ros::Timer simulation_timer;
  tf::TransformBroadcaster tf_br_;
  double noise_uav_,noise_rpm_, noise_payload_,noise_acc_uav_,noise_acc_payload_,noise_vel_uav_,noise_vel_payload_; 
  std::default_random_engine generator;
  std::normal_distribution<double> distribution{0.0,1.0};
  double l_length;
  double true_force_rate_ = 100.0;
  ros::Duration odom_period_;
  ros::Duration true_force_period_;
  ros::Time next_odom_pub_time_;
  ros::Time next_true_force_pub_time_;
  bool time_initialized_ = false;
  // parameters
  int simulation_rate_ = 1e3;
  int odom_rate_ = 400;
  // msgs
  nav_msgs::Odometry odom_msg_;
  nav_msgs::Odometry odom_payload_msg_;
  sensor_msgs::Imu imu_msg_;
  sensor_msgs::Imu load_imu_msg_;
  sensor_msgs::Imu cable_info_msg_;  // TODO: Change the message type
  mavros_msgs::ESCTelemetry rpm_msg_;
  visualization_msgs::MarkerArray vis_msg_;
  double payload_size_;
  double motor_yaw_[4] = {0,0,0,0};
  std::unique_ptr<wind::WindFieldBase> wind_field_;
  geometry_msgs::Vector3Stamped wind_msg_;

  // 三维高斯噪声生成器
  // @param std 噪声标准差
  // @return Eigen::Vector3d 含噪声的三维向量
  Eigen::Vector3d guass_random(double std)
  {
    Eigen::Vector3d noise(distribution(generator),distribution(generator),distribution(generator));
    return noise*std;
  }
  // 四维高斯噪声生成器（扩展三维版本）
  // @param std 噪声标准差  
  // @return Eigen::Vector4d 含噪声的四维向量
    Eigen::Vector4d guass_random4d(double std)
  {
    Eigen::Vector4d noise(distribution(generator),distribution(generator),distribution(generator),distribution(generator));
    return noise*std;
  }

  // 控制指令回调函数
  // @param cmd_msg 来自mavros的AttitudeTarget控制指令
  // 功能：解析并存储推力/机体角速率指令到cmd_结构体
  void cmd_callback(const mavros_msgs::AttitudeTarget::ConstPtr& cmd_msg) {
    cmd_.thrust = cmd_msg->thrust;
    cmd_.body_rates[0] = cmd_msg->body_rate.x;
    cmd_.body_rates[1] = cmd_msg->body_rate.y;
    cmd_.body_rates[2] = cmd_msg->body_rate.z;
  }
  
  // 主仿真循环定时器回调
  // 功能：执行无人机动力学仿真、传感器数据生成、坐标变换发布、可视化更新
  // 流程：
  // 1. 计算电机控制量
  // 2. 执行动力学仿真步进
  // 3. 添加传感器噪声
  // 4. 发布TF/里程计/IMU/RPM等数据
  // 5. 更新螺旋桨/负载可视化
  void timer_callback(const ros::TimerEvent& event) {
    const ros::Time current_time = event.current_expected;
    if (!time_initialized_) {
      next_odom_pub_time_ = current_time;
      next_true_force_pub_time_ = current_time;
      time_initialized_ = true;
    }
    auto last_control = control_;
    control_ = quadrotorPtr_->getControl(cmd_);
    for (size_t i = 0; i < 4; ++i) {
      if (std::isnan(control_.rpm[i]))
        control_.rpm[i] = last_control.rpm[i];
    }
    quadrotorPtr_->setInput(control_.rpm[0], control_.rpm[1], control_.rpm[2], control_.rpm[3]);
    quadrotorPtr_->step(1.0/simulation_rate_);
    Eigen::Vector3d wind_quad = Eigen::Vector3d::Zero();
    Eigen::Vector3d wind_load = Eigen::Vector3d::Zero();
    if (wind_field_) {
      wind_quad = wind_field_->sample(quadrotorPtr_->getPos(), current_time.toSec());
      wind_load = wind_field_->sample(quadrotorPtr_->getLoadPos(), current_time.toSec());
    }
    quadrotorPtr_->setWindVelocity(wind_quad, wind_load);
    if (wind_pub_) {
      wind_msg_.header.stamp = current_time;
      wind_msg_.vector.x = wind_quad.x();
      wind_msg_.vector.y = wind_quad.y();
      wind_msg_.vector.z = wind_quad.z();
      wind_pub_.publish(wind_msg_);
    }

    if (vis_msg_.markers.size() < 8)
    {
      ROS_WARN_THROTTLE(1.0, "[so3_quadrotor_nodelet] Marker array not ready yet (size=%zu)", vis_msg_.markers.size());
      return;
    }
    const Eigen::Vector3d&     pos = quadrotorPtr_->getPos() + guass_random(noise_uav_);
    const Eigen::Vector3d&     vel = quadrotorPtr_->getVel() + guass_random(noise_vel_uav_);
    Eigen::Vector3d     acc = quadrotorPtr_->getQuadAcc() + guass_random(noise_acc_uav_);
    const Eigen::Quaterniond& quat = quadrotorPtr_->getQuat();
    const Eigen::Vector3d&   omega = quadrotorPtr_->getOmega();
    const Eigen::Vector3d&   pos_l = quadrotorPtr_->getLoadPos() + guass_random(noise_payload_);
    const Eigen::Vector3d&   vel_l = quadrotorPtr_->getLoadVel() + guass_random(noise_vel_payload_);
    // const Eigen::Vector3d&   q = quadrotorPtr_->getLoadq();
    const Eigen::Vector3d&   dq = quadrotorPtr_->getLoaddq();
    const Eigen::Vector3d&    pos_real = quadrotorPtr_->getPos();
    const Eigen::Vector3d&    pos_l_real = quadrotorPtr_->getLoadPos();
    const Eigen::Vector4d&    rpm = quadrotorPtr_->getRpm() +  guass_random4d(noise_rpm_);
    Eigen::Vector3d     acc_load = quadrotorPtr_->getLoadAcc() +guass_random(noise_acc_payload_);
    Eigen::Vector3d fake_q = pos_l - pos;
    fake_q.normalize();
    Eigen::Vector3d fake_dq = (vel_l - vel)/l_length;
    acc_load = acc_load + Eigen::Vector3d(0,0,quadrotorPtr_->config.g); //Convert to imu like data
    acc = acc + Eigen::Vector3d(0,0,quadrotorPtr_->config.g);

    acc = quat.toRotationMatrix().transpose() *acc; //Convert to inertial frame 转换为惯性坐标系

    if (current_time >= next_odom_pub_time_) {
      next_odom_pub_time_ += odom_period_;
      // const Eigen::Vector3d&     pos = quadrotorPtr_->getPos() + guass_random(noise_uav_);
      // const Eigen::Vector3d&     vel = quadrotorPtr_->getVel() ;
      // const Eigen::Vector3d&     acc = quadrotorPtr_->getAcc() ;
      // const Eigen::Quaterniond& quat = quadrotorPtr_->getQuat();
      // const Eigen::Vector3d&   omega = quadrotorPtr_->getOmega();
      // const Eigen::Vector3d&   pos_l = quadrotorPtr_->getLoadPos() + guass_random(noise_payload_);
      // const Eigen::Vector3d&   vel_l = quadrotorPtr_->getLoadVel() ;
      // const Eigen::Vector3d&   q = quadrotorPtr_->getLoadq();
      // const Eigen::Vector3d&   dq = quadrotorPtr_->getLoaddq();
      // tf
      tf::Transform transform;
      transform.setOrigin( tf::Vector3(pos.x(), pos.y(), pos.z()) );
      transform.setRotation( tf::Quaternion(quat.x(), quat.y(), quat.z(), quat.w()) );
      tf_br_.sendTransform(tf::StampedTransform(transform, current_time, "world", "body"));
      // odom
      odom_msg_.header.stamp = current_time;
      odom_msg_.pose.pose.position.x = pos(0) ;
      odom_msg_.pose.pose.position.y = pos(1) ;
      odom_msg_.pose.pose.position.z = pos(2) ;
      odom_msg_.pose.pose.orientation.x = quat.x();
      odom_msg_.pose.pose.orientation.y = quat.y();
      odom_msg_.pose.pose.orientation.z = quat.z();
      odom_msg_.pose.pose.orientation.w = quat.w();

      odom_msg_.twist.twist.linear.x = vel(0);
      odom_msg_.twist.twist.linear.y = vel(1);
      odom_msg_.twist.twist.linear.z = vel(2);

      odom_msg_.twist.twist.angular.x = omega(0);
      odom_msg_.twist.twist.angular.y = omega(1);
      odom_msg_.twist.twist.angular.z = omega(2);
      odom_pub_.publish(odom_msg_);

      // odom payload
      odom_payload_msg_.header.stamp = current_time;
      odom_payload_msg_.pose.pose.position.x = pos_l(0) ;
      odom_payload_msg_.pose.pose.position.y = pos_l(1) ;
      odom_payload_msg_.pose.pose.position.z = pos_l(2) ;
      odom_payload_msg_.pose.pose.orientation.x = 0;
      odom_payload_msg_.pose.pose.orientation.y = 0;
      odom_payload_msg_.pose.pose.orientation.z = 0;
      odom_payload_msg_.pose.pose.orientation.w = 1;

      odom_payload_msg_.twist.twist.linear.x = vel_l(0);
      odom_payload_msg_.twist.twist.linear.y = vel_l(1);
      odom_payload_msg_.twist.twist.linear.z = vel_l(2);

      odom_payload_msg_.twist.twist.angular.x = dq(0);
      odom_payload_msg_.twist.twist.angular.y = dq(1);
      odom_payload_msg_.twist.twist.angular.z = dq(2);
      odom_payload_pub_.publish(odom_payload_msg_);
      // imu
      imu_msg_.header.stamp = current_time;
      imu_msg_.orientation.x = quat.x();
      imu_msg_.orientation.y = quat.y();
      imu_msg_.orientation.z = quat.z();
      imu_msg_.orientation.w = quat.w();

      imu_msg_.angular_velocity.x = omega(0);
      imu_msg_.angular_velocity.y = omega(1);
      imu_msg_.angular_velocity.z = omega(2);

      imu_msg_.linear_acceleration.x = acc[0];
      imu_msg_.linear_acceleration.y = acc[1];
      imu_msg_.linear_acceleration.z = acc[2];
      imu_pub_.publish(imu_msg_);

      //cable info
      cable_info_msg_.header.stamp = current_time;
      cable_info_msg_.orientation.x = 0;
      cable_info_msg_.orientation.y = 0;
      cable_info_msg_.orientation.z = 0;
      cable_info_msg_.orientation.w = 1;

      cable_info_msg_.angular_velocity.x = fake_q(0);
      cable_info_msg_.angular_velocity.y = fake_q(1);
      cable_info_msg_.angular_velocity.z = fake_q(2);

      cable_info_msg_.linear_acceleration.x = fake_dq[0];
      cable_info_msg_.linear_acceleration.y = fake_dq[1];
      cable_info_msg_.linear_acceleration.z = fake_dq[2];
      cable_info_pub_.publish(cable_info_msg_);

      //RPM 
      rpm_msg_.header = imu_msg_.header;
      rpm_msg_.esc_telemetry[0].header = imu_msg_.header;
      rpm_msg_.esc_telemetry[0].rpm = rpm[0];
      rpm_msg_.esc_telemetry[1].header = imu_msg_.header;
      rpm_msg_.esc_telemetry[1].rpm = rpm[1];
      rpm_msg_.esc_telemetry[2].header = imu_msg_.header;
      rpm_msg_.esc_telemetry[2].rpm = rpm[2];
      rpm_msg_.esc_telemetry[3].header = imu_msg_.header;
      rpm_msg_.esc_telemetry[3].rpm = rpm[3];
      rpm_pub_.publish(rpm_msg_);

      // IMU payload
      load_imu_msg_.header = imu_msg_.header;
      load_imu_msg_.orientation.x = 0;
      load_imu_msg_.orientation.y = 0;
      load_imu_msg_.orientation.z = 0;
      load_imu_msg_.orientation.w = 1;
      load_imu_msg_.angular_velocity.x = 0;
      load_imu_msg_.angular_velocity.y = 0;
      load_imu_msg_.angular_velocity.z = 0;
      load_imu_msg_.linear_acceleration.x = acc_load[0];
      load_imu_msg_.linear_acceleration.y = acc_load[1];
      load_imu_msg_.linear_acceleration.z = acc_load[2];
      load_imu_msg_.angular_velocity.x = 0;
      load_imu_msg_.angular_velocity.y = 0;
      load_imu_msg_.angular_velocity.z = 0; 
      load_imu_pub_.publish(load_imu_msg_);

      // drone visualization
      std::vector<Eigen::Vector3d> propellers;
      propellers.emplace_back(0, +quadrotorPtr_->config.arm_length/2, 0.02);
      propellers.emplace_back(0, -quadrotorPtr_->config.arm_length/2, 0.02);
      propellers.emplace_back(-quadrotorPtr_->config.arm_length/2, 0, 0.02);
      propellers.emplace_back(+quadrotorPtr_->config.arm_length/2, 0, 0.02);
      for (size_t i=0; i<4; ++i) {
        double rpm = quadrotorPtr_->state.motor_rpm.coeff(i);
        if (i/2) {
          motor_yaw_[i] += rpm*60 / odom_rate_;
        } else {
          motor_yaw_[i] -= rpm*60 / odom_rate_;
        }
        motor_yaw_[i] = std::fmod(motor_yaw_[i], 2*M_PI);
        Eigen::Quaterniond quat_propeller = quat * uav_utils::ypr_to_quaternion(
          Eigen::Vector3d(motor_yaw_[i], 0, 0));
        Eigen::Vector3d pos_propeller = pos_real + quat.toRotationMatrix() * propellers[i];

        vis_msg_.markers[i].pose.position.x = pos_propeller(0);
        vis_msg_.markers[i].pose.position.y = pos_propeller(1);
        vis_msg_.markers[i].pose.position.z = pos_propeller(2);
        vis_msg_.markers[i].pose.orientation.x = quat_propeller.x();
        vis_msg_.markers[i].pose.orientation.y = quat_propeller.y();
        vis_msg_.markers[i].pose.orientation.z = quat_propeller.z();
        vis_msg_.markers[i].pose.orientation.w = quat_propeller.w();
      }
      for (size_t i=4; i<6; ++i) {
        vis_msg_.markers[i].pose.position.x = pos_real(0);
        vis_msg_.markers[i].pose.position.y = pos_real(1);
        vis_msg_.markers[i].pose.position.z = pos_real(2);
        vis_msg_.markers[i].pose.orientation.x = quat.x();
        vis_msg_.markers[i].pose.orientation.y = quat.y();
        vis_msg_.markers[i].pose.orientation.z = quat.z();
        vis_msg_.markers[i].pose.orientation.w = quat.w();
        vis_msg_.markers[i].id = i;
      }
      vis_msg_.markers[6].pose.orientation.w = 1;
      vis_msg_.markers[6].pose.position.x = pos_l_real(0);
      vis_msg_.markers[6].pose.position.y = pos_l_real(1);
      vis_msg_.markers[6].pose.position.z = pos_l_real(2);
      
      vis_msg_.markers[7].points.clear();
      vis_msg_.markers[7].pose.orientation.w = 1;
      geometry_msgs::Point p;
      p.x = pos_l_real(0);
      p.y = pos_l_real(1);
      p.z = pos_l_real(2);
      vis_msg_.markers[7].points.push_back(p);
      p.x = pos_real(0);
      p.y = pos_real(1);
      p.z = pos_real(2);
      vis_msg_.markers[7].points.push_back(p);

      vis_pub_.publish(vis_msg_);
    }

    if (true_force_pub_ && current_time >= next_true_force_pub_time_) {
      next_true_force_pub_time_ += true_force_period_;
      geometry_msgs::AccelStamped true_force_msg;
      true_force_msg.header.stamp = current_time;
      true_force_msg.header.frame_id = "world";
      const Eigen::Vector3d fq_true = quadrotorPtr_->getTrueQuadForce();
      const Eigen::Vector3d fl_true = quadrotorPtr_->getTrueLoadForce();
      true_force_msg.accel.linear.x = fl_true.x();
      true_force_msg.accel.linear.y = fl_true.y();
      true_force_msg.accel.linear.z = fl_true.z();
      true_force_msg.accel.angular.x = fq_true.x();
      true_force_msg.accel.angular.y = fq_true.y();
      true_force_msg.accel.angular.z = fq_true.z();
      true_force_pub_.publish(true_force_msg);
    }
  }
 public:
  //这段代码实现了Nodelet类的onInit方法，这是ROS节点初始化的核心入口函数，负责完成仿真器的所有初始化工作
  //包括参数加载、模型创建、通信接口设置和可视化配置等。
  void onInit() override
  {
    ros::NodeHandle nh(getPrivateNodeHandle());
  // 1. 加载初始位置和姿态参数
  // - 从参数服务器加载无人机和负载的初始位置（init_x, init_y, init_z）
  // - 加载负载相对于无人机的初始位置（init_quadx, init_quady, init_quadz）
  // - 加载初始偏航角（init_yaw）
  double init_x, init_y, init_z, init_yaw;
  double init_quadx, init_quady, init_quadz;
  nh.getParam("init_x", init_x);
  nh.getParam("init_y", init_y);
  nh.getParam("init_z", init_z);
  nh.getParam("init_quadx", init_quadx);
  nh.getParam("init_quady", init_quady);
  nh.getParam("init_quadz", init_quadz);
  nh.getParam("init_yaw", init_yaw);
  // nh.param("init_q1", init_q1,0.0);
  // nh.param("init_q2", init_q2,0.0);
  // nh.param("init_q3", init_q3,-1.0);

  // config of quadrotor
  // 配置四旋翼物理参数（质量、惯性矩阵、电机参数等）
  Config config;
  nh.getParam("g",       config.g);
  nh.getParam("mass", config.mass); 
  nh.getParam("mass_load", config.mass_l);
  nh.getParam("l_length", config.l_length);
  l_length = config.l_length;
  double Ixx, Iyy, Izz;
  nh.getParam("Ixx", Ixx);
  nh.getParam("Iyy", Iyy);
  nh.getParam("Izz", Izz);
  config.J = Eigen::Vector3d(Ixx, Iyy, Izz).asDiagonal();
  nh.getParam("kf", config.kf);
  double prop_radius;
  nh.getParam("prop_radius", prop_radius);
  config.km = 0.07 * (3 * prop_radius) * config.kf;
  nh.getParam("arm_length", config.arm_length);
  nh.getParam("motor_time_constant", config.motor_time_constant);
  nh.getParam("max_rpm", config.max_rpm);
  nh.getParam("min_rpm", config.min_rpm);
  nh.getParam("simulation_rate", simulation_rate_);
  nh.getParam("odom_rate", odom_rate_);
  nh.param("true_force_rate", true_force_rate_, 100.0);
  nh.getParam("payload_size", payload_size_);

  if (odom_rate_ <= 0) {
    ROS_WARN("Invalid odom_rate (%d), fallback to 400 Hz", odom_rate_);
    odom_rate_ = 400;
  }
  odom_period_ = ros::Duration(1.0 / static_cast<double>(odom_rate_));

  if (true_force_rate_ <= 0.0) {
    ROS_WARN("Invalid true_force_rate (%.2f), fallback to odom_rate", true_force_rate_);
    true_force_rate_ = static_cast<double>(odom_rate_);
  }
  true_force_period_ = ros::Duration(1.0 / true_force_rate_);
  
  // 加载噪声参数
  nh.param("noise_uav", noise_uav_,0.0);
  nh.param("noise_payload", noise_payload_,0.0);
  nh.param("noise_vel_uav", noise_vel_uav_,0.0);
  nh.param("noise_vel_payload", noise_vel_payload_,0.0);
  nh.param("noise_acc_uav", noise_acc_uav_,0.0);
  nh.param("noise_acc_payload", noise_acc_payload_,0.0);
  nh.param("noise_rpm", noise_rpm_,0.0);
  
  std::vector<double> kom;
  if (!nh.getParam("kOmega", kom) || kom.size() < 3)
  {
    ROS_ERROR("[so3_quadrotor] Parameter kOmega missing or incomplete; using defaults [3,3,3].");
    kom = {3.0, 3.0, 3.0};
  }
  for (size_t i = 0; i < 3; i++)
  {
    config.kOmega[i] = kom[i];
  }
  std::vector<double> kdom;
  if (!nh.getParam("kdOmega", kdom) || kdom.size() < 3)
  {
    ROS_ERROR("[so3_quadrotor] Parameter kdOmega missing or incomplete; using defaults [15,15,15].");
    kdom = {15.0, 15.0, 15.0};
  }
  for (size_t i = 0; i < 3; i++)
  {
    config.kdOmega[i] = kdom[i];
  }
  
  rpm_msg_.esc_telemetry.resize(4);
  

  // 2. 动力学模型初始化
  // - 设置初始位置和姿态
  // - 创建四旋翼动力学模型实例
  // - 设置初始偏航角
  // - 计算并设置悬停所需的初始电机转速
  // - 检查并设置无人机和负载的初始位置
  // - 计算并设置初始缆绳方向向量
  // - 设置初始控制指令（悬停推力）
  quadrotorPtr_ = std::make_shared<Quadrotor>(config);
  quadrotorPtr_->setYpr(Eigen::Vector3d(init_yaw, 0, 0));
  double rpm = sqrt(config.mass * config.g / 4 / config.kf);
  quadrotorPtr_->setRpm(Eigen::Vector4d(rpm, rpm, rpm, rpm));

  Eigen::Vector3d q_unnorm = Eigen::Vector3d(init_x, init_y, init_z)-Eigen::Vector3d(init_quadx, init_quady, init_quadz);
  if (q_unnorm.norm() < 1.0e-8 || q_unnorm.norm() > (config.l_length+1.0e-8))
  {
    ROS_WARN("Invalid initial position of quadrotor, using the default value");
    init_quadx = init_x;
    init_quady = init_y;
    init_quadz = init_z + config.l_length;
    q_unnorm = Eigen::Vector3d(init_x, init_y, init_z)-Eigen::Vector3d(init_quadx, init_quady, init_quadz);
  }
  quadrotorPtr_->setPos(Eigen::Vector3d(init_x, init_y, init_z));
  quadrotorPtr_->setQuadPos(Eigen::Vector3d(init_quadx, init_quady, init_quadz));
  auto initq = q_unnorm.normalized();
  quadrotorPtr_->setq(initq);
  quadrotorPtr_->setYpr(Eigen::Vector3d(init_yaw,0.0,0.0));
  cmd_.thrust =  (config.mass+config.mass_l) * config.g;
  cmd_.body_rates.setZero();

  // 3. ROS通信接口设置
  // - 创建发布者（odom, imu, vis等）
  // - 创建订阅者（so3cmd, rpm等）
  // - 设置定时器（用于发布状态和控制指令）
  odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 100);
  odom_payload_pub_ = nh.advertise<nav_msgs::Odometry>("odom_payload", 100);
  imu_pub_  = nh.advertise<sensor_msgs::Imu>("imu", 10);
  cable_info_pub_  = nh.advertise<sensor_msgs::Imu>("cable_info", 10);
  load_imu_pub_ = nh.advertise<sensor_msgs::Imu>("load_imu", 10);
  vis_pub_= nh.advertise<visualization_msgs::MarkerArray>("vis", 10);
  wind_pub_ = nh.advertise<geometry_msgs::Vector3Stamped>("wind", 10);
  true_force_pub_ = nh.advertise<geometry_msgs::AccelStamped>("true_force", 10);
  cmd_sub_  = nh.subscribe<mavros_msgs::AttitudeTarget>("so3cmd", 10, &Nodelet::cmd_callback, this, ros::TransportHints().tcpNoDelay());
  rpm_pub_ = nh.advertise<mavros_msgs::ESCTelemetry>("rpm", 10);
  odom_msg_.header.frame_id = "world";
  imu_msg_ .header.frame_id = "world";

  ros::NodeHandle wind_nh(nh, "wind");
  ROS_INFO("[so3_quadrotor_nodelet] Loading wind field configuration.");
  wind_field_ = wind::createWindField(wind_nh);
  if (!wind_field_)
  {
    ROS_WARN("[so3_quadrotor_nodelet] Wind field instance is null; defaulting to zero wind.");
  }
  wind_msg_.header.frame_id = "world";

  // 4. 可视化设置
  // - 创建可视化标记发布者（四旋翼、负载、缆绳等） - 设置标记属性（颜色、大小、形状等）
  // - 螺旋桨标记 （4个）：使用圆柱体表示，设置尺寸、颜色等属性
  // - 无人机机体标记 （2个）：使用球体表示，形成十字交叉形状
  // - 负载标记 （1个）：使用球体表示
  // - 缆绳标记 （1个）：使用线段表示，连接无人机和负载
  vis_msg_.markers.resize(4);
  visualization_msgs::Marker propeller;
  propeller.header.frame_id = "world";
  propeller.type = visualization_msgs::Marker::CYLINDER;
  propeller.action = visualization_msgs::Marker::ADD;
  propeller.scale.x = 2 * prop_radius;
  propeller.scale.y = 0.1 * prop_radius;
  propeller.scale.z = 0.05 * prop_radius;
  propeller.color.a = 0.7;
  propeller.color.r = 0.6;
  propeller.color.g = 0.7;
  propeller.color.b = 0.8;
  for (size_t i=0; i<4; ++i) {
    vis_msg_.markers[i] = propeller;
    vis_msg_.markers[i].id = i;
  }
  visualization_msgs::Marker drone_body = propeller;
  drone_body.type = visualization_msgs::Marker::SPHERE;
  drone_body.color.a = 1.0;
  drone_body.scale.x = config.arm_length;
  drone_body.scale.y = 0.1 * config.arm_length;
  drone_body.scale.z = 0.05 * config.arm_length;
  vis_msg_.markers.push_back(drone_body);
  drone_body.scale.y = config.arm_length;
  drone_body.scale.x = 0.1 * config.arm_length;
  vis_msg_.markers.push_back(drone_body);
  
  visualization_msgs::Marker payload = propeller;
  payload.type = visualization_msgs::Marker::SPHERE;
  payload.color.a = 1.0;
  payload.color.r = 0.3;
  payload.color.g = 0.4;
  payload.color.b = 0.5;
  payload.scale.x = payload_size_*2;
  payload.scale.y = payload_size_*2;
  payload.scale.z = payload_size_*2;
  payload.id = (int32_t)vis_msg_.markers.size();
  vis_msg_.markers.push_back(payload);
  visualization_msgs::Marker payload_line = payload;
  payload_line.type = visualization_msgs::Marker::LINE_STRIP;
  payload_line.scale.x = 0.02;
  payload_line.id = (int32_t)vis_msg_.markers.size();
    vis_msg_.markers.push_back(payload_line);
    // drone_body.scale.y = config.arm_length;
    // drone_body.scale.x = 0.1 * config.arm_length;
    // vis_msg_.markers.push_back(payload);

  simulation_timer = nh.createTimer(ros::Duration(1.0/simulation_rate_), &Nodelet::timer_callback, this);
  }
};

} // so3_quadrotor

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(so3_quadrotor::Nodelet, nodelet::Nodelet);
