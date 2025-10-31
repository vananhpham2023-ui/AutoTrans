// ================================================================
// 文件：mpc_fsm.cpp 
// 职责：飞行状态机管理
// 主要功能：
//   - 三状态切换逻辑(MANUAL_CTRL/AUTO_HOVER/CMD_CTRL)
//   - 异常状态检测与处理(速度超限/通信超时)
//   - 飞行模式切换安全校验
//   - 紧急情况处理(自动降落/重启)
// ================================================================

#include "mpc_fsm.h"
#include <uav_utils/converters.h>
#include "geometry_msgs/Accel.h"
#include "std_msgs/Float64MultiArray.h"
#include "visualization_msgs/Marker.h"
#include <std_srvs/Empty.h>
#include <ros/package.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>
#include <cerrno>
#include <Eigen/Geometry>
#include <cmath>
using namespace std;
using namespace uav_utils;
#define USE_PX4_OR_ARDUPILOT 1 // 0: px4 1:ardupilot
namespace PayloadMPC
{

	MPCFSM::MPCFSM(const ros::NodeHandle &nh, MpcParams &params, MpcController &controller) : nh_(nh),
																							  params_(params),
																							  controller_(controller)
	{
		fsm_state = MANUAL_CTRL;
		exec_traj_state_ = HOVER;
		hover_pose_.setZero();
		hover_yaw_ = 0;
		analytic_start_time_ = ros::Time::now();

		pub_predicted_trajectory_ =
			nh_.advertise<nav_msgs::Path>("mpc/trajectory_predicted", 1);
		pub_payload_predicted_trajectory_ =
			nh_.advertise<nav_msgs::Path>("mpc/payload_trajectory_predicted", 1);
		pub_all_ref_data_ =
			nh_.advertise<nav_msgs::Path>("mpc/all_ref_data", 1);

		pub_reference_trajectory_ =
			nh_.advertise<nav_msgs::Path>("mpc/reference_trajectory", 1);
		pub_payload_reference_trajectory_ =
			nh_.advertise<nav_msgs::Path>("mpc/reference_payload_trajectory", 1);
		reference_geometry_pub_ =
			nh_.advertise<visualization_msgs::MarkerArray>("mpc/reference_geometry", 1);
	controller_.resetThrustMapping();

	pub_force_marker_ = nh_.advertise<visualization_msgs::Marker>("mpc/force_marker", 1);
	pub_force_ = nh_.advertise<geometry_msgs::Accel>("mpc/force", 1);

	pub_cable_ = nh_.advertise<geometry_msgs::PoseStamped>("mpc/cable_dir_reference", 1);

	pub_rmse_info_ = nh_.advertise<std_msgs::Float64MultiArray>("mpc/rmse_info", 1);
	reference_geometry_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("mpc/reference_geometry", 1);
	analytic_rmse_sum_quad_ = 0.0;
	analytic_rmse_sum_payload_ = 0.0;
	analytic_rmse_samples_ = 0;
	analytic_cycles_completed_ = 0;
	analytic_target_cycles_ = 10;
	analytic_cycle_time_ = std::numeric_limits<double>::infinity();
	analytic_total_duration_ = std::numeric_limits<double>::infinity();
	analytic_next_cycle_time_ = std::numeric_limits<double>::infinity();
	analytic_rmse_reported_ = false;
	analytic_plot_generated_ = false;
	actual_history_quad_.clear();
	actual_history_payload_.clear();

	force_estimator_.init(params_);
}

	void MPCFSM::setAnalyticTrajectory(std::unique_ptr<AnalyticTrajectory> trajectory)
	{
		analytic_traj_ = std::move(trajectory);
	if (analytic_traj_)
	{
		analytic_traj_->reset(0.0);
		analytic_start_time_ = ros::Time::now();
			reference_history_.clear();
			reference_payload_history_.clear();
			actual_history_quad_.clear();
			actual_history_payload_.clear();
			analytic_plot_generated_ = false;
		analytic_rmse_sum_quad_ = 0.0;
		analytic_rmse_sum_payload_ = 0.0;
		analytic_rmse_samples_ = 0;
		analytic_cycles_completed_ = 0;
		analytic_rmse_reported_ = false;
		analytic_cycle_time_ = analytic_traj_->getCycleTime();
		analytic_total_duration_ = analytic_traj_->getDuration();
		analytic_target_cycles_ = 10;
		if (std::isfinite(analytic_cycle_time_) && analytic_cycle_time_ > 0.0)
		{
			if (std::isfinite(analytic_total_duration_))
			{
				double available = analytic_total_duration_ / analytic_cycle_time_;
				int available_int = static_cast<int>(std::floor(available + 1e-6));
				if (available_int <= 0)
					available_int = 1;
				analytic_target_cycles_ = std::min(analytic_target_cycles_, available_int);
			}
			analytic_next_cycle_time_ = analytic_cycle_time_;
		}
		else
		{
			analytic_cycle_time_ = std::numeric_limits<double>::infinity();
			analytic_next_cycle_time_ = analytic_total_duration_;
			if (!std::isfinite(analytic_next_cycle_time_) || analytic_next_cycle_time_ <= 0.0)
			{
				analytic_next_cycle_time_ = std::numeric_limits<double>::infinity();
			}
		}
		if (analytic_target_cycles_ <= 0)
		{
			analytic_target_cycles_ = 1;
		}
	}
	}

	void MPCFSM::resetAnalyticStart(const ros::Time &stamp)
	{
		analytic_start_time_ = stamp;
	if (analytic_traj_)
	{
		analytic_traj_->reset(0.0);
	}
	reference_history_.clear();
	reference_payload_history_.clear();
	analytic_rmse_sum_quad_ = 0.0;
	analytic_rmse_sum_payload_ = 0.0;
	analytic_rmse_samples_ = 0;
	analytic_cycles_completed_ = 0;
	analytic_rmse_reported_ = false;
	analytic_plot_generated_ = false;
	actual_history_quad_.clear();
	actual_history_payload_.clear();
	if (std::isfinite(analytic_cycle_time_) && analytic_cycle_time_ > 0.0)
	{
		analytic_next_cycle_time_ = analytic_cycle_time_;
	}
	else if (std::isfinite(analytic_total_duration_) && analytic_total_duration_ > 0.0)
	{
		analytic_next_cycle_time_ = analytic_total_duration_;
	}
	else
	{
		analytic_next_cycle_time_ = std::numeric_limits<double>::infinity();
	}
}

	/*
			Finite State Machine

			   system start
				   /
				  /
				 v
	----- > MANUAL_CTRL
	|         ^   |
	|         |   |
	|         |   |
	|         |   |
	|         |   |
	|         |   v
	|       AUTO_HOVER
	|         ^   |
	|         |   |
	|         |	  |
	|         |   |
	|         |   v
	-------- CMD_CTRL

	*/
	
	/**
	* @brief 主状态机处理函数
	* @details 处理状态机切换逻辑，包括手动控制、自动悬停和指令控制三种状态
	* @note 需要在ROS主循环中周期性调用
	*/
	void MPCFSM::process()
	{
		ros::Time now_time = ros::Time::now(); 											//获取当前ROS时间

		setEstimateState(odom_data, payload_odom_data, cable_info_data);				//更新状态估计（无人机/负载位姿、线缆状态）
		setForceEstimation();															//执行力估计计算	
		
		//仿真模式处理 ：
		if (params_.use_simulation_)						//如果使用仿真模式
		{
			fsm_state = CMD_CTRL;
			rc_data.is_hover_mode = true;
			rc_data.is_command_mode = true;
			static bool is_first_time = true;
			if (is_first_time)
			{
				is_first_time = false;
				update_hover_pose();				    	// 首次进入时初始化悬停点
			}
		}

		//状态机主循环 ：
		switch (fsm_state)
		{
		case MANUAL_CTRL:
		{
			if (rc_data.enter_hover_mode) // Try to jump to AUTO_HOVER
			{
				if (!odom_is_received(now_time))
				{
					ROS_ERROR("[MPCctrl] Reject AUTO_HOVER(L2). No odom!");
					break;
				}
				if (odom_data.v.norm() > 3.0)
				{
					ROS_ERROR("[MPCctrl] Reject AUTO_HOVER(L2). Odom_Vel=%fm/s, which seems that the locolization module goes wrong!", odom_data.v.norm());
					break;
				}

				trajectory_data.exec_traj = 0; // clean the trajectory data
				update_hover_pose();
				controller_.resetThrustMapping();
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				fsm_state = AUTO_HOVER;
				toggle_offboard_mode(true);

				ROS_INFO("\033[32m[MPCctrl] MANUAL_CTRL(L1) --> AUTO_HOVER(L2)\033[32m");
			}

			if (rc_data.toggle_reboot) // Try to reboot. EKF2 based PX4 FCU requires reboot when its state estimator goes wrong.
			{
				if (state_data.current_state.armed)
				{
					ROS_ERROR("[MPCctrl] Reject reboot! Disarm the drone first!");
					break;
				}
				reboot_FCU();
			}

			break;
		}

		case AUTO_HOVER:
		{
			if (!rc_data.is_hover_mode || !odom_is_received(now_time))
			{
				fsm_state = MANUAL_CTRL;
				toggle_offboard_mode(false);

				ROS_WARN("[MPCctrl] AUTO_HOVER(L2) --> MANUAL_CTRL(L1)");
			}
			else if (rc_data.is_command_mode)
			{
				if (((USE_PX4_OR_ARDUPILOT == 1) && (state_data.current_state.mode == "GUIDED_NOGPS")) || ((USE_PX4_OR_ARDUPILOT == 0) && (state_data.current_state.mode == "OFFBOARD")))
				{
					update_hover_pose();
					controller_.setHoverReference(hover_pose_, hover_yaw_);
					controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
					fsm_state = CMD_CTRL;
					exec_traj_state_ = HOVER;
					ROS_INFO("\033[32m[MPCctrl] AUTO_HOVER(L1) --> CMD_CTRL(L2)\033[32m");
					// haojia MPC add
					publish_trigger(odom_data.msg);
					ROS_INFO("\033[32m[MPCctrl] TRIGGER sent, allow user command.\033[32m");
				}
			}
			else
			{
				update_hover_with_rc();
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				if (rc_data.enter_command_mode)
				{
					publish_trigger(odom_data.msg);
					ROS_INFO("\033[32m[MPCctrl] TRIGGER sent, allow user command.\033[32m");
				}

				// cout << "des.p=" << des.p.transpose() << endl;
			}

			break;
		}

		case CMD_CTRL:
		{
			if (!rc_data.is_hover_mode || !odom_is_received(now_time))
			{
				fsm_state = MANUAL_CTRL;
				toggle_offboard_mode(false);
				exec_traj_state_ = HOVER;

				ROS_WARN("[MPCctrl] From CMD_CTRL(L3) to MANUAL_CTRL(L1)!");
			}
			else if (!rc_data.is_command_mode)
			{
				fsm_state = AUTO_HOVER;
				exec_traj_state_ = HOVER; // Reset the state
				update_hover_pose();
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				ROS_INFO("[MPCctrl] From CMD_CTRL(L3) to AUTO_HOVER(L2)!");
			}
			else
			{
				CMD_CTRL_process();
			}

			break;
		}
		default:
			break;
		}

		// 公共执行段
		// 控制指令生成 ：
		if (fsm_state == AUTO_HOVER || fsm_state == CMD_CTRL)
		{
			controller_.estimateThrustModel(imu_data.a, odom_data.q, cable_info_data.w, cable_info_data.a, rpm_data.rpm_vec, bat_data.volt, params_);	 // 推力模型估计
			// controller_.resetThrustMapping();
			publish_bodyrate_ctrl(mpc_predicted_inputs_.col(0), now_time);											 			// 发布机体速率控制指令		
			publishPrediction(controller_.reference_states_, mpc_predicted_states_, now_time, controller_.getTimeStep());		// 发布预测轨迹
		}

		// 步骤6：清除超出其生命周期的标志
		rc_data.enter_hover_mode = false;
		rc_data.enter_command_mode = false;
		rc_data.toggle_reboot = false;
	}
	/*
		Finite State Machine

		   CMD_CTRL
			   /
			  /
			 v
		  HOVER
		  ^   |
		  |   |
		  |   |
		  |   |
		  |   |
		  |   v
		POLY_TRAJ


	*/
	/**
	* @brief 指令控制状态处理函数
	* @details 处理轨迹跟踪状态切换和跟踪过程
	* @note 仅在CMD_CTRL状态下被调用
	*/
	void MPCFSM::CMD_CTRL_process()
	{
		ros::Time now_time = ros::Time::now();
		const bool use_planner = params_.reference_config().use_planner;
		if (!use_planner && analytic_traj_)
		{
			if (exec_traj_state_ == HOVER)
			{
				resetAnalyticStart(now_time);
				exec_traj_state_ = ANALYTIC;
				ROS_INFO("[MPCctrl] HOVER --> ANALYTIC");
			}
		}
		else if (exec_traj_state_ == ANALYTIC)
		{
			resetAnalyticStart(now_time);
			exec_traj_state_ = HOVER;
			ROS_INFO("[MPCctrl] ANALYTIC --> HOVER");
		}
		switch (exec_traj_state_)
		{
		case HOVER:
		{
			if (now_time >= trajectory_data.total_traj_start_time &&
				now_time <= trajectory_data.total_traj_end_time &&
				trajectory_data.exec_traj == 1 && (!trajectory_data.traj_queue.empty()))
			{
				// same as the below
				update_hover_pose();
				oneTraj_Data_t *traj_info = &trajectory_data.traj_queue.front();
				traj_info = &trajectory_data.traj_queue.front();
				trajectory_data.total_traj_start_time = traj_info->traj_start_time;

				double traj_time = (now_time - traj_info->traj_start_time).toSec();
				controller_.setTrajectoyReference(traj_info->traj, traj_time, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);

				exec_traj_state_ = POLY_TRAJ;
				ROS_INFO("[MPCctrl] Receive the trajectory. HOVER --> POLY_TRAJ");
			}
			else
			{
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
			}
		}

		break;

		// POLY_TRAJ 状态表示当前正在执行多项式轨迹跟踪状态。在这个状态下，无人机将按照预设的多项式轨迹进行飞行。
		case POLY_TRAJ:
		{
			if (now_time < (trajectory_data.total_traj_start_time) || now_time > trajectory_data.total_traj_end_time || trajectory_data.exec_traj != 1 || trajectory_data.traj_queue.empty())
			{
				if (params_.use_trajectory_ending_pos_ && trajectory_data.exec_traj != -1)
				{
					// tracking the end point of the trajectory
					//  the hover pose is the end point of the trajectory
					auto &traj_info = trajectory_data.traj_queue.front().traj;
					hover_pose_ = traj_info.getJuncPos(traj_info.getPieceNum());
					hover_pose_(2) += params_.dyn_params_.l_length; // the hover pose is the quadrotor pose
					hover_yaw_ = get_yaw_from_quaternion(odom_data.q);
				}
				else
				{
					update_hover_pose();
				}
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				exec_traj_state_ = HOVER;
				ROS_INFO("[MPCctrl] Stop execute the trajectory. POLY_TRAJ --> HOVER");
				trajectory_data.exec_traj = 0;
				printandresetRMSE();
			}
			else
			{
				update_hover_pose();
				oneTraj_Data_t *traj_info = &trajectory_data.traj_queue.front();
				if (now_time < (traj_info->traj_start_time))
				{ // the start time of first trajectory should be whole trajectory start time
					trajectory_data.total_traj_start_time = traj_info->traj_start_time;
					controller_.setHoverReference(hover_pose_, hover_yaw_);
					controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				}
				else
				{
					if (trajectory_data.traj_queue.size() > 1)
					{
						oneTraj_Data_t *next_traj_info = &trajectory_data.traj_queue.at(1);
						while (now_time > next_traj_info->traj_start_time)
						{ // finish the first trajectory
							trajectory_data.traj_queue.pop_front();
							traj_info = &trajectory_data.traj_queue.front();
							trajectory_data.total_traj_start_time = traj_info->traj_start_time;
							trajectory_data.total_traj_end_time = trajectory_data.traj_queue.back().traj_end_time;
							if (trajectory_data.traj_queue.size() == 1)
							{
								break;
							}
							next_traj_info = &trajectory_data.traj_queue.at(1);
						}
					}

					double traj_time = (now_time - traj_info->traj_start_time).toSec();
					addRMSE();
					controller_.setTrajectoyReference(traj_info->traj, traj_time, hover_yaw_);
					controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				}
			}
		}
		break;

		case ANALYTIC:
		{
			if (!analytic_traj_)
			{
				update_hover_pose();
				controller_.setHoverReference(hover_pose_, hover_yaw_);
				controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);
				exec_traj_state_ = HOVER;
				break;
			}

			update_hover_pose();
			double traj_time = (now_time - analytic_start_time_).toSec();
			if (traj_time < 0.0)
			{
				traj_time = 0.0;
				resetAnalyticStart(now_time);
			}

			auto sampler = [this](double query_t) {
				return analytic_traj_->sample(query_t);
			};

		controller_.setAnalyticReference(sampler, traj_time, hover_yaw_);
		controller_.execMPC(est_state_, mpc_predicted_states_, mpc_predicted_inputs_);

		Eigen::Vector3d quad_actual(est_state_(kPosX), est_state_(kPosY), est_state_(kPosZ));
		Eigen::Vector3d payload_actual(est_state_(kPayloadX), est_state_(kPayloadY), est_state_(kPayloadZ));
		Eigen::Vector3d quad_ref(controller_.reference_states_(kPosX, 0), controller_.reference_states_(kPosY, 0), controller_.reference_states_(kPosZ, 0));
		Eigen::Vector3d payload_ref(controller_.reference_states_(kPayloadX, 0), controller_.reference_states_(kPayloadY, 0), controller_.reference_states_(kPayloadZ, 0));
		analytic_rmse_sum_quad_ += (quad_actual - quad_ref).squaredNorm();
		analytic_rmse_sum_payload_ += (payload_actual - payload_ref).squaredNorm();
		analytic_rmse_samples_++;
		if (std::isfinite(analytic_cycle_time_) && analytic_cycle_time_ > 0.0)
		{
			while (!analytic_rmse_reported_ && analytic_cycles_completed_ < analytic_target_cycles_ && traj_time >= analytic_next_cycle_time_)
			{
				analytic_cycles_completed_++;
				analytic_next_cycle_time_ += analytic_cycle_time_;
			}
		}
		else if (std::isfinite(analytic_total_duration_) && traj_time >= analytic_total_duration_)
		{
			analytic_cycles_completed_ = analytic_target_cycles_;
		}
	if (!analytic_rmse_reported_ && analytic_rmse_samples_ > 0 && (analytic_cycles_completed_ >= analytic_target_cycles_ || (std::isfinite(analytic_total_duration_) && traj_time >= analytic_total_duration_)))
	{
		double quad_rmse = std::sqrt(analytic_rmse_sum_quad_ / static_cast<double>(analytic_rmse_samples_));
		double payload_rmse = std::sqrt(analytic_rmse_sum_payload_ / static_cast<double>(analytic_rmse_samples_));
		ROS_INFO_STREAM("[MPCctrl] ANALYTIC RMSE (" << analytic_cycles_completed_ << "/" << analytic_target_cycles_
				       << " cycles) quad=" << quad_rmse << " m, payload=" << payload_rmse
				       << " m over " << analytic_rmse_samples_ << " samples.");
		analytic_rmse_reported_ = true;
		finalizeAnalyticRun(quad_rmse, payload_rmse);
	}

		const double duration = analytic_traj_->getDuration();
		if (std::isfinite(duration) && traj_time >= duration)
		{
			ROS_INFO_THROTTLE(5.0, "[MPCctrl] ANALYTIC trajectory reached duration %.2f s, holding terminal state.", duration);
			}
			else
			{
				ROS_INFO_THROTTLE(2.0, "[MPCctrl] ANALYTIC trajectory tracking t=%.2f s", traj_time);
			}
		}
		break;

		case POINTS:
		{
			exec_traj_state_ = HOVER;
			ROS_ERROR("[MPCctrl] Unknown exec_traj_state_! Jump to hover");
		}
		break;

		default:
		{
			exec_traj_state_ = HOVER;
			ROS_ERROR("[MPCctrl] Unknown exec_traj_state_! Jump to hover");
		}

		break;
		}
	}

	/**
	* @brief 设置估计状态
	* @param[in] odom_est_state 无人机里程计数据
	* @param[in] odom_payload_state 负载里程计数据 
	* @param[in] cable_info_data 缆绳IMU数据
	* @post 更新内部状态估计est_state_
	*/
	void MPCFSM::setEstimateState(const Odom_Data_t &odom_est_state, const Odom_Data_t &odom_payload_state, const Imu_Data_t &cable_info_data)
	{
		est_state_(kPosX) = odom_est_state.p[0];
		est_state_(kPosY) = odom_est_state.p[1];
		est_state_(kPosZ) = odom_est_state.p[2];
		auto rot_q = odom_est_state.q;
		rot_q.normalize();
		est_state_(kOriW) = rot_q.w();
		est_state_(kOriX) = rot_q.x();
		est_state_(kOriY) = rot_q.y();
		est_state_(kOriZ) = rot_q.z();
		est_state_(kVelX) = odom_est_state.v[0];
		est_state_(kVelY) = odom_est_state.v[1];
		est_state_(kVelZ) = odom_est_state.v[2];
		est_state_(kPayloadX) = odom_payload_state.p[0];
		est_state_(kPayloadY) = odom_payload_state.p[1];
		est_state_(kPayloadZ) = odom_payload_state.p[2];
		est_state_(kPayloadVx) = odom_payload_state.v[0];
		est_state_(kPayloadVy) = odom_payload_state.v[1];
		est_state_(kPayloadVz) = odom_payload_state.v[2];
		est_state_(kCableX) = cable_info_data.w[0];
		est_state_(kCableY) = cable_info_data.w[1];
		est_state_(kCableZ) = cable_info_data.w[2];

		est_state_(kDcableX) = cable_info_data.a[0];
		est_state_(kDcableY) = cable_info_data.a[1];
		est_state_(kDcableZ) = cable_info_data.a[2];
	}

	void MPCFSM::addNewForceObseverState()
	{
		if (rpm_data.filtered_rpm[0] < 1000 && params_.force_estimator_param_.use_force_estimator) // Must have the real num
		{
			ROS_ERROR_THROTTLE(1.0, "[MPC CTRL]RPM TOO LOW");
			return;
		}
		// force_estimator_.setSystemState(imu_data.filtered_a,imu_data.q,payload_imu_data.filtered_a,payload_imu_data.q,cable_info_data.filtered_w,rpm_data.filtered_rpm);
		Eigen::Vector3d cable_dir = cable_info_data.filtered_w.normalized();
		force_estimator_.setSystemState(imu_data.filtered_a, odom_data.q, payload_imu_data.filtered_a, payload_imu_data.q, cable_dir, rpm_data.filtered_rpm);
		// force_estimator_.setSystemState(imu_data.a,imu_data.q,cable_imu_data.a,cable_imu_data.q,cable_info_data.w,rpm_data.rpm_vec);
	}

	/**
	* @brief 力估计处理函数
	* @details 计算并发布负载和无人机的估计力
	* @post 更新fl_和fq_向量，并通过controller_.setExternalForce()设置到控制器
	*/
	void MPCFSM::setForceEstimation()
	{

		force_estimator_.caculate_force(fl_, fq_);

		controller_.setExternalForce(fl_, fq_);
		// controller_.setExternalForce(Eigen::Vector3d::Zero(),fq_);  						// 仅设置无人机力
		// controller_.setExternalForce(Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero()); 	// 清零所有外力

		// Publish the force
		if (pub_force_.getNumSubscribers() > 0 || pub_force_marker_.getNumSubscribers() > 0)
		{
			geometry_msgs::Accel force_msg;
			force_msg.linear.x = fl_(0);
			force_msg.linear.y = fl_(1);
			force_msg.linear.z = fl_(2);
			force_msg.angular.x = fq_(0);
			force_msg.angular.y = fq_(1);
			force_msg.angular.z = fq_(2);
			pub_force_.publish(force_msg);

			visualization_msgs::Marker force_marker;
			force_marker.header.frame_id = "world";
			force_marker.header.stamp = ros::Time::now();
			force_marker.ns = "force";
			force_marker.id = 0;
			force_marker.type = visualization_msgs::Marker::ARROW;
			force_marker.action = visualization_msgs::Marker::ADD;
			force_marker.pose.position.x = est_state_(kPosX);
			force_marker.pose.position.y = est_state_(kPosY);
			force_marker.pose.position.z = est_state_(kPosZ);
			Eigen::Quaterniond q_fq(Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), fq_));
			q_fq.normalize();
			force_marker.pose.orientation.x = q_fq.x();
			force_marker.pose.orientation.y = q_fq.y();
			force_marker.pose.orientation.z = q_fq.z();
			force_marker.pose.orientation.w = q_fq.w();
			force_marker.scale.x = fq_.norm();
			force_marker.scale.y = 0.05;
			force_marker.scale.z = 0.05;
			force_marker.color.a = 1.0;
			force_marker.color.r = 1.0;
			force_marker.color.g = 0.0;
			force_marker.color.b = 0.0;
			pub_force_marker_.publish(force_marker);

			force_marker.id = 1;
			force_marker.pose.position.x = est_state_(kPosX) + cable_info_data.w(0) * params_.dyn_params_.l_length;
			force_marker.pose.position.y = est_state_(kPosY) + cable_info_data.w(1) * params_.dyn_params_.l_length;
			force_marker.pose.position.z = est_state_(kPosZ) + cable_info_data.w(2) * params_.dyn_params_.l_length;
			Eigen::Quaterniond q_fl(Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), fl_));
			q_fl.normalize();
			force_marker.pose.orientation.x = q_fl.x();
			force_marker.pose.orientation.y = q_fl.y();
			force_marker.pose.orientation.z = q_fl.z();
			force_marker.pose.orientation.w = q_fl.w();
			force_marker.scale.x = fl_.norm();
			pub_force_marker_.publish(force_marker);
		}
	}
	/**
	* @brief 计算并重置轨迹跟踪的RMSE统计指标
	* @details 该函数实现以下功能：
	* - 计算无人机和负载的三维空间RMSE及XY平面RMSE
	* - 计算跟踪过程中的最大位置偏差
	* - 通过ROS_INFO输出统计信息（需开启print_info参数）
	* - 发布包含8个统计指标的Float64MultiArray消息（顺序为：
	*   [0]无人机三维RMSE, [1]负载三维RMSE, 
	*   [2]无人机XY RMSE, [3]负载XY RMSE,
	*   [4]无人机最大偏差, [5]负载最大偏差,
	*   [6]无人机XY最大偏差, [7]负载XY最大偏差）
	* - 重置所有统计计数器
	* @note 该函数会同时清除历史统计数据，调用后统计从零开始
	*/
	void MPCFSM::printandresetRMSE()
	{
		double drone_rmse = sqrt(rmse_sum_ / rmse_cnt_);
		double payload_rmse = sqrt(rmse_payload_sum_ / rmse_cnt_);
		double drone_rmse_xy = sqrt(rmse_xy_sum_ / rmse_cnt_);
		double payload_rmse_xy = sqrt(rmse_payload_xy_sum_ / rmse_cnt_);
		double drone_max = sqrt(drone_max_);
		double payload_max = sqrt(payload_max_);
		double drone_max_xy = sqrt(drone_max_xy_);
		double payload_max_xy = sqrt(payload_max_xy_);
		if (params_.print_info_)
		{
			ROS_INFO("\033[32m[MPCctrl] Tracking Drone RMSE = %lf m.\033[32m", drone_rmse);
			ROS_INFO("\033[32m[MPCctrl] Tracking Payload RMSE = %lf m.\033[32m", payload_rmse);
			ROS_INFO("\033[32m[MPCctrl] Tracking Drone RMSE_XY = %lf m.\033[32m", drone_rmse_xy);
			ROS_INFO("\033[32m[MPCctrl] Tracking Payload RMSE_XY = %lf m.\033[32m", payload_rmse_xy);
			ROS_INFO("\033[32m[MPCctrl] Tracking Drone MAX = %lf m.\033[32m", drone_max);
			ROS_INFO("\033[32m[MPCctrl] Tracking Payload MAX = %lf m.\033[32m", payload_max);
			ROS_INFO("\033[32m[MPCctrl] Tracking Drone MAX_XY = %lf m.\033[32m", drone_max_xy);
			ROS_INFO("\033[32m[MPCctrl] Tracking Payload MAX_XY = %lf m.\033[32m", payload_max_xy);
		}

		std_msgs::Float64MultiArray msg;
		msg.data.resize(8);
		msg.data[0] = drone_rmse;
		msg.data[1] = payload_rmse;
		msg.data[2] = drone_rmse_xy;
		msg.data[3] = payload_rmse_xy;
		msg.data[4] = drone_max;
		msg.data[5] = payload_max;
		msg.data[6] = drone_max_xy;
		msg.data[7] = payload_max_xy;
		pub_rmse_info_.publish(msg);

		rmse_cnt_ = 0;
		rmse_sum_ = 0;
		rmse_payload_sum_ = 0;
		rmse_xy_sum_ = 0;
		rmse_payload_xy_sum_ = 0;
		payload_max_ = 0;
		drone_max_ = 0;
		payload_max_xy_ = 0;
		drone_max_xy_ = 0;
	}

	void MPCFSM::addRMSE()
	{
		rmse_cnt_++;
		double pos_err = pow((odom_data.p[0] - controller_.reference_states_(kPosX, 0)), 2) +
						 pow((odom_data.p[1] - controller_.reference_states_(kPosY, 0)), 2) +
						 pow((odom_data.p[2] - controller_.reference_states_(kPosZ, 0)), 2);

		double pos_xy_err = pow((odom_data.p[0] - controller_.reference_states_(kPosX, 0)), 2) +
							pow((odom_data.p[1] - controller_.reference_states_(kPosY, 0)), 2);
		double pos_payload_err = pow((payload_odom_data.p[0] - controller_.reference_states_(kPayloadX, 0)), 2) +
								 pow((payload_odom_data.p[1] - controller_.reference_states_(kPayloadY, 0)), 2) +
								 pow((payload_odom_data.p[2] - controller_.reference_states_(kPayloadZ, 0)), 2);
		double pos_payload_xy_err = pow((payload_odom_data.p[0] - controller_.reference_states_(kPayloadX, 0)), 2) +
									pow((payload_odom_data.p[1] - controller_.reference_states_(kPayloadY, 0)), 2);

		rmse_sum_ += pos_err;
		rmse_xy_sum_ += pos_xy_err;
		rmse_payload_sum_ += pos_payload_err;
		rmse_payload_xy_sum_ += pos_payload_xy_err;

		if (drone_max_ < pos_err)
		{
			drone_max_ = pos_err;
		}
		if (drone_max_xy_ < pos_xy_err)
		{
			drone_max_xy_ = pos_xy_err;
		}
		if (payload_max_ < pos_payload_err)
		{
			payload_max_ = pos_payload_err;
		}
		if (payload_max_xy_ < pos_payload_xy_err)
		{
			payload_max_xy_ = pos_payload_xy_err;
		}
	}

	/**
	* @brief 更新悬停位置
	* @details 根据当前里程计数据更新悬停位置和偏航角
	* @post 更新hover_pose_和hover_yaw_
	*/
	void MPCFSM::update_hover_pose()
	{
		last_set_hover_pose_time = ros::Time::now();
		hover_pose_ = odom_data.p;
		// hover_pose_(0) = params_.pos_x_;
		// hover_pose_(1) = params_.pos_y_;
		// hover_pose_(2) = params_.takeoff_height_;

		hover_yaw_ = get_yaw_from_quaternion(odom_data.q);
	}

	/**
	* @brief 根据遥控器输入更新悬停位置
	* @details 根据遥控器通道输入计算新的悬停位置
	* @note 仅在AUTO_HOVER状态下被调用
	*/
	void MPCFSM::update_hover_with_rc()
	{
		ros::Time now = ros::Time::now();
		double delta_t = (now - last_set_hover_pose_time).toSec();
		last_set_hover_pose_time = now;

		hover_pose_(0) += rc_data.ch[1] * params_.max_manual_vel_ * delta_t * (params_.rc_reverse_.pitch ? 1 : -1);
		hover_pose_(1) += rc_data.ch[0] * params_.max_manual_vel_ * delta_t * (params_.rc_reverse_.roll ? 1 : -1);
		hover_pose_(2) += rc_data.ch[2] * params_.max_manual_vel_ * delta_t * (params_.rc_reverse_.throttle ? 1 : -1);
		hover_yaw_ += rc_data.ch[3] * params_.max_manual_vel_ * delta_t * (params_.rc_reverse_.yaw ? 1 : -1);

		if (hover_pose_(2) < -0.3)
			hover_pose_(2) = -0.3;
	}

	/**
	* @brief 发布预测轨迹
	* @param[in] reference_states 参考状态序列
	* @param[in] predicted_traj 预测轨迹状态序列  
	* @param[in] time 当前时间戳
	* @param[in] dt 时间步长
	* @post 发布四种轨迹消息到对应ROS话题
	*/
	void MPCFSM::publishPrediction(
		const Eigen::Ref<const Eigen::Matrix<real_t, kStateSize, kSamples + 1>> reference_states,
		const Eigen::Ref<const Eigen::Matrix<real_t, kStateSize, kSamples + 1>> predicted_traj,
		ros::Time &time, double dt)
	{
		nav_msgs::Path path_msg;
		nav_msgs::Path payload_path_msg;

		nav_msgs::Path reference_path_msg;
		nav_msgs::Path reference_payload_path_msg;
		path_msg.header.stamp = time;
		path_msg.header.frame_id = "world";
		payload_path_msg.header = path_msg.header;
		reference_path_msg.header = path_msg.header;
		reference_payload_path_msg.header = path_msg.header;
		geometry_msgs::PoseStamped pose;
		geometry_msgs::PoseStamped payload_pose;

		geometry_msgs::PoseStamped cable_dir_ref;

		geometry_msgs::PoseStamped reference_pose;
		geometry_msgs::PoseStamped reference_payload_pose;
		// std::cout << "pub_prediction" << std:: endl;
	cable_dir_ref.header = path_msg.header;
	cable_dir_ref.pose.position.x = reference_states(kCableX, 0);
	cable_dir_ref.pose.position.y = reference_states(kCableY, 0);
	cable_dir_ref.pose.position.z = reference_states(kCableZ, 0);

	for (int i = 0; i < kSamples + 1; i++)
	{
		// pub predicted
			pose.header.stamp = time + ros::Duration(i * dt);
			pose.header.seq = i;
			payload_pose.header = pose.header;
			pose.pose.position.x = predicted_traj(kPosX, i);
			pose.pose.position.y = predicted_traj(kPosY, i);
			pose.pose.position.z = predicted_traj(kPosZ, i);
			;
			pose.pose.orientation.w = predicted_traj(kOriW, i);
			pose.pose.orientation.x = predicted_traj(kOriX, i);
			pose.pose.orientation.y = predicted_traj(kOriY, i);
			pose.pose.orientation.z = predicted_traj(kOriZ, i);

			payload_pose.pose.position.x = predicted_traj(kPayloadX, i);
			payload_pose.pose.position.y = predicted_traj(kPayloadY, i);
			payload_pose.pose.position.z = predicted_traj(kPayloadZ, i);
			payload_pose.pose.orientation.w = 1.0;
			payload_pose.pose.orientation.x = 0.0;
			payload_pose.pose.orientation.y = 0.0;
			payload_pose.pose.orientation.z = 0.0;

			path_msg.poses.push_back(pose);
			payload_path_msg.poses.push_back(payload_pose);

			reference_pose.header.stamp = time + ros::Duration(i * dt);
			reference_pose.header.seq = i;
			// reference_payload_pose.header = reference_pose.header;
			reference_pose.pose.position.x = reference_states(kPosX, i);
			reference_pose.pose.position.y = reference_states(kPosY, i);
			reference_pose.pose.position.z = reference_states(kPosZ, i);
			;
			reference_pose.pose.orientation.w = reference_states(kOriW, i);
			reference_pose.pose.orientation.x = reference_states(kOriX, i);
			reference_pose.pose.orientation.y = reference_states(kOriY, i);
			reference_pose.pose.orientation.z = reference_states(kOriZ, i);

			reference_payload_pose.pose.position.x = reference_states(kPayloadX, i);
			reference_payload_pose.pose.position.y = reference_states(kPayloadY, i);
			reference_payload_pose.pose.position.z = reference_states(kPayloadZ, i);
			reference_payload_pose.pose.orientation.w = 1.0;
			reference_payload_pose.pose.orientation.x = 0.0;
			reference_payload_pose.pose.orientation.y = 0.0;
			reference_payload_pose.pose.orientation.z = 0.0;

			reference_path_msg.poses.push_back(reference_pose);
			reference_payload_path_msg.poses.push_back(reference_payload_pose);
		}

	if (exec_traj_state_ != ANALYTIC)
	{
		reference_history_.clear();
		reference_payload_history_.clear();
		actual_history_quad_.clear();
		actual_history_payload_.clear();
	}
	else
	{
		reference_history_.reserve(reference_history_.size() + 1);
		reference_payload_history_.reserve(reference_payload_history_.size() + 1);
	}

	if (!reference_path_msg.poses.empty())
	{
		const auto &front = reference_path_msg.poses.front();
		bool append_sample = true;
		if (!reference_history_.empty())
		{
			const auto &last = reference_history_.back();
			double dx = front.pose.position.x - last.pose.position.x;
			double dy = front.pose.position.y - last.pose.position.y;
			double dz = front.pose.position.z - last.pose.position.z;
			if ((dx * dx + dy * dy + dz * dz) < 1e-6)
			{
				append_sample = false;
			}
		}
		if (append_sample)
		{
			reference_history_.push_back(front);
			reference_payload_history_.push_back(reference_payload_path_msg.poses.front());
		}
	}

	nav_msgs::Path reference_full_msg = reference_path_msg;
	nav_msgs::Path reference_payload_full_msg = reference_payload_path_msg;
	reference_full_msg.header = reference_path_msg.header;
	reference_payload_full_msg.header = reference_payload_path_msg.header;

	if (!reference_history_.empty())
	{
		reference_full_msg.poses.assign(reference_history_.begin(), reference_history_.end());
		reference_payload_full_msg.poses.assign(reference_payload_history_.begin(), reference_payload_history_.end());
		if (reference_path_msg.poses.size() > 1)
		{
			reference_full_msg.poses.insert(reference_full_msg.poses.end(), reference_path_msg.poses.begin() + 1, reference_path_msg.poses.end());
			reference_payload_full_msg.poses.insert(reference_payload_full_msg.poses.end(), reference_payload_path_msg.poses.begin() + 1, reference_payload_path_msg.poses.end());
		}
	}

	if (exec_traj_state_ == ANALYTIC)
	{
		auto append_actual = [](std::vector<geometry_msgs::PoseStamped> &history, const geometry_msgs::PoseStamped &pose) {
			if (history.empty())
			{
				history.push_back(pose);
				return;
			}
			const auto &last = history.back();
			double dx = pose.pose.position.x - last.pose.position.x;
			double dy = pose.pose.position.y - last.pose.position.y;
			double dz = pose.pose.position.z - last.pose.position.z;
			if ((dx * dx + dy * dy + dz * dz) > 1e-6)
			{
				history.push_back(pose);
			}
		};

		geometry_msgs::PoseStamped actual_quad_pose;
		actual_quad_pose.header = path_msg.header;
		actual_quad_pose.header.stamp = time;
		actual_quad_pose.pose.position.x = est_state_(kPosX);
		actual_quad_pose.pose.position.y = est_state_(kPosY);
		actual_quad_pose.pose.position.z = est_state_(kPosZ);
		actual_quad_pose.pose.orientation.w = est_state_(kOriW);
		actual_quad_pose.pose.orientation.x = est_state_(kOriX);
		actual_quad_pose.pose.orientation.y = est_state_(kOriY);
		actual_quad_pose.pose.orientation.z = est_state_(kOriZ);
		append_actual(actual_history_quad_, actual_quad_pose);

		geometry_msgs::PoseStamped actual_payload_pose;
		actual_payload_pose.header = path_msg.header;
		actual_payload_pose.header.stamp = time;
		actual_payload_pose.pose.position.x = est_state_(kPayloadX);
		actual_payload_pose.pose.position.y = est_state_(kPayloadY);
		actual_payload_pose.pose.position.z = est_state_(kPayloadZ);
		actual_payload_pose.pose.orientation.w = 1.0;
		actual_payload_pose.pose.orientation.x = 0.0;
		actual_payload_pose.pose.orientation.y = 0.0;
		actual_payload_pose.pose.orientation.z = 0.0;
		append_actual(actual_history_payload_, actual_payload_pose);
	}

	pub_predicted_trajectory_.publish(path_msg);
	pub_payload_predicted_trajectory_.publish(payload_path_msg);
	pub_reference_trajectory_.publish(reference_full_msg);
	pub_payload_reference_trajectory_.publish(reference_payload_full_msg);
	pub_cable_.publish(cable_dir_ref);

	visualization_msgs::MarkerArray geometry_markers;
	geometry_markers.markers.reserve(6);

	auto make_sphere_marker = [&](int id, const std::string &ns, const Eigen::Vector3d &pos, const Eigen::Quaterniond &quat,
			 float r, float g, float b, float a, double scale) {
		visualization_msgs::Marker marker;
		marker.header = path_msg.header;
		marker.ns = ns;
		marker.id = id;
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = pos.x();
		marker.pose.position.y = pos.y();
		marker.pose.position.z = pos.z();
		marker.pose.orientation.w = quat.w();
		marker.pose.orientation.x = quat.x();
		marker.pose.orientation.y = quat.y();
		marker.pose.orientation.z = quat.z();
		marker.scale.x = marker.scale.y = marker.scale.z = scale;
		marker.color.r = r;
		marker.color.g = g;
		marker.color.b = b;
		marker.color.a = a;
		return marker;
	};

	auto make_cable_marker = [&](int id, const std::string &ns, const Eigen::Vector3d &start, const Eigen::Vector3d &end,
		 float r, float g, float b, float a) {
		visualization_msgs::Marker marker;
		marker.header = path_msg.header;
		marker.ns = ns;
		marker.id = id;
		marker.type = visualization_msgs::Marker::LINE_STRIP;
		marker.action = visualization_msgs::Marker::ADD;
		marker.scale.x = 0.02;
		marker.color.r = r;
		marker.color.g = g;
		marker.color.b = b;
		marker.color.a = a;
		geometry_msgs::Point p;
		p.x = start.x();
		p.y = start.y();
		p.z = start.z();
		marker.points.push_back(p);
		p.x = end.x();
		p.y = end.y();
		p.z = end.z();
		marker.points.push_back(p);
		return marker;
	};

	Eigen::Vector3d predicted_quad(predicted_traj(kPosX, 0), predicted_traj(kPosY, 0), predicted_traj(kPosZ, 0));
	Eigen::Vector3d predicted_payload(predicted_traj(kPayloadX, 0), predicted_traj(kPayloadY, 0), predicted_traj(kPayloadZ, 0));
	Eigen::Quaterniond predicted_quat(predicted_traj(kOriW, 0), predicted_traj(kOriX, 0), predicted_traj(kOriY, 0), predicted_traj(kOriZ, 0));
	Eigen::Vector3d reference_quad(reference_states(kPosX, 0), reference_states(kPosY, 0), reference_states(kPosZ, 0));
	Eigen::Vector3d reference_payload(reference_states(kPayloadX, 0), reference_states(kPayloadY, 0), reference_states(kPayloadZ, 0));
	Eigen::Quaterniond reference_quat(reference_states(kOriW, 0), reference_states(kOriX, 0), reference_states(kOriY, 0), reference_states(kOriZ, 0));

	geometry_markers.markers.push_back(make_sphere_marker(0, "predicted", predicted_quad, predicted_quat, 0.2f, 0.6f, 1.0f, 0.9f, 0.4));
	geometry_markers.markers.push_back(make_sphere_marker(1, "predicted", predicted_payload, Eigen::Quaterniond::Identity(), 1.0f, 0.5f, 0.2f, 0.9f, 0.28));
	geometry_markers.markers.push_back(make_cable_marker(2, "predicted", predicted_quad, predicted_payload, 0.2f, 0.6f, 1.0f, 0.9f));
	geometry_markers.markers.push_back(make_sphere_marker(3, "reference", reference_quad, reference_quat, 1.0f, 0.9f, 0.2f, 0.8f, 0.35));
	geometry_markers.markers.push_back(make_sphere_marker(4, "reference", reference_payload, Eigen::Quaterniond::Identity(), 1.0f, 0.3f, 0.3f, 0.8f, 0.24));
	geometry_markers.markers.push_back(make_cable_marker(5, "reference", reference_quad, reference_payload, 1.0f, 0.9f, 0.2f, 0.8f));

	reference_geometry_pub_.publish(geometry_markers);

		nav_msgs::Path all_ref;
		all_ref.header = path_msg.header;
		geometry_msgs::PoseStamped data_point;
		data_point.header = path_msg.header;

		// Position & atiiude 0
		data_point.pose.position.x = reference_states(kPosX, 0);
		data_point.pose.position.y = reference_states(kPosY, 0);
		data_point.pose.position.z = reference_states(kPosZ, 0);
		data_point.pose.orientation.w = reference_states(kOriW, 0);
		data_point.pose.orientation.x = reference_states(kOriX, 0);
		data_point.pose.orientation.y = reference_states(kOriY, 0);
		data_point.pose.orientation.z = reference_states(kOriZ, 0);
		all_ref.poses.push_back(data_point);
		// Velocity 1
		data_point.pose.position.x = reference_states(kVelX, 0);
		data_point.pose.position.y = reference_states(kVelY, 0);
		data_point.pose.position.z = reference_states(kVelZ, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);

		// payload position & atiiude 2
		data_point.pose.position.x = reference_states(kPayloadX, 0);
		data_point.pose.position.y = reference_states(kPayloadY, 0);
		data_point.pose.position.z = reference_states(kPayloadZ, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);
		// payload velocity 3
		data_point.pose.position.x = reference_states(kPayloadVx, 0);
		data_point.pose.position.y = reference_states(kPayloadVy, 0);
		data_point.pose.position.z = reference_states(kPayloadVz, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);

		// cable direction 4
		data_point.pose.position.x = reference_states(kCableX, 0);
		data_point.pose.position.y = reference_states(kCableY, 0);
		data_point.pose.position.z = reference_states(kCableZ, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);

		// cable velocity 5
		data_point.pose.position.x = reference_states(kDcableX, 0);
		data_point.pose.position.y = reference_states(kDcableY, 0);
		data_point.pose.position.z = reference_states(kDcableZ, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);

		// angular veolcity 6
		data_point.pose.position.x = controller_.reference_inputs_(kRateX, 0);
		data_point.pose.position.y = controller_.reference_inputs_(kRateY, 0);
		data_point.pose.position.z = controller_.reference_inputs_(kRateZ, 0);
		data_point.pose.orientation.w = 1.0;
		data_point.pose.orientation.x = 0.0;
		data_point.pose.orientation.y = 0.0;
		data_point.pose.orientation.z = 0.0;
		all_ref.poses.push_back(data_point);
		// Thrust 7
		data_point.pose.position.x = controller_.reference_inputs_(kThrust, 0);
		data_point.pose.position.y = 0.0;
		data_point.pose.position.z = 0.0;
		all_ref.poses.push_back(data_point);

		pub_all_ref_data_.publish(all_ref);
	}

	bool MPCFSM::rc_is_received(const ros::Time &now_time)
	{
		return (now_time - rc_data.rcv_stamp).toSec() < params_.msg_timeout_.rc;
	}

	bool MPCFSM::odom_is_received(const ros::Time &now_time)
	{
		return (now_time - odom_data.rcv_stamp).toSec() < params_.msg_timeout_.odom;
	}

	bool MPCFSM::imu_is_received(const ros::Time &now_time)
	{
		return (now_time - imu_data.rcv_stamp).toSec() < params_.msg_timeout_.imu;
	}

	bool MPCFSM::bat_is_received(const ros::Time &now_time)
	{
		return (now_time - bat_data.rcv_stamp).toSec() < params_.msg_timeout_.bat;
	}
	bool MPCFSM::recv_new_odom()
	{
		if (odom_data.rcv_new_msg)
		{
			odom_data.rcv_new_msg = false;
			return true;
		}
		return false;
	}

	void MPCFSM::publish_bodyrate_ctrl(const Eigen::Ref<const Eigen::Matrix<real_t, kInputSize, 1>> predicted_input,
									   const ros::Time &stamp)
	{
		mavros_msgs::AttitudeTarget msg;

		msg.header.stamp = stamp;
		msg.header.frame_id = std::string("FCU");

		msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;

		// double collective_thrust;
		Eigen::Vector3d bodyrates;

		// collective_thrust = input_bounded(INPUT_BODYRATE::kThrust);
		bodyrates << predicted_input(INPUT_BODYRATE::kRateX), predicted_input(INPUT_BODYRATE::kRateY),
			predicted_input(INPUT_BODYRATE::kRateZ);

		msg.body_rate.x = bodyrates[0];
		msg.body_rate.y = bodyrates[1];
		msg.body_rate.z = bodyrates[2];

		// if(collective_thrust_normalized > 0.7) collective_thrust_normalized = 0.7;
		if (params_.use_simulation_)
		{
			msg.thrust = predicted_input(INPUT_BODYRATE::kThrust);
		}
		else
		{
			msg.thrust = controller_.convertThrust(predicted_input(INPUT_BODYRATE::kThrust), bat_data.volt);
		}

		ctrl_FCU_pub.publish(msg);
	}

	void MPCFSM::publish_trigger(const nav_msgs::Odometry &odom_msg)
	{
		geometry_msgs::PoseStamped msg;
		msg.header.frame_id = "world";
		msg.pose = odom_msg.pose.pose;

		traj_start_trigger_pub.publish(msg);
	}

	bool MPCFSM::toggle_offboard_mode(bool on_off)
	{
		mavros_msgs::SetMode offb_set_mode;

#if (USE_PX4_OR_ARDUPILOT == 0)
		{
			if (on_off)
			{
				state_data.state_before_offboard = state_data.current_state;
				if (state_data.state_before_offboard.mode == "OFFBOARD") // Not allowed
					state_data.state_before_offboard.mode = "MANUAL";

				offb_set_mode.request.custom_mode = "OFFBOARD";
				if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
				{
					ROS_ERROR("Enter OFFBOARD rejected by PX4!");
					return false;
				}
			}
			else
			{
				offb_set_mode.request.custom_mode = state_data.state_before_offboard.mode;
				if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
				{
					ROS_ERROR("Exit OFFBOARD rejected by PX4!");
					return false;
				}
			}
		}
#else
		{
			if (on_off)
			{
				state_data.state_before_offboard = state_data.current_state;
				if (state_data.state_before_offboard.mode == "GUIDED_NOGPS") // Not allowed
					state_data.state_before_offboard.mode = "STABILIZE";

				offb_set_mode.request.custom_mode = "GUIDED_NOGPS";
				if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
				{
					ROS_ERROR("Enter OFFBOARD rejected by ARDUPILOT!");
					return false;
				}
			}
			else
			{
				offb_set_mode.request.custom_mode = state_data.state_before_offboard.mode;
				if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
				{
					ROS_ERROR("Exit OFFBOARD rejected by ARDUPILOT!");
					return false;
				}
			}
		}
#endif

		return true;

		// if (params_.print_dbg)
		// 	printf("offb_set_mode mode_sent=%d(uint8_t)\n", offb_set_mode.response.mode_sent);
	}

	bool MPCFSM::toggle_arm_disarm(bool arm)
	{
		mavros_msgs::CommandBool arm_cmd;
		arm_cmd.request.value = arm;
		if (!(arming_client_srv.call(arm_cmd) && arm_cmd.response.success))
		{
			if (arm)
				ROS_ERROR("ARM rejected by PX4!");
			else
				ROS_ERROR("DISARM rejected by PX4!");

			return false;
		}

		return true;
	}

	void MPCFSM::reboot_FCU()
	{
		// https://mavlink.io/en/messages/common.html, MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN(#246)
		mavros_msgs::CommandLong reboot_srv;
		reboot_srv.request.broadcast = false;
		reboot_srv.request.command = 246; // MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
		reboot_srv.request.param1 = 1;	  // Reboot autopilot
		reboot_srv.request.param2 = 0;	  // Do nothing for onboard computer
		reboot_srv.request.confirmation = true;

		reboot_FCU_srv.call(reboot_srv);

		ROS_INFO("Reboot FCU");

		// if (params_.print_dbg)
		// 	printf("reboot result=%d(uint8_t), success=%d(uint8_t)\n", reboot_srv.response.result, reboot_srv.response.success);
	}
void MPCFSM::finalizeAnalyticRun(double quad_rmse, double payload_rmse)
{
	if (analytic_plot_generated_)
	{
		return;
	}
	analytic_plot_generated_ = true;
	std::string package_path = ros::package::getPath("payload_mpc_controller");
	std::string plot_dir = package_path + "/plots";
	if (!plot_dir.empty())
	{
		if (::mkdir(plot_dir.c_str(), 0755) != 0 && errno != EEXIST)
		{
			ROS_WARN_STREAM("[MPCctrl] Failed to create plot directory: " << plot_dir);
		}
	}
	std::string csv_path = plot_dir + "/analytic_xy.csv";
	std::ofstream csv(csv_path.c_str());
	if (!csv.is_open())
	{
		ROS_WARN_STREAM("[MPCctrl] Unable to write plot data to " << csv_path);
	}else{
		csv << "quad_ref_x,quad_ref_y,quad_actual_x,quad_actual_y,payload_ref_x,payload_ref_y,payload_actual_x,payload_actual_y\n";
		size_t max_len = std::max(reference_history_.size(), actual_history_quad_.size());
		max_len = std::max(max_len, reference_payload_history_.size());
		max_len = std::max(max_len, actual_history_payload_.size());
		for (size_t i = 0; i < max_len; ++i)
		{
			auto fetch = [](const std::vector<geometry_msgs::PoseStamped> &hist, size_t idx) -> std::pair<bool, const geometry_msgs::PoseStamped *>
			{
				if (idx < hist.size())
				{
					return {true, &hist[idx]};
				}
				return {false, nullptr};
			};
			auto quad_ref = fetch(reference_history_, i);
			auto quad_act = fetch(actual_history_quad_, i);
			auto payload_ref = fetch(reference_payload_history_, i);
			auto payload_act = fetch(actual_history_payload_, i);
			auto emit = [&](const std::pair<bool, const geometry_msgs::PoseStamped *> &item, char axis) {
				if (!item.first)
				{
					csv << "nan";
				}
				else
				{
					const auto &pose = *item.second;
					switch (axis)
					{
					case 'x': csv << pose.pose.position.x; break;
					case 'y': csv << pose.pose.position.y; break;
					default: csv << "nan"; break;
					}
				}
			};
			emit(quad_ref, 'x'); csv << ',';
			emit(quad_ref, 'y'); csv << ',';
			emit(quad_act, 'x'); csv << ',';
			emit(quad_act, 'y'); csv << ',';
			emit(payload_ref, 'x'); csv << ',';
			emit(payload_ref, 'y'); csv << ',';
			emit(payload_act, 'x'); csv << ',';
			emit(payload_act, 'y'); csv << '\n';
		}
		csv.close();
	}
	std::string script = package_path + "/scripts/plot_xy.py";
	std::string svg_path = plot_dir + "/analytic_xy.svg";
	std::ostringstream cmd;
	cmd << "python3 " << script << ' ' << csv_path << ' ' << quad_rmse << ' ' << payload_rmse << ' ' << svg_path;
	int ret = std::system(cmd.str().c_str());
	if (ret != 0)
	{
		ROS_WARN_STREAM("[MPCctrl] Plot script returned code " << ret);
	}
	else
	{
		ROS_INFO_STREAM("[MPCctrl] Analytical XY plot saved to " << svg_path);
	}
	std_srvs::Empty srv;
	ros::ServiceClient quit_client = nh_.serviceClient<std_srvs::Empty>("/rviz/force_quit");
	bool rviz_closed = false;
	if (quit_client.waitForExistence(ros::Duration(0.5)))
	{
		if (quit_client.call(srv))
		{
			rviz_closed = true;
		}
	}
	if (!rviz_closed)
	{
		int kill_ret = std::system("rosnode kill /rviz >/dev/null 2>&1");
		if (kill_ret != 0)
		{
			ROS_WARN_STREAM("[MPCctrl] Unable to close RViz automatically. Please close it manually.");
		}
	}
}

} // namespace PayloadMPC
