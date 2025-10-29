// ================================================================
// 文件：mpc_controller_node.cpp
// 职责：ROS节点主循环，处理消息订阅和发布
// 主要功能：
//   - 传感器数据订阅(IMU/Odometry/RC等)
//   - 控制指令发布(mavros_msgs/AttitudeTarget)
//   - 多线程定时器调度(控制周期/状态观测周期)
//   - 系统启动自检与安全校验
//   -调用`execMPC`进行求解，并发布控制指令。
// ================================================================

#include "mpc_wrapper.h"
#include "mpc_fsm.h"
#include "mpc_controller.h"
#include "reference_generator.h"
#include <ros/ros.h>
#include <memory>

std::unique_ptr<PayloadMPC::MPCFSM> fsm_ptr;
void MPC_controller_main(const ros::TimerEvent &)
{
    fsm_ptr->process();
}

void system_state_update_main(const ros::TimerEvent &)
{
    fsm_ptr->addNewForceObseverState();
}

/**
 * @brief MPC控制器主节点入口函数
 * 
 * @details 该函数是ROS节点的入口点，负责:
 *          - 初始化ROS节点和参数
 *          - 设置所有必要的订阅者(传感器数据、控制指令等)
 *          - 配置状态机(FSM)和控制器
 *          - 创建定时器进行周期性控制计算
 *          - 处理系统启动和安全检查
 * 
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出状态码(0表示正常退出)
 * 
 * @note 主要订阅话题包括:
 *       - /mavros/state (无人机状态)
 *       - /mavros/extended_state (扩展状态)
 *       - odom (里程计信息)
 *       - 各种传感器数据(IMU、电池、ESC等)
 *       - 控制指令和触发信号
 * 
 * @note 主要发布话题包括:
 *       - /mavros/setpoint_raw/attitude (姿态控制指令)
 *       - 各种触发信号和调试信息
 * 
 * @note 在仿真模式下会跳过RC检查，实际飞行时需等待RC信号
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "MPCctrl");
    ros::NodeHandle nh("~");

    PayloadMPC::MpcParams param;
    param.config_from_ros_handle(nh);

    PayloadMPC::MpcController controller(param);
    fsm_ptr.reset(new PayloadMPC::MPCFSM(nh, param, controller));

    // 创建一个对 `fsm_ptr` 所指向的 `PayloadMPC::MPCFSM` 对象的引用 `fsm`。
    // 后续代码可以通过这个引用 `fsm` 方便地访问和操作 `PayloadMPC::MPCFSM` 对象的成员，
    // 避免每次都通过解引用 `fsm_ptr` 来操作对象，提高代码的可读性和简洁性。
    PayloadMPC::MPCFSM &fsm = *fsm_ptr;

    const PayloadMPC::MpcParams::ReferenceConfig &ref_cfg = param.reference_config();
    const bool use_planner = ref_cfg.use_planner;
    if (!use_planner)
    {
        std::unique_ptr<PayloadMPC::AnalyticTrajectory> analytic_traj;
        if (ref_cfg.mode == "circle")
        {
            analytic_traj = std::make_unique<PayloadMPC::CircleTrajectory>(ref_cfg.circle.radius,
                                                                           ref_cfg.circle.angular_velocity,
                                                                           ref_cfg.circle.center,
                                                                           ref_cfg.circle.altitude);
        }
        else if (ref_cfg.mode == "figure_eight")
        {
            analytic_traj = std::make_unique<PayloadMPC::FigureEightTrajectory>(ref_cfg.figure_eight.radius,
                                                                                ref_cfg.figure_eight.angular_velocity,
                                                                                ref_cfg.figure_eight.center,
                                                                                ref_cfg.figure_eight.altitude);
        }
        else if (ref_cfg.mode == "helix")
        {
            analytic_traj = std::make_unique<PayloadMPC::HelixTrajectory>(ref_cfg.helix.radius,
                                                                          ref_cfg.helix.angular_velocity,
                                                                          ref_cfg.helix.center,
                                                                          ref_cfg.helix.base_altitude,
                                                                          ref_cfg.helix.vertical_rate,
                                                                          ref_cfg.helix.revolutions);
        }
        else
        {
            ROS_ERROR_STREAM("[MPCctrl] Unknown analytic trajectory mode: " << ref_cfg.mode);
        }

        if (analytic_traj)
        {
            ROS_INFO_STREAM("[MPCctrl] Using analytic trajectory mode: " << ref_cfg.mode);
            fsm.setAnalyticTrajectory(std::move(analytic_traj));
        }
        else
        {
            ROS_WARN("[MPCctrl] Analytic trajectory not configured. Falling back to hover.");
        }
    }
    else
    {
        ROS_INFO("[MPCctrl] Planner mode enabled. Waiting for polynomial trajectories.");
    }
    
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state",
                                                                 10,
                                                                 boost::bind(&State_Data_t::feed, &fsm.state_data, _1));

    ros::Subscriber extended_state_sub = nh.subscribe<mavros_msgs::ExtendedState>("/mavros/extended_state",
                                                                                  10,
                                                                                  boost::bind(&ExtendedState_Data_t::feed, &fsm.extended_state_data, _1));

    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>("odom",
                                         100,
                                         boost::bind(&Odom_Data_t::feed, &fsm.odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());

    ros::Subscriber start_trig_sub;
    ros::Subscriber cmd_trig_sub;
    ros::Subscriber mpc_traj_sub;
    ros::Subscriber cmd_sub;
    if (use_planner)
    {
        start_trig_sub =
            nh.subscribe<geometry_msgs::PoseStamped>("start_trigger",
                                                     10,
                                                     boost::bind(&Start_Trigger_Data_t::feed, &fsm.start_trigger_data, _1));

        cmd_trig_sub =
            nh.subscribe<geometry_msgs::PoseStamped>("cmd_trigger",
                                                     10,
                                                     boost::bind(&Cmd_Trigger_Data_t::feed, &fsm.cmd_trigger_data, _1));

        mpc_traj_sub =
            nh.subscribe<quadrotor_msgs::PolynomialTraj>("traj",
                                                         100,
                                                         boost::bind(&Trajectory_Data_t::feed, &fsm.trajectory_data, _1),
                                                         ros::VoidConstPtr(),
                                                         ros::TransportHints().tcpNoDelay());

        cmd_sub =
            nh.subscribe<quadrotor_msgs::PositionCommand>("cmd",
                                                          100,
                                                          boost::bind(&Command_Data_t::feed, &fsm.cmd_data, _1),
                                                          ros::VoidConstPtr(),
                                                          ros::TransportHints().tcpNoDelay());
    }
    else
    {
        ROS_INFO("[MPCctrl] Planner topic subscriptions are disabled (use_planner=false).");
    }

    ros::Subscriber imu_sub =
        nh.subscribe<sensor_msgs::Imu>("drone_imu/data",
                                       100,
                                       boost::bind(&Imu_Data_t::feed, &fsm.imu_data, _1),
                                       ros::VoidConstPtr(),
                                       ros::TransportHints().tcpNoDelay());
    fsm.imu_data.set_filter_params(param.filter_param_.sample_freq_quad_acc, param.filter_param_.cutoff_freq_quad_acc,
                                   param.filter_param_.sample_freq_quad_omg, param.filter_param_.cutoff_freq_quad_omg);
    ros::Subscriber cable_info_data_sub =
        nh.subscribe<sensor_msgs::Imu>("cable_info/data",
                                       100,
                                       boost::bind(&Imu_Data_t::feed, &fsm.cable_info_data, _1),
                                       ros::VoidConstPtr(),
                                       ros::TransportHints().tcpNoDelay());
    fsm.cable_info_data.set_filter_params(param.filter_param_.sample_freq_dcable, param.filter_param_.cutoff_freq_dcable,
                                          param.filter_param_.sample_freq_cable, param.filter_param_.cutoff_freq_cable);
    ros::Subscriber payload_imu_data_sub =
        nh.subscribe<sensor_msgs::Imu>("load_imu/data",
                                       100,
                                       boost::bind(&Imu_Data_t::feed, &fsm.payload_imu_data, _1),
                                       ros::VoidConstPtr(),
                                       ros::TransportHints().tcpNoDelay());
    fsm.payload_imu_data.set_filter_params(param.filter_param_.sample_freq_load_acc, param.filter_param_.cutoff_freq_load_acc,
                                           param.filter_param_.sample_freq_load_omg, param.filter_param_.cutoff_freq_load_omg);
    ros::Subscriber payload_odom_sub =
        nh.subscribe<nav_msgs::Odometry>("payload_odom",
                                         100,
                                         boost::bind(&Odom_Data_t::feed, &fsm.payload_odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());

    ros::Subscriber rc_sub;
    {
        rc_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in",
                                                 10,
                                                 boost::bind(&RC_Data_t::feed, &fsm.rc_data, _1));
    }

    ros::Subscriber bat_sub =
        nh.subscribe<sensor_msgs::BatteryState>("/mavros/battery",
                                                100,
                                                boost::bind(&Battery_Data_t::feed, &fsm.bat_data, _1),
                                                ros::VoidConstPtr(),
                                                ros::TransportHints().tcpNoDelay());
    ros::Subscriber esc_sub =
        nh.subscribe<mavros_msgs::ESCTelemetry>("/mavros/esc_telemetry",
                                                100,
                                                boost::bind(&Rpm_Data_t::feed, &fsm.rpm_data, _1),
                                                ros::VoidConstPtr(),
                                                ros::TransportHints().tcpNoDelay());
    fsm.rpm_data.set_filter_params(param.filter_param_.sample_freq_rpm, param.filter_param_.cutoff_freq_rpm);

    if (use_planner)
    {
        fsm.planning_stop_pub_ = nh.advertise<std_msgs::Empty>("/planning_stop_trigger", 10);
        fsm.planning_restart_pub_ = nh.advertise<std_msgs::Empty>("/planning_restart_trigger", 10);
        fsm.traj_start_trigger_pub = nh.advertise<geometry_msgs::PoseStamped>("/traj_start_trigger", 10);
    }
    else
    {
        fsm.planning_stop_pub_ = ros::Publisher();
        fsm.planning_restart_pub_ = ros::Publisher();
        fsm.traj_start_trigger_pub = ros::Publisher();
    }
    fsm.ctrl_FCU_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
    fsm.des_yaw_pub = nh.advertise<nav_msgs::Odometry>("/des_yaw_pub", 10);

    // fsm.debug_pub = nh.advertise<quadrotor_msgs::Px4ctrlDebug>("/debugPx4ctrl", 10); // debug

    fsm.set_FCU_mode_srv = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    fsm.arming_client_srv = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    fsm.reboot_FCU_srv = nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");

    ros::Duration(0.5).sleep();
    if (!param.use_simulation_)
    {
        ROS_INFO("[MPCCTRL] Waiting for RC");
        while (ros::ok())
        {
            ros::spinOnce();
            if (fsm.rc_is_received(ros::Time::now()))
            {
                ROS_INFO("[MPCCTRL] RC received.");
                break;
            }
            ros::Duration(0.1).sleep();
        }
        int trials = 0;
        while (ros::ok() && !fsm.state_data.current_state.connected)
        {
            ros::spinOnce();
            ros::Duration(1.0).sleep();
            if (trials++ > 5)
                ROS_ERROR("Unable to connnect to PX4!!!");
        }
    }
    else
    {
        ROS_WARN("[MPCCTRL] Remote controller disabled, be careful!");
    }

    // 通过ROS定时器实现了力观测器的周期性状态更新，
    //- 作为力估计模块的数据采集前端
    //- 保证力估计器能获取连续、稳定的状态序列
    //- 与控制周期解耦，可独立配置采样频率
    ros::Timer force_state_timer = nh.createTimer(ros::Duration(1.0 / param.force_estimator_param_.force_observer_freq), system_state_update_main);
    
    
    // Create a ROS timer for controller
    // 创建一个ROS定时器 `MPC_controller_main_timer`，该定时器会按照指定的周期执行 `MPC_controller_main` 函数。
    // 定时器的周期由 `param.ctrl_freq_max_` 决定，具体计算方式是将 1 除以 `param.ctrl_freq_max_` 得到的时间间隔，单位为秒。
    // 这样 `MPC_controller_main` 函数就会以 `param.ctrl_freq_max_` 的频率被周期性调用，用于执行MPC控制器的主要逻辑。
    ros::Timer MPC_controller_main_timer = nh.createTimer(ros::Duration(1.0 / param.ctrl_freq_max_), MPC_controller_main);


    // 我们并不依赖反馈作为触发条件，因为通过我们的测试，（使用反馈）并没有显著的性能差异。

    ros::spin();

    return 0;
}
