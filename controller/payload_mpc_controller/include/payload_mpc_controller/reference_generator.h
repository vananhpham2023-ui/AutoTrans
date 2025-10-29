#pragma once

#include <Eigen/Dense>
#include <limits>

namespace PayloadMPC
{

	class AnalyticTrajectory
	{
	public:
		using Sample = Eigen::Matrix<double, 3, 6>;

		virtual ~AnalyticTrajectory() = default;
		virtual Sample sample(double t) const = 0;
		virtual double getDuration() const = 0;
		virtual void reset(double start_time) = 0;
	};

	class CircleTrajectory final : public AnalyticTrajectory
	{
	public:
		CircleTrajectory(double radius, double angular_velocity, Eigen::Vector2d center, double altitude);

		Sample sample(double t) const override;
		double getDuration() const override;
		void reset(double start_time) override;

	private:
		double radius_;
		double omega_;
		double omega_sq_;
		double omega_cu_;
		double omega_qu_;
		double omega_qui_;
		Eigen::Vector2d center_;
		double altitude_;
		double start_time_{0.0};
	};

	class FigureEightTrajectory final : public AnalyticTrajectory
	{
	public:
		FigureEightTrajectory(double radius, double angular_velocity, Eigen::Vector2d center, double altitude);

		Sample sample(double t) const override;
		double getDuration() const override;
		void reset(double start_time) override;

	private:
		double radius_;
		double omega_;
		double omega_sq_;
		double omega_cu_;
		double omega_qu_;
		double omega_qui_;
		Eigen::Vector2d center_;
		double altitude_;
		double start_time_{0.0};
	};

	class HelixTrajectory final : public AnalyticTrajectory
	{
	public:
		HelixTrajectory(double radius,
						double angular_velocity,
						Eigen::Vector2d center,
						double base_altitude,
						double vertical_rate,
						double revolutions);

		Sample sample(double t) const override;
		double getDuration() const override;
		void reset(double start_time) override;

	private:
		double radius_;
		double omega_;
		double omega_sq_;
		double omega_cu_;
		double omega_qu_;
		double omega_qui_;
		Eigen::Vector2d center_;
		double base_altitude_;
		double vertical_rate_;
		double revolutions_;
		double duration_;
		double start_time_{0.0};
	};

} // namespace PayloadMPC
