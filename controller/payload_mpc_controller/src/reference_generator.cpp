#include "payload_mpc_controller/reference_generator.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace PayloadMPC
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTwoPi = 6.28318530717958647692;

double clamp_non_negative(double value)
{
	return value < 0.0 ? 0.0 : value;
}
} // namespace

CircleTrajectory::CircleTrajectory(double radius, double angular_velocity, Eigen::Vector2d center, double altitude)
	: radius_(radius),
	  omega_(angular_velocity),
	  omega_sq_(omega_ * omega_),
	  omega_cu_(omega_sq_ * omega_),
	  omega_qu_(omega_sq_ * omega_sq_),
	  omega_qui_(omega_qu_ * omega_),
	  center_(std::move(center)),
	  altitude_(altitude)
{
}

CircleTrajectory::Sample CircleTrajectory::sample(double t) const
{
	const double tau = clamp_non_negative(t - start_time_);
	const double theta = omega_ * tau;
	const double c = std::cos(theta);
	const double s = std::sin(theta);

	Sample out = Sample::Zero();
	out(0, 0) = center_.x() + radius_ * c;
	out(1, 0) = center_.y() + radius_ * s;
	out(2, 0) = altitude_;

	out(0, 1) = -radius_ * omega_ * s;
	out(1, 1) = radius_ * omega_ * c;

	out(0, 2) = -radius_ * omega_sq_ * c;
	out(1, 2) = -radius_ * omega_sq_ * s;

	out(0, 3) = radius_ * omega_cu_ * s;
	out(1, 3) = -radius_ * omega_cu_ * c;

	out(0, 4) = radius_ * omega_qu_ * c;
	out(1, 4) = radius_ * omega_qu_ * s;

	out(0, 5) = -radius_ * omega_qui_ * s;
	out(1, 5) = radius_ * omega_qui_ * c;

	return out;
}

double CircleTrajectory::getDuration() const
{
	return kInf;
}

void CircleTrajectory::reset(double start_time)
{
	start_time_ = start_time;
}

FigureEightTrajectory::FigureEightTrajectory(double radius, double angular_velocity, Eigen::Vector2d center, double altitude)
	: radius_(radius),
	  omega_(angular_velocity),
	  omega_sq_(omega_ * omega_),
	  omega_cu_(omega_sq_ * omega_),
	  omega_qu_(omega_sq_ * omega_sq_),
	  omega_qui_(omega_qu_ * omega_),
	  center_(std::move(center)),
	  altitude_(altitude)
{
}

FigureEightTrajectory::Sample FigureEightTrajectory::sample(double t) const
{
	const double tau = clamp_non_negative(t - start_time_);
	const double theta = omega_ * tau;
	const double s = std::sin(theta);
	const double c = std::cos(theta);
	const double sin2 = std::sin(2.0 * theta);
	const double cos2 = std::cos(2.0 * theta);

	Sample out = Sample::Zero();
	out(0, 0) = center_.x() + radius_ * s;
	out(1, 0) = center_.y() + 0.5 * radius_ * sin2;
	out(2, 0) = altitude_;

	out(0, 1) = radius_ * omega_ * c;
	out(1, 1) = radius_ * omega_ * cos2;

	out(0, 2) = -radius_ * omega_sq_ * s;
	out(1, 2) = -2.0 * radius_ * omega_sq_ * sin2;

	out(0, 3) = -radius_ * omega_cu_ * c;
	out(1, 3) = -4.0 * radius_ * omega_cu_ * cos2;

	out(0, 4) = radius_ * omega_qu_ * s;
	out(1, 4) = 8.0 * radius_ * omega_qu_ * sin2;

	out(0, 5) = radius_ * omega_qui_ * c;
	out(1, 5) = 16.0 * radius_ * omega_qui_ * cos2;

	return out;
}

double FigureEightTrajectory::getDuration() const
{
	return kInf;
}

void FigureEightTrajectory::reset(double start_time)
{
	start_time_ = start_time;
}

HelixTrajectory::HelixTrajectory(double radius,
								 double angular_velocity,
								 Eigen::Vector2d center,
								 double base_altitude,
								 double vertical_rate,
								 double revolutions)
	: radius_(radius),
	  omega_(angular_velocity),
	  omega_sq_(omega_ * omega_),
	  omega_cu_(omega_sq_ * omega_),
	  omega_qu_(omega_sq_ * omega_sq_),
	  omega_qui_(omega_qu_ * omega_),
	  center_(std::move(center)),
	  base_altitude_(base_altitude),
	  vertical_rate_(vertical_rate),
	  revolutions_(revolutions),
	  duration_((angular_velocity > 0.0 && revolutions > 0.0) ? (kTwoPi * revolutions / angular_velocity) : kInf)
{
}

HelixTrajectory::Sample HelixTrajectory::sample(double t) const
{
	const double raw_tau = t - start_time_;
	const double tau = clamp_non_negative(raw_tau);
	const bool bounded = std::isfinite(duration_);
	const bool after_end = bounded && (tau >= duration_);
	const double eval_tau = (bounded && after_end) ? duration_ : tau;
	const double theta = omega_ * eval_tau;
	const double c = std::cos(theta);
	const double s = std::sin(theta);

	Sample out = Sample::Zero();
	out(0, 0) = center_.x() + radius_ * c;
	out(1, 0) = center_.y() + radius_ * s;
	out(2, 0) = base_altitude_ + vertical_rate_ * eval_tau;

	if (!after_end)
	{
		out(0, 1) = -radius_ * omega_ * s;
		out(1, 1) = radius_ * omega_ * c;
		out(2, 1) = vertical_rate_;

		out(0, 2) = -radius_ * omega_sq_ * c;
		out(1, 2) = -radius_ * omega_sq_ * s;

		out(0, 3) = radius_ * omega_cu_ * s;
		out(1, 3) = -radius_ * omega_cu_ * c;

		out(0, 4) = radius_ * omega_qu_ * c;
		out(1, 4) = radius_ * omega_qu_ * s;

		out(0, 5) = -radius_ * omega_qui_ * s;
		out(1, 5) = radius_ * omega_qui_ * c;
	}

	return out;
}

double HelixTrajectory::getDuration() const
{
	return duration_;
}

void HelixTrajectory::reset(double start_time)
{
	start_time_ = start_time;
}

} // namespace PayloadMPC
