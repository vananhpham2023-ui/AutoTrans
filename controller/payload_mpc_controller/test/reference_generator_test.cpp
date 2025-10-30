#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "payload_mpc_controller/reference_generator.h"

namespace PayloadMPC
{
namespace
{

constexpr double kTol = 1e-9;

template <typename DerivedA, typename DerivedB>
void ExpectMatrixNear(const Eigen::MatrixBase<DerivedA> &actual,
                      const Eigen::MatrixBase<DerivedB> &expected,
                      double tol)
{
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  for (int r = 0; r < actual.rows(); ++r)
  {
    for (int c = 0; c < actual.cols(); ++c)
    {
      EXPECT_NEAR(actual(r, c), expected(r, c), tol)
          << "Mismatch at (" << r << ", " << c << ")";
    }
  }
}

TEST(CircleTrajectoryTest, SampleMatchesClosedForm)
{
  const double radius = 5.0;
  const double omega = 0.5;
  const Eigen::Vector2d center(1.0, -2.0);
  const double altitude = 3.0;
  CircleTrajectory traj(radius, omega, center, altitude);

  const double t = 4.2; // seconds
  const double theta = omega * t;
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  CircleTrajectory::Sample expected = CircleTrajectory::Sample::Zero();
  expected(0, 0) = center.x() + radius * c;
  expected(1, 0) = center.y() + radius * s;
  expected(2, 0) = altitude;

  expected(0, 1) = -radius * omega * s;
  expected(1, 1) = radius * omega * c;

  const double omega_sq = omega * omega;
  const double omega_cu = omega_sq * omega;
  const double omega_qu = omega_sq * omega_sq;
  const double omega_qui = omega_qu * omega;

  expected(0, 2) = -radius * omega_sq * c;
  expected(1, 2) = -radius * omega_sq * s;

  expected(0, 3) = radius * omega_cu * s;
  expected(1, 3) = -radius * omega_cu * c;

  expected(0, 4) = radius * omega_qu * c;
  expected(1, 4) = radius * omega_qu * s;

  expected(0, 5) = -radius * omega_qui * s;
  expected(1, 5) = radius * omega_qui * c;

  ExpectMatrixNear(traj.sample(t), expected, kTol);
}

TEST(FigureEightTrajectoryTest, SampleMatchesClosedForm)
{
  const double radius = 2.5;
  const double omega = 0.8;
  const Eigen::Vector2d center(-0.6, 0.9);
  const double altitude = 1.7;
  FigureEightTrajectory traj(radius, omega, center, altitude);

  const double t = 1.3;
  const double theta = omega * t;
  const double s = std::sin(theta);
  const double c = std::cos(theta);
  const double sin2 = std::sin(2.0 * theta);
  const double cos2 = std::cos(2.0 * theta);

  FigureEightTrajectory::Sample expected = FigureEightTrajectory::Sample::Zero();
  expected(0, 0) = center.x() + radius * s;
  expected(1, 0) = center.y() + 0.5 * radius * sin2;
  expected(2, 0) = altitude;

  expected(0, 1) = radius * omega * c;
  expected(1, 1) = radius * omega * cos2;

  const double omega_sq = omega * omega;
  const double omega_cu = omega_sq * omega;
  const double omega_qu = omega_sq * omega_sq;
  const double omega_qui = omega_qu * omega;

  expected(0, 2) = -radius * omega_sq * s;
  expected(1, 2) = -2.0 * radius * omega_sq * sin2;

  expected(0, 3) = -radius * omega_cu * c;
  expected(1, 3) = -4.0 * radius * omega_cu * cos2;

  expected(0, 4) = radius * omega_qu * s;
  expected(1, 4) = 8.0 * radius * omega_qu * sin2;

  expected(0, 5) = radius * omega_qui * c;
  expected(1, 5) = 16.0 * radius * omega_qui * cos2;

  ExpectMatrixNear(traj.sample(t), expected, kTol);
}

TEST(HelixTrajectoryTest, DurationAndPostCompletionBehavior)
{
  const double radius = 1.2;
  const double omega = 0.7;
  const Eigen::Vector2d center(0.0, 0.0);
  const double base_altitude = 0.5;
  const double vertical_rate = 0.3;
  const double revolutions = 2.5;

  HelixTrajectory traj(radius,
                       omega,
                       center,
                       base_altitude,
                       vertical_rate,
                       revolutions);

  const double expected_duration = (2.0 * M_PI * revolutions) / omega;
  EXPECT_NEAR(traj.getDuration(), expected_duration, 1e-12);

  // Sample mid-flight.
  const double t_mid = expected_duration * 0.25;
  const double theta = omega * t_mid;
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  HelixTrajectory::Sample expected_mid = HelixTrajectory::Sample::Zero();
  const double omega_sq = omega * omega;
  const double omega_cu = omega_sq * omega;
  const double omega_qu = omega_sq * omega_sq;
  const double omega_qui = omega_qu * omega;

  expected_mid(0, 0) = radius * c;
  expected_mid(1, 0) = radius * s;
  expected_mid(2, 0) = base_altitude + vertical_rate * t_mid;

  expected_mid(0, 1) = -radius * omega * s;
  expected_mid(1, 1) = radius * omega * c;
  expected_mid(2, 1) = vertical_rate;

  expected_mid(0, 2) = -radius * omega_sq * c;
  expected_mid(1, 2) = -radius * omega_sq * s;

  expected_mid(0, 3) = radius * omega_cu * s;
  expected_mid(1, 3) = -radius * omega_cu * c;

  expected_mid(0, 4) = radius * omega_qu * c;
  expected_mid(1, 4) = radius * omega_qu * s;

  expected_mid(0, 5) = -radius * omega_qui * s;
  expected_mid(1, 5) = radius * omega_qui * c;

  ExpectMatrixNear(traj.sample(t_mid), expected_mid, kTol);

  // Compare samples at terminal time and beyond terminal time; values should freeze.
  const auto sample_terminal = traj.sample(expected_duration);
  const double t_after = expected_duration + 3.0;
  const auto sample_after = traj.sample(t_after);
  ExpectMatrixNear(sample_after, sample_terminal, kTol);

  // Reset shifts the start time; querying before start clamps to zero offset.
  traj.reset(5.0);
  const auto sample_pre_start = traj.sample(4.7);
  const auto sample_at_start = traj.sample(5.0);
  ExpectMatrixNear(sample_pre_start, sample_at_start, kTol);
}

} // namespace
} // namespace PayloadMPC
