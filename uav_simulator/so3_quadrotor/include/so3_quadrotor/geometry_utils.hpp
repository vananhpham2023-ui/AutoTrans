#pragma once
#include <Eigen/Geometry>

/**
 * @file geometry_utils.hpp
 * @brief 提供四旋翼无人机仿真和控制中常用的几何变换工具函数
 * @details 该文件包含了角度转换、旋转矩阵、四元数、欧拉角等几何运算的工具函数，
 * 用于四旋翼无人机的姿态表示、变换和计算
 */  
namespace uav_utils {

/**
 * @brief 将角度从度数转换为弧度
 * @tparam Scalar_t 数据类型（如float, double等）
 * @param x 角度值（度数）
 * @return 转换后的弧度值
 */
template <typename Scalar_t>
Scalar_t toRad(const Scalar_t& x) {
  return x / 180.0 * M_PI;
}

/**
 * @brief 将角度从弧度转换为度数
 * @tparam Scalar_t 数据类型
 * @param x 角度值（弧度）
 * @return 转换后的度数
 */
template <typename Scalar_t>
Scalar_t toDeg(const Scalar_t& x) {
  return x * 180.0 / M_PI;
}

/**
 * @brief 生成绕X轴旋转的旋转矩阵
 * @tparam Scalar_t 数据类型
 * @param t 旋转角度（弧度）
 * @return 3x3的旋转矩阵
 */
template <typename Scalar_t>
Eigen::Matrix<Scalar_t, 3, 3> rotx(Scalar_t t) {
  Eigen::Matrix<Scalar_t, 3, 3> R;
  R(0, 0) = 1.0;
  R(0, 1) = 0.0;
  R(0, 2) = 0.0;
  R(1, 0) = 0.0;
  R(1, 1) = std::cos(t);
  R(1, 2) = -std::sin(t);
  R(2, 0) = 0.0;
  R(2, 1) = std::sin(t);
  R(2, 2) = std::cos(t);

  return R;
}

/**
 * @brief 生成绕Y轴旋转的旋转矩阵
 * @tparam Scalar_t 数据类型
 * @param t 旋转角度（弧度）
 * @return 3x3的旋转矩阵
 */
template <typename Scalar_t>
Eigen::Matrix<Scalar_t, 3, 3> roty(Scalar_t t) {
  Eigen::Matrix<Scalar_t, 3, 3> R;
  R(0, 0) = std::cos(t);
  R(0, 1) = 0.0;
  R(0, 2) = std::sin(t);
  R(1, 0) = 0.0;
  R(1, 1) = 1.0;
  R(1, 2) = 0;
  R(2, 0) = -std::sin(t);
  R(2, 1) = 0.0;
  R(2, 2) = std::cos(t);

  return R;
}

/**
 * @brief 生成绕Z轴旋转的旋转矩阵
 * @tparam Scalar_t 数据类型
 * @param t 旋转角度（弧度）
 * @return 3x3的旋转矩阵
 */
template <typename Scalar_t>
Eigen::Matrix<Scalar_t, 3, 3> rotz(Scalar_t t) {
  Eigen::Matrix<Scalar_t, 3, 3> R;
  R(0, 0) = std::cos(t);
  R(0, 1) = -std::sin(t);
  R(0, 2) = 0.0;
  R(1, 0) = std::sin(t);
  R(1, 1) = std::cos(t);
  R(1, 2) = 0.0;
  R(2, 0) = 0.0;
  R(2, 1) = 0.0;
  R(2, 2) = 1.0;

  return R;
}

/**
 * @brief 将YPR角（偏航角、俯仰角、横滚角）转换为旋转矩阵
 * @tparam Derived Eigen矩阵类型
 * @param ypr 包含YPR角的3x1向量 [偏航角, 俯仰角, 横滚角]（弧度）
 * @return 3x3的旋转矩阵，旋转顺序为Z-Y-X
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr_to_R(const Eigen::DenseBase<Derived>& ypr) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  typename Derived::Scalar c, s;

  // 计算绕Z轴旋转的矩阵（偏航角）
  Eigen::Matrix<typename Derived::Scalar, 3, 3> Rz = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
  typename Derived::Scalar y = ypr(0);
  c = cos(y);
  s = sin(y);
  Rz(0, 0) = c;
  Rz(1, 0) = s;
  Rz(0, 1) = -s;
  Rz(1, 1) = c;
  Rz(2, 2) = 1;

  // 计算绕Y轴旋转的矩阵（俯仰角）
  Eigen::Matrix<typename Derived::Scalar, 3, 3> Ry = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
  typename Derived::Scalar p = ypr(1);
  c = cos(p);
  s = sin(p);
  Ry(0, 0) = c;
  Ry(2, 0) = -s;
  Ry(0, 2) = s;
  Ry(2, 2) = c;
  Ry(1, 1) = 1;

  // 计算绕X轴旋转的矩阵（横滚角）
  Eigen::Matrix<typename Derived::Scalar, 3, 3> Rx = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Zero();
  typename Derived::Scalar r = ypr(2);
  c = cos(r);
  s = sin(r);
  Rx(1, 1) = c;
  Rx(2, 1) = s;
  Rx(1, 2) = -s;
  Rx(2, 2) = c;
  Rx(0, 0) = 1;

  // 按Z-Y-X顺序组合旋转矩阵
  Eigen::Matrix<typename Derived::Scalar, 3, 3> R = Rz * Ry * Rx;
  return R;
}

/**
 * @brief 将旋转矩阵转换为YPR角（偏航角、俯仰角、横滚角）
 * @tparam Derived Eigen矩阵类型
 * @param R 3x3的旋转矩阵
 * @return 包含YPR角的3x1向量 [偏航角, 俯仰角, 横滚角]（弧度）
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> R_to_ypr(const Eigen::DenseBase<Derived>& R) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  // 提取旋转矩阵的列向量（单位向量）
  Eigen::Matrix<typename Derived::Scalar, 3, 1> n = R.col(0); // 北向向量
  Eigen::Matrix<typename Derived::Scalar, 3, 1> o = R.col(1); // 东向向量
  Eigen::Matrix<typename Derived::Scalar, 3, 1> a = R.col(2); // 天向向量

  Eigen::Matrix<typename Derived::Scalar, 3, 1> ypr(3);
  // 计算偏航角 yaw
  typename Derived::Scalar y = atan2(n(1), n(0));
  // 计算俯仰角 pitch
  typename Derived::Scalar p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  // 计算横滚角 roll
  typename Derived::Scalar r = 
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr;
}

/**
 * @brief 将YPR角转换为四元数
 * @tparam Derived Eigen矩阵类型
 * @param ypr 包含YPR角的3x1向量 [偏航角, 俯仰角, 横滚角]（弧度）
 * @return 表示相同旋转的四元数
 */
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> ypr_to_quaternion(const Eigen::DenseBase<Derived>& ypr) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  // 计算各角度半角的正弦和余弦值
  const typename Derived::Scalar cy = cos(ypr(0) / 2.0); // 偏航角半角余弦
  const typename Derived::Scalar sy = sin(ypr(0) / 2.0); // 偏航角半角正弦
  const typename Derived::Scalar cp = cos(ypr(1) / 2.0); // 俯仰角半角余弦
  const typename Derived::Scalar sp = sin(ypr(1) / 2.0); // 俯仰角半角正弦
  const typename Derived::Scalar cr = cos(ypr(2) / 2.0); // 横滚角半角余弦
  const typename Derived::Scalar sr = sin(ypr(2) / 2.0); // 横滚角半角正弦

  Eigen::Quaternion<typename Derived::Scalar> q;

  // 计算四元数各分量
  q.w() = cr * cp * cy + sr * sp * sy;
  q.x() = sr * cp * cy - cr * sp * sy;
  q.y() = cr * sp * cy + sr * cp * sy;
  q.z() = cr * cp * sy - sr * sp * cy;

  // 归一化四元数
  return q.normalized();
}

/**
 * @brief 将四元数转换为YPR角
 * @tparam Scalar_t 数据类型
 * @param q_ 输入四元数
 * @return 包含YPR角的3x1向量 [偏航角, 俯仰角, 横滚角]（弧度）
 */
template <typename Scalar_t>
Eigen::Matrix<Scalar_t, 3, 1> quaternion_to_ypr(const Eigen::Quaternion<Scalar_t>& q_) {
  // 确保四元数被归一化
  Eigen::Quaternion<Scalar_t> q = q_.normalized();

  Eigen::Matrix<Scalar_t, 3, 1> ypr;
  // 计算横滚角 roll
  ypr(2) = atan2(2 * (q.w() * q.x() + q.y() * q.z()), 1 - 2 * (q.x() * q.x() + q.y() * q.y()));
  // 计算俯仰角 pitch
  ypr(1) = asin(2 * (q.w() * q.y() - q.z() * q.x()));
  // 计算偏航角 yaw
  ypr(0) = atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));

  return ypr;
}

/**
 * @brief 从四元数中提取偏航角
 * @tparam Scalar_t 数据类型
 * @param q 输入四元数
 * @return 偏航角（弧度）
 */
template <typename Scalar_t>
Scalar_t get_yaw_from_quaternion(const Eigen::Quaternion<Scalar_t>& q) {
  return quaternion_to_ypr(q)(0);
}

/**
 * @brief 将偏航角转换为四元数（仅绕Z轴旋转）
 * @tparam Scalar_t 数据类型
 * @param yaw 偏航角（弧度）
 * @return 表示绕Z轴旋转的四元数
 */
template <typename Scalar_t>
Eigen::Quaternion<Scalar_t> yaw_to_quaternion(Scalar_t yaw) {
  return Eigen::Quaternion<Scalar_t>(rotz(yaw));
}

/**
 * @brief 将角度归一化到[-π, π]范围内
 * @tparam Scalar_t 数据类型
 * @param a 输入角度（弧度）
 * @return 归一化后的角度（弧度），范围在[-π, π]之间
 */
template <typename Scalar_t>
Scalar_t normalize_angle(Scalar_t a) {
  int cnt = 0;
  while (true) {
    cnt++;

    if (a < -M_PI) {
        a += M_PI * 2.0;
    } else if (a > M_PI) {
        a -= M_PI * 2.0;
    }

    if (-M_PI <= a && a <= M_PI) {
        break;
    };

    assert(cnt < 10 && "[uav_utils/geometry_msgs] INVALID INPUT ANGLE");
  }

  return a;
}

/**
 * @brief 角度加法，确保结果在[-π, π]范围内
 * @tparam Scalar_t 数据类型
 * @param a 第一个角度（弧度）
 * @param b 第二个角度（弧度）
 * @return 角度之和，已归一化到[-π, π]范围
 */
template <typename Scalar_t>
Scalar_t angle_add(Scalar_t a, Scalar_t b) {
  Scalar_t c = a + b;
  c = normalize_angle(c);
  assert(-M_PI <= c && c <= M_PI);
  return c;
}

/**
 * @brief 偏航角加法，与普通角度加法相同
 * @tparam Scalar_t 数据类型
 * @param a 第一个偏航角（弧度）
 * @param b 第二个偏航角（弧度）
 * @return 偏航角之和，已归一化到[-π, π]范围
 */
template <typename Scalar_t>
Scalar_t yaw_add(Scalar_t a, Scalar_t b) {
  return angle_add(a, b);
}

/**
 * @brief 计算三维向量的斜对称矩阵（叉乘矩阵）
 * @tparam Derived Eigen矩阵类型
 * @param v 输入的3x1向量
 * @return 3x3的斜对称矩阵，满足 v × w = M * w
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> get_skew_symmetric(const Eigen::DenseBase<Derived>& v) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  Eigen::Matrix<typename Derived::Scalar, 3, 3> M;
  M.setZero();
  M(0, 1) = -v(2);
  M(0, 2) = v(1);
  M(1, 0) = v(2);
  M(1, 2) = -v(0);
  M(2, 0) = -v(1);
  M(2, 1) = v(0);
  return M;
}

/**
 * @brief 从斜对称矩阵中提取原始向量
 * @tparam Derived Eigen矩阵类型
 * @param M 3x3的斜对称矩阵
 * @return 原始的3x1向量v，满足M是v的斜对称矩阵
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> from_skew_symmetric(const Eigen::DenseBase<Derived>& M) {
  EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
  EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
  EIGEN_STATIC_ASSERT(Derived::ColsAtCompileTime == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);

  Eigen::Matrix<typename Derived::Scalar, 3, 1> v;
  v(0) = M(2, 1);
  v(1) = -M(2, 0);
  v(2) = M(1, 0);
  
  // 验证提取的向量是否正确
  assert(v.isApprox(Eigen::Matrix<typename Derived::Scalar, 3, 1>(-M(1, 2), M(0, 2), -M(0, 1))));
  
  return v;
}


}  // namespace uav_utils
