#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht_laser_ = H_.transpose();
  MatrixXd S = H_ * P_ * Ht_laser_ + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt_laser_ = P_ * Ht_laser_;
  MatrixXd K = PHt_laser_ * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  // check division by zero
  float eps = 0.000001F;
  if (fabs(px) < eps && fabs(py) < eps) {
    px = eps;
    py = eps;
  }
  else if (fabs(px) < eps) {
    px = eps;
  }

  // Recalculate x object state to rho, theta, rho_dot coordinates
  float rho     = sqrtf(powf(px, 2) + powf(py, 2));
  float theta   = atan2f(py, px);
  float rho_dot = (px * vx + py * vy) / rho;

  VectorXd h = VectorXd(3);   // h(x_)
  h << rho, theta, rho_dot;

  VectorXd y = z - h;

  // normalizing the angle using - https://stackoverflow.com/questions/24234609/standard-way-to-normalize-an-angle-to-%CF%80-radians-in-java
  y[1] -= (2 * M_PI) * floor((y[1] + M_PI) / (2 * M_PI));
  //y[1] = atan2(sin(theta), cos(theta));
  MatrixXd Hjt_ = H_.transpose();
  MatrixXd S = H_ * P_ * Hjt_ + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHjt_ = P_ * Hjt_;
  MatrixXd K = PHjt_ * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
