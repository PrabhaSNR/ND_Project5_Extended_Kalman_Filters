#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
   H_laser_ << 1, 0, 0, 0,
               0, 1, 0, 0; 
			  			  
   ekf_.P_ = MatrixXd(4, 4);
   ekf_.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0, 
              0, 0, 0, 1000;  
			  
   ekf_.F_ = MatrixXd(4, 4);
   ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */
	cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
	// Convert radar from polar to cartesian coordinates
      cout<<"Radar Intialization Started "<< endl;
      float ro = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      //float rho_dot = measurement_pack.raw_measurements_(2);
      float px = ro*cos(theta);
      float py = ro*sin(theta);
      //float vx = rho_dot*cos(theta);
      //float vy = rho_dot*sin(theta);

      // Initialize state
      ekf_.x_ << px,
                 py,
				 0,
				 0;
      // first measurement
      cout << "EKF: " << ekf_.x_ << endl;
      // done initializing, no need to predict or update
      cout<<"Radar Intialization Completed "<< endl;
      previous_timestamp_ = measurement_pack.timestamp_;

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
	  // set the state with the initial location and zero velocity
      cout<<"Laser Intialization in Progress.. "<< endl;
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
                measurement_pack.raw_measurements_[1], 
                0, 
                0; 
      // first measurement
      cout << "EKF: " << ekf_.x_ << endl;
      
      previous_timestamp_ = measurement_pack.timestamp_;     
      cout<<"Laser Intialization done "<< endl;
            
    }
	float eps = 0.000001F;
    if ((fabs(ekf_.x_(0)) < eps) && fabs(ekf_.x_(1)) < eps) {
      ekf_.x_(0) = eps;
	  ekf_.x_(1) = eps;
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
      // compute the time elapsed between the current and previous measurements
      // dt - expressed in seconds
      float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
      previous_timestamp_ = measurement_pack.timestamp_;
      
      float dt_2 = dt * dt;
      float dt_3 = dt_2 * dt;
      float dt_4 = dt_3 * dt;
      
      // Modify the F matrix so that the time is integrated
      ekf_.F_(0, 2) = dt;
      ekf_.F_(1, 3) = dt;
      
      // set the process covariance matrix Q
	  float noise_ax = 9;
      float noise_ay = 9;
      ekf_.Q_ = MatrixXd(4, 4);
      ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
             0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
             dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
             0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  if (dt >= 0.000001) {
  ekf_.Predict();
  }
  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    if ((ekf_.x_(0) != 0) || (ekf_.x_(1) != 0)) {
    cout<<"Radar Update Started" << endl;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_); 
	cout<<"Jacobian"<<ekf_.H_<<endl;
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    cout<<"Radar Update done "<< endl;
    }
  } else {
    // TODO: Laser updates
    cout<<"Laser Update Started.. "<< endl;
	ekf_.H_ = H_laser_; 
	ekf_.R_ = R_laser_;
	ekf_.Update(measurement_pack.raw_measurements_);
    cout<<"Laser Update done "<< endl;

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
