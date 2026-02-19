// Copyright 2025 Rusty Racer Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file control_node.cpp
 * @brief Rusty Racer control node implementation
 * @author zx
 * @date 2025-12
 */

#include "rusty_racer_control/control_node.h"

#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

// TF2 includes
#include <tf2/LinearMath/Matrix3x3.h>  // NOLINT(build/include_order)
#include <tf2/LinearMath/Quaternion.h>  // NOLINT(build/include_order)
#include <tf2/utils.h>  // NOLINT(build/include_order)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rusty_racer_control/common.h"
#include "rusty_racer_control/laengsfuehrung_controller.h"
#include "rusty_racer_control/lateral_controller.h"
#include "rusty_racer_control/motor_mapping.h"
#include "rusty_racer_interfaces/msg/lane_deviation.hpp"
#include "rusty_racer_interfaces/msg/motor_command.hpp"
#include "rusty_racer_interfaces/msg/traffic_sign.hpp"

// =============================================================================
//  Constructor
// =============================================================================

ControlNode::ControlNode()
: Node("control_node")
{
  // -- Vehicle parameters ------------------------------------------------
  const double v_init = 0.01;
  const double l = 0.257;        // Wheelbase [m]
  const double l_h = 0.0;        // Sensor offset [m]

  // Lateral controller gains (gentle steering for low speed)
  const double lat_kp = 0.8;
  const double lat_kd = 0.2;

  lateral_controller_ =
    std::make_unique<LateralController>(v_init, l, l_h, lat_kp, lat_kd);

  // -- Longitudinal PI parameters ----------------------------------------
  pi_params_.Kp = 1.5;
  pi_params_.Ki = 0.015;
  pi_params_.v_min = 0.0;
  pi_params_.v_max = 2.0;

  pi_state_ = init_pi();

  // -- Traffic FSM parameters --------------------------------------------
  traffic_enabled_ = this->declare_parameter("traffic_enabled", true);
  traffic_params_.conf_th = 0.0f;
  traffic_params_.v_default = 1.5f;
  traffic_params_.v_speed30 = 1.2f;
  traffic_params_.v_highway = 1.8f;
  traffic_params_.v_yield = 0.5f;
  traffic_params_.d_stop_trigger = 1.0f;
  traffic_params_.d_yield_trigger = 1.0f;
  traffic_params_.d_release = 3.0f;
  traffic_params_.stop_hold_ms = 3000;
  traffic_params_.on_count = 3;
  traffic_params_.off_count = 3;
  traffic_params_.start_off_count = 6;
  traffic_params_.start_stop_lock_dist_m = 0.30f;
  traffic_params_.same_sign_block_dist_m = 1.0f;

  traffic_fsm_.reset();
  last_traffic_update_ = this->now();

  // -- Target values -----------------------------------------------------
  v_ref_ = 1.6;        // Cruise speed [m/s]
  y_target_ = -0.06;   // Lateral offset [m] (negative = left bias)

  // -- Initial state -----------------------------------------------------
  current_v_ = 0.0;
  current_psi_k_ = 0.0;
  last_update_time_ = this->now();

  // -- Subscribers -------------------------------------------------------
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", 10,
    std::bind(&ControlNode::odomCallback, this, std::placeholders::_1));

  lane_sub_ = this->create_subscription<rusty_racer_interfaces::msg::LaneDeviation>(
    "/lane_deviation", 10,
    std::bind(&ControlNode::laneCallback, this, std::placeholders::_1));

  traffic_sign_sub_ = this->create_subscription<rusty_racer_interfaces::msg::TrafficSign>(
    "/traffic_sign", 10,
    std::bind(&ControlNode::trafficSignCallback, this, std::placeholders::_1));

  // -- Publishers --------------------------------------------------------
  motor_cmd_pub_ = this->create_publisher<rusty_racer_interfaces::msg::MotorCommand>(
    "/motor_command", 10);

  // -- Startup log -------------------------------------------------------
  RCLCPP_INFO(this->get_logger(), "=================================");
  RCLCPP_INFO(this->get_logger(), "Control Node Started");
  RCLCPP_INFO(this->get_logger(), "Lateral:      PD (no curvature feedforward)");
  RCLCPP_INFO(this->get_logger(), "Longitudinal: PI + Motor Mapping");
  RCLCPP_INFO(
    this->get_logger(), "Traffic FSM:  %s",
    traffic_enabled_ ? "ENABLED" : "DISABLED");
  RCLCPP_INFO(this->get_logger(), "=================================");
  RCLCPP_INFO(
    this->get_logger(), "Lateral  Kp=%.2f  Kd=%.2f  y_target=%.3fm",
    lat_kp, lat_kd, y_target_);
  RCLCPP_INFO(
    this->get_logger(), "PI  Kp=%.2f  Ki=%.4f  v_max=%.2fm/s",
    pi_params_.Kp, pi_params_.Ki, pi_params_.v_max);
  RCLCPP_INFO(this->get_logger(), "v_ref=%.2f m/s", v_ref_);
  RCLCPP_INFO(this->get_logger(), "=================================");
}

// =============================================================================
//  Odometry callback
// =============================================================================

void ControlNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_v_ = msg->twist.twist.linear.x;

  // Explicit quaternion-to-yaw conversion (avoids tf2::getYaw linker issue)
  tf2::Quaternion q;
  tf2::fromMsg(msg->pose.pose.orientation, q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  current_psi_k_ = yaw;

  lateral_controller_->updateVelocity(current_v_);
  last_update_time_ = this->now();
}

// =============================================================================
//  Traffic sign callback (async, updates cached CamFrame)
// =============================================================================

void ControlNode::trafficSignCallback(
  const rusty_racer_interfaces::msg::TrafficSign::SharedPtr msg)
{
  // toCamFrame is a template in traffic_fsm2.h; handles int32 sign_id via
  // signIdToType(int) overload. Returns empty CamFrame for unknown / invalid.
  latest_cam_frame_ = toCamFrame(*msg);
  last_traffic_update_ = this->now();
}

// =============================================================================
//  Lane deviation callback (main control loop, ~50 Hz)
// =============================================================================

void ControlNode::laneCallback(
  const rusty_racer_interfaces::msg::LaneDeviation::SharedPtr msg)
{
  const double y = msg->lateral_error;
  const double phi_k = msg->heading_error;

  // -- Time step ---------------------------------------------------------
  auto current_time = this->now();
  double dt = (current_time - last_update_time_).seconds();
  if (dt <= 0.0 || dt > 0.1) {
    dt = 0.02;  // Default 50 Hz
  }

  // -- Determine velocity reference --------------------------------------
  double v_target = v_ref_;         // Default: fixed cruise speed
  bool emergency_stop = false;

  if (traffic_enabled_) {
    // Check for sign-input timeout: if the vision node hasn't published
    // within kTrafficSignTimeout seconds, fall back to cruise speed.
    double sign_age = (current_time - last_traffic_update_).seconds();

    if (sign_age <= kTrafficSignTimeout) {
      // FSM step uses lane callback interval as dt
      uint32_t dt_ms = static_cast<uint32_t>(dt * 1000.0);
      if (dt_ms == 0) {
        dt_ms = 20;
      }

      // TrafficFSM::step(CamFrame, dt_ms, TrafficParams) -> DecisionOut
      DecisionOut decision = traffic_fsm_.step(
        latest_cam_frame_, dt_ms, traffic_params_);

      v_target = decision.v_ref_mps;
      emergency_stop = decision.must_stop;
    } else {
      // Timeout: vision node may have crashed. Use cruise speed to keep moving.
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "TrafficSign timeout (%.1fs). Falling back to cruise v_ref=%.2f m/s",
        sign_age, v_ref_);
      v_target = v_ref_;
    }
  }

  // Emergency stop overrides everything
  if (emergency_stop) {
    v_target = 0.0;
  }

  // -- Longitudinal control: PI + motor mapping --------------------------
  double v_cmd = pi_step(pi_params_, pi_state_, v_target, current_v_, dt);
  double motor_level = speed_to_motor_level(v_cmd, pi_params_.v_max);

  // -- Lateral control: 3-param PD (no curvature feedforward) ------------
  double delta = lateral_controller_->compute(y, y_target_, phi_k);

  // -- Publish motor command ---------------------------------------------
  auto cmd = rusty_racer_interfaces::msg::MotorCommand();
  cmd.header = msg->header;
  cmd.motor_level = motor_level;
  // Negate delta: coordinate system correction (trajectory already inverted)
  cmd.steering_angle = -delta;
  motor_cmd_pub_->publish(cmd);
}

// =============================================================================
//  Entry point
// =============================================================================

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlNode>());
  rclcpp::shutdown();
  return 0;
}
