#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

#from px4_mpc.osqp_solver.cpg_solver import cpg_solve
from px4_mpc.osqp_solver.cpg_solver import cpg_solve

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleAngularVelocity
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import VehicleTorqueSetpoint
from px4_msgs.msg import VehicleThrustSetpoint
from px4_msgs.msg import ManualControlSetpoint

from mpc_msgs.srv import SetPose

from cvxpy import *
import numpy as np
import jax.numpy as jnp
import cvxpy as cp
from jax import jacfwd, jacrev
import time
from jax import jit
import jax
jax.config.update('jax_platform_name', 'cpu')
from cvxpygen import cpg
import json

def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id = frame_id
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg

@jit
def drone_dynamics_eqn(x, u):
    J = jnp.array([0.03, 0.03, 0.06])
    mass = 0.5
    g = jnp.array([0, 0, -9.81])

    angle = x[3:7]
    vel = x[7:10]
    a_rate = x[10:13]

    thrust_acc = jnp.array([[0], [0], [u[0]]]) / mass

    # Derivatives
    d_rate = jnp.array([
            1 / J[0] * (u[1] + (J[1] - J[2]) * a_rate[1] * a_rate[2]),
            1 / J[1] * (-u[2] + (J[2] - J[0]) * a_rate[2] * a_rate[0]),
            1 / J[2] * (u[3] + (J[0] - J[1]) * a_rate[0] * a_rate[1])])
    d_velocity = jnp.ravel(v_dot_q(thrust_acc, angle)) + g
    d_attitude = 1 / 2 * skew_symmetric(a_rate).dot(angle)
    d_position = vel

    x_dot = jnp.concatenate([d_position, d_attitude, d_velocity, d_rate])
    return x_dot

@jit
def skew_symmetric(v):
    return jnp.array(
        [
            [0, -v[0], -v[1], -v[2]],
            [v[0], 0, v[2], -v[1]],
            [v[1], -v[2], 0, v[0]],
            [v[2], v[1], -v[0], 0],
        ]
    )
    
@jit
def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    return rot_mat.dot(v)

@jit
def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    
@jit
def calculate_jacobians(x_op, u_op):
    A = jacfwd(lambda x: drone_dynamics_eqn(x, u_op))(x_op)
    B = jacfwd(lambda u: drone_dynamics_eqn(x_op, u))(u_op)
    return A, B

@jit
def euler_discretize(A, B, dt):
    I = jnp.eye(A.shape[0])  # Identity matrix
    A_d = I + A * dt  # Euler approximation for A_d
    B_d = B * dt      # Euler approximation for B_d
    return A_d, B_d

class SpacecraftMPC(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        # Get mode; rate, wrench, direct_allocation
        self.mode = 'wrench'#self.declare_parameter('mode', 'rate').value

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)

        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile)
        self.angular_vel_sub = self.create_subscription(
            VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile)
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
        self.manual_control_sub = self.create_subscription(
            ManualControlSetpoint,
            '/fmu/out/manual_control_setpoint',
            self.manual_control_callback,
            qos_profile)

        self.set_pose_srv = self.create_service(SetPose, '/set_pose', self.add_set_pos_callback)

        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_rates_setpoint = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.publisher_direct_actuator = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.publisher_thrust_setpoint = self.create_publisher(VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.publisher_torque_setpoint = self.create_publisher(VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.predicted_path_pub = self.create_publisher(Path, '/px4_mpc/predicted_path', 10)
        self.reference_pub = self.create_publisher(Marker, "/px4_mpc/reference", 10)
        self.reference_pub = self.create_publisher(Marker, "/px4_mpc/reference", 10)

        timer_period = 0.0001  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 6.0])
        
        # Constraints
        tmax = 1.0
        umin = jnp.array([0.0, -tmax, -tmax, -tmax])
        umax = jnp.array([10.0, tmax, tmax, tmax])

        # Objective function
        with open("/mpc_config/qr_weights.json", 'r') as file:
            data = json.load(file)
        Q = jnp.diag(jnp.array(data['Q']))
        R = jnp.diag(jnp.array(data['R']))            

        #Q = jnp.diag(jnp.array([200, 200, 200, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
        #R = jnp.eye(4)*1.0

        # Initial and reference states
        x0 = jnp.array([1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        xr = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u0 = jnp.array([9.8*0.5, 0.0, 0.0, 0.0])
        self.u_prev = u0
        ur = jnp.array([0.0, 0.0, 0.0, 0.0])
        N = 10
        self.horizon = N
        dt = 0.05
        A, B = calculate_jacobians(x0, u0)
        Ad, Bd = euler_discretize(A, B, dt)
        self.Ad_param = Parameter(Ad.shape)
        self.Bd_param = Parameter(Bd.shape)
        nx = 13
        nu = 4
        self.u = Variable((nu, N))
        self.x = Variable((nx, N+1))
        self.x_init = Parameter(nx)
        objective = 0
        constraints = [self.x[:,0] == self.x_init]
        for k in range(N):
            objective += quad_form(self.x[:,k] - xr, Q) + quad_form(self.u[:,k] - ur, R)
            constraints += [self.x[:,k+1] == self.Ad_param@self.x[:,k] + self.Bd_param@self.u[:,k]]
            constraints += [umin <= self.u[:,k], self.u[:,k] <= umax]
        objective += quad_form(self.x[:,N] - xr, Q)
        self.prob = Problem(Minimize(objective), constraints)
        # cpg.generate_code(self.prob, code_dir='osqp_solver', solver='OSQP', solver_opts={'eps_abs': 1e-3, 'eps_rel': 1e-3, 'warm_start': True})
        self.prob.register_solve('CPG', cpg_solve)

    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_angular_velocity_callback(self, msg):
        # TODO: handle NED->ENU transformation
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    def vehicle_status_callback(self, msg):
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_rate_setpoint(self, u_pred):
        thrust_rates = u_pred[0, :]
        # Hover thrust = 0.73
        thrust_command = thrust_rates[0:3] * 0.07  # NOTE: Tune in thrust multiplier
        rates_setpoint_msg = VehicleRatesSetpoint()
        rates_setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        rates_setpoint_msg.roll  = float(thrust_rates[3])
        rates_setpoint_msg.pitch = -float(thrust_rates[4])
        rates_setpoint_msg.yaw   = -float(thrust_rates[5])
        rates_setpoint_msg.thrust_body[0] = float(thrust_command[0])
        rates_setpoint_msg.thrust_body[1] = -float(thrust_command[1])
        rates_setpoint_msg.thrust_body[2] = -float(thrust_command[2])
        self.publisher_rates_setpoint.publish(rates_setpoint_msg)
                
    def publish_rate_setpoint_wrench(self, x_pred, u_pred):
        rates = x_pred[10:13, 0]
        thrust_rates = u_pred
        
        # Hover thrust = 0.73
        thrust_command = -thrust_rates[0]#(thrust_rates[0] * 0.07 + 0.0)
        setpoint_msg = VehicleRatesSetpoint()
        setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        setpoint_msg.roll = float(rates[0])
        setpoint_msg.pitch = float(-rates[1])
        setpoint_msg.yaw = float(-rates[2])
        setpoint_msg.thrust_body[0] = 0.0
        setpoint_msg.thrust_body[1] = 0.0
        setpoint_msg.thrust_body[2] = float(thrust_command)
        self.publisher_rates_setpoint.publish(setpoint_msg)

    def publish_wrench_setpoint(self, u_pred):
        thrust_outputs_msg = VehicleThrustSetpoint()
        thrust_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        torque_outputs_msg = VehicleTorqueSetpoint()
        torque_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        thrust_outputs_msg.xyz = [0.0, 0.0, -u_pred[0]]
        torque_outputs_msg.xyz = [u_pred[1], -u_pred[2], -u_pred[3]]

        self.publisher_thrust_setpoint.publish(thrust_outputs_msg)
        self.publisher_torque_setpoint.publish(torque_outputs_msg)

    def publish_direct_actuator_setpoint(self, u_pred):
        actuator_outputs_msg = ActuatorMotors()
        actuator_outputs_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        # NOTE:
        # Output is float[16]
        # u1 needs to be divided between 1 and 2
        # u2 needs to be divided between 3 and 4
        # u3 needs to be divided between 5 and 6
        # u4 needs to be divided between 7 and 8
        # positve component goes for the first, the negative for the second
        thrust = u_pred[0, :] / self.model.max_thrust  # normalizes w.r.t. max thrust
        # print("Thrust rates: ", thrust[0:4])

        thrust_command = np.zeros(12, dtype=np.float32)
        thrust_command[0] = 0.0 if thrust[0] <= 0.0 else thrust[0]
        thrust_command[1] = 0.0 if thrust[0] >= 0.0 else -thrust[0]

        thrust_command[2] = 0.0 if thrust[1] <= 0.0 else thrust[1]
        thrust_command[3] = 0.0 if thrust[1] >= 0.0 else -thrust[1]

        thrust_command[4] = 0.0 if thrust[2] <= 0.0 else thrust[2]
        thrust_command[5] = 0.0 if thrust[2] >= 0.0 else -thrust[2]

        thrust_command[6] = 0.0 if thrust[3] <= 0.0 else thrust[3]
        thrust_command[7] = 0.0 if thrust[3] >= 0.0 else -thrust[3]

        actuator_outputs_msg.control = thrust_command.flatten()
        self.publisher_direct_actuator.publish(actuator_outputs_msg)

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.direct_actuator = False
        #offboard_msg.body_rate = True
        offboard_msg.thrust_and_torque = True
        self.publisher_offboard_mode.publish(offboard_msg)

        error_position = self.vehicle_local_position - self.setpoint_position

        x0 = np.array([error_position[0],
                        error_position[1],
                        error_position[2],
                        self.vehicle_attitude[0],
                        self.vehicle_attitude[1],
                        self.vehicle_attitude[2],
                        self.vehicle_attitude[3],
                        self.vehicle_local_velocity[0],
                        self.vehicle_local_velocity[1],
                        self.vehicle_local_velocity[2],
                        self.vehicle_angular_velocity[0],
                        self.vehicle_angular_velocity[1],
                        self.vehicle_angular_velocity[2]]).reshape(13, 1)
            
        self.x_init.value = x0.reshape(13)
        dt = 0.05
        A, B = calculate_jacobians(jnp.array(x0.reshape(13)), self.u_prev)
        Ad, Bd = euler_discretize(A, B, dt)
        self.Ad_param.value = np.array(Ad)
        self.Bd_param.value = np.array(Bd)
        # self.prob.solve(solver=cp.OSQP, warm_start=True, eps_rel=1e-3, eps_abs=1e-3)
        self.prob.solve(method='CPG')
        u_pred = self.u[:,0].value
        x_pred = self.x.value
        self.u_prev = u_pred

        idx = 0
        predicted_path_msg = Path()
        for i in range(self.horizon):
            idx = idx + 1
            # Publish time history of the vehicle path
            predicted_pose_msg = vector2PoseMsg('map', x_pred[0:3, i] + self.setpoint_position, np.array([1.0, 0.0, 0.0, 0.0]))
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        self.publish_wrench_setpoint(u_pred)
        # if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        #     if self.mode == 'rate':
        #         pass
        #         #self.publish_rate_setpoint(u_pred)
        #     elif self.mode == 'direct_allocation':
        #         self.publish_direct_actuator_setpoint(u_pred)
        #     elif self.mode == 'wrench':
        #         # self.publish_wrench_setpoint(u_pred)
        #         self.publish_rate_setpoint_wrench(x_pred, u_pred)

    def add_set_pos_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z

        return response

    def manual_control_callback(self, msg):
        if abs(msg.pitch) > 0.0:
            self.setpoint_position[0] = self.vehicle_local_position[0] + (msg.pitch * self.scale_x_control * 0.5/0.66)
        if abs(msg.roll) > 0.0:
            self.setpoint_position[1] = self.vehicle_local_position[1] + (msg.roll * self.scale_y_control * 0.5/0.66)
        if abs(msg.throttle) > 0.0:
            self.setpoint_position[2] = self.vehicle_local_position[2] + (msg.throttle * self.scale_z_control * 0.5/0.66)

def main(args=None):
    rclpy.init(args=args)

    spacecraft_mpc = SpacecraftMPC()

    rclpy.spin(spacecraft_mpc)

    spacecraft_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
