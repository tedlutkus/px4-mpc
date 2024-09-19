uint64 timestamp				# time since system start (microseconds)

float32 voltage

float32 thrust_n				# thrust force setpoint in Newtons
float32 throttle				# throttle setpoint in %
float32 yaw_torque_nm			# yaw torque setpoint in Newton-meters
float32 control_torque_nm		# control torque setpoint in Newton-meters
float32 control_direction_deg	# control direction setpoint in degrees

float32[2] motors_rpm			# motor RPM setpoints
float32[2] motors_out			# motor output setpoints
float32[4] flaps_out			# flap output setpoints