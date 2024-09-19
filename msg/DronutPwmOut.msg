# Motor control message
uint64 timestamp			# time since system start (microseconds)
uint64 timestamp_sample	    # the timestamp the data this control response is based on was sampled

uint32[2] motor_pwm
uint32[4] servo_pwm
