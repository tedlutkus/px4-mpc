# Servo control message
uint64 timestamp			# time since system start (microseconds)
uint64 timestamp_sample	    # the timestamp the data this control response is based on was sampled

float32[4] flaps 			# range: [0, 1], where 1 means maximum positive position
