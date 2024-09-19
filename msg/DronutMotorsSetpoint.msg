# Motor control message
uint64 timestamp			# time since system start (microseconds)
uint64 timestamp_sample	    # the timestamp the data this control response is based on was sampled

float32[2] motors 			# range: [0, 1], where 1 means maximum positive thrust
