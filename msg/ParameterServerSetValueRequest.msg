# ParameterServerSetValueRequest : Used by a client to update the value for a parameter

uint64 timestamp                        # time since system start (microseconds)

char[17] parameter_name     # Parameter name with null termination
int32 int_value             # Optional value for an integer parameter
float32 float_value         # Optional value for a float parameter
