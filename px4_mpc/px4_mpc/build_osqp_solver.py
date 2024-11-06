import numpy as np
import jax.numpy as jnp
import cvxpy as cp
from jax import jacfwd, jacrev
import time
from jax import jit
import jax
jax.config.update('jax_platform_name', 'cpu')
from cvxpygen import cpg

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

# Constraints
tmax = 1.0
umin = jnp.array([0.0, -tmax, -tmax, -tmax])
umax = jnp.array([10.0, tmax, tmax, tmax])      

Q = jnp.diag(jnp.array([200, 200, 200, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
R = jnp.eye(4)*1.0

# Initial and reference states
x0 = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
xr = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u0 = jnp.array([9.8*0.5, 0.0, 0.0, 0.0])
u_prev = u0
ur = jnp.array([0.0, 0.0, 0.0, 0.0])
N = 10
horizon = N
dt = 0.05
A, B = calculate_jacobians(x0, u0)
Ad, Bd = euler_discretize(A, B, dt)
Ad_param = cp.Parameter(Ad.shape)
Bd_param = cp.Parameter(Bd.shape)
nx = 13
nu = 4
u = cp.Variable((nu, N))
x = cp.Variable((nx, N+1))
x_init = cp.Parameter(nx)
objective = 0
constraints = [x[:,0] == x_init]
for k in range(N):
    objective += cp.quad_form(x[:,k] - xr, Q) + cp.quad_form(u[:,k] - ur, R)
    constraints += [x[:,k+1] == Ad_param@x[:,k] + Bd_param@u[:,k]]
    constraints += [umin <= u[:,k], u[:,k] <= umax]
objective += cp.quad_form(x[:,N] - xr, Q)
prob = cp.Problem(cp.Minimize(objective), constraints)
cpg.generate_code(prob, code_dir='osqp_solver', solver='OSQP', solver_opts={'eps_abs': 1e-3, 'eps_rel': 1e-3, 'warm_start': True})
exit()

from osqp_solver.cpg_solver import cpg_solve
prob.register_solve('CPG', cpg_solve)


x_init.value = np.array(x0)
dt = 0.05
A, B = calculate_jacobians(jnp.array(x0), np.array(u0))
Ad, Bd = euler_discretize(A, B, dt)
Ad_param.value = np.array(Ad)
Bd_param.value = np.array(Bd)

prob.solve(method='CPG')
prob.solve(method='CPG')
prob.solve(method='CPG')
prob.solve(method='CPG')
prob.solve(method='CPG')



import time
t0 = time.time()
prob.solve(method='CPG')
tf = time.time()
print(((tf-t0)*1000.0))

A, B = calculate_jacobians(x0, u0)
Ad, Bd = euler_discretize(A, B, dt)
t0 = time.time()
A, B = calculate_jacobians(x0, u0)
Ad, Bd = euler_discretize(A, B, dt)
tf = time.time()
print(((tf-t0)*1000.0))