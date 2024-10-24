############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
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

from acados_template import AcadosModel
import casadi as cs
import l4casadi as l4c
import onyxengine as onyx
from onyxengine.modeling import *
import numpy as np

class MultirotorRateModelResidual():
    def __init__(self):
        self.name = 'multirotor_rate_model'

        # constants
        self.mass = 0.5 #500g
        hover_thrust = 0.73
        self.max_thrust = 10.0#self.mass * 9.81/hover_thrust # 10N
        self.max_rate = 3.0
        
        # l4c model
        pytorch_model = onyx.load_model('cleo_residual_model_no_base')
        self.l4c_model = l4c.realtime.RealTimeL4CasADi(pytorch_model, approximation_order=1)
        self.parameter_values = None

    def get_acados_model(self) -> AcadosModel:
        def skew_symmetric(v):
            return cs.vertcat(cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0))

        def q_to_rot_mat(q):
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]

            rot_mat = cs.vertcat(
                cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

            return rot_mat

        def v_dot_q(v, q):
            rot_mat = q_to_rot_mat(q)

            return cs.mtimes(rot_mat, v)

        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3)
        v      = cs.MX.sym('v', 3)
        q = cs.MX.sym('q', 4)

        x = cs.vertcat(p, v, q)

        F = cs.MX.sym('F')
        w = cs.MX.sym('w', 3)
        u = cs.vertcat(F, w)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)

        xdot = cs.vertcat(p_dot, v_dot, q_dot)
        g = cs.vertcat(0.0, 0.0, -9.81) # gravity constant [m/s^2]
        
        force = cs.vertcat(0.0, 0.0, F)

        a_thrust = v_dot_q(force, q)/self.mass
        
        # Residual model
        x_model = cs.vertcat(v, q, u)
        res_model = self.l4c_model(x_model)
        p = self.l4c_model.get_sym_params() #symbolic taylor approximation
        model.p = p
        residual_dyn = cs.vertcat(0.0, 0.0, 0.0, res_model)
        self.parameter_values = self.l4c_model.get_params(np.array([0]*11))

        f_expl = cs.vertcat(v,
                        a_thrust + g,
                        1 / 2 * cs.mtimes(skew_symmetric(w), q)
                        ) + residual_dyn
        
        # # dynamics
        # f_expl = cs.vertcat(v,
        #                 a_thrust + g,
        #                 1 / 2 * cs.mtimes(skew_symmetric(w), q)
        #                 )

        f_impl = xdot - f_expl


        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model
    