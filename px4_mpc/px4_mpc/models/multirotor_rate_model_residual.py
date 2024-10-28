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
# import onyxengine as onyx
# from onyxengine.modeling import *
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import Literal

class MLPConfig(BaseModel):
    num_inputs: int = 11
    num_outputs: int = 7
    sequence_length: int = 1
    hidden_layers: int = 3
    hidden_size: int = 64
    activation: Literal['relu', 'tanh', 'sigmoid'] = 'tanh'
    dropout: float = 0.2
    bias: bool = True

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        nn.Module.__init__(self)
        self.config = config
        num_inputs = config.num_inputs * config.sequence_length
        num_outputs = config.num_outputs
        hidden_layers = config.hidden_layers
        hidden_size = config.hidden_size
        activation = None
        if config.activation == 'relu':
            activation = nn.ReLU()
        elif config.activation == 'tanh':
            activation = nn.Tanh()
        elif config.activation == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {config.activation} not supported")
        dropout = config.dropout
        bias = config.bias
        layers = []
        
        # Add first hidden layer
        layers.append(nn.Linear(num_inputs, hidden_size, bias=bias))
        layers.append(activation)
        layers.append(nn.Dropout(dropout))
        
        # Add remaining hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, num_outputs, bias=bias))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights close to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Sequence input shape (batch_size, sequence_length, num_inputs)
        return self.model(x.view(x.size(0), -1))

class MultirotorRateModelResidual():
    def __init__(self):
        self.name = 'NN_multirotor_rate_model'

        # constants
        self.mass = 0.5 #500g
        self.max_thrust = 10.0
        self.max_rate = 100.0#3.0
        
        # l4c model
        pytorch_model = MLP(MLPConfig())
        pytorch_model.load_state_dict(torch.load('model_weights/cleo_residual_model_body_frame2.pt'))
        pytorch_model.eval()
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
        
        def v_dot_q_to_body(v, q):
            rot_mat = q_to_rot_mat(q).T

            return cs.mtimes(rot_mat, v)

        model = AcadosModel()

        # set up states & controls
        p      = cs.MX.sym('p', 3) # world frame
        v      = cs.MX.sym('v', 3) # body frame
        q = cs.MX.sym('q', 4) # body frame

        x = cs.vertcat(p, v, q)

        F = cs.MX.sym('F') # body frame
        w = cs.MX.sym('w', 3) # body frame
        u = cs.vertcat(F, w)

        # xdot
        p_dot      = cs.MX.sym('p_dot', 3)
        v_dot      = cs.MX.sym('v_dot', 3)
        q_dot      = cs.MX.sym('q_dot', 4)

        xdot = cs.vertcat(p_dot, v_dot, q_dot)
        g = cs.vertcat(0.0, 0.0, -9.81) # gravity constant [m/s^2]
        
        force = cs.vertcat(0.0, 0.0, F)

        a_thrust = force/self.mass
        a_g = v_dot_q_to_body(g, q)
        
        
        # Put velocity in body frame for model
        x_model = cs.vertcat(v, q, u)
        res_model = self.l4c_model(x_model)
        
        model_p = self.l4c_model.get_sym_params() #symbolic taylor approximation
        model.p = model_p
        self.parameter_values = self.l4c_model.get_params(np.array([0]*11))

        residual_dyn = cs.vertcat(0.0, 0.0, 0.0, res_model)

        f_expl = cs.vertcat(v_dot_q(v, q),
                        a_thrust + a_g,
                        1 / 2 * cs.mtimes(skew_symmetric(w), q)
                        ) + residual_dyn
        

        f_impl = xdot - f_expl


        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = self.name

        return model

    def v_dot_q_to_body_func(self, v, q):
        rot_mat = self.q_to_rot_mat_func(q).T
        if isinstance(q, np.ndarray):
            return rot_mat.dot(v)

        return cs.mtimes(rot_mat, v)

    def q_to_rot_mat_func(self, q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        if isinstance(q, np.ndarray):
            rot_mat = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

        else:
            rot_mat = cs.vertcat(
                cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

        return rot_mat