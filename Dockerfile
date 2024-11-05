ARG FROM_IMAGE=ros:humble
ARG OVERLAY_WS=/opt/ros/overlay_ws
#ARG TARGETPLATFORM=linux/arm64/v8  # Default to arm64/v8

# multi-stage for caching
#FROM --platform=$TARGETPLATFORM $FROM_IMAGE AS cacher
FROM $FROM_IMAGE AS cacher

# clone overlay source
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
#RUN date > /cache-bust
#ADD cache-bust.txt /tmp/cache-bust.txt
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src
RUN echo "\
repositories: \n\
  ros2/demos: \n\
    type: git \n\
    url: https://github.com/ros2/demos.git \n\
    version: ${ROS_DISTRO} \n\
  px4_msgs: \n\
    type: git \n\
    url: https://github.com/PX4/px4_msgs.git \n\
    version: main \n\
  px4_mpc: \n\
    type: git \n\
    url: https://github.com/tedlutkus/px4-mpc.git \n\
    version: master \n\
  px4_offboard: \n\
    type: git \n\
    url: https://github.com/Jaeyoung-Lim/px4-offboard.git \n\
    version: master \n\
" > ../overlay.repos

#px4_ros_com: \n\
#type: git \n\
#url: https://github.com/PX4/px4_ros_com.git \n\
#version: main \n\

# Import all repositories using vcs
RUN vcs import ./ < ../overlay.repos

# Copy local msg into px4_msgs
COPY ./msg px4_msgs/msg

# copy manifests for caching
WORKDIR /opt
RUN mkdir -p /tmp/opt && \
    find ./ -name "package.xml" | \
      xargs cp --parents -t /tmp/opt && \
    find ./ -name "COLCON_IGNORE" | \
      xargs cp --parents -t /tmp/opt || true

# multi-stage for building
FROM $FROM_IMAGE AS builder

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
COPY --from=cacher /tmp/$OVERLAY_WS/src ./src
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && rosdep install -y \
      --from-paths \
        src/ros2/demos/demo_nodes_cpp \
        src/ros2/demos/demo_nodes_py \
        src/px4_msgs \
        #src/px4_ros_com \
        src/px4_mpc \
        src/px4_offboard \
      --ignore-src \
    && rm -rf /var/lib/apt/lists/*

# build overlay source
COPY --from=cacher $OVERLAY_WS/src ./src
ARG OVERLAY_MIXINS="release"
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
      --parallel-workers 4 \
      --packages-select \
        demo_nodes_cpp \
        demo_nodes_py \
        px4_msgs \
        #px4_ros_com \
        px4_mpc \
        mpc_msgs \
        px4_offboard \
      --mixin $OVERLAY_MIXINS

# Install dependencies for acados
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      cmake \
      git \
      libblas-dev \
      liblapack-dev \
      libhdf5-dev \
      python3-pip \
      wget \
      cargo && \
    rm -rf /var/lib/apt/lists/*

# Clone and build acados
WORKDIR /opt
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. && \
    make install -j4

# Install acados_template Python package
RUN pip3 install -e /opt/acados/interfaces/acados_template

# Set environment variables for acados
#ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/acados/lib"
ENV LD_LIBRARY_PATH="/opt/acados/lib"
ENV ACADOS_SOURCE_DIR="/opt/acados"

# Download and install the tera_renderer binary 
#RUN wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-osx -O /opt/acados/bin/t_renderer && \
#RUN wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux -O /opt/acados/bin/t_renderer && \
#    chmod +x /opt/acados/bin/t_renderer

# Compile tera renderer library from source
WORKDIR /opt/acados/bin
RUN git clone https://github.com/acados/tera_renderer.git && \
    cd tera_renderer && \
    cargo build --verbose --release && \
    cd .. && \
    mv tera_renderer/target/release/t_renderer .
# git pull from https://github.com/acados/tera_renderer/tree/master
# run command "cargo build --verbose --release"
WORKDIR /opt

# Install dependencies for l4casadi
# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip cmake libopenblas-dev
#RUN pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu
#RUN pip install setuptools==68.1 scikit-build>=0.17 cmake>=3.27 ninja>=1.11
#RUN pip install l4casadi --no-build-isolation
RUN pip install ninja
RUN pip install ecos==2.0.5
RUN pip install setuptools
RUN pip install numpy
RUN pip install meson
RUN pip install osqp scipy
RUN pip install cvxpy
RUN pip install jax
RUN pip install cvxpygen

WORKDIR /opt/ros/overlay_ws/src/px4_mpc/px4_mpc/px4_mpc
RUN python3 build_osqp_solver.py
RUN mv /opt/ros/overlay_ws/src/px4_mpc/px4_mpc/px4_mpc/cpg_solver.py /opt/ros/overlay_ws/src/px4_mpc/px4_mpc/px4_mpc/osqp_solver/
# move cpg_solver.py file into osqp_solver/ directory
WORKDIR /opt/ros/overlay_ws/src/px4_mpc/px4_mpc/px4_mpc/osqp_solver
RUN pip install -e .

WORKDIR $OVERLAY_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
      --parallel-workers 4 \
      --packages-select \
        demo_nodes_cpp \
        demo_nodes_py \
        px4_msgs \
        #px4_ros_com \
        px4_mpc \
        mpc_msgs \
        px4_offboard \
      --mixin $OVERLAY_MIXINS

# source entrypoint setup
ENV OVERLAY_WS $OVERLAY_WS
RUN sed --in-place --expression \
      '$isource "$OVERLAY_WS/install/setup.bash"' \
      /ros_entrypoint.sh

# run launch file
CMD ["ros2", "launch", "px4_mpc", "mpc_quadrotor_launch.py"]
#CMD ["ros2", "run", "px4_mpc", "mpc_quadrotor"]

#export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"/opt/acados/lib"
# python3 examples/acados_python/getting_started/minimal_example_ocp.py

# docker run -d -v $(pwd)/mpc_qr_weights/qr_weights.json:/mpc_config/qr_weights.json px4_mpc
# docker run -d -v $(pwd)/mpc_qr_weights/qr_weights.json:/mpc_config/qr_weights.json localhost:5000/px4_mpc
# ros2 service call /set_pose mpc_msgs/srv/SetPose "{pose: {position: {x: 0.0, y: 0.0, z: 1.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
# ros2 service call /set_pose mpc_msgs/srv/SetPose "{pose: {position: {x: 0.0, y: 0.0, z: 1.0}}}"
# px4-qshell ekf2 stop
# px4-qshell ekf2 start
# docker push 192.168.8.1:5000/px4_mpc:latest
# docker pull localhost:5000/px4_mpc:latest


# Instructions to run the drone:
# ssh into voxl
# cd /
# (might want to restart state estimates) -> px4-qshell ekf2 stop -> px4-qshell ekf2 start
# docker run -d -v $(pwd)/mpc_qr_weights/qr_weights.json:/mpc_config/qr_weights.json localhost:5000/px4_mpc

# To change the gains:
# edit the json at /mpc_qr_weights/qr_weights.json
# Q: state cost (higher = more important) [pos[3], vel[3], quaternion_attitude[4]]
# R: control cost (higher = more cost for using control) [thrust, rate_x, rate_y, rate_z]
# scale_control: multiplier to joystick input in setting moving position target in front of drone to reach

# docker run -d -v $(pwd)/mpc_config/:/mpc_config/ localhost:5000/px4_mpc
# docker run -d -v $(pwd)/mpc_config:/mpc_config localhost:5000/px4_mpc_nn