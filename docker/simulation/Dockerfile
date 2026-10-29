FROM g1-unifolm-vla:0.2

WORKDIR /workspace/unifolm-vla

ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV TF_CPP_MIN_LOG_LEVEL=2

# Preserve NVIDIA GPU libraries
RUN python -c 'import importlib.metadata as m; \
print("torch=="+m.version("torch")); \
print("torchvision=="+m.version("torchvision"))' \
> /tmp/nvidia-constraints.txt

# Install Unitree LIBERO evaluation dependencies
RUN python -m pip install \
    --constraint /tmp/nvidia-constraints.txt \
    "robosuite==1.4.1" \
    "mujoco==3.3.5" \
    "bddl" \
    "easydict" \
    "cloudpickle" \
    "gym" \
    "imageio[ffmpeg]" \
    "tyro==0.9.35" \
    "requests" \
    "tqdm"

CMD ["/bin/bash"]
