FROM nvcr.io/nvidia/pytorch:25.11-py3

WORKDIR /workspace/unifolm-vla

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Protect NVIDIA's working PyTorch and torchvision builds
RUN python -c 'import importlib.metadata as m; \
print("torch=="+m.version("torch")); \
print("torchvision=="+m.version("torchvision"))' \
> /tmp/nvidia-constraints.txt

# First group of inference dependencies
RUN python -m pip install \
    --constraint /tmp/nvidia-constraints.txt \
    "transformers==4.52.3" \
    "accelerate==1.5.2" \
    "omegaconf" \
    "diffusers==0.35.1" \
    "qwen-vl-utils" \
    "json-numpy" \
    "fastapi" \
    "uvicorn" \
    "pydantic==2.10.6"

CMD ["/bin/bash"]
