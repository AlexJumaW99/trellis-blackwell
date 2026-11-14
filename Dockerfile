# TRELLIS (image-to-3D) for NVIDIA Blackwell GPUs (RTX 50-series, sm_120)
#
# Builds THIS repository's code (not upstream), with:
#   - stable PyTorch 2.11.0 + CUDA 12.8 (sm_120 support), pinned
#   - xformers 0.0.35 as the attention backend (flash-attn is not used)
#   - nvdiffrast, diffoctreerast and diff-gaussian-rasterization compiled for sm_120, pinned to commits
#   - Kaolin not installed (see the flexicubes step below)
#
# Build:  docker compose build        Run:  see README.md
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3.10 python3.10-dev python3-pip build-essential ninja-build \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Compile CUDA extensions for Blackwell only
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=8

# PyTorch + xformers, pinned (all from the official cu128 index)
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu128

# Ubuntu 22.04's pip can't read pyproject.toml metadata: the git installs below would all be
# registered as "UNKNOWN" and uninstall each other. Use a newer, pinned toolchain.
RUN python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1

# Runtime libraries Open3D links against
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libusb-1.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# TRELLIS Python dependencies (pinned versions in requirements-docker.txt)
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# CUDA extensions, compiled from pinned commits
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    git -C /tmp/nvdiffrast checkout 253ac4fcea7de5f396371124af597e6cc957bfae && \
    pip install /tmp/nvdiffrast --no-build-isolation && rm -rf /tmp/nvdiffrast
RUN git clone https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    git -C /tmp/diffoctreerast checkout b09c20b84ec3aace4729e6e18a613112320eca3a && \
    git -C /tmp/diffoctreerast submodule update --init --recursive && \
    pip install /tmp/diffoctreerast --no-build-isolation && rm -rf /tmp/diffoctreerast
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    git -C /tmp/mip-splatting checkout dda02ab5ecf45d6edb8c540d9bb65c7e451345a9 && \
    git -C /tmp/mip-splatting submodule update --init --recursive && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation && \
    rm -rf /tmp/mip-splatting

# This repository's code
WORKDIR /app
COPY . /app

# FlexiCubes imports Kaolin only for `check_tensor`, a debug shape assertion. Kaolin has no wheels for
# this PyTorch version, so fall back to a no-op when it's missing. The build fails if the patch didn't apply.
RUN f=trellis/representations/mesh/flexicubes/flexicubes.py && \
    test -f "$f" || { echo "FlexiCubes submodule missing: clone with --recurse-submodules" >&2; exit 1; } && \
    sed -i 's/^from kaolin.utils.testing import check_tensor$/try:\n    from kaolin.utils.testing import check_tensor\nexcept ImportError:\n    def check_tensor(*args, **kwargs):\n        return True/' "$f" && \
    grep -q "except ImportError" "$f"

# Writable folders for the non-root runtime user (compose runs as your UID)
RUN mkdir -p /app/tmp /app/outputs /cache && chmod 777 /app/tmp /app/outputs /cache

ENV ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    XFORMERS_DISABLED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRELLIS_LOW_VRAM=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    U2NET_HOME=/cache/u2net \
    HOME=/tmp \
    USER=trellis \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860
CMD ["python", "app.py"]
