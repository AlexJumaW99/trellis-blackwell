FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Environment variables
ENV ATTN_BACKEND=xformers
ENV SPCONV_ALGO=native
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PATH="/usr/local/cuda/bin:${PATH}"

WORKDIR /app

# PyTorch nightly with CUDA 12.8 (Blackwell sm_120 support)
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# xformers (stable ABI, works with PyTorch 2.10+)
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

# Basic TRELLIS dependencies
RUN pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
    scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix \
    igraph transformers gradio

# utils3d
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# spconv (cu120 works with any CUDA 12.x)
RUN pip install spconv-cu120

# nvdiffrast (compile from source for sm_120)
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install /tmp/nvdiffrast --no-build-isolation && \
    rm -rf /tmp/nvdiffrast

# diffoctreerast (compile with submodules)
RUN git clone --recursive https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    pip install /tmp/diffoctreerast --no-build-isolation && \
    rm -rf /tmp/diffoctreerast

# mip-splatting (diff-gaussian-rasterization)
RUN git clone --recursive https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation && \
    rm -rf /tmp/mip-splatting

# Clone TRELLIS
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /app/TRELLIS

# Patch flexicubes to remove kaolin dependency
RUN sed -i 's/from kaolin.utils.testing import check_tensor/def check_tensor(*args, **kwargs): return True/' \
    /app/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py

WORKDIR /app/TRELLIS

# Pre-download DINOv2 and patch for Blackwell compatibility
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)" 2>/dev/null || true
RUN find /root/.cache/torch/hub -name "attention.py" -path "*/dinov2/*" -exec \
    sed -i 's/XFORMERS_AVAILABLE = True/XFORMERS_AVAILABLE = False/' {} \; && \
    find /root/.cache/torch/hub -name "swiglu_ffn.py" -path "*/dinov2/*" -exec \
    sed -i 's/XFORMERS_AVAILABLE = True/XFORMERS_AVAILABLE = False/' {} \; && \
    find /root/.cache/torch/hub -name "block.py" -path "*/dinov2/*" -exec \
    sed -i 's/XFORMERS_AVAILABLE = True/XFORMERS_AVAILABLE = False/' {} \;

EXPOSE 7860

# Default: run Gradio web demo
CMD ["python", "app.py"]
