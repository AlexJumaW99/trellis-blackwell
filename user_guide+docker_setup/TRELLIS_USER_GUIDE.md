# 🏗️ TRELLIS on RTX 5070 Ti (Blackwell) — Complete Guide

> **Tested on:** Pop!_OS 24.04 · RTX 5070 Ti (sm_120) · Driver 580.x · Docker image: PyTorch 2.11.0+cu128, xformers 0.0.35  
> **Model:** TRELLIS-image-large (1.2B params) — the best available TRELLIS v1 model

---

## 📑 Table of Contents

1. [Architecture Overview](#-architecture-overview)
2. [Quick Start (Docker)](#-quick-start-docker)
3. [Quick Start (Conda)](#-quick-start-conda)
4. [Running Inference](#-running-inference)
5. [Web Demo (Gradio)](#-web-demo-gradio)
6. [Multi-View Input](#-multi-view-input)
7. [Background Removal](#-background-removal)
8. [Input Image Best Practices](#-input-image-best-practices)
9. [Training on Custom Data](#-training-on-custom-data)
10. [Content Filtering / NSFW](#-content-filtering--nsfw)
11. [Output Formats](#-output-formats)
12. [Troubleshooting](#-troubleshooting)
13. [Improving the Pipeline](#-improving-the-pipeline)

---

## 🔄 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRELLIS Pipeline                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Input    │───▶│  DINOv2      │───▶│  Stage 1: Sparse      │  │
│  │  Image    │    │  Encoder     │    │  Structure Sampling    │  │
│  │ (RGBA)    │    │  (1.1B)      │    │  (25 steps)           │  │
│  └──────────┘    └──────────────┘    └──────────┬────────────┘  │
│                                                  │               │
│                                                  ▼               │
│                                      ┌───────────────────────┐  │
│                                      │  Stage 2: SLAT        │  │
│                                      │  Latent Sampling      │  │
│                                      │  (25 steps)           │  │
│                                      └──────────┬────────────┘  │
│                                                  │               │
│                        ┌─────────────────────────┼───────┐      │
│                        ▼                         ▼       ▼      │
│               ┌──────────────┐  ┌──────────┐  ┌────────────┐   │
│               │  3D Gaussians│  │ Radiance │  │   Mesh     │   │
│               │  (.ply)      │  │ Field    │  │  (.glb)    │   │
│               └──────────────┘  └──────────┘  └────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key specs for your hardware:**

| Component | Detail |
|-----------|--------|
| GPU | RTX 5070 Ti (16GB VRAM, Blackwell sm_120) |
| PyTorch | 2.11.0+cu128 (stable; sm_120 is supported since PyTorch 2.7) |
| Attention | xformers 0.0.35 (`ATTN_BACKEND=xformers`; flash-attn is not used) |
| DINOv2 | `XFORMERS_DISABLED=1` makes it use PyTorch's native SDPA |
| Kaolin | Not installed; the FlexiCubes import falls back to a no-op (it's only a tensor shape check) |
| Memory | `expandable_segments:True` reduces fragmentation on 16GB |

---

## 🐳 Quick Start (Docker)

```bash
git clone --recurse-submodules https://github.com/AlexJumaW99/trellis-blackwell.git
cd trellis-blackwell
mkdir -p inputs outputs ~/.cache/huggingface ~/.cache/torch ~/.u2net

# Build, then run the web demo at http://localhost:7860
docker compose build
docker compose up

# Or generate from one image without the web UI
docker compose run --rm trellis python example_hq.py --image inputs/my_object.png --preset high --output outputs/my_object
```

The image is built from this repository's code. The Dockerfile and `docker-compose.yml` are in the repository root; see the root README for details.

**Rebuild from scratch:**
```bash
docker compose build --no-cache
```

---

## 🐍 Quick Start (Conda)

If you already have the environment set up:

```bash
conda activate trellis-nightly_v2
cd ~/Desktop/TRELLIS
python example.py
```

Environment variables are auto-loaded via `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`:
- `ATTN_BACKEND=xformers`
- `SPCONV_ALGO=native`
- `CUDA_HOME=$CONDA_PREFIX`
- `TORCH_CUDA_ARCH_LIST="12.0"`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 🚀 Running Inference

### Single Image to 3D

```python
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load the best model (auto-downloads ~4GB on first run)
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load your image (RGBA with transparent background works best)
image = Image.open("my_image.png")

# Generate 3D
outputs = pipeline.run(
    image,
    seed=42,
    # Tune quality vs speed:
    # sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
    # slat_sampler_params={"steps": 25, "cfg_strength": 3},
)

# Save all outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("output_gaussian.mp4", video, fps=30)

video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("output_mesh.mp4", video, fps=30)

# Export GLB (usable in Blender, Unity, Unreal)
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export("output.glb")

# Export Gaussian PLY
outputs['gaussian'][0].save_ply("output.ply")
```

### ⚙️ Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `seed` | Random | Reproducibility |
| `steps` (both stages) | 12 | Higher = better quality, slower. Try 25 for best results |
| `cfg_strength` (stage 1) | 7.5 | How closely to follow input. Higher = more faithful |
| `cfg_strength` (stage 2) | 3.0 | Refinement intensity. Keep lower to avoid artifacts |
| `simplify` | 0.95 | Mesh triangle reduction ratio (0.0-1.0) |
| `texture_size` | 1024 | Texture resolution. Use 2048 for high detail |

---

## 🌐 Web Demo (Gradio)

```bash
# Install Gradio if not already
pip install gradio

# Run the web UI
cd ~/Desktop/TRELLIS
python app.py
```

Access at the URL shown in terminal (typically `http://127.0.0.1:7860`).

The Gradio app provides:
- 📤 Image upload with drag-and-drop
- ⚙️ Adjustable sampling parameters
- 🖼️ Live 3D preview
- 💾 GLB/PLY download

---

## 👁️ Multi-View Input

**Yes, TRELLIS supports multi-view input.** See `example_multi_image.py`:

```python
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()

# Load multiple views of the same object
images = [
    Image.open("view_front.png"),
    Image.open("view_side.png"),
    Image.open("view_back.png"),
]

outputs = pipeline.run_multi_image(
    images,
    seed=42,
)
```

**Important caveats:**
- This is a **tuning-free algorithm** — no specialized multi-view model was trained
- Results may not be optimal for all input combinations
- All images should show the **same object** from different angles
- All images should have **transparent backgrounds**
- Multi-view is **NOT done automatically** — single-image mode infers unseen sides from DINOv2 features

---

## ✂️ Background Removal

**Background removal IS automatic.** From the TRELLIS documentation:

> Images should have transparent backgrounds (alpha channel) or will be automatically processed by `rembg` for background removal.

So you have two options:

1. **Let TRELLIS handle it** — just pass any image, `rembg` runs automatically
2. **Pre-process yourself** — for better control:
   ```bash
   # Using rembg CLI
   rembg i input.jpg output.png
   
   # Or in Python
   from rembg import remove
   from PIL import Image
   output = remove(Image.open("input.jpg"))
   output.save("output.png")
   ```

**Recommendation:** Pre-processing gives you better results because you can verify the background removal quality before feeding it to TRELLIS.

---

## 🖼️ Input Image Best Practices

TRELLIS was trained on **TRELLIS-500K**, a dataset of 500K diverse 3D objects from Objaverse, ABO (Amazon products), 3D-FUTURE (furniture), HSSD (household items), and Toys4k.

### ✅ Works Best With

| Category | Examples | Quality |
|----------|----------|---------|
| **Discrete objects** | Furniture, vehicles, tools, toys | ⭐⭐⭐⭐⭐ |
| **Characters/figures** | Stylized characters, figurines, statues | ⭐⭐⭐⭐ |
| **Product shots** | Shoes, electronics, packaging | ⭐⭐⭐⭐⭐ |
| **Animals** | Dogs, cats, birds (single subject) | ⭐⭐⭐⭐ |
| **Stylized/Pixar** | Cartoon-style characters, game assets | ⭐⭐⭐⭐⭐ |
| **Architecture** | Simple buildings, structures | ⭐⭐⭐⭐ |

### ⚠️ Challenging

| Category | Issue |
|----------|-------|
| **Full human bodies** | Fine details (hands, faces) may have artifacts |
| **Thin structures** | Wire, grass, hair — may collapse |
| **Scenes** | Multiple separate objects in one image |
| **Flat images** | Logos, text, paintings — no 3D depth to infer |
| **Transparent objects** | Glass, water (TRELLIS v1 limitation; v2 handles these) |

### 📐 Image Preparation Tips

```
┌──────────────────────────────────────────────┐
│              IDEAL INPUT IMAGE               │
│                                              │
│  ✅ Single object, centered                  │
│  ✅ Transparent or clean background          │
│  ✅ Good lighting, clear details visible     │
│  ✅ 3/4 view angle (shows depth)             │
│  ✅ Resolution: 512px+ on shortest side      │
│  ✅ PNG with alpha channel                   │
│                                              │
│  ❌ Multiple objects                         │
│  ❌ Busy/cluttered background                │
│  ❌ Extreme close-up or far away             │
│  ❌ Heavy occlusion                          │
│  ❌ Low resolution or blurry                 │
└──────────────────────────────────────────────┘
```

---

## 🏋️ Training on Custom Data

### ⚠️ Honest Assessment

Training TRELLIS from scratch or fine-tuning requires:
- **Hardware:** NVIDIA A100 GPUs (the models were trained on these). Your RTX 5070 Ti with 16GB is too limited for full training
- **Data:** 3D assets (meshes), not 2D images. You cannot train on photos alone
- **Dataset format:** Must match TRELLIS-500K structure (rendered views + 3D meshes)

### What You CAN Do

**1. Prepare a custom dataset:**
```bash
# Install dataset toolkits
cd ~/Desktop/TRELLIS
. ./dataset_toolkits/setup.sh

# Build metadata from your 3D assets
python dataset_toolkits/build_metadata.py ObjaverseXL \
    --source sketchfab \
    --output_dir datasets/my_dataset
```

**2. Fine-tune a decoder (most feasible on 16GB):**
```bash
python train.py \
    --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
    --output_dir outputs/my_finetune \
    --data_dir /path/to/your/3d/dataset \
    --num_gpus 1
```

**3. Dry run to test configuration:**
```bash
python train.py \
    --config configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json \
    --output_dir outputs/test \
    --data_dir /path/to/dataset \
    --tryrun
```

### Training Data Requirements

Your custom data needs to be **3D meshes** (OBJ, GLB, etc.), not photographs. The pipeline:

```
3D Meshes → Rendered Views (multi-angle) → Paired Training Data
         ↘ Structured Latent Encoding ↗
```

If you only have 2D images, you cannot directly train TRELLIS. You would need to first reconstruct 3D models (e.g., via photogrammetry or NeRF) and then use those as training data.

---

## 🔓 Content Filtering / NSFW

**TRELLIS has NO content filter, safety mode, or NSFW blocking.**

There is no code in the repository performing content moderation, prompt filtering, or image classification for safety. The pipeline processes whatever image you provide without restriction. There is nothing to turn off.

The model was trained on publicly available 3D asset datasets (Objaverse, ABO, etc.) which were "reviewed to ensure there is no personally identifiable information or harmful content" according to Microsoft's documentation, but this refers to training data curation, not inference-time filtering.

---

## 📦 Output Formats

| Format | File | Use Case |
|--------|------|----------|
| **3D Gaussians** | `.ply` | Real-time rendering, novel view synthesis |
| **Radiance Field** | (video only) | Visualization, research |
| **Mesh + Texture** | `.glb` | Blender, Unity, Unreal, 3D printing, AR/VR |

### GLB is the most versatile output

Import directly into:
- 🎮 **Game engines:** Unity, Unreal Engine, Godot
- 🎨 **3D editors:** Blender, Maya, 3ds Max
- 🌐 **Web viewers:** Three.js, model-viewer
- 📱 **AR/VR:** ARKit, ARCore, WebXR

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `sm_120 is not compatible` | You're using stable PyTorch. Switch to nightly cu128 |
| `No module named 'flash_attn'` | Set `export ATTN_BACKEND=xformers` before running |
| `CUDA out of memory` | Set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and close other GPU apps |
| `XFORMERS_AVAILABLE` warning from DINOv2 | Patch the cached DINOv2 files (see setup steps) |
| `kaolin` import error | Already patched out. If it recurs, re-run the `sed` command on `flexicubes.py` |
| `glm/glm.hpp not found` | Run `git submodule update --init --recursive` inside the extension repo |
| Models re-downloading | Set `HF_HOME` to a persistent directory |
| Environment variables lost | They persist via `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` |

### VRAM Budget (RTX 5070 Ti, 16GB)

```
Desktop compositing (cosmic-comp):  ~1.4 GB
TRELLIS model loading:              ~4.5 GB
Stage 1 sampling:                   ~3.0 GB
Stage 2 sampling:                   ~4.0 GB
Mesh decoding (peak):               ~9.0 GB  ← OOM risk
                                    ─────────
Estimated peak:                     ~13-15 GB
```

**Tips to stay under 16GB:**
- Close browsers and unnecessary apps
- Use `expandable_segments:True` (already set in env_vars)
- Lower `texture_size` from 1024 to 512 if needed

---

## 🚀 Improving the Pipeline

### Short-term Improvements

1. **Upgrade to TRELLIS.2** — Microsoft's newer 4B parameter model with better quality, PBR materials, and transparency support. Requires 24GB+ VRAM for full resolution, but supports lower resolutions
2. **Better input images** — Use AI upscalers (Real-ESRGAN) and background removers (BRIA-RMBG-2.0) before feeding to TRELLIS
3. **Multi-view input** — Provide 2-4 views for significantly better geometry

### Long-term Improvements

1. **Wait for stable PyTorch sm_120 support** — will simplify the entire setup
2. **Consider TRELLIS.2 with ComfyUI** — node-based workflow with more control
3. **Use cloud GPUs** — RunPod or Massed Compute with A100/H100 for training
4. **Post-process in Blender** — fix mesh artifacts, retopologize, add materials

---

## 📝 Quick Reference Card

```bash
# Activate environment
conda activate trellis-nightly_v2

# Run single image generation
cd ~/Desktop/TRELLIS
python example.py

# Run web demo
python app.py

# Run multi-image generation
python example_multi_image.py

# Run text-to-3D (less quality than image-to-3D)
python example_text.py

# Check GPU status
nvidia-smi

# Verify setup
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

---

*Guide created April 2026. Tested on RTX 5070 Ti (Blackwell sm_120) with PyTorch 2.12.0.dev+cu128.*
