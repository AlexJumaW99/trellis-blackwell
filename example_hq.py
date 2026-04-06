"""
TRELLIS High-Quality Generation Script
=======================================
Optimized for RTX 5070 Ti (16GB VRAM, Blackwell sm_120)

Features:
- Model offloading between stages to free VRAM for post-processing
- 2048 texture support on 16GB cards
- Configurable quality presets
- Single image and multi-image support

Usage:
    python example_hq.py                              # Default example image
    python example_hq.py --image my_photo.png         # Custom image
    python example_hq.py --image my_photo.png --preset max   # Maximum quality
    python example_hq.py --images front.png side.png  # Multi-view
"""

import os
import gc
import argparse

os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------
PRESETS = {
    "fast": {
        "sparse_structure_sampler_params": {"steps": 12, "cfg_strength": 7.5},
        "slat_sampler_params": {"steps": 12, "cfg_strength": 3},
        "simplify": 0.95,
        "texture_size": 1024,
    },
    "balanced": {
        "sparse_structure_sampler_params": {"steps": 20, "cfg_strength": 7.5},
        "slat_sampler_params": {"steps": 20, "cfg_strength": 3},
        "simplify": 0.93,
        "texture_size": 1024,
    },
    "high": {
        "sparse_structure_sampler_params": {"steps": 25, "cfg_strength": 7.5},
        "slat_sampler_params": {"steps": 25, "cfg_strength": 3},
        "simplify": 0.90,
        "texture_size": 2048,
    },
    "max": {
        "sparse_structure_sampler_params": {"steps": 50, "cfg_strength": 7.5},
        "slat_sampler_params": {"steps": 50, "cfg_strength": 3},
        "simplify": 0.85,
        "texture_size": 2048,
    },
}


def free_pipeline_vram(pipeline):
    """Move all pipeline models to CPU and free CUDA memory."""
    for name, model in pipeline.models.items():
        model.cpu()
        print(f"  [OFFLOAD] {name} -> CPU")
    torch.cuda.empty_cache()
    gc.collect()
    vram_free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  [VRAM] {vram_free:.1f} GB free after offload")


def reload_pipeline(pipeline):
    """Move all pipeline models back to GPU."""
    for name, model in pipeline.models.items():
        model.cuda()
        print(f"  [RELOAD] {name} -> GPU")


def generate_single(pipeline, image_path, preset_name, seed, output_prefix):
    """Generate 3D from a single image with model offloading."""
    preset = PRESETS[preset_name]
    print(f"\n{'='*60}")
    print(f"  TRELLIS HQ Generation")
    print(f"  Image:   {image_path}")
    print(f"  Preset:  {preset_name}")
    print(f"  Seed:    {seed}")
    print(f"  Texture: {preset['texture_size']}px")
    print(f"  Simplify:{preset['simplify']}")
    print(f"{'='*60}\n")

    image = Image.open(image_path)

    # --- Stage 1 & 2: Sampling (models on GPU) ---
    print("[1/4] Running sampling stages...")
    outputs = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params=preset["sparse_structure_sampler_params"],
        slat_sampler_params=preset["slat_sampler_params"],
    )
    print("  Sampling complete.\n")

    # --- Offload models to free VRAM for post-processing ---
    print("[2/4] Offloading models to CPU for high-quality post-processing...")
    free_pipeline_vram(pipeline)

    # --- Stage 3: Render videos ---
    print("[3/4] Rendering videos...")
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{output_prefix}_gaussian.mp4", video_gs, fps=30)
    print(f"  Saved {output_prefix}_gaussian.mp4")

    video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f"{output_prefix}_radiance.mp4", video_rf, fps=30)
    print(f"  Saved {output_prefix}_radiance.mp4")

    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{output_prefix}_mesh.mp4", video_mesh, fps=30)
    print(f"  Saved {output_prefix}_mesh.mp4")

    # --- Stage 4: GLB export (high-quality, benefits from freed VRAM) ---
    print(f"[4/4] Exporting GLB (texture={preset['texture_size']}, simplify={preset['simplify']})...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=preset["simplify"],
        texture_size=preset["texture_size"],
    )
    glb.export(f"{output_prefix}.glb")
    print(f"  Saved {output_prefix}.glb")

    # Save PLY
    outputs['gaussian'][0].save_ply(f"{output_prefix}.ply")
    print(f"  Saved {output_prefix}.ply")

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    # Reload models for next generation (if needed)
    reload_pipeline(pipeline)

    print(f"\n  Done! Output files: {output_prefix}.[glb|ply|mp4]")


def generate_multi(pipeline, image_paths, preset_name, seed, output_prefix):
    """Generate 3D from multiple views with model offloading."""
    preset = PRESETS[preset_name]
    print(f"\n{'='*60}")
    print(f"  TRELLIS HQ Multi-View Generation")
    print(f"  Images:  {image_paths}")
    print(f"  Preset:  {preset_name}")
    print(f"{'='*60}\n")

    images = [Image.open(p) for p in image_paths]

    print("[1/4] Running multi-view sampling...")
    outputs = pipeline.run_multi_image(
        images,
        seed=seed,
        sparse_structure_sampler_params=preset["sparse_structure_sampler_params"],
        slat_sampler_params=preset["slat_sampler_params"],
    )
    print("  Sampling complete.\n")

    print("[2/4] Offloading models to CPU...")
    free_pipeline_vram(pipeline)

    print("[3/4] Rendering videos...")
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{output_prefix}_gaussian.mp4", video_gs, fps=30)

    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{output_prefix}_mesh.mp4", video_mesh, fps=30)

    print(f"[4/4] Exporting GLB (texture={preset['texture_size']})...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=preset["simplify"],
        texture_size=preset["texture_size"],
    )
    glb.export(f"{output_prefix}.glb")
    outputs['gaussian'][0].save_ply(f"{output_prefix}.ply")

    torch.cuda.empty_cache()
    gc.collect()
    reload_pipeline(pipeline)

    print(f"\n  Done! Output files: {output_prefix}.[glb|ply|mp4]")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS HQ 3D Generation (16GB VRAM optimized)")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="Paths to multiple view images")
    parser.add_argument("--preset", type=str, default="high", choices=PRESETS.keys(),
                        help="Quality preset: fast, balanced, high, max")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    args = parser.parse_args()

    # Default to example image if nothing specified
    if args.image is None and args.images is None:
        args.image = "assets/example_image/T.png"

    # Determine output prefix
    if args.output:
        output_prefix = args.output
    elif args.image:
        output_prefix = os.path.splitext(os.path.basename(args.image))[0] + f"_{args.preset}"
    else:
        output_prefix = f"multi_view_{args.preset}"

    # Load pipeline
    print("Loading TRELLIS-image-large pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print("Pipeline loaded.\n")

    if args.images:
        generate_multi(pipeline, args.images, args.preset, args.seed, output_prefix)
    else:
        generate_single(pipeline, args.image, args.preset, args.seed, output_prefix)


if __name__ == "__main__":
    main()
