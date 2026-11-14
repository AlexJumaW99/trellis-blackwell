"""
TRELLIS Batch Processing Script
================================
Process a folder of images into 3D models.

Usage:
    python batch_generate.py --input_dir ./my_images --output_dir ./my_3d_outputs
    python batch_generate.py --input_dir ./my_images --output_dir ./my_3d_outputs --preset max
"""

import os
import gc
import glob
import argparse
import time

os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Keep weights on the CPU and move each model to the GPU only for its stage (set to 0 on 24 GB+ cards)
os.environ.setdefault('TRELLIS_LOW_VRAM', '1')

import torch
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


PRESETS = {
    "fast":     {"steps": 12, "cfg": 7.5, "slat_cfg": 3, "simplify": 0.95, "tex": 1024},
    "balanced": {"steps": 20, "cfg": 7.5, "slat_cfg": 3, "simplify": 0.93, "tex": 1024},
    "high":     {"steps": 25, "cfg": 7.5, "slat_cfg": 3, "simplify": 0.90, "tex": 2048},
    "max":      {"steps": 50, "cfg": 7.5, "slat_cfg": 3, "simplify": 0.85, "tex": 2048},
}

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}


def process_image(pipeline, image_path, output_dir, preset, seed):
    """Process a single image with model offloading."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    prefix = os.path.join(output_dir, name)

    print(f"\n  Processing: {image_path}")
    start = time.time()

    image = Image.open(image_path)

    # Sample
    outputs = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params={"steps": preset["steps"], "cfg_strength": preset["cfg"]},
        slat_sampler_params={"steps": preset["steps"], "cfg_strength": preset["slat_cfg"]},
    )

    # Offload models for post-processing
    for model in pipeline.models.values():
        model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Render
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{prefix}_gaussian.mp4", video, fps=30)

    # Export GLB
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=preset["simplify"],
        texture_size=preset["tex"],
    )
    glb.export(f"{prefix}.glb")
    outputs['gaussian'][0].save_ply(f"{prefix}.ply")

    # Clean and reload
    torch.cuda.empty_cache()
    gc.collect()
    pipeline.cuda()  # no-op in low-VRAM mode

    elapsed = time.time() - start
    print(f"  Done: {name} ({elapsed:.1f}s) -> {prefix}.glb")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="TRELLIS Batch 3D Generation")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder for output 3D files")
    parser.add_argument("--preset", type=str, default="high", choices=PRESETS.keys())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    preset = PRESETS[args.preset]

    # Find all images
    images = sorted([
        f for f in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])

    if not images:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(images)} images. Preset: {args.preset}")
    print(f"Output: {args.output_dir}\n")

    # Load pipeline once
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    total_time = 0
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", end="")
        elapsed = process_image(pipeline, img_path, args.output_dir, preset, args.seed)
        total_time += elapsed

    print(f"\n{'='*60}")
    print(f"  Batch complete: {len(images)} images in {total_time:.1f}s")
    print(f"  Average: {total_time/len(images):.1f}s per image")
    print(f"  Output:  {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
