#!/usr/bin/env python
# coding=utf-8
"""
Stage 1 + Clean-Only Inference Pipeline - SDXL VERSION

Simplified pipeline that generates:
- Stage 1: SDXL + LoRA text-to-clear-image
- Vanilla VAE decode (clear.png)
- Optional Stage 2 clean pass (T=1, B=0)
- Optional DepthPro depth estimation (saved as .npy)
- Optional clear latent (saved as .pt for later underwater variant generation)

Skips underwater variant generation - designed to be followed by a separate
underwater pass that loads the saved latents and depth maps.

Features:
- Auto-resume: skips prompts with existing outputs (--skip_existing)
- Multi-scale LoRA comparison
"""

import argparse
import os
import sys
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Add DepthPro to path if DEPTH_PRO_SRC is set, e.g.
#   export DEPTH_PRO_SRC=/path/to/ml-depth-pro/src
_depth_pro_src = os.environ.get("DEPTH_PRO_SRC", "")
if _depth_pro_src:
    sys.path.insert(0, _depth_pro_src)

# Diffusers imports
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import make_image_grid

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.myvae_single_cond_decoder_sdxl import MyVAE_SingleCondDecoder_SDXL

# SDXL VAE scaling factor
SDXL_SCALING_FACTOR = 0.13025


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 + Clean-Only Inference - SDXL Version")
    
    # Input
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single text prompt for generation")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="File with multiple prompts (one per line)")
    parser.add_argument("--prompt_prefix", type=str, default="crystal clear underwater, ",
                        help="Prefix to add to all prompts")
    parser.add_argument("--prompt_filter", type=str, default=None,
                        help="Only process prompts containing this case-insensitive substring before prefixing")
    
    # Stage 1 model (SDXL)
    parser.add_argument("--stage1_model", type=str, 
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to SDXL base model")
    parser.add_argument("--stage1_vae", type=str, default="madebyollin/sdxl-vae-fp16-fix",
                        help="Path to SDXL VAE (default: madebyollin/sdxl-vae-fp16-fix)")
    parser.add_argument("--stage1_lora", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA scale factor (0.0-1.0). Lower values preserve base model quality. Default: 1.0")
    parser.add_argument("--lora_scales", type=str, default=None,
                        help="Comma-separated LoRA scales (e.g., '0.0,0.3,0.6,0.9,1.0'). If provided, overrides --lora_scale and creates multi-scale comparison grid.")
    parser.add_argument("--save_multiscale_grid", dest="save_multiscale_grid", action="store_true", default=True,
                        help="Save multi-scale comparison grid when multiple scales are used")
    parser.add_argument("--no_save_multiscale_grid", dest="save_multiscale_grid", action="store_false",
                        help="Disable multi-scale comparison grid generation")
    parser.add_argument("--save_clear_latent", dest="save_clear_latent", action="store_true", default=True,
                        help="Save clear_latent.pt for later reuse")
    parser.add_argument("--no_save_clear_latent", dest="save_clear_latent", action="store_false",
                        help="Do not encode/save clear_latent.pt")
    
    # Stage 2 model (SDXL VAE)
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 conditional decoder checkpoint (.pth)")
    parser.add_argument("--skip_stage2_clean_pass", action="store_true",
                        help="Skip Stage 2 clean-pass decoding and only save Stage 1 outputs")
    
    # Generation parameters
    parser.add_argument("--num_clear_images", type=int, default=1,
                        help="Number of clear images per prompt (each with different seed)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale for Stage 1")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of diffusion steps for Stage 1")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Output resolution")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    
    # Auto-resume
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip prompts that already have complete outputs")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    # DepthPro checkpoint
    parser.add_argument("--depth_checkpoint", type=str,
                        default=os.environ.get("DEPTH_CHECKPOINT", "checkpoints/depth_pro.pt"),
                        help="Path to DepthPro checkpoint")
    parser.add_argument("--skip_depth_pro", action="store_true",
                        help="Skip DepthPro loading/inference and do not save depth outputs")
    
    args = parser.parse_args()
    
    # Validation
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt_file must be provided")
    if not args.skip_stage2_clean_pass and args.stage2_checkpoint is None:
        parser.error("--stage2_checkpoint is required unless --skip_stage2_clean_pass is set")
    
    return args


def is_prompt_completed(output_dir, prompt_idx, lora_scales, seed, skip_stage2=False, skip_depth=False, save_clear_latent=True):
    """
    Check if all outputs for a prompt exist (for auto-resume).
    
    Args:
        output_dir: Output directory
        prompt_idx: Prompt index
        lora_scales: List of LoRA scales
        seed: Random seed
        
    Returns:
        True if all required outputs exist, False otherwise
    """
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}")
    
    for scale in lora_scales:
        scale_dir = os.path.join(prompt_dir, f"scale_{scale:.1f}_seed{seed}")
        required_files = ["clear.png"]
        if save_clear_latent:
            required_files.append("clear_latent.pt")
        if not skip_stage2:
            required_files.append("clean_stage2.png")
        if not skip_depth:
            required_files.append("depth.npy")
        
        for f in required_files:
            if not os.path.exists(os.path.join(scale_dir, f)):
                return False
    
    return True


def load_stage1_pipeline(model_id, lora_path, device, vae_path=None, dtype=torch.float16):
    """Load SDXL pipeline with LoRA weights and optional custom VAE.
    
    Note: LoRA weights are NOT fused - use lora_scale at inference time
    to control the strength of domain adaptation.
    """
    print(f"Loading Stage 1 SDXL pipeline from {model_id}...")
    
    # Build pipeline kwargs
    pipeline_kwargs = {
        "torch_dtype": dtype,
    }
    
    # Load custom VAE if specified
    if vae_path:
        print(f"Loading SDXL VAE from {vae_path}...")
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        pipeline_kwargs["vae"] = vae
    
    # Load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        **pipeline_kwargs,
    )
    
    # Use DDIM scheduler for deterministic generation
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights (WITHOUT fusing - apply scale at inference time)
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}...")
        pipe.load_lora_weights(lora_path)
    else:
        print(f"Warning: LoRA path not found: {lora_path}")
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def load_stage2_model(checkpoint_path, device, dtype=torch.float32):
    """Load Stage 2 conditional decoder model (SDXL version)."""
    print(f"Loading Stage 2 SDXL model from {checkpoint_path}...")
    
    # MyVAE_SingleCondDecoder_SDXL uses default SDXL config
    model = MyVAE_SingleCondDecoder_SDXL(
        skip_connection=True,
        mid_control=False,
        residual=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).to(dtype)
    model.eval()
    
    print(f"  Loaded from step {checkpoint.get('global_step', 'unknown')}")
    
    return model


def load_depth_model(device, checkpoint_path="checkpoints/depth_pro.pt"):
    """Load DepthPro model for depth estimation."""
    print("Loading DepthPro model...")
    
    from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig
    
    # Create config with absolute checkpoint path
    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=checkpoint_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    
    model, transform = create_model_and_transforms(config=config, device=device)
    model = model.to(device)
    model.eval()
    
    return model, transform


def estimate_depth(depth_model, depth_transform, image_pil, device):
    """
    Estimate depth from PIL image.
    
    Args:
        depth_model: DepthPro model
        depth_transform: DepthPro transform
        image_pil: PIL Image
        device: torch device
        
    Returns:
        depth_map: numpy array [H, W] in meters
    """
    import depth_pro
    
    # Save temp image for DepthPro (it expects file path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        image_pil.save(temp_path)
    
    # Load and transform
    image, _, f_px = depth_pro.load_rgb(temp_path)
    image = depth_transform(image)
    image = image.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = depth_model.infer(image, f_px=f_px)
        depth = prediction["depth"]
    
    depth_map = depth.cpu().numpy()
    if depth_map.ndim != 2:
        depth_map = depth_map.squeeze()
    
    # Cleanup
    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass
    
    return depth_map


@torch.no_grad()
def generate_clear_latent(pipe, prompt, num_steps, guidance_scale, generator, resolution, lora_scale=1.0, return_latent=True):
    """
    Generate clear image latent using SDXL + LoRA.
    
    Args:
        pipe: StableDiffusionXLPipeline with LoRA loaded
        prompt: Text prompt
        num_steps: Number of inference steps
        guidance_scale: CFG scale
        generator: Random generator
        resolution: Output resolution
        lora_scale: LoRA weight scale (0.0=base model only, 1.0=full LoRA). 
    
    Returns:
        latent: Clear image latent [1, 4, H/8, W/8]
        clear_image: PIL Image (vanilla VAE decoded)
    """
    # Run standard pipeline to get image with dynamic LoRA scale
    output = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=resolution,
        width=resolution,
        output_type="pil",
        cross_attention_kwargs={"scale": lora_scale},
    )
    clear_image = output.images[0]
    
    if not return_latent:
        return None, clear_image
    
    # Encode to latent for Stage 2 or later reuse
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    clear_tensor = transform(clear_image).unsqueeze(0).to(pipe.device, pipe.dtype)
    
    # Encode to latent
    clear_tensor = clear_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    latent = pipe.vae.encode(clear_tensor).latent_dist.sample()
    latent = latent * SDXL_SCALING_FACTOR  # SDXL VAE scaling factor
    
    return latent, clear_image


@torch.no_grad()
def stage2_decode(model, clear_latent, T, B, device, dtype=torch.float32):
    """
    Decode clear latent with T/B conditioning using Stage 2 model (SDXL version).
    
    Args:
        model: Stage 2 MyVAE_SingleCondDecoder_SDXL
        clear_latent: Clear image latent [1, 4, H/8, W/8]
        T: Transmission map [1, 3, H, W] in [0, 1]
        B: Backscatter map [1, 3, H, W] in [0, 1]
        device: torch device
        dtype: torch dtype
        
    Returns:
        output_image: PIL Image
    """
    # Prepare conditioning
    conditioning_tb = torch.cat([T, B], dim=1)  # [1, 6, H, W]
    conditioning_tb = conditioning_tb * 2.0 - 1.0  # [0,1] -> [-1,1]
    conditioning_tb = conditioning_tb.to(device, dtype)
    
    clear_latent = clear_latent.to(device, dtype)
    
    # Forward pass
    output = model(clear_latent, conditioning_tb)
    
    # Convert to PIL
    output = (output / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
    output = output[0].cpu().permute(1, 2, 0).numpy()
    output = (output * 255).astype(np.uint8)
    output_pil = Image.fromarray(output)
    
    return output_pil


def save_multi_scale_outputs(output_dir, prompt_idx, prompt, scale_results, seed=None, save_grid=True):
    """
    Save outputs for multi-scale LoRA comparison (clean-only version).
    
    Args:
        output_dir: Output directory
        prompt_idx: Prompt index
        prompt: Original prompt text
        scale_results: List of (lora_scale, clear_image, clean_output, depth_map, clear_latent)
        seed: Random seed used
    """
    from PIL import ImageDraw
    
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}")
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Save prompt text
    prompt_file = os.path.join(prompt_dir, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    
    # Save individual scale results
    for lora_scale, clear_image, clean_output, depth_map, clear_latent in scale_results:
        scale_dir = os.path.join(prompt_dir, f"scale_{lora_scale:.1f}_seed{seed}")
        os.makedirs(scale_dir, exist_ok=True)
        
        # Save clear image (vanilla VAE decoded)
        clear_image.save(os.path.join(scale_dir, "clear.png"))
        
        if clean_output is not None:
            # Save clean output (Stage 2 with T=1, B=0)
            clean_output.save(os.path.join(scale_dir, "clean_stage2.png"))
        
        if depth_map is not None:
            # Save depth as .npy (raw meters from DepthPro)
            np.save(os.path.join(scale_dir, "depth.npy"), depth_map)
            
            # Save depth visualization as .png
            depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            Image.fromarray(depth_vis).save(os.path.join(scale_dir, "depth.png"))
        
        if clear_latent is not None:
            # Save latent for later reuse
            torch.save(clear_latent.cpu(), os.path.join(scale_dir, "clear_latent.pt"))
    
    # Create multi-scale comparison grid (optional)
    # Row format: [Scale Label | Clear | optional Clean_Stage2]
    num_scales = len(scale_results)
    if num_scales == 0:
        return
    if not save_grid or num_scales <= 1:
        print(f"  Saved outputs to {prompt_dir} (multi-scale grid skipped)")
        return
    
    # Get first result to determine size
    first_result = scale_results[0]
    img_size = first_result[1].size  # (width, height)
    
    # Label column width
    label_width = 80
    
    # Create label images for each scale
    def create_label_image(text, width, height):
        """Create a label image with text."""
        img = Image.new('RGB', (width, height), color=(40, 40, 40))
        try:
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            draw.text((x, y), text, fill=(255, 255, 255))
        except:
            pass
        return img
    
    include_clean = any(result[2] is not None for result in scale_results)
    
    # Build grid rows
    rows = []
    for lora_scale, clear_image, clean_output, depth_map, clear_latent in scale_results:
        row_images = []
        
        # Label
        label = create_label_image(f"s={lora_scale:.1f}", label_width, img_size[1])
        row_images.append(label)
        
        # Clear image
        row_images.append(clear_image)
        
        # Clean Stage2 output
        if include_clean and clean_output is not None:
            row_images.append(clean_output)
        
        # Concatenate horizontally
        row_width = sum(img.size[0] for img in row_images)
        row_height = img_size[1]
        row_img = Image.new('RGB', (row_width, row_height))
        x_offset = 0
        for img in row_images:
            row_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        rows.append(row_img)
    
    # Add header row
    header_labels = ["LoRA", "Clear"]
    if include_clean:
        header_labels.append("Clean")
    header_widths = [label_width] + [img_size[0]] * (len(header_labels) - 1)
    header_row = Image.new('RGB', (sum(header_widths), 30), color=(60, 60, 60))
    try:
        draw = ImageDraw.Draw(header_row)
        x_offset = 0
        for label, width in zip(header_labels, header_widths):
            bbox = draw.textbbox((0, 0), label)
            text_width = bbox[2] - bbox[0]
            x = x_offset + (width - text_width) // 2
            draw.text((x, 8), label, fill=(255, 255, 255))
            x_offset += width
    except:
        pass
    
    # Combine all rows
    total_width = rows[0].size[0]
    total_height = 30 + sum(row.size[1] for row in rows)
    combined_grid = Image.new('RGB', (total_width, total_height))
    
    # Paste header
    combined_grid.paste(header_row, (0, 0))
    
    # Paste rows
    y_offset = 30
    for row in rows:
        combined_grid.paste(row, (0, y_offset))
        y_offset += row.size[1]
    
    # Save combined grid
    combined_grid.save(os.path.join(prompt_dir, f"multi_scale_grid_seed{seed}.png"))
    
    print(f"  Saved multi-scale outputs to {prompt_dir}")


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts
    prompt_entries = []
    if args.prompt:
        prompt_entries.append((0, args.prompt))
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt_entries.extend((idx, line.strip()) for idx, line in enumerate(f) if line.strip())
    
    total_prompts = len(prompt_entries)
    if args.prompt_filter:
        prompt_filter = args.prompt_filter.lower()
        prompt_entries = [
            (prompt_idx, prompt)
            for prompt_idx, prompt in prompt_entries
            if prompt_filter in prompt.lower()
        ]
        print(f"Prompt filter '{args.prompt_filter}': kept {len(prompt_entries)} of {total_prompts} prompt(s)")
    
    print(f"Processing {len(prompt_entries)} prompt(s)")
    
    # Parse LoRA scales
    if args.lora_scales:
        lora_scales = [float(s.strip()) for s in args.lora_scales.split(",")]
        print(f"Multi-scale mode: testing LoRA scales {lora_scales}")
    else:
        lora_scales = [args.lora_scale]
    
    # Check for prompts to skip (auto-resume)
    prompts_to_process = []
    for prompt_idx, prompt in prompt_entries:
        current_seed = args.seed + prompt_idx * 1000
        if args.skip_existing and is_prompt_completed(
            args.output_dir,
            prompt_idx,
            lora_scales,
            current_seed,
            skip_stage2=args.skip_stage2_clean_pass,
            skip_depth=args.skip_depth_pro,
            save_clear_latent=args.save_clear_latent,
        ):
            print(f"Skipping prompt {prompt_idx} (already completed)")
        else:
            prompts_to_process.append((prompt_idx, prompt))
    
    if not prompts_to_process:
        print("All prompts already completed. Nothing to do.")
        return
    
    print(f"Processing {len(prompts_to_process)} prompt(s) (skipped {len(prompt_entries) - len(prompts_to_process)} existing)")
    
    # Load models
    print("\n" + "="*60)
    print("Loading SDXL models...")
    print("="*60)
    
    # Stage 1: SDXL + LoRA (with optional custom VAE)
    pipe = load_stage1_pipeline(args.stage1_model, args.stage1_lora, device, args.stage1_vae)
    
    # Stage 2: Conditional decoder (SDXL version)
    stage2_model = None
    if not args.skip_stage2_clean_pass:
        stage2_model = load_stage2_model(args.stage2_checkpoint, device)
    
    # DepthPro
    depth_model, depth_transform = None, None
    if not args.skip_depth_pro:
        depth_model, depth_transform = load_depth_model(device, args.depth_checkpoint)
    
    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Starting generation (clean-only mode)...")
    print("="*60)
    
    # Import zoom here to avoid repeated imports
    scipy_zoom = None
    if not args.skip_depth_pro:
        from scipy.ndimage import zoom as scipy_zoom
    
    # Process each prompt
    for prompt_idx, prompt in tqdm(prompts_to_process, desc="Generating"):
        print(f"\nPrompt index {prompt_idx}: {prompt[:50]}...")
        
        # Add prefix
        full_prompt = args.prompt_prefix + prompt
        
        # Generate clear images
        for clear_idx in range(args.num_clear_images):
            current_seed = args.seed + prompt_idx * 1000 + clear_idx
            
            print(f"  Multi-scale generation (seed={current_seed})...")
            
            target_h, target_w = args.resolution, args.resolution
            
            scale_results = []
            need_clear_latent = args.save_clear_latent or not args.skip_stage2_clean_pass
            
            for lora_scale in lora_scales:
                print(f"    LoRA scale={lora_scale}...")
                
                # Reset generator for same noise across scales
                generator = torch.Generator(device=device).manual_seed(current_seed)
                
                # Stage 1: Generate clear latent
                clear_latent, clear_image = generate_clear_latent(
                    pipe, full_prompt, args.num_inference_steps,
                    args.guidance_scale, generator, args.resolution,
                    lora_scale=lora_scale,
                    return_latent=need_clear_latent,
                )
                
                depth_map = None
                if not args.skip_depth_pro:
                    # Estimate depth for THIS scale's clear image
                    depth_map = estimate_depth(depth_model, depth_transform, clear_image, device)
                    if depth_map.shape[0] != target_h or depth_map.shape[1] != target_w:
                        zoom_h = target_h / depth_map.shape[0]
                        zoom_w = target_w / depth_map.shape[1]
                        depth_map = scipy_zoom(depth_map, (zoom_h, zoom_w), order=1)
                
                clean_output = None
                if not args.skip_stage2_clean_pass:
                    # Stage 2: Clean output (T=1, B=0)
                    ones_T = torch.ones(1, 3, args.resolution, args.resolution)
                    zeros_B = torch.zeros(1, 3, args.resolution, args.resolution)
                    clean_output = stage2_decode(stage2_model, clear_latent, ones_T, zeros_B, device)
                
                scale_results.append((lora_scale, clear_image, clean_output, depth_map, clear_latent))
            
            # Save multi-scale outputs
            save_multi_scale_outputs(
                args.output_dir,
                prompt_idx,
                prompt,
                scale_results,
                seed=current_seed,
                save_grid=args.save_multiscale_grid,
            )
    
    print("\n" + "="*60)
    print(f"Generation complete! Results saved to: {args.output_dir}")
    print("="*60)
    print("\nOutput structure per prompt:")
    print("  prompt_XXX/")
    print("    prompt.txt")
    print("    scale_X.X_seedY/")
    print("      clear.png         # Vanilla VAE decoded")
    if args.save_clear_latent:
        print("      clear_latent.pt   # Latent for later underwater pass")
    if not args.skip_stage2_clean_pass:
        print("      clean_stage2.png  # Stage 2 with T=1, B=0")
    if not args.skip_depth_pro:
        print("      depth.npy         # Raw depth in meters")
        print("      depth.png         # Depth visualization")
    print("    multi_scale_grid_seedY.png  # Only when enabled and >1 scale")


if __name__ == "__main__":
    main()
