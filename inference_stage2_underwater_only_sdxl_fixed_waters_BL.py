#!/usr/bin/env python
# coding=utf-8
"""
Stage 2 Underwater-Only Inference Pipeline - SDXL VERSION (CUSTOM BL COLORS)

Loads pre-saved clear_latent.pt and depth.npy from Stage 1 outputs,
uses FIXED water parameters (T, B) with multiple CUSTOM background-light colors,
and generates underwater variants using Stage 2 conditional decoding.

CUSTOM BL COLORS / WATER TYPES:
  1. Deep Blue         - dark navy-blue water
  2. Blue              - standard ocean blue
  3. Blue-Green        - teal / cyan coastal water
  4. Green             - green-dominant water
  5. Deep Green        - dark murky green water
  6. Turquoise Shallow - bright shallow tropical water
  7. Milky Turbid      - bright, low-saturation milky water
  8. Night Blue        - very dark deep water
  9. Steel Cyan        - muted cyan-gray water (BL=[51, 91, 100])
 10. Clear             - nearly clear / very shallow water (minimal color cast, ALWAYS LAST)

Each variant also uses a corresponding beta_c from the table.

This script PRESERVES existing variants/ folder and saves to variants_fixed/.

Input Structure:
  input_dir/
    prompt_000/
      clear_latent.pt     # Pre-saved latent tensor
      depth.npy           # Pre-saved depth in meters
      clear.png           # Reference clear image

Output Structure:
  input_dir/
    prompt_000/
      variants_fixed/     # 10 variants with custom BL colors
        uw_001.png        # Deep Blue
        uw_001_T.png
        uw_001_B.png
        uw_002.png        # Blue
        uw_003.png        # Blue-Green
        uw_004.png        # Green
        uw_005.png        # Deep Green
        uw_006.png        # Turquoise Shallow
        uw_007.png        # Milky Turbid
        uw_008.png        # Night Blue
        uw_009.png        # Steel Cyan
        uw_010.png        # Clear (last)
      underwater_grid_fixed.png  # Combined grid
"""

import argparse
import os
import sys
import json
import random
import glob
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Diffusers imports
from diffusers.utils import make_image_grid

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.myvae_single_cond_decoder_sdxl import MyVAE_SingleCondDecoder_SDXL

# SDXL VAE scaling factor
SDXL_SCALING_FACTOR = 0.13025


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Underwater-Only Inference - SDXL Version")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root directory containing prompt folders with pre-saved latents/depth")
    parser.add_argument("--scale_filter", type=str, default=None,
                        help="Only process specific scale (e.g., '0.9'). If None, process all scales.")
    parser.add_argument("--prompt_filter", type=str, default=None,
                        help="Only process specific prompt(s). Comma-separated numbers or names "
                             "(e.g., '0,1,5' or 'prompt_000,prompt_005')")
    
    # Stage 2 model
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                        help="Path to Stage 2 conditional decoder checkpoint (.pth)")
    
    # Generation parameters
    parser.add_argument("--num_underwater_variants", type=int, default=5,
                        help="[IGNORED] Number of variants is fixed by the number of custom BL colors (currently 11)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Output resolution (should match saved latent resolution)")
    
    # Auto-resume
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip scale directories that already have variants")
    
    # Testing
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to process (for testing). None = process all.")
    
    # Depth rescaling
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Multiplicative factor applied to the depth map before computing T/B. "
                             "E.g. 2.0 doubles all depths → stronger underwater effect. Default: 1.0")
    parser.add_argument("--depth_far_percentile", type=float, default=0.0,
                        help="Percentile threshold (0-100) above which depth values get extra scaling. "
                             "0 = disabled. E.g. 50 → only the farthest 50%% of pixels are rescaled. "
                             "Applied AFTER --depth_scale. Default: 0 (disabled)")
    parser.add_argument("--depth_far_scale", type=float, default=2.0,
                        help="Extra scale factor applied to depth values above --depth_far_percentile. "
                             "The near portion stays unchanged; the far portion is stretched by this factor "
                             "relative to the percentile boundary. Default: 2.0")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for water parameter sampling")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed water parameter info for each variant")
    
    return parser.parse_args()


class WaterParameterSampler:
    """
    Samples water parameters (T, B) for underwater image synthesis.
    Uses multiple CUSTOM background-light (BL) RGB colors and corresponding beta_c values.
    
    Custom BL colors (RGB, 0-255 range):
      0: Deep Blue         - dark navy-blue ocean
      1: Blue              - standard ocean blue
      2: Blue-Green        - teal / cyan coastal water
      3: Green             - green-dominant water
      4: Deep Green        - dark murky green water
      5: Turquoise Shallow - bright shallow tropical water
      6: Milky Turbid      - bright, low-saturation milky water
      7: Night Blue        - very dark deep water
      8: Steel Cyan        - muted cyan-gray water (BL=[51, 91, 100])
      9: Clear             - nearly clear / very shallow water (minimal color cast, ALWAYS LAST)
    """
    
    # ---------- Custom BL colors (RGB, 0-255) ----------
    # Each entry: (name, np.array([R, G, B]))
    # Tuned to match reference water-type categories
    CUSTOM_BL_LIST = [
        ("Deep Blue",         np.array([ 10.0,  20.0, 120.0])),  # dark navy-blue ocean
        ("Blue",              np.array([ 30.0,  60.0, 180.0])),  # standard ocean blue
        ("Blue-Green",        np.array([  2.0, 147.0, 159.0])),  # murky cyan/teal coastal water
        ("Green",             np.array([ 80.0, 240.0,  60.0])),  # bright, saturated green
        ("Deep Green",        np.array([  7.0,  71.0,   9.0])),  # clear but dark-green tinted water
        ("Turquoise Shallow", np.array([ 40.0, 210.0, 200.0])),  # bright shallow tropical water
        ("Milky Turbid",      np.array([140.0, 230.0, 170.0])),  # bright milky water with green tint
        ("Night Blue",        np.array([  5.0,  10.0,  60.0])),  # very dark deep water
        ("Steel Cyan",        np.array([ 51.0,  91.0, 100.0])),  # muted cyan-gray water
        ("Clear",             np.array([180.0, 220.0, 240.0])),  # very clear / shallow water (ALWAYS LAST)
    ]
    
    # Beta_c table (attenuation coefficients) - BGR format, one per custom BL
    # Direction controls which color channels survive at depth:
    #   high R (in BGR col 2) → red attenuates fast → water looks blue/green
    #   low G (in BGR col 1)  → green preserved
    # BETA_C_TABLE = [
    #     np.array([0.02206, 0.04805, 0.22922]),  # Deep Blue        - R attenuates fastest, B preserved
    #     np.array([0.02696, 0.05082, 0.23083]),  # Blue             - similar, slightly higher
    #     np.array([0.086,   0.1034,  0.2766]),   # Blue-Green       - R attenuates fastest → murky cyan/teal
    #     np.array([0.4818,  0.4339,  0.537 ]),   # Green            - R attenuated hard, G preserved → murky green
    #     np.array([0.4818,  0.4339,  0.537 ]),   # Deep Green       - clear dark-green tint
    #     np.array([0.0500,  0.0700,  0.2200]),   # Turquoise Shallow- moderate attenuation, bright water
    #     np.array([0.0900,  0.0900,  0.1400]),   # Milky Turbid     - soft attenuation, low contrast, milky look
    #     np.array([0.0400,  0.0600,  0.2600]),   # Night Blue       - more attenuation overall, dark blue
    #     np.array([0.0600,  0.0800,  0.2100]),   # Steel Cyan       - balanced attenuation, cyan-gray look
    #     np.array([0.0300,  0.0700,  0.2300]),   # Deep Teal        - moderate attenuation, teal-blue
    #     np.array([0.02206, 0.04805, 0.22922]),  # Clear            - very low attenuation, minimal color change
    # ]
    
    # Target beta_c magnitude (BGR) and per-variant scale factors
    # Higher scale = murkier (more attenuation overall)
    BETA_C_LEVELS = [
    np.array([0.05,  0.10,  0.15]),      # Level 1 (lowest)  - magnitude ~0.19
    np.array([0.10,  0.15,  0.25]),      # Level 2 (low)     - magnitude ~0.30
    np.array([0.20,  0.25,  0.35]),      # Level 3 (medium)  - magnitude ~0.47
    np.array([0.4818, 0.4339, 0.537]),   # Level 4 (current) - magnitude ~0.84
    np.array([0.60,  0.55,  0.70]),      # Level 5 (high)    - magnitude ~1.07
    np.array([0.75,  0.70,  0.85]),      # Level 6 (high)    - magnitude ~1.33
    np.array([1.00,  0.95,  1.10]),      # Level 7 (very high) - magnitude ~1.77
    np.array([1.30,  1.25,  1.40]),      # Level 8 (extremely high) - magnitude ~2.29
    np.array([1.60,  1.55,  1.70]),      # Level 9 (maximum) - magnitude ~2.81
    np.array([2.00,  1.95,  2.10]),      # Level 10 (ultra high) - magnitude ~3.50
    np.array([2.50,  2.45,  2.60]),      # Level 11 (extreme) - magnitude ~4.37
    np.array([3.00,  2.95,  3.10]),      # Level 12 (maximum extreme) - magnitude ~5.24
]
    TARGET_BETA_C = BETA_C_LEVELS[0]
    # Use similar attenuation magnitude for the colored water types so that
    # hue differences dominate, and keep Clear with very small attenuation.
    BETA_SCALE_FACTORS = [
        0.4,   # Deep Blue
        0.4,   # Blue
        0.4,   # Blue-Green
        0.4,   # Green
        0.4,   # Deep Green
        0.4,   # Turquoise Shallow
        0.4,   # Milky Turbid
        0.4,   # Night Blue
        0.4,   # Steel Cyan
        0.02   # Clear (almost no attenuation)
    ]
    BETA_SCALE_FACTORS = [x * 1.5 for x in BETA_SCALE_FACTORS]
    
    # Per-variant scale applied to the BL RGB value
    BL_VALUE_SCALE_FACTORS = [
        1.2,   # Deep Blue
        1.2,   # Blue
        1.0,   # Blue-Green
        0.5,   # Green
        0.5,   # Deep Green
        1.3,   # Turquoise Shallow (bright)
        1.1,   # Milky Turbid (bright but low saturation)
        0.8,   # Night Blue (darker)
        1.0,   # Steel Cyan
        0.001  # Clear
    ]
    
    def __init__(self, seed=42):
        """
        Args:
            seed: Random seed (kept for compatibility)
        """
        self.num_variants = len(self.CUSTOM_BL_LIST)
        
        # ----- beta_c -----
        target_norm = np.linalg.norm(self.TARGET_BETA_C)
        scale_factors = self.BETA_SCALE_FACTORS
        self.fixed_beta_c_per_variant = []
        self.fixed_beta_c_scale_per_variant = []
        
        if len(scale_factors) != self.num_variants:
            raise ValueError(f"BETA_SCALE_FACTORS must have {self.num_variants} values, got {len(scale_factors)}")
        if len(self.BL_VALUE_SCALE_FACTORS) != self.num_variants:
            raise ValueError(f"BL_VALUE_SCALE_FACTORS must have {self.num_variants} values, got {len(self.BL_VALUE_SCALE_FACTORS)}")
        
        print(f"  Setting up {self.num_variants} custom BL water variants:")
        print()
        print("  beta_c per variant (BGR, normalized + scaled):")
        # Use a single shared beta_c direction for all colored water types so that
        # only the BL color controls hue; scale factor still controls overall strength.
        base_beta_c = self.TARGET_BETA_C.copy()
        for i in range(self.num_variants):
            beta_norm = np.linalg.norm(base_beta_c)
            sf = scale_factors[i]
            if beta_norm == 0:
                scaled_beta_c = self.TARGET_BETA_C.copy() * sf
            else:
                scaled_beta_c = (base_beta_c / beta_norm) * target_norm * sf
            self.fixed_beta_c_per_variant.append(scaled_beta_c)
            self.fixed_beta_c_scale_per_variant.append(sf)
            name = self.CUSTOM_BL_LIST[i][0]
            print(f"    Variant {i} [{name}]: shared_beta_c_base={base_beta_c} -> scaled_BGR={scaled_beta_c.round(4)} (scale={sf:.3f})")
        
        # ----- custom BL colors -----
        self.fixed_bl_per_variant = []
        print()
        print("  Custom BL colors (RGB, 0-255) after BL scaling:")
        for i, (name, bl_rgb) in enumerate(self.CUSTOM_BL_LIST):
            bl_scale = self.BL_VALUE_SCALE_FACTORS[i]
            scaled_bl = bl_rgb * bl_scale
            self.fixed_bl_per_variant.append(scaled_bl)
            print(f"    Variant {i} [{name}]: BL_RGB_scaled=[{scaled_bl[0]:.1f}, {scaled_bl[1]:.1f}, {scaled_bl[2]:.1f}] "
                  f"(BL_scale={bl_scale:.3f})")

        # ----- summary of final parameters used for T and B -----
        print()
        print("  Final parameters used to compute T and B (per variant):")
        for i in range(self.num_variants):
            name = self.CUSTOM_BL_LIST[i][0]
            beta_c_bgr = self.fixed_beta_c_per_variant[i]
            beta_c_rgb = beta_c_bgr[[2, 1, 0]]
            bl_rgb_scaled = self.fixed_bl_per_variant[i]
            bl_rgb_unit = bl_rgb_scaled / 255.0
            print(
                f"    Variant {i} [{name}]: "
                f"beta_c_rgb={beta_c_rgb.round(4)}, "
                f"BL_rgb_255=[{bl_rgb_scaled[0]:.1f}, {bl_rgb_scaled[1]:.1f}, {bl_rgb_scaled[2]:.1f}], "
                f"BL_rgb_0_1=[{bl_rgb_unit[0]:.4f}, {bl_rgb_unit[1]:.4f}, {bl_rgb_unit[2]:.4f}]"
            )
    
    def sample(self, depth_map, variant_idx, verbose=False):
        """
        Compute T and B maps from depth map using custom BL water parameters.
        
        Args:
            depth_map: Depth map [H, W] in meters
            variant_idx: Index of custom BL variant to use (0-4)
            verbose: If True, return additional info dict
            
        Returns:
            T: Transmission map [3, H, W] in [0, 1]
            B: Backscatter map [3, H, W] in [0, 1]
            info: dict (only if verbose=True)
        """
        # Fixed beta_c - convert BGR to RGB
        beta_c_original = self.fixed_beta_c_per_variant[variant_idx].copy()
        beta_c = beta_c_original[[2, 1, 0]]  # BGR to RGB
        
        # Custom background light (already RGB)
        background_light = self.fixed_bl_per_variant[variant_idx]
        bl_name = self.CUSTOM_BL_LIST[variant_idx][0]
        
        # Compute T = exp(-beta_c * depth)
        T = np.exp(-beta_c[np.newaxis, np.newaxis, :] * depth_map[:, :, np.newaxis])
        T = T.transpose(2, 0, 1)  # [3, H, W]
        
        # Compute B = background_light * (1 - T)
        B = background_light[:, np.newaxis, np.newaxis] * (1 - T)
        
        # Normalize
        T = np.clip(T, 0, 1).astype(np.float32)
        B = np.clip(B / 255.0, 0, 1).astype(np.float32)
        
        if verbose:
            info = {
                'beta_c_original_bgr': beta_c_original,
                'beta_c_rgb': beta_c,
                'beta_c_table_bgr': self.BETA_C_TABLE[variant_idx],
                'beta_c_scale': self.fixed_beta_c_scale_per_variant[variant_idx],
                'variant_idx': variant_idx,
                'bl_name': bl_name,
                'background_light': background_light,
                'depth_max': depth_map.max(),
                'T_min': T.min(),
                'T_mean': T.mean(),
                'B_max': B.max(),
            }
            return T, B, info
        
        return T, B


def load_stage2_model(checkpoint_path, device, dtype=torch.float32):
    """Load Stage 2 conditional decoder model (SDXL version)."""
    print(f"Loading Stage 2 SDXL model from {checkpoint_path}...")
    
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


@torch.no_grad()
def stage2_decode(model, clear_latent, T, B, device, dtype=torch.float32):
    """
    Decode clear latent with T/B conditioning using Stage 2 model.
    
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


def find_scale_directories(input_dir, scale_filter=None):
    """
    Find all scale directories in the input directory structure.
    Supports both nested structure (prompt_*/scale_*_seed*/) and flat structure (prompt_*/).
    
    Args:
        input_dir: Root directory containing prompt folders
        scale_filter: Optional scale to filter (e.g., "0.9")
        
    Returns:
        List of (prompt_dir, scale_dir_path) tuples
    """
    scale_dirs = []
    
    # Find all prompt directories
    prompt_dirs = sorted(glob.glob(os.path.join(input_dir, "prompt_*")))
    
    for prompt_dir in prompt_dirs:
        # First, check for flat structure (files directly in prompt_dir)
        latent_path_flat = os.path.join(prompt_dir, "clear_latent.pt")
        depth_path_flat = os.path.join(prompt_dir, "depth.npy")
        
        if os.path.exists(latent_path_flat) and os.path.exists(depth_path_flat):
            # Flat structure found - use prompt_dir as the "scale_dir"
            scale_dirs.append((prompt_dir, prompt_dir))
            continue  # Skip nested search for this prompt
        
        # Otherwise, check for nested structure (scale_*_seed* subdirectories)
        scale_pattern = os.path.join(prompt_dir, "scale_*_seed*")
        for scale_dir in sorted(glob.glob(scale_pattern)):
            # Check if this matches the scale filter
            if scale_filter is not None:
                # Extract scale from directory name (e.g., "scale_0.9_seed42" -> "0.9")
                dir_name = os.path.basename(scale_dir)
                match = re.match(r"scale_(\d+\.\d+)_seed", dir_name)
                if match:
                    dir_scale = match.group(1)
                    if dir_scale != scale_filter:
                        continue
            
            # Check if required files exist
            latent_path = os.path.join(scale_dir, "clear_latent.pt")
            depth_path = os.path.join(scale_dir, "depth.npy")
            
            if os.path.exists(latent_path) and os.path.exists(depth_path):
                scale_dirs.append((prompt_dir, scale_dir))
    
    return scale_dirs


def has_existing_variants(scale_dir, num_variants=10):
    """Check if fixed variants already exist in the scale directory."""
    variants_dir = os.path.join(scale_dir, "variants_fixed")
    if not os.path.exists(variants_dir):
        return False
    
    # Check if all variant files exist
    for i in range(num_variants):
        variant_path = os.path.join(variants_dir, f"uw_{i+1:03d}.png")
        if not os.path.exists(variant_path):
            return False
    
    return True


def save_underwater_outputs(scale_dir, underwater_outputs, T_maps, B_maps, clear_image_path=None):
    """
    Save underwater variants and create a combined grid.
    
    Args:
        scale_dir: Directory to save outputs
        underwater_outputs: List of PIL Images
        T_maps: List of T maps [3, H, W]
        B_maps: List of B maps [3, H, W]
        clear_image_path: Path to clear image for grid (optional)
    """
    variants_dir = os.path.join(scale_dir, "variants_fixed")
    os.makedirs(variants_dir, exist_ok=True)
    
    # Save underwater variants
    for i, (uw_img, T_map, B_map) in enumerate(zip(underwater_outputs, T_maps, B_maps)):
        uw_img.save(os.path.join(variants_dir, f"uw_{i+1:03d}.png"))
        
        # Save T and B maps
        T_vis = (T_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        B_vis = (B_map.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(T_vis).save(os.path.join(variants_dir, f"uw_{i+1:03d}_T.png"))
        Image.fromarray(B_vis).save(os.path.join(variants_dir, f"uw_{i+1:03d}_B.png"))
    
    # Create combined grid
    grid_images = []
    
    # Add clear image if available
    if clear_image_path and os.path.exists(clear_image_path):
        clear_image = Image.open(clear_image_path)
        grid_images.append(clear_image)
    
    # Add clean_stage2 if available
    clean_path = os.path.join(scale_dir, "clean_stage2.png")
    if os.path.exists(clean_path):
        clean_image = Image.open(clean_path)
        grid_images.append(clean_image)
    
    # Add underwater variants
    grid_images.extend(underwater_outputs)
    
    if len(grid_images) > 0:
        grid = make_image_grid(grid_images, rows=1, cols=len(grid_images))
        grid.save(os.path.join(scale_dir, "underwater_grid_fixed.png"))


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
    
    # Find scale directories to process
    print(f"\nScanning input directory: {args.input_dir}")
    if args.scale_filter:
        print(f"Filtering for scale: {args.scale_filter}")
    
    scale_dirs = find_scale_directories(args.input_dir, args.scale_filter)
    print(f"Found {len(scale_dirs)} scale directories with latent/depth files")
    
    # Filter by prompt if specified
    if args.prompt_filter:
        prompt_patterns = []
        for p in args.prompt_filter.split(','):
            p = p.strip()
            if p.isdigit():
                prompt_patterns.append(f"prompt_{int(p):03d}")
            else:
                prompt_patterns.append(p)
        
        print(f"Filtering for prompts: {prompt_patterns}")
        # Exact-match filtering on directory basename, so "205" only matches "prompt_205"
        def _matches_prompt(prompt_dir, scale_dir, patterns):
            prompt_base = os.path.basename(prompt_dir)
            scale_base = os.path.basename(scale_dir)
            return (prompt_base in patterns) or (scale_base in patterns)

        scale_dirs = [
            (prompt_dir, scale_dir)
            for prompt_dir, scale_dir in scale_dirs
            if _matches_prompt(prompt_dir, scale_dir, prompt_patterns)
        ]
        print(f"After prompt filter: {len(scale_dirs)} directories")
        # List which directories are being processed after the prompt filter
        for i, (prompt_dir, scale_dir) in enumerate(scale_dirs):
            print(f"  [{i}] prompt_dir={prompt_dir}, scale_dir={scale_dir}")
    
    if not scale_dirs:
        print("No scale directories found with required files. Exiting.")
        return
    
    # Filter out existing if requested
    if args.skip_existing:
        original_count = len(scale_dirs)
        scale_dirs = [(p, s) for p, s in scale_dirs 
                      if not has_existing_variants(s)]
        skipped = original_count - len(scale_dirs)
        if skipped > 0:
            print(f"Skipping {skipped} directories with existing variants")
        print(f"Processing {len(scale_dirs)} directories")
    
    if not scale_dirs:
        print("All directories already have variants. Nothing to do.")
        return
    
    # Limit samples for testing
    if args.max_samples is not None and args.max_samples > 0:
        scale_dirs = scale_dirs[:args.max_samples]
        print(f"Testing mode: limiting to {len(scale_dirs)} sample(s)")
    
    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    
    # Stage 2: Conditional decoder
    stage2_model = load_stage2_model(args.stage2_checkpoint, device)
    
    # Water parameter sampler (with custom BL colors / water types)
    print("\nInitializing water parameter sampler with custom BL colors / water types...")
    water_sampler = WaterParameterSampler(seed=args.seed)
    
    # Save config
    config = vars(args)
    config_path = os.path.join(args.input_dir, "stage2_underwater_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("Starting underwater variant generation...")
    print("="*60)
    
    # Process each scale directory
    for prompt_dir, scale_dir in tqdm(scale_dirs, desc="Processing"):
        # Load pre-saved latent and depth
        latent_path = os.path.join(scale_dir, "clear_latent.pt")
        depth_path = os.path.join(scale_dir, "depth.npy")
        clear_image_path = os.path.join(scale_dir, "clear.png")
        
        clear_latent = torch.load(latent_path, map_location='cpu')
        depth_map = np.load(depth_path)
        
        # Resize depth if needed
        target_h, target_w = args.resolution, args.resolution
        if depth_map.shape[0] != target_h or depth_map.shape[1] != target_w:
            from scipy.ndimage import zoom as scipy_zoom
            zoom_h = target_h / depth_map.shape[0]
            zoom_w = target_w / depth_map.shape[1]
            depth_map = scipy_zoom(depth_map, (zoom_h, zoom_w), order=1)
        
        # Rescale depth by the user-specified factor
        if args.depth_scale != 1.0:
            if args.verbose:
                print(f"  Depth before scale: min={depth_map.min():.2f}, max={depth_map.max():.2f}")
            depth_map = depth_map * args.depth_scale
            if args.verbose:
                print(f"  Depth after  scale (×{args.depth_scale}): min={depth_map.min():.2f}, max={depth_map.max():.2f}")
        
        # Selectively rescale only the farthest portion of depth
        if args.depth_far_percentile > 0:
            threshold = np.percentile(depth_map, args.depth_far_percentile)
            far_mask = depth_map > threshold
            if args.verbose:
                print(f"  Depth far-rescale: percentile={args.depth_far_percentile}%, "
                      f"threshold={threshold:.2f}, far_pixels={far_mask.sum()} "
                      f"({100*far_mask.mean():.1f}%)")
                print(f"    Before: min={depth_map.min():.2f}, max={depth_map.max():.2f}")
            # Stretch the far portion: new = threshold + (old - threshold) * far_scale
            depth_map[far_mask] = threshold + (depth_map[far_mask] - threshold) * args.depth_far_scale
            if args.verbose:
                print(f"    After (far ×{args.depth_far_scale}): min={depth_map.min():.2f}, max={depth_map.max():.2f}")
        
        # Generate underwater variants
        underwater_outputs = []
        T_maps = []
        B_maps = []
        
        # Print header for this prompt (only if verbose)
        if args.verbose:
            prompt_name = os.path.basename(scale_dir)
            print(f"\n{'='*70}")
            print(f"Processing: {prompt_name}")
            print(f"{'='*70}")
        
        # Generate one variant per custom BL color / water type (num_variants total)
        num_variants = water_sampler.num_variants
        for v in range(num_variants):
            if args.verbose:
                T, B, info = water_sampler.sample(depth_map, variant_idx=v, verbose=True)
                
                # Print sampled parameters
                print(f"\n  Variant {v+1} [{info['bl_name']}]:")
                print(f"    beta_c table (BGR): [{info['beta_c_table_bgr'][0]:.4f}, {info['beta_c_table_bgr'][1]:.4f}, {info['beta_c_table_bgr'][2]:.4f}]")
                print(f"    beta_c scaled (BGR): [{info['beta_c_original_bgr'][0]:.4f}, {info['beta_c_original_bgr'][1]:.4f}, {info['beta_c_original_bgr'][2]:.4f}] (scale={info['beta_c_scale']:.3f})")
                print(f"    beta_c (RGB): [{info['beta_c_rgb'][0]:.4f}, {info['beta_c_rgb'][1]:.4f}, {info['beta_c_rgb'][2]:.4f}]")
                print(f"    BL color: {info['bl_name']}")
                print(f"    background_light (RGB): [{info['background_light'][0]:.1f}, {info['background_light'][1]:.1f}, {info['background_light'][2]:.1f}]")
                print(f"    depth_max: {info['depth_max']:.2f}m")
                print(f"    T_min: {info['T_min']:.4f}, T_mean: {info['T_mean']:.4f}")
                print(f"    B_max: {info['B_max']:.4f}")
            else:
                T, B = water_sampler.sample(depth_map, variant_idx=v, verbose=False)
            
            T_maps.append(T)
            B_maps.append(B)
            
            # Convert to tensor
            T_tensor = torch.from_numpy(T).unsqueeze(0)  # [1, 3, H, W]
            B_tensor = torch.from_numpy(B).unsqueeze(0)  # [1, 3, H, W]
            
            # Generate underwater image
            uw_output = stage2_decode(stage2_model, clear_latent, T_tensor, B_tensor, device)
            underwater_outputs.append(uw_output)
        
        # Save outputs
        save_underwater_outputs(scale_dir, underwater_outputs, T_maps, B_maps, clear_image_path)
    
    print("\n" + "="*60)
    print(f"Generation complete!")
    print(f"Processed {len(scale_dirs)} directories")
    print("Custom BL colors / water types:")
    print("  1. Deep Blue")
    print("  2. Blue")
    print("  3. Blue-Green")
    print("  4. Green")
    print("  5. Deep Green")
    print("  6. Turquoise Shallow")
    print("  7. Milky Turbid")
    print("  8. Night Blue")
    print("  9. Steel Cyan")
    print(" 10. Clear (last)")
    print("="*60)
    print("\nOutput structure (existing variants/ preserved):")
    print("  prompt_XXX/")
    print("    variants_fixed/    <-- NEW (custom BL colors)")
    print("      uw_001.png, uw_001_T.png, uw_001_B.png  (Deep Blue)")
    print("      uw_002.png, uw_002_T.png, uw_002_B.png  (Blue)")
    print("      uw_003.png, uw_003_T.png, uw_003_B.png  (Blue-Green)")
    print("      uw_004.png, uw_004_T.png, uw_004_B.png  (Green)")
    print("      uw_005.png, uw_005_T.png, uw_005_B.png  (Deep Green)")
    print("      uw_006.png, uw_006_T.png, uw_006_B.png  (Turquoise Shallow)")
    print("      uw_007.png, uw_007_T.png, uw_007_B.png  (Milky Turbid)")
    print("      uw_008.png, uw_008_T.png, uw_008_B.png  (Night Blue)")
    print("      uw_009.png, uw_009_T.png, uw_009_B.png  (Steel Cyan)")
    print("      uw_010.png, uw_010_T.png, uw_010_B.png  (Clear)")
    print("    underwater_grid_fixed.png")


if __name__ == "__main__":
    main()
