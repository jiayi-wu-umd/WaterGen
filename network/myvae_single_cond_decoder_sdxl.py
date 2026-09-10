"""
MyVAE with Single Conditional Decoder for T/B conditioning - SDXL Version.

Key differences from SD1.5 version:
- Uses SDXL VAE scaling factor (0.13025 instead of 0.18215)
- Compatible with madebyollin/sdxl-vae-fp16-fix

Simplified baseline for Stage 2:
- Input: Clear image latent (4-channel, potentially from degraded clear image)
- Conditioning: T + B concatenated (6-channel)
- Output: Conditioned image (can be underwater or clean depending on T/B values)

Two forward passes per training step:
1. T, B (real) → underwater image prediction
2. T=1, B=0 → clean image prediction (identity conditioning)

UIFM physics loss: ||underwater_pred - (clean_pred * T + B)||
"""

from .myvae import Encoder, Decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def zero_module(module):
    """Zero out module parameters."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ZeroConv(nn.Module):
    """Zero-initialized convolution for skip connections."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = zero_module(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, x):
        return self.conv(x)


class MyVAE_SingleCondDecoder_SDXL(nn.Module):
    """
    Single conditional decoder VAE for T/B conditioning baseline - SDXL Version.
    
    Architecture:
    - ConditioningEncoder: Encodes T+B (6-channel) to skip features
    - Decoder: Takes clear_latent + T/B skip features → output image
    - Same decoder used for both underwater (T,B) and clean (T=1,B=0)
    
    The decoder learns to apply the physical transformation:
    - When T=1, B=0: output ≈ input (identity/clean)
    - When T<1, B>0: output = degraded underwater
    
    Key difference from SD1.5: Uses SDXL scaling factor (0.13025)
    
    Args:
        skip_connection: Whether to use skip connections from conditioning encoder
        mid_control: Whether to use mid-block control
        residual: Whether to use residual learning
        config: VAE config dict (optional, will use SDXL defaults if not provided)
    """
    
    # Default SDXL VAE config (from madebyollin/sdxl-vae-fp16-fix)
    DEFAULT_SDXL_CONFIG = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "block_out_channels": [128, 256, 512, 512],
        "layers_per_block": 2,
        "act_fn": "silu",
        "latent_channels": 4,
        "norm_num_groups": 32,
        "sample_size": 512,
        "scaling_factor": 0.13025,  # SDXL uses different scaling factor
    }
    
    def __init__(
        self,
        skip_connection=True,
        mid_control=False,
        residual=False,
        config=None,
    ):
        super(MyVAE_SingleCondDecoder_SDXL, self).__init__()
        
        # Use provided config or default SDXL config
        if config is None:
            config = self.DEFAULT_SDXL_CONFIG.copy()
        
        # SDXL uses different scaling factor
        self.scale = config.get("scaling_factor", 0.13025)
        
        # Quantization convolutions (from pretrained VAE)
        self.quant_conv = nn.Conv2d(2 * config["latent_channels"], 2 * config["latent_channels"], 1)
        self.post_quant_conv = nn.Conv2d(config["latent_channels"], config["latent_channels"], 1)
        
        # Single conditional decoder
        self.Decoder = Decoder(
            in_channels=config["latent_channels"],
            out_channels=config["out_channels"],
            up_block_types=config["up_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"]
        )
        
        # Conditioning encoder for 6-channel input (T + B)
        self.ConditioningEncoder = Encoder(
            in_channels=6,  # 6-channel: T (3) + B (3)
            out_channels=config["latent_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            act_fn=config["act_fn"],
            double_z=True,
        )
        
        self.residual = residual
        self.skip_connection = skip_connection
        self.mid_control = mid_control
        self.dtype = torch.float32

        # Zero convolutions for skip connections
        # Channel dimensions match encoder output to decoder input requirements
        self.zero_conv_0 = ZeroConv(128, 256)   # After conv_in (128) -> first up_block input (256)
        self.zero_conv_1 = ZeroConv(128, 512)   # After down_block_0 (128) -> second up_block (512)
        self.zero_conv_2 = ZeroConv(256, 512)   # After down_block_1 (256) -> third up_block (512)
        self.zero_conv_3 = ZeroConv(512, 512)   # After down_block_2 (512) -> mid_block (512)

        if self.mid_control:
            self.zero_conv_4 = ZeroConv(8, 4)
    
    def get_trainable_params(self):
        """Get list of trainable parameters."""
        params = []
        params.extend(self.Decoder.parameters())
        params.extend(self.ConditioningEncoder.parameters())
        params.extend(self.zero_conv_0.parameters())
        params.extend(self.zero_conv_1.parameters())
        params.extend(self.zero_conv_2.parameters())
        params.extend(self.zero_conv_3.parameters())
        params.extend(self.post_quant_conv.parameters())
        if self.mid_control:
            params.extend(self.zero_conv_4.parameters())
        return params

    def load_vae(self, autoencoder):
        """
        Load pretrained VAE weights from SDXL VAE.
        
        Args:
            autoencoder: Pretrained SDXL VAE model (e.g., madebyollin/sdxl-vae-fp16-fix)
        """
        print("Loading SDXL VAE weights...")
        print(f"  VAE scaling factor: {self.scale}")
        
        # Load quantization convs
        self.quant_conv.load_state_dict(copy.deepcopy(autoencoder.quant_conv.state_dict()), strict=True)
        self.post_quant_conv.load_state_dict(copy.deepcopy(autoencoder.post_quant_conv.state_dict()), strict=True)
        
        # Load decoder from pretrained
        self.Decoder.load_state_dict(copy.deepcopy(autoencoder.decoder.state_dict()), strict=True)
        
        # Initialize conditioning encoder from pretrained encoder
        self._init_conditioning_encoder_from_pretrained(autoencoder.encoder)
        
        print("SDXL VAE weights loaded successfully")
    
    def _init_conditioning_encoder_from_pretrained(self, pretrained_encoder):
        """
        Initialize the 6-channel conditioning encoder from a pretrained 3-channel encoder.
        """
        state_dict = pretrained_encoder.state_dict()
        cond_state_dict = {}
        
        for key, value in state_dict.items():
            if 'conv_in' not in key:
                cond_state_dict[key] = copy.deepcopy(value)
        
        # Load compatible weights
        self.ConditioningEncoder.load_state_dict(cond_state_dict, strict=False)
        
        # Initialize conv_in for 6 channels
        pretrained_conv_in = pretrained_encoder.conv_in
        with torch.no_grad():
            # First 3 channels (T) get pretrained weights
            self.ConditioningEncoder.conv_in.weight[:, 0:3, :, :] = pretrained_conv_in.weight.clone()
            # Last 3 channels (B) get pretrained weights
            self.ConditioningEncoder.conv_in.weight[:, 3:6, :, :] = pretrained_conv_in.weight.clone()
            if pretrained_conv_in.bias is not None:
                self.ConditioningEncoder.conv_in.bias.copy_(pretrained_conv_in.bias)
        
        print("Conditioning encoder initialized from pretrained SDXL encoder")
        print("  - Channels 0-2: Transmission (T) weights")
        print("  - Channels 3-5: Backscatter (B) weights")

    def forward(self, clear_latents, conditioning_tb):
        """
        Forward pass: decode clear image latents conditioned on T+B.
        
        Args:
            clear_latents: Clear image latents [B, 4, H/8, W/8] (already scaled by SDXL factor 0.13025)
            conditioning_tb: T+B concatenated [B, 6, H, W] in range [-1, 1]
        
        Returns:
            output_image: [B, 3, H, W] in range [-1, 1]
        """
        # Unscale latents using SDXL scaling factor
        clear_latents = clear_latents / self.scale
        
        # Post quant conv on clear latents
        clear_latents_processed = self.post_quant_conv(clear_latents)
        
        # Get conditioning features from T+B
        conditioning_latent_list = self.ConditioningEncoder(
            conditioning_tb, 
            mid_control=self.mid_control
        )
        
        if self.skip_connection:
            # Apply zero convolutions to conditioning features
            conditioning_latent_list[0] = self.zero_conv_0(conditioning_latent_list[0])
            conditioning_latent_list[1] = self.zero_conv_1(conditioning_latent_list[1])
            conditioning_latent_list[2] = self.zero_conv_2(conditioning_latent_list[2])
            conditioning_latent_list[3] = self.zero_conv_3(conditioning_latent_list[3])

            if self.mid_control:
                conditioning_latent_list[4] = self.quant_conv(conditioning_latent_list[4])
                conditioning_latent_list[4] = self.zero_conv_4(conditioning_latent_list[4])
        
        # Decode with conditioning
        output_img = self.Decoder(
            clear_latents_processed,
            composite_latents=conditioning_latent_list,
            mid_control=self.mid_control
        )

        if self.residual:
            # Residual learning: output = T - decoded
            output_img = conditioning_tb[:, 0:3, :, :] - output_img
        
        return output_img


def compute_uifm_loss(clear_pred, transmission_T, backscatter_B, underwater_pred, 
                       bidirectional=False, inverse_weight=0.5, eps=1e-3):
    """
    UIFM Physics Loss: I_uw = J * T + B
    
    Validates that the underwater prediction matches the physics model
    applied to the clean prediction.
    
    Args:
        clear_pred: Predicted clear image [B, 3, H, W] in [0, 1]
        transmission_T: Transmission map [B, 3, H, W] in [0, 1]
        backscatter_B: Backscatter map [B, 3, H, W] in [0, 1]
        underwater_pred: Predicted underwater image [B, 3, H, W] in [0, 1]
        bidirectional: If True, also enforce inverse physics constraint
        inverse_weight: Weight for inverse loss (default 0.5)
        eps: Small value to avoid division by zero
    
    Returns:
        UIFM loss (L1 between underwater prediction and physics model)
    """
    # Forward: Compute physics-based underwater image: I_uw = J * T + B
    I_uw_physics = clear_pred * transmission_T + backscatter_B
    I_uw_physics = I_uw_physics.clamp(0, 1)
    
    # Forward loss: L1 between underwater prediction and physics model
    forward_loss = F.l1_loss(underwater_pred, I_uw_physics)
    
    if not bidirectional:
        return forward_loss
    
    # Inverse: J = (I_uw - B) / T
    # Enforce that clean_pred can be recovered from underwater_pred using physics
    T_safe = transmission_T.clamp(min=eps)  # Avoid division by zero
    J_recovered = (underwater_pred - backscatter_B) / T_safe
    J_recovered = J_recovered.clamp(0, 1)
    
    # Inverse loss: L1 between recovered clean and predicted clean
    inverse_loss = F.l1_loss(clear_pred, J_recovered)
    
    # Combined bidirectional loss
    total_loss = forward_loss + inverse_weight * inverse_loss
    
    return total_loss

