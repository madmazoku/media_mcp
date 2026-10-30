#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native In-Canvas Latent Prefix Conditioning for LTX-2.3 (Media-MCP).
Replaces the flawed LTXVAddLatentGuide + LTXVCropGuides pattern (which appended tokens
to the attention sequence while leaving the canvas as pure white noise).

1. In-Canvas Latent Replacement:
   Directly pins prefix latent frames into the target video latent canvas:
   samples[:, :, :k] = prefix_latent[:, :, :k]
2. Diffusion Forcing (Noise Masking):
   Generates a denoise mask:
   noise_mask[:, :, :k] = 0.0 (completely freezes prefix frames from white noise addition)
   noise_mask[:, :, k:] = 1.0 (allows full diffusion for newly generated core frames)
3. Spatial Upscaling & Matching:
   Ensures prefix spatial size matches target latent size (height // 32, width // 32).
"""

from __future__ import annotations
import torch
import comfy.model_management
import comfy.utils


class LTXLatentPrefixToVideo:
    """
    LTX-2.3 In-Canvas Latent Prefix Conditioning.
    Pins the prefix latent directly into the target latent tensor and sets noise_mask=0
    for the prefix frames, allowing seamless diffusion of subsequent core frames.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "prefix_latent": ("LATENT",),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ramp_latents": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/ltxv"

    def execute(self, latent, prefix_latent, strength=1.0, ramp_latents=0):
        samples = latent["samples"].clone()
        src_samples = prefix_latent["samples"]

        device = samples.device
        dtype = samples.dtype

        # src_samples shape: [B, 128, T, H, W] for LTX-2.3
        p_samples = src_samples.to(device=device, dtype=dtype)

        # Match batch size if needed
        if p_samples.shape[0] < samples.shape[0]:
            p_samples = p_samples.repeat(samples.shape[0], 1, 1, 1, 1)
        elif p_samples.shape[0] > samples.shape[0]:
            p_samples = p_samples[:samples.shape[0]]

        # Match spatial dimensions if needed (LTX latent spatial stride is 32)
        if p_samples.shape[-2:] != samples.shape[-2:]:
            p_samples = comfy.utils.common_upscale(
                p_samples, samples.shape[-1], samples.shape[-2], "bilinear", "center"
            )

        k = min(p_samples.shape[2], samples.shape[2])

        # 1. In-canvas latent injection: pin the prefix frames directly into the canvas
        samples[:, :, :k] = p_samples[:, :, :k]

        # 2. Diffusion Forcing: build denoise mask
        # 0.0 = completely preserved (no noise added / no denoise)
        # 1.0 = fully diffused
        noise_mask = torch.ones(
            (samples.shape[0], 1, samples.shape[2], 1, 1),
            dtype=torch.float32,
            device=device,
        )

        unmasked_val = max(0.0, 1.0 - float(strength))
        noise_mask[:, :, :k] = unmasked_val

        # Optional soft ramp over boundary frames to smooth gradient transition
        if ramp_latents > 0 and k < samples.shape[2]:
            ramp_end = min(k + ramp_latents, samples.shape[2])
            ramp_count = ramp_end - k
            ramp_vals = torch.linspace(unmasked_val, 1.0, ramp_count + 1, device=device, dtype=torch.float32)
            noise_mask[:, :, k:ramp_end] = ramp_vals[:ramp_count].view(1, 1, ramp_count, 1, 1)

        out = dict(latent)
        out["samples"] = samples
        out["noise_mask"] = noise_mask
        return (out,)
