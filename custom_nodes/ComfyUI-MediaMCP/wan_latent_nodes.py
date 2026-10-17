import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import node_helpers

class WanLatentPrefixToVideo:
    """
    Unified Wan 2.2 Conditioning node supporting:
    1. Direct LATENT prefix chaining (zero-loss temporal continuity, no VAE re-encoding).
    2. Optional single/multi IMAGE start input (fallback for Block 1 anchor image).
    Fills future ungenerated frames with standard flat 0.5 gray image encoded by VAE (matching Wan 2.2 training).
    Sets concat_mask = 0.0 for all prefix frames (conditioned) and 1.0 for newly diffused frames.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 61, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "prefix_latent": ("LATENT",),
                "start_image": ("IMAGE",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/wan"

    def execute(self, positive, negative, vae, width, height, length, batch_size,
                prefix_latent=None, start_image=None, clip_vision_output=None):
        device = comfy.model_management.intermediate_device()
        t_latents = ((length - 1) // 4) + 1
        h_lat = height // 8
        w_lat = width // 8

        if prefix_latent is not None and "samples" in prefix_latent:
            # Mode A: Direct Latent Prefix (True Zero-Loss Conditioning)
            src_samples = prefix_latent["samples"].to(device=device)
            # In ComfyUI (samplers.py:1238), KSampler already calls process_latent_out() before returning samples.
            # Therefore, src_samples from SaveLatent is ALREADY in native raw VAE space.
            # Do NOT call process_out() here, as doing so doubles the variance and blows out colors/contrast.
            prefix_vae = src_samples
            k_lat = min(prefix_vae.shape[2], t_latents)

            # Ensure spatial dimensions match
            if prefix_vae.shape[-2:] != (h_lat, w_lat):
                prefix_vae = comfy.utils.common_upscale(prefix_vae, w_lat, h_lat, "bilinear", "center")

            # Wan I2V conditioning padding:
            # Wan I2V was trained with flat 0.5 gray image encoded by VAE for unconditioned frames.
            gray_img = torch.ones((length, height, width, 3), device="cpu", dtype=torch.float32) * 0.5
            concat_latent = vae.encode(gray_img[:, :, :, :3]).to(device=device)
            concat_latent[:, :, :k_lat] = prefix_vae[:batch_size, :, :k_lat].to(dtype=concat_latent.dtype)

            mask = torch.ones((batch_size, 1, t_latents, h_lat, w_lat), device=device, dtype=torch.float32)
            mask[:, :, :k_lat] = 0.0

            # Clean unmasked diffusion latents matching native Wan I2V conditioning:
            out_latent = {
                "samples": torch.zeros([batch_size, 16, t_latents, h_lat, w_lat], device=device, dtype=torch.float32)
            }
        elif start_image is not None:
            # Mode B: Image Input (for Block 1 initial anchor frame)
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image
            encoded_image_latent = vae.encode(image[:, :, :, :3]).to(device=device)
            concat_latent = encoded_image_latent.repeat(batch_size, 1, 1, 1, 1)
            k_masked = ((start_image.shape[0] - 1) // 4) + 1
            mask = torch.ones((batch_size, 1, t_latents, h_lat, w_lat), device=device, dtype=torch.float32)
            mask[:, :, :k_masked] = 0.0
            out_latent = {"samples": torch.zeros([batch_size, 16, t_latents, h_lat, w_lat], device=device, dtype=torch.float32)}
        else:
            concat_latent = torch.zeros([batch_size, 16, t_latents, h_lat, w_lat], device=device, dtype=torch.float32)
            mask = torch.ones((batch_size, 1, t_latents, h_lat, w_lat), device=device, dtype=torch.float32)
            out_latent = {"samples": torch.zeros([batch_size, 16, t_latents, h_lat, w_lat], device=device, dtype=torch.float32)}

        pos_out = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent, "concat_mask": mask})
        neg_out = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent, "concat_mask": mask})

        if clip_vision_output is not None:
            pos_out = node_helpers.conditioning_set_values(pos_out, {"clip_vision_output": clip_vision_output})
            neg_out = node_helpers.conditioning_set_values(neg_out, {"clip_vision_output": clip_vision_output})

        return (pos_out, neg_out, out_latent)


class WanLatentPrefixLock:
    """
    Native Diffusion Forcing & Boundary Motion Smoothing for Wan 2.2 (Option B).
    1. Temporal CFG Ramp-Up (sampler_cfg_function):
       Eliminates "prompt shock" and boundary motion tear at the seam between blocks.
       - Prefix frames (0..k-1): CFG = min_cfg (1.0, zero text guidance delta).
       - Ramp frames (k..k+ramp_latents-1): Smooth linear ramp from min_cfg to full cond_scale.
       - Core frames (k+ramp_latents..T): Full target CFG (e.g. 6.0) for vivid action.
    2. Pristine Latent Clamping (sampler_post_cfg_function):
       Clamps x_0 prediction on prefix frames 0..k-1 to pristine prefix latents during diffusion.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prefix_latent": ("LATENT",),
                "ramp_latents": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "min_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/wan"

    def execute(self, model, prefix_latent, ramp_latents=3, min_cfg=1.0):
        m = model.clone()
        src_samples = prefix_latent["samples"]  # [B, C, T, H, W] in raw VAE space

        # 1. Temporal CFG Ramp-Up
        if ramp_latents > 0 or min_cfg != 1.0:
            def prefix_cfg_ramp(args):
                cond = args["cond"]        # [B, C, T, H, W]
                uncond = args["uncond"]    # [B, C, T, H, W]
                cond_scale = args["cond_scale"]

                if cond.ndim != 5:
                    return uncond + cond_scale * (cond - uncond)

                B, C, T, H, W = cond.shape
                scale = torch.full((1, 1, T, 1, 1), cond_scale, device=cond.device, dtype=cond.dtype)

                k = min(src_samples.shape[2], T)
                if k > 0:
                    scale[:, :, :k, :, :] = min_cfg

                if ramp_latents > 0 and k < T:
                    ramp_end = min(k + ramp_latents, T)
                    ramp_steps = ramp_end - k
                    ramp_vals = torch.linspace(min_cfg, cond_scale, ramp_steps + 1, device=cond.device, dtype=cond.dtype)
                    scale[:, :, k:ramp_end, :, :] = ramp_vals[:ramp_steps].view(1, 1, ramp_steps, 1, 1)

                return uncond + scale * (cond - uncond)

            m.set_model_sampler_cfg_function(prefix_cfg_ramp)

        # 2. Pristine Latent Clamping (Diffusion Forcing)
        def prefix_clamp_hook(args):
            denoised = args["denoised"]
            inner_model = args["model"]

            # Convert raw VAE samples to model's normalized diffusion space on current device/dtype
            p_samples = src_samples.to(device=denoised.device, dtype=denoised.dtype)
            if p_samples.shape[-2:] != (denoised.shape[-2], denoised.shape[-1]):
                p_samples = comfy.utils.common_upscale(
                    p_samples, denoised.shape[-1], denoised.shape[-2], "bilinear", "center"
                )

            p_norm = inner_model.process_latent_in(p_samples)
            k = min(p_norm.shape[2], denoised.shape[2])
            b = min(p_norm.shape[0], denoised.shape[0])

            out_denoised = denoised.clone()
            if out_denoised.shape[2] > k and k > 0:
                pref_mean = p_norm[:b, :, :k].mean(dim=(2, 3, 4), keepdim=True)
                pref_std = p_norm[:b, :, :k].std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)
                core_mean = out_denoised[:b, :, k:].mean(dim=(2, 3, 4), keepdim=True)
                core_std = out_denoised[:b, :, k:].std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)
                out_denoised[:b, :, k:] = (out_denoised[:b, :, k:] - core_mean) / core_std * pref_std + pref_mean

            out_denoised[:b, :, :k] = p_norm[:b, :, :k]
            return out_denoised

        m.set_model_sampler_post_cfg_function(prefix_clamp_hook)
        return (m,)


class WanLatentPrefixRestore:
    """
    Ensures bit-exact prefix latent frames on KSampler output before VAE decode.
    Replaces frames 0..k_lat with the pristine raw VAE prefix latent.
    Applies Latent AdaIN (Adaptive Instance Normalization) to newly diffused core frames
    to strictly prevent channel drift, contrast blow-out, and color desaturation across chained blocks.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "prefix_latent": ("LATENT",),
            },
            "optional": {
                "color_lock_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/wan"

    def execute(self, samples, prefix_latent, color_lock_strength=1.0):
        out = {k: v for k, v in samples.items()}
        s = samples["samples"].clone()
        p = prefix_latent["samples"].to(device=s.device, dtype=s.dtype)
        if p.shape[-2:] != (s.shape[-2], s.shape[-1]):
            p = comfy.utils.common_upscale(p, s.shape[-1], s.shape[-2], "bilinear", "center")
        k = min(p.shape[2], s.shape[2])
        b = min(p.shape[0], s.shape[0])

        # Apply Latent AdaIN to prevent channel mean drift and variance explosion across blocks
        if color_lock_strength > 0 and s.shape[2] > k and k > 0:
            pref_mean = p[:b, :, :k].mean(dim=(2, 3, 4), keepdim=True)
            pref_std = p[:b, :, :k].std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

            core_mean = s[:b, :, k:].mean(dim=(2, 3, 4), keepdim=True)
            core_std = s[:b, :, k:].std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-5)

            normalized_core = (s[:b, :, k:] - core_mean) / core_std * pref_std + pref_mean
            s[:b, :, k:] = s[:b, :, k:] * (1.0 - color_lock_strength) + normalized_core * color_lock_strength

        # Restore pristine prefix latents bit-for-bit
        s[:b, :, :k] = p[:b, :, :k]
        out["samples"] = s
        return (out,)
