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
            # KSampler outputs normalized diffusion latents.
            # Convert to raw VAE space so DiT's self.process_latent_in() reconstructs it with machine precision:
            latent_format = comfy.latent_formats.Wan21()
            prefix_vae = latent_format.process_out(src_samples)
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

            # KSampler initialization: lock prefix frames and guide diffusion via noise_mask
            latent_samples = torch.zeros([batch_size, 16, t_latents, h_lat, w_lat], device=device, dtype=torch.float32)
            latent_samples[:, :, :k_lat] = src_samples[:batch_size, :, :k_lat].to(dtype=latent_samples.dtype)
            out_latent = {
                "samples": latent_samples,
                "noise_mask": mask
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
