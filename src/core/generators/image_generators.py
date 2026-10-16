#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generators for media_mcp.
Polymorphic strategy implementations for SDXL / Pony (A1111), FLUX.1 Dev (ComfyUI), and FLUX.2 Klein (ComfyUI).
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import random
import base64
import datetime
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from ..config import OUTPUT_IMG_DIR
from .. import comfy_client
from .. import a1111_client
from .base import BaseGenerator, ImageRequest, GenerationResult, snap_to_16


def log(msg: str):
    sys.stderr.write(f"[image-generators] {msg}\n")
    sys.stderr.flush()


_IMAGE_PROGRESS_CALLBACK = None


def set_image_progress_callback(cb):
    global _IMAGE_PROGRESS_CALLBACK
    _IMAGE_PROGRESS_CALLBACK = cb


class BaseImageGenerator(BaseGenerator):
    """Abstract base class for image generators."""

    kind = "image"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "settings": dict(self.settings)
        }


class SdxlImageGenerator(BaseImageGenerator):
    """
    SDXL / Pony / Illustrious generator communicating with Automatic1111 REST API.
    Handles negative prompts, CFG scaling, Euler a sampler, and SDXL low-res upscaling.
    Supports multiple LoRA tags dynamically appended to prompt.
    """

    service = "automatic1111"

    def generate(self, request: ImageRequest) -> GenerationResult:
        model = request.model or self.settings.get("model") or self.settings.get("checkpoint")
        current_model = a1111_client.get_current_model() or ""
        if model:
            if current_model and model.lower() not in current_model.lower():
                log(f"Switching A1111 model to: {model}")
                a1111_client.set_model(model)
                current_model = model

        model_lower = (str(model or "") + " " + str(current_model)).lower()
        is_sdxl = any(x in model_lower for x in ["xl", "pony", "illustrious", "animagine"])
        is_pony = any(x in model_lower for x in ["pony", "score_"]) or "score_" in request.prompt.lower()

        cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 7.0))
        if is_pony and abs(cfg - 7.0) < 0.01:
            cfg = 5.5

        sampler = request.sampler or self.settings.get("sampler") or self.settings.get("sampler_name")
        if not sampler and is_pony:
            sampler = "Euler a"

        target_w = snap_to_16(request.width or self.settings.get("width", 832))
        target_h = snap_to_16(request.height or self.settings.get("height", 480))

        # SDXL models require ~1MP. Generating directly at low resolutions causes attention starvation.
        if is_sdxl and (target_w * target_h < 750_000):
            aspect = target_w / max(1, target_h)
            if abs(aspect - (16.0 / 9.0)) < 0.15:
                gen_w, gen_h = 1280, 720
            else:
                scale = max(1.5, 1024.0 / max(target_w, target_h))
                gen_w = snap_to_16(int(target_w * scale))
                gen_h = snap_to_16(int(target_h * scale))
            log(f"SDXL low resolution detected ({target_w}x{target_h}). Generating at full high-res {gen_w}x{gen_h}.")
        else:
            gen_w = target_w
            gen_h = target_h

        steps = int(request.steps or self.settings.get("steps", 30))
        seed = int(request.seed) if request.seed not in (-1, None) else -1
        negative_prompt = request.negative_prompt or self.settings.get("negative_prompt", "ugly, deformed, low quality, bad anatomy")

        # Process prompt and append multiple LoRAs if requested
        prompt_text = request.prompt.strip()
        if request.loras:
            lora_tags = []
            for item in request.loras:
                l_name = item.get("name", "").strip()
                l_strength = float(item.get("strength", 0.8))
                if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                    # Strip .safetensors if passed
                    base_name = l_name[:-12] if l_name.endswith(".safetensors") else l_name
                    lora_tags.append(f"<lora:{base_name}:{l_strength}>")
            if lora_tags:
                prompt_text = f"{prompt_text} {' '.join(lora_tags)}"

        payload = {
            "prompt": prompt_text,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg,
            "width": gen_w,
            "height": gen_h,
            "seed": seed
        }
        if sampler:
            payload["sampler_name"] = sampler

        if request.is_img2img:
            if not Path(request.image_path).is_file():
                return GenerationResult(success=False, error=f"Source image not found on disk: {request.image_path}")
            with open(request.image_path, "rb") as f:
                init_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload["init_images"] = [init_b64]
            payload["denoising_strength"] = float(request.denoise)
            if request.is_inpaint and request.mask_path:
                if not Path(request.mask_path).is_file():
                    return GenerationResult(success=False, error=f"Mask image not found on disk: {request.mask_path}")
                with open(request.mask_path, "rb") as f:
                    mask_b64 = base64.b64encode(f.read()).decode("utf-8")
                payload["mask"] = mask_b64
                payload["mask_blur"] = 0
                payload["inpainting_fill"] = 1  # preserve original image under mask
                payload["inpaint_full_res"] = False
                payload["mask_round"] = False  # Prevent A1111 from binarizing continuous mask (np.around)
                if "override_settings" not in payload:
                    payload["override_settings"] = {}
                payload["override_settings"]["overlay_inpaint"] = False

        cb = _IMAGE_PROGRESS_CALLBACK
        stop_event = threading.Event()

        def poll_heartbeat():
            start_t = time.time()
            while not stop_event.wait(3.0):
                try:
                    p_data = a1111_client.get_progress()
                    prog = float(p_data.get("progress", 0.0))
                    eta = float(p_data.get("eta_relative", 0.0))
                    state = p_data.get("state", {})
                    step = state.get("sampling_step", 0)
                    total_steps = state.get("sampling_steps", 0)
                    elapsed = int(time.time() - start_t)

                    if total_steps > 0:
                        msg = f"A1111 rendering: step {step}/{total_steps} ({int(prog * 100)}%, ETA: {eta:.1f}s, elapsed: {elapsed}s)"
                    else:
                        msg = f"A1111 rendering in progress (elapsed: {elapsed}s)..."

                    if cb:
                        try:
                            cb(msg, prog, 1.0)
                        except Exception:
                            pass
                    log(msg)
                except Exception:
                    pass

        t = threading.Thread(target=poll_heartbeat, daemon=True)
        t.start()
        try:
            if request.is_img2img:
                res = a1111_client.img2img(payload)
            else:
                res = a1111_client.txt2img(payload)
            images = res.get("images", [])
            if not images:
                return GenerationResult(success=False, error="A1111 returned no images")

            raw_b64 = images[0].split(",", 1)[-1]
            raw_bytes = base64.b64decode(raw_b64)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            save_filename = request.custom_filename or f"{timestamp}-{req_hash}-final.png"
            p_save = Path(save_filename)
            final_path = (p_save if p_save.is_absolute() else (Path(OUTPUT_IMG_DIR) / p_save)).resolve().as_posix()

            with open(final_path, "wb") as f:
                f.write(raw_bytes)

            meta = {
                "engine": "sdxl",
                "mode": "inpaint" if request.is_inpaint else ("img2img" if request.is_img2img else "txt2img"),
                "model": current_model,
                "width": gen_w,
                "height": gen_h,
                "target_width": target_w,
                "target_height": target_h,
                "steps": steps,
                "cfg": cfg,
                "sampler": sampler,
                "seed": seed,
                "loras": request.loras or []
            }
            if request.is_img2img:
                meta["source_image_path"] = request.image_path
                meta["denoise"] = float(request.denoise)
            if request.is_inpaint:
                meta["mask_path"] = request.mask_path
            if request.fit:
                meta["fit"] = request.fit

            return GenerationResult(success=True, image_path=final_path, file_path=final_path, base64_data=raw_b64, metadata=meta)
        except Exception as ex:
            return GenerationResult(success=False, error=str(ex))
        finally:
            stop_event.set()
            t.join(timeout=1.0)


class Flux1ImageGenerator(BaseImageGenerator):
    """
    FLUX.1 Dev generator via ComfyUI.
    Uses DualCLIPLoader (T5XXL + CLIP-L), FluxGuidance, and standard Euler/beta sampling.
    Dynamically chains LoraLoaderModelOnly nodes for multiple LoRAs.
    Negative prompts are deliberately ignored as FLUX does not support them.
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 28))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        unet_name = request.model or self.settings.get("model") or self.settings.get("checkpoint", "persephoneFluxNSFWSFW_20FP16.safetensors")
        clip_name1 = self.settings.get("text_encoder", "t5xxl_fp16.safetensors")
        clip_name2 = self.settings.get("text_encoder_2", "clip_l.safetensors")
        vae_name = self.settings.get("vae", "ae.safetensors")
        full_prompt = request.prompt.strip()

        wf: Dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": clip_name1, "clip_name2": clip_name2, "type": "flux"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": guidance}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": full_prompt}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "media_mcp_flux1", "images": ["8", 0]}}
        }

        # Dynamic LoRA chaining
        sampler_model = ["1", 0]
        if request.loras:
            lora_node_id = 20
            for lora_entry in request.loras:
                l_name = lora_entry.get("name", "").strip()
                l_strength = float(lora_entry.get("strength", 0.8))
                if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                    lora_fn = l_name if l_name.endswith(".safetensors") else f"{l_name}.safetensors"
                    node_key = str(lora_node_id)
                    wf[node_key] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {"lora_name": lora_fn, "model": sampler_model, "strength_model": l_strength}
                    }
                    sampler_model = [node_key, 0]
                    lora_node_id += 1

        if request.is_img2img:
            if not Path(request.image_path).is_file():
                return GenerationResult(success=False, error=f"Source image not found on disk: {request.image_path}")
            try:
                comfy_input_fn = comfy_client.copy_to_comfy_input(request.image_path)
            except Exception as ex:
                return GenerationResult(success=False, error=f"Failed to prepare input image for ComfyUI: {ex}")
            wf["6"] = {"class_type": "LoadImage", "inputs": {"image": comfy_input_fn}}
            wf["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["3", 0]}}
            latent_source = ["10", 0]
            denoise_val = float(request.denoise)
            if request.is_inpaint and request.mask_path:
                if not Path(request.mask_path).is_file():
                    return GenerationResult(success=False, error=f"Mask image not found on disk: {request.mask_path}")
                try:
                    comfy_mask_fn = comfy_client.copy_to_comfy_input(request.mask_path)
                except Exception as ex:
                    return GenerationResult(success=False, error=f"Failed to prepare input mask for ComfyUI: {ex}")
                wf["12"] = {"class_type": "LoadImageMask", "inputs": {"image": comfy_mask_fn, "channel": "red"}}
                wf["13"] = {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["10", 0], "mask": ["12", 0]}
                }
                latent_source = ["13", 0]
        else:
            latent_source = ["6", 0]
            denoise_val = 1.0

        wf["7"] = {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 1.0, "denoise": denoise_val, "latent_image": latent_source,
                "model": sampler_model, "negative": ["5", 0], "positive": ["4", 0],
                "sampler_name": request.sampler or "euler", "scheduler": "beta", "seed": seed, "steps": steps
            }
        }

        log(f"Queuing FLUX.1 Dev image generation: '{full_prompt[:60]}...' ({width}x{height})")
        ok, prompt_id = comfy_client.queue_prompt(wf)
        if not ok:
            return GenerationResult(success=False, error=prompt_id)

        ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=300)
        if not ok:
            return GenerationResult(success=False, error=str(history))

        outputs = history.get("outputs", {})
        images_meta = []
        for node_out in outputs.values():
            if "images" in node_out:
                images_meta.extend(node_out["images"])

        if not images_meta:
            return GenerationResult(success=False, error="No output images produced by ComfyUI")

        first_img = images_meta[0]
        raw_bytes = comfy_client.download_output_file(
            filename=first_img.get("filename"),
            subfolder=first_img.get("subfolder", ""),
            file_type=first_img.get("type", "output")
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        req_hash = uuid.uuid4().hex[:6]
        save_filename = request.custom_filename or f"{timestamp}-{req_hash}-final.png"
        p_save = Path(save_filename)
        final_path = (p_save if p_save.is_absolute() else (Path(OUTPUT_IMG_DIR) / p_save)).resolve().as_posix()

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        meta = {
            "engine": "flux1",
            "mode": "inpaint" if request.is_inpaint else ("img2img" if request.is_img2img else "txt2img"),
            "model": unet_name,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "sampler": request.sampler or "euler",
            "loras": request.loras or []
        }
        if request.is_img2img:
            meta["source_image_path"] = request.image_path
            meta["denoise"] = float(request.denoise)
        if request.is_inpaint:
            meta["mask_path"] = request.mask_path
        if request.fit:
            meta["fit"] = request.fit

        return GenerationResult(success=True, image_path=final_path, file_path=final_path, base64_data=b64_data, metadata=meta)


class Flux2ImageGenerator(BaseImageGenerator):
    """
    FLUX.2 Klein generator via ComfyUI.
    Uses Qwen CLIPLoader, Flux2Scheduler, SamplerCustomAdvanced, and prepends 'flux2, uncensored, '.
    Incompatible with FLUX.1 LoRAs (safeguard enforced).
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        if request.loras:
            return GenerationResult(
                success=False,
                error="FLUX.2 Klein does not support FLUX.1 LoRAs (architecture and tensor shape mismatch)."
            )

        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 30))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        unet_name = request.model or self.settings.get("model") or self.settings.get("checkpoint", "aishaNSFWFlux2Klein_betaV88.safetensors")
        clip_name = self.settings.get("text_encoder", "qwen_3_8b_fp8mixed.safetensors")
        vae_name = self.settings.get("vae", "flux2-vae.safetensors")
        prompt_prefix = "flux2, uncensored, "
        full_prompt = f"{prompt_prefix}{request.prompt}".strip()

        wf: Dict[str, Any] = {
            "1001": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "1002": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "flux2"}},
            "1004": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1002", 0], "text": full_prompt}},
            "1025": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["1004", 0], "guidance": guidance}},
            "1006": {"class_type": "BasicGuider", "inputs": {"model": ["1001", 0], "conditioning": ["1025", 0]}},
            "1007": {"class_type": "EmptySD3LatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "1022": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "1023": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": request.sampler or "euler"}},
            "1024": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
            "1010": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "1009": {"class_type": "VAEDecode", "inputs": {"samples": ["1021", 0], "vae": ["1010", 0]}},
            "1011": {"class_type": "SaveImage", "inputs": {"filename_prefix": "media_mcp_flux2", "images": ["1009", 0]}}
        }

        if request.is_img2img:
            if not Path(request.image_path).is_file():
                return GenerationResult(success=False, error=f"Source image not found on disk: {request.image_path}")
            try:
                comfy_input_fn = comfy_client.copy_to_comfy_input(request.image_path)
            except Exception as ex:
                return GenerationResult(success=False, error=f"Failed to prepare input image for ComfyUI: {ex}")
            wf["1007"] = {"class_type": "LoadImage", "inputs": {"image": comfy_input_fn}}
            wf["1008"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1007", 0], "vae": ["1010", 0]}}
            latent_source = ["1008", 0]
            wf["1026"] = {
                "class_type": "SplitSigmasDenoise",
                "inputs": {
                    "sigmas": ["1024", 0],
                    "denoise": float(request.denoise)
                }
            }
            sigmas_source = ["1026", 1]
            if request.is_inpaint and request.mask_path:
                if not Path(request.mask_path).is_file():
                    return GenerationResult(success=False, error=f"Mask image not found on disk: {request.mask_path}")
                try:
                    comfy_mask_fn = comfy_client.copy_to_comfy_input(request.mask_path)
                except Exception as ex:
                    return GenerationResult(success=False, error=f"Failed to prepare input mask for ComfyUI: {ex}")
                wf["1012"] = {"class_type": "LoadImageMask", "inputs": {"image": comfy_mask_fn, "channel": "red"}}
                wf["1013"] = {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {"samples": ["1008", 0], "mask": ["1012", 0]}
                }
                latent_source = ["1013", 0]
        else:
            latent_source = ["1007", 0]
            sigmas_source = ["1024", 0]

        wf["1021"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["1022", 0], "guider": ["1006", 0], "sampler": ["1023", 0],
                "sigmas": sigmas_source, "latent_image": latent_source
            }
        }

        log(f"Queuing FLUX.2 Klein image generation: '{full_prompt[:60]}...' ({width}x{height})")
        ok, prompt_id = comfy_client.queue_prompt(wf)
        if not ok:
            return GenerationResult(success=False, error=prompt_id)

        ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=300)
        if not ok:
            return GenerationResult(success=False, error=str(history))

        outputs = history.get("outputs", {})
        images_meta = []
        for node_out in outputs.values():
            if "images" in node_out:
                images_meta.extend(node_out["images"])

        if not images_meta:
            return GenerationResult(success=False, error="No output images produced by ComfyUI")

        first_img = images_meta[0]
        raw_bytes = comfy_client.download_output_file(
            filename=first_img.get("filename"),
            subfolder=first_img.get("subfolder", ""),
            file_type=first_img.get("type", "output")
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        req_hash = uuid.uuid4().hex[:6]
        save_filename = request.custom_filename or f"{timestamp}-{req_hash}-final.png"
        p_save = Path(save_filename)
        final_path = (p_save if p_save.is_absolute() else (Path(OUTPUT_IMG_DIR) / p_save)).resolve().as_posix()

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        meta = {
            "engine": "flux2",
            "mode": "inpaint" if request.is_inpaint else ("img2img" if request.is_img2img else "txt2img"),
            "model": unet_name,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "sampler": request.sampler or "euler"
        }
        if request.is_img2img:
            meta["source_image_path"] = request.image_path
            meta["denoise"] = float(request.denoise)
        if request.is_inpaint:
            meta["mask_path"] = request.mask_path
        if request.fit:
            meta["fit"] = request.fit

        return GenerationResult(success=True, image_path=final_path, file_path=final_path, base64_data=b64_data, metadata=meta)
