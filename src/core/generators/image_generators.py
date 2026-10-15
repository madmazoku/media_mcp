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
from typing import Any, Dict, Optional, Tuple

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
    """

    service = "automatic1111"

    def generate(self, request: ImageRequest) -> GenerationResult:
        model_name = request.model_name or self.settings.get("model") or self.settings.get("checkpoint")
        current_model = a1111_client.get_current_model() or ""
        if model_name:
            if current_model and model_name.lower() not in current_model.lower():
                log(f"Switching A1111 model to: {model_name}")
                a1111_client.set_model(model_name)
                current_model = model_name

        model_lower = (str(model_name or "") + " " + str(current_model)).lower()
        is_sdxl = any(x in model_lower for x in ["xl", "pony", "illustrious", "animagine"])
        is_pony = any(x in model_lower for x in ["pony", "score_"]) or "score_" in request.prompt.lower()

        cfg_scale = float(request.cfg_scale if request.cfg_scale is not None else self.settings.get("cfg", 7.0))
        if is_pony and abs(cfg_scale - 7.0) < 0.01:
            cfg_scale = 5.5

        sampler_name = request.sampler_name or self.settings.get("sampler") or self.settings.get("sampler_name")
        if not sampler_name and is_pony:
            sampler_name = "Euler a"

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

        payload = {
            "prompt": request.prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": gen_w,
            "height": gen_h,
            "seed": seed
        }
        if sampler_name:
            payload["sampler_name"] = sampler_name

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
            res = a1111_client.txt2img(payload)
            images = res.get("images", [])
            if not images:
                return GenerationResult(success=False, error="A1111 returned no images")

            raw_b64 = images[0].split(",", 1)[-1]
            raw_bytes = base64.b64decode(raw_b64)

            if request.custom_filename:
                save_filename = request.custom_filename if request.custom_filename.endswith(".png") else f"{request.custom_filename}.png"
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                req_hash = uuid.uuid4().hex[:6]
                save_filename = f"{timestamp}-{req_hash}-sdxl.png"
            final_path = os.path.join(OUTPUT_IMG_DIR, save_filename)

            with open(final_path, "wb") as f:
                f.write(raw_bytes)

            meta = {
                "model": current_model,
                "width": gen_w,
                "height": gen_h,
                "target_width": target_w,
                "target_height": target_h,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler": sampler_name,
                "seed": seed
            }
            return GenerationResult(success=True, file_path=final_path, base64_data=raw_b64, metadata=meta)
        except Exception as ex:
            return GenerationResult(success=False, error=str(ex))
        finally:
            stop_event.set()
            t.join(timeout=1.0)


class Flux1ImageGenerator(BaseImageGenerator):
    """
    FLUX.1 Dev generator via ComfyUI.
    Uses DualCLIPLoader (T5XXL + CLIP-L), FluxGuidance, and standard Euler/beta sampling.
    Negative prompts are deliberately ignored as FLUX does not support them.
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 28))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        unet_name = self.settings.get("checkpoint", "persephoneFluxNSFWSFW_20FP16.safetensors")
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

        sampler_model = ["1", 0]
        lora_name = request.lora_name or self.settings.get("lora")
        lora_strength = float(request.lora_strength if request.lora_strength is not None else self.settings.get("lora_strength", 0.8))

        if lora_name and str(lora_name).lower() not in ("none", "disabled", "false", ""):
            lora_fn = lora_name if lora_name.endswith(".safetensors") else f"{lora_name}.safetensors"
            wf["10"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": lora_fn, "model": ["1", 0], "strength_model": lora_strength}
            }
            sampler_model = ["10", 0]

        wf["7"] = {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 1.0, "denoise": 1.0, "latent_image": ["6", 0],
                "model": sampler_model, "negative": ["5", 0], "positive": ["4", 0],
                "sampler_name": "euler", "scheduler": "beta", "seed": seed, "steps": steps
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

        if request.custom_filename:
            save_filename = request.custom_filename if request.custom_filename.endswith(".png") else f"{request.custom_filename}.png"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            save_filename = f"{timestamp}-{req_hash}-flux1.png"
        final_path = os.path.join(OUTPUT_IMG_DIR, save_filename)

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        meta = {
            "model": "flux1_dev",
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "lora": lora_name
        }
        return GenerationResult(success=True, file_path=final_path, base64_data=b64_data, metadata=meta)


class Flux2ImageGenerator(BaseImageGenerator):
    """
    FLUX.2 Klein generator via ComfyUI.
    Uses Qwen CLIPLoader, Flux2Scheduler, SamplerCustomAdvanced, and prepends 'flux2, uncensored, '.
    Incompatible with FLUX.1 LoRAs (safeguard enforced).
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 30))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        unet_name = self.settings.get("checkpoint", "aishaNSFWFlux2Klein_betaV88.safetensors")
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
            "1023": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "1024": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
            "1021": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["1022", 0], "guider": ["1006", 0], "sampler": ["1023", 0],
                "sigmas": ["1024", 0], "latent_image": ["1007", 0]
            }},
            "1010": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "1009": {"class_type": "VAEDecode", "inputs": {"samples": ["1021", 0], "vae": ["1010", 0]}},
            "1011": {"class_type": "SaveImage", "inputs": {"filename_prefix": "media_mcp_flux2", "images": ["1009", 0]}}
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

        if request.custom_filename:
            save_filename = request.custom_filename if request.custom_filename.endswith(".png") else f"{request.custom_filename}.png"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            save_filename = f"{timestamp}-{req_hash}-flux2.png"
        final_path = os.path.join(OUTPUT_IMG_DIR, save_filename)

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        meta = {
            "model": "flux2",
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance": guidance
        }
        return GenerationResult(success=True, file_path=final_path, base64_data=b64_data, metadata=meta)
