#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generators for media_mcp.
Polymorphic strategy implementations for SDXL / Pony (A1111), FLUX.1 Dev (ComfyUI), and FLUX.2 Klein (ComfyUI).
"""

from __future__ import annotations
import json
import sys
import time
import uuid
import random
import base64
import datetime
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config
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
    Handles negative prompts, CFG scaling, Euler a sampler, and dynamic LoRA tags.
    """

    service = "automatic1111"

    def generate(self, request: ImageRequest) -> GenerationResult:
        model = str(request.model or self.settings.get("model") or self.settings.get("checkpoint") or "").strip()
        if not model:
            return GenerationResult(success=False, error="Parameter 'model' is required for SDXL image generation.")

        current_model = a1111_client.get_current_model() or ""
        if not current_model or model.lower() not in current_model.lower():
            log(f"Switching A1111 model to: {model}")
            if not a1111_client.set_model(model):
                return GenerationResult(
                    success=False,
                    error=f"Failed to switch Automatic1111 checkpoint to '{model}'."
                )
            current_model = model

        model_lower = (model + " " + current_model).lower()
        is_pony = any(x in model_lower for x in ("pony", "score_")) or "score_" in request.prompt.lower()

        cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 7.0))
        if is_pony and abs(cfg - 7.0) < 0.01:
            cfg = 5.5

        sampler = request.sampler or self.settings.get("sampler") or self.settings.get("sampler_name")
        if not sampler and is_pony:
            sampler = "Euler a"

        gen_w = snap_to_16(request.width or self.settings.get("width", 1024))
        gen_h = snap_to_16(request.height or self.settings.get("height", 1024))

        steps = int(request.steps or self.settings.get("steps", 30))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        negative_prompt = request.negative_prompt or self.settings.get("negative_prompt", "ugly, deformed, low quality, bad anatomy")

        # Process prompt and append multiple LoRAs if requested
        prompt_text = request.prompt.strip()
        if request.loras:
            lora_tags = []
            for item in request.loras:
                l_name = item.get("name", "").strip()
                l_strength = float(item.get("strength", 0.8))
                if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                    base_name = l_name[:-12] if l_name.endswith(".safetensors") else l_name
                    lora_tags.append(f"<lora:{base_name}:{l_strength}>")
            if lora_tags:
                prompt_text = f"{prompt_text} {' '.join(lora_tags)}"

        payload: Dict[str, Any] = {
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

        cb = request.progress_callback or _IMAGE_PROGRESS_CALLBACK
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

            info_raw = res.get("info")
            if isinstance(info_raw, str) and info_raw.strip():
                try:
                    info_obj = json.loads(info_raw)
                    if isinstance(info_obj, dict) and info_obj.get("seed") is not None:
                        seed = int(info_obj["seed"])
                except Exception:
                    pass

            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            run_prefix = request.run_prefix or f"{timestamp}-{req_hash}"
            save_filename = request.custom_filename or f"{run_prefix}-final.png"
            p_save = Path(save_filename)
            out_dir = Path(config.OUTPUT_IMG_DIR) if config.OUTPUT_IMG_DIR else Path.cwd()
            final_path = (p_save if p_save.is_absolute() else (out_dir / p_save)).resolve().as_posix()

            with open(final_path, "wb") as f:
                f.write(raw_bytes)

            meta = {
                "engine": "sdxl",
                "mode": "inpaint" if request.is_inpaint else ("img2img" if request.is_img2img else "txt2img"),
                "model": current_model,
                "width": gen_w,
                "height": gen_h,
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

            return GenerationResult(success=True, image_path=final_path, file_path=final_path, metadata=meta)
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
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        unet_name = str(request.model or self.settings.get("model") or self.settings.get("checkpoint") or "").strip()
        if not unet_name:
            return GenerationResult(success=False, error="Parameter 'model' is required for FLUX.1 image generation.")

        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 28))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        req_hash = uuid.uuid4().hex[:6]
        run_prefix = request.run_prefix or f"{timestamp}-{req_hash}"

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
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{run_prefix}-flux1", "images": ["8", 0]}}
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
                comfy_input_fn = comfy_client.upload_file_to_comfy(
                    request.image_path,
                    run_prefix=run_prefix,
                    suffix="input"
                )
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
                    comfy_mask_fn = comfy_client.upload_file_to_comfy(
                        request.mask_path,
                        run_prefix=run_prefix,
                        suffix="mask"
                    )
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

        cb = request.progress_callback or _IMAGE_PROGRESS_CALLBACK
        ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=300, progress_callback=cb)
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

        save_filename = request.custom_filename or f"{run_prefix}-final.png"
        p_save = Path(save_filename)
        out_dir = Path(config.OUTPUT_IMG_DIR) if config.OUTPUT_IMG_DIR else Path.cwd()
        final_path = (p_save if p_save.is_absolute() else (out_dir / p_save)).resolve().as_posix()

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

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

        return GenerationResult(success=True, image_path=final_path, file_path=final_path, metadata=meta)


class Flux2ImageGenerator(BaseImageGenerator):
    """
    FLUX.2 Klein generator via ComfyUI.
    Uses Qwen CLIPLoader, Flux2Scheduler, and SamplerCustomAdvanced.
    Incompatible with FLUX.1 LoRAs (safeguard enforced).
    """

    service = "comfyui"

    def generate(self, request: ImageRequest) -> GenerationResult:
        unet_name = str(request.model or self.settings.get("model") or self.settings.get("checkpoint") or "").strip()
        if not unet_name:
            return GenerationResult(success=False, error="Parameter 'model' is required for FLUX.2 image generation.")

        width = snap_to_16(request.width or self.settings.get("width", 1024))
        height = snap_to_16(request.height or self.settings.get("height", 1024))
        seed = random.randint(1, 10**14) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 4))
        guidance = float(request.guidance if request.guidance is not None else self.settings.get("guidance", 3.5))

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        req_hash = uuid.uuid4().hex[:6]
        run_prefix = request.run_prefix or f"{timestamp}-{req_hash}"

        clip_name = self.settings.get("text_encoder", "qwen_3_8b_fp8mixed.safetensors")
        vae_name = self.settings.get("vae", "flux2-vae.safetensors")
        full_prompt = request.prompt.strip()

        wf: Dict[str, Any] = {
            "1001": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "1002": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "flux2"}},
            "1004": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1002", 0], "text": full_prompt}},
            "1025": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["1004", 0], "guidance": guidance}},
            "1007": {"class_type": "EmptySD3LatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "1022": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "1023": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": request.sampler or "euler"}},
            "1024": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
            "1010": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "1009": {"class_type": "VAEDecode", "inputs": {"samples": ["1021", 0], "vae": ["1010", 0]}},
            "1011": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"{run_prefix}-flux2", "images": ["1009", 0]}}
        }

        guider_model = ["1001", 0]
        if request.loras:
            lora_node_id = 1030
            for lora_entry in request.loras:
                l_name = lora_entry.get("name", "").strip()
                l_strength = float(lora_entry.get("strength", 0.8))
                if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                    lora_fn = l_name if l_name.endswith(".safetensors") else f"{l_name}.safetensors"
                    node_key = str(lora_node_id)
                    wf[node_key] = {
                        "class_type": "LoraLoaderModelOnly",
                        "inputs": {"lora_name": lora_fn, "model": guider_model, "strength_model": l_strength}
                    }
                    guider_model = [node_key, 0]
                    lora_node_id += 1

        wf["1006"] = {"class_type": "BasicGuider", "inputs": {"model": guider_model, "conditioning": ["1025", 0]}}


        if request.is_img2img:
            if not Path(request.image_path).is_file():
                return GenerationResult(success=False, error=f"Source image not found on disk: {request.image_path}")
            try:
                comfy_input_fn = comfy_client.upload_file_to_comfy(
                    request.image_path,
                    run_prefix=run_prefix,
                    suffix="input"
                )
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
                    comfy_mask_fn = comfy_client.upload_file_to_comfy(
                        request.mask_path,
                        run_prefix=run_prefix,
                        suffix="mask"
                    )
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

        cb = request.progress_callback or _IMAGE_PROGRESS_CALLBACK
        ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=300, progress_callback=cb)
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

        save_filename = request.custom_filename or f"{run_prefix}-final.png"
        p_save = Path(save_filename)
        out_dir = Path(config.OUTPUT_IMG_DIR) if config.OUTPUT_IMG_DIR else Path.cwd()
        final_path = (p_save if p_save.is_absolute() else (out_dir / p_save)).resolve().as_posix()

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

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

        return GenerationResult(success=True, image_path=final_path, file_path=final_path, metadata=meta)
