#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Engine for media_mcp.
Supports FLUX.1 Dev, FLUX.2 Klein (ComfyUI) and SD 1.5, SDXL, Pony, Illustrious (A1111).
Returns local file path, metadata, and base64 representation.
"""

import os
import sys
import uuid
import time
import base64
import random
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .config import OUTPUT_IMG_DIR, OUTPUT_DIR
from . import comfy_client
from . import a1111_client

def log(msg: str):
    sys.stderr.write(f"[image-engine] {msg}\n")
    sys.stderr.flush()

_GLOBAL_PROGRESS_CALLBACK = None

def set_global_progress_callback(cb):
    global _GLOBAL_PROGRESS_CALLBACK
    _GLOBAL_PROGRESS_CALLBACK = cb

def snap_to_16(dim: int, default: int = 1024) -> int:
    try:
        val = int(dim)
        return max(64, (val // 16) * 16)
    except Exception:
        return default

# ---------------------------------------------------------------------------
# FLUX.1 & FLUX.2 Klein generation via ComfyUI
# ---------------------------------------------------------------------------

def generate_flux_image(
    prompt: str,
    model_type: str = "flux1_dev",
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    guidance: float = 3.5,
    seed: int = -1,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Generates an image via FLUX.1 or FLUX.2 Klein in ComfyUI.
    Returns: (success, file_path, base64_data, metadata)
    """
    width = snap_to_16(width, 1024)
    height = snap_to_16(height, 1024)
    seed = random.randint(1, 10**14) if seed in (-1, None) else int(seed)

    is_flux2 = (model_type.lower() == "flux2")
    steps = steps or (30 if is_flux2 else 28)

    # Prevent FLUX.1 LoRA on FLUX.2 Klein
    if is_flux2 and lora_name:
        if any(x in str(lora_name).lower() for x in ["flux-v0.2", "flux1", "aidmansfwunlock"]):
            log(f"Warning: FLUX.1 LoRA '{lora_name}' is incompatible with FLUX.2 Klein. Disabling.")
            lora_name = None

    if is_flux2:
        unet_name = "aishaNSFWFlux2Klein_betaV88.safetensors"
        clip_name = "qwen_3_8b_fp8mixed.safetensors"
        vae_name = "flux2-vae.safetensors"
        prompt_prefix = "flux2, uncensored, "
        full_prompt = f"{prompt_prefix}{prompt}".strip()

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
            "1011": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"media_mcp_flux_{model_type}", "images": ["1009", 0]}}
        }
    else:
        unet_name = "persephoneFluxNSFWSFW_20FP16.safetensors"
        clip_name1 = "t5xxl_fp16.safetensors"
        clip_name2 = "clip_l.safetensors"
        vae_name = "ae.safetensors"
        full_prompt = prompt.strip()

        wf: Dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": clip_name1, "clip_name2": clip_name2, "type": "flux"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": guidance}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": full_prompt}},
            "6": {"class_type": "EmptySD3LatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": f"media_mcp_flux_{model_type}", "images": ["8", 0]}}
        }

        sampler_model = ["1", 0]
        if lora_name and str(lora_name).lower() not in ("none", "disabled", "false", ""):
            lora_fn = lora_name if lora_name.endswith(".safetensors") else f"{lora_name}.safetensors"
            wf["10"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": lora_fn, "model": ["1", 0], "strength_model": float(lora_strength)}
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

    log(f"Queuing FLUX ({model_type.upper()}) image generation: '{full_prompt[:60]}...' ({width}x{height})")
    ok, prompt_id = comfy_client.queue_prompt(wf)
    if not ok:
        return False, None, None, {"error": prompt_id}

    ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=300)
    if not ok:
        return False, None, None, history

    outputs = history.get("outputs", {})
    images_meta = []
    for node_out in outputs.values():
        if "images" in node_out:
            images_meta.extend(node_out["images"])

    if not images_meta:
        return False, None, None, {"error": "No output images produced by ComfyUI"}

    first_img = images_meta[0]
    raw_bytes = comfy_client.download_output_file(
        filename=first_img.get("filename"),
        subfolder=first_img.get("subfolder", ""),
        file_type=first_img.get("type", "output")
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    req_uuid = uuid.uuid4().hex[:8]
    save_filename = f"flux_{model_type}_{timestamp}_{req_uuid}.png"
    final_path = os.path.join(OUTPUT_IMG_DIR, save_filename)

    with open(final_path, "wb") as f:
        f.write(raw_bytes)

    b64_data = base64.b64encode(raw_bytes).decode("utf-8")
    meta = {
        "model": model_type,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "lora": lora_name
    }
    return True, final_path, b64_data, meta


# ---------------------------------------------------------------------------
# SD 1.5, SDXL, Pony generation via Automatic1111
# ---------------------------------------------------------------------------

def generate_a1111_image(
    prompt: str,
    negative_prompt: str = "ugly, deformed, low quality, bad anatomy",
    model_name: Optional[str] = None,
    width: int = 832,
    height: int = 480,
    steps: int = 30,
    cfg_scale: float = 7.0,
    sampler_name: Optional[str] = None,
    seed: int = -1,
    highres_fix: bool = True
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Generates an image via Automatic1111 WebUI.
    Automatically handles high-res generation and Lanczos downscaling for SDXL/Pony models
    when target resolution is below SDXL optimal dimensions (e.g. 832x480).
    Returns: (success, file_path, base64_data, metadata)
    """
    current_model = a1111_client.get_current_model() or ""
    if model_name:
        if current_model and model_name.lower() not in current_model.lower():
            log(f"Switching A1111 model to: {model_name}")
            a1111_client.set_model(model_name)
            current_model = model_name

    model_lower = (str(model_name or "") + " " + str(current_model)).lower()
    is_sdxl = any(x in model_lower for x in ["xl", "pony", "illustrious", "animagine"])
    is_pony = any(x in model_lower for x in ["pony", "score_"]) or "score_" in prompt.lower()

    # Pony character tag sanitization:
    # In Pony Diffusion / Danbooru, '1girl' strictly means a human female girl.
    # Combining '1girl' with 'unicorn/pony' produces an uncanny human girl face pasted onto horse legs.
    is_mlp_pony = any(x in prompt.lower() for x in ["pony", "unicorn", "pegasus", "alicorn", "mare", "stallion", "filly", "colt"])
    if is_pony and is_mlp_pony:
        import re
        prompt = re.sub(r'\b1girl\b,?\s*', '', prompt, flags=re.IGNORECASE).strip()
        if "quadruped" not in prompt.lower() and "feral" not in prompt.lower():
            prompt = f"{prompt}, quadruped, feral pony, cute pony muzzle, mlp:fim style"
        # Enforce anti-human & anti-anthro tags in negative prompt
        for neg_tag in ["anthro", "human", "1girl", "humanoid face", "human face", "bipedal"]:
            if neg_tag not in negative_prompt.lower():
                negative_prompt = f"{negative_prompt}, {neg_tag}"
        log("Pony character detected: removed '1girl', enforced quadruped & anti-human tags")

    # Pony XL models overbake and produce burnt contrast/halos at CFG >= 7.0.
    # Optimal CFG for Pony XL is 5.0 - 5.5.
    if is_pony and abs(cfg_scale - 7.0) < 0.01:
        cfg_scale = 5.5
        log(f"Auto-adjusting Pony CFG scale to {cfg_scale} to prevent overbaking")

    # Pony-friendly negative prompt if default generic one is provided
    if is_pony and negative_prompt == "ugly, deformed, low quality, bad anatomy":
        negative_prompt = "score_4, score_5, score_6, worst quality, low quality, bad anatomy, bad hands, deformed, extra limbs, bipedal, anthro, human, 1girl, blurry"

    if not sampler_name and is_pony:
        sampler_name = "Euler a"

    # High-resolution generation for SDXL/Pony:
    # SDXL models require ~1MP (1280x720, 1024x1024). Generating directly at low resolutions (< 750k pixels)
    # causes attention starvation, mangled anatomy, and weird limbs.
    # When low resolution (e.g. 832x480) is requested, we scale up to full HD (1280x720) and KEEP the high-res file!
    target_w = snap_to_16(width)
    target_h = snap_to_16(height)

    if highres_fix and is_sdxl and (target_w * target_h < 750_000):
        # Scale to 1280x720 for 16:9, preserving aspect ratio
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

    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": gen_w,
        "height": gen_h,
        "seed": seed
    }
    if sampler_name:
        payload["sampler_name"] = sampler_name

    cb = globals().get("_GLOBAL_PROGRESS_CALLBACK", None)

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
            return False, None, None, {"error": "A1111 returned no images"}

        raw_b64 = images[0].split(",", 1)[-1]
        raw_bytes = base64.b64decode(raw_b64)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        req_uuid = uuid.uuid4().hex[:8]
        save_filename = f"a1111_{timestamp}_{req_uuid}.png"
        final_path = os.path.join(OUTPUT_IMG_DIR, save_filename)

        with open(final_path, "wb") as f:
            f.write(raw_bytes)

        return True, final_path, raw_b64, {
            "width": gen_w,
            "height": gen_h,
            "target_width": target_w,
            "target_height": target_h,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
    except Exception as ex:
        return False, None, None, {"error": str(ex)}
    finally:
        stop_event.set()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Universal Dispatcher (Used by video pipeline for Anchor Keyframes)
# ---------------------------------------------------------------------------

def generate_anchor_frame(anchor_meta: Dict[str, Any], width: int = 832, height: int = 480) -> Tuple[bool, str, Optional[str]]:
    """
    Generates or verifies an anchor keyframe for a scene.
    Supports:
    1. Ready image path: {"image_path": "G:/..."} -> validates and returns existing image + base64.
    2. Prompt-based generation: {"prompt": "...", "model": "...", "lora": "..."} -> generates on the fly.
    Returns: (success, file_path, base64_data)
    """
    # CASE 1: Existing image file provided
    img_path = anchor_meta.get("image_path") or anchor_meta.get("start_image") or anchor_meta.get("source_image")
    if img_path:
        p = Path(img_path)
        if p.exists() and p.is_file():
            log(f"Anchor keyframe already exists: {p}")
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return True, str(p.resolve()), b64
            except Exception as ex:
                log(f"Warning: could not read existing anchor image base64: {ex}")
                return True, str(p.resolve()), None
        else:
            log(f"Specified anchor image path not found: {img_path}. Falling back to prompt generation.")

    # CASE 2: Generate via prompt
    prompt = anchor_meta.get("prompt", "")
    if not prompt:
        return False, "Neither valid 'image_path' nor 'prompt' provided for anchor frame", None

    model = anchor_meta.get("model") or anchor_meta.get("model_alias") or "flux1_dev"
    backend = anchor_meta.get("backend", "auto").lower()
    lora = anchor_meta.get("lora") or anchor_meta.get("lora_alias")
    lora_strength = float(anchor_meta.get("lora_strength", 0.8))

    use_a1111 = (backend == "a1111") or (backend == "auto" and any(x in str(model).lower() for x in ["pony", "sdxl", "illustrious", "animagine"]))

    if use_a1111:
        cfg = float(anchor_meta.get("cfg_scale", anchor_meta.get("cfg", 5.5 if "pony" in str(model).lower() else 7.0)))
        sampler = anchor_meta.get("sampler_name", anchor_meta.get("sampler", "Euler a" if "pony" in str(model).lower() else None))
        steps = int(anchor_meta.get("steps", 30))
        seed = int(anchor_meta.get("seed", -1))
        neg = anchor_meta.get("negative_prompt", "ugly, deformed, low quality, bad anatomy")

        ok, path, b64, err = generate_a1111_image(
            prompt=prompt,
            negative_prompt=neg,
            model_name=model,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg,
            sampler_name=sampler,
            seed=seed,
            highres_fix=bool(anchor_meta.get("highres_fix", True))
        )
        if not ok:
            return False, f"A1111 anchor generation failed: {err.get('error')}", None
        return True, path, b64
    else:
        m_type = "flux2" if ("flux2" in str(model).lower() or "klein" in str(model).lower()) else "flux1_dev"
        ok, path, b64, err = generate_flux_image(
            prompt=prompt,
            model_type=m_type,
            width=width,
            height=height,
            lora_name=lora,
            lora_strength=lora_strength
        )
        if not ok:
            return False, f"FLUX anchor generation failed: {err.get('error')}", None
        return True, path, b64
