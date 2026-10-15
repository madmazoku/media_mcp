#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Engine for media_mcp.
Routes requests through the polymorphic Generator Strategy & Factory pattern.
Supports FLUX.1 Dev, FLUX.2 Klein (ComfyUI) and SD 1.5, SDXL, Pony, Illustrious (A1111).
Returns local file path, metadata, and base64 representation.
"""

import os
import sys
import shutil
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .config import OUTPUT_IMG_DIR
from .generators import (
    create_image_generator,
    ImageRequest,
    GenerationResult,
    snap_to_16,
    set_image_progress_callback,
)


def log(msg: str):
    sys.stderr.write(f"[image-engine] {msg}\n")
    sys.stderr.flush()


def set_global_progress_callback(cb):
    set_image_progress_callback(cb)


def generate_flux_image(
    prompt: str,
    model_type: str = "flux1_dev",
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    guidance: float = 3.5,
    seed: int = -1,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    custom_filename: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Generates an image via FLUX.1 or FLUX.2 Klein using the Generator Strategy pattern.
    Returns: (success, file_path, base64_data, metadata)
    """
    engine_name = "flux2" if str(model_type).lower() == "flux2" else "flux1"
    generator = create_image_generator(engine_name)
    req = ImageRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        lora_name=lora_name,
        lora_strength=lora_strength,
        custom_filename=custom_filename
    )
    res = generator.generate(req)
    if not res.success:
        return False, None, None, {"error": res.error or "FLUX generation failed"}
    return True, res.file_path, res.base64_data, res.metadata


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
    highres_fix: bool = True,
    custom_filename: Optional[str] = None
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Generates an image via Automatic1111 WebUI using SdxlImageGenerator.
    Automatically handles high-res generation for SDXL/Pony when target resolution < 750k px.
    Returns: (success, file_path, base64_data, metadata)
    """
    generator = create_image_generator("sdxl")
    req = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_name=model_name,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler_name,
        seed=seed,
        highres_fix=highres_fix,
        custom_filename=custom_filename
    )
    res = generator.generate(req)
    if not res.success:
        return False, None, None, {"error": res.error or "A1111 generation failed"}
    return True, res.file_path, res.base64_data, res.metadata


def generate_anchor_frame(
    anchor_meta: Dict[str, Any],
    width: int = 832,
    height: int = 480,
    custom_filename: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Generates or verifies an anchor keyframe for a scene.
    Supports:
    1. Ready image path: {"image_path": "G:/..."} -> validates and returns existing image + base64.
    2. Prompt-based generation: polymorphic dispatch via create_image_generator.
    Returns: (success, file_path, base64_data)
    """
    # CASE 1: Existing image file provided
    img_path = anchor_meta.get("image_path") or anchor_meta.get("start_image") or anchor_meta.get("source_image")
    if img_path:
        p = Path(img_path)
        if p.exists() and p.is_file():
            log(f"Anchor keyframe already exists: {p}")
            target_p = p
            if custom_filename:
                target_p = Path(OUTPUT_IMG_DIR) / (custom_filename if custom_filename.endswith(".png") else f"{custom_filename}.png")
                try:
                    shutil.copy2(p, target_p)
                except Exception as ex:
                    log(f"Warning: could not copy existing anchor to custom filename: {ex}")
                    target_p = p
            try:
                with open(target_p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return True, str(target_p.resolve()), b64
            except Exception as ex:
                log(f"Warning: could not read existing anchor image base64: {ex}")
                return True, str(target_p.resolve()), None
        else:
            log(f"Specified anchor image path not found: {img_path}. Falling back to prompt generation.")

    # CASE 2: Generate via prompt
    prompt = anchor_meta.get("prompt", "")
    if not prompt:
        return False, "Neither valid 'image_path' nor 'prompt' provided for anchor frame", None

    model = anchor_meta.get("model") or anchor_meta.get("image_model") or anchor_meta.get("model_alias")
    engine_val = str(anchor_meta.get("image_engine") or anchor_meta.get("backend") or "auto").lower()
    lora = anchor_meta.get("lora") or anchor_meta.get("lora_alias")
    lora_strength = float(anchor_meta.get("lora_strength", 0.8))

    use_a1111 = (engine_val in ("a1111", "sdxl", "pony")) or (
        engine_val == "auto" and model and any(x in str(model).lower() for x in ["sdxl", "pony", "illustrious", "animagine"])
    )

    if use_a1111:
        engine_name = "sdxl"
    elif engine_val in ("flux2", "klein") or (model and any(x in str(model).lower() for x in ["flux2", "klein"])):
        engine_name = "flux2"
    else:
        engine_name = "flux1"

    generator = create_image_generator(engine_name, settings=anchor_meta)
    cfg = float(anchor_meta.get("cfg_scale", anchor_meta.get("cfg", 5.5 if engine_name == "sdxl" else 3.5)))
    sampler = anchor_meta.get("sampler_name", anchor_meta.get("sampler", "Euler a" if engine_name == "sdxl" else "euler"))
    steps = int(anchor_meta.get("steps", 30))
    seed = int(anchor_meta.get("seed", -1))
    neg = anchor_meta.get("negative_prompt", "score_6, score_5, score_4, ugly, deformed, low quality, bad anatomy")

    req = ImageRequest(
        prompt=prompt,
        negative_prompt=neg,
        model_name=model,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg,
        guidance=cfg,
        sampler_name=sampler,
        seed=seed,
        lora_name=lora,
        lora_strength=lora_strength,
        highres_fix=bool(anchor_meta.get("highres_fix", False)),
        custom_filename=custom_filename
    )
    res = generator.generate(req)
    if not res.success:
        return False, f"{engine_name.upper()} anchor generation failed: {res.error}", None
    return True, res.file_path, res.base64_data
