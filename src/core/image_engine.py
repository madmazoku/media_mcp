#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Engine for media_mcp.
Routes requests through the polymorphic Generator Strategy & Factory pattern.
Supports FLUX.1 Dev, FLUX.2 Klein (ComfyUI) and SDXL / Pony / Illustrious (A1111).
Returns local file path, metadata, and base64 representation.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from . import config
from .config import OUTPUT_IMG_DIR
from .generators import (
    create_image_generator,
    ImageRequest,
    set_image_progress_callback,
)


def log(msg: str):
    sys.stderr.write(f"[image-engine] {msg}\n")
    sys.stderr.flush()


def set_global_progress_callback(cb):
    set_image_progress_callback(cb)


def generate_image(params: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Unified entry point for image generation across all engines (SDXL, Pony, FLUX.1, FLUX.2).
    Takes a canonical image_spec dictionary matching schemas/image_spec.json.
    Returns: (success, image_path, base64_data, metadata)
    """
    prompt = params.get("prompt", "").strip()
    if not prompt:
        return False, None, None, {"error": "Parameter 'prompt' is required for image generation"}

    raw_engine = str(params.get("engine") or "sdxl").strip().lower()
    if "flux2" in raw_engine:
        engine_name = "flux2"
    elif "flux" in raw_engine:
        engine_name = "flux1"
    else:
        engine_name = "sdxl"

    width = int(params.get("width", 1024 if "flux" in engine_name else 832))
    height = int(params.get("height", 1024 if "flux" in engine_name else 480))
    steps = int(params["steps"]) if params.get("steps") is not None else None
    seed = int(params.get("seed", -1))
    negative_prompt = params.get("negative_prompt", "")
    model = params.get("model") or params.get("checkpoint")
    sampler = params.get("sampler") or params.get("sampler_name")

    # CFG / Guidance resolution
    default_cfg = 7.0 if engine_name == "sdxl" else 1.0
    cfg_val = float(params.get("cfg", default_cfg))
    guidance_val = float(params.get("guidance", 3.5 if "flux" in engine_name else cfg_val))

    # LoRAs list resolution
    loras: Optional[List[Dict[str, Any]]] = None
    if "loras" in params and isinstance(params["loras"], list):
        loras = params["loras"]
    elif "lora" in params and params["lora"]:
        loras = [{"name": str(params["lora"]), "strength": float(params.get("lora_strength", 0.8))}]

    # img2img source path and denoise resolution
    source_img_path = None
    denoise_val = float(params.get("denoise", 0.65))
    if params.get("image_path"):
        raw_src = params["image_path"]
        p = Path(raw_src)
        if not p.is_file():
            found = False
            for parent_cand in [Path.cwd(), Path(config.OUTPUT_IMG_DIR), Path(config.WORKFLOWS_DIR), Path(config.REPO_DIR)]:
                cand = parent_cand / raw_src
                if cand.is_file():
                    p = cand
                    found = True
                    break
            if not found:
                return False, None, None, {"error": f"Source image for img2img not found on disk: {raw_src}"}
        source_img_path = str(p.resolve())
    elif denoise_val != 1.0:
        log(f"Notice: Normalizing denoise from {denoise_val} to 1.0 for pure txt2img generation.")
        denoise_val = 1.0

    try:
        generator = create_image_generator(engine_name, settings=params)
    except Exception as ex:
        return False, None, None, {"error": f"Failed to instantiate generator for engine '{engine_name}': {ex}"}

    req = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg_val,
        guidance=guidance_val,
        sampler=sampler,
        seed=seed,
        loras=loras,
        image_path=source_img_path,
        denoise=denoise_val
    )

    res = generator.generate(req)
    if not res.success:
        return False, None, None, {"error": res.error or f"{engine_name.upper()} generation failed"}

    # Extract run_prefix from image_path filename: {timestamp}-{hash}-final.png -> {timestamp}-{hash}
    image_basename = os.path.basename(res.image_path)
    if "-" in image_basename:
        run_prefix = image_basename.rsplit("-", 1)[0]
    else:
        run_prefix = os.path.splitext(image_basename)[0]

    actual_seed = res.metadata.get("seed", seed)
    actual_width = res.metadata.get("target_width", res.metadata.get("width", width))
    actual_height = res.metadata.get("target_height", res.metadata.get("height", height))

    resolved_spec = dict(params)
    resolved_spec["seed"] = actual_seed
    resolved_spec["width"] = actual_width
    resolved_spec["height"] = actual_height
    if source_img_path:
        resolved_spec["image_path"] = source_img_path
        resolved_spec["denoise"] = denoise_val
    if res.metadata.get("model") and "model" not in resolved_spec:
        resolved_spec["model"] = res.metadata["model"]

    spec_filename = f"{run_prefix}-spec.json"
    spec_dir = config.WORKFLOWS_DIR if config.WORKFLOWS_DIR and os.path.isdir(config.WORKFLOWS_DIR) else os.path.dirname(res.image_path)
    spec_path = os.path.join(spec_dir, spec_filename)
    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(resolved_spec, f, indent=2, ensure_ascii=False)
        log(f"Saved resolved image specification: {spec_path}")
    except Exception as ex:
        log(f"Warning: Failed to save image spec to {spec_path}: {ex}")

    res.metadata["run_prefix"] = run_prefix
    res.metadata["spec_path"] = spec_path
    res.metadata["seed"] = actual_seed
    res.metadata["width"] = actual_width
    res.metadata["height"] = actual_height

    return True, res.image_path, res.base64_data, res.metadata


def generate_anchor_frame(
    anchor_meta: Dict[str, Any],
    width: int = 832,
    height: int = 480
) -> Tuple[bool, str, Optional[str]]:
    """
    Generates or verifies an anchor keyframe for a scene using strict schema fields.
    Supports:
    1. Ready keyframe: {"keyframe_path": "path/to/image.png"} -> returns existing file directly (no generation).
    2. img2img refinement: {"image_path": "path/to/image.png", "prompt": "..."} -> runs img2img through generate_image().
    3. txt2img generation: {"prompt": "..."} (no image_path) -> runs txt2img through generate_image().
    Returns: (success, image_path, base64_data)
    """
    keyframe_path = anchor_meta.get("keyframe_path")
    if keyframe_path:
        p = Path(keyframe_path)
        if not p.is_file():
            for base in [Path.cwd(), Path(config.OUTPUT_IMG_DIR), Path(config.WORKFLOWS_DIR), Path(config.REPO_DIR)]:
                cand = base / keyframe_path
                if cand.is_file():
                    p = cand
                    break
        if p.is_file():
            log(f"Using ready anchor keyframe: {p}")
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                return True, str(p.resolve()), b64
            except Exception as ex:
                log(f"Warning: could not read existing anchor keyframe base64: {ex}")
                return True, str(p.resolve()), None
        return False, f"Anchor keyframe_path not found on disk: {keyframe_path}", None

    has_prompt = bool(anchor_meta.get("prompt", "").strip())
    if not has_prompt:
        return False, "Prompt is required for anchor image generation when keyframe_path is not specified", None

    # Prepare merged generation parameters
    gen_params = dict(anchor_meta)
    if "width" not in gen_params:
        gen_params["width"] = width
    if "height" not in gen_params:
        gen_params["height"] = height

    success, out_path, b64_data, meta = generate_image(gen_params)
    if not success or not out_path:
        return False, meta.get("error", "Anchor image generation failed"), None
    return True, out_path, b64_data
