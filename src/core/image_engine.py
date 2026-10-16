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
        loras=loras
    )

    res = generator.generate(req)
    if not res.success:
        return False, None, None, {"error": res.error or f"{engine_name.upper()} generation failed"}

    # Extract run_prefix from image_path filename: {timestamp}-{hash}-{engine}.png -> {timestamp}-{hash}
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
    1. Ready image path: {"image_path": "path/to/image.png"} -> validates and returns existing image + base64.
    2. Prompt-based generation: delegates directly to generate_image(anchor_meta).
    Returns: (success, image_path, base64_data)
    """
    img_path = anchor_meta.get("image_path")
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
        elif not anchor_meta.get("prompt"):
            return False, f"Anchor image_path not found on disk: {img_path}", None
        else:
            log(f"Specified anchor image_path not found: {img_path}. Falling back to prompt generation.")

    if not anchor_meta.get("prompt"):
        return False, "Neither valid 'image_path' nor 'prompt' provided for anchor frame", None

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
