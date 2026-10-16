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
    snap_to_16,
    set_image_progress_callback,
)


def log(msg: str):
    sys.stderr.write(f"[image-engine] {msg}\n")
    sys.stderr.flush()


def get_temp_dir() -> Path:
    """Returns configured TEMP_DIR if valid, or a standard safe system temporary directory."""
    if config.TEMP_DIR and os.path.isdir(config.TEMP_DIR):
        return Path(config.TEMP_DIR)
    import tempfile
    t = Path(tempfile.gettempdir()) / "media_mcp"
    t.mkdir(parents=True, exist_ok=True)
    return t


def set_global_progress_callback(cb):
    set_image_progress_callback(cb)


def calculate_native_dimensions(
    image_input: Any,
    target_area: int = 1024 * 1024,
    multiple_of: int = 16,
    min_dim: int = 512,
    max_dim: int = 1536
) -> Tuple[int, int]:
    """
    Calculates target (width, height) preserving the source image aspect ratio,
    targeting native model capacity (~1.0 MP for SDXL/FLUX), rounded to multiples of 16.
    Clamps dimensions to [min_dim, max_dim] to prevent degenerate diffusion geometries.
    """
    from PIL import Image
    import math

    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            src_w, src_h = img.size
    else:
        src_w, src_h = image_input.size

    if src_w <= 0 or src_h <= 0:
        return (1024, 1024)

    ar = src_w / src_h
    ideal_h = math.sqrt(target_area / ar)
    ideal_w = ideal_h * ar

    target_w = max(min_dim, min(max_dim, int(round(ideal_w / multiple_of) * multiple_of)))
    target_h = max(min_dim, min(max_dim, int(round(ideal_h / multiple_of) * multiple_of)))

    return target_w, target_h


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
    # ---------------------------------------------------------------------------
    # fit & dimension validation rules
    # ---------------------------------------------------------------------------
    fit_mode = params.get("fit")
    if fit_mode:
        if not params.get("image_path"):
            return False, None, None, {"error": "Parameter 'fit' requires 'image_path' to be provided."}
        if fit_mode not in ("crop", "outpaint"):
            return False, None, None, {"error": f"Invalid fit mode '{fit_mode}'. Allowed values: 'crop', 'outpaint'."}
        if not (params.get("width") and params.get("height")):
            return False, None, None, {"error": "Parameter 'fit' requires both 'width' and 'height' to be explicitly specified."}
        width = snap_to_16(int(params["width"]))
        height = snap_to_16(int(params["height"]))
    elif params.get("image_path") and (params.get("width") or params.get("height")):
        return False, None, None, {
            "error": "Manual 'width' and 'height' cannot be combined with 'image_path' unless 'fit' is specified ('crop' or 'outpaint')."
        }

    # img2img source path and denoise resolution
    source_img_path = None
    original_source_path = None
    default_denoise = 0.85 if fit_mode == "outpaint" else 0.65
    denoise_val = float(params.get("denoise", default_denoise))
    prepared_mask_path = None
    original_mask_path = None

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
        original_source_path = str(p.resolve())
        source_img_path = original_source_path

        from PIL import Image, ImageFilter
        import numpy as np

        # Resolve optional user-provided mask path
        raw_user_mask = params.get("mask_path")
        user_mask_l = None
        if raw_user_mask:
            p_mask = Path(raw_user_mask)
            if not p_mask.is_file():
                found_m = False
                for parent_cand in [Path.cwd(), Path(config.OUTPUT_IMG_DIR), Path(config.WORKFLOWS_DIR), Path(config.REPO_DIR)]:
                    cand_m = parent_cand / raw_user_mask
                    if cand_m.is_file():
                        p_mask = cand_m
                        found_m = True
                        break
                if not found_m:
                    return False, None, None, {"error": f"Inpainting mask file not found on disk: {raw_user_mask}"}
            original_mask_path = str(p_mask.resolve())

            # Validate aspect ratio: |AR_mask - AR_src| / AR_src <= 0.01
            try:
                with Image.open(original_source_path) as s_im, Image.open(original_mask_path) as m_im:
                    s_w, s_h = s_im.size
                    m_w, m_h = m_im.size
                    ar_src = s_w / max(1, s_h)
                    ar_mask = m_w / max(1, m_h)
                    diff_pct = abs(ar_mask - ar_src) / ar_src
                    if diff_pct > 0.01:
                        return False, None, None, {
                            "error": (
                                f"Aspect ratio mismatch between source image ({s_w}x{s_h}, AR {ar_src:.2f}) "
                                f"and inpainting mask ({m_w}x{m_h}, AR {ar_mask:.2f}, difference {diff_pct * 100:.1f}% > 1.0%). "
                                f"Mask aspect ratio must match source image aspect ratio."
                            )
                        }
                    if m_im.mode in ("RGBA", "LA"):
                        alpha = m_im.split()[-1]
                        extrema = alpha.getextrema()
                        user_mask_l = alpha if extrema[0] != extrema[1] else m_im.convert("L")
                    else:
                        user_mask_l = m_im.convert("L")
            except Exception as ex:
                return False, None, None, {"error": f"Failed to validate inpainting mask: {ex}"}

        # Process image and mask according to fit_mode
        try:
            temp_dir = get_temp_dir()
            with Image.open(source_img_path) as src_im:
                cur_w, cur_h = src_im.size

                if fit_mode == "crop":
                    # Scale to cover target (width, height)
                    scale = max(width / cur_w, height / cur_h)
                    new_w = max(1, round(cur_w * scale))
                    new_h = max(1, round(cur_h * scale))
                    scaled_im = src_im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - width) // 2
                    top = (new_h - height) // 2
                    cropped_im = scaled_im.crop((left, top, left + width, top + height))

                    norm_path = temp_dir / f"crop_{Path(source_img_path).stem}_{width}x{height}.png"
                    cropped_im.save(norm_path, format="PNG")
                    source_img_path = str(norm_path.resolve())
                    log(f"Cropped img2img source with fit='crop' from {cur_w}x{cur_h} to {width}x{height}: {source_img_path}")

                    if user_mask_l is not None:
                        scaled_mask = user_mask_l.resize((new_w, new_h), Image.Resampling.BILINEAR)
                        cropped_mask = scaled_mask.crop((left, top, left + width, top + height))
                        norm_mask_path = temp_dir / f"norm_mask_crop_{Path(original_mask_path).stem}_{width}x{height}.png"
                        cropped_mask.save(norm_mask_path, format="PNG")
                        prepared_mask_path = str(norm_mask_path.resolve())
                        log(f"Cropped inpainting mask with fit='crop' to {width}x{height}: {prepared_mask_path}")

                elif fit_mode == "outpaint":
                    # Scale to fit inside target (width, height)
                    scale = min(width / cur_w, height / cur_h)
                    new_w = max(1, round(cur_w * scale))
                    new_h = max(1, round(cur_h * scale))
                    scaled_im = src_im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)

                    offset_x = (width - new_w) // 2
                    offset_y = (height - new_h) // 2

                    scaled_arr = np.array(scaled_im)
                    pad_top = offset_y
                    pad_bottom = height - (offset_y + new_h)
                    pad_left = offset_x
                    pad_right = width - (offset_x + new_w)

                    padded_arr = np.pad(scaled_arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
                    padded_im = Image.fromarray(padded_arr)
                    norm_path = temp_dir / f"outpaint_{Path(source_img_path).stem}_{width}x{height}.png"
                    padded_im.save(norm_path, format="PNG")
                    source_img_path = str(norm_path.resolve())
                    log(f"Padded img2img source with fit='outpaint' from {cur_w}x{cur_h} to {width}x{height}: {source_img_path}")

                    # Outpaint mask: 255 (new content) in border areas, 0 (preserved) or user mask in center
                    base_mask_arr = np.full((height, width), 255, dtype=np.uint8)
                    if user_mask_l is not None:
                        scaled_user_m = user_mask_l.resize((new_w, new_h), Image.Resampling.BILINEAR)
                        base_mask_arr[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = np.array(scaled_user_m)
                    else:
                        base_mask_arr[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = 0

                    # Apply gentle Gaussian feathering on the boundary (radius 4)
                    base_mask_img = Image.fromarray(base_mask_arr)
                    feathered_mask = base_mask_img.filter(ImageFilter.GaussianBlur(radius=4))

                    norm_mask_path = temp_dir / f"norm_mask_outpaint_{Path(source_img_path).stem}_{width}x{height}.png"
                    feathered_mask.save(norm_mask_path, format="PNG")
                    prepared_mask_path = str(norm_mask_path.resolve())
                    log(f"Generated feathered outpaint mask at {width}x{height}: {prepared_mask_path}")

                else:
                    # Native auto-resolution calculation
                    auto_w, auto_h = calculate_native_dimensions(source_img_path)
                    width = auto_w
                    height = auto_h
                    log(f"Auto-calculated native resolution for img2img: {width}x{height} (source: {source_img_path})")

                    if cur_w != width or cur_h != height:
                        resized_im = src_im.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
                        norm_path = temp_dir / f"norm_{Path(source_img_path).stem}_{width}x{height}.png"
                        resized_im.save(norm_path, format="PNG")
                        source_img_path = str(norm_path.resolve())
                        log(f"Resized img2img source image from {cur_w}x{cur_h} to native {width}x{height}: {source_img_path}")

                    if user_mask_l is not None:
                        resized_mask = user_mask_l.resize((width, height), Image.Resampling.BILINEAR)
                        norm_mask_path = temp_dir / f"norm_mask_{Path(original_mask_path).stem}_{width}x{height}.png"
                        resized_mask.save(norm_mask_path, format="PNG")
                        prepared_mask_path = str(norm_mask_path.resolve())
                        log(f"Prepared normalized inpainting mask at native {width}x{height}: {prepared_mask_path}")

        except Exception as ex:
            return False, None, None, {"error": f"Failed to pre-process source image/mask: {ex}"}

    elif denoise_val != 1.0:
        log(f"Notice: Normalizing denoise from {denoise_val} to 1.0 for pure txt2img generation.")
        denoise_val = 1.0
    elif params.get("mask_path"):
        return False, None, None, {"error": "Parameter 'mask_path' requires 'image_path' to be provided."}

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
        mask_path=prepared_mask_path,
        fit=fit_mode,
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
    if fit_mode:
        resolved_spec["fit"] = fit_mode
    if original_source_path or source_img_path:
        resolved_spec["image_path"] = original_source_path or source_img_path
        resolved_spec["denoise"] = denoise_val
    if original_mask_path:
        resolved_spec["mask_path"] = original_mask_path
    elif prepared_mask_path and fit_mode == "outpaint":
        resolved_spec["mask_path"] = prepared_mask_path
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
    if not (gen_params.get("image_path") and not gen_params.get("fit")):
        if "width" not in gen_params:
            gen_params["width"] = width
        if "height" not in gen_params:
            gen_params["height"] = height

    success, out_path, b64_data, meta = generate_image(gen_params)
    if not success or not out_path:
        return False, meta.get("error", "Anchor image generation failed"), None
    return True, out_path, b64_data
