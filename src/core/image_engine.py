#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Image Engine for media_mcp.
Routes requests through the polymorphic Generator Strategy & Factory pattern.
Supports FLUX.1 Dev, FLUX.2 Klein (ComfyUI) and SDXL / Pony / Illustrious (A1111).
Renders all images at optimal ~1.0 MP native dimensions (64px grid for SDXL, 16px grid for FLUX)
with matching aspect ratio, then resizes via Lanczos to the requested target resolution.
"""

from __future__ import annotations
import sys
import json
import math
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from . import config
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
    return config.get_temp_dir()


def set_global_progress_callback(cb):
    set_image_progress_callback(cb)


def calculate_native_dimensions(
    image_input: Any,
    target_area: int = 1024 * 1024,
    multiple_of: Optional[int] = None,
    min_dim: int = 512,
    max_dim: int = 1536,
    engine: str = "sdxl"
) -> Tuple[int, int]:
    """
    Calculates optimal ~1.0 MP native (width, height) preserving the input aspect ratio:
    - Uses 64px quantization step for SDXL UNet (8x VAE * 8x UNet downsampling)
    - Uses 16px quantization step for FLUX DiT (8x VAE * 2x2 patch size)
    Clamps dimensions to [min_dim, max_dim] to prevent degenerate diffusion geometries.
    """
    from PIL import Image

    if isinstance(image_input, (tuple, list)) and len(image_input) >= 2:
        src_w, src_h = int(image_input[0]), int(image_input[1])
    elif isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            src_w, src_h = img.size
    else:
        src_w, src_h = image_input.size

    if src_w <= 0 or src_h <= 0:
        return (1024, 1024)

    if multiple_of is None:
        eng_lower = str(engine or "sdxl").strip().lower()
        multiple_of = 64 if eng_lower == "sdxl" else 16

    ar = src_w / src_h
    ideal_h = math.sqrt(target_area / ar)
    ideal_w = ideal_h * ar

    native_w = max(min_dim, min(max_dim, int(round(ideal_w / multiple_of) * multiple_of)))
    native_h = max(min_dim, min(max_dim, int(round(ideal_h / multiple_of) * multiple_of)))

    return native_w, native_h


def _make_Mask_for_outpaint(
    width: int,
    height: int,
    new_w: int,
    new_h: int,
    offset_x: int,
    offset_y: int,
    user_mask_l: Any = None
) -> Any:
    """
    Creates an 8-bit grayscale ('L') outpainting mask using pure PIL (no numpy required)
    with 255 in padded borders, 0 (or user_mask_l) in center, and radius-4 Gaussian feathering.
    """
    from PIL import Image, ImageFilter

    base_mask = Image.new("L", (width, height), 255)
    if user_mask_l is not None:
        scaled_user_m = user_mask_l.resize((new_w, new_h), Image.Resampling.BILINEAR)
        base_mask.paste(scaled_user_m, (offset_x, offset_y))
    else:
        center_zero = Image.new("L", (new_w, new_h), 0)
        base_mask.paste(center_zero, (offset_x, offset_y))

    return base_mask.filter(ImageFilter.GaussianBlur(radius=4))


def _pad_image_edge(scaled_im: Any, width: int, height: int, offset_x: int, offset_y: int) -> Any:
    """
    Pads scaled_im into a (width, height) canvas replicating edge pixels using pure PIL.
    """
    from PIL import Image

    new_w, new_h = scaled_im.size
    canvas = Image.new("RGB", (width, height))
    canvas.paste(scaled_im, (offset_x, offset_y))

    # Left & right edge replication
    if offset_x > 0:
        left_strip = scaled_im.crop((0, 0, 1, new_h)).resize((offset_x, new_h), Image.Resampling.NEAREST)
        canvas.paste(left_strip, (0, offset_y))
    right_pad = width - (offset_x + new_w)
    if right_pad > 0:
        right_strip = scaled_im.crop((new_w - 1, 0, new_w, new_h)).resize((right_pad, new_h), Image.Resampling.NEAREST)
        canvas.paste(right_strip, (offset_x + new_w, offset_y))

    # Top & bottom edge replication (across full width)
    if offset_y > 0:
        top_strip = canvas.crop((0, offset_y, width, offset_y + 1)).resize((width, offset_y), Image.Resampling.NEAREST)
        canvas.paste(top_strip, (0, 0))
    bottom_pad = height - (offset_y + new_h)
    if bottom_pad > 0:
        bottom_strip = canvas.crop((0, offset_y + new_h - 1, width, offset_y + new_h)).resize((width, bottom_pad), Image.Resampling.NEAREST)
        canvas.paste(bottom_strip, (0, offset_y + new_h))

    return canvas


def generate_image(params: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str], Dict[str, Any]]:
    """
    Unified entry point for image generation across all engines (SDXL, Pony, FLUX.1, FLUX.2).
    Takes a canonical image_spec dictionary matching schemas/requests/image_generate.json.
    Returns: (success, image_path, None, metadata)
    """
    from PIL import Image

    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return False, None, None, {"error": "Parameter 'prompt' is required for image generation"}

    raw_engine = str(params.get("engine") or "").strip().lower()
    if not raw_engine:
        return False, None, None, {"error": "Parameter 'engine' is required for image generation"}
    if "flux2" in raw_engine:
        engine_name = "flux2"
    elif "flux" in raw_engine:
        engine_name = "flux1"
    elif raw_engine == "sdxl":
        engine_name = "sdxl"
    else:
        return False, None, None, {"error": f"Unsupported image engine '{raw_engine}'. Allowed: 'sdxl', 'flux1', 'flux2'."}

    model = str(params.get("model") or params.get("checkpoint") or "").strip()
    if not model:
        return False, None, None, {"error": "Parameter 'model' is required for image generation"}

    steps = int(params["steps"]) if params.get("steps") is not None else None
    seed = int(params.get("seed", -1))
    negative_prompt = params.get("negative_prompt", "")
    sampler = params.get("sampler") or params.get("sampler_name")

    default_cfg = 7.0 if engine_name == "sdxl" else 1.0
    cfg_val = float(params.get("cfg", default_cfg))
    guidance_val = float(params.get("guidance", 3.5 if "flux" in engine_name else cfg_val))

    loras: Optional[List[Dict[str, Any]]] = None
    if "loras" in params and isinstance(params["loras"], list):
        loras = params["loras"]
    elif "lora" in params and params["lora"]:
        loras = [{"name": str(params["lora"]), "strength": float(params.get("lora_strength", 0.8))}]

    # Fit & dimension validation rules
    fit_mode = params.get("fit")
    if fit_mode:
        if not params.get("image_path"):
            return False, None, None, {"error": "Parameter 'fit' requires 'image_path' to be provided."}
        if fit_mode not in ("crop", "outpaint"):
            return False, None, None, {"error": f"Invalid fit mode '{fit_mode}'. Allowed values: 'crop', 'outpaint'."}
        if not (params.get("width") and params.get("height")):
            return False, None, None, {"error": "Parameter 'fit' requires both 'width' and 'height' to be explicitly specified."}
        target_w = snap_to_16(int(params["width"]))
        target_h = snap_to_16(int(params["height"]))
    elif params.get("image_path") and (params.get("width") or params.get("height")):
        return False, None, None, {
            "error": "Manual 'width' and 'height' cannot be combined with 'image_path' unless 'fit' is specified ('crop' or 'outpaint')."
        }
    else:
        target_w = snap_to_16(int(params.get("width", 1024)))
        target_h = snap_to_16(int(params.get("height", 1024)))

    source_img_path = None
    original_source_path = None
    default_denoise = 0.85 if fit_mode == "outpaint" else 0.65
    denoise_val = float(params.get("denoise", default_denoise))
    prepared_mask_path = None
    original_mask_path = None
    warnings: List[str] = []

    now_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rand_h = uuid.uuid4().hex[:6]
    run_prefix = str(params.get("run_prefix") or "").strip()
    custom_fn = params.get("custom_filename")
    if not run_prefix:
        if custom_fn:
            stem = Path(custom_fn).stem
            for suffix in ("-final", "-poster", "-anchor"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            run_prefix = stem
        else:
            run_prefix = f"{now_ts}-{rand_h}"

    if not custom_fn:
        custom_fn = f"{run_prefix}-final.png"

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

        try:
            temp_dir = get_temp_dir()
            with Image.open(source_img_path) as src_im:
                cur_w, cur_h = src_im.size

                if not fit_mode:
                    # Without fit, target resolution is the optimal ~1MP native bucket for the source aspect ratio
                    native_w, native_h = calculate_native_dimensions((cur_w, cur_h), engine=engine_name)
                    target_w, target_h = native_w, native_h
                else:
                    native_w, native_h = calculate_native_dimensions((target_w, target_h), engine=engine_name)

                if fit_mode == "crop":
                    scale = max(native_w / cur_w, native_h / cur_h)
                    new_w = max(1, round(cur_w * scale))
                    new_h = max(1, round(cur_h * scale))
                    scaled_im = src_im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
                    left = (new_w - native_w) // 2
                    top = (new_h - native_h) // 2
                    cropped_im = scaled_im.crop((left, top, left + native_w, top + native_h))

                    norm_path = temp_dir / f"{run_prefix}-crop.png"
                    cropped_im.save(norm_path, format="PNG")
                    source_img_path = norm_path.resolve().as_posix()
                    log(f"Cropped img2img source with fit='crop' from {cur_w}x{cur_h} to native {native_w}x{native_h}: {source_img_path}")

                    if user_mask_l is not None:
                        scaled_mask = user_mask_l.resize((new_w, new_h), Image.Resampling.BILINEAR)
                        cropped_mask = scaled_mask.crop((left, top, left + native_w, top + native_h))
                        norm_mask_path = temp_dir / f"{run_prefix}-crop-mask.png"
                        cropped_mask.save(norm_mask_path, format="PNG")
                        prepared_mask_path = norm_mask_path.resolve().as_posix()
                        log(f"Cropped inpainting mask with fit='crop' to native {native_w}x{native_h}: {prepared_mask_path}")

                elif fit_mode == "outpaint":
                    scale = min(native_w / cur_w, native_h / cur_h)
                    new_w = max(1, round(cur_w * scale))
                    new_h = max(1, round(cur_h * scale))
                    scaled_im = src_im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)

                    offset_x = (native_w - new_w) // 2
                    offset_y = (native_h - new_h) // 2

                    padded_im = _pad_image_edge(scaled_im, native_w, native_h, offset_x, offset_y)
                    norm_path = temp_dir / f"{run_prefix}-outpaint.png"
                    padded_im.save(norm_path, format="PNG")
                    source_img_path = norm_path.resolve().as_posix()
                    log(f"Padded img2img source with fit='outpaint' from {cur_w}x{cur_h} to native {native_w}x{native_h}: {source_img_path}")

                    feathered_mask = _make_Mask_for_outpaint(
                        native_w, native_h, new_w, new_h, offset_x, offset_y, user_mask_l
                    )
                    norm_mask_path = temp_dir / f"{run_prefix}-outpaint-mask.png"
                    feathered_mask.save(norm_mask_path, format="PNG")
                    prepared_mask_path = norm_mask_path.resolve().as_posix()
                    log(f"Generated feathered outpaint mask at native {native_w}x{native_h}: {prepared_mask_path}")

                else:
                    log(f"Auto-calculated native resolution for img2img: {native_w}x{native_h} (source: {source_img_path})")
                    if cur_w != native_w or cur_h != native_h:
                        resized_im = src_im.convert("RGB").resize((native_w, native_h), Image.Resampling.LANCZOS)
                        norm_path = temp_dir / f"{run_prefix}-norm.png"
                        resized_im.save(norm_path, format="PNG")
                        source_img_path = norm_path.resolve().as_posix()
                        log(f"Resized img2img source image from {cur_w}x{cur_h} to native {native_w}x{native_h}: {source_img_path}")

                    if user_mask_l is not None:
                        resized_mask = user_mask_l.resize((native_w, native_h), Image.Resampling.BILINEAR)
                        norm_mask_path = temp_dir / f"{run_prefix}-norm-mask.png"
                        resized_mask.save(norm_mask_path, format="PNG")
                        prepared_mask_path = norm_mask_path.resolve().as_posix()
                        log(f"Prepared normalized inpainting mask at native {native_w}x{native_h}: {prepared_mask_path}")

        except Exception as ex:
            return False, None, None, {"error": f"Failed to pre-process source image/mask: {ex}"}

    elif params.get("mask_path"):
        return False, None, None, {"error": "Parameter 'mask_path' requires 'image_path' to be provided."}
    else:
        if denoise_val != 1.0:
            log(f"Notice: Normalizing denoise from {denoise_val} to 1.0 for pure txt2img generation.")
            denoise_val = 1.0
        native_w, native_h = calculate_native_dimensions((target_w, target_h), engine=engine_name)

    # Check optimal aspect ratio and resolution bounds -> emit non-fatal warnings
    req_ar = target_w / max(1, target_h)
    if req_ar > 2.4 or req_ar < (1.0 / 2.4):
        w_msg = (
            f"Requested aspect ratio ({target_w}x{target_h}, AR={req_ar:.2f}) exceeds optimal diffusion "
            f"bounds [1:2.4 .. 2.4:1] for engine '{engine_name}'."
        )
        warnings.append(w_msg)
        log(f"Warning: {w_msg}")

    if target_w * target_h > int(1.1 * 1024 * 1024):
        w_msg = (
            f"Target resolution ({target_w}x{target_h}) exceeds optimal ~1.0 MP native generation "
            f"({native_w}x{native_h}); output was upscaled via Lanczos."
        )
        warnings.append(w_msg)
        log(f"Warning: {w_msg}")

    try:
        generator = create_image_generator(engine_name, settings=params)
    except Exception as ex:
        return False, None, None, {"error": f"Failed to instantiate generator for engine '{engine_name}': {ex}"}

    req = ImageRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        width=native_w,
        height=native_h,
        steps=steps,
        cfg=cfg_val,
        guidance=guidance_val,
        sampler=sampler,
        seed=seed,
        loras=loras,
        image_path=source_img_path,
        mask_path=prepared_mask_path,
        fit=fit_mode,
        denoise=denoise_val,
        custom_filename=custom_fn,
        run_prefix=run_prefix
    )

    try:
        res = generator.generate(req)
        if not res.success or not res.image_path:
            return False, None, None, {"error": res.error or f"{engine_name.upper()} generation failed"}

        p_img = Path(res.image_path)

        # Post-generation Lanczos resize from optimal native (native_w, native_h) to requested (target_w, target_h)
        if (native_w, native_h) != (target_w, target_h) and p_img.is_file():
            try:
                with Image.open(p_img) as gen_im:
                    if gen_im.size != (target_w, target_h):
                        resized_out = gen_im.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                        resized_out.save(p_img, format="PNG")
                        log(f"Resized generated image from native {native_w}x{native_h} to target {target_w}x{target_h}: {p_img}")
            except Exception as ex:
                log(f"Warning: Failed to resize generated image to target {target_w}x{target_h}: {ex}")

        actual_seed = res.metadata.get("seed", seed)

        resolved_spec = {
            k: v for k, v in params.items()
            if k not in ("custom_filename", "run_prefix")
        }
        resolved_spec["engine"] = engine_name
        resolved_spec["model"] = res.metadata.get("model") or model
        resolved_spec["seed"] = actual_seed
        if not (params.get("image_path") and not fit_mode):
            resolved_spec["width"] = target_w
            resolved_spec["height"] = target_h
        if fit_mode:
            resolved_spec["fit"] = fit_mode
        if original_source_path or source_img_path:
            resolved_spec["image_path"] = original_source_path or source_img_path
            resolved_spec["denoise"] = denoise_val
        if original_mask_path:
            resolved_spec["mask_path"] = original_mask_path
        elif prepared_mask_path and fit_mode == "outpaint":
            resolved_spec["mask_path"] = prepared_mask_path

        spec_filename = f"{run_prefix}-spec.json"
        workflows_dir = Path(config.WORKFLOWS_DIR) if config.WORKFLOWS_DIR and Path(config.WORKFLOWS_DIR).is_dir() else p_img.parent
        spec_path = (workflows_dir / spec_filename).resolve().as_posix()
        try:
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(resolved_spec, f, indent=2, ensure_ascii=False)
            log(f"Saved resolved image specification: {spec_path}")
        except Exception as ex:
            log(f"Warning: Failed to save image spec to {spec_path}: {ex}")

        res.metadata["run_prefix"] = run_prefix
        res.metadata["spec_path"] = spec_path
        res.metadata["seed"] = actual_seed
        res.metadata["width"] = target_w
        res.metadata["height"] = target_h
        res.metadata["native_width"] = native_w
        res.metadata["native_height"] = native_h
        if warnings:
            res.metadata["warnings"] = warnings

        return True, res.image_path, None, res.metadata
    finally:
        from . import comfy_client
        if comfy_client.is_interrupted():
            comfy_client.reset_interrupt()


def generate_anchor_frame(
    anchor_meta: Dict[str, Any],
    width: int = 832,
    height: int = 480
) -> Tuple[bool, str, Optional[str], Dict[str, Any]]:
    """
    Generates or verifies an anchor keyframe for a scene using strict schema fields.
    Supports:
    1. Ready keyframe: {"keyframe_path": "path/to/image.png"} -> returns existing file directly (no generation).
    2. img2img refinement: {"image_path": "path/to/image.png", "prompt": "..."} -> runs img2img through generate_image().
    3. txt2img generation: {"prompt": "..."} (no image_path) -> runs txt2img through generate_image().
    Returns: (success, image_path_or_error, None, metadata)
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
            return True, str(p.resolve()), None, {}
        return False, f"Anchor keyframe_path not found on disk: {keyframe_path}", None, {}

    has_prompt = bool(anchor_meta.get("prompt", "").strip())
    if not has_prompt:
        return False, "Prompt is required for anchor image generation when keyframe_path is not specified", None, {}

    gen_params = dict(anchor_meta)
    if not (gen_params.get("image_path") and not gen_params.get("fit")):
        if "width" not in gen_params:
            gen_params["width"] = width
        if "height" not in gen_params:
            gen_params["height"] = height

    success, out_path, _, meta = generate_image(gen_params)
    if not success or not out_path:
        return False, meta.get("error", "Anchor image generation failed"), None, meta
    return True, out_path, None, meta
