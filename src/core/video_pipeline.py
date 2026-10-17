#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Scene-based Video Pipeline with True Forced Prefix Continuity.
Routes block rendering through the polymorphic Generator Strategy & Factory pattern.
Supports Wan 2.2 Fast Mode (4-step LoRA), Wan 2.2 Full (20-step UNet), and LTX-2.3 22B.
Seamless FFmpeg stream copy assembly with zero-torch safetensors header inspection.
"""

import sys
import json
import time
import math
import shutil
import random
import datetime
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from . import config
from .schema_validator import validate_scenario
from . import ffmpeg_engine
from . import comfy_client
from . import image_engine as img_engine_mod
from .lora_inspector import read_latent_temporal_frames
from .generators import (
    create_video_generator,
    VideoRequest,
)


def log(msg: str):
    sys.stderr.write(f"[video-pipeline] {msg}\n")
    sys.stderr.flush()


_GLOBAL_PROGRESS_CALLBACK = None

_PIPELINE_STATUS = {
    "is_running": False,
    "scenario_name": None,
    "scene_index": 0,
    "scene_name": None,
    "total_scenes": 0,
    "block_index": 0,
    "block_name": None,
    "total_blocks": 0,
    "phase": "idle",
    "elapsed_seconds": 0.0,
    "started_at": 0.0
}


def get_pipeline_status() -> Dict[str, Any]:
    st = dict(_PIPELINE_STATUS)
    if st["is_running"] and st["started_at"] > 0:
        st["elapsed_seconds"] = round(time.time() - st["started_at"], 1)
    return st


_INTERRUPT_REQUESTED = False


def request_interrupt():
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True
    comfy_client.interrupt()


def is_interrupted() -> bool:
    return _INTERRUPT_REQUESTED or comfy_client.is_interrupted()


def set_global_progress_callback(cb):
    global _GLOBAL_PROGRESS_CALLBACK
    _GLOBAL_PROGRESS_CALLBACK = cb
    comfy_client.set_global_progress_callback(cb)


def notify_progress(msg: str, progress: Optional[float] = None, total: Optional[float] = None, cb: Optional[Any] = None):
    callback = cb or _GLOBAL_PROGRESS_CALLBACK
    if callback:
        try:
            callback(msg, progress, total)
        except Exception:
            pass
    log(msg)


# ---------------------------------------------------------------------------
# Deep Merge & Prompt Inheritance Helpers
# ---------------------------------------------------------------------------

def deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively deep merges overlay dict onto base dict."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay if overlay is not None else base
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        elif v is not None:
            merged[k] = v
    return merged


def combine_prompts(base: Optional[str], local: Optional[str]) -> str:
    """Combines local prompt with base prompt (local first for block/scene specificity, base last, separated by newline)."""
    b = str(base).strip() if base else ""
    l = str(local).strip() if local else ""
    if b and l:
        return f"{l}\n{b}"
    return l or b


def combine_negative_prompts(base: Optional[str], local: Optional[str]) -> str:
    """Combines base and local negative prompts, deduplicating tags if comma-separated."""
    b = str(base).strip() if base else ""
    l = str(local).strip() if local else ""
    if b and l:
        if "," in b or "," in l:
            tags = [t.strip() for t in (b + ", " + l).split(",") if t.strip()]
            seen = set()
            deduped = []
            for t in tags:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    deduped.append(t)
            return ", ".join(deduped)
        return f"{l}\n{b}"
    return b or l


def _merge_loras(global_loras: Optional[List[Dict[str, Any]]], local_loras: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merges global and local LoRA lists, with local strength overriding global on matching name."""
    merged: Dict[str, float] = {}
    for l in (global_loras or []):
        if isinstance(l, dict) and l.get("name"):
            merged[str(l["name"]).strip()] = float(l.get("strength", 0.8))
    for l in (local_loras or []):
        if isinstance(l, dict) and l.get("name"):
            merged[str(l["name"]).strip()] = float(l.get("strength", 0.8))
    return [{"name": k, "strength": v} for k, v in merged.items()]


merge_lora_lists = _merge_loras



# ---------------------------------------------------------------------------
# Hierarchical Scene Normalizer & Timing Validator
# ---------------------------------------------------------------------------

def resolve_scenario_dict(raw_data: Any, request_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolves base scenario loading via 'scenario_path', deep merges overrides,
    and returns a pure scenario dictionary matching schemas/scenario.json.
    Consumes 'scenario_path' immediately to prevent reference leaks and recursion.
    """
    scenario_dict: Dict[str, Any] = {}
    base_file_path = None

    if isinstance(raw_data, str):
        base_file_path = raw_data
    elif isinstance(raw_data, dict):
        scenario_dict = json.loads(json.dumps(raw_data))
        base_file_path = scenario_dict.pop("scenario_path", None)

    if base_file_path:
        p = Path(base_file_path)
        if not p.is_file():
            found = False
            for parent_cand in [Path.cwd(), Path(config.WORKFLOWS_DIR), Path(config.REPO_DIR)]:
                cand = parent_cand / base_file_path
                if cand.is_file():
                    p = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Scenario file not found on disk: {base_file_path}")
        with open(p, "r", encoding="utf-8") as f:
            base_dict = json.load(f)
        base_dict.pop("scenario_path", None)
        scenario_dict = deep_merge(base_dict, scenario_dict)

    if request_overrides:
        clean_overrides = dict(request_overrides)
        clean_overrides.pop("scenario_path", None)
        scenario_dict = deep_merge(scenario_dict, clean_overrides)

    if not isinstance(scenario_dict, dict) or "scenes" not in scenario_dict:
        raise ValueError("Scenario must be a JSON object matching schemas/scenario.json containing 'scenes' list")

    validate_scenario(scenario_dict)
    return scenario_dict


def normalize_scenario_structure(raw_data: Any, request_overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Validates scenario against schemas/scenario.json and normalizes it into a list of Scene objects.
    Preserves strict separation: technical parameters come from root .image and .video,
    while local anchor_image and block settings override narrative/framing fields without
    corrupting the schema-valid scenario dict.
    """
    scenario_dict = resolve_scenario_dict(raw_data, request_overrides)

    image_cfg = scenario_dict.get("image", {})
    video_cfg = scenario_dict.get("video", {})

    root_video_engine = video_cfg["engine"]
    root_video_model = video_cfg["model"]
    root_video_negative = video_cfg.get("negative_prompt", "")
    root_video_prompt = video_cfg.get("prompt", "")
    root_video_loras = video_cfg.get("loras", [])

    root_image_engine = image_cfg.get("engine", "sdxl")
    root_image_model = image_cfg.get("model", "")
    root_image_negative = image_cfg.get("negative_prompt", "")
    root_image_prompt = image_cfg.get("prompt", "")
    root_image_loras = image_cfg.get("loras", [])

    raw_scenes = scenario_dict["scenes"]

    normalized_scenes = []
    for s_idx, sc in enumerate(raw_scenes):
        sc_negative = combine_negative_prompts(root_video_negative, sc.get("negative_prompt"))
        sc_name = sc.get("name") or f"Scene #{s_idx + 1}"

        raw_anchor = sc["anchor_image"]
        if isinstance(raw_anchor, dict) and raw_anchor.get("keyframe_path"):
            anchor = {"keyframe_path": raw_anchor["keyframe_path"]}
        else:
            anchor = dict(raw_anchor)
            anchor["engine"] = root_image_engine
            anchor["model"] = root_image_model

            # Inherit global image technical parameters only if not already set on local anchor
            for k in ("width", "height", "steps", "cfg", "guidance", "sampler", "seed", "fit"):
                if k not in anchor and k in image_cfg:
                    anchor[k] = image_cfg[k]

            if anchor.get("image_path"):
                if "denoise" not in anchor and "denoise" in image_cfg:
                    anchor["denoise"] = image_cfg["denoise"]
                elif "denoise" not in anchor:
                    anchor["denoise"] = 0.65
            else:
                anchor["denoise"] = 1.0

            anchor["loras"] = _merge_loras(root_image_loras, anchor.get("loras"))

            if anchor.get("prompt") or root_image_prompt:
                anchor["prompt"] = combine_prompts(root_image_prompt, anchor.get("prompt"))
            if anchor.get("negative_prompt") or root_image_negative:
                anchor["negative_prompt"] = combine_negative_prompts(root_image_negative, anchor.get("negative_prompt"))

        raw_blocks = sc["blocks"]
        norm_blocks = []
        for b_idx, blk in enumerate(raw_blocks):
            b_prompt = combine_prompts(root_video_prompt, blk["motion_prompt"])
            b_neg = combine_negative_prompts(sc_negative, blk.get("negative_prompt"))
            b_loras = _merge_loras(root_video_loras, blk.get("loras"))

            norm_blocks.append({
                "name": blk.get("name", f"Block #{b_idx + 1}"),
                "duration": float(blk.get("duration", 5.0)),
                "overlap": float(blk.get("overlap") or 0.0) if b_idx < len(raw_blocks) - 1 else 0.0,
                "motion_prompt": b_prompt,
                "negative_prompt": b_neg,
                "engine": root_video_engine,
                "model": root_video_model,
                "steps": video_cfg.get("steps"),
                "cfg": video_cfg.get("cfg"),
                "seed": video_cfg.get("seed"),
                "loras": b_loras,
                "weight_dtype": video_cfg.get("weight_dtype"),
                "ramp_latents": video_cfg.get("ramp_latents"),
                "min_cfg": video_cfg.get("min_cfg"),
            })

        normalized_scenes.append({
            "name": sc_name,
            "scene_name": sc_name,
            "engine": root_video_engine,
            "model": root_video_model,
            "negative_prompt": sc_negative,
            "anchor_image": anchor,
            "blocks": norm_blocks
        })

    return normalized_scenes


def collect_scenario_timing_errors(scenes: List[Dict[str, Any]], max_duration: Optional[float] = None) -> List[str]:
    """
    Validates dynamic block merging timing constraints inside each scene (montage):
    1. overlap_duration < clip_duration
    2. prefix_duration + clip_duration < max_duration
    Returns a list of human-readable error messages.
    """
    limit = float(max_duration if max_duration is not None else config.MAX_VIDEO_DURATION)
    errors: List[str] = []
    for s_idx, sc in enumerate(scenes):
        sc_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
        blocks = sc.get("blocks", [])
        for b_idx, blk in enumerate(blocks):
            b_name = blk.get("name", f"Block #{b_idx + 1}")
            clip_duration = float(blk.get("duration", 5.0))
            prefix_duration = float(blocks[b_idx - 1].get("overlap") or 0.0) if b_idx > 0 else 0.0
            overlap_duration = float(blk.get("overlap") or 0.0) if b_idx < len(blocks) - 1 else 0.0

            if overlap_duration >= clip_duration:
                errors.append(
                    f"[{sc_name} : {b_name}] Invalid overlap: overlap_duration ({overlap_duration}s) >= "
                    f"clip_duration ({clip_duration}s). Condition 'overlap_duration < clip_duration' is required."
                )

            if prefix_duration + clip_duration >= limit:
                errors.append(
                    f"[{sc_name} : {b_name}] Generation window exceeded: prefix_duration ({prefix_duration}s) + "
                    f"clip_duration ({clip_duration}s) = {prefix_duration + clip_duration:.2f}s >= "
                    f"max_duration ({limit}s). Condition 'prefix_duration + clip_duration < max_duration' is required."
                )
    return errors


def collect_scenario_aspect_ratio_errors(
    scenes: List[Dict[str, Any]],
    video_width: int = 832,
    video_height: int = 480,
    tolerance: float = 0.05
) -> List[str]:
    """
    Validates that anchor images (keyframe_path, image_path, or explicit txt2img resolution)
    have an aspect ratio matching the target video resolution within the given tolerance (default 5%).
    """
    from PIL import Image
    errors: List[str] = []
    if video_height <= 0 or video_width <= 0:
        return errors

    ar_vid = video_width / video_height

    for s_idx, sc in enumerate(scenes):
        sc_name = sc.get("scene_name") or sc.get("name", f"Scene #{s_idx + 1}")
        anchor = sc.get("anchor_image", {})
        if anchor.get("fit"):
            continue

        img_ref = anchor.get("keyframe_path") or anchor.get("image_path")
        if img_ref:
            p = Path(img_ref)
            if not p.is_file():
                for base in [Path.cwd(), Path(config.OUTPUT_IMG_DIR), Path(config.WORKFLOWS_DIR), Path(config.REPO_DIR)]:
                    cand = base / img_ref
                    if cand.is_file():
                        p = cand
                        break
            if p.is_file():
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                    if w > 0 and h > 0:
                        ar_img = w / h
                        diff_pct = abs(ar_img - ar_vid) / ar_vid
                        if diff_pct > tolerance:
                            errors.append(
                                f"[{sc_name}] Aspect ratio mismatch: anchor image '{img_ref}' has resolution {w}x{h} "
                                f"(aspect ratio {ar_img:.2f}), but target video is {video_width}x{video_height} "
                                f"(aspect ratio {ar_vid:.2f}, difference {diff_pct * 100:.1f}% > {tolerance * 100:.1f}%). "
                                f"Anchor image aspect ratio must match video aspect ratio to prevent distortion."
                            )
                except Exception as ex:
                    errors.append(f"[{sc_name}] Failed to inspect anchor image '{img_ref}': {ex}")
            else:
                errors.append(f"[{sc_name}] Anchor image file not found on disk: {img_ref}")

        elif anchor.get("width") and anchor.get("height"):
            w = int(anchor["width"])
            h = int(anchor["height"])
            if w > 0 and h > 0:
                ar_img = w / h
                diff_pct = abs(ar_img - ar_vid) / ar_vid
                if diff_pct > tolerance:
                    errors.append(
                        f"[{sc_name}] Aspect ratio mismatch: anchor resolution {w}x{h} (aspect ratio {ar_img:.2f}) "
                        f"does not match target video {video_width}x{video_height} (aspect ratio {ar_vid:.2f}, "
                        f"difference {diff_pct * 100:.1f}% > {tolerance * 100:.1f}%)."
                    )

    return errors


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def execute_scenario_pipeline(
    scenario_input: Any = None,
    scenario_path: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
    **extra_kwargs
) -> Dict[str, Any]:
    """
    Executes a multi-scene video scenario using polymorphic generator dispatch and 5D Latent-Chaining.
    Supports base loading via scenario_path, top-level 'image' and 'video' structured parameters and Deep Merge.
    """
    if scenario_input is None and scenario_path:
        scenario_input = {"scenario_path": scenario_path}
    elif isinstance(scenario_input, dict) and scenario_path and "scenario_path" not in scenario_input:
        scenario_input = dict(scenario_input)
        scenario_input["scenario_path"] = scenario_path

    req_ov = dict(request_overrides or {})
    if scenario_path:
        req_ov["scenario_path"] = scenario_path
    if image:
        req_ov["image"] = deep_merge(req_ov.get("image", {}), image)
    if video:
        req_ov["video"] = deep_merge(req_ov.get("video", {}), video)

    resolved_scenario_dict = resolve_scenario_dict(scenario_input, request_overrides=req_ov)

    # Resolve concrete global seeds up front so *-spec.json is 100% deterministic and schema-valid
    if "image" in resolved_scenario_dict and isinstance(resolved_scenario_dict["image"], dict):
        img_seed = resolved_scenario_dict["image"].get("seed", -1)
        if img_seed in (-1, None):
            resolved_scenario_dict["image"]["seed"] = random.randint(1, 10**14)

    if "video" in resolved_scenario_dict and isinstance(resolved_scenario_dict["video"], dict):
        vid_seed = resolved_scenario_dict["video"].get("seed", -1)
        if vid_seed in (-1, None):
            resolved_scenario_dict["video"]["seed"] = random.randint(1, 1_000_000_000)

    scenes = normalize_scenario_structure(resolved_scenario_dict)

    effective_vid = resolved_scenario_dict.get("video", {})
    v_engine = effective_vid["engine"].lower()
    v_model = effective_vid["model"]
    width = int(effective_vid.get("width", 832))
    height = int(effective_vid.get("height", 480))
    fps = int(effective_vid.get("fps", 16))
    max_duration = float(effective_vid.get("max_duration", config.MAX_VIDEO_DURATION))

    timing_errors = collect_scenario_timing_errors(scenes, max_duration=max_duration)
    ar_errors = collect_scenario_aspect_ratio_errors(scenes, video_width=width, video_height=height)
    combined_errors = timing_errors + ar_errors
    if combined_errors:
        raise ValueError("; ".join(combined_errors))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}-{run_hash}"
    log(
        f"Executing scenario pipeline with {len(scenes)} scenes "
        f"(run_id={run_prefix}, video_engine={v_engine}, model={v_model}, {width}x{height} @ {fps}fps, max_duration={max_duration}s)..."
    )

    all_block_cores: List[str] = []
    pipeline_warnings: List[str] = []
    first_poster_path: Optional[str] = None

    total_scenes = len(scenes)
    total_blocks = sum(len(sc.get("blocks", [])) for sc in scenes)
    processed_blocks = 0

    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False
    comfy_client.reset_interrupt()

    _PIPELINE_STATUS["is_running"] = True
    _PIPELINE_STATUS["scenario_name"] = resolved_scenario_dict.get("name") or f"{run_prefix}-final"
    _PIPELINE_STATUS["total_scenes"] = total_scenes
    _PIPELINE_STATUS["total_blocks"] = total_blocks
    _PIPELINE_STATUS["processed_blocks"] = 0
    _PIPELINE_STATUS["started_at"] = time.time()
    _PIPELINE_STATUS["phase"] = "starting"

    temp_dir = config.get_temp_dir()

    try:
        for s_idx, sc in enumerate(scenes):
            if is_interrupted():
                raise RuntimeError("Scenario execution was interrupted by user request.")

            scene_idx = s_idx + 1
            scene_name = sc.get("scene_name", f"Scene #{scene_idx}")
            anchor_meta = dict(sc.get("anchor_image", {}))
            blocks = sc.get("blocks", [])

            log("\n==========================================")
            log(f"[{scene_name}] Preparing Scene Anchor Frame...")
            log("==========================================")

            _PIPELINE_STATUS["scene_index"] = scene_idx
            _PIPELINE_STATUS["scene_name"] = scene_name
            _PIPELINE_STATUS["phase"] = f"anchor_generation (Scene {scene_idx}/{total_scenes})"

            notify_progress(
                f"Scene {scene_idx}/{total_scenes} [{scene_name}]: Preparing anchor keyframe...",
                progress=processed_blocks,
                total=total_blocks,
                cb=progress_callback
            )

            scene_anchor_prefix = f"{run_prefix}-s{scene_idx:03d}"
            scene_anchor_filename = f"{scene_anchor_prefix}-anchor.png"
            anchor_meta["run_prefix"] = scene_anchor_prefix
            if not anchor_meta.get("custom_filename"):
                anchor_meta["custom_filename"] = (temp_dir / scene_anchor_filename).resolve().as_posix()

            ok, anchor_img_path, _, anchor_res_meta = img_engine_mod.generate_anchor_frame(
                anchor_meta=anchor_meta,
                width=width,
                height=height
            )
            if not ok:
                raise RuntimeError(f"Failed to obtain anchor keyframe for {scene_name}: {anchor_img_path}")

            if isinstance(anchor_res_meta, dict) and anchor_res_meta.get("warnings"):
                for w in anchor_res_meta["warnings"]:
                    if w not in pipeline_warnings:
                        pipeline_warnings.append(w)

            if s_idx == 0:
                first_poster_path = anchor_img_path

            next_tail_latent: Optional[str] = None
            next_tail_latent_count = 0
            next_tail_image: Optional[str] = None

            for b_idx, blk in enumerate(blocks):
                if is_interrupted():
                    raise RuntimeError("Scenario execution was interrupted by user request.")

                block_idx = b_idx + 1
                b_name = blk.get("name", f"Block {block_idx}")
                dur = float(blk.get("duration", 5.0))
                is_last = (b_idx == len(blocks) - 1)
                overlap = float(blk.get("overlap") or 0.0) if not is_last else 0.0

                motion_prompt = blk["motion_prompt"]
                b_neg = blk.get("negative_prompt") or ""
                b_engine_raw = (blk.get("engine") or v_engine).lower()
                b_model = blk.get("model") or v_model

                b_settings = dict(effective_vid)
                b_settings["model"] = b_model
                if blk.get("loras"):
                    b_settings["loras"] = blk["loras"]

                generator = create_video_generator(b_engine_raw, settings=b_settings)
                stride = int(getattr(generator, "temporal_stride", 4))

                block_prefix = f"{run_prefix}-s{scene_idx:03d}-b{block_idx:03d}"
                out_core_name = f"{block_prefix}-core.mp4"

                _PIPELINE_STATUS["block_index"] = block_idx
                _PIPELINE_STATUS["block_name"] = b_name
                _PIPELINE_STATUS["phase"] = f"{generator.name}_diffusion ({scene_name} - {b_name})"

                notify_progress(
                    f"Scene {scene_idx}/{total_scenes} [{scene_name}] - {b_name} ({dur}s): Diffusing {generator.name.upper()}...",
                    progress=processed_blocks,
                    total=total_blocks,
                    cb=progress_callback
                )

                blk_steps = blk.get("steps") or effective_vid.get("steps")
                blk_cfg = blk.get("cfg") if blk.get("cfg") is not None else effective_vid.get("cfg")
                blk_seed = blk.get("seed") if blk.get("seed") is not None else effective_vid.get("seed", -1)
                blk_ramp = blk.get("ramp_latents") if blk.get("ramp_latents") is not None else effective_vid.get("ramp_latents", 3)
                blk_min_cfg = blk.get("min_cfg") if blk.get("min_cfg") is not None else effective_vid.get("min_cfg", 1.0)
                blk_weight_dtype = blk.get("weight_dtype") or effective_vid.get("weight_dtype")

                # Calculate tail latents needed for the NEXT block from overlap duration
                if not is_last and overlap > 0:
                    overlap_pixels = max(1, int(round(overlap * fps)))
                    k_tail_lat = max(2, math.ceil((overlap_pixels - 1) / stride) + 1)
                    save_tail = True
                    tail_frames = k_tail_lat
                else:
                    k_tail_lat = 0
                    save_tail = False
                    tail_frames = 0

                if b_idx == 0:
                    start_img = anchor_img_path
                    pref_latent = None
                    pref_pixel_frames = 0
                    k_core = max(1, round(dur * fps / stride))
                    core_frames = stride * k_core + 1
                    num_frames = core_frames
                elif next_tail_latent and Path(next_tail_latent).is_file():
                    # Mode A: Pure 5D Latent prefix (previous block had overlap > 0)
                    start_img = None
                    pref_latent = next_tail_latent
                    k_pref_lat = read_latent_temporal_frames(pref_latent, default=next_tail_latent_count)
                    if k_pref_lat <= 0:
                        k_pref_lat = max(2, next_tail_latent_count)
                    log(f"Verified prefix latent temporal frames from header: latents={k_pref_lat}")

                    pref_pixel_frames = (k_pref_lat - 1) * stride + 1
                    k_core = max(1, round(dur * fps / stride))
                    core_frames = stride * k_core
                    num_frames = pref_pixel_frames + core_frames
                elif next_tail_image and Path(next_tail_image).is_file():
                    # Mode B: 1-frame Image Chaining (previous block had overlap == 0)
                    start_img = next_tail_image
                    pref_latent = None
                    k_core = max(1, round(dur * fps / stride))
                    pref_pixel_frames = 1
                    core_frames = stride * k_core
                    num_frames = stride * k_core + 1
                    log(f"Configuring 1-frame I2V chaining from previous block: {Path(next_tail_image).name} (diffusing {num_frames} frames, slicing core 1..{core_frames})")
                else:
                    start_img = anchor_img_path
                    pref_latent = None
                    pref_pixel_frames = 0
                    k_core = max(1, round(dur * fps / stride))
                    core_frames = stride * k_core + 1
                    num_frames = core_frames

                log(f"\n[{scene_name} : {b_name}] Engine: {generator.name.upper()}, Duration: {dur}s, Overlap: {overlap}s")
                log(f"Quantized frames (stride={stride}): total={num_frames}, prefix={pref_pixel_frames}, core={core_frames}, next_tail_latents={tail_frames}")
                log(f"Motion prompt: '{motion_prompt[:60]}...'")

                save_tail_image = (not is_last and overlap == 0)

                req = VideoRequest(
                    image_path=start_img,
                    prefix_latent_path=pref_latent,
                    prefix_frames=pref_pixel_frames,
                    core_frames=core_frames,
                    tail_frames=tail_frames,
                    save_tail=save_tail,
                    save_tail_image=save_tail_image,
                    prompt=motion_prompt,
                    negative_prompt=b_neg,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    steps=blk_steps,
                    cfg=blk_cfg,
                    seed=blk_seed,
                    ramp_latents=int(blk_ramp),
                    min_cfg=float(blk_min_cfg),
                    weight_dtype=blk_weight_dtype,
                    model=b_model,
                    loras=blk.get("loras") or [],
                    out_video_name=out_core_name,
                    run_prefix=block_prefix,
                    progress_callback=progress_callback
                )
                res = generator.generate(req)
                if not res.success or not res.file_path:
                    raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {res.error}")

                if not is_last:
                    if overlap > 0 and save_tail and res.tail_latent_path and Path(res.tail_latent_path).is_file():
                        next_tail_latent = res.tail_latent_path
                        next_tail_latent_count = k_tail_lat
                        next_tail_image = None
                        log(f"Captured tail latent for next block (overlap={overlap}s): {Path(next_tail_latent).name} ({k_tail_lat} latents)")
                    elif res.tail_image_path and Path(res.tail_image_path).is_file():
                        next_tail_latent = None
                        next_tail_latent_count = 0
                        next_tail_image = res.tail_image_path
                        log(f"Received pristine workflow tail image for next block: {Path(next_tail_image).name}")
                    else:
                        next_tail_latent = None
                        next_tail_latent_count = 0
                        next_tail_image = None
                else:
                    next_tail_latent = None
                    next_tail_latent_count = 0
                    next_tail_image = None

                processed_blocks += 1
                _PIPELINE_STATUS["processed_blocks"] = processed_blocks
                all_block_cores.append(res.file_path)

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg (-c copy)...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

        # Final movie assembly across all blocks in a single lossless Stream Copy (-c copy) pass
        final_name = f"{run_prefix}-final.mp4"
        out_vid_dir = Path(config.OUTPUT_VID_DIR) if config.OUTPUT_VID_DIR and Path(config.OUTPUT_VID_DIR).is_dir() else (temp_dir / "videos")
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        final_path = (out_vid_dir / final_name).resolve().as_posix()
        ffmpeg_engine.concat_videos(all_block_cores, final_path)
        total_duration = ffmpeg_engine.get_duration(final_path)

        # Poster handling: prioritize pristine anchor keyframe from Scene 1
        poster_filename = f"{run_prefix}-poster.png"
        out_img_dir = Path(config.OUTPUT_IMG_DIR) if config.OUTPUT_IMG_DIR and Path(config.OUTPUT_IMG_DIR).is_dir() else (temp_dir / "images")
        out_img_dir.mkdir(parents=True, exist_ok=True)
        poster_path = (out_img_dir / poster_filename).resolve().as_posix()
        if first_poster_path and Path(first_poster_path).is_file():
            try:
                shutil.copy2(first_poster_path, poster_path)
            except Exception as e:
                log(f"Warning: Failed to copy anchor image to poster: {e}")
        else:
            poster_path = ffmpeg_engine.extract_poster(final_path, poster_path)
            poster_path = Path(poster_path).resolve().as_posix()

        # Save resolved scenario spec artifact for replay/audit (preserving schema validity)
        spec_filename = f"{run_prefix}-spec.json"
        workflows_dir = Path(config.WORKFLOWS_DIR) if config.WORKFLOWS_DIR and Path(config.WORKFLOWS_DIR).is_dir() else out_vid_dir
        spec_path = (workflows_dir / spec_filename).resolve().as_posix()
        try:
            with open(spec_path, "w", encoding="utf-8") as f:
                json.dump(resolved_scenario_dict, f, indent=2, ensure_ascii=False)
            log(f"Saved resolved scenario specification: {spec_path}")
        except Exception as ex:
            log(f"Warning: Failed to save resolved scenario spec to {spec_path}: {ex}")

        result_payload: Dict[str, Any] = {
            "run_prefix": run_prefix,
            "video_path": final_path,
            "spec_path": spec_path,
            "poster_path": poster_path,
            "duration": total_duration,
            "width": width,
            "height": height,
            "fps": fps,
            "scenes_count": len(scenes),
            "blocks_count": total_blocks
        }
        if pipeline_warnings:
            result_payload["warnings"] = pipeline_warnings
        return result_payload
    finally:
        if comfy_client.is_interrupted():
            comfy_client.reset_interrupt()
        _PIPELINE_STATUS["is_running"] = False
        _PIPELINE_STATUS["phase"] = "idle"
