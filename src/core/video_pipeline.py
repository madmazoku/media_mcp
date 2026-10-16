#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Scene-based Video Pipeline with True Forced Prefix Continuity.
Routes block rendering through the polymorphic Generator Strategy & Factory pattern.
Supports Wan 2.2 Fast Mode (4-step LoRA), Wan 2.2 Full (20-step UNet), and LTX-2.3 22B.
Seamless FFmpeg stream copy assembly + dynamic countdown HUD.
"""

import os
import sys
import time
import math
import shutil
import base64
import datetime
import uuid
from typing import List, Dict, Any, Optional

from .config import OUTPUT_VID_DIR, OUTPUT_IMG_DIR, MAX_VIDEO_DURATION
from .schema_validator import validate_scenario
from . import ffmpeg_engine
from . import comfy_client
from . import image_engine as img_engine_mod
from .generators import (
    create_video_generator,
    VideoRequest,
    LtxVideoGenerator,
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
    """Combines local prompt with base prompt (local first, base last, separated by newline)."""
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


# ---------------------------------------------------------------------------
# Hierarchical Scene Normalizer & Timing Validator
# ---------------------------------------------------------------------------

def normalize_scenario_structure(raw_data: Any, request_overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Validates scenario against schemas/scenario.json and normalizes it into a list of Scene objects.
    Supports top-level 'image' and 'video' sections, prompt inheritance, and deep merge.
    """
    if not isinstance(raw_data, dict):
        raise ValueError("Scenario must be a JSON object matching schemas/scenario.json")

    scenario_dict = dict(raw_data)
    if request_overrides:
        scenario_dict = deep_merge(scenario_dict, request_overrides)

    # Validate merged scenario against centralized JSON Schema
    validate_scenario(scenario_dict)

    image_cfg = scenario_dict.get("image", {})
    video_cfg = scenario_dict.get("video", {})

    root_video_engine = video_cfg.get("engine", "wan-fast")
    root_video_model = video_cfg.get("model")
    root_video_checkpoint = video_cfg.get("checkpoint")
    root_video_negative = video_cfg.get("negative_prompt", "")
    root_video_prompt = video_cfg.get("prompt", "")

    root_image_engine = image_cfg.get("engine", "sdxl")
    root_image_negative = image_cfg.get("negative_prompt", "")
    root_image_prompt = image_cfg.get("prompt", "")

    raw_scenes = scenario_dict["scenes"]

    normalized_scenes = []
    for s_idx, sc in enumerate(raw_scenes):
        sc_engine = sc.get("engine") or root_video_engine
        sc_negative = combine_negative_prompts(root_video_negative, sc.get("negative_prompt"))
        sc_model = sc.get("model") or root_video_model
        sc_checkpoint = sc.get("checkpoint") or root_video_checkpoint
        sc_name = sc.get("scene_name", f"Scene #{s_idx + 1}")

        anchor = dict(sc["anchor_image"])
        anchor["engine"] = anchor.get("engine") or root_image_engine
        for k in (
            "model", "checkpoint", "width", "height", "steps", "cfg",
            "guidance", "sampler", "seed", "lora",
            "lora_strength"
        ):
            if k in image_cfg and k not in anchor:
                anchor[k] = image_cfg[k]

        if anchor.get("prompt") or root_image_prompt:
            anchor["prompt"] = combine_prompts(root_image_prompt, anchor.get("prompt"))
        if anchor.get("negative_prompt") or root_image_negative:
            anchor["negative_prompt"] = combine_negative_prompts(root_image_negative, anchor.get("negative_prompt"))

        raw_blocks = sc["blocks"]
        norm_blocks = []
        for b_idx, blk in enumerate(raw_blocks):
            b_engine = blk.get("engine") or sc_engine
            b_prompt = combine_prompts(root_video_prompt, blk["motion_prompt"])
            b_neg = combine_negative_prompts(sc_negative, blk.get("negative_prompt"))

            norm_blocks.append({
                "name": blk.get("name", f"Block #{b_idx + 1}"),
                "duration": float(blk.get("duration", 5.0)),
                "overlap": float(blk.get("overlap") or 0.0) if b_idx < len(raw_blocks) - 1 else 0.0,
                "motion_prompt": b_prompt,
                "negative_prompt": b_neg,
                "engine": b_engine,
                "model": blk.get("model") or sc_model,
                "checkpoint": blk.get("checkpoint") or sc_checkpoint,
                "steps": blk.get("steps"),
                "cfg": blk.get("cfg"),
                "seed": blk.get("seed"),
                "lora_strength": blk.get("lora_strength"),
                "weight_dtype": blk.get("weight_dtype"),
                "ramp_latents": blk.get("ramp_latents"),
                "min_cfg": blk.get("min_cfg"),
            })

        normalized_scenes.append({
            "scene_name": sc_name,
            "engine": sc_engine,
            "negative_prompt": sc_negative,
            "anchor_image": anchor,
            "blocks": norm_blocks
        })

    return normalized_scenes


def collect_scenario_timing_errors(scenes: List[Dict[str, Any]], max_duration: float = MAX_VIDEO_DURATION) -> List[str]:
    """
    Validates dynamic block merging timing constraints inside each scene (montage):
    1. overlap_duration < clip_duration
    2. prefix_duration + clip_duration < max_duration
    Returns a list of human-readable error messages.
    """
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

            if prefix_duration + clip_duration >= max_duration:
                errors.append(
                    f"[{sc_name} : {b_name}] Generation window exceeded: prefix_duration ({prefix_duration}s) + "
                    f"clip_duration ({clip_duration}s) = {prefix_duration + clip_duration:.2f}s >= "
                    f"max_duration ({max_duration}s). Condition 'prefix_duration + clip_duration < max_duration' is required."
                )
    return errors


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def execute_scenario_pipeline(
    scenario_input: Any,
    enable_hud: bool = False,
    progress_callback: Optional[Any] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a multi-scene video scenario using polymorphic generator dispatch and Latent-Chaining.
    Supports top-level 'image' and 'video' structured parameters and Deep Merge.
    Returns structured results including video paths, duration, and poster base64.
    """
    req_ov = dict(request_overrides or {})
    if image:
        req_ov["image"] = deep_merge(req_ov.get("image", {}), image)
    if video:
        req_ov["video"] = deep_merge(req_ov.get("video", {}), video)

    scenes = normalize_scenario_structure(scenario_input, request_overrides=req_ov)

    scenario_dict = scenario_input if isinstance(scenario_input, dict) else {}
    effective_vid = deep_merge(scenario_dict.get("video", {}), req_ov.get("video", {}))
    v_engine = effective_vid.get("engine", "wan-fast").lower()
    width = int(effective_vid.get("width", 832))
    height = int(effective_vid.get("height", 480))
    fps = int(effective_vid.get("fps", 16))
    max_duration = float(effective_vid.get("max_duration", MAX_VIDEO_DURATION))

    timing_errors = collect_scenario_timing_errors(scenes, max_duration=max_duration)
    if timing_errors:
        raise ValueError("; ".join(timing_errors))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}-{run_hash}"
    log(f"Executing scenario pipeline with {len(scenes)} scenes (run_id={run_prefix}, video_engine={v_engine}, {width}x{height} @ {fps}fps, max_duration={max_duration}s)...")

    all_block_cores = []
    generated_block_metadata = []
    first_poster_b64 = None
    first_poster_path = None

    total_scenes = len(scenes)
    total_blocks = sum(len(sc.get("blocks", [])) for sc in scenes)
    processed_blocks = 0

    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False
    comfy_client.reset_interrupt()

    _PIPELINE_STATUS["is_running"] = True
    _PIPELINE_STATUS["scenario_name"] = scenario_dict.get("name") or f"{run_prefix}-final"
    _PIPELINE_STATUS["total_scenes"] = total_scenes
    _PIPELINE_STATUS["total_blocks"] = total_blocks
    _PIPELINE_STATUS["processed_blocks"] = 0
    _PIPELINE_STATUS["started_at"] = time.time()
    _PIPELINE_STATUS["phase"] = "starting"

    try:
        for s_idx, sc in enumerate(scenes):
            if is_interrupted():
                raise RuntimeError("Scenario execution was interrupted by user request.")

            scene_idx = s_idx + 1
            scene_name = sc.get("scene_name", f"Scene #{scene_idx}")
            anchor_meta = sc.get("anchor_image", {})
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

            anchor_filename = f"{run_prefix}-s{scene_idx:03d}-anchor.png"
            ok, anchor_img_path, anchor_b64 = img_engine_mod.generate_anchor_frame(
                anchor_meta=anchor_meta,
                width=width,
                height=height,
                custom_filename=anchor_filename
            )
            if not ok:
                raise RuntimeError(f"Failed to obtain anchor keyframe for {scene_name}: {anchor_img_path}")

            if s_idx == 0 and anchor_b64:
                first_poster_b64 = anchor_b64
                first_poster_path = anchor_img_path

            next_tail_latent = None
            next_tail_latent_count = 0
            next_tail_image = None

            for b_idx, blk in enumerate(blocks):
                if is_interrupted():
                    raise RuntimeError("Scenario execution was interrupted by user request.")

                block_idx = b_idx + 1
                b_name = blk.get("name", f"Block {block_idx}")
                dur = float(blk.get("duration", 5.0))
                overlap = float(blk.get("overlap") or 0.0) if b_idx < len(blocks) - 1 else 0.0

                motion_prompt = blk["motion_prompt"]
                b_neg = blk.get("negative_prompt") or ""
                b_engine_raw = (blk.get("engine") or v_engine).lower()

                # Instantiate the generator strategy with scenario and block overrides
                b_settings = dict(effective_vid)
                for override_key in ("checkpoint", "model", "lora_strength", "weight_dtype"):
                    if blk.get(override_key) is not None:
                        b_settings[override_key] = blk[override_key]

                generator = create_video_generator(b_engine_raw, settings=b_settings)
                is_ltx = isinstance(generator, LtxVideoGenerator)

                is_last = (b_idx == len(blocks) - 1)
                stride = 8 if is_ltx else 4

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

                if is_ltx:
                    if not is_last and overlap > 0:
                        overlap_pixels = max(1, int(round(overlap * fps)))
                        # LTX temporal stride: 1 latent past first = 8 pixel frames
                        k_tail_lat = max(2, math.ceil((overlap_pixels - 1) / stride) + 1)
                        save_tail = True
                        tail_frames = k_tail_lat
                    else:
                        k_tail_lat = 0
                        save_tail = False
                        tail_frames = 0

                    if b_idx == 0:
                        ltx_start_img = anchor_img_path
                        ltx_pref_latent = None
                        ltx_pref_pixel_frames = 0
                        k_core = max(1, round(dur * fps / stride))
                        ltx_core_frames = stride * k_core + 1
                        ltx_num_frames = ltx_core_frames
                    elif next_tail_latent and os.path.exists(next_tail_latent):
                        # Mode A: Latent prefix (previous block had overlap > 0)
                        ltx_start_img = None
                        ltx_pref_latent = next_tail_latent
                        try:
                            import safetensors.torch
                            _t_dict = safetensors.torch.load_file(ltx_pref_latent, device="cpu")
                            _t_val = _t_dict.get("latent_tensor", _t_dict.get("samples"))
                            k_pref_lat = int(_t_val.shape[2])
                            log(f"Verified LTX prefix latent shape from safetensors: {_t_val.shape} (latents={k_pref_lat})")
                        except Exception as e:
                            k_pref_lat = next_tail_latent_count
                            log(f"Using tracked LTX prefix latents={k_pref_lat} (safetensors read note: {e})")

                        ltx_pref_pixel_frames = (k_pref_lat - 1) * stride + 1
                        k_core = max(1, round(dur * fps / stride))
                        ltx_core_frames = stride * k_core
                        ltx_num_frames = ltx_pref_pixel_frames + ltx_core_frames
                    elif next_tail_image and os.path.exists(next_tail_image):
                        # Mode B: 1-frame Image Chaining (previous block had overlap == 0)
                        ltx_start_img = next_tail_image
                        ltx_pref_latent = None
                        k_core = max(1, round(dur * fps / stride))
                        ltx_pref_pixel_frames = 1
                        ltx_core_frames = stride * k_core
                        ltx_num_frames = stride * k_core + 1
                        log(f"Configuring LTX 1-frame I2V chaining: {os.path.basename(next_tail_image)} (core={ltx_core_frames})")
                    else:
                        ltx_start_img = anchor_img_path
                        ltx_pref_latent = None
                        ltx_pref_pixel_frames = 0
                        k_core = max(1, round(dur * fps / stride))
                        ltx_core_frames = stride * k_core + 1
                        ltx_num_frames = ltx_core_frames

                    log(f"\n[{scene_name} : {b_name}] Engine: {generator.name.upper()}, Duration: {dur}s, Overlap: {overlap}s")
                    log(f"LTX quantized frames: total={ltx_num_frames}, core={ltx_core_frames}, prefix_pixel={ltx_pref_pixel_frames}, tail_latents={tail_frames}")
                    log(f"Motion prompt: '{motion_prompt[:60]}...'")

                    save_tail_image = (not is_last and overlap == 0)

                    req = VideoRequest(
                        image_path=ltx_start_img,
                        prefix_latent_path=ltx_pref_latent,
                        prefix_frames=ltx_pref_pixel_frames,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=ltx_num_frames,
                        core_frames=ltx_core_frames,
                        tail_frames=tail_frames,
                        save_tail=save_tail,
                        save_tail_image=save_tail_image,
                        fps=fps,
                        steps=blk_steps,
                        cfg=blk_cfg,
                        seed=blk_seed,
                        out_video_name=out_core_name,
                        progress_callback=progress_callback
                    )
                    res = generator.generate(req)
                    if not res.success or not res.file_path:
                        raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {res.error}")

                    if not is_last:
                        if overlap > 0 and save_tail and res.tail_latent_path and os.path.exists(res.tail_latent_path):
                            next_tail_latent = res.tail_latent_path
                            next_tail_latent_count = tail_frames
                            next_tail_image = None
                            log(f"Received pristine workflow tail latent for next block (LTX): {os.path.basename(next_tail_latent)} ({tail_frames} latents)")
                        elif res.tail_image_path and os.path.exists(res.tail_image_path):
                            next_tail_latent = None
                            next_tail_latent_count = 0
                            next_tail_image = res.tail_image_path
                            log(f"Received pristine workflow tail image for next block: {os.path.basename(next_tail_image)}")
                        else:
                            next_tail_latent = None
                            next_tail_latent_count = 0
                            next_tail_image = None
                    else:
                        next_tail_latent = None
                        next_tail_latent_count = 0
                        next_tail_image = None
                    block_core_frames = ltx_core_frames
                else:
                    # Calculate tail latents needed for the NEXT block from overlap duration
                    if not is_last and overlap > 0:
                        overlap_pixels = max(1, int(round(overlap * fps)))
                        # Wan 2.2 temporal stride: 1 latent past first = 4 pixel frames
                        k_tail_lat = max(2, math.ceil((overlap_pixels - 1) / 4) + 1)
                        save_tail = True
                        tail_frames = k_tail_lat
                    else:
                        k_tail_lat = 0
                        save_tail = False
                        tail_frames = 0

                    if b_idx == 0:
                        wan_start_img = anchor_img_path
                        wan_pref_latent = None
                        wan_pref_pixel_frames = 0
                        k_core = max(1, round(dur * fps / 4))
                        wan_core_frames = 4 * k_core + 1
                        wan_num_frames = wan_core_frames
                    elif next_tail_latent and os.path.exists(next_tail_latent):
                        # Mode A: Latent prefix (previous block had overlap > 0)
                        wan_start_img = None
                        wan_pref_latent = next_tail_latent
                        try:
                            import safetensors.torch
                            _t_dict = safetensors.torch.load_file(wan_pref_latent, device="cpu")
                            _t_val = _t_dict.get("latent_tensor", _t_dict.get("samples"))
                            k_pref_lat = int(_t_val.shape[2])
                            log(f"Verified prefix latent shape from safetensors: {_t_val.shape} (latents={k_pref_lat})")
                        except Exception as e:
                            k_pref_lat = next_tail_latent_count
                            log(f"Using tracked prefix latents={k_pref_lat} (safetensors read note: {e})")

                        wan_pref_pixel_frames = (k_pref_lat - 1) * 4 + 1
                        k_core = max(1, round(dur * fps / 4))
                        wan_core_frames = 4 * k_core
                        wan_num_frames = wan_pref_pixel_frames + wan_core_frames
                    elif next_tail_image and os.path.exists(next_tail_image):
                        # Mode B: 1-frame Image Chaining (previous block had overlap == 0)
                        wan_start_img = next_tail_image
                        wan_pref_latent = None
                        k_core = max(1, round(dur * fps / 4))
                        wan_pref_pixel_frames = 1
                        wan_core_frames = 4 * k_core
                        wan_num_frames = 4 * k_core + 1
                        log(f"Configuring 1-frame I2V chaining from previous block: {os.path.basename(next_tail_image)} (diffusing {wan_num_frames} frames, slicing core 1..{wan_core_frames})")
                    else:
                        wan_start_img = anchor_img_path
                        wan_pref_latent = None
                        wan_pref_pixel_frames = 0
                        k_core = max(1, round(dur * fps / 4))
                        wan_core_frames = 4 * k_core + 1
                        wan_num_frames = wan_core_frames

                    blk_ramp = blk.get("ramp_latents") if blk.get("ramp_latents") is not None else effective_vid.get("ramp_latents", 3)
                    blk_min_cfg = blk.get("min_cfg") if blk.get("min_cfg") is not None else effective_vid.get("min_cfg", 1.0)

                    log(f"\n[{scene_name} : {b_name}] Engine: {generator.name.upper()}, Duration: {dur}s, Overlap: {overlap}s")
                    log(f"Wan frames: total={wan_num_frames}, prefix={wan_pref_pixel_frames} ({k_pref_lat if (b_idx > 0 and wan_pref_latent) else 0} latents), core={wan_core_frames}, next_tail={tail_frames} latents")
                    log(f"Ramp latents: {blk_ramp}, min_cfg: {blk_min_cfg}")
                    if wan_pref_latent:
                        log(f"Input prefix latent: {os.path.basename(wan_pref_latent)}")
                    else:
                        log(f"Start anchor image: {os.path.basename(wan_start_img) if wan_start_img else 'None'}")
                    log(f"Motion prompt: '{motion_prompt[:60]}...'")

                    wan_save_tail_image = (not is_last and overlap == 0)

                    req = VideoRequest(
                        image_path=wan_start_img,
                        prefix_latent_path=wan_pref_latent,
                        prefix_frames=wan_pref_pixel_frames,
                        core_frames=wan_core_frames,
                        tail_frames=tail_frames,
                        save_tail=save_tail,
                        save_tail_image=wan_save_tail_image,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=wan_num_frames,
                        fps=fps,
                        steps=blk_steps,
                        cfg=blk_cfg,
                        seed=blk_seed,
                        ramp_latents=int(blk_ramp),
                        min_cfg=float(blk_min_cfg),
                        out_video_name=out_core_name,
                        progress_callback=progress_callback
                    )
                    res = generator.generate(req)
                    if not res.success or not res.file_path:
                        raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {res.error}")

                    if not is_last:
                        if overlap > 0 and res.tail_latent_path and os.path.exists(res.tail_latent_path):
                            next_tail_latent = res.tail_latent_path
                            next_tail_latent_count = k_tail_lat
                            next_tail_image = None
                            log(f"Captured tail latent for next block (overlap={overlap}s): {os.path.basename(next_tail_latent)} ({k_tail_lat} latents)")
                        elif res.tail_image_path and os.path.exists(res.tail_image_path):
                            next_tail_latent = None
                            next_tail_latent_count = 0
                            next_tail_image = res.tail_image_path
                            log(f"Received pristine workflow tail image for next block: {os.path.basename(next_tail_image)}")
                        else:
                            next_tail_latent = None
                            next_tail_latent_count = 0
                            next_tail_image = None
                    else:
                        next_tail_latent = None
                        next_tail_latent_count = 0
                        next_tail_image = None
                    block_core_frames = wan_core_frames

                processed_blocks += 1
                _PIPELINE_STATUS["processed_blocks"] = processed_blocks

                all_block_cores.append(res.file_path)
                actual_core_dur = block_core_frames / fps
                generated_block_metadata.append((res.file_path, actual_core_dur, overlap if not is_last else 0.0))

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg (-c copy)...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

        # Final movie assembly across all blocks in a single lossless Stream Copy (-c copy) pass
        final_name = f"{run_prefix}-final.mp4"
        final_path = os.path.join(OUTPUT_VID_DIR, final_name)
        ffmpeg_engine.concat_videos(all_block_cores, final_path)
        total_duration = ffmpeg_engine.get_duration(final_path)

        # Poster handling: prioritize pristine anchor keyframe from Scene 1
        poster_filename = f"{run_prefix}-final.png"
        poster_path = os.path.join(OUTPUT_IMG_DIR, poster_filename)
        if first_poster_path and os.path.exists(first_poster_path):
            try:
                shutil.copy2(first_poster_path, poster_path)
            except Exception as e:
                log(f"Warning: Failed to copy anchor image to poster: {e}")
            poster_b64 = first_poster_b64
            if not poster_b64:
                try:
                    with open(poster_path, "rb") as f:
                        poster_b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    pass
        else:
            poster_path, poster_b64 = ffmpeg_engine.extract_poster(final_path, poster_path)

        # Optional HUD generation
        hud_path = None
        if enable_hud and generated_block_metadata:
            hud_name = f"{run_prefix}-hud.mp4"
            hud_final = os.path.join(OUTPUT_VID_DIR, hud_name)
            try:
                ffmpeg_engine.generate_hud_overlay(final_path, generated_block_metadata, hud_final)
                hud_path = hud_final
            except Exception as ex:
                log(f"Warning: HUD generation failed: {ex}")

        return {
            "video_path": final_path,
            "hud_video_path": hud_path,
            "poster_path": poster_path,
            "poster_base64": poster_b64,
            "duration": total_duration,
            "width": width,
            "height": height,
            "fps": fps,
            "scenes_count": len(scenes)
        }
    finally:
        _PIPELINE_STATUS["is_running"] = False
        _PIPELINE_STATUS["phase"] = "idle"
