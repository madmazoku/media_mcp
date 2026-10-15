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
import json
import time
import shutil
import base64
import random
import datetime
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

from .config import OUTPUT_VID_DIR, OUTPUT_IMG_DIR, TEMP_DIR
from . import ffmpeg_engine
from . import comfy_client
from . import image_engine as img_engine_mod
from .generators import (
    create_video_generator,
    VideoRequest,
    GenerationResult,
    LtxVideoGenerator,
    snap_to_16,
    snap_to_32,
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
# Latent Slicing & File Helpers (Standalone fallback utility)
# ---------------------------------------------------------------------------

def slice_latent_file(
    in_path: str,
    out_path: str,
    tail_frames: int,
    start_idx: Optional[int] = None
) -> bool:
    """
    Slices `tail_frames` along the temporal dimension of a .latent (safetensors) file.
    If start_idx is provided (e.g. -k), slices from start_idx to start_idx + tail_frames.
    Otherwise slices the last tail_frames.
    """
    try:
        import safetensors.torch
        import torch
        data = safetensors.torch.load_file(in_path)
        tensor = data.get("latent_tensor")
        if tensor is None:
            for v in data.values():
                if hasattr(v, "ndim") and v.ndim == 5:
                    tensor = v
                    break
        if tensor is None:
            log(f"Warning: No 5D latent tensor found in {in_path}. Keys: {list(data.keys())}")
            return False

        t_dim = 2
        total_t = tensor.shape[t_dim]
        if start_idx is not None:
            s_idx = start_idx if start_idx >= 0 else max(0, total_t + start_idx)
            e_idx = min(total_t, s_idx + max(1, tail_frames))
            sliced = tensor[:, :, s_idx:e_idx, :, :].contiguous()
            slice_t = sliced.shape[t_dim]
        else:
            slice_t = min(total_t, max(1, tail_frames))
            sliced = tensor[:, :, -slice_t:, :, :].contiguous()

        save_dict = {
            "latent_tensor": sliced,
            "latent_format_version_0": torch.tensor([])
        }
        safetensors.torch.save_file(save_dict, out_path)
        log(f"Sliced latent: {slice_t} frames saved to {out_path} (from total {total_t} frames)")
        return True
    except Exception as ex:
        log(f"slice_latent_file error: {ex}")
        return False


# ---------------------------------------------------------------------------
# Generator Strategy Facades (Backward Compatibility for direct callers)
# ---------------------------------------------------------------------------

def generate_wan_segment(
    image_path: Optional[Union[str, List[str]]] = None,
    prefix_video_path: Optional[str] = None,
    prefix_latent_path: Optional[str] = None,
    prefix_frames: int = 0,
    end_image_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "blurry, jitter, static, low quality, deformed, extra limbs, bad anatomy, text, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    core_frames: Optional[int] = None,
    tail_frames: int = 0,
    save_tail: bool = True,
    save_tail_latent: Optional[bool] = None,
    fps: int = 16,
    seed: Optional[int] = None,
    fast_mode: bool = False,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    out_video_name: Optional[str] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Submits a Wan 2.2 Video segment via WanFastVideoGenerator or WanFullVideoGenerator.
    Outputs core video directly and exports tail latent for seamless zero-loss momentum chaining.
    Returns: (success, out_video_path, out_tail_latent_path)
    """
    engine_name = "wan-fast" if fast_mode else "wan"
    generator = create_video_generator(engine_name)
    if save_tail_latent is not None:
        save_tail = save_tail_latent

    req = VideoRequest(
        image_path=image_path,
        prefix_video_path=prefix_video_path,
        prefix_latent_path=prefix_latent_path,
        prefix_frames=prefix_frames,
        end_image_path=end_image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        core_frames=core_frames,
        tail_frames=tail_frames,
        save_tail=save_tail,
        fps=fps,
        seed=seed,
        steps=steps,
        cfg=cfg,
        out_video_name=out_video_name,
        progress_callback=progress_callback
    )
    res = generator.generate(req)
    return res.success, (res.file_path or res.error or ""), res.tail_latent_path


def generate_ltx_segment(
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    prefix_latent_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    core_frames: Optional[int] = None,
    save_tail: bool = True,
    save_tail_latent: Optional[bool] = None,
    fps: int = 24,
    seed: Optional[int] = None,
    cfg: float = 3.0,
    steps: int = 20,
    out_video_name: Optional[str] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Submits an LTX-2.3 (22B) Video segment via LtxVideoGenerator.
    Outputs core video directly and optionally exports clean tail video for next block chaining.
    Returns: (success, out_video_path, out_tail_path)
    """
    generator = create_video_generator("ltx")
    if save_tail_latent is not None:
        save_tail = save_tail_latent

    req = VideoRequest(
        image_path=image_path,
        prefix_video_path=prefix_video_path,
        prefix_latent_path=prefix_latent_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        core_frames=core_frames,
        save_tail=save_tail,
        fps=fps,
        seed=seed,
        cfg=cfg,
        steps=steps,
        out_video_name=out_video_name,
        progress_callback=progress_callback
    )
    res = generator.generate(req)
    return res.success, (res.file_path or res.error or ""), res.tail_video_path


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
    """Combines base prompt with local prompt using newline formatting."""
    b = str(base).strip() if base else ""
    l = str(local).strip() if local else ""
    if b and l:
        return f"{b}\n {l}"
    return b or l


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
        return f"{b}\n {l}"
    return b or l


# ---------------------------------------------------------------------------
# Hierarchical Scene Normalizer
# ---------------------------------------------------------------------------

def normalize_scenario_structure(raw_data: Any, request_overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Normalizes scenario input into a clean list of Scene objects:
    [
      {
        "scene_name": str,
        "engine": "wan" | "wan-fast" | "ltx",
        "negative_prompt": str,
        "anchor_image": { "image_path": Optional[str], "prompt": Optional[str], ... },
        "blocks": [ { "name": str, "duration": float, "overlap": float, "motion_prompt": str, "negative_prompt": str, "engine": str, ... } ]
      }
    ]
    Supports top-level 'image' and 'video' sections, prompt inheritance, and deep merge.
    """
    if isinstance(raw_data, list):
        scenario_dict = {"scenes": raw_data}
    elif isinstance(raw_data, dict):
        scenario_dict = dict(raw_data)
    else:
        raise ValueError("Invalid scenario structure")

    if request_overrides:
        scenario_dict = deep_merge(scenario_dict, request_overrides)

    image_cfg = scenario_dict.get("image", {})
    video_cfg = scenario_dict.get("video", {})

    root_video_engine = video_cfg.get("engine", "wan")
    root_video_model = video_cfg.get("model")
    root_video_negative = video_cfg.get("negative_prompt", "")
    root_video_prompt = video_cfg.get("prompt", "")

    root_image_engine = image_cfg.get("engine", "sdxl")
    root_image_model = image_cfg.get("model")
    root_image_negative = image_cfg.get("negative_prompt", "")
    root_image_prompt = image_cfg.get("prompt", "")

    raw_scenes = scenario_dict.get("scenes", [])
    if not raw_scenes and "blocks" in scenario_dict:
        raw_scenes = [scenario_dict]

    normalized_scenes = []
    for s_idx, sc in enumerate(raw_scenes):
        sc_engine = sc.get("engine") or root_video_engine
        sc_negative = combine_negative_prompts(root_video_negative, sc.get("negative_prompt"))
        sc_model = sc.get("model") or root_video_model
        sc_name = sc.get("scene_name", f"Scene #{s_idx + 1}")

        anchor = sc.get("anchor_image") or sc.get("anchor", {})
        anchor["image_engine"] = anchor.get("image_engine") or root_image_engine
        if root_image_model and "model" not in anchor:
            anchor["model"] = root_image_model
        for k in ("cfg", "cfg_scale", "steps", "sampler", "sampler_name"):
            if k in image_cfg and k not in anchor:
                anchor[k] = image_cfg[k]

        anchor["prompt"] = combine_prompts(root_image_prompt, anchor.get("prompt"))
        anchor["negative_prompt"] = combine_negative_prompts(root_image_negative, anchor.get("negative_prompt"))

        raw_blocks = sc.get("blocks", [])
        norm_blocks = []
        for b_idx, blk in enumerate(raw_blocks):
            b_engine = blk.get("engine") or sc_engine
            b_prompt = combine_prompts(root_video_prompt, blk.get("motion_prompt") or blk.get("prompt", ""))
            b_neg = combine_negative_prompts(sc_negative, blk.get("negative_prompt"))

            norm_blocks.append({
                "name": blk.get("name", f"Block #{b_idx + 1}"),
                "duration": float(blk.get("duration", 5.0)),
                "overlap": float(blk.get("overlap", 1.5 if b_idx < len(raw_blocks) - 1 else 0.0)),
                "motion_prompt": b_prompt,
                "negative_prompt": b_neg,
                "engine": b_engine,
                "use_next_as_end": blk.get("use_next_as_end", False),
                "steps": blk.get("steps"),
                "cfg": blk.get("cfg") or blk.get("cfg_scale")
            })

        normalized_scenes.append({
            "scene_name": sc_name,
            "engine": sc_engine,
            "negative_prompt": sc_negative,
            "anchor_image": anchor,
            "blocks": norm_blocks
        })

    return normalized_scenes


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def execute_scenario_pipeline(
    scenario_input: Any,
    output_name: Optional[str] = None,
    enable_hud: bool = False,
    progress_callback: Optional[Any] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
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

    effective_vid = req_ov.get("video", {})
    v_engine = effective_vid.get("engine", "wan").lower()
    width = int(effective_vid.get("width", 832))
    height = int(effective_vid.get("height", 480))
    fps = int(effective_vid.get("fps", 16))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}_{run_hash}"
    log(f"Executing scenario pipeline with {len(scenes)} scenes (run_id={run_prefix}, video_engine={v_engine}, {width}x{height} @ {fps}fps)...")

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
    _PIPELINE_STATUS["scenario_name"] = output_name or f"{run_prefix}_final"
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

            start_img_override = kwargs.get("global_start_image") or kwargs.get("start_image") or req_ov.get("image", {}).get("image_path")
            if s_idx == 0 and start_img_override:
                anchor_meta["image_path"] = start_img_override

            log(f"\n==========================================")
            log(f"[{scene_name}] Preparing Scene Anchor Frame...")
            log(f"==========================================")

            _PIPELINE_STATUS["scene_index"] = scene_idx
            _PIPELINE_STATUS["scene_name"] = scene_name
            _PIPELINE_STATUS["phase"] = f"anchor_generation (Scene {scene_idx}/{total_scenes})"

            notify_progress(
                f"Scene {scene_idx}/{total_scenes} [{scene_name}]: Preparing anchor keyframe...",
                progress=processed_blocks,
                total=total_blocks,
                cb=progress_callback
            )

            anchor_filename = f"{run_prefix}_s{scene_idx:03d}_anchor.png"
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

            prefix_latent_path = None
            prefix_video_path = None
            next_tail_latent = None
            next_tail_image = None

            for b_idx, blk in enumerate(blocks):
                if is_interrupted():
                    raise RuntimeError("Scenario execution was interrupted by user request.")

                block_idx = b_idx + 1
                b_name = blk.get("name", f"Block {block_idx}")
                dur = float(blk.get("duration", 5.0))
                overlap = float(blk.get("overlap", 1.5 if b_idx < len(blocks) - 1 else 0.0))

                # Enforce non-colliding overlap constraint:
                # prefix_length + tail_length < clip_duration
                prefix_length = float(blocks[b_idx - 1].get("overlap", 1.5)) if b_idx > 0 else 0.0
                tail_length = overlap if b_idx < len(blocks) - 1 else 0.0
                if prefix_length + tail_length >= dur:
                    raise ValueError(
                        f"[{scene_name} : {b_name}] Overlap constraint violated: prefix_length ({prefix_length}s) + tail_length ({tail_length}s) = "
                        f"{prefix_length + tail_length:.2f}s >= clip_duration ({dur}s). Condition 'prefix_length + tail_length < clip_duration' is required so transition zones do not collide."
                    )

                motion_prompt = blk.get("motion_prompt") or blk.get("prompt", "")
                b_neg = blk.get("negative_prompt") or ""
                b_engine_raw = (blk.get("engine") or v_engine).lower()

                # Instantiate the generator strategy
                generator = create_video_generator(b_engine_raw, settings=effective_vid)
                is_ltx = isinstance(generator, LtxVideoGenerator) or any(x in b_engine_raw for x in ["ltx", "22b"])

                is_last = (b_idx == len(blocks) - 1)
                stride = 8 if is_ltx else 4

                # Dynamic time quantization matching model VAE stride:
                block_prefix = f"{run_prefix}_s{scene_idx:03d}_b{block_idx:03d}"
                out_core_name = f"{block_prefix}_core.mp4"

                _PIPELINE_STATUS["block_index"] = block_idx
                _PIPELINE_STATUS["block_name"] = b_name
                _PIPELINE_STATUS["phase"] = f"{generator.name}_diffusion ({scene_name} - {b_name})"

                notify_progress(
                    f"Scene {scene_idx}/{total_scenes} [{scene_name}] - {b_name} ({dur}s): Diffusing {generator.name.upper()}...",
                    progress=processed_blocks,
                    total=total_blocks,
                    cb=progress_callback
                )

                if is_ltx:
                    raw_frames = int(round(dur * fps))
                    k_total = max(1, round((raw_frames - 1) / stride))
                    num_frames = stride * k_total + 1

                    if not is_last and overlap > 0:
                        k_overlap = max(1, round(round(overlap * fps) / stride))
                        overlap_frames = k_overlap * stride
                        core_frames = max(stride, (num_frames - 1) - overlap_frames)
                        tail_frames = num_frames - core_frames
                        save_tail = True
                    else:
                        core_frames = num_frames
                        tail_frames = 0
                        save_tail = False

                    if prefix_video_path and os.path.exists(prefix_video_path):
                        start_img = None
                        pref_vid = prefix_video_path
                    else:
                        start_img = anchor_img_path
                        pref_vid = None

                    log(f"\n[{scene_name} : {b_name}] Engine: {generator.name.upper()}, Duration: {dur}s, Overlap: {overlap}s")
                    log(f"LTX quantized frames: total={num_frames}, core={core_frames}, tail_frames={tail_frames}")
                    log(f"Motion prompt: '{motion_prompt[:60]}...'")

                    req = VideoRequest(
                        image_path=start_img,
                        prefix_video_path=pref_vid,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        core_frames=core_frames,
                        save_tail=save_tail,
                        fps=fps,
                        out_video_name=out_core_name,
                        progress_callback=progress_callback
                    )
                    res = generator.generate(req)
                    if save_tail and res.tail_video_path and os.path.exists(res.tail_video_path):
                        prefix_video_path = res.tail_video_path
                    else:
                        prefix_video_path = None
                    block_core_frames = core_frames
                else:
                    k_dur = max(1, round(dur * fps / 4))
                    base_frames = 4 * k_dur + 1

                    if b_idx == 0:
                        wan_start_img = anchor_img_path
                        wan_pref_frames = 0
                        wan_num_frames = base_frames
                        wan_core_frames = base_frames
                    else:
                        wan_start_img = next_tail_image
                        wan_pref_frames = 1
                        wan_num_frames = base_frames
                        wan_core_frames = base_frames - 1

                    blk_steps = blk.get("steps") or effective_vid.get("steps")
                    blk_cfg = blk.get("cfg") or blk.get("cfg_scale") or effective_vid.get("cfg")

                    log(f"\n[{scene_name} : {b_name}] Engine: {generator.name.upper()}, Duration: {dur}s, Frames: {wan_num_frames} (core={wan_core_frames})")
                    log(f"Start anchor image: {os.path.basename(wan_start_img) if wan_start_img else 'None'}")
                    log(f"Motion prompt: '{motion_prompt[:60]}...'")

                    req = VideoRequest(
                        image_path=wan_start_img,
                        prefix_frames=wan_pref_frames,
                        core_frames=wan_core_frames,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=wan_num_frames,
                        fps=fps,
                        steps=blk_steps,
                        cfg=blk_cfg,
                        out_video_name=out_core_name,
                        progress_callback=progress_callback
                    )
                    res = generator.generate(req)
                    if not is_last and res.success and res.file_path:
                        tail_img_name = f"{block_prefix}_last.png"
                        tail_img_path = os.path.join(TEMP_DIR, tail_img_name)
                        ffmpeg_engine.extract_last_frame(res.file_path, tail_img_path)
                        next_tail_image = tail_img_path
                        log(f"Extracted pristine tail frame for next block anchor: {os.path.basename(next_tail_image)}")
                    block_core_frames = wan_core_frames

                if not res.success or not res.file_path:
                    raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {res.error}")

                processed_blocks += 1
                _PIPELINE_STATUS["processed_blocks"] = processed_blocks

                all_block_cores.append(res.file_path)
                actual_core_dur = block_core_frames / fps
                generated_block_metadata.append((res.file_path, actual_core_dur, overlap if not is_last else 0.0))

        # Clean up any leftover temporary tail videos
        if prefix_video_path and os.path.exists(prefix_video_path):
            try:
                os.remove(prefix_video_path)
            except Exception:
                pass

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg (-c copy)...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

        # Final movie assembly across all blocks in a single lossless Stream Copy (-c copy) pass
        if output_name:
            clean_out = os.path.splitext(os.path.basename(output_name))[0]
            final_name = f"{run_prefix}_{clean_out}.mp4"
        else:
            final_name = f"{run_prefix}_final.mp4"

        final_path = os.path.join(OUTPUT_VID_DIR, final_name)
        ffmpeg_engine.concat_videos(all_block_cores, final_path, smooth_transitions=False)
        total_duration = ffmpeg_engine.get_duration(final_path)

        # Poster handling: prioritize pristine anchor keyframe from Scene 1
        poster_filename = f"{os.path.splitext(final_name)[0]}_poster.png"
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
            hud_name = f"{os.path.splitext(final_name)[0]}_hud.mp4"
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
