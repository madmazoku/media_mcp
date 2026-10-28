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
        scenario_dict.pop("dry_run", None)

    if base_file_path:
        p = Path(base_file_path)
        if not p.is_file():
            found = False
            for parent_cand in [Path.cwd(), Path(config.WORKFLOWS_DIR)]:
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
        base_dict.pop("dry_run", None)
        scenario_dict = deep_merge(base_dict, scenario_dict)

    if request_overrides:
        clean_overrides = dict(request_overrides)
        clean_overrides.pop("scenario_path", None)
        clean_overrides.pop("dry_run", None)
        scenario_dict = deep_merge(scenario_dict, clean_overrides)

    if not isinstance(scenario_dict, dict) or "scenes" not in scenario_dict:
        raise ValueError("Scenario must be a JSON object matching schemas/scenario.json containing 'scenes' list")

    validate_scenario(scenario_dict)
    return scenario_dict


def _normalize_keyframe_config(raw_kf: Any, image_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_kf, dict):
        return {}
    if raw_kf.get("keyframe_path"):
        return {"keyframe_path": raw_kf["keyframe_path"]}

    kf = dict(raw_kf)
    root_image_engine = image_cfg.get("engine", "sdxl")
    root_image_model = image_cfg.get("model", "")
    root_image_negative = image_cfg.get("negative_prompt", "")
    root_image_prompt = image_cfg.get("prompt", "")
    root_image_loras = image_cfg.get("loras", [])

    kf["engine"] = root_image_engine
    kf["model"] = root_image_model

    for k in ("width", "height", "steps", "cfg", "guidance", "sampler", "seed", "fit"):
        if k not in kf and k in image_cfg:
            kf[k] = image_cfg[k]

    combined_prompt = combine_prompts(root_image_prompt, kf.get("prompt")) if (kf.get("prompt") or root_image_prompt) else ""
    if combined_prompt:
        kf["prompt"] = combined_prompt
    if kf.get("negative_prompt") or root_image_negative:
        kf["negative_prompt"] = combine_negative_prompts(root_image_negative, kf.get("negative_prompt"))

    if kf.get("image_path"):
        if "denoise" in kf:
            pass
        elif "denoise" in image_cfg and (combined_prompt or kf.get("mask_path")):
            kf["denoise"] = image_cfg["denoise"]
        elif combined_prompt or kf.get("mask_path"):
            kf["denoise"] = 0.65
        # If no prompt and no mask_path, do not inject denoise so passive crop triggers
    else:
        kf["denoise"] = 1.0

    kf["loras"] = _merge_loras(root_image_loras, kf.get("loras"))
    return kf

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

        anchor = _normalize_keyframe_config(sc["anchor_image"], image_cfg)

        raw_blocks = sc["blocks"]
        norm_blocks = []
        for b_idx, blk in enumerate(raw_blocks):
            b_prompt = combine_prompts(root_video_prompt, blk["prompt"])
            b_neg = combine_negative_prompts(sc_negative, blk.get("negative_prompt"))
            b_loras = _merge_loras(root_video_loras, blk.get("loras"))
            raw_end_img = blk.get("end_image")
            norm_end_img = _normalize_keyframe_config(raw_end_img, image_cfg) if raw_end_img is not None else None

            norm_blocks.append({
                "name": blk.get("name", f"Block #{b_idx + 1}"),
                "duration": float(blk.get("duration", 5.0)),
                "overlap": float(blk.get("overlap") or 0.0) if b_idx < len(raw_blocks) - 1 else 0.0,
                "prompt": b_prompt,
                "negative_prompt": b_neg,
                "end_image": norm_end_img,
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


ENGINE_MAX_DURATIONS: Dict[str, float] = {
    "wan-fast": 6.0,
    "wan-full": 6.0,
    "ltx": 16.0,
}


def collect_scenario_timing_errors(scenes: List[Dict[str, Any]], default_engine: str = "wan-fast") -> List[str]:
    """
    Validates dynamic block merging timing constraints inside each scene (montage):
    1. overlap_duration < clip_duration
    2. prefix_duration + clip_duration < engine_limit (immutable hardware/architecture constant per engine)
    Returns a list of human-readable error messages.
    """
    errors: List[str] = []
    for s_idx, sc in enumerate(scenes):
        sc_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
        sc_engine = str(sc.get("engine") or "").lower()
        blocks = sc.get("blocks", [])
        for b_idx, blk in enumerate(blocks):
            b_name = blk.get("name", f"Block #{b_idx + 1}")
            b_engine = str(blk.get("engine") or sc_engine or default_engine).lower()
            limit = float(ENGINE_MAX_DURATIONS.get(b_engine, 6.0))
            clip_duration = float(blk.get("duration", 5.0))
            prefix_duration = float(blocks[b_idx - 1].get("overlap") or 0.0) if b_idx > 0 else 0.0
            overlap_duration = float(blk.get("overlap") or 0.0) if b_idx < len(blocks) - 1 else 0.0

            if overlap_duration >= clip_duration:
                errors.append(
                    f"scene #{s_idx + 1} ('{sc_name}'): block #{b_idx + 1} ('{b_name}'): failed video: Invalid overlap: overlap_duration ({overlap_duration}s) >= "
                    f"clip_duration ({clip_duration}s). Condition 'overlap_duration < clip_duration' is required."
                )

            if prefix_duration + clip_duration >= limit:
                errors.append(
                    f"scene #{s_idx + 1} ('{sc_name}'): block #{b_idx + 1} ('{b_name}'): failed video: Generation window exceeded: prefix_duration ({prefix_duration}s) + "
                    f"clip_duration ({clip_duration}s) = {prefix_duration + clip_duration:.2f}s >= "
                    f"engine limit ({limit}s for engine '{b_engine}'). Condition 'prefix_duration + clip_duration < engine limit' is required."
                )
    return errors


def _validate_keyframe_spec(
    kf: Dict[str, Any],
    label_prefix: str,
    entity_label: str,
    ar_vid: float,
    video_width: int,
    video_height: int,
    tolerance: float = 0.05,
    allow_runtime_refs: bool = False
) -> List[str]:
    from PIL import Image
    errors: List[str] = []
    if kf is None or not isinstance(kf, dict):
        return errors

    fit_mode = kf.get("fit")
    img_ref = kf.get("keyframe_path") or kf.get("image_path")

    if img_ref in ("$anchor", "$start"):
        if not allow_runtime_refs:
            errors.append(
                f"{label_prefix}: Runtime reference '{img_ref}' is only permitted inside block end_image, "
                f"not in scene anchor_image."
            )
        elif kf.get("image_path") == img_ref and not kf.get("prompt") and not kf.get("mask_path"):
            errors.append(
                f"{label_prefix}: Prompt is required when using '{img_ref}' in end_image image_path. "
                f"To reuse the frame unmodified, use keyframe_path instead."
            )
        return errors

    if img_ref:
        p = Path(img_ref)
        if not p.is_file():
            for base in [Path.cwd(), Path(config.OUTPUT_IMG_DIR), Path(config.WORKFLOWS_DIR)]:
                cand = base / img_ref
                if cand.is_file():
                    p = cand
                    break
        if p.is_file():
            if not fit_mode:
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                    if w > 0 and h > 0:
                        ar_img = w / h
                        diff_pct = abs(ar_img - ar_vid) / ar_vid
                        if diff_pct > tolerance:
                            errors.append(
                                f"{label_prefix}: Aspect ratio mismatch: "
                                f"{entity_label.lower()} '{img_ref}' has resolution {w}x{h} (aspect ratio {ar_img:.2f}), "
                                f"but target video is {video_width}x{video_height} (aspect ratio {ar_vid:.2f}, "
                                f"difference {diff_pct * 100:.1f}% > {tolerance * 100:.1f}%). "
                                f"{entity_label} aspect ratio must match video aspect ratio to prevent distortion."
                            )
                except Exception as ex:
                    errors.append(f"{label_prefix}: Failed to inspect {entity_label.lower()} '{img_ref}': {ex}")
        else:
            errors.append(f"{label_prefix}: {entity_label} file not found on disk: {img_ref}")

    elif not kf.get("prompt"):
        errors.append(
            f"{label_prefix}: Prompt is required for {entity_label.lower()} "
            f"generation when neither keyframe_path nor image_path is specified."
        )

    elif kf.get("width") and kf.get("height") and not fit_mode:
        w = int(kf["width"])
        h = int(kf["height"])
        if w > 0 and h > 0:
            ar_img = w / h
            diff_pct = abs(ar_img - ar_vid) / ar_vid
            if diff_pct > tolerance:
                errors.append(
                    f"{label_prefix}: Aspect ratio mismatch: "
                    f"resolution {w}x{h} (aspect ratio {ar_img:.2f}) does not match target video "
                    f"{video_width}x{video_height} (aspect ratio {ar_vid:.2f}, difference {diff_pct * 100:.1f}% > {tolerance * 100:.1f}%)."
                )

    return errors

def collect_scenario_aspect_ratio_errors(
    scenes: List[Dict[str, Any]],
    video_width: int = 832,
    video_height: int = 480,
    tolerance: float = 0.05
) -> List[str]:
    """
    Validates anchor and block end keyframes for each scene:
    1. Existence of keyframe_path / image_path on disk (always checked, even if fit is set).
    2. Aspect ratio matching target video resolution within tolerance (checked when fit is not set).
    3. Mandatory prompt when neither keyframe_path nor image_path is specified (pure txt2img).
    """
    errors: List[str] = []
    if video_height <= 0 or video_width <= 0:
        return errors

    ar_vid = video_width / video_height

    for s_idx, sc in enumerate(scenes):
        sc_name = sc.get("scene_name") or sc.get("name", f"Scene #{s_idx + 1}")
        anchor = sc.get("anchor_image", {})
        errors.extend(_validate_keyframe_spec(
            kf=anchor,
            label_prefix=f"scene #{s_idx + 1} ('{sc_name}'): failed anchor_image",
            entity_label="Anchor image",
            ar_vid=ar_vid,
            video_width=video_width,
            video_height=video_height,
            tolerance=tolerance,
            allow_runtime_refs=False
        ))

        blocks = sc.get("blocks", [])
        for b_idx, blk in enumerate(blocks):
            b_name = blk.get("name", f"Block #{b_idx + 1}")
            end_img = blk.get("end_image")
            if end_img is not None:
                errors.extend(_validate_keyframe_spec(
                    kf=end_img,
                    label_prefix=f"scene #{s_idx + 1} ('{sc_name}'): block #{b_idx + 1} ('{b_name}'): failed end_image",
                    entity_label="End image",
                    ar_vid=ar_vid,
                    video_width=video_width,
                    video_height=video_height,
                    tolerance=tolerance,
                    allow_runtime_refs=True
                ))

    return errors


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def prepare_scenario_pipeline(
    scenario_input: Any = None,
    scenario_path: Optional[str] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Performs fast synchronous pre-flight preparation and validation of a video scenario.
    Validates JSON schema, timing laws, aspect ratios, resolves seeds, assigns canonical run_prefix,
    and immediately writes {run_prefix}-spec.json to disk.
    Returns prepared pipeline dictionary ready for queued background execution.
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

    timing_errors = collect_scenario_timing_errors(scenes, default_engine=v_engine)
    ar_errors = collect_scenario_aspect_ratio_errors(scenes, video_width=width, video_height=height)
    combined_errors = timing_errors + ar_errors
    if combined_errors:
        scen_title = resolved_scenario_dict.get("name") or (Path(scenario_path).stem if scenario_path else "untitled")
        raise ValueError(f"failed scenario '{scen_title}': {'; '.join(combined_errors)}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}-{run_hash}"

    # Write specification file immediately to disk
    spec_filename = f"{run_prefix}-spec.json"
    workflows_dir = Path(config.WORKFLOWS_DIR) if config.WORKFLOWS_DIR else Path(config.OUTPUT_DIR) / 'workflows'
    workflows_dir.mkdir(parents=True, exist_ok=True)
    spec_path = (workflows_dir / spec_filename).resolve().as_posix()
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(resolved_scenario_dict, f, indent=2, ensure_ascii=False)
    log(f"Pre-flight: Saved resolved scenario specification: {spec_path}")

    total_scenes = len(scenes)
    total_blocks = sum(len(sc.get("blocks", [])) for sc in scenes)

    return {
        "run_prefix": run_prefix,
        "spec_path": spec_path,
        "resolved_scenario_dict": resolved_scenario_dict,
        "scenes": scenes,
        "effective_vid": effective_vid,
        "v_engine": v_engine,
        "v_model": v_model,
        "width": width,
        "height": height,
        "fps": fps,
        "total_scenes": total_scenes,
        "total_blocks": total_blocks,
    }


def execute_scenario_pipeline(
    scenario_input: Any = None,
    scenario_path: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
    prepared_pipeline_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a multi-scene video scenario using polymorphic generator dispatch and 5D Latent-Chaining.
    Supports base loading via scenario_path, top-level 'image' and 'video' structured parameters and Deep Merge.
    Can accept pre-validated prepared_pipeline_data from prepare_scenario_pipeline.
    """
    if prepared_pipeline_data:
        prep = prepared_pipeline_data
    else:
        prep = prepare_scenario_pipeline(
            scenario_input=scenario_input,
            scenario_path=scenario_path,
            image=image,
            video=video,
            request_overrides=request_overrides
        )

    run_prefix = prep["run_prefix"]
    spec_path = prep["spec_path"]
    resolved_scenario_dict = prep["resolved_scenario_dict"]
    scenes = prep["scenes"]
    effective_vid = prep["effective_vid"]
    v_engine = prep["v_engine"]
    v_model = prep["v_model"]
    width = prep["width"]
    height = prep["height"]
    fps = prep["fps"]
    total_scenes = prep["total_scenes"]
    total_blocks = prep["total_blocks"]

    log(
        f"Executing scenario pipeline with {total_scenes} scenes "
        f"(run_id={run_prefix}, video_engine={v_engine}, model={v_model}, {width}x{height} @ {fps}fps)..."
    )

    all_block_cores: List[str] = []
    pipeline_warnings: List[str] = []
    first_poster_path: Optional[str] = None
    first_anchor_res_meta: Dict[str, Any] = {}
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

            try:
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
                    raise RuntimeError(f"failed anchor_image: {anchor_img_path}")

                if isinstance(anchor_res_meta, dict) and anchor_res_meta.get("warnings"):
                    for w in anchor_res_meta["warnings"]:
                        if w not in pipeline_warnings:
                            pipeline_warnings.append(w)

                if s_idx == 0:
                    first_poster_path = anchor_img_path
                    first_anchor_res_meta = anchor_res_meta or {}

                next_tail_latent: Optional[str] = None
                next_tail_latent_count = 0
                next_tail_image: Optional[str] = None
                last_block_tail_image: Optional[str] = None

                for b_idx, blk in enumerate(blocks):
                    if is_interrupted():
                        raise RuntimeError("Scenario execution was interrupted by user request.")

                    block_idx = b_idx + 1
                    b_name = blk.get("name", f"Block {block_idx}")

                    try:
                        dur = float(blk.get("duration", 5.0))
                        is_last = (b_idx == len(blocks) - 1)
                        overlap = float(blk.get("overlap") or 0.0) if not is_last else 0.0

                        block_prompt = blk["prompt"]
                        b_neg = blk.get("negative_prompt") or ""
                        b_engine_raw = (blk.get("engine") or v_engine).lower()
                        b_model = blk.get("model") or v_model

                        b_settings = dict(effective_vid)
                        b_settings["model"] = b_model
                        if blk.get("loras"):
                            b_settings["loras"] = blk["loras"]

                        generator = create_video_generator(b_engine_raw, settings=b_settings)
                        stride = int(getattr(generator, "temporal_stride", 4))

                        if b_engine_raw.startswith("wan") and dur > 5.5:
                            log(
                                f"Warning: [{scene_name} : {b_name}] Duration ({dur}s) exceeds Wan 2.2 native training horizon "
                                f"(5.06s / 81 frames). Consider keeping blocks <= 5.0s for optimal image quality across chaining."
                            )

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
                        log(f"Prompt: '{block_prompt[:60]}...'")

                        current_block_start_image = anchor_img_path if b_idx == 0 else (last_block_tail_image or anchor_img_path)

                        end_img_meta = blk.get("end_image")
                        block_end_img_path = None
                        if end_img_meta:
                            block_end_prefix = f"{run_prefix}-s{scene_idx:03d}-b{block_idx:03d}"
                            block_end_filename = f"{block_end_prefix}-end.png"
                            end_meta_copy = dict(end_img_meta)
                            end_meta_copy["run_prefix"] = block_end_prefix
                            if not end_meta_copy.get("custom_filename"):
                                end_meta_copy["custom_filename"] = (temp_dir / block_end_filename).resolve().as_posix()

                            # Dynamic runtime resolution for $anchor and $start references
                            if end_meta_copy.get("image_path") == "$anchor":
                                end_meta_copy["image_path"] = anchor_img_path
                                log(f"Resolved end_image image_path reference '$anchor' -> {Path(anchor_img_path).name}")
                            elif end_meta_copy.get("image_path") == "$start":
                                end_meta_copy["image_path"] = current_block_start_image
                                log(f"Resolved end_image image_path reference '$start' -> {Path(current_block_start_image).name}")

                            if end_meta_copy.get("keyframe_path") == "$anchor":
                                end_meta_copy["keyframe_path"] = anchor_img_path
                                log(f"Resolved end_image keyframe_path reference '$anchor' -> {Path(anchor_img_path).name}")
                            elif end_meta_copy.get("keyframe_path") == "$start":
                                end_meta_copy["keyframe_path"] = current_block_start_image
                                log(f"Resolved end_image keyframe_path reference '$start' -> {Path(current_block_start_image).name}")

                            _PIPELINE_STATUS["phase"] = f"end_image_generation ({scene_name} - {b_name})"
                            notify_progress(
                                f"Scene {scene_idx}/{total_scenes} [{scene_name}] - {b_name}: Preparing end keyframe...",
                                progress=processed_blocks,
                                total=total_blocks,
                                cb=progress_callback
                            )
                            ok_end, end_path, _, end_res_meta = img_engine_mod.generate_anchor_frame(
                                anchor_meta=end_meta_copy,
                                width=width,
                                height=height
                            )
                            if not ok_end:
                                raise RuntimeError(f"failed end_image: {end_path}")
                            block_end_img_path = end_path
                            if isinstance(end_res_meta, dict) and end_res_meta.get("warnings"):
                                for w in end_res_meta["warnings"]:
                                    if w not in pipeline_warnings:
                                        pipeline_warnings.append(w)

                        save_tail_image = not is_last

                        req = VideoRequest(
                            image_path=start_img,
                            end_image_path=block_end_img_path,
                            prefix_latent_path=pref_latent,
                            prefix_frames=pref_pixel_frames,
                            core_frames=core_frames,
                            tail_frames=tail_frames,
                            save_tail=save_tail,
                            save_tail_image=save_tail_image,
                            prompt=block_prompt,
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
                            raise RuntimeError(f"failed video: {res.error}")

                        if not is_last:
                            if res.tail_image_path and Path(res.tail_image_path).is_file():
                                last_block_tail_image = res.tail_image_path

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

                    except Exception as b_ex:
                        if is_interrupted() or "interrupted by user request" in str(b_ex):
                            raise
                        raise RuntimeError(f"block #{block_idx} ('{b_name}'): {b_ex}") from b_ex

            except Exception as sc_ex:
                if is_interrupted() or "interrupted by user request" in str(sc_ex):
                    raise
                raise RuntimeError(f"scene #{scene_idx} ('{scene_name}'): {sc_ex}") from sc_ex

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg (-c copy)...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

        final_name = f"{run_prefix}-final.mp4"
        out_vid_dir = Path(config.OUTPUT_VID_DIR) if config.OUTPUT_VID_DIR and Path(config.OUTPUT_VID_DIR).is_dir() else (temp_dir / "videos")
        out_vid_dir.mkdir(parents=True, exist_ok=True)
        final_path = (out_vid_dir / final_name).resolve().as_posix()
        try:
            ffmpeg_engine.concat_videos(all_block_cores, final_path)
            total_duration = ffmpeg_engine.get_duration(final_path)
        except Exception as f_ex:
            raise RuntimeError(f"failed assembly: {f_ex}") from f_ex

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
            "blocks_count": total_blocks,
            "anchor_metadata": first_anchor_res_meta
        }
        if pipeline_warnings:
            result_payload["warnings"] = pipeline_warnings
        return result_payload

    except Exception as ex:
        if is_interrupted() or "interrupted by user request" in str(ex):
            raise
        scen_title = resolved_scenario_dict.get("name") or (Path(scenario_path).stem if scenario_path else "untitled")
        raise RuntimeError(f"failed scenario '{scen_title}': {ex}") from ex
    finally:
        _INTERRUPT_REQUESTED = False
        if comfy_client.is_interrupted():
            comfy_client.reset_interrupt()
        _PIPELINE_STATUS["is_running"] = False
        _PIPELINE_STATUS["phase"] = "idle"
