#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tool Dispatcher & Execution Engine for media_mcp.
Decouples core business logic from transport layers (CLI, MCP Stdio, SSE HTTP).
Handles spec resolution, schema validation, fail-fast configuration checks,
progress routing, and standardized MCP/CLI output formatting.
"""

from __future__ import annotations
import os
import sys
import json
import time
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

from . import (
    config,
    schema_validator,
    discovery,
    comfy_client,
    a1111_client,
)


def log(msg: str):
    sys.stderr.write(f"[core.tools] {msg}\n")
    sys.stderr.flush()


CANONICAL_TOOLS = [
    "image_generate",
    "video_generate",
    "video_validate",
    "media_get_models",
    "media_get_samplers",
    "media_get_loras",
    "media_get_status",
    "media_interrupt",
    "media_reload_modules",
]

CONFIG_REQUIRED_TOOLS = {"image_generate", "video_generate"}


# ---------------------------------------------------------------------------
# Deep Merge & File Helpers
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


def load_spec_file(path_str: str) -> Dict[str, Any]:
    """
    Locates and loads a JSON specification/scenario file on disk.
    Searches absolute path, relative to cwd, WORKFLOWS_DIR, and REPO_DIR.
    """
    p = Path(path_str)
    if not p.is_file():
        found = False
        candidates = []
        if config.WORKFLOWS_DIR:
            candidates.append(Path(config.WORKFLOWS_DIR))
        if config.REPO_DIR:
            candidates.append(Path(config.REPO_DIR))
        candidates.append(Path.cwd())
        for parent_cand in candidates:
            cand = parent_cand / path_str
            if cand.is_file():
                p = cand
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Specification file not found on disk: {path_str}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def reload_core_modules() -> List[str]:
    """
    Dynamically reloads all loaded modules in sys.modules under 'core' in reverse dependency order,
    ensuring code updates take effect immediately without restarting servers.
    """
    core_mods = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
    core_mods.sort(key=lambda s: s.count("."), reverse=True)
    reloaded = []
    for mod_name in core_mods:
        m = sys.modules.get(mod_name)
        if m and hasattr(m, "__file__") and m.__file__:
            try:
                importlib.reload(m)
                reloaded.append(mod_name)
            except Exception as e:
                log(f"Warning: Failed to reload {mod_name}: {e}")
    return reloaded


# ---------------------------------------------------------------------------
# Individual Tool Executors
# ---------------------------------------------------------------------------

def execute_image_generate(args: Dict[str, Any], progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    from . import image_engine
    if progress_cb:
        image_engine.set_global_progress_callback(progress_cb)

    success, img_path, b64_data, metadata = image_engine.generate_image(args)
    if not success or not img_path:
        err_msg = metadata.get("error", "Image generation failed")
        return {
            "content": [{"type": "text", "text": f"Error: {err_msg}"}],
            "error": err_msg,
            "isError": True,
            "metadata": metadata
        }

    run_prefix = metadata.get("run_prefix", "")
    spec_path = metadata.get("spec_path", "")
    seed = metadata.get("seed")
    engine_name = metadata.get("engine", "image").upper()
    width = metadata.get("width", "N/A")
    height = metadata.get("height", "N/A")
    model = metadata.get("model", "default")
    file_uri = Path(img_path).resolve().as_uri()

    summary_text = (
        f"Image Generated Successfully ({engine_name}):\n"
        f"- Run Prefix: `{run_prefix}`\n"
        f"- Spec: `{spec_path}`\n"
        f"- Output: `{img_path}`\n"
        f"- Resolution: {width}x{height} | Seed: {seed} | Model: `{model}`\n\n"
        f"File link: [Open Image]({file_uri})\n"
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": summary_text}]
    if b64_data:
        content.append({"type": "image", "data": b64_data, "mimeType": "image/png"})

    return {
        "content": content,
        "run_prefix": run_prefix,
        "image_path": img_path,
        "spec_path": spec_path,
        "seed": seed,
        "width": width,
        "height": height,
        "metadata": metadata
    }


def execute_video_generate(args: Dict[str, Any], progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    from . import video_pipeline
    if progress_cb:
        video_pipeline.set_global_progress_callback(progress_cb)

    enable_hud = bool(args.get("enable_hud", False))
    res = video_pipeline.execute_scenario_pipeline(
        scenario_input=args,
        enable_hud=enable_hud,
        progress_callback=progress_cb
    )

    run_prefix = res.get("run_prefix", "")
    final_path = res["video_path"]
    spec_path = res.get("spec_path", "")
    hud_path = res.get("hud_video_path")
    poster_path = res.get("poster_path")
    poster_b64 = res.get("poster_base64")
    dur = res.get("duration", 0.0)
    w = res.get("width", 832)
    h = res.get("height", 480)
    fps = res.get("fps", 16)
    scenes_cnt = res.get("scenes_count", 1)

    file_uri = Path(final_path).resolve().as_uri()
    hud_uri = Path(hud_path).resolve().as_uri() if hud_path else None

    result_text = (
        f"Multi-Scene Video Scenario Rendered Successfully:\n"
        f"- Run Prefix: `{run_prefix}`\n"
        f"- Spec: `{spec_path}`\n"
        f"- Output: `{final_path}`\n"
        f"- Poster: `{poster_path}`\n"
        f"- Scenes: {scenes_cnt} | Total Duration: {dur:.2f}s\n"
        f"- Resolution: {w}x{h} @ {fps} fps\n\n"
        f"File link: [Open Final Video]({file_uri})\n"
    )
    if hud_path and hud_uri:
        result_text += f"- Diagnostic HUD: [Open HUD Video]({hud_uri})\n"

    content: List[Dict[str, Any]] = [{"type": "text", "text": result_text}]
    if poster_b64:
        content.append({"type": "image", "data": poster_b64, "mimeType": "image/png"})

    return {
        "content": content,
        "run_prefix": run_prefix,
        "video_path": final_path,
        "spec_path": spec_path,
        "poster_path": poster_path,
        "hud_video_path": hud_path,
        "duration": dur,
        "width": w,
        "height": h,
        "fps": fps,
        "scenes_count": scenes_cnt
    }


def execute_video_validate(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    errors = []
    warnings = []
    scene_summaries = []
    total_projected_dur = 0.0

    try:
        scenes = video_pipeline.normalize_scenario_structure(args)
    except Exception as ex:
        return {
            "content": [{"type": "text", "text": f"Scenario Normalization/Schema Error:\n{ex}"}],
            "is_valid": False,
            "errors": [str(ex)],
            "warnings": [],
            "scenes": []
        }

    v_cfg = args.get("video", {})
    vid_w = int(v_cfg.get("width", 832))
    vid_h = int(v_cfg.get("height", 480))
    timing_errors = video_pipeline.collect_scenario_timing_errors(scenes)
    ar_errors = video_pipeline.collect_scenario_aspect_ratio_errors(scenes, video_width=vid_w, video_height=vid_h)
    errors.extend(timing_errors)
    errors.extend(ar_errors)

    for sc in scenes:
        s_name = sc.get("name", "Scene")
        anchor = sc.get("anchor_image", {})
        blocks = sc.get("blocks", [])

        has_keyframe = bool(anchor.get("keyframe_path"))
        has_image = bool(anchor.get("image_path"))
        has_prompt = bool(anchor.get("prompt"))

        if not has_keyframe and not has_image and not has_prompt:
            errors.append(f"{s_name}: Neither 'keyframe_path', 'image_path' nor 'prompt' specified for anchor_image.")

        anchor_type = "Ready Keyframe" if has_keyframe else ("img2img Refinement" if has_image else "txt2img Prompt")
        s_dur = sum(float(blk.get("duration", 5.0)) for blk in blocks)
        total_projected_dur += s_dur
        scene_summaries.append({
            "name": s_name,
            "blocks_count": len(blocks),
            "projected_duration": round(s_dur, 2),
            "anchor_type": anchor_type
        })

    is_valid = (len(errors) == 0)
    report = (
        f"Scenario Validation Report:\n"
        f"- Status: {'VALID [OK]' if is_valid else 'INVALID [ERRORS DETECTED]'}\n"
        f"- Scenes count: {len(scenes)}\n"
        f"- Projected total duration: ~{total_projected_dur:.2f} seconds\n"
    )

    if errors:
        report += "\nErrors:\n" + "\n".join(f"  [X] {e}" for e in errors)
    if warnings:
        report += "\nWarnings:\n" + "\n".join(f"  [!] {w}" for w in warnings)

    report += "\nScene Breakdown:\n"
    for s in scene_summaries:
        report += f"- **{s['name']}** ({s['blocks_count']} blocks, ~{s['projected_duration']}s) | Anchor: {s['anchor_type']}\n"

    return {
        "content": [{"type": "text", "text": report}],
        "is_valid": is_valid,
        "total_duration": total_projected_dur,
        "errors": errors,
        "warnings": warnings,
        "scenes": scene_summaries
    }


def execute_media_get_models(args: Dict[str, Any]) -> Dict[str, Any]:
    type_filter = args.get("type")
    engine_filter = args.get("engine")
    tree = discovery.get_models_tree(type_filter=type_filter, engine_filter=engine_filter)
    json_text = json.dumps(tree, indent=2, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": json_text}],
        **tree
    }


def execute_media_get_samplers(args: Dict[str, Any]) -> Dict[str, Any]:
    engine = args["engine"]
    model = args["model"]
    samplers = discovery.get_samplers_for_model(engine=engine, model=model)
    json_text = json.dumps(samplers, indent=2, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": json_text}],
        "samplers": samplers
    }


def execute_media_get_loras(args: Dict[str, Any]) -> Dict[str, Any]:
    engine = args["engine"]
    model = args["model"]
    pattern = args.get("pattern")
    try:
        loras = discovery.get_loras_for_model(engine=engine, model=model, pattern=pattern)
    except ValueError as ex:
        return {
            "error": str(ex),
            "isError": True,
            "content": [{"type": "text", "text": f"Error: {ex}"}],
            "loras": []
        }
    json_text = json.dumps(loras, indent=2, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": json_text}],
        "loras": loras
    }


def execute_media_get_status(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    status_lines = []

    pipe_st = video_pipeline.get_pipeline_status()
    if pipe_st.get("is_running"):
        status_lines.append(
            f"🎬 Video Pipeline: RUNNING\n"
            f"  - Target: {pipe_st.get('scenario_name')}\n"
            f"  - Scene: {pipe_st.get('scene_index')}/{pipe_st.get('total_scenes')} ({pipe_st.get('scene_name')})\n"
            f"  - Block: {pipe_st.get('block_name', 'N/A')}\n"
            f"  - Phase: {pipe_st.get('phase')}\n"
            f"  - Elapsed: {pipe_st.get('elapsed_seconds')}s"
        )
    else:
        status_lines.append("🎬 Video Pipeline: IDLE")

    a1111_avail = a1111_client.is_available()
    if a1111_avail:
        cur_m = a1111_client.get_current_model()
        p_data = a1111_client.get_progress()
        prog = float(p_data.get("progress", 0.0))
        st_state = p_data.get("state", {})
        job = st_state.get("job", "")
        status_lines.append(
            f"🎨 A1111 WebUI: ONLINE (Model: `{cur_m}`)\n"
            f"  - Rendering Progress: {int(prog * 100)}% ({job or 'Idle'})"
        )
    else:
        status_lines.append(f"🎨 A1111 WebUI: OFFLINE ({config.SD_WEBUI_URL} unreachable)")

    comfy_avail = comfy_client.is_available()
    if comfy_avail:
        q_data = comfy_client.get_queue_status()
        exec_info = q_data.get("queue_executing", [])
        pending = q_data.get("queue_pending", [])
        status_lines.append(
            f"⚡ ComfyUI: ONLINE ({config.COMFY_URL})\n"
            f"  - Active: {len(exec_info)} job(s) | Pending: {len(pending)} job(s)"
        )
    else:
        status_lines.append(f"⚡ ComfyUI: OFFLINE ({config.COMFY_URL} unreachable)")

    text_resp = "\n\n".join(status_lines)
    return {
        "content": [{"type": "text", "text": text_resp}],
        "pipeline": pipe_st,
        "a1111_available": a1111_avail,
        "comfy_available": comfy_avail
    }


def execute_media_interrupt(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    comfy_client.interrupt()
    a1111_client.interrupt()
    video_pipeline.request_interrupt()
    msg = "Interrupt signals dispatched to ComfyUI, A1111 WebUI, and Active Video Pipelines."
    log(msg)
    return {"content": [{"type": "text", "text": msg}], "interrupted": True}


def execute_media_reload_modules(args: Dict[str, Any]) -> Dict[str, Any]:
    reloaded = reload_core_modules()
    msg = f"Hot-reloaded {len(reloaded)} core modules in reverse dependency order:\n" + "\n".join(f"- {m}" for m in reloaded)
    log(msg)
    return {
        "content": [{"type": "text", "text": msg}],
        "reloaded_modules": reloaded
    }


TOOL_REGISTRY = {
    "image_generate": execute_image_generate,
    "video_generate": execute_video_generate,
    "video_validate": execute_video_validate,
    "media_get_models": execute_media_get_models,
    "media_get_samplers": execute_media_get_samplers,
    "media_get_loras": execute_media_get_loras,
    "media_get_status": execute_media_get_status,
    "media_interrupt": execute_media_interrupt,
    "media_reload_modules": execute_media_reload_modules,
}


def get_tools_list() -> List[Dict[str, Any]]:
    return [
        {
            "name": "image_generate",
            "description": "Generate high-quality images via SDXL/Pony (Automatic1111) or FLUX.1/FLUX.2 (ComfyUI) using a unified image_spec dictionary.",
            "inputSchema": schema_validator.get_mcp_tool_schema("image_generate")
        },
        {
            "name": "video_generate",
            "description": "Generate multi-scene or single-block videos using Wan 2.2 Fast/Full or LTX-2.3 with Latent-Chaining and True Forced Prefix continuity. Takes scenario_path and/or inline scenario scenes.",
            "inputSchema": schema_validator.get_mcp_tool_schema("video_generate")
        },
        {
            "name": "video_validate",
            "description": "Dry-run validation of a video scenario against JSON Schema and timing constraints without executing generation. Takes scenario_path and/or inline scenario scenes.",
            "inputSchema": schema_validator.get_mcp_tool_schema("video_validate")
        },
        {
            "name": "media_get_models",
            "description": "Inspect installed checkpoint and UNET models across image and video engines. Returns invariant tree structure filtered by type or engine.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_get_models")
        },
        {
            "name": "media_get_samplers",
            "description": "Get recommended and compatible samplers for a specific engine and model. Index 0 in returned list is the recommended default.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_get_samplers")
        },
        {
            "name": "media_get_loras",
            "description": "Inspect compatible LoRA models and their trained activation triggers for a specific engine and model. Supports optional 'pattern' (case-insensitive Python regex) to filter by LoRA name or triggers.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_get_loras")
        },
        {
            "name": "media_get_status",
            "description": "Check generation progress, queue depths, and current pipeline execution status across backends.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_get_status")
        },
        {
            "name": "media_interrupt",
            "description": "Interrupt and cancel currently running image and video generations across all engines.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_interrupt")
        },
        {
            "name": "media_reload_modules",
            "description": "Developer tool: hot-reload all python modules under 'core' in memory on demand.",
            "inputSchema": schema_validator.get_mcp_tool_schema("media_reload_modules")
        }
    ]


# ---------------------------------------------------------------------------
# Central Entry Point for all Transports (CLI, MCP, SSE)
# ---------------------------------------------------------------------------

def execute_tool(
    tool_name: str,
    payload: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Unified entry point for all media_mcp tools.
    Transport layers (CLI, Stdio MCP, SSE HTTP) MUST delegate directly to this function.
    Guarantees identical request processing, validation, spec resolution, and execution.
    """
    if tool_name not in TOOL_REGISTRY:
        return {
            "error": f"Unknown tool '{tool_name}'. Available tools: {CANONICAL_TOOLS}",
            "isError": True,
            "content": [{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}]
        }

    args = dict(payload or {})

    # 1. Environment configuration check (Fail-Fast)
    if tool_name in CONFIG_REQUIRED_TOOLS and not config.is_configured():
        err_msg = (
            f"Tool '{tool_name}' requires an environment configuration file.\n"
            f"Please specify --config <path_to_config.json> or set MEDIA_CONFIG environment variable."
        )
        return {
            "error": err_msg,
            "isError": True,
            "content": [{"type": "text", "text": f"Configuration Error:\n{err_msg}"}]
        }

    # 2. Schema validation against canonical JSON Schema
    try:
        schema_validator.validate_mcp_args(tool_name, args)
    except schema_validator.SchemaValidationError as ex:
        err_details = "\n".join(f"- {e}" for e in ex.errors)
        return {
            "error": f"Schema validation failed for tool '{tool_name}':\n{err_details}",
            "errors": ex.errors,
            "isError": True,
            "content": [{"type": "text", "text": f"Validation Error for '{tool_name}':\n{err_details}"}]
        }
    except Exception as ex:
        return {
            "error": f"Validation exception: {ex}",
            "isError": True,
            "content": [{"type": "text", "text": f"Validation Exception: {ex}"}]
        }

    # 3. Execution via registered tool handler
    handler = TOOL_REGISTRY[tool_name]
    try:
        # Pass progress_callback if handler accepts it
        if tool_name in ("image_generate", "video_generate"):
            res = handler(args, progress_cb=progress_callback)
        else:
            res = handler(args)

        if "isError" not in res and "error" in res and "content" not in res:
            res["isError"] = True
            res["content"] = [{"type": "text", "text": f"Error: {res['error']}"}]

        return res
    except Exception as ex:
        log(f"Execution exception in tool '{tool_name}': {ex}")
        return {
            "error": str(ex),
            "isError": True,
            "content": [{"type": "text", "text": f"Execution error in '{tool_name}': {ex}"}]
        }
