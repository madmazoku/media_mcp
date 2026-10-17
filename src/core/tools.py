#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tool Dispatcher & Execution Engine for media_mcp.
Decouples core business logic from transport layers (CLI, MCP Stdio, SSE HTTP).
Handles spec resolution, schema validation, fail-fast configuration checks,
progress routing, and standardized MCP/CLI output formatting.
"""

from __future__ import annotations
import sys
import json
import threading
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

TOPOLOGICAL_CORE_MODULES = [
    "core.config",
    "core.schema_validator",
    "core.lora_inspector",
    "core.comfy_client",
    "core.a1111_client",
    "core.ffmpeg_engine",
    "core.discovery",
    "core.generators.base",
    "core.generators.image_generators",
    "core.generators.video_generators",
    "core.generators.factory",
    "core.generators",
    "core.image_engine",
    "core.video_pipeline",
    "core.schema_formatter",
    "core.tools",
    "core",
]


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
    Dynamically reloads all loaded modules in sys.modules under 'core' in explicit
    topological dependency order, preserving active runtime configuration and _GPU_LOCK.
    """
    saved_cfg = None
    if "core.config" in sys.modules:
        cfg_mod = sys.modules["core.config"]
        if hasattr(cfg_mod, "cfg_json") and cfg_mod.cfg_json:
            saved_cfg = dict(cfg_mod.cfg_json)

    saved_gpu_lock = _GPU_LOCK

    loaded_core_mods = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
    ordered_mods: List[str] = []
    for mod_name in TOPOLOGICAL_CORE_MODULES:
        if mod_name in loaded_core_mods:
            ordered_mods.append(mod_name)
    for mod_name in loaded_core_mods:
        if mod_name not in ordered_mods:
            ordered_mods.append(mod_name)

    reloaded: List[str] = []
    for mod_name in ordered_mods:
        m = sys.modules.get(mod_name)
        if m and hasattr(m, "__file__") and m.__file__:
            try:
                importlib.reload(m)
                if mod_name == "core.config" and saved_cfg:
                    m.load_config_dict(saved_cfg)
                elif mod_name == "core.tools":
                    setattr(m, "_GPU_LOCK", saved_gpu_lock)
                reloaded.append(mod_name)
            except Exception as e:
                log(f"Warning: Failed to reload {mod_name}: {e}")

    return reloaded


# ---------------------------------------------------------------------------
# Individual Tool Executors & GPU Serialization
# ---------------------------------------------------------------------------

_GPU_LOCK = threading.Lock()


def execute_image_generate(args: Dict[str, Any], progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    with _GPU_LOCK:
        from . import image_engine
        if progress_cb:
            image_engine.set_global_progress_callback(progress_cb)

        success, img_path, _b64_data, metadata = image_engine.generate_image(args)
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
        seed = metadata.get("seed", 0)
        width = int(metadata.get("width", 1024))
        height = int(metadata.get("height", 1024))
        native_width = int(metadata.get("native_width", width))
        native_height = int(metadata.get("native_height", height))

        resp: Dict[str, Any] = {
            "run_prefix": run_prefix,
            "image_path": Path(img_path).resolve().as_posix() if img_path else "",
            "spec_path": Path(spec_path).resolve().as_posix() if spec_path else "",
            "engine": str(metadata.get("engine", "sdxl")).lower(),
            "model": str(metadata.get("model", "")),
            "seed": int(seed) if seed is not None else 0,
            "width": width,
            "height": height,
            "native_width": native_width,
            "native_height": native_height,
        }
        if metadata.get("warnings"):
            resp["warnings"] = list(metadata["warnings"])
        return resp


def execute_video_generate(args: Dict[str, Any], progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    with _GPU_LOCK:
        from . import video_pipeline
        if progress_cb:
            video_pipeline.set_global_progress_callback(progress_cb)

        res = video_pipeline.execute_scenario_pipeline(
            scenario_input=args,
            progress_callback=progress_cb
        )

        run_prefix = res.get("run_prefix", "")
        final_path = res["video_path"]
        spec_path = res.get("spec_path", "")
        poster_path = res.get("poster_path", "")
        dur = float(res.get("duration", 0.0))
        w = int(res.get("width", 832))
        h = int(res.get("height", 480))
        fps = int(res.get("fps", 16))
        scenes_cnt = int(res.get("scenes_count", 1))
        blocks_cnt = int(res.get("blocks_count", 0))

        resp: Dict[str, Any] = {
            "run_prefix": run_prefix,
            "video_path": Path(final_path).resolve().as_posix() if final_path else "",
            "spec_path": Path(spec_path).resolve().as_posix() if spec_path else "",
            "poster_path": Path(poster_path).resolve().as_posix() if poster_path else "",
            "duration": dur,
            "width": w,
            "height": h,
            "fps": fps,
            "scenes_count": scenes_cnt,
            "blocks_count": blocks_cnt,
        }
        if res.get("warnings"):
            resp["warnings"] = list(res["warnings"])
        return resp


def execute_video_validate(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    errors: List[str] = []
    warnings: List[str] = []
    scene_summaries: List[Dict[str, Any]] = []
    total_projected_dur = 0.0

    try:
        scenario_dict = video_pipeline.resolve_scenario_dict(args)
        scenes = video_pipeline.normalize_scenario_structure(scenario_dict)
    except Exception as ex:
        return {
            "content": [{"type": "text", "text": f"Scenario Normalization/Schema Error:\n{ex}"}],
            "is_valid": False,
            "errors": [str(ex)],
            "warnings": [],
            "scenes": []
        }

    v_cfg = scenario_dict.get("video", {})
    vid_w = int(v_cfg.get("width", 832))
    vid_h = int(v_cfg.get("height", 480))
    max_dur = float(v_cfg.get("max_duration", config.MAX_VIDEO_DURATION))

    timing_errors = video_pipeline.collect_scenario_timing_errors(scenes, max_duration=max_dur)
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
    return {
        "is_valid": is_valid,
        "total_duration": round(total_projected_dur, 2),
        "errors": errors,
        "warnings": warnings,
        "scenes": scene_summaries
    }


def execute_media_get_models(args: Dict[str, Any]) -> Dict[str, Any]:
    type_filter = args.get("type")
    engine_filter = args.get("engine")
    return discovery.get_models_tree(type_filter=type_filter, engine_filter=engine_filter)


def execute_media_get_samplers(args: Dict[str, Any]) -> Dict[str, Any]:
    engine = args["engine"]
    model = args["model"]
    samplers = discovery.get_samplers_for_model(engine=engine, model=model)
    return {
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
    return {
        "loras": loras
    }


def execute_media_get_status(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    status_lines = []

    gpu_busy = _GPU_LOCK.locked()
    status_lines.append(
        f"🔒 GPU Lock: {'BUSY (active generation in progress)' if gpu_busy else 'IDLE (ready for generation)'}"
    )

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
        exec_info = q_data.get("queue_running", q_data.get("queue_executing", []))
        pending = q_data.get("queue_pending", [])
        status_lines.append(
            f"⚡ ComfyUI: ONLINE ({config.COMFY_URL})\n"
            f"  - Active: {len(exec_info)} job(s) | Pending: {len(pending)} job(s)"
        )
    else:
        status_lines.append(f"⚡ ComfyUI: OFFLINE ({config.COMFY_URL} unreachable)")

    text_resp = "\n\n".join(status_lines)
    return {
        "pipeline": pipe_st,
        "comfy_online": comfy_avail,
        "a1111_available": a1111_avail,
        "report": text_resp
    }


def execute_media_interrupt(args: Dict[str, Any]) -> Dict[str, Any]:
    from . import video_pipeline
    comfy_client.interrupt()
    a1111_client.interrupt()
    video_pipeline.request_interrupt()
    msg = "Interrupt signals dispatched to ComfyUI, A1111 WebUI, and Active Video Pipelines."
    log(msg)
    return {"interrupted": True, "message": msg}


def execute_media_reload_modules(args: Dict[str, Any]) -> Dict[str, Any]:
    reloaded = reload_core_modules()
    msg = f"Hot-reloaded {len(reloaded)} core modules in topological order:\n" + "\n".join(f"- {m}" for m in reloaded)
    log(msg)
    return {
        "reloaded_modules": reloaded,
        "count": len(reloaded)
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
            "description": "Generate multi-scene or single-block videos using Wan 2.2 Fast (wan-fast), Wan 2.2 Full (wan-full), or LTX-Video (ltx) with 5D Latent-Chaining and True Forced Prefix continuity. Takes scenario_path and/or inline scenario scenes.",
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

    # 1. Environment configuration check (Fail-Fast for all tools)
    if not config.is_configured():
        err_msg = (
            f"Tool '{tool_name}' requires an environment configuration file.\n"
            f"Please specify --config <path_to_config.json> or set MEDIA_CONFIG environment variable."
        )
        return {
            "error": err_msg,
            "isError": True,
            "content": [{"type": "text", "text": f"Configuration Error:\n{err_msg}"}]
        }

    # 2. Schema validation against canonical JSON Schema (input)
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

        if res.get("isError") or "error" in res:
            res["isError"] = True
            if "content" not in res:
                res["content"] = [{"type": "text", "text": f"Error: {res.get('error', 'Execution failed')}"}]
            return res

        # 4. Mandatory Response Schema Validation (output)
        clean_res = {k: v for k, v in res.items() if k not in ("content", "isError")}
        schema_validator.validate_mcp_response(tool_name, clean_res)

        # 5. Standard MCP text content with pure formatted JSON
        res["content"] = [{"type": "text", "text": json.dumps(clean_res, indent=2, ensure_ascii=False)}]

        return res
    except Exception as ex:
        log(f"Execution exception in tool '{tool_name}': {ex}")
        return {
            "error": str(ex),
            "isError": True,
            "content": [{"type": "text", "text": f"Execution error in '{tool_name}': {ex}"}]
        }
