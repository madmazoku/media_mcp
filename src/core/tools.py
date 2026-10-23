#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Tool Dispatcher & Execution Engine for media_mcp.
Decouples core business logic from transport layers (CLI, MCP Stdio, SSE HTTP).
Handles spec resolution, schema validation, fail-fast configuration checks,
asynchronous video job delegation, and standardized MCP/CLI output formatting.
"""

from __future__ import annotations
import sys
import json
import queue
import threading
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable

from . import (
    config,
    schema_validator,
    discovery,
    comfy_client,
    a1111_client,
)
from .job_manager import JOB_MANAGER, JobRecord


def log(msg: str):
    sys.stderr.write(f"[core.tools] {msg}\n")
    sys.stderr.flush()


TOPOLOGICAL_CORE_MODULES = [
    "core.config",
    "core.schema_validator",
    "core.lora_inspector",
    "core.comfy_client",
    "core.a1111_client",
    "core.ffmpeg_engine",
    "core.discovery",
    "core.job_manager",
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
    Searches absolute path, relative to cwd, and WORKFLOWS_DIR.
    """
    p = Path(path_str)
    if not p.is_file():
        found = False
        candidates = []
        candidates.append(Path.cwd())
        if config.WORKFLOWS_DIR:
            candidates.append(Path(config.WORKFLOWS_DIR))
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
    topological dependency order, preserving active runtime configuration, _GPU_LOCK,
    JOB_MANAGER, and background worker queue.
    """
    saved_cfg = None
    if "core.config" in sys.modules:
        cfg_mod = sys.modules["core.config"]
        if hasattr(cfg_mod, "cfg_json") and cfg_mod.cfg_json:
            saved_cfg = dict(cfg_mod.cfg_json)

    saved_gpu_lock = _GPU_LOCK
    saved_job_manager = JOB_MANAGER
    saved_queue = _VIDEO_JOB_QUEUE
    saved_worker = _VIDEO_WORKER_THREAD

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
                elif mod_name == "core.job_manager":
                    setattr(m, "JOB_MANAGER", saved_job_manager)
                elif mod_name == "core.tools":
                    setattr(m, "_GPU_LOCK", saved_gpu_lock)
                    setattr(m, "JOB_MANAGER", saved_job_manager)
                    setattr(m, "_VIDEO_JOB_QUEUE", saved_queue)
                    setattr(m, "_VIDEO_WORKER_THREAD", saved_worker)
                reloaded.append(mod_name)
            except Exception as e:
                log(f"Warning: Failed to reload {mod_name}: {e}")

    return reloaded


# ---------------------------------------------------------------------------
# Individual Tool Executors & GPU Serialization
# ---------------------------------------------------------------------------

_GPU_LOCK = threading.Lock()
_VIDEO_JOB_QUEUE: queue.Queue = queue.Queue()
_VIDEO_WORKER_THREAD: Optional[threading.Thread] = None
_VIDEO_WORKER_LOCK = threading.Lock()


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


def _execute_video_job_pipeline(job: JobRecord, prep: Dict[str, Any]):
    """Executes scenario video pipeline and registers completion or failure."""
    from . import video_pipeline

    JOB_MANAGER.mark_started(job.run_prefix)
    try:
        def worker_progress_cb(msg: str, progress: Optional[float] = None, total: Optional[float] = None):
            pct = int((progress / total) * 100) if progress is not None and total else None
            pipe_st = video_pipeline.get_pipeline_status()
            JOB_MANAGER.update_progress(
                run_prefix=job.run_prefix,
                phase=pipe_st.get("phase", "diffusing"),
                percent=pct,
                scene_index=pipe_st.get("scene_index", 1),
                total_scenes=pipe_st.get("total_scenes", 1),
                block_index=pipe_st.get("block_index", 0),
                total_blocks=pipe_st.get("total_blocks", 0),
                message=msg
            )

        res = video_pipeline.execute_scenario_pipeline(
            prepared_pipeline_data=prep,
            progress_callback=worker_progress_cb
        )

        final_vid = res.get("video_path")
        poster_img = res.get("poster_path")
        w = int(res.get("width", 832))
        h = int(res.get("height", 480))
        dur = float(res.get("duration", 0.0))
        fps = int(res.get("fps", 16))
        scenes_cnt = int(res.get("scenes_count", 1))
        blocks_cnt = int(res.get("blocks_count", 0))

        video_data = {
            "video_path": Path(final_vid).resolve().as_posix() if final_vid else "",
            "duration": dur,
            "width": w,
            "height": h,
            "fps": fps,
            "scenes_count": scenes_cnt,
            "blocks_count": blocks_cnt
        }

        anchor_meta = res.get("anchor_metadata") or {}
        first_scene_anchor = {}
        if prep.get("scenes") and len(prep["scenes"]) > 0:
            first_scene_anchor = prep["scenes"][0].get("anchor_image", {})
        img_cfg = prep.get("resolved_scenario_dict", {}).get("image", {})

        seed = int(anchor_meta.get("seed") or first_scene_anchor.get("seed") or img_cfg.get("seed", 0))
        img_w = int(anchor_meta.get("width") or first_scene_anchor.get("width") or img_cfg.get("width", w))
        img_h = int(anchor_meta.get("height") or first_scene_anchor.get("height") or img_cfg.get("height", h))
        nat_w = int(anchor_meta.get("native_width") or img_cfg.get("native_width") or img_w)
        nat_h = int(anchor_meta.get("native_height") or img_cfg.get("native_height") or img_h)

        image_data = {
            "image_path": Path(poster_img).resolve().as_posix() if poster_img else "",
            "seed": seed,
            "width": img_w,
            "height": img_h,
            "native_width": nat_w,
            "native_height": nat_h
        }
        JOB_MANAGER.complete_job(
            job.run_prefix,
            video_data=video_data,
            image_data=image_data,
            warnings=res.get("warnings")
        )
    except Exception as ex:
        log(f"Background worker failed for job '{job.run_prefix}': {ex}")
        JOB_MANAGER.fail_job(job.run_prefix, str(ex))


def _run_video_job_worker(job: JobRecord, prep: Dict[str, Any]):
    """Executes a single video job under _GPU_LOCK with cancellation check."""
    snap = JOB_MANAGER.get_job_snapshot(job.run_prefix)
    if not snap or snap.get("status") == "cancelled":
        log(f"Skipping cancelled job '{job.run_prefix}' before acquiring GPU lock.")
        return

    with _GPU_LOCK:
        snap = JOB_MANAGER.get_job_snapshot(job.run_prefix)
        if not snap or snap.get("status") == "cancelled":
            log(f"Skipping cancelled job '{job.run_prefix}' after acquiring GPU lock.")
            return

        _execute_video_job_pipeline(job, prep)


def _video_worker_loop():
    """Single dedicated background worker thread executing video scenario pipeline under _GPU_LOCK in FIFO order."""
    while True:
        try:
            job, prep = _VIDEO_JOB_QUEUE.get()
        except Exception:
            break
        try:
            _run_video_job_worker(job, prep)
        finally:
            _VIDEO_JOB_QUEUE.task_done()


def _ensure_video_worker_started():
    global _VIDEO_WORKER_THREAD
    with _VIDEO_WORKER_LOCK:
        if _VIDEO_WORKER_THREAD is None or not _VIDEO_WORKER_THREAD.is_alive():
            _VIDEO_WORKER_THREAD = threading.Thread(
                target=_video_worker_loop,
                name="VideoJobQueueWorker",
                daemon=True
            )
            _VIDEO_WORKER_THREAD.start()


def execute_video_generate(args: Dict[str, Any], progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
    from . import video_pipeline
    dry_run = bool(args.get("dry_run", False))
    prep = video_pipeline.prepare_scenario_pipeline(scenario_input=args)
    run_prefix = prep["run_prefix"]
    spec_path = prep["spec_path"]

    if dry_run:
        # Pre-flight validation only: do not create job or queue on GPU
        total_dur = 0.0
        for sc in prep.get("scenes", []):
            for blk in sc.get("blocks", []):
                dur = float(blk.get("duration", 0.0))
                overlap = float(blk.get("overlap", 0.0))
                total_dur += max(0.0, dur - overlap)

        resp: Dict[str, Any] = {
            "run_prefix": run_prefix,
            "status": "validated",
            "spec_path": Path(spec_path).resolve().as_posix(),
            "is_valid": True,
            "total_duration": round(total_dur, 2),
            "scenes_count": prep["total_scenes"],
            "blocks_count": prep["total_blocks"]
        }
        if prep.get("warnings"):
            resp["warnings"] = list(prep["warnings"])
        return resp

    job = JOB_MANAGER.create_job(
        run_prefix=run_prefix,
        spec_path=spec_path,
        scenario_dict=prep["resolved_scenario_dict"],
        total_scenes=prep["total_scenes"],
        total_blocks=prep["total_blocks"]
    )

    _ensure_video_worker_started()
    _VIDEO_JOB_QUEUE.put((job, prep))

    resp: Dict[str, Any] = {
        "run_prefix": run_prefix,
        "status": "queued",
        "spec_path": Path(spec_path).resolve().as_posix()
    }
    return resp


def execute_media_get_job(args: Dict[str, Any]) -> Dict[str, Any]:
    run_prefix = args["run_prefix"]
    wait = int(args.get("wait", 0))

    snap = JOB_MANAGER.get_job_snapshot(run_prefix, wait_seconds=wait)
    if not snap:
        return {
            "run_prefix": run_prefix,
            "status": "not_found",
            "error": f"Job '{run_prefix}' was not found in active jobs registry."
        }

    return snap


def execute_media_remove_job(args: Dict[str, Any]) -> Dict[str, Any]:
    run_prefix = args["run_prefix"]
    status, msg = JOB_MANAGER.cancel_or_remove_job(run_prefix)
    return {
        "run_prefix": run_prefix,
        "status": status,
        "message": msg
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

    q_info = JOB_MANAGER.get_queue_info()
    status_lines.append(
        f"📋 Jobs Registry: {q_info['total_records']} record(s) "
        f"(Running: {q_info['running_jobs']}, Queued: {q_info['queued_jobs']})"
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
    cancelled_job = JOB_MANAGER.cancel_active_job()
    if cancelled_job:
        msg = f"Interrupt signals dispatched; active job '{cancelled_job}' cancelled."
    else:
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


# ---------------------------------------------------------------------------
# CoreTool Object Definition & Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoreTool:
    name: str
    description: str
    handler: Callable[..., Dict[str, Any]]
    takes_progress: bool = False

    @property
    def input_schema(self) -> Dict[str, Any]:
        return schema_validator.get_mcp_tool_schema(self.name)

    @property
    def output_schema(self) -> Dict[str, Any]:
        return schema_validator.get_mcp_response_schema(self.name)

    def execute(self, payload: Optional[Dict[str, Any]] = None, progress_cb: Optional[Callable] = None) -> Dict[str, Any]:
        return execute_tool(self.name, payload=payload, progress_callback=progress_cb)

    def to_mcp_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


CORE_TOOLS: Dict[str, CoreTool] = {
    "image_generate": CoreTool(
        name="image_generate",
        description="Generate high-quality images via SDXL/Pony (Automatic1111) or FLUX.1/FLUX.2 (ComfyUI) using a unified image_spec dictionary.",
        handler=execute_image_generate,
        takes_progress=True,
    ),
    "video_generate": CoreTool(
        name="video_generate",
        description="Generate multi-scene or single-block videos asynchronously using Wan 2.2 Fast (wan-fast), Wan 2.2 Full (wan-full), or LTX-Video (ltx) with 5D Latent-Chaining. Synchronously validates and returns run_prefix immediately.",
        handler=execute_video_generate,
    ),
    "media_get_job": CoreTool(
        name="media_get_job",
        description="Poll or retrieve the status, progress, and completed video/poster outputs of an asynchronous video generation job by run_prefix. Supports blocking wait.",
        handler=execute_media_get_job,
    ),
    "media_remove_job": CoreTool(
        name="media_remove_job",
        description="Cancel a queued or active video generation job, or remove completed job metadata from the server registry.",
        handler=execute_media_remove_job,
    ),
    "media_get_models": CoreTool(
        name="media_get_models",
        description="Inspect installed checkpoint and UNET models across image and video engines. Returns invariant tree structure filtered by type or engine.",
        handler=execute_media_get_models,
    ),
    "media_get_samplers": CoreTool(
        name="media_get_samplers",
        description="Get recommended and compatible samplers for a specific engine and model. Index 0 in returned list is the recommended default.",
        handler=execute_media_get_samplers,
    ),
    "media_get_loras": CoreTool(
        name="media_get_loras",
        description="Inspect compatible LoRA models and their trained activation triggers for a specific engine and model. Supports optional 'pattern' filter.",
        handler=execute_media_get_loras,
    ),
    "media_get_status": CoreTool(
        name="media_get_status",
        description="Check cluster health, GPU lock status, active job progress, and backend availability across ComfyUI and Automatic1111.",
        handler=execute_media_get_status,
    ),
    "media_interrupt": CoreTool(
        name="media_interrupt",
        description="Emergency stop: interrupt and cancel currently running image and video generations across all engines.",
        handler=execute_media_interrupt,
    ),
    "media_reload_modules": CoreTool(
        name="media_reload_modules",
        description="Developer tool: hot-reload all python modules under 'core' in memory on demand.",
        handler=execute_media_reload_modules,
    ),
}

CANONICAL_TOOLS = list(CORE_TOOLS.keys())
TOOL_REGISTRY = {t.name: t.handler for t in CORE_TOOLS.values()}


def get_tools_list() -> List[Dict[str, Any]]:
    return [t.to_mcp_dict() for t in CORE_TOOLS.values()]


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
    if tool_name not in CORE_TOOLS:
        return {
            "error": f"Unknown tool '{tool_name}'. Available tools: {list(CORE_TOOLS.keys())}",
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
    core_tool = CORE_TOOLS[tool_name]
    try:
        if core_tool.takes_progress:
            res = core_tool.handler(args, progress_cb=progress_callback)
        else:
            res = core_tool.handler(args)

        if res.get("isError"):
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
