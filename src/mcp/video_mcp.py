#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Generation MCP Server for media_mcp.
Thin wrapper exposing Hierarchical Scenario Video Generation, Wan 2.2 / LTX-2.3 Forced Prefix continuity,
and FFmpeg assembly over JSON-RPC 2.0.
Supports Stdio (Antigravity / Cursor / Claude) and SSE HTTP (Open-WebUI).
"""

import os
import sys
import json
import uuid
import queue
import threading
import shutil
import base64
import urllib.parse
import contextlib
import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
from core import config, a1111_client, comfy_client, ffmpeg_engine, image_engine, video_pipeline, schema_validator
from core.generators import create_video_generator, VideoRequest


def reload_core_modules():
    """
    Dynamically reloads all loaded modules in sys.modules under 'core' in reverse dependency order,
    ensuring all code changes take effect immediately without stale memory caches.
    """
    global config, a1111_client, comfy_client, ffmpeg_engine, image_engine, video_pipeline, schema_validator
    global create_video_generator, VideoRequest
    core_mods = [k for k in list(sys.modules.keys()) if k == "core" or k.startswith("core.")]
    core_mods.sort(key=lambda s: s.count("."), reverse=True)
    for mod_name in core_mods:
        m = sys.modules.get(mod_name)
        if m and hasattr(m, "__file__") and m.__file__:
            try:
                importlib.reload(m)
            except Exception as e:
                log(f"Warning: Failed to reload {mod_name}: {e}")
    try:
        from core import (
            config as _c,
            a1111_client as _a,
            comfy_client as _cm,
            ffmpeg_engine as _f,
            image_engine as _i,
            video_pipeline as _v,
            schema_validator as _sv,
        )
        from core.generators import create_video_generator as _cvg, VideoRequest as _vr
        config, a1111_client, comfy_client, ffmpeg_engine, image_engine, video_pipeline, schema_validator = (
            _c, _a, _cm, _f, _i, _v, _sv
        )
        create_video_generator, VideoRequest = _cvg, _vr
    except Exception as e:
        log(f"Warning: Failed to re-bind core modules: {e}")


if hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_REAL_STDOUT = sys.stdout
_STDOUT_LOCK = threading.Lock()
_SSE_CLIENTS = {}
_CURRENT_PROGRESS_TOKEN = None


def log(msg: str):
    sys.stderr.write(f"[video-mcp] {msg}\n")
    sys.stderr.flush()


def send_response(resp: dict):
    line = json.dumps(resp, ensure_ascii=False)
    with _STDOUT_LOCK:
        _REAL_STDOUT.write(line + "\n")
        _REAL_STDOUT.flush()


def broadcast_mcp_progress(message: str, progress: Optional[float] = None, total: Optional[float] = None):
    log_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "logger": "media-mcp-video",
            "data": message
        }
    }
    send_response(log_notif)

    if _CURRENT_PROGRESS_TOKEN is not None:
        p_params: Dict[str, Any] = {
            "progressToken": _CURRENT_PROGRESS_TOKEN,
            "progress": float(progress) if progress is not None else 0.0,
            "message": message
        }
        if total is not None:
            p_params["total"] = float(total)
        send_response({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": p_params
        })

    if _SSE_CLIENTS:
        for q in list(_SSE_CLIENTS.values()):
            try:
                q.put(log_notif)
            except Exception:
                pass


video_pipeline.set_global_progress_callback(broadcast_mcp_progress)


def resolve_scenario_path(scenario_path: str) -> Optional[Path]:
    p = Path(scenario_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        Path.cwd() / scenario_path,
        Path(config.TEMP_DIR) / scenario_path,
        Path(config.WORKFLOWS_DIR) / scenario_path
    ]
    for c in candidates:
        if c.exists():
            return c

    if not scenario_path.endswith(".json"):
        for base in [Path.cwd(), Path(config.TEMP_DIR), Path(config.WORKFLOWS_DIR)]:
            c = base / f"{scenario_path}.json"
            if c.exists():
                return c

    return None


@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def tool_video_run_scenario(args: dict) -> dict:
    reload_core_modules()
    scenario_data = args.get("scenario_data")
    scenario_path = args.get("scenario_path")
    enable_hud = bool(args.get("enable_hud", False))

    if scenario_data is None and scenario_path:
        p = resolve_scenario_path(scenario_path)
        if not p:
            return {
                "error": f"Scenario file not found: '{scenario_path}'. Checked cwd, temp/ ({config.TEMP_DIR}), and workflows/ ({config.WORKFLOWS_DIR})."
            }
        try:
            with open(p, "r", encoding="utf-8") as f:
                scenario_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse scenario file '{p}': {e}"}

    if scenario_data is None:
        return {"error": "Scenario data is required (pass 'scenario_path' or 'scenario_data')."}

    image_cfg = args.get("image") or {}
    video_cfg = args.get("video") or {}

    log(f"Starting scenario execution (image={image_cfg}, video={video_cfg}, HUD={enable_hud})")

    try:
        res = video_pipeline.execute_scenario_pipeline(
            scenario_input=scenario_data,
            enable_hud=enable_hud,
            image=image_cfg,
            video=video_cfg
        )
    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        log(f"Pipeline execution failed: {ex}\n{tb}")
        return {"error": f"Scenario execution failed: {str(ex)}\n{tb}"}

    final_path = res["video_path"]
    hud_path = res.get("hud_video_path")
    total_dur = res["duration"]
    poster_path = res.get("poster_path")
    poster_b64 = res.get("poster_base64")
    scenes_count = res.get("scenes_count", 1)
    width = res.get("width", 832)
    height = res.get("height", 480)
    fps = res.get("fps", 16)

    video_filename = os.path.basename(final_path)
    openwebui_url = f"http://host.docker.internal:{config.VIDEO_MCP_PORT}/video/{video_filename}"

    result_text = (
        f"Video Scenario Successfully Completed!\n\n"
        f"- Output Video: `{final_path}`\n"
        f"- Total Duration: {total_dur:.2f} seconds\n"
        f"- Resolution: {width}x{height} @ {fps} fps\n"
        f"- Scenes executed: {scenes_count}\n"
    )
    if hud_path and os.path.exists(hud_path):
        result_text += f"- Dynamic Timeline HUD Video: `{hud_path}`\n"

    result_text += (
        f"\nFile link: [Open Video](file:///{final_path.replace(os.sep, '/')})\n"
        f"Open-WebUI Stream URL: {openwebui_url}\n"
        f"\n<video controls width=\"100%\" src=\"{openwebui_url}\"></video>\n"
    )

    content = [{"type": "text", "text": result_text}]
    if poster_b64:
        content.append({
            "type": "image",
            "data": poster_b64,
            "mimeType": "image/png"
        })

    return {
        "content": content,
        "video_path": final_path,
        "hud_video_path": hud_path,
        "poster_path": poster_path,
        "duration": total_dur,
        "width": width,
        "height": height,
        "fps": fps,
        "stream_url": openwebui_url
    }


def tool_video_validate_scenario(args: dict) -> dict:
    reload_core_modules()
    scenario_data = args.get("scenario_data")
    scenario_path = args.get("scenario_path")

    if scenario_data is None and scenario_path:
        p = resolve_scenario_path(scenario_path)
        if not p:
            return {
                "error": f"Scenario file not found: '{scenario_path}'. Checked cwd, temp/ ({config.TEMP_DIR}), and workflows/ ({config.WORKFLOWS_DIR})."
            }
        try:
            with open(p, "r", encoding="utf-8") as f:
                scenario_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse scenario file '{p}': {e}"}

    if scenario_data is None:
        return {"error": "Scenario data is required (pass 'scenario_path' or 'scenario_data')."}

    try:
        scenes = video_pipeline.normalize_scenario_structure(scenario_data)
    except Exception as ex:
        return {
            "content": [{"type": "text", "text": f"Hierarchical Scenario Validation Report:\n- Status: INVALID [SCHEMA ERROR]\n\nErrors:\n  [X] {ex}"}],
            "is_valid": False,
            "total_duration": 0.0,
            "errors": [str(ex)],
            "warnings": [],
            "scenes": []
        }

    max_duration = float(scenario_data.get("video", {}).get("max_duration", config.MAX_VIDEO_DURATION))
    errors = video_pipeline.collect_scenario_timing_errors(scenes, max_duration=max_duration)
    warnings = []
    total_projected_dur = 0.0
    scene_summaries = []

    for s_idx, sc in enumerate(scenes):
        s_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
        anchor = sc.get("anchor_image", {})
        blocks = sc.get("blocks", [])

        has_image = bool(anchor.get("image_path"))
        has_prompt = bool(anchor.get("prompt"))

        if not has_image and not has_prompt:
            errors.append(f"{s_name}: Neither 'image_path' nor 'prompt' specified for anchor_image.")

        if has_image:
            img_p = Path(anchor["image_path"])
            if not img_p.exists():
                if has_prompt:
                    warnings.append(
                        f"{s_name}: Anchor image_path not found on disk: {img_p} (will fallback to prompt)."
                    )
                else:
                    errors.append(
                        f"{s_name}: Anchor image_path not found on disk: {img_p} and no fallback prompt provided."
                    )

        # In prefix-addition merging, each block outputs its full clip_duration (T_core)
        s_dur = sum(float(blk.get("duration", 5.0)) for blk in blocks)
        total_projected_dur += s_dur
        scene_summaries.append({
            "scene_name": s_name,
            "blocks_count": len(blocks),
            "projected_duration": round(s_dur, 2),
            "anchor_type": "Existing File" if has_image else "Auto-generated via Prompt"
        })

    is_valid = (len(errors) == 0)
    report = (
        f"Hierarchical Scenario Validation Report:\n"
        f"- Status: {'VALID [OK]' if is_valid else 'INVALID [ERRORS DETECTED]'}\n"
        f"- Scenes count: {len(scenes)}\n"
        f"- Max single-pass block window: {max_duration:.1f}s (prefix_duration + clip_duration < max_duration)\n"
        f"- Projected total duration: ~{total_projected_dur:.2f} seconds\n"
    )

    if errors:
        report += "\nErrors:\n" + "\n".join(f"  [X] {e}" for e in errors)
    if warnings:
        report += "\nWarnings:\n" + "\n".join(f"  [!] {w}" for w in warnings)

    report += "\nScene Breakdown:\n"
    for s in scene_summaries:
        report += f"- **{s['scene_name']}** ({s['blocks_count']} blocks, ~{s['projected_duration']}s) | Anchor: {s['anchor_type']}\n"

    return {
        "content": [{"type": "text", "text": report}],
        "is_valid": is_valid,
        "total_duration": total_projected_dur,
        "errors": errors,
        "warnings": warnings,
        "scenes": scene_summaries
    }


def tool_video_generate_block(args: dict) -> dict:
    reload_core_modules()
    prompt = args["prompt"]
    engine = str(args.get("engine", "wan-fast")).lower()
    negative_prompt = args.get("negative_prompt") or "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark"
    duration = float(args.get("duration", 5.0))
    start_image = args.get("start_image")
    prefix_latent = args.get("prefix_latent")
    width = int(args.get("width", 832))
    height = int(args.get("height", 480))
    fps = int(args.get("fps", 24 if engine == "ltx" else 16))
    steps = args.get("steps")
    cfg = args.get("cfg")
    seed = int(args.get("seed", -1))

    if start_image and not os.path.exists(start_image):
        return {"error": f"start_image not found on disk: {start_image}"}
    if prefix_latent and not os.path.exists(prefix_latent):
        return {"error": f"prefix_latent not found on disk: {prefix_latent}"}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    req_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}-{req_hash}"
    out_name = f"{run_prefix}-{engine}.mp4"

    stride = 8 if engine == "ltx" else 4
    raw_frames = int(round(duration * fps))
    k = max(1, round((raw_frames - 1) / stride))
    num_frames = stride * k + 1

    generator = create_video_generator(engine)
    req = VideoRequest(
        image_path=start_image,
        prefix_latent_path=prefix_latent,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        save_tail=False,
        fps=fps,
        steps=steps,
        cfg=cfg,
        seed=seed,
        out_video_name=out_name
    )
    res = generator.generate(req)
    if not res.success or not res.file_path:
        return {"error": f"Block generation failed: {res.error}"}

    final_path = os.path.join(config.OUTPUT_VID_DIR, out_name)
    shutil.copy2(res.file_path, final_path)
    actual_dur = ffmpeg_engine.get_duration(final_path)

    poster_filename = f"{run_prefix}-{engine}.png"
    poster_path = os.path.join(config.OUTPUT_IMG_DIR, poster_filename)
    if start_image and os.path.exists(start_image):
        try:
            shutil.copy2(start_image, poster_path)
        except Exception:
            pass
        try:
            with open(poster_path, "rb") as f:
                poster_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            poster_b64 = None
    else:
        poster_path, poster_b64 = ffmpeg_engine.extract_poster(final_path, poster_path)

    result_text = (
        f"Video Block Generated Successfully ({engine.upper()}):\n"
        f"- Output: `{final_path}`\n"
        f"- Duration: {actual_dur:.2f}s | Resolution: {width}x{height} @ {fps} fps\n\n"
        f"File link: [Open Video](file:///{final_path.replace(os.sep, '/')})\n"
    )

    content = [{"type": "text", "text": result_text}]
    if poster_b64:
        content.append({"type": "image", "data": poster_b64, "mimeType": "image/png"})

    return {"content": content, "video_path": final_path, "duration": actual_dur, "engine": engine}


def tool_media_get_status(args: dict) -> dict:
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

    try:
        q = comfy_client.get_queue()
        running = q.get("queue_running", [])
        pending = q.get("queue_pending", [])
        status_lines.append(f"\n⚡ ComfyUI (Wan 2.2 / FLUX):\n  - Running tasks: {len(running)}\n  - Pending in queue: {len(pending)}")
        if running:
            status_lines.append(f"  - Active prompt ID: {running[0][1]}")
            try:
                p_st = comfy_client.get_prompt_status()
                node_id = p_st.get("executing", {}).get("node")
                if node_id:
                    status_lines.append(f"  - Executing node: {node_id}")
            except Exception:
                pass
    except Exception as ex:
        status_lines.append(f"\n⚡ ComfyUI: Connection error ({ex})")

    try:
        a_prog = a1111_client.get_progress()
        prog = a_prog.get("progress", 0.0)
        job = a_prog.get("state", {}).get("job", "")
        job_count = a_prog.get("state", {}).get("job_count", 0)
        step = a_prog.get("state", {}).get("sampling_step", 0)
        total_steps = a_prog.get("state", {}).get("sampling_steps", 0)

        if job_count > 0 or prog > 0.0 or step > 0:
            status_lines.append(
                f"\n🎨 Automatic1111 (SD / Pony / Anchor Frames):\n"
                f"  - Status: ACTIVE ({round(prog * 100, 1)}%)\n"
                f"  - Step: {step}/{total_steps}\n"
                f"  - Job: {job or 'Processing'}"
            )
        else:
            status_lines.append("\n🎨 Automatic1111: IDLE")
    except Exception as ex:
        status_lines.append(f"\n🎨 Automatic1111: Connection error ({ex})")

    full_text = "\n".join(status_lines)
    return {
        "content": [{"type": "text", "text": full_text}],
        "pipeline": pipe_st
    }


def tool_media_interrupt(args: dict) -> dict:
    reload_core_modules()
    video_pipeline.request_interrupt()
    comfy_client.interrupt()
    a1111_client.interrupt()
    return {"content": [{"type": "text", "text": "Successfully sent interrupt signal to Video Pipeline, ComfyUI, and Automatic1111."}]}


TOOLS_SPEC = [
    {
        "name": "video_run_scenario",
        "description": "Execute a hierarchical multi-scene video scenario with Wan 2.2 or LTX-2.3 (22B) using Latent-Chaining and lossless Stream Copy assembly.",
        "inputSchema": schema_validator.get_mcp_tool_schema("video_run_scenario")
    },
    {
        "name": "video_validate_scenario",
        "description": "Validate a hierarchical scenario structure before execution (checks JSON Schema, anchor images, timing, and overlap constraints).",
        "inputSchema": schema_validator.get_mcp_tool_schema("video_validate_scenario")
    },
    {
        "name": "video_generate_block",
        "description": "Generate a single video segment using Wan 2.2 or LTX-2.3 (22B) with a start image or prefix latent.",
        "inputSchema": schema_validator.get_mcp_tool_schema("video_generate_block")
    },
    {
        "name": "media_get_status",
        "description": "Query real-time generation and queue status across Video Pipeline, ComfyUI (Wan/FLUX), and Automatic1111 (SD/Pony).",
        "inputSchema": schema_validator.get_mcp_tool_schema("media_get_status")
    },
    {
        "name": "media_interrupt",
        "description": "Interrupt any active generation across Video Pipeline, ComfyUI (Wan/FLUX), and Automatic1111 (SD/Pony).",
        "inputSchema": schema_validator.get_mcp_tool_schema("media_interrupt")
    }
]

TOOL_HANDLERS = {
    "video_run_scenario": tool_video_run_scenario,
    "video_validate_scenario": tool_video_validate_scenario,
    "video_generate_block": tool_video_generate_block,
    "media_get_status": tool_media_get_status,
    "media_interrupt": tool_media_interrupt
}


def handle_jsonrpc(req: dict) -> Optional[dict]:
    global _CURRENT_PROGRESS_TOKEN
    msg_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "media-mcp-video", "version": "2.0.0"}
            }
        }
    elif method in ("notifications/initialized", "initialized"):
        return None
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS_SPEC}}
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        _CURRENT_PROGRESS_TOKEN = params.get("_meta", {}).get("progressToken")
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown tool: '{tool_name}'"}}

        try:
            schema_validator.validate_mcp_args(tool_name, tool_args)
            with redirect_stdout_to_stderr():
                res = handler(tool_args)
            if "error" in res and "content" not in res:
                return {
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"Error: {res['error']}"}], "isError": True}
                }
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        except Exception as ex:
            log(f"Tool execution exception: {ex}")
            return {
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"content": [{"type": "text", "text": f"Execution error: {str(ex)}"}], "isError": True}
            }
        finally:
            _CURRENT_PROGRESS_TOKEN = None
    else:
        if msg_id is not None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method '{method}' not implemented"}}
        return None


# ---------------------------------------------------------------------------
# SSE HTTP Server (Open-WebUI / Network clients)
# ---------------------------------------------------------------------------
_SSE_CLIENTS = {}


class VideoSseHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log(f"[sse-http] {format % args}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/video/"):
            filename = os.path.basename(parsed.path)
            fp = os.path.join(config.OUTPUT_VID_DIR, filename)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404, "Not Found")
            return

        if parsed.path in ("/sse", "/", ""):
            session_id = uuid.uuid4().hex
            q = queue.Queue()
            _SSE_CLIENTS[session_id] = q
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            self.wfile.write(f"event: endpoint\r\ndata: /messages?session_id={session_id}\r\n\r\n".encode("utf-8"))
            self.wfile.flush()
            log(f"SSE client connected: {session_id}")

            try:
                while True:
                    try:
                        msg = q.get(timeout=15.0)
                        if msg is None:
                            break
                        self.wfile.write(f"event: message\r\ndata: {json.dumps(msg, ensure_ascii=False)}\r\n\r\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": ping\r\n\r\n")
                        self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                log(f"SSE client disconnected: {session_id}")
                _SSE_CLIENTS.pop(session_id, None)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/messages", "/sse", "/", ""):
            query_params = urllib.parse.parse_qs(parsed.query)
            session_id = query_params.get("session_id", [None])[0]
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                req = json.loads(raw_body.decode("utf-8"))
                resp = handle_jsonrpc(req)
            except Exception as e:
                resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}

            if session_id and session_id in _SSE_CLIENTS:
                self.send_response(202)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Accepted")
                if resp is not None:
                    _SSE_CLIENTS[session_id].put(resp)
            else:
                resp_bytes = json.dumps(resp or {}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
        else:
            self.send_error(404, "Not Found")


def run_sse_server(host: str = "0.0.0.0", port: int = 7866):
    server = ThreadingHTTPServer((host, port), VideoSseHandler)
    log(f"Video MCP SSE Server listening on http://{host}:{port}/sse")
    print(f"[*] Video MCP SSE Server running on http://{host}:{port}/sse")
    print(f"[*] Для Open-WebUI URL: http://host.docker.internal:{port}/sse")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def _async_tool_worker(req: dict):
    resp = handle_jsonrpc(req)
    if resp is not None:
        send_response(resp)


def run_stdio_server():
    log("Starting Video MCP Server over stdio...")
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        params = req.get("params", {})
        tool_name = params.get("name") if method == "tools/call" else None

        if method == "tools/call" and tool_name in ("video_run_scenario", "video_generate_block"):
            threading.Thread(target=_async_tool_worker, args=(req,), daemon=True).start()
        else:
            resp = handle_jsonrpc(req)
            if resp is not None:
                send_response(resp)


def run_cli_test():
    print("Testing Video MCP Server...")
    print(f"ComfyUI URL: {config.COMFY_URL}")
    try:
        st = tool_media_get_status({})
        print(st["content"][0]["text"])
    except Exception as e:
        print(f"[WARN] Status check: {e}")
    print("\nVideo MCP Server is ready.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_cli_test()
    elif "--sse" in sys.argv or "--port" in sys.argv:
        p = config.VIDEO_MCP_PORT
        for i, a in enumerate(sys.argv):
            if a == "--port" and i + 1 < len(sys.argv):
                p = int(sys.argv[i + 1])
        run_sse_server(port=p)
    else:
        run_stdio_server()
