#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Generation MCP Server for media_mcp.
Thin wrapper exposing Hierarchical Scenario Video Generation, Wan 2.2 Forced Prefix continuity,
and FFmpeg assembly over JSON-RPC 2.0.
Supports Stdio (Antigravity / Cursor / Claude) and SSE HTTP (Open-WebUI).
"""

import os
import sys
import json
import uuid
import queue
import urllib.parse
import contextlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, comfy_client, ffmpeg_engine, video_pipeline

if hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_REAL_STDOUT = sys.stdout

def log(msg: str):
    sys.stderr.write(f"[video-mcp] {msg}\n")
    sys.stderr.flush()

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
    scenario_path = args.get("scenario_path")
    scenario_data = args.get("scenario_data")
    start_image = args.get("start_image")
    output_name = args.get("output_name")
    width = int(args.get("width", 832))
    height = int(args.get("height", 480))
    fps = int(args.get("fps", 16))
    enable_hud = bool(args.get("enable_hud", False))

    if scenario_path:
        p = Path(scenario_path)
        if not p.is_absolute():
            candidate = Path(config.WORKFLOWS_DIR) / scenario_path
            if not candidate.exists():
                candidate_gemini = Path(r"G:\Git\Gemini\workflows") / scenario_path
                p = candidate_gemini if candidate_gemini.exists() else candidate
            else:
                p = candidate
        if not p.exists():
            return {"error": f"Scenario file not found: {p}"}
        try:
            with open(p, "r", encoding="utf-8") as f:
                scenario_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse scenario file: {e}"}

    if not scenario_data:
        return {"error": "Either 'scenario_path' or 'scenario_data' is required."}

    if isinstance(scenario_data, str):
        try:
            scenario_data = json.loads(scenario_data)
        except Exception as e:
            return {"error": f"Failed to parse 'scenario_data' JSON: {e}"}

    log(f"Starting scenario execution: {output_name or 'unnamed'} ({width}x{height} @ {fps}fps, HUD={enable_hud})")

    try:
        res = video_pipeline.execute_scenario_pipeline(
            scenario_input=scenario_data,
            global_start_image=start_image,
            output_name=output_name,
            width=width,
            height=height,
            fps=fps,
            enable_hud=enable_hud
        )
    except Exception as ex:
        log(f"Pipeline execution failed: {ex}")
        return {"error": f"Scenario execution failed: {str(ex)}"}

    final_path = res["video_path"]
    hud_path = res.get("hud_video_path")
    total_dur = res["duration"]
    poster_path = res.get("poster_path")
    poster_b64 = res.get("poster_base64")
    scenes_count = res.get("scenes_count", 1)

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
    # Automatically offload first-frame poster in chat!
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
    scenario_path = args.get("scenario_path")
    scenario_data = args.get("scenario_data")

    if scenario_path:
        p = Path(scenario_path)
        if not p.is_absolute():
            candidate = Path(config.WORKFLOWS_DIR) / scenario_path
            if not candidate.exists():
                candidate_gemini = Path(r"G:\Git\Gemini\workflows") / scenario_path
                p = candidate_gemini if candidate_gemini.exists() else candidate
            else:
                p = candidate
        if not p.exists():
            return {"error": f"Scenario file not found: {p}"}
        try:
            with open(p, "r", encoding="utf-8") as f:
                scenario_data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse scenario JSON: {e}"}

    if not scenario_data:
        return {"error": "Either 'scenario_path' or 'scenario_data' is required."}

    if isinstance(scenario_data, str):
        try:
            scenario_data = json.loads(scenario_data)
        except Exception as e:
            return {"error": f"Failed to parse 'scenario_data' JSON: {e}"}

    try:
        scenes = video_pipeline.normalize_scenario_structure(scenario_data)
    except Exception as ex:
        return {"error": f"Scenario validation failed: {ex}"}

    errors = []
    warnings = []
    total_projected_dur = 0.0
    scene_summaries = []

    for s_idx, sc in enumerate(scenes):
        s_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
        anchor = sc.get("anchor_image", {})
        blocks = sc.get("blocks", [])

        has_image = bool(anchor.get("image_path") or anchor.get("start_image"))
        has_prompt = bool(anchor.get("prompt"))

        if not has_image and not has_prompt:
            errors.append(f"{s_name}: Neither 'image_path' nor 'prompt' specified for anchor frame.")

        if has_image:
            img_p = Path(anchor.get("image_path") or anchor.get("start_image"))
            if not img_p.exists():
                warnings.append(f"{s_name}: Anchor image path not found on disk: {img_p} (will fallback to prompt if present).")

        s_dur = 0.0
        for b_idx, blk in enumerate(blocks):
            dur = float(blk.get("duration", 5.0))
            overlap = float(blk.get("overlap", 1.5 if b_idx < len(blocks) - 1 else 0.0))
            if dur <= 0:
                errors.append(f"{s_name} / Block #{b_idx + 1}: duration must be positive.")
            if overlap < 0:
                errors.append(f"{s_name} / Block #{b_idx + 1}: overlap cannot be negative.")
            if overlap >= dur:
                errors.append(f"{s_name} / Block #{b_idx + 1}: overlap ({overlap}s) must be < duration ({dur}s).")

            if b_idx < len(blocks) - 1:
                s_dur += max(0.0, dur - overlap)
            else:
                s_dur += dur

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
    prompt = args.get("prompt")
    if not prompt:
        return {"error": "Prompt is required."}

    duration = float(args.get("duration", 5.0))
    start_image = args.get("start_image")
    end_image = args.get("end_image")
    prefix_video = args.get("prefix_video")
    width = int(args.get("width", 832))
    height = int(args.get("height", 480))
    fps = int(args.get("fps", 16))
    out_name = args.get("output_name") or f"single_{int(time.time())}.mp4"

    raw_frames = int(round(duration * fps))
    k = max(4, round((raw_frames - 1) / 4))
    num_frames = 4 * k + 1

    ok, raw_path = video_pipeline.generate_wan_segment(
        image_path=start_image,
        prefix_video_path=prefix_video,
        end_image_path=end_image,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        out_video_name=out_name
    )
    if not ok:
        return {"error": f"Block generation failed: {raw_path}"}

    final_path = os.path.join(config.OUTPUT_VID_DIR, out_name)
    ffmpeg_engine.cut_duration(raw_path, duration=duration, out_path=final_path)
    actual_dur = ffmpeg_engine.get_duration(final_path)

    poster_path = os.path.join(config.OUTPUT_IMG_DIR, f"{os.path.splitext(out_name)[0]}_poster.png")
    poster_path, poster_b64 = ffmpeg_engine.extract_poster(final_path, poster_path)

    result_text = (
        f"Video Block Generated Successfully:\n"
        f"- Output: `{final_path}`\n"
        f"- Duration: {actual_dur:.2f}s | Resolution: {width}x{height} @ {fps} fps\n\n"
        f"File link: [Open Video](file:///{final_path.replace(os.sep, '/')})\n"
    )

    content = [{"type": "text", "text": result_text}]
    if poster_b64:
        content.append({"type": "image", "data": poster_b64, "mimeType": "image/png"})

    return {"content": content, "video_path": final_path, "duration": actual_dur}

def tool_video_create_scenario(args: dict) -> dict:
    name = args.get("name", "").strip()
    if not name:
        return {"error": "Scenario name is required."}
    if not name.endswith(".json"):
        name = f"{name}.json"

    data = args.get("scenario") or args.get("scenes") or args.get("blocks")
    if not data:
        return {"error": "Scenario content ('scenes' or 'blocks') is required."}

    fp = os.path.join(config.WORKFLOWS_DIR, name)
    try:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return {"error": f"Failed to save scenario: {e}"}

    return {
        "content": [{
            "type": "text",
            "text": f"Scenario saved successfully to `{fp}`.\nLink: [Open Scenario](file:///{fp.replace(os.sep, '/')})"
        }],
        "path": fp
    }

def tool_video_list_scenarios(args: dict) -> dict:
    folder = args.get("directory") or config.WORKFLOWS_DIR
    if not os.path.isabs(folder):
        folder = os.path.join(config.OUTPUT_DIR, folder)

    if not os.path.exists(folder):
        return {"error": f"Directory not found: {folder}"}

    scenarios = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json") and not any(fn.startswith(x) for x in ["comfy_", "video_wan2_2", "Wan2.2", "Vantage-"]):
            fp = os.path.join(folder, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                scenes = video_pipeline.normalize_scenario_structure(data)
                total_dur = sum(
                    sum(max(0.5, b.get("duration", 5.0) - b.get("overlap", 1.5)) for b in sc.get("blocks", []))
                    for sc in scenes
                )
                scenarios.append({
                    "filename": fn,
                    "path": fp,
                    "scenes_count": len(scenes),
                    "estimated_duration_sec": round(total_dur, 2)
                })
            except Exception:
                continue

    text = f"Found {len(scenarios)} video scenarios in '{folder}':\n\n"
    for sc in scenarios:
        text += f"- **{sc['filename']}** ({sc['scenes_count']} scenes, ~{sc['estimated_duration_sec']}s)\n  Path: `{sc['path']}`\n\n"

    return {"content": [{"type": "text", "text": text}], "scenarios": scenarios}

def tool_video_get_status(args: dict) -> dict:
    try:
        q = comfy_client.get_queue()
        running = q.get("queue_running", [])
        pending = q.get("queue_pending", [])
        text = f"ComfyUI Video Generator Status:\n- Running: {len(running)}\n- Pending: {len(pending)}\n"
        if running:
            text += f"- Active Prompt: {running[0][1]}\n"
        else:
            text += "- Status: Idle (Ready for generation)\n"
        return {"content": [{"type": "text", "text": text}], "running_count": len(running), "pending_count": len(pending)}
    except Exception as ex:
        return {"content": [{"type": "text", "text": f"Error querying ComfyUI: {ex}"}], "isError": True}

def tool_video_interrupt(args: dict) -> dict:
    comfy_client.interrupt()
    return {"content": [{"type": "text", "text": "Successfully sent interrupt signal to ComfyUI."}]}


TOOLS_SPEC = [
    {
        "name": "video_run_scenario",
        "description": "Execute a hierarchical multi-scene video scenario with Wan 2.2 using True Forced Prefix continuity and FFmpeg assembly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_path": {"type": "string", "description": "Path to scenario JSON file (in workflows/ or absolute)"},
                "scenario_data": {"description": "Hierarchical scenario object or array"},
                "start_image": {"type": "string", "description": "Optional global starting image path for Scene 1"},
                "output_name": {"type": "string", "description": "Custom final output video filename (e.g. 'movie.mp4')"},
                "width": {"type": "integer", "description": "Width in pixels (default: 832, multiple of 16)"},
                "height": {"type": "integer", "description": "Height in pixels (default: 480, multiple of 16)"},
                "fps": {"type": "integer", "description": "Frame rate (default: 16)"},
                "enable_hud": {"type": "boolean", "description": "Generate debug video with countdown timeline overlay"}
            }
        }
    },
    {
        "name": "video_validate_scenario",
        "description": "Validate a hierarchical scenario structure before execution (checks anchor images, timing, overlaps, and models).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_path": {"type": "string", "description": "Path to scenario JSON file"},
                "scenario_data": {"description": "Scenario JSON object or array"}
            }
        }
    },
    {
        "name": "video_generate_block",
        "description": "Generate a single video segment using Wan 2.2 with optional seed, end keyframe, or prefix video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Motion and camera prompt"},
                "duration": {"type": "number", "description": "Duration in seconds (default: 5.0)"},
                "start_image": {"type": "string", "description": "Starting image path"},
                "end_image": {"type": "string", "description": "Target ending keyframe image path"},
                "prefix_video": {"type": "string", "description": "Prefix video for forced continuity"},
                "width": {"type": "integer", "description": "Width (default: 832)"},
                "height": {"type": "integer", "description": "Height (default: 480)"},
                "fps": {"type": "integer", "description": "FPS (default: 16)"},
                "output_name": {"type": "string", "description": "Output filename"}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "video_create_scenario",
        "description": "Create and save a new scenario JSON file in workflows/ directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Scenario filename (e.g. 'my_scene.json')"},
                "scenario": {"description": "Hierarchical scenario structure or array of scenes/blocks"}
            },
            "required": ["name", "scenario"]
        }
    },
    {
        "name": "video_list_scenarios",
        "description": "List all scenario files in workflows/ directory with duration estimates.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "video_get_status",
        "description": "Query ComfyUI video generator queue status and active tasks.",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "video_interrupt",
        "description": "Interrupt currently active video generation in ComfyUI.",
        "inputSchema": {"type": "object", "properties": {}}
    }
]

TOOL_HANDLERS = {
    "video_run_scenario": tool_video_run_scenario,
    "video_validate_scenario": tool_video_validate_scenario,
    "video_generate_block": tool_video_generate_block,
    "video_create_scenario": tool_video_create_scenario,
    "video_list_scenarios": tool_video_list_scenarios,
    "video_get_status": tool_video_get_status,
    "video_interrupt": tool_video_interrupt
}

def send_response(resp: dict):
    line = json.dumps(resp, ensure_ascii=False)
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()

def handle_jsonrpc(req: dict) -> Optional[dict]:
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
        tool_args = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown tool: '{tool_name}'"}}

        try:
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
        resp = handle_jsonrpc(req)
        if resp is not None:
            send_response(resp)

def run_cli_test():
    print(f"Testing Video MCP Server...")
    print(f"ComfyUI URL: {config.COMFY_URL}")
    try:
        st = tool_video_get_status({})
        print(st["content"][0]["text"])
        sc = tool_video_list_scenarios({})
        print(f"[OK] Scenarios detected: {len(sc.get('scenarios', []))}")
    except Exception as e:
        print(f"[WARN] ComfyUI connection: {e}")
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
