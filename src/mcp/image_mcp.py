#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Generation MCP Server for media_mcp.
Thin wrapper exposing FLUX, FLUX.2 Klein, and SDXL / Pony generation over JSON-RPC 2.0.
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
from typing import Optional

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
from core import config, comfy_client, a1111_client, image_engine, schema_validator


def reload_core_modules():
    """
    Dynamically reloads all loaded modules in sys.modules under 'core' in reverse dependency order,
    ensuring all code changes take effect immediately without stale memory caches.
    """
    global config, comfy_client, a1111_client, image_engine, schema_validator
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
        from core import config as _c, comfy_client as _cm, a1111_client as _a, image_engine as _i, schema_validator as _sv
        config, comfy_client, a1111_client, image_engine, schema_validator = _c, _cm, _a, _i, _sv
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
_SSE_CLIENTS = {}
_CURRENT_PROGRESS_TOKEN = None


def log(msg: str):
    sys.stderr.write(f"[image-mcp] {msg}\n")
    sys.stderr.flush()


def send_response(resp: dict):
    line = json.dumps(resp, ensure_ascii=False)
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()


def broadcast_mcp_progress(message: str, progress: Optional[float] = None, total: Optional[float] = None):
    log_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "logger": "media-mcp-image",
            "data": message
        }
    }
    send_response(log_notif)

    if _CURRENT_PROGRESS_TOKEN is not None:
        p_params = {
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


comfy_client.set_global_progress_callback(broadcast_mcp_progress)
image_engine.set_global_progress_callback(broadcast_mcp_progress)


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

def tool_comfy_flux_txt2img(args: dict) -> dict:
    reload_core_modules()
    prompt = args["prompt"]
    model_type = args.get("model_type", "flux1_dev")
    width = int(args.get("width", 1024))
    height = int(args.get("height", 1024))
    steps = int(args["steps"]) if args.get("steps") is not None else None
    guidance = float(args.get("guidance", 3.5))
    seed = int(args.get("seed", -1))
    lora_name = args.get("lora_name")
    lora_strength = float(args.get("lora_strength", 0.8))

    ok, path, b64, meta = image_engine.generate_flux_image(
        prompt=prompt,
        model_type=model_type,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        lora_name=lora_name,
        lora_strength=lora_strength
    )
    if not ok:
        return {"error": meta.get("error", "FLUX generation failed")}

    result_text = (
        f"Image generated successfully via ComfyUI ({model_type.upper()}):\n"
        f"- Output: `{path}`\n"
        f"- Dimensions: {width}x{height} | Guidance: {guidance}\n"
        f"- Seed: {meta.get('seed')}\n\n"
        f"File link: [Open Image](file:///{path.replace(os.sep, '/')})\n"
    )

    content = [{"type": "text", "text": result_text}]
    if b64:
        content.append({"type": "image", "data": b64, "mimeType": "image/png"})

    return {"content": content, "file_path": path}


def tool_sd_txt2img(args: dict) -> dict:
    reload_core_modules()
    prompt = args["prompt"]
    negative_prompt = args.get("negative_prompt", "ugly, deformed, low quality, bad anatomy")
    steps = int(args.get("steps", 30))
    cfg_scale = float(args.get("cfg_scale", 7.0))
    width = int(args.get("width", 1280))
    height = int(args.get("height", 720))
    seed = int(args.get("seed", -1))
    sampler_name = args.get("sampler_name")
    model_name = args.get("model_name")

    ok, path, b64, meta = image_engine.generate_a1111_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_name=model_name,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        seed=seed,
        sampler_name=sampler_name
    )
    if not ok:
        return {"error": meta.get("error", "A1111 generation failed")}

    actual_w = meta.get("width", width)
    actual_h = meta.get("height", height)

    result_text = (
        f"Image generated successfully via Automatic1111:\n"
        f"- Output: `{path}`\n"
        f"- Dimensions: {actual_w}x{actual_h} | Steps: {steps}\n\n"
        f"File link: [Open Image](file:///{path.replace(os.sep, '/')})\n"
    )

    content = [{"type": "text", "text": result_text}]
    if b64:
        content.append({"type": "image", "data": b64, "mimeType": "image/png"})

    return {"content": content, "file_path": path}


def tool_sd_get_models(args: dict) -> dict:
    try:
        models = a1111_client.get_models()
        cur = a1111_client.get_current_model()
        titles = [m.get("title", m.get("model_name")) for m in models]
        text = f"Current active model: {cur}\n\nAvailable models ({len(models)}):\n" + "\n".join(f"- {t}" for t in titles)
        return {"content": [{"type": "text", "text": text}], "models": titles, "active_model": cur}
    except Exception as ex:
        return {"error": f"Failed to get models: {ex}"}


def tool_sd_set_model(args: dict) -> dict:
    model_name = args["model_name"]
    ok = a1111_client.set_model(model_name)
    if ok:
        return {"content": [{"type": "text", "text": f"Successfully set model to: {model_name}"}]}
    return {"error": f"Failed to set model to: {model_name}"}


def tool_sd_get_loras(args: dict) -> dict:
    try:
        loras = a1111_client.get_loras()
        names = [l.get("name") for l in loras]
        return {"content": [{"type": "text", "text": f"Available LoRAs ({len(names)}):\n" + "\n".join(f"- {n}" for n in names)}], "loras": names}
    except Exception as ex:
        return {"error": f"Failed to get LoRAs: {ex}"}


TOOLS_SPEC = [
    {
        "name": "comfy_flux_txt2img",
        "description": "Generate high-quality images via FLUX.1 Dev or FLUX.2 Klein in ComfyUI.",
        "inputSchema": schema_validator.get_mcp_tool_schema("comfy_flux_txt2img")
    },
    {
        "name": "sd_txt2img",
        "description": "Generate images via Automatic1111 WebUI (SDXL, Pony, Illustrious, SD 1.5).",
        "inputSchema": schema_validator.get_mcp_tool_schema("sd_txt2img")
    },
    {
        "name": "sd_get_models",
        "description": "List available checkpoint models in Automatic1111 WebUI.",
        "inputSchema": schema_validator.get_mcp_tool_schema("sd_get_models")
    },
    {
        "name": "sd_set_model",
        "description": "Switch active checkpoint model in Automatic1111.",
        "inputSchema": schema_validator.get_mcp_tool_schema("sd_set_model")
    },
    {
        "name": "sd_get_loras",
        "description": "List registered LoRAs in Automatic1111.",
        "inputSchema": schema_validator.get_mcp_tool_schema("sd_get_loras")
    }
]

TOOL_HANDLERS = {
    "comfy_flux_txt2img": tool_comfy_flux_txt2img,
    "sd_txt2img": tool_sd_txt2img,
    "sd_get_models": tool_sd_get_models,
    "sd_set_model": tool_sd_set_model,
    "sd_get_loras": tool_sd_get_loras
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
                "serverInfo": {"name": "media-mcp-image", "version": "2.0.0"}
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


class ImageSseHandler(BaseHTTPRequestHandler):
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
        if parsed.path.startswith("/images/"):
            filename = os.path.basename(parsed.path)
            fp = os.path.join(config.OUTPUT_IMG_DIR, filename)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if filename.endswith(".png") else "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
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


def run_sse_server(host: str = "0.0.0.0", port: int = 7865):
    server = ThreadingHTTPServer((host, port), ImageSseHandler)
    log(f"Image MCP SSE Server listening on http://{host}:{port}/sse")
    print(f"[*] Image MCP SSE Server running on http://{host}:{port}/sse")
    print(f"[*] Для Open-WebUI URL: http://host.docker.internal:{port}/sse")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def run_stdio_server():
    log("Starting Image MCP Server over stdio...")
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
    print("Testing Image MCP Server...")
    print(f"ComfyUI URL: {config.COMFY_URL}")
    print(f"A1111 URL: {config.SD_WEBUI_URL}")
    try:
        models = a1111_client.get_models()
        print(f"[OK] A1111 Models: {len(models)} found")
    except Exception as e:
        print(f"[WARN] A1111 connection: {e}")
    try:
        q = comfy_client.get_queue()
        print(f"[OK] ComfyUI queue connected: {len(q.get('queue_running', []))} running")
    except Exception as e:
        print(f"[WARN] ComfyUI connection: {e}")
    print("\nImage MCP Server is ready.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_cli_test()
    elif "--sse" in sys.argv or "--port" in sys.argv:
        p = config.IMAGE_MCP_PORT
        for i, a in enumerate(sys.argv):
            if a == "--port" and i + 1 < len(sys.argv):
                p = int(sys.argv[i + 1])
        run_sse_server(port=p)
    else:
        run_stdio_server()
