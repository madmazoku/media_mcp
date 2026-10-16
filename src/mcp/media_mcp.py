#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Media MCP Server for media_mcp.
Single production-grade server combining Image and Video generation into a canonical 8-tool suite.
Supports Stdio (Antigravity / Claude / Cursor) and SSE HTTP (Open-WebUI).
"""

from __future__ import annotations
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
import importlib
import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import (
    config,
    comfy_client,
    a1111_client,
    discovery,
    image_engine,
    video_pipeline,
    schema_validator,
)
from core.tools import (
    execute_tool,
    CANONICAL_TOOLS,
    get_tools_list,
    reload_core_modules,
)


def log(msg: str):
    sys.stderr.write(f"[media-mcp] {msg}\n")
    sys.stderr.flush()


def reload_core_modules() -> List[str]:
    """
    Dynamically reloads all loaded modules in sys.modules under 'core' in reverse dependency order,
    ensuring code updates take effect immediately without restarting the server.
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


if hasattr(sys.stdin, "reconfigure"):
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_REAL_STDOUT = sys.stdout
_STDOUT_LOCK = threading.Lock()
_SSE_CLIENTS: Dict[str, queue.Queue] = {}
_CURRENT_PROGRESS_TOKEN = None


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
            "logger": "media-mcp",
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

    for s_id, q in list(_SSE_CLIENTS.items()):
        try:
            q.put_nowait(log_notif)
        except Exception:
            pass


@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# Tool Handlers Registry (Delegated to core.tools)
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    name: (lambda args, name=name: execute_tool(name, args, progress_callback=broadcast_mcp_progress))
    for name in CANONICAL_TOOLS
}


def handle_jsonrpc(req: dict) -> Optional[dict]:
    msg_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "media-mcp",
                    "version": "2.0.0"
                }
            }
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": get_tools_list()}
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        tool_args = params.get("arguments") or {}
        global _CURRENT_PROGRESS_TOKEN
        _CURRENT_PROGRESS_TOKEN = params.get("_meta", {}).get("progressToken")

        try:
            with redirect_stdout_to_stderr():
                res = execute_tool(tool_name, tool_args, progress_callback=broadcast_mcp_progress)
            if res.get("isError"):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": res,
                    "isError": True
                }
            return {"jsonrpc": "2.0", "id": msg_id, "result": res}
        except Exception as ex:
            log(f"Tool execution exception: {ex}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": f"Execution error: {str(ex)}"}], "isError": True}
            }
        finally:
            _CURRENT_PROGRESS_TOKEN = None
    else:
        if msg_id is not None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method '{method}' not implemented"}}
        return None


# ---------------------------------------------------------------------------
# HTTP File Streaming & Range Requests Helper
# ---------------------------------------------------------------------------

def _serve_file_with_range(handler: BaseHTTPRequestHandler, filepath: str, content_type: str):
    file_size = Path(filepath).stat().st_size
    range_header = handler.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        parts = range_header.split("=")[1].split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1

        handler.send_response(206)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        handler.send_header("Content-Length", str(length))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        with open(filepath, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                buf = f.read(min(remaining, chunk_size))
                if not buf:
                    break
                handler.wfile.write(buf)
                remaining -= len(buf)
    else:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(file_size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        with open(filepath, "rb") as f:
            chunk_size = 64 * 1024
            while True:
                buf = f.read(chunk_size)
                if not buf:
                    break
                handler.wfile.write(buf)


# ---------------------------------------------------------------------------
# SSE HTTP Server (Open-WebUI / Network clients)
# ---------------------------------------------------------------------------

class MediaSseHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log(f"[sse-http] {format % args}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        # Serve generated images
        if parsed.path.startswith(("/images/", "/image/")):
            filename = Path(parsed.path).name
            fp = Path(config.OUTPUT_IMG_DIR) / filename
            if fp.is_file():
                ctype = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                _serve_file_with_range(self, fp.as_posix(), ctype)
                return
            self.send_error(404, "Image Not Found")
            return

        # Serve generated videos with Range requests
        if parsed.path.startswith(("/video/", "/videos/")):
            filename = Path(parsed.path).name
            fp = Path(config.OUTPUT_VID_DIR) / filename
            if fp.is_file():
                _serve_file_with_range(self, fp.as_posix(), "video/mp4")
                return
            self.send_error(404, "Video Not Found")
            return

        # SSE Endpoint
        if parsed.path in ("/sse", "/", ""):
            session_id = uuid.uuid4().hex
            q: queue.Queue = queue.Queue()
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
    server = ThreadingHTTPServer((host, port), MediaSseHandler)
    log(f"Media MCP Unified SSE Server listening on http://{host}:{port}/sse")
    print(f"[*] Media MCP SSE Server running on http://{host}:{port}/sse")
    print(f"[*] For Open-WebUI URL: http://host.docker.internal:{port}/sse")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def _async_tool_worker(req: dict):
    resp = handle_jsonrpc(req)
    if resp is not None:
        send_response(resp)


def run_stdio_server():
    log("Starting Unified Media MCP Server over stdio...")
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
        except Exception as e:
            send_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {str(e)}"}})
            continue

        method = req.get("method")
        if method == "tools/call":
            t = threading.Thread(target=_async_tool_worker, args=(req,), daemon=True)
            t.start()
        else:
            resp = handle_jsonrpc(req)
            if resp is not None:
                send_response(resp)


if __name__ == "__main__":
    if not config.is_configured():
        sys.stderr.write(
            "[media-mcp] Error: Server requires a valid configuration file.\n"
            "[media-mcp] Please specify --config <path_to_config.json> or set MEDIA_CONFIG=<path>.\n"
        )
        sys.exit(1)

    if "--sse" in sys.argv:
        port = config.MEDIA_MCP_PORT
        for arg in sys.argv:
            if arg.startswith("--port="):
                port = int(arg.split("=")[1])
        run_sse_server(port=port)
    else:
        run_stdio_server()
