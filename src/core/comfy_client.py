#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI HTTP client for queuing, monitoring, and retrieving artifacts.
"""

import sys
import json
import time
import os
import shutil
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, Tuple
from .config import COMFY_URL, COMFY_INPUT_DIR, COMFY_OUTPUT_DIR


def log(msg: str):
    sys.stderr.write(f"[comfy-client] {msg}\n")
    sys.stderr.flush()


def make_request(endpoint: str, data: dict = None, method: str = "GET", raw: bool = False, timeout: int = 300):
    url = f"{COMFY_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    if not raw and data is not None:
        req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "media-mcp/1.0")

    body = json.dumps(data).encode("utf-8") if (data is not None and not raw) else None
    with urllib.request.urlopen(req, data=body, timeout=timeout) as resp:
        if raw:
            return resp.read()
        return json.loads(resp.read().decode("utf-8"))


def get_queue() -> Dict[str, Any]:
    return make_request("/queue")


def get_prompt_status() -> Dict[str, Any]:
    return make_request("/prompt")


_INTERRUPT_FLAG = False


def interrupt() -> bool:
    global _INTERRUPT_FLAG
    _INTERRUPT_FLAG = True
    try:
        make_request("/interrupt", data={}, method="POST")
        return True
    except Exception as ex:
        log(f"Interrupt error: {ex}")
        return False


def reset_interrupt():
    global _INTERRUPT_FLAG
    _INTERRUPT_FLAG = False


def is_interrupted() -> bool:
    return _INTERRUPT_FLAG


def queue_prompt(workflow: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {"prompt": workflow}
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data_bytes, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            prompt_id = res.get("prompt_id")
            if not prompt_id:
                return False, f"No prompt_id returned: {res}"
            return True, prompt_id
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        return False, f"ComfyUI HTTP {e.code} error: {err_msg}"
    except Exception as e:
        return False, f"Failed to queue prompt: {e}"


_GLOBAL_PROGRESS_CALLBACK = None


def set_global_progress_callback(cb):
    global _GLOBAL_PROGRESS_CALLBACK
    _GLOBAL_PROGRESS_CALLBACK = cb


def wait_for_prompt(
    prompt_id: str,
    timeout: int = 600,
    poll_interval: float = 1.0,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, Dict[str, Any]]:
    start_time = time.time()
    last_notify = 0.0
    cb = progress_callback or _GLOBAL_PROGRESS_CALLBACK

    while time.time() - start_time < timeout:
        if _INTERRUPT_FLAG:
            log(f"Aborting wait_for_prompt for {prompt_id} due to interrupt.")
            return False, {"error": "Execution interrupted by user request"}

        elapsed = time.time() - start_time
        if elapsed - last_notify >= 5.0:
            last_notify = elapsed
            msg = f"ComfyUI prompt {prompt_id[:8]} executing (elapsed: {int(elapsed)}s)"
            if cb:
                try:
                    cb(msg, None, None)
                except Exception:
                    pass
            log(msg)

        try:
            history = make_request(f"/history/{prompt_id}")
            if prompt_id in history:
                prompt_hist = history[prompt_id]
                status = prompt_hist.get("status", {})
                if status.get("status_str") == "error":
                    return False, {"error": status.get("messages", "ComfyUI reported execution error")}
                if status.get("completed", False) or (prompt_hist.get("outputs") and len(prompt_hist["outputs"]) > 0):
                    return True, prompt_hist
        except Exception:
            pass
        time.sleep(poll_interval)
    return False, {"error": f"Timeout after {timeout}s waiting for prompt {prompt_id}"}


def download_output_file(filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
    return make_request(f"/view?{query}", raw=True)


def upload_file_to_comfy(src_path: str, custom_filename: Optional[str] = None) -> str:
    """
    Copies a local file into ComfyUI's input directory so nodes like LoadImage
    or LoadLatent can reliably access it.
    Returns the filename.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    filename = custom_filename or os.path.basename(src_path)
    dest_path = os.path.join(COMFY_INPUT_DIR, filename)
    shutil.copy2(src_path, dest_path)
    return filename


def find_comfy_output_file(filename: str, subfolder: str = "") -> Optional[str]:
    """
    Directly checks COMFY_OUTPUT_DIR for an output file to avoid network transfer.
    """
    p = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename) if subfolder else os.path.join(COMFY_OUTPUT_DIR, filename)
    if os.path.exists(p):
        return p
    return None
