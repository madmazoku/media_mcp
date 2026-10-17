#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI HTTP client for queuing, monitoring, and retrieving artifacts.
"""

import datetime
import sys
import time
import shutil
import urllib.parse
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


def log(msg: str):
    sys.stderr.write(f"[comfy-client] {msg}\n")
    sys.stderr.flush()


_SESSION: Optional[requests.Session] = None
_RUN_PREFIX_RGX = re.compile(r"^\d{8}[-_]\d{6}-[0-9a-fA-F]{6}")


def get_session() -> requests.Session:
    """Returns a singleton requests.Session with HTTP Keep-Alive connection pooling."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.trust_env = False
        retries = Retry(total=1, backoff_factor=0.1, status_forcelist=[502, 503, 504], raise_on_status=False)
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retries)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": "media-mcp/1.0"})
        _SESSION = s
    return _SESSION


def _format_url(endpoint: str) -> str:
    base = (config.COMFY_URL or "").strip()
    if not base:
        raise ValueError("ComfyUI URL is not configured. Please set 'comfy_url' in config.json.")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base.rstrip('/')}{ep}"


def make_request(endpoint: str, data: dict = None, method: str = "GET", raw: bool = False, timeout: int = 30):
    url = _format_url(endpoint)
    s = get_session()
    m = method.upper()
    if m == "GET":
        resp = s.get(url, timeout=timeout)
    elif m == "POST":
        resp = s.post(url, json=data if not raw else None, data=data if raw else None, timeout=timeout)
    else:
        resp = s.request(m, url, json=data if not raw else None, data=data if raw else None, timeout=timeout)
    resp.raise_for_status()
    if raw:
        return resp.content
    return resp.json()


def get_queue() -> Dict[str, Any]:
    return make_request("/queue", timeout=5)


def get_queue_status() -> Dict[str, Any]:
    try:
        return make_request("/queue", timeout=5)
    except Exception:
        return {}


def is_available() -> bool:
    try:
        make_request("/system_stats", timeout=3)
        return True
    except Exception:
        return False


def get_node_input_combo_list(node_class: str, input_field: str) -> List[str]:
    """Retrieves allowed combo list values for a node class from ComfyUI /object_info/{node_class}."""
    try:
        info = make_request(f"/object_info/{node_class}", timeout=5)
        node_def = info.get(node_class, {})
        req_inputs = node_def.get("input", {}).get("required", {})
        field_def = req_inputs.get(input_field, [])
        if field_def and isinstance(field_def[0], list):
            return [str(item) for item in field_def[0]]
    except Exception:
        pass
    return []


def get_unet_models() -> List[str]:
    """Returns available diffusion models / UNets from ComfyUI."""
    return get_node_input_combo_list("UNETLoader", "unet_name")


def get_checkpoint_models() -> List[str]:
    """Returns available checkpoint models from ComfyUI."""
    return get_node_input_combo_list("CheckpointLoaderSimple", "ckpt_name")


def get_loras() -> List[str]:
    """Returns available LoRA models from ComfyUI."""
    return get_node_input_combo_list("LoraLoader", "lora_name")


_INTERRUPT_FLAG = False


def interrupt() -> bool:
    global _INTERRUPT_FLAG
    _INTERRUPT_FLAG = True
    try:
        make_request("/interrupt", data={}, method="POST", timeout=5)
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
    try:
        res = make_request("/prompt", data=payload, method="POST", timeout=30)
        prompt_id = res.get("prompt_id")
        if not prompt_id:
            return False, f"No prompt_id returned: {res}"
        return True, prompt_id
    except requests.exceptions.HTTPError as e:
        err_msg = e.response.text if e.response is not None else str(e)
        return False, f"ComfyUI HTTP {e.response.status_code if e.response else 'error'}: {err_msg}"
    except Exception as e:
        return False, f"Failed to queue prompt: {e}"


_GLOBAL_PROGRESS_CALLBACK = None


def set_global_progress_callback(cb):
    global _GLOBAL_PROGRESS_CALLBACK
    _GLOBAL_PROGRESS_CALLBACK = cb


def wait_for_prompt(
    prompt_id: str,
    timeout: int = 600,
    poll_interval: float = 0.5,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Monitors prompt execution in ComfyUI with active queue tracking and HTTP Keep-Alive.
    """
    start_time = time.time()
    execution_start_time: Optional[float] = None
    last_notify = 0.0
    was_seen_in_queue = False
    cb = progress_callback or _GLOBAL_PROGRESS_CALLBACK
    max_queue_wait = 1800.0

    while True:
        if _INTERRUPT_FLAG:
            log(f"Aborting wait_for_prompt for {prompt_id} due to interrupt.")
            return False, {"error": "Execution interrupted by user request"}

        now = time.time()
        total_elapsed = now - start_time

        # 1. Fast history check
        try:
            history = make_request(f"/history/{prompt_id}", timeout=5)
            if prompt_id in history:
                prompt_hist = history[prompt_id]
                status = prompt_hist.get("status", {})
                if status.get("status_str") == "error":
                    return False, {"error": status.get("messages", "ComfyUI reported execution error")}
                if status.get("completed", False) or (prompt_hist.get("outputs") and len(prompt_hist["outputs"]) > 0):
                    return True, prompt_hist
        except Exception:
            pass

        # 2. Check position in ComfyUI queue
        in_pending = False
        in_running = False
        queue_pos = None

        try:
            q_data = make_request("/queue", timeout=5)
            running_list = q_data.get("queue_running", [])
            for item in running_list:
                if len(item) > 1 and item[1] == prompt_id:
                    in_running = True
                    break

            if not in_running:
                pending_list = q_data.get("queue_pending", [])
                for idx, item in enumerate(pending_list):
                    if len(item) > 1 and item[1] == prompt_id:
                        in_pending = True
                        queue_pos = idx + 1
                        break
        except Exception:
            pass

        if in_pending or in_running:
            was_seen_in_queue = True

        if in_pending:
            if total_elapsed > max_queue_wait:
                return False, {"error": f"Queue wait timeout after {int(max_queue_wait)}s waiting for prompt {prompt_id}"}

            if total_elapsed - last_notify >= 3.0:
                last_notify = total_elapsed
                msg = f"ComfyUI prompt {prompt_id[:8]} waiting in queue (position: {queue_pos}, wait: {int(total_elapsed)}s)"
                if cb:
                    try:
                        cb(msg, None, None)
                    except Exception:
                        pass
                log(msg)

        elif in_running:
            if execution_start_time is None:
                execution_start_time = now
                log(f"ComfyUI prompt {prompt_id[:8]} started execution on GPU.")

            exec_elapsed = now - execution_start_time
            if exec_elapsed > timeout:
                return False, {"error": f"Execution timeout after {timeout}s diffusing prompt {prompt_id}"}

            if total_elapsed - last_notify >= 3.0:
                last_notify = total_elapsed
                msg = f"ComfyUI prompt {prompt_id[:8]} diffusing on GPU (elapsed: {int(exec_elapsed)}s)"
                if cb:
                    try:
                        cb(msg, None, None)
                    except Exception:
                        pass
                log(msg)

        else:
            if was_seen_in_queue or total_elapsed >= 5.0:
                time.sleep(0.5)
                try:
                    history = make_request(f"/history/{prompt_id}", timeout=5)
                    if prompt_id in history:
                        prompt_hist = history[prompt_id]
                        status = prompt_hist.get("status", {})
                        if status.get("status_str") == "error":
                            return False, {"error": status.get("messages", "ComfyUI reported execution error")}
                        if status.get("completed", False) or (prompt_hist.get("outputs") and len(prompt_hist["outputs"]) > 0):
                            return True, prompt_hist
                except Exception:
                    pass

                return False, {
                    "error": f"ComfyUI prompt {prompt_id[:8]} disappeared from queue without completing (possible crash, CUDA error, or manual cancellation)."
                }

        time.sleep(poll_interval)


def download_output_file(filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
    query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
    return make_request(f"/view?{query}", raw=True)


def upload_file_to_comfy(
    src_path: str,
    custom_filename: Optional[str] = None,
    run_prefix: Optional[str] = None,
    suffix: str = "input"
) -> str:
    """
    Copies a local file into ComfyUI's input directory using the unified artifact naming convention
    ({run_prefix}-{suffix}.{ext}) so all artifacts of a run remain traceable and collision-free.
    """
    p_src = Path(src_path)
    if not p_src.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    input_dir = config.COMFY_INPUT_DIR
    if not input_dir or not Path(input_dir).is_dir():
        raise RuntimeError(
            f"ComfyUI input directory is not configured or does not exist on disk: '{input_dir}'. "
            f"Please set a valid 'comfy_input_dir' in config.json."
        )

    ext = p_src.suffix or ".png"
    if custom_filename:
        filename = custom_filename
    elif run_prefix:
        clean_suffix = suffix.lstrip("-")
        filename = f"{run_prefix}-{clean_suffix}{ext}"
    elif _RUN_PREFIX_RGX.match(p_src.stem):
        # Already carries a run_prefix (e.g. from temp/)
        filename = p_src.name
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        short_h = uuid.uuid4().hex[:6]
        clean_suffix = suffix.lstrip("-")
        filename = f"{ts}-{short_h}-{clean_suffix}{ext}"

    dest_path = (Path(input_dir) / filename).as_posix()
    shutil.copy2(src_path, dest_path)
    return filename


# Compatibility alias
copy_to_comfy_input = upload_file_to_comfy


def find_comfy_output_file(filename: str, subfolder: str = "") -> Optional[str]:
    """
    Directly checks COMFY_OUTPUT_DIR for an output file.
    Requires 'comfy_output_dir' to be configured and exist on disk.
    """
    output_dir = config.COMFY_OUTPUT_DIR
    if not output_dir or not Path(output_dir).is_dir():
        raise RuntimeError(
            f"ComfyUI output directory is not configured or does not exist on disk: '{output_dir}'. "
            f"Please set a valid 'comfy_output_dir' in config.json."
        )

    p = (Path(output_dir) / subfolder / filename) if subfolder else (Path(output_dir) / filename)
    if p.is_file():
        return p.resolve().as_posix()
    return None
