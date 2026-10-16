#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic1111 WebUI HTTP client for SD 1.5, SDXL, and Pony models.
"""

import sys
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, List
from .config import SD_WEBUI_URL


def log(msg: str):
    sys.stderr.write(f"[a1111-client] {msg}\n")
    sys.stderr.flush()


def make_request(endpoint: str, data: dict = None, method: str = "GET", timeout: int = 300) -> dict:
    url = f"{SD_WEBUI_URL}{endpoint}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "media-mcp/1.0")

    body = json.dumps(data).encode("utf-8") if data is not None else None
    with urllib.request.urlopen(req, data=body, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_models() -> List[Dict[str, Any]]:
    return make_request("/sdapi/v1/sd-models")


def get_current_model() -> Optional[str]:
    try:
        opt = make_request("/sdapi/v1/options")
        return opt.get("sd_model_checkpoint")
    except Exception:
        return None


def set_model(model_title: str) -> bool:
    try:
        make_request("/sdapi/v1/options", {"sd_model_checkpoint": model_title}, method="POST")
        return True
    except Exception as ex:
        log(f"Failed to set model: {ex}")
        return False


def get_loras() -> List[Dict[str, Any]]:
    return make_request("/sdapi/v1/loras")


def txt2img(payload: Dict[str, Any]) -> Dict[str, Any]:
    return make_request("/sdapi/v1/txt2img", payload, method="POST")


def interrupt() -> bool:
    try:
        make_request("/sdapi/v1/interrupt", method="POST")
        return True
    except Exception:
        return False


def get_progress() -> Dict[str, Any]:
    try:
        return make_request("/sdapi/v1/progress?skip_current_image=true", timeout=5)
    except Exception:
        return {}
