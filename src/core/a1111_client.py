#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic1111 WebUI HTTP client for SD 1.5, SDXL, and Pony models.
"""

import sys
from typing import Dict, Any, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


def log(msg: str):
    sys.stderr.write(f"[a1111-client] {msg}\n")
    sys.stderr.flush()


_SESSION: Optional[requests.Session] = None


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
    base = (config.SD_WEBUI_URL or "").strip()
    if not base:
        raise ValueError("Automatic1111 WebUI URL is not configured. Please set 'sd_webui_url' in config.json.")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base.rstrip('/')}{ep}"


def make_request(endpoint: str, data: dict = None, method: str = "GET", timeout: int = 30) -> dict:
    url = _format_url(endpoint)
    s = get_session()
    m = method.upper()
    if m == "GET":
        resp = s.get(url, timeout=timeout)
    elif m == "POST":
        resp = s.post(url, json=data, timeout=timeout)
    else:
        resp = s.request(m, url, json=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_models() -> List[Dict[str, Any]]:
    return make_request("/sdapi/v1/sd-models", timeout=10)


def get_current_model() -> Optional[str]:
    try:
        opt = make_request("/sdapi/v1/options", timeout=5)
        return opt.get("sd_model_checkpoint")
    except Exception:
        return None


def set_model(model_title: str) -> bool:
    try:
        make_request("/sdapi/v1/options", {"sd_model_checkpoint": model_title}, method="POST", timeout=120)
        return True
    except Exception as ex:
        log(f"Failed to set model: {ex}")
        return False


def get_loras() -> List[Dict[str, Any]]:
    return make_request("/sdapi/v1/loras", timeout=10)


def get_samplers() -> List[Dict[str, Any]]:
    return make_request("/sdapi/v1/samplers", timeout=10)


def is_available() -> bool:
    try:
        make_request("/sdapi/v1/progress", timeout=3)
        return True
    except Exception:
        return False


def txt2img(payload: Dict[str, Any]) -> Dict[str, Any]:
    return make_request("/sdapi/v1/txt2img", payload, method="POST", timeout=600)


def img2img(payload: Dict[str, Any]) -> Dict[str, Any]:
    return make_request("/sdapi/v1/img2img", payload, method="POST", timeout=600)


def interrupt() -> bool:
    try:
        make_request("/sdapi/v1/interrupt", method="POST", timeout=5)
        return True
    except Exception:
        return False


def get_progress() -> Dict[str, Any]:
    try:
        return make_request("/sdapi/v1/progress?skip_current_image=true", timeout=3)
    except Exception:
        return {}
