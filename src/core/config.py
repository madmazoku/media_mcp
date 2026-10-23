#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for media_mcp.
Strictly requires explicit configuration via:
1. CLI argument: --config <path_to_config.json> (or -c)
2. Environment variable: MEDIA_CONFIG=<path_to_config.json>

Zero heuristic auto-discovery in CWD. Zero hardcoded local paths.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from core.schema_validator import validate_config

# 1. Resolve configuration JSON file path
config_file_path: Optional[Path] = None

# Check CLI arguments: --config <path> or -c <path>
for idx, arg in enumerate(sys.argv):
    if arg in ("--config", "-c") and idx + 1 < len(sys.argv):
        config_file_path = Path(sys.argv[idx + 1])
        break

# Check environment variable
if not config_file_path and os.environ.get("MEDIA_CONFIG"):
    config_file_path = Path(os.environ["MEDIA_CONFIG"])

cfg_json: Dict[str, Any] = {}
COMFY_URL: str = ""
SD_WEBUI_URL: str = ""
OUTPUT_DIR: str = ""
OUTPUT_IMG_DIR: str = ""
OUTPUT_VID_DIR: str = ""
TEMP_DIR: str = ""
WORKFLOWS_DIR: str = ""
COMFY_DIR: str = ""
COMFY_INPUT_DIR: str = ""
COMFY_OUTPUT_DIR: str = ""
MEDIA_MCP_PORT: int = 7865
MAX_VIDEO_DURATION: float = 10.0
FFMPEG_BIN: str = ""
FFPROBE_BIN: str = ""


def _apply_config(data: Dict[str, Any]) -> None:
    global cfg_json, COMFY_URL, SD_WEBUI_URL, OUTPUT_DIR
    global OUTPUT_IMG_DIR, OUTPUT_VID_DIR, TEMP_DIR, WORKFLOWS_DIR
    global COMFY_DIR, COMFY_INPUT_DIR, COMFY_OUTPUT_DIR
    global MEDIA_MCP_PORT, MAX_VIDEO_DURATION, FFMPEG_BIN, FFPROBE_BIN

    cfg_json = dict(data)
    validate_config(cfg_json)

    # Network endpoints
    COMFY_URL = str(cfg_json.get("comfy_url", "")).rstrip("/")
    SD_WEBUI_URL = str(cfg_json.get("sd_webui_url", "")).rstrip("/")

    # Storage Directories (derived from root output_dir)
    OUTPUT_DIR = Path(str(cfg_json.get("output_dir", ""))).resolve().as_posix() if cfg_json.get("output_dir") else ""
    OUTPUT_IMG_DIR = (Path(OUTPUT_DIR) / "images").as_posix() if OUTPUT_DIR else ""
    OUTPUT_VID_DIR = (Path(OUTPUT_DIR) / "videos").as_posix() if OUTPUT_DIR else ""
    TEMP_DIR = (Path(OUTPUT_DIR) / "temp").as_posix() if OUTPUT_DIR else ""
    WORKFLOWS_DIR = (Path(OUTPUT_DIR) / "workflows").as_posix() if OUTPUT_DIR else ""

    # ComfyUI directory integration (derived from comfy_dir)
    COMFY_DIR = Path(str(cfg_json.get("comfy_dir", ""))).resolve().as_posix() if cfg_json.get("comfy_dir") else ""
    COMFY_INPUT_DIR = (Path(COMFY_DIR) / "input").as_posix() if COMFY_DIR else ""
    COMFY_OUTPUT_DIR = (Path(COMFY_DIR) / "output").as_posix() if COMFY_DIR else ""

    # MCP SSE server port (single unified port)
    MEDIA_MCP_PORT = int(cfg_json.get("media_mcp_port", 7865))

    # Single-pass block diffusion window
    MAX_VIDEO_DURATION = float(cfg_json.get("max_video_duration", 10.0))

    # Binaries: explicit config takes precedence, fallback to system PATH via shutil.which
    cfg_ffmpeg = cfg_json.get("ffmpeg_bin")
    if cfg_ffmpeg:
        FFMPEG_BIN = str(cfg_ffmpeg)
    else:
        FFMPEG_BIN = shutil.which("ffmpeg") or ""

    cfg_ffprobe = cfg_json.get("ffprobe_bin")
    if cfg_ffprobe:
        FFPROBE_BIN = str(cfg_ffprobe)
    else:
        FFPROBE_BIN = shutil.which("ffprobe") or ""

    ensure_directories()


def is_configured() -> bool:
    """Returns True if a valid configuration file was loaded."""
    return bool(cfg_json)


def get_temp_dir() -> Path:
    """Returns configured TEMP_DIR Path if valid, or a standard safe system temporary directory."""
    if TEMP_DIR and Path(TEMP_DIR).is_dir():
        return Path(TEMP_DIR)
    import tempfile
    t = Path(tempfile.gettempdir()) / "media_mcp"
    t.mkdir(parents=True, exist_ok=True)
    return t


def require_configured(context: str = "This operation") -> None:
    """Raises RuntimeError if no valid configuration file was loaded."""
    if not is_configured():
        raise RuntimeError(
            f"{context} requires a configuration file.\n"
            f"Please specify --config <path_to_config.json> or set the MEDIA_CONFIG environment variable."
        )


def require_ffmpeg(context: str = "Video processing") -> None:
    """Raises RuntimeError if FFmpeg or FFprobe executable is not available."""
    if not FFMPEG_BIN or (Path(FFMPEG_BIN).is_absolute() and not Path(FFMPEG_BIN).is_file()):
        raise RuntimeError(
            f"{context} requires FFmpeg.\n"
            f"Please install FFmpeg into your system PATH or specify 'ffmpeg_bin' in config.json."
        )
    if not FFPROBE_BIN or (Path(FFPROBE_BIN).is_absolute() and not Path(FFPROBE_BIN).is_file()):
        raise RuntimeError(
            f"{context} requires FFprobe.\n"
            f"Please install FFprobe into your system PATH or specify 'ffprobe_bin' in config.json."
        )


def ensure_directories():
    """Ensures configured output and workflow storage directories exist."""
    for d in [OUTPUT_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR, TEMP_DIR, WORKFLOWS_DIR]:
        if d:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


def load_config_from_file(path: Path) -> None:
    """Load, validate, and apply configuration from a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _apply_config(data)
    sys.stderr.write(f"[config] Loaded and validated configuration from {path.resolve()}\n")


def load_config_dict(data: Dict[str, Any]) -> None:
    """Apply an in-memory configuration dictionary (used in tests and embeddings)."""
    _apply_config(data)


# Auto-load on module import if config path was explicitly provided
if config_file_path and config_file_path.is_file():
    load_config_from_file(config_file_path)
