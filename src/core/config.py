#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for media-mcp.
Supports:
1. CLI argument: --config <path_to_config.json>
2. Environment variable: MEDIA_CONFIG=<path_to_config.json>
3. Local config file: media_mcp/config.json or config.json
4. Automatic fallback to portable defaults (zero hardcoded personal paths).
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Any

from core.schema_validator import validate_config

# Base repository directory
REPO_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Resolve configuration JSON file path
config_file_path = None

# Check CLI arguments: --config <path> or -c <path>
for idx, arg in enumerate(sys.argv):
    if arg in ("--config", "-c") and idx + 1 < len(sys.argv):
        config_file_path = Path(sys.argv[idx + 1])
        break

# Check environment variable
if not config_file_path and os.environ.get("MEDIA_CONFIG"):
    config_file_path = Path(os.environ["MEDIA_CONFIG"])

# Check CWD candidates
if not config_file_path:
    for candidate in [Path("media_mcp/config.json"), Path("config.json"), REPO_DIR / "config.json"]:
        if candidate.exists():
            config_file_path = candidate
            break

# Load and validate JSON configuration if present
cfg_json: dict = {}
if config_file_path and config_file_path.exists():
    with open(config_file_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    validate_config(cfg_json)
    sys.stderr.write(f"[config] Loaded and validated configuration from {config_file_path.resolve()}\n")

# Network endpoints
COMFY_URL = str(cfg_json.get("comfy_url", "http://127.0.0.1:8188")).rstrip("/")
SD_WEBUI_URL = str(cfg_json.get("sd_webui_url", "http://127.0.0.1:7860")).rstrip("/")

# Binaries (default: find in PATH, no hardcoded drive letters)
default_ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
default_ffprobe = shutil.which("ffprobe") or "ffprobe"
FFMPEG_BIN = str(cfg_json.get("ffmpeg_bin", default_ffmpeg))
FFPROBE_BIN = str(cfg_json.get("ffprobe_bin", default_ffprobe))

# Storage Directories (configured via config.json)
OUTPUT_DIR = str(cfg_json.get("output_dir", ""))
OUTPUT_IMG_DIR = str(cfg_json.get("output_img_dir", os.path.join(OUTPUT_DIR, "images") if OUTPUT_DIR else ""))
OUTPUT_VID_DIR = str(cfg_json.get("output_vid_dir", os.path.join(OUTPUT_DIR, "videos") if OUTPUT_DIR else ""))
TEMP_DIR = str(cfg_json.get("temp_dir", os.path.join(OUTPUT_DIR, "temp") if OUTPUT_DIR else ""))
WORKFLOWS_DIR = str(cfg_json.get("workflows_dir", ""))

# ComfyUI filesystem integration (configured via config.json)
COMFY_INPUT_DIR = str(cfg_json.get("comfy_input_dir", ""))
COMFY_OUTPUT_DIR = str(cfg_json.get("comfy_output_dir", ""))

# MCP SSE server ports
MEDIA_MCP_PORT = int(cfg_json.get("media_mcp_port", cfg_json.get("image_mcp_port", 7865)))
IMAGE_MCP_PORT = MEDIA_MCP_PORT
VIDEO_MCP_PORT = int(cfg_json.get("video_mcp_port", 7866))

# Maximum single-pass block diffusion window (prefix_duration + clip_duration < max_video_duration)
MAX_VIDEO_DURATION = float(cfg_json.get("max_video_duration", 10.0))


def is_configured() -> bool:
    """Returns True if a valid configuration file was loaded."""
    return bool(cfg_json)


def require_configured(context: str = "This operation") -> None:
    """Raises RuntimeError if no valid configuration file was loaded."""
    if not is_configured():
        raise RuntimeError(
            f"{context} requires a configuration file. "
            f"Please specify --config <path_to_config.json> or set MEDIA_CONFIG=<path>."
        )


def ensure_directories():
    """Ensures configured output and workflow storage directories exist."""
    for d in [OUTPUT_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR, TEMP_DIR, WORKFLOWS_DIR]:
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass


if is_configured():
    ensure_directories()

