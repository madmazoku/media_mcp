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

# Base repository directory
REPO_DIR = Path(__file__).resolve().parent.parent.parent

# 1. Resolve configuration JSON file path
config_file_path = None

# Check CLI arguments: --config <path>
for idx, arg in enumerate(sys.argv):
    if arg == "--config" and idx + 1 < len(sys.argv):
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

# Load JSON configuration if present
cfg_json: dict = {}
if config_file_path and config_file_path.exists():
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        sys.stderr.write(f"[config] Loaded configuration from {config_file_path.resolve()}\n")
    except Exception as ex:
        sys.stderr.write(f"[config] Warning: Failed to read {config_file_path}: {ex}\n")

def get_setting(key: str, env_name: str, default: Any) -> Any:
    """Settings priority: Environment Variable > config.json > Default."""
    if env_name in os.environ:
        return os.environ[env_name]
    if key in cfg_json:
        return cfg_json[key]
    return default

# Network endpoints
COMFY_URL = str(get_setting("comfy_url", "COMFY_URL", "http://127.0.0.1:8188")).rstrip("/")
SD_WEBUI_URL = str(get_setting("sd_webui_url", "SD_WEBUI_URL", "http://127.0.0.1:7860")).rstrip("/")

# Binaries (default: find in PATH, no hardcoded drive letters)
default_ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
default_ffprobe = shutil.which("ffprobe") or "ffprobe"
FFMPEG_BIN = str(get_setting("ffmpeg_bin", "FFMPEG_BIN", default_ffmpeg))
FFPROBE_BIN = str(get_setting("ffprobe_bin", "FFPROBE_BIN", default_ffprobe))

# Storage Directories
OUTPUT_DIR = str(get_setting("output_dir", "MEDIA_OUTPUT_DIR", str(REPO_DIR / "output")))
OUTPUT_IMG_DIR = str(get_setting("output_img_dir", "MEDIA_OUTPUT_IMG_DIR", os.path.join(OUTPUT_DIR, "images")))
OUTPUT_VID_DIR = str(get_setting("output_vid_dir", "MEDIA_OUTPUT_VID_DIR", os.path.join(OUTPUT_DIR, "videos")))
TEMP_DIR = str(get_setting("temp_dir", "MEDIA_TEMP_DIR", os.path.join(OUTPUT_DIR, "temp")))
WORKFLOWS_DIR = str(get_setting("workflows_dir", "WORKFLOWS_DIR", str(REPO_DIR / "workflows")))

# ComfyUI filesystem integration (optional for local setups)
COMFY_INPUT_DIR = str(get_setting("comfy_input_dir", "COMFY_INPUT_DIR", ""))
COMFY_OUTPUT_DIR = str(get_setting("comfy_output_dir", "COMFY_OUTPUT_DIR", ""))

# MCP SSE server ports
IMAGE_MCP_PORT = int(get_setting("image_mcp_port", "IMAGE_MCP_PORT", 7865))
VIDEO_MCP_PORT = int(get_setting("video_mcp_port", "VIDEO_MCP_PORT", 7866))

def ensure_directories():
    for d in [OUTPUT_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR, TEMP_DIR, WORKFLOWS_DIR]:
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as ex:
                sys.stderr.write(f"[config] Notice: could not create directory {d}: {ex}\n")

ensure_directories()

