#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for media-mcp.
Reads from environment variables with sensible defaults.
"""

import os
import sys
from pathlib import Path

# Base repository directory
REPO_DIR = Path(__file__).resolve().parent.parent.parent

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
SD_WEBUI_URL = os.environ.get("SD_WEBUI_URL", "http://127.0.0.1:7860").rstrip("/")

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", r"E:\Git\ffmpeg\bin\ffmpeg.exe")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", r"E:\Git\ffmpeg\bin\ffprobe.exe")

OUTPUT_DIR = os.environ.get("MEDIA_OUTPUT_DIR", r"G:\Git\Gemini\output")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_VID_DIR = os.path.join(OUTPUT_DIR, "video")
TEMP_DIR = os.environ.get("MEDIA_TEMP_DIR", str(REPO_DIR / "temp"))
WORKFLOWS_DIR = os.environ.get("WORKFLOWS_DIR", str(REPO_DIR / "workflows"))

COMFY_INPUT_DIR = os.environ.get("COMFY_INPUT_DIR", r"G:\Git\ComfyUI\input")
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR", r"G:\Git\ComfyUI\output")

IMAGE_MCP_PORT = int(os.environ.get("IMAGE_MCP_PORT", "7865"))
VIDEO_MCP_PORT = int(os.environ.get("VIDEO_MCP_PORT", "7866"))

def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VID_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(WORKFLOWS_DIR, exist_ok=True)

ensure_directories()
