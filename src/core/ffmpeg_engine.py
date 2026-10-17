#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFmpeg processing engine for duration probing, lossless Stream Copy concat, and poster extraction.
"""

import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import List

from . import config


def log(msg: str):
    sys.stderr.write(f"[ffmpeg-engine] {msg}\n")
    sys.stderr.flush()


def get_duration(path: str) -> float:
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def concat_videos(video_list: List[str], out_path: str) -> str:
    """Concatenate videos using lossless FFmpeg Stream Copy (-c copy)."""
    if not video_list:
        return out_path
    if len(video_list) == 1:
        shutil.copy2(video_list[0], out_path)
        return out_path

    temp_dir = config.get_temp_dir()
    concat_path = Path(temp_dir) / f"concat-{int(time.time() * 1000)}.txt"
    concat_txt = concat_path.as_posix()
    with open(concat_txt, "w", encoding="utf-8") as f:
        for v in video_list:
            f.write(f"file '{Path(v).as_posix()}'\n")

    cmd = [
        config.FFMPEG_BIN,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        out_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    concat_path.unlink(missing_ok=True)
    return out_path


def extract_poster(video_path: str, out_image_path: str) -> str:
    """Extracts first frame of video_path as poster image on disk and returns out_image_path."""
    cmd = [
        config.FFMPEG_BIN,
        "-y",
        "-ss", "0.0",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_image_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as ex:
        log(f"Failed to extract poster: {ex}")
    return out_image_path
