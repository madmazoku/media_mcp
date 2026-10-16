#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFmpeg processing engine for duration probing, lossless Stream Copy concat, poster extraction, and HUD.
"""

import os
import sys
import time
import shutil
import base64
import subprocess
from typing import List, Tuple, Optional
from .config import FFMPEG_BIN, FFPROBE_BIN, TEMP_DIR


def log(msg: str):
    sys.stderr.write(f"[ffmpeg-engine] {msg}\n")
    sys.stderr.flush()


def get_duration(path: str) -> float:
    cmd = [
        FFPROBE_BIN,
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

    concat_txt = os.path.join(TEMP_DIR, f"concat_{int(time.time()*1000)}.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for v in video_list:
            f.write(f"file '{v.replace(os.sep, '/')}'\n")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        out_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    try:
        os.remove(concat_txt)
    except Exception:
        pass
    return out_path


def extract_poster(video_path: str, out_image_path: str) -> Tuple[str, Optional[str]]:
    """Extracts first frame as poster and returns (file_path, base64_string)."""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", "0.0",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_image_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        with open(out_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return out_image_path, b64
    except Exception as ex:
        log(f"Failed to extract poster: {ex}")
        return out_image_path, None


def generate_hud_overlay(video_path: str, generated_blocks: List[Tuple[str, float, float]], out_hud_path: str) -> str:
    """Generate debug HUD video where each block spans its full core clip duration."""
    escaped_font = "C\\:/Windows/Fonts/arial.ttf"
    hud_filters = []
    cur_t = 0.0

    for i, (_b_path, b_dur, b_overlap) in enumerate(generated_blocks):
        block_end = cur_t + b_dur
        is_last = i == len(generated_blocks) - 1
        ov_info = f" | overlap->{b_overlap:.1f}s" if (not is_last and b_overlap > 0) else ""
        label = f"Block {i+1}{ov_info}" if not is_last else f"Block {i+1} [Final]"
        color = "white" if not is_last else "cyan"
        cd_expr = f"%{{eif\\:max(0,({block_end:.2f}-t))\\:d}}.%{{eif\\:mod(max(0,({block_end:.2f}-t)*10)\\,10)\\:d}}s left"
        draw = (
            f"drawtext=fontfile='{escaped_font}':"
            f"text='{label} ({cd_expr})':"
            f"fontsize=24:fontcolor={color}:"
            f"box=1:boxcolor=black@0.65:boxborderw=8:"
            f"x=(w-text_w)/2:y=24:"
            f"enable='between(t\\,{cur_t:.2f}\\,{block_end:.2f})'"
        )
        hud_filters.append(draw)
        cur_t = block_end

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-vf", ",".join(hud_filters),
        "-c:v", "h264_nvenc",
        "-preset", "p6",
        "-cq", "18",
        "-pix_fmt", "yuv420p",
        out_hud_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_hud_path
