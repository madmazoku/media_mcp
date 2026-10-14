#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFmpeg processing engine for cutting, seamless concat, poster extraction, and HUD.
"""

import os
import sys
import time
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

def extract_tail_clip(video_path: str, tail_duration: float, out_clip_path: str) -> str:
    dur = get_duration(video_path)
    start_time = max(0.0, dur - tail_duration)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(tail_duration),
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-cq", "12",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        out_clip_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_clip_path

def cut_duration(video_path: str, duration: float, out_path: str) -> str:
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-t", str(duration),
        "-i", video_path,
        "-c:v", "h264_nvenc",
        "-preset", "p7",
        "-cq", "12",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_path

def concat_videos(video_list: List[str], out_path: str, smooth_transitions: bool = False) -> str:
    concat_txt = os.path.join(TEMP_DIR, f"concat_{int(time.time()*1000)}.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for v in video_list:
            f.write(f"file '{v.replace(os.sep, '/')}'\n")

    vf_filters = []
    if smooth_transitions and len(video_list) > 1:
        # Stabilize luminance (Y) and temporal chroma (U/V) across prefix stitch points
        vf_filters.append("deflicker=mode=pm:size=7")
        vf_filters.append("tmidequalizer=radius=5:sigma=0.3")

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt
    ]
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend([
        "-c:v", "h264_nvenc",
        "-preset", "p6",
        "-cq", "18",
        "-pix_fmt", "yuv420p",
        out_path
    ])
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
    escaped_font = "C\\:/Windows/Fonts/arial.ttf"
    hud_filters = []
    cur_t = 0.0

    for i, (b_path, b_dur, b_overlap) in enumerate(generated_blocks):
        pure_dur = b_dur - b_overlap if i < len(generated_blocks) - 1 else b_dur
        phase_end = cur_t + pure_dur
        label = f"Блок {i+1}" if i < len(generated_blocks) - 1 else f"Блок {i+1} [Финал]"
        color = "white" if i < len(generated_blocks) - 1 else "cyan"
        cd_expr = f"%{{eif\\:max(0,({phase_end:.2f}-t))\\:d}}.%{{eif\\:mod(max(0,({phase_end:.2f}-t)*10)\\,10)\\:d}}s left"
        draw = (
            f"drawtext=fontfile='{escaped_font}':"
            f"text='{label} ({cd_expr})':"
            f"fontsize=24:fontcolor={color}:"
            f"box=1:boxcolor=black@0.65:boxborderw=8:"
            f"x=(w-text_w)/2:y=24:"
            f"enable='between(t\\,{cur_t:.2f}\\,{phase_end:.2f})'"
        )
        hud_filters.append(draw)
        cur_t = phase_end

        if i < len(generated_blocks) - 1 and b_overlap > 0:
            overlap_end = cur_t + b_overlap
            ov_label = f"Блок {i+1}+{i+2} [Пересечение {b_overlap:.1f}с]"
            cd_ov = f"%{{eif\\:max(0,({overlap_end:.2f}-t))\\:d}}.%{{eif\\:mod(max(0,({overlap_end:.2f}-t)*10)\\,10)\\:d}}s left"
            draw_ov = (
                f"drawtext=fontfile='{escaped_font}':"
                f"text='{ov_label} ({cd_ov})':"
                f"fontsize=24:fontcolor=yellow:"
                f"box=1:boxcolor=black@0.65:boxborderw=8:"
                f"x=(w-text_w)/2:y=24:"
                f"enable='between(t\\,{cur_t:.2f}\\,{overlap_end:.2f})'"
            )
            hud_filters.append(draw_ov)
            cur_t = overlap_end

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
