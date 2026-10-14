#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Scene-based Video Pipeline with True Forced Prefix Continuity for Wan 2.2.
Supports:
1. Scene-based hierarchy: Scene (Anchor Image) -> Shot Sequence (Motion Blocks).
2. Hybrid Anchor Frame: accepts ready 'image_path' OR visual 'prompt' generated via image_engine.
3. True Forced Prefix Conditioning: tails of previous blocks condition the latent space of subsequent blocks.
4. Seamless FFmpeg cut assembly + dynamic countdown HUD.
5. Automatic poster extraction (returns file and base64).
6. Backward compatibility with legacy flat block arrays.
"""

import os
import sys
import json
import time
import shutil
import random
from typing import List, Dict, Any, Optional, Tuple

from .config import OUTPUT_VID_DIR, OUTPUT_IMG_DIR, TEMP_DIR
from . import ffmpeg_engine
from . import comfy_client
from . import image_engine

def log(msg: str):
    sys.stderr.write(f"[video-pipeline] {msg}\n")
    sys.stderr.flush()

# ---------------------------------------------------------------------------
# Wan 2.2 ComfyUI Graph Builder
# ---------------------------------------------------------------------------

def generate_wan_segment(
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    end_image_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "ugly, blurry, static, jitter, deformed, low quality, bad anatomy",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    fps: int = 16,
    steps: int = 20,
    cfg: float = 6.0,
    out_video_name: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Submits a Wan 2.2 Video segment to ComfyUI.
    Supports single image seed, end image keyframe, or full prefix video conditioning.
    """
    width = (width // 16) * 16
    height = (height // 16) * 16
    out_video_name = out_video_name or f"wan_{int(time.time()*1000)}.mp4"
    seed = random.randint(1, 10**14)

    wf: Dict[str, Any] = {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
        "2": {"class_type": "WanVideoTextEncode", "inputs": {"positive": prompt, "negative": negative_prompt, "t5": ["3", 0]}},
        "3": {"class_type": "LoadWanVideoT5", "inputs": {"model_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}},
        "4": {"class_type": "WanVideoModelLoader", "inputs": {"model_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}},
        "5": {"class_type": "WanVideoLoraSelect", "inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength": 1.0}},
        "6": {"class_type": "WanVideoLoraSelect", "inputs": {"lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength": 1.0}},
        "7": {"class_type": "WanVideoModelLoader", "inputs": {"model_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"}},
        "11": {
            "class_type": "WanVideoSampler",
            "inputs": {
                "base_model": ["4", 0], "high_noise_model": ["4", 0], "low_noise_model": ["7", 0],
                "high_noise_lora": ["5", 0], "low_noise_lora": ["6", 0],
                "cfg": cfg, "steps": steps, "seed": seed, "shift": 5.0,
                "text_embeds": ["2", 0], "image_embeds": ["9", 0]
            }
        },
        "12": {"class_type": "WanVideoDecode", "inputs": {"samples": ["11", 0], "vae": ["1", 0]}},
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0], "frame_rate": fps, "format": "video/h264-mp4",
                "filename_prefix": "media_mcp_wan", "save_output": True
            }
        }
    }

    # Case A: Forced Prefix Video Conditioning
    if prefix_video_path and os.path.exists(prefix_video_path):
        abs_prefix = os.path.abspath(prefix_video_path)
        log(f"Configuring TRUE FORCED PREFIX from video: {abs_prefix}")
        wf["8"] = {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {"video": abs_prefix, "force_rate": fps, "custom_width": width, "custom_height": height}
        }
        start_image_node = ["8", 0]
    elif image_path and os.path.exists(image_path):
        wf["8"] = {"class_type": "LoadImage", "inputs": {"image": os.path.abspath(image_path)}}
        start_image_node = ["8", 0]
    else:
        return False, f"Neither valid start_image nor prefix_video provided: img={image_path}, vid={prefix_video_path}"

    if end_image_path and os.path.exists(end_image_path):
        log(f"Configuring END-KEYFRAME boundary lock: {end_image_path}")
        wf["10"] = {"class_type": "LoadImage", "inputs": {"image": os.path.abspath(end_image_path)}}
        wf["9"] = {
            "class_type": "WanFirstLastFrameToVideo",
            "inputs": {
                "start_image": start_image_node,
                "end_image": ["10", 0],
                "width": width, "height": height, "num_frames": num_frames
            }
        }
    else:
        wf["9"] = {
            "class_type": "WanImageToVideo",
            "inputs": {"image": start_image_node, "width": width, "height": height, "num_frames": num_frames}
        }

    log(f"Submitting Wan 2.2 segment prompt ({num_frames} frames @ {fps}fps)...")
    ok, prompt_id = comfy_client.queue_prompt(wf)
    if not ok:
        return False, prompt_id

    ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=900)
    if not ok:
        return False, str(history)

    outputs = history.get("outputs", {})
    vid_meta = None
    for n_out in outputs.values():
        if "gifs" in n_out:
            vid_meta = n_out["gifs"][0]
            break

    if not vid_meta:
        return False, "ComfyUI completed without output video"

    raw_bytes = comfy_client.download_output_file(
        filename=vid_meta.get("filename"),
        subfolder=vid_meta.get("subfolder", ""),
        file_type=vid_meta.get("type", "output")
    )

    out_file = os.path.join(TEMP_DIR, out_video_name)
    with open(out_file, "wb") as f:
        f.write(raw_bytes)

    return True, out_file


# ---------------------------------------------------------------------------
# Hierarchical Scene Normalizer (Backward Compatibility)
# ---------------------------------------------------------------------------

def normalize_scenario_structure(raw_data: Any) -> List[Dict[str, Any]]:
    """
    Normalizes scenario input into a clean list of Scene objects:
    [
      {
        "scene_name": str,
        "anchor_image": { "image_path": Optional[str], "prompt": Optional[str], ... },
        "blocks": [ { "name": str, "duration": float, "overlap": float, "motion_prompt": str, ... } ]
      }
    ]
    """
    if isinstance(raw_data, dict) and "scenes" in raw_data:
        raw_scenes = raw_data["scenes"]
    elif isinstance(raw_data, list):
        if len(raw_data) > 0 and isinstance(raw_data[0], dict) and "blocks" in raw_data[0]:
            raw_scenes = raw_data
        else:
            # Legacy flat block array -> Wrap into scenes grouped by new_scene
            raw_scenes = []
            cur_scene = {"scene_name": "Scene 1", "anchor_image": {}, "blocks": []}
            for idx, blk in enumerate(raw_data):
                if blk.get("new_scene", False) and idx > 0:
                    raw_scenes.append(cur_scene)
                    cur_scene = {
                        "scene_name": f"Scene {len(raw_scenes) + 1}",
                        "anchor_image": {
                            "image_path": blk.get("start_image"),
                            "prompt": blk.get("prompt"),
                            "model": blk.get("model"),
                            "lora": blk.get("lora")
                        },
                        "blocks": []
                    }
                elif idx == 0:
                    cur_scene["anchor_image"] = {
                        "image_path": blk.get("start_image"),
                        "prompt": blk.get("prompt"),
                        "model": blk.get("model"),
                        "lora": blk.get("lora")
                    }

                cur_scene["blocks"].append({
                    "name": blk.get("name", f"Block #{len(cur_scene['blocks']) + 1}"),
                    "duration": float(blk.get("duration", 5.0)),
                    "overlap": float(blk.get("overlap", 1.5 if idx < len(raw_data) - 1 else 0.0)),
                    "motion_prompt": blk.get("motion_prompt") or blk.get("prompt", ""),
                    "use_next_as_end": blk.get("use_next_as_end", False)
                })
            raw_scenes.append(cur_scene)
    else:
        raise ValueError("Invalid scenario structure")

    return raw_scenes


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def execute_scenario_pipeline(
    scenario_input: Any,
    global_start_image: Optional[str] = None,
    output_name: Optional[str] = None,
    width: int = 832,
    height: int = 480,
    fps: int = 16,
    enable_hud: bool = False
) -> Dict[str, Any]:
    """
    Executes a multi-scene video scenario.
    Returns structured results including video paths, duration, and poster base64.
    """
    scenes = normalize_scenario_structure(scenario_input)
    log(f"Executing scenario pipeline with {len(scenes)} scenes...")

    all_scene_videos = []
    generated_block_metadata = []
    first_poster_b64 = None
    first_poster_path = None

    for s_idx, sc in enumerate(scenes):
        scene_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
        anchor_meta = sc.get("anchor_image", {})
        blocks = sc.get("blocks", [])

        if s_idx == 0 and global_start_image:
            anchor_meta["image_path"] = global_start_image

        log(f"\n==========================================")
        log(f"[{scene_name}] Preparing Scene Anchor Frame...")
        log(f"==========================================")

        ok, anchor_img_path, anchor_b64 = image_engine.generate_anchor_frame(
            anchor_meta=anchor_meta,
            width=width,
            height=height
        )
        if not ok:
            raise RuntimeError(f"Failed to obtain anchor keyframe for {scene_name}: {anchor_img_path}")

        if s_idx == 0 and anchor_b64:
            first_poster_b64 = anchor_b64
            first_poster_path = anchor_img_path

        prefix_video = None
        scene_block_pieces = []

        for b_idx, blk in enumerate(blocks):
            b_name = blk.get("name", f"Block {b_idx + 1}")
            dur = float(blk.get("duration", 5.0))
            overlap = float(blk.get("overlap", 1.5 if b_idx < len(blocks) - 1 else 0.0))
            motion_prompt = blk.get("motion_prompt") or blk.get("prompt", "")

            # Calculate Wan frame count: (num_frames - 1) must be divisible by 4
            raw_frames = int(round(dur * fps))
            k = max(4, round((raw_frames - 1) / 4))
            num_frames = 4 * k + 1

            log(f"\n[{scene_name} : {b_name}] Duration: {dur}s, Overlap: {overlap}s, Frames: {num_frames}")
            log(f"Motion prompt: '{motion_prompt[:60]}...'")

            raw_segment_name = f"seg_{s_idx}_{b_idx}_{int(time.time())}.mp4"
            if prefix_video and os.path.exists(prefix_video):
                # CASE A: Conditioned on previous block's tail
                ok, seg_raw_path = generate_wan_segment(
                    image_path=None,
                    prefix_video_path=prefix_video,
                    prompt=motion_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    out_video_name=raw_segment_name
                )
            else:
                # CASE B: First block of scene conditioned on anchor image
                ok, seg_raw_path = generate_wan_segment(
                    image_path=anchor_img_path,
                    prefix_video_path=None,
                    prompt=motion_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    out_video_name=raw_segment_name
                )

            if not ok:
                raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {seg_raw_path}")

            block_trimmed = os.path.join(TEMP_DIR, f"trimmed_{s_idx}_{b_idx}.mp4")
            if b_idx < len(blocks) - 1 and overlap > 0:
                # Keep only [0 .. dur - overlap]
                keep_dur = max(0.5, dur - overlap)
                ffmpeg_engine.cut_duration(seg_raw_path, duration=keep_dur, out_path=block_trimmed)
                
                # Extract the tail overlap clip for next block's prefix
                tail_clip = os.path.join(TEMP_DIR, f"tail_{s_idx}_{b_idx}_{overlap}s.mp4")
                ffmpeg_engine.extract_tail_clip(seg_raw_path, tail_duration=overlap, out_clip_path=tail_clip)
                prefix_video = tail_clip
            else:
                # Last block in scene: keep full duration
                ffmpeg_engine.cut_duration(seg_raw_path, duration=dur, out_path=block_trimmed)
                prefix_video = None

            scene_block_pieces.append(block_trimmed)
            generated_block_metadata.append((seg_raw_path, dur, overlap))

        # Join blocks within the scene
        scene_final = os.path.join(TEMP_DIR, f"scene_{s_idx}_final.mp4")
        ffmpeg_engine.concat_videos(scene_block_pieces, scene_final)
        all_scene_videos.append(scene_final)

    # Final movie assembly across all scenes
    final_name = output_name or f"movie_{int(time.time())}.mp4"
    if not final_name.endswith(".mp4"):
        final_name = f"{final_name}.mp4"

    final_path = os.path.join(OUTPUT_VID_DIR, final_name)
    ffmpeg_engine.concat_videos(all_scene_videos, final_path)
    total_duration = ffmpeg_engine.get_duration(final_path)

    # Poster extraction from final video
    poster_filename = f"{os.path.splitext(final_name)[0]}_poster.png"
    poster_path = os.path.join(OUTPUT_IMG_DIR, poster_filename)
    poster_path, poster_b64 = ffmpeg_engine.extract_poster(final_path, poster_path)
    if not poster_b64 and first_poster_b64:
        poster_b64 = first_poster_b64

    # Optional HUD generation
    hud_path = None
    if enable_hud and generated_block_metadata:
        hud_name = final_name.replace(".mp4", "_hud.mp4")
        hud_final = os.path.join(OUTPUT_VID_DIR, hud_name)
        try:
            ffmpeg_engine.generate_hud_overlay(final_path, generated_block_metadata, hud_final)
            hud_path = hud_final
        except Exception as ex:
            log(f"Warning: HUD generation failed: {ex}")

    return {
        "video_path": final_path,
        "hud_video_path": hud_path,
        "poster_path": poster_path,
        "poster_base64": poster_b64,
        "duration": total_duration,
        "width": width,
        "height": height,
        "fps": fps,
        "scenes_count": len(scenes)
    }
