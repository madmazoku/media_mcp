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

_GLOBAL_PROGRESS_CALLBACK = None

_PIPELINE_STATUS = {
    "is_running": False,
    "scenario_name": None,
    "scene_index": 0,
    "scene_name": None,
    "total_scenes": 0,
    "block_index": 0,
    "block_name": None,
    "total_blocks": 0,
    "phase": "idle",
    "elapsed_seconds": 0.0,
    "started_at": 0.0
}

def get_pipeline_status() -> Dict[str, Any]:
    st = dict(_PIPELINE_STATUS)
    if st["is_running"] and st["started_at"] > 0:
        st["elapsed_seconds"] = round(time.time() - st["started_at"], 1)
    return st

_INTERRUPT_REQUESTED = False

def request_interrupt():
    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = True
    comfy_client.interrupt()

def is_interrupted() -> bool:
    return _INTERRUPT_REQUESTED or comfy_client.is_interrupted()

def set_global_progress_callback(cb):
    global _GLOBAL_PROGRESS_CALLBACK
    _GLOBAL_PROGRESS_CALLBACK = cb
    comfy_client.set_global_progress_callback(cb)

def notify_progress(msg: str, progress: Optional[float] = None, total: Optional[float] = None, cb: Optional[Any] = None):
    callback = cb or _GLOBAL_PROGRESS_CALLBACK
    if callback:
        try:
            callback(msg, progress, total)
        except Exception:
            pass
    log(msg)

# ---------------------------------------------------------------------------
# Wan 2.2 ComfyUI Graph Builder
# ---------------------------------------------------------------------------

def generate_wan_segment(
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    end_image_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形，blurry, jitter, static, low quality, deformed, extra limbs, bad anatomy, text, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    fps: int = 16,
    seed: Optional[int] = None,
    out_video_name: Optional[str] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    Submits a Wan 2.2 Video segment to ComfyUI matching the working implementation in video_extender.py.
    Supports single image seed, end image keyframe, or full prefix video conditioning.
    """
    width = (width // 16) * 16
    height = (height // 16) * 16
    out_video_name = out_video_name or f"wan_{int(time.time()*1000)}.mp4"
    if seed is None or seed < 0:
        seed = random.randint(1, 1_000_000_000)

    prefix_str = f"wan22_{int(time.time())}"

    wf: Dict[str, Any] = {
        # 1. CLIP / Text Encoder
        "clip_loader": {
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default"
            },
            "class_type": "CLIPLoader"
        },
        # 2. VAE
        "vae_loader": {
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        # 3. High Noise Diffusion Model
        "unet_high": {
            "inputs": {
                "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        # 4. Low Noise Diffusion Model
        "unet_low": {
            "inputs": {
                "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        # 5. LoRA for High Noise (4-steps distillation)
        "lora_high": {
            "inputs": {
                "model": ["unet_high", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        },
        # 6. LoRA for Low Noise (4-steps distillation)
        "lora_low": {
            "inputs": {
                "model": ["unet_low", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        },
        # 7. ModelSamplingSD3 (shift = 5.0)
        "sampling_high": {
            "inputs": {
                "model": ["lora_high", 0],
                "shift": 5.0
            },
            "class_type": "ModelSamplingSD3"
        },
        "sampling_low": {
            "inputs": {
                "model": ["lora_low", 0],
                "shift": 5.0
            },
            "class_type": "ModelSamplingSD3"
        },
        # 8. Text Encoders
        "positive_cond": {
            "inputs": {
                "clip": ["clip_loader", 0],
                "text": prompt
            },
            "class_type": "CLIPTextEncode"
        },
        "negative_cond": {
            "inputs": {
                "clip": ["clip_loader", 0],
                "text": negative_prompt
            },
            "class_type": "CLIPTextEncode"
        }
    }

    # 9. Conditioning start image or full video prefix
    if prefix_video_path and os.path.exists(prefix_video_path):
        prefix_fn = comfy_client.upload_file_to_comfy(
            prefix_video_path,
            custom_filename=f"wan_prefix_{int(time.time()*1000)}.mp4"
        )
        abs_prefix_in_comfy = os.path.join(comfy_client.COMFY_INPUT_DIR, prefix_fn)
        log(f"Configuring TRUE FORCED PREFIX from video: {abs_prefix_in_comfy}")
        wf["prefix_video_loader"] = {
            "inputs": {
                "video": os.path.abspath(abs_prefix_in_comfy),
                "force_rate": float(fps),
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "Wan"
            },
            "class_type": "VHS_LoadVideoPath"
        }
        start_image_node = ["prefix_video_loader", 0]
    elif image_path and os.path.exists(image_path):
        img_fn = comfy_client.upload_file_to_comfy(image_path)
        log(f"Configuring start image: {img_fn}")
        wf["start_image"] = {
            "inputs": {
                "image": img_fn,
                "upload": "image"
            },
            "class_type": "LoadImage"
        }
        start_image_node = ["start_image", 0]
    else:
        return False, f"Neither valid start_image nor prefix_video provided: img={image_path}, vid={prefix_video_path}"

    # 10. Wan I2V conditioning: FirstLastFrame or Start-only
    if end_image_path and os.path.exists(end_image_path):
        end_fn = comfy_client.upload_file_to_comfy(end_image_path)
        log(f"Configuring END-KEYFRAME boundary lock: {end_fn}")
        wf["end_image"] = {
            "inputs": {
                "image": end_fn,
                "upload": "image"
            },
            "class_type": "LoadImage"
        }
        wf["wan_i2v"] = {
            "inputs": {
                "positive": ["positive_cond", 0],
                "negative": ["negative_cond", 0],
                "vae": ["vae_loader", 0],
                "start_image": start_image_node,
                "end_image": ["end_image", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            },
            "class_type": "WanFirstLastFrameToVideo"
        }
    else:
        wf["wan_i2v"] = {
            "inputs": {
                "positive": ["positive_cond", 0],
                "negative": ["negative_cond", 0],
                "vae": ["vae_loader", 0],
                "start_image": start_image_node,
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            },
            "class_type": "WanImageToVideo"
        }

    # 11. 2-Stage Distilled KSamplers (High Noise 0..2, Low Noise 2..4)
    wf.update({
        "sampler_stage1": {
            "inputs": {
                "model": ["sampling_high", 0],
                "positive": ["wan_i2v", 0],
                "negative": ["wan_i2v", 1],
                "latent_image": ["wan_i2v", 2],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable"
            },
            "class_type": "KSamplerAdvanced"
        },
        "sampler_stage2": {
            "inputs": {
                "model": ["sampling_low", 0],
                "positive": ["wan_i2v", 0],
                "negative": ["wan_i2v", 1],
                "latent_image": ["sampler_stage1", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable"
            },
            "class_type": "KSamplerAdvanced"
        },
        "vae_decode": {
            "inputs": {
                "samples": ["sampler_stage2", 0],
                "vae": ["vae_loader", 0]
            },
            "class_type": "VAEDecode"
        },
        "create_video": {
            "inputs": {
                "images": ["vae_decode", 0],
                "fps": fps
            },
            "class_type": "CreateVideo"
        },
        "save_video": {
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": prefix_str,
                "format": "mp4",
                "codec": "h264"
            },
            "class_type": "SaveVideo"
        }
    })

    log(f"Submitting Wan 2.2 segment prompt ({num_frames} frames @ {fps}fps, seed={seed})...")
    ok, prompt_id = comfy_client.queue_prompt(wf)
    if not ok:
        return False, prompt_id

    ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=900, progress_callback=progress_callback)
    if not ok:
        return False, str(history)

    outputs = history.get("outputs", {})
    save_video_out = outputs.get("save_video", {})
    videos = save_video_out.get("videos", []) or save_video_out.get("gifs", []) or save_video_out.get("images", [])

    found_local_file = None
    if videos:
        fn = videos[0]["filename"]
        sub = videos[0].get("subfolder", "")
        found_local_file = comfy_client.find_comfy_output_file(fn, sub)

    if not found_local_file:
        for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
            for f in files:
                if f.startswith(prefix_str) and f.endswith(".mp4"):
                    found_local_file = os.path.join(root, f)
                    break
            if found_local_file:
                break

    out_file = os.path.join(TEMP_DIR, out_video_name)
    if found_local_file and os.path.exists(found_local_file):
        shutil.copy2(found_local_file, out_file)
        return True, out_file

    if videos:
        fn = videos[0]["filename"]
        sub = videos[0].get("subfolder", "")
        raw_bytes = comfy_client.download_output_file(filename=fn, subfolder=sub)
        with open(out_file, "wb") as f:
            f.write(raw_bytes)
        return True, out_file

    return False, "Wan 2.2 execution completed but output mp4 file could not be located"


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
    enable_hud: bool = False,
    progress_callback: Optional[Any] = None
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

    total_scenes = len(scenes)
    total_blocks = sum(len(sc.get("blocks", [])) for sc in scenes)
    processed_blocks = 0

    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False
    comfy_client.reset_interrupt()

    _PIPELINE_STATUS["is_running"] = True
    _PIPELINE_STATUS["scenario_name"] = output_name or "video_scenario"
    _PIPELINE_STATUS["total_scenes"] = total_scenes
    _PIPELINE_STATUS["total_blocks"] = total_blocks
    _PIPELINE_STATUS["processed_blocks"] = 0
    _PIPELINE_STATUS["started_at"] = time.time()
    _PIPELINE_STATUS["phase"] = "starting"

    try:
        for s_idx, sc in enumerate(scenes):
            if is_interrupted():
                raise RuntimeError("Scenario execution was interrupted by user request.")

            scene_name = sc.get("scene_name", f"Scene #{s_idx + 1}")
            anchor_meta = sc.get("anchor_image", {})
            blocks = sc.get("blocks", [])

            if s_idx == 0 and global_start_image:
                anchor_meta["image_path"] = global_start_image

            log(f"\n==========================================")
            log(f"[{scene_name}] Preparing Scene Anchor Frame...")
            log(f"==========================================")

            _PIPELINE_STATUS["scene_index"] = s_idx + 1
            _PIPELINE_STATUS["scene_name"] = scene_name
            _PIPELINE_STATUS["phase"] = f"anchor_generation (Scene {s_idx + 1}/{total_scenes})"

            notify_progress(
                f"Scene {s_idx + 1}/{total_scenes} [{scene_name}]: Preparing anchor keyframe...",
                progress=processed_blocks,
                total=total_blocks,
                cb=progress_callback
            )

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
                if is_interrupted():
                    raise RuntimeError("Scenario execution was interrupted by user request.")

                b_name = blk.get("name", f"Block {b_idx + 1}")
                dur = float(blk.get("duration", 5.0))
                overlap = float(blk.get("overlap", 1.5 if b_idx < len(blocks) - 1 else 0.0))
                motion_prompt = blk.get("motion_prompt") or blk.get("prompt", "")

                # Calculate Wan frame count: (num_frames - 1) must be divisible by 4
                raw_frames = int(round(dur * fps))
                k = max(4, round((raw_frames - 1) / 4))
                num_frames = 4 * k + 1

                _PIPELINE_STATUS["block_index"] = b_idx + 1
                _PIPELINE_STATUS["block_name"] = b_name
                _PIPELINE_STATUS["phase"] = f"wan_diffusion ({scene_name} - {b_name})"

                notify_progress(
                    f"Scene {s_idx + 1}/{total_scenes} [{scene_name}] - {b_name} ({dur}s): Diffusing Wan 2.2...",
                    progress=processed_blocks,
                    total=total_blocks,
                    cb=progress_callback
                )

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
                        out_video_name=raw_segment_name,
                        progress_callback=progress_callback
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
                        out_video_name=raw_segment_name,
                        progress_callback=progress_callback
                    )

                if not ok:
                    raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {seg_raw_path}")

                processed_blocks += 1
                _PIPELINE_STATUS["processed_blocks"] = processed_blocks

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

            # Join blocks within the scene with deflickering & temporal color stabilization
            scene_final = os.path.join(TEMP_DIR, f"scene_{s_idx}_final.mp4")
            ffmpeg_engine.concat_videos(scene_block_pieces, scene_final, smooth_transitions=True)
            all_scene_videos.append(scene_final)

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

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
    finally:
        _PIPELINE_STATUS["is_running"] = False
        _PIPELINE_STATUS["phase"] = "idle"
