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
import base64
import random
import datetime
import uuid
from typing import List, Dict, Any, Optional, Tuple

from .config import OUTPUT_VID_DIR, OUTPUT_IMG_DIR, TEMP_DIR
from . import ffmpeg_engine
from . import comfy_client
from . import image_engine as img_engine_mod

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
# Latent Slicing & File Helpers
# ---------------------------------------------------------------------------

def slice_latent_file(
    in_path: str,
    out_path: str,
    tail_frames: int,
    start_idx: Optional[int] = None
) -> bool:
    """
    Slices `tail_frames` along the temporal dimension of a .latent (safetensors) file.
    If start_idx is provided (e.g. -k), slices from start_idx to start_idx + tail_frames.
    Otherwise slices the last tail_frames.
    KSampler outputs pure raw VAE latents directly; no re-scaling or denormalization is applied.
    """
    try:
        import safetensors.torch
        import torch
        data = safetensors.torch.load_file(in_path)
        tensor = data.get("latent_tensor")
        if tensor is None:
            for v in data.values():
                if hasattr(v, "ndim") and v.ndim == 5:
                    tensor = v
                    break
        if tensor is None:
            log(f"Warning: No 5D latent tensor found in {in_path}. Keys: {list(data.keys())}")
            return False

        t_dim = 2
        total_t = tensor.shape[t_dim]
        if start_idx is not None:
            s_idx = start_idx if start_idx >= 0 else max(0, total_t + start_idx)
            e_idx = min(total_t, s_idx + max(1, tail_frames))
            sliced = tensor[:, :, s_idx:e_idx, :, :].contiguous()
            slice_t = sliced.shape[t_dim]
        else:
            slice_t = min(total_t, max(1, tail_frames))
            sliced = tensor[:, :, -slice_t:, :, :].contiguous()

        save_dict = {
            "latent_tensor": sliced,
            "latent_format_version_0": torch.tensor([])
        }
        safetensors.torch.save_file(save_dict, out_path)
        log(f"Sliced latent: {slice_t} frames saved to {out_path} (from total {total_t} frames)")
        return True
    except Exception as ex:
        log(f"slice_latent_file error: {ex}")
        return False

# ---------------------------------------------------------------------------
# Wan 2.2 ComfyUI Graph Builder
# ---------------------------------------------------------------------------

def generate_wan_segment(
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    prefix_latent_path: Optional[str] = None,
    prefix_frames: int = 0,
    end_image_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "blurry, jitter, static, low quality, deformed, extra limbs, bad anatomy, text, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    core_frames: Optional[int] = None,
    save_tail: bool = True,
    save_tail_latent: Optional[bool] = None,
    fps: int = 16,
    seed: Optional[int] = None,
    out_video_name: Optional[str] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Submits a Wan 2.2 Video segment to ComfyUI.
    Supports single image seed, end image keyframe, or true forced prefix latent conditioning.
    Outputs core video directly and exports tail latent for seamless chaining.
    Returns: (success, out_video_path, out_tail_latent_path)
    """
    if save_tail_latent is not None:
        save_tail = save_tail_latent

    width = (width // 16) * 16
    height = (height // 16) * 16
    out_video_name = out_video_name or f"wan_{int(time.time()*1000)}.mp4"
    if seed is None or seed < 0:
        seed = random.randint(1, 1_000_000_000)

    clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
    if clean_base.endswith("_core"):
        clean_base = clean_base[:-5]
    prefix_str = clean_base

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

    # 9. Conditioning start: Prefix Video (Forced Prefix) > Start Image Keyframe
    if prefix_video_path and os.path.exists(prefix_video_path):
        prefix_fn = comfy_client.upload_file_to_comfy(
            prefix_video_path,
            custom_filename=f"wan_prefix_{int(time.time()*1000)}.mp4"
        )
        abs_prefix_in_comfy = os.path.join(comfy_client.COMFY_INPUT_DIR, prefix_fn)
        log(f"Configuring TRUE MULTI-FRAME FORCED PREFIX from video: {abs_prefix_in_comfy}")
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
        return False, f"No valid start input provided: img={image_path}, vid={prefix_video_path}", None

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
    cond_positive = ["wan_i2v", 0]
    cond_negative = ["wan_i2v", 1]
    latent_input = ["wan_i2v", 2]

    if prefix_latent_path and os.path.exists(prefix_latent_path):
        prefix_fn = comfy_client.upload_file_to_comfy(
            prefix_latent_path,
            custom_filename=f"wan_pref_{int(time.time()*1000)}.latent"
        )
        log(f"Configuring TRUE FORCED PREFIX LATENT: {prefix_fn}")
        wf["load_prefix_latent"] = {
            "inputs": {"latent": prefix_fn},
            "class_type": "LoadLatent"
        }
        wf["replace_latent"] = {
            "inputs": {
                "destination": ["wan_i2v", 2],
                "source": ["load_prefix_latent", 0],
                "index": 0
            },
            "class_type": "ReplaceVideoLatentFrames"
        }
        latent_input = ["replace_latent", 0]

    # 10. 2-Stage Distilled KSamplers (High Noise 0..2, Low Noise 2..4) + SaveVideo
    wf.update({
        "sampler_stage1": {
            "inputs": {
                "model": ["sampling_high", 0],
                "positive": cond_positive,
                "negative": cond_negative,
                "latent_image": latent_input,
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
                "positive": cond_positive,
                "negative": cond_negative,
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
        }
    })

    slice_batch_idx = 1 if prefix_frames > 0 else 0
    if core_frames is not None and core_frames < num_frames:
        wf["slice_core"] = {
            "inputs": {
                "image": ["vae_decode", 0],
                "batch_index": int(slice_batch_idx),
                "length": int(core_frames)
            },
            "class_type": "ImageFromBatch"
        }
        video_images_input = ["slice_core", 0]
        log(f"Configured ComfyUI ImageFromBatch core: start={slice_batch_idx}, length={core_frames}")
    else:
        video_images_input = ["vae_decode", 0]

    wf.update({
        "create_video": {
            "inputs": {
                "images": video_images_input,
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

    if save_tail:
        end_idx = int(core_frames - 1) if slice_batch_idx == 0 else int(core_frames)
        wf["slice_end_frame"] = {
            "inputs": {
                "image": ["vae_decode", 0],
                "batch_index": end_idx,
                "length": 1
            },
            "class_type": "ImageFromBatch"
        }
        wf["save_end_frame"] = {
            "inputs": {
                "images": ["slice_end_frame", 0],
                "filename_prefix": f"{prefix_str}_end"
            },
            "class_type": "SaveImage"
        }
        log(f"Configured ComfyUI to export end keyframe (frame {end_idx}) for next block")

    log(f"Submitting Wan 2.2 segment prompt ({num_frames} frames @ {fps}fps, seed={seed})...")
    ok, prompt_id = comfy_client.queue_prompt(wf)
    if not ok:
        return False, prompt_id, None, None, None

    ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=900, progress_callback=progress_callback)
    if not ok:
        return False, str(history), None, None, None

    outputs = history.get("outputs", {})

    # Locate generated core video
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
                if f.startswith(prefix_str) and not f.startswith(f"{prefix_str}_tail") and f.endswith(".mp4"):
                    found_local_file = os.path.join(root, f)
                    break
            if found_local_file:
                break

    out_file = os.path.join(TEMP_DIR, out_video_name)
    if found_local_file and os.path.exists(found_local_file):
        shutil.copy2(found_local_file, out_file)
    elif videos:
        fn = videos[0]["filename"]
        sub = videos[0].get("subfolder", "")
        raw_bytes = comfy_client.download_output_file(filename=fn, subfolder=sub)
        with open(out_file, "wb") as f:
            f.write(raw_bytes)
    else:
        return False, "Wan 2.2 execution completed but output mp4 file could not be located", None

    # Locate generated end keyframe
    out_end_frame_path = None
    if save_tail:
        found_end = None
        for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
            for f in files:
                if f.startswith(f"{prefix_str}_end") and (f.endswith(".png") or f.endswith(".jpg")):
                    found_end = os.path.join(root, f)
                    break
            if found_end:
                break
        if found_end and os.path.exists(found_end):
            out_end_frame_path = os.path.join(TEMP_DIR, f"{prefix_str}_end.png")
            shutil.copy2(found_end, out_end_frame_path)
            log(f"Successfully captured end keyframe for next block: {out_end_frame_path}")

    return True, out_file, out_end_frame_path


# ---------------------------------------------------------------------------
# LTX-2.3 (22B) ComfyUI Graph Builder
# ---------------------------------------------------------------------------

def generate_ltx_segment(
    image_path: Optional[str] = None,
    prefix_video_path: Optional[str] = None,
    prefix_latent_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    core_frames: Optional[int] = None,
    save_tail: bool = True,
    save_tail_latent: Optional[bool] = None,
    fps: int = 24,
    seed: Optional[int] = None,
    cfg: float = 3.0,
    steps: int = 20,
    out_video_name: Optional[str] = None,
    progress_callback: Optional[Any] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Submits an LTX-2.3 (22B) Video segment to ComfyUI with CFG 3.0 and distilled LoRA 1.1.
    Supports single image seed or multi-frame forced prefix video conditioning.
    Outputs core video directly and optionally exports clean tail video for next block chaining.
    Returns: (success, out_video_path, out_tail_path)
    """
    if save_tail_latent is not None:
        save_tail = save_tail_latent

    width = (width // 32) * 32
    height = (height // 32) * 32
    out_video_name = out_video_name or f"ltx_{int(time.time()*1000)}.mp4"
    if seed is None or seed < 0:
        seed = random.randint(1, 1_000_000_000)

    prefix_str = f"ltx23_{int(time.time()*1000)}"

    wf: Dict[str, Any] = {
        # 1. UNET Loader (LTX-2.3 22B Dev FP8)
        "unet_loader": {
            "inputs": {
                "unet_name": "ltx-2.3-22b-dev-fp8.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        # 2. Distilled LoRA 1.1 Loader
        "lora_loader": {
            "inputs": {
                "model": ["unet_loader", 0],
                "lora_name": "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        },
        # 3. Dual CLIP Loader
        "clip_loader": {
            "inputs": {
                "clip_name1": "ltx-2-3-22b-text_encoder.safetensors",
                "clip_name2": "ltx-2-3-22b-text_encoder.safetensors",
                "type": "ltxv"
            },
            "class_type": "DualCLIPLoader"
        },
        # 4. VAE Loader
        "vae_loader": {
            "inputs": {
                "vae_name": "ltx-2-3-22b-VAE.safetensors"
            },
            "class_type": "VAELoader"
        },
        # 5. Prompts
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
        },
        # 6. LTXV Conditioning
        "ltxv_cond": {
            "inputs": {
                "positive": ["positive_cond", 0],
                "negative": ["negative_cond", 0],
                "frame_rate": float(fps)
            },
            "class_type": "LTXVConditioning"
        },
        # 7. Empty Video Latent
        "empty_latent": {
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            },
            "class_type": "EmptyLTXVLatentVideo"
        }
    }

    # 8. Start Conditioning (Prefix Video, Prefix Latent, or Image)
    if prefix_video_path and os.path.exists(prefix_video_path):
        prefix_fn = comfy_client.upload_file_to_comfy(
            prefix_video_path,
            custom_filename=f"ltx_prefix_{int(time.time()*1000)}.mp4"
        )
        abs_prefix_in_comfy = os.path.join(comfy_client.COMFY_INPUT_DIR, prefix_fn)
        log(f"LTX multi-frame prefix from video: {abs_prefix_in_comfy}")
        wf["prefix_video_loader"] = {
            "inputs": {
                "video": os.path.abspath(abs_prefix_in_comfy),
                "force_rate": float(fps),
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "LTXV"
            },
            "class_type": "VHS_LoadVideoPath"
        }
        start_image_node = ["prefix_video_loader", 0]
    elif prefix_latent_path and os.path.exists(prefix_latent_path):
        prefix_fn = comfy_client.upload_file_to_comfy(
            prefix_latent_path,
            custom_filename=f"ltx_prefix_{int(time.time()*1000)}.latent"
        )
        log(f"LTX Latent-Chaining from: {prefix_fn}")
        wf["prefix_latent_loader"] = {
            "inputs": {"latent": prefix_fn},
            "class_type": "LoadLatent"
        }
        wf["prefix_latent_decode"] = {
            "inputs": {
                "samples": ["prefix_latent_loader", 0],
                "vae": ["vae_loader", 0]
            },
            "class_type": "VAEDecode"
        }
        start_image_node = ["prefix_latent_decode", 0]
    elif image_path and os.path.exists(image_path):
        img_fn = comfy_client.upload_file_to_comfy(image_path)
        log(f"LTX start image: {img_fn}")
        wf["start_image"] = {
            "inputs": {
                "image": img_fn,
                "upload": "image"
            },
            "class_type": "LoadImage"
        }
        start_image_node = ["start_image", 0]
    else:
        start_image_node = None

    if start_image_node:
        wf["inplace_cond"] = {
            "inputs": {
                "image": start_image_node,
                "vae": ["vae_loader", 0],
                "latent": ["empty_latent", 0]
            },
            "class_type": "LTXVImgToVideoInplace"
        }
        active_latent_node = ["inplace_cond", 0]
    else:
        active_latent_node = ["empty_latent", 0]

    # 9. CFGGuider + LTXVScheduler + SamplerCustomAdvanced
    wf.update({
        "cfg_guider": {
            "inputs": {
                "model": ["lora_loader", 0],
                "positive": ["ltxv_cond", 0],
                "negative": ["ltxv_cond", 1],
                "cfg": float(cfg)
            },
            "class_type": "CFGGuider"
        },
        "scheduler": {
            "inputs": {
                "steps": int(steps),
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": active_latent_node
            },
            "class_type": "LTXVScheduler"
        },
        "sampler_select": {
            "inputs": {
                "sampler_name": "euler"
            },
            "class_type": "KSamplerSelect"
        },
        "random_noise": {
            "inputs": {
                "noise_seed": seed
            },
            "class_type": "RandomNoise"
        },
        "sampler": {
            "inputs": {
                "noise": ["random_noise", 0],
                "guider": ["cfg_guider", 0],
                "sampler": ["sampler_select", 0],
                "sigmas": ["scheduler", 0],
                "latent_image": active_latent_node
            },
            "class_type": "SamplerCustomAdvanced"
        },
        "vae_decode": {
            "inputs": {
                "samples": ["sampler", 0],
                "vae": ["vae_loader", 0]
            },
            "class_type": "VAEDecode"
        }
    })

    if core_frames is not None and core_frames < num_frames:
        wf["slice_core"] = {
            "inputs": {
                "image": ["vae_decode", 0],
                "batch_index": 0,
                "length": int(core_frames)
            },
            "class_type": "ImageFromBatch"
        }
        video_images_input = ["slice_core", 0]
        log(f"Configured ComfyUI ImageFromBatch to output exactly {core_frames} core frames directly into MP4 (LTX)")
    else:
        video_images_input = ["vae_decode", 0]

    wf.update({
        "create_video": {
            "inputs": {
                "images": video_images_input,
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

    if save_tail and core_frames is not None and core_frames < num_frames:
        tail_len = int(num_frames - core_frames)
        wf["slice_tail"] = {
            "inputs": {
                "image": ["vae_decode", 0],
                "batch_index": int(core_frames),
                "length": tail_len
            },
            "class_type": "ImageFromBatch"
        }
        wf["create_tail_video"] = {
            "inputs": {
                "images": ["slice_tail", 0],
                "fps": fps
            },
            "class_type": "CreateVideo"
        }
        wf["save_tail_video"] = {
            "inputs": {
                "video": ["create_tail_video", 0],
                "filename_prefix": f"{prefix_str}_tail",
                "format": "mp4",
                "codec": "h264"
            },
            "class_type": "SaveVideo"
        }
        log(f"Configured ComfyUI to export tail video ({tail_len} frames, start={core_frames}) directly for multi-frame forced prefix chaining (LTX)")

    log(f"Submitting LTX-2.3 (22B) prompt ({num_frames} frames @ {fps}fps, CFG={cfg}, seed={seed})...")
    ok, prompt_id = comfy_client.queue_prompt(wf)
    if not ok:
        return False, prompt_id, None

    ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=900, progress_callback=progress_callback)
    if not ok:
        return False, str(history), None

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
                if f.startswith(prefix_str) and not f.startswith(f"{prefix_str}_tail") and f.endswith(".mp4"):
                    found_local_file = os.path.join(root, f)
                    break
            if found_local_file:
                break

    out_file = os.path.join(TEMP_DIR, out_video_name)
    if found_local_file and os.path.exists(found_local_file):
        shutil.copy2(found_local_file, out_file)
    elif videos:
        fn = videos[0]["filename"]
        sub = videos[0].get("subfolder", "")
        raw_bytes = comfy_client.download_output_file(filename=fn, subfolder=sub)
        with open(out_file, "wb") as f:
            f.write(raw_bytes)
    else:
        return False, "LTX-2.3 execution completed but output mp4 file could not be located", None

    # Locate generated tail video (if requested)
    out_tail_path = None
    if save_tail and core_frames is not None and core_frames < num_frames:
        save_tail_out = outputs.get("save_tail_video", {})
        tail_videos = save_tail_out.get("videos", []) or save_tail_out.get("gifs", []) or save_tail_out.get("images", [])
        found_tail_file = None
        if tail_videos:
            fn = tail_videos[0]["filename"]
            sub = tail_videos[0].get("subfolder", "")
            found_tail_file = comfy_client.find_comfy_output_file(fn, sub)

        if not found_tail_file:
            for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
                for f in files:
                    if f.startswith(f"{prefix_str}_tail") and f.endswith(".mp4"):
                        found_tail_file = os.path.join(root, f)
                        break
                if found_tail_file:
                    break

        if found_tail_file and os.path.exists(found_tail_file):
            out_tail_path = os.path.join(TEMP_DIR, f"{prefix_str}_tail.mp4")
            shutil.copy2(found_tail_file, out_tail_path)
            log(f"Successfully captured tail video for forced prefix (LTX): {out_tail_path}")

    return True, out_file, out_tail_path



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Deep Merge & Prompt Inheritance Helpers
# ---------------------------------------------------------------------------

def deep_merge(base: Any, overlay: Any) -> Any:
    """Recursively deep merges overlay dict onto base dict."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay if overlay is not None else base
    merged = dict(base)
    for k, v in overlay.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        elif v is not None:
            merged[k] = v
    return merged

def combine_prompts(base: Optional[str], local: Optional[str]) -> str:
    """Combines base prompt with local prompt using newline formatting."""
    b = str(base).strip() if base else ""
    l = str(local).strip() if local else ""
    if b and l:
        return f"{b}\n {l}"
    return b or l

def combine_negative_prompts(base: Optional[str], local: Optional[str]) -> str:
    """Combines base and local negative prompts, deduplicating tags if comma-separated."""
    b = str(base).strip() if base else ""
    l = str(local).strip() if local else ""
    if b and l:
        if "," in b or "," in l:
            tags = [t.strip() for t in (b + ", " + l).split(",") if t.strip()]
            seen = set()
            deduped = []
            for t in tags:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    deduped.append(t)
            return ", ".join(deduped)
        return f"{b}\n {l}"
    return b or l


# ---------------------------------------------------------------------------
# Hierarchical Scene Normalizer (Backward Compatibility & Structured Config)
# ---------------------------------------------------------------------------

def normalize_scenario_structure(raw_data: Any, request_overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Normalizes scenario input into a clean list of Scene objects:
    [
      {
        "scene_name": str,
        "engine": "wan" | "ltx",
        "negative_prompt": str,
        "anchor_image": { "image_path": Optional[str], "prompt": Optional[str], ... },
        "blocks": [ { "name": str, "duration": float, "overlap": float, "motion_prompt": str, "negative_prompt": str, "engine": str, ... } ]
      }
    ]
    Supports top-level 'image' and 'video' sections, prompt inheritance, and deep merge.
    """
    if isinstance(raw_data, list):
        scenario_dict = {"scenes": raw_data}
    elif isinstance(raw_data, dict):
        scenario_dict = dict(raw_data)
    else:
        raise ValueError("Invalid scenario structure")

    if request_overrides:
        scenario_dict = deep_merge(scenario_dict, request_overrides)

    image_cfg = scenario_dict.get("image", {})
    video_cfg = scenario_dict.get("video", {})

    root_video_engine = video_cfg.get("engine") or scenario_dict.get("engine", "wan")
    root_video_model = video_cfg.get("model") or scenario_dict.get("model")
    root_video_negative = video_cfg.get("negative_prompt") or scenario_dict.get("default_negative_prompt") or scenario_dict.get("negative_prompt", "")
    root_video_prompt = video_cfg.get("prompt", "")

    root_image_engine = image_cfg.get("engine") or scenario_dict.get("image_engine") or "sdxl"
    root_image_model = image_cfg.get("model") or scenario_dict.get("image_model")
    root_image_negative = image_cfg.get("negative_prompt", "")
    root_image_prompt = image_cfg.get("prompt", "")

    raw_scenes = scenario_dict.get("scenes", [])
    if not raw_scenes and "blocks" in scenario_dict:
        raw_scenes = [scenario_dict]

    normalized_scenes = []
    for s_idx, sc in enumerate(raw_scenes):
        sc_engine = sc.get("engine") or root_video_engine
        sc_negative = combine_negative_prompts(root_video_negative, sc.get("negative_prompt"))
        sc_model = sc.get("model") or root_video_model
        sc_name = sc.get("scene_name", f"Scene #{s_idx + 1}")

        anchor = sc.get("anchor_image") or sc.get("anchor", {})
        anchor["image_engine"] = anchor.get("image_engine") or root_image_engine
        if root_image_model and "model" not in anchor:
            anchor["model"] = root_image_model
        for k in ("cfg", "cfg_scale", "steps", "sampler", "sampler_name"):
            if k in image_cfg and k not in anchor:
                anchor[k] = image_cfg[k]

        # Combine anchor prompts: base prompt + "\n " + local prompt
        anchor["prompt"] = combine_prompts(root_image_prompt, anchor.get("prompt"))
        anchor["negative_prompt"] = combine_negative_prompts(root_image_negative, anchor.get("negative_prompt"))

        raw_blocks = sc.get("blocks", [])
        norm_blocks = []
        for b_idx, blk in enumerate(raw_blocks):
            b_engine = blk.get("engine") or sc_engine
            # Combine motion prompts: base video prompt + "\n " + local block prompt
            b_prompt = combine_prompts(root_video_prompt, blk.get("motion_prompt") or blk.get("prompt", ""))
            b_neg = combine_negative_prompts(sc_negative, blk.get("negative_prompt"))

            norm_blocks.append({
                "name": blk.get("name", f"Block #{b_idx + 1}"),
                "duration": float(blk.get("duration", 5.0)),
                "overlap": float(blk.get("overlap", 1.5 if b_idx < len(raw_blocks) - 1 else 0.0)),
                "motion_prompt": b_prompt,
                "negative_prompt": b_neg,
                "engine": b_engine,
                "use_next_as_end": blk.get("use_next_as_end", False)
            })

        normalized_scenes.append({
            "scene_name": sc_name,
            "engine": sc_engine,
            "negative_prompt": sc_negative,
            "anchor_image": anchor,
            "blocks": norm_blocks
        })

    return normalized_scenes


# ---------------------------------------------------------------------------
# Main Pipeline Executor
# ---------------------------------------------------------------------------

def execute_scenario_pipeline(
    scenario_input: Any,
    global_start_image: Optional[str] = None,
    output_name: Optional[str] = None,
    default_negative_prompt: Optional[str] = None,
    video_engine: Optional[str] = None,
    image_engine: Optional[str] = None,
    image_model: Optional[str] = None,
    default_engine: Optional[str] = None,
    default_model: Optional[str] = None,
    width: int = 832,
    height: int = 480,
    fps: int = 16,
    enable_hud: bool = False,
    progress_callback: Optional[Any] = None,
    image: Optional[Dict[str, Any]] = None,
    video: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a multi-scene video scenario using Latent-Chaining and clean core saving.
    Supports top-level 'image' and 'video' structured parameters and Deep Merge.
    Returns structured results including video paths, duration, and poster base64.
    """
    req_ov = dict(request_overrides or {})
    if image:
        req_ov["image"] = deep_merge(req_ov.get("image", {}), image)
    if video:
        req_ov["video"] = deep_merge(req_ov.get("video", {}), video)

    # Flat arguments support (backward compatibility)
    if image_engine or image_model:
        img_ov = req_ov.get("image", {})
        if image_engine and image_engine != "auto":
            img_ov["engine"] = image_engine
        if image_model:
            img_ov["model"] = image_model
        req_ov["image"] = img_ov

    if video_engine or default_engine or default_negative_prompt or width != 832 or height != 480 or fps != 16:
        vid_ov = req_ov.get("video", {})
        v_eng = video_engine or default_engine
        if v_eng:
            vid_ov["engine"] = v_eng
        if default_negative_prompt:
            vid_ov["negative_prompt"] = default_negative_prompt
        if width != 832:
            vid_ov["width"] = width
        if height != 480:
            vid_ov["height"] = height
        if fps != 16:
            vid_ov["fps"] = fps
        req_ov["video"] = vid_ov

    scenes = normalize_scenario_structure(scenario_input, request_overrides=req_ov)

    effective_vid = req_ov.get("video", {})
    v_engine = effective_vid.get("engine", "wan").lower()
    width = int(effective_vid.get("width", width))
    height = int(effective_vid.get("height", height))
    fps = int(effective_vid.get("fps", fps))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_hash = uuid.uuid4().hex[:6]
    run_prefix = f"{timestamp}_{run_hash}"
    log(f"Executing scenario pipeline with {len(scenes)} scenes (run_id={run_prefix}, video_engine={v_engine}, {width}x{height} @ {fps}fps)...")

    all_block_cores = []
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
    _PIPELINE_STATUS["scenario_name"] = output_name or f"{run_prefix}_final"
    _PIPELINE_STATUS["total_scenes"] = total_scenes
    _PIPELINE_STATUS["total_blocks"] = total_blocks
    _PIPELINE_STATUS["processed_blocks"] = 0
    _PIPELINE_STATUS["started_at"] = time.time()
    _PIPELINE_STATUS["phase"] = "starting"

    try:
        for s_idx, sc in enumerate(scenes):
            if is_interrupted():
                raise RuntimeError("Scenario execution was interrupted by user request.")

            scene_idx = s_idx + 1
            scene_name = sc.get("scene_name", f"Scene #{scene_idx}")
            anchor_meta = sc.get("anchor_image", {})
            blocks = sc.get("blocks", [])

            if s_idx == 0 and global_start_image:
                anchor_meta["image_path"] = global_start_image

            log(f"\n==========================================")
            log(f"[{scene_name}] Preparing Scene Anchor Frame...")
            log(f"==========================================")

            _PIPELINE_STATUS["scene_index"] = scene_idx
            _PIPELINE_STATUS["scene_name"] = scene_name
            _PIPELINE_STATUS["phase"] = f"anchor_generation (Scene {scene_idx}/{total_scenes})"

            notify_progress(
                f"Scene {scene_idx}/{total_scenes} [{scene_name}]: Preparing anchor keyframe...",
                progress=processed_blocks,
                total=total_blocks,
                cb=progress_callback
            )

            anchor_filename = f"{run_prefix}_s{scene_idx:03d}_anchor.png"
            ok, anchor_img_path, anchor_b64 = img_engine_mod.generate_anchor_frame(
                anchor_meta=anchor_meta,
                width=width,
                height=height,
                custom_filename=anchor_filename
            )
            if not ok:
                raise RuntimeError(f"Failed to obtain anchor keyframe for {scene_name}: {anchor_img_path}")

            if s_idx == 0 and anchor_b64:
                first_poster_b64 = anchor_b64
                first_poster_path = anchor_img_path

            prefix_latent_path = None
            prefix_video_path = None
            next_end_frame_path = None

            for b_idx, blk in enumerate(blocks):
                if is_interrupted():
                    raise RuntimeError("Scenario execution was interrupted by user request.")

                block_idx = b_idx + 1
                b_name = blk.get("name", f"Block {block_idx}")
                dur = float(blk.get("duration", 5.0))
                overlap = float(blk.get("overlap", 1.5 if b_idx < len(blocks) - 1 else 0.0))
                # Enforce non-colliding overlap constraint:
                # prefix_length + tail_length < clip_duration
                prefix_length = float(blocks[b_idx - 1].get("overlap", 1.5)) if b_idx > 0 else 0.0
                tail_length = overlap if b_idx < len(blocks) - 1 else 0.0
                if prefix_length + tail_length >= dur:
                    raise ValueError(
                        f"[{scene_name} : {b_name}] Overlap constraint violated: prefix_length ({prefix_length}s) + tail_length ({tail_length}s) = "
                        f"{prefix_length + tail_length:.2f}s >= clip_duration ({dur}s). Condition 'prefix_length + tail_length < clip_duration' is required so transition zones do not collide."
                    )
                motion_prompt = blk.get("motion_prompt") or blk.get("prompt", "")
                b_neg = blk.get("negative_prompt") or default_negative_prompt or ""
                b_engine = (blk.get("engine") or v_engine).lower()

                is_last = (b_idx == len(blocks) - 1)
                stride = 8 if b_engine == "ltx" else 4

                # Dynamic time quantization strictly matching model VAE stride:
                # 1. Total block frames quantized to stride * k_total + 1
                raw_frames = int(round(dur * fps))
                k_total = max(1, round((raw_frames - 1) / stride))
                num_frames = stride * k_total + 1

                # 2. Overlap quantization & core frames:
                if not is_last and overlap > 0:
                    k_overlap = max(1, round(round(overlap * fps) / stride))
                    overlap_frames = k_overlap * stride
                    # Core frames for video: exactly (num_frames - 1) - overlap_frames
                    # so block core ends at frame N-1, and next block starts at frame N (no duplicate boundary frame!)
                    core_frames = max(stride, (num_frames - 1) - overlap_frames)
                    tail_frames = num_frames - core_frames
                    save_tail = True
                else:
                    core_frames = num_frames
                    tail_frames = 0
                    save_tail = False

                _PIPELINE_STATUS["block_index"] = block_idx
                _PIPELINE_STATUS["block_name"] = b_name
                _PIPELINE_STATUS["phase"] = f"{b_engine}_diffusion ({scene_name} - {b_name})"

                notify_progress(
                    f"Scene {scene_idx}/{total_scenes} [{scene_name}] - {b_name} ({dur}s): Diffusing {b_engine.upper()}...",
                    progress=processed_blocks,
                    total=total_blocks,
                    cb=progress_callback
                )

                log(f"\n[{scene_name} : {b_name}] Engine: {b_engine.upper()}, Duration: {dur}s, Overlap: {overlap}s")
                log(f"Quantized frames: total={num_frames}, core={core_frames}, tail_frames={tail_frames}")
                log(f"Motion prompt: '{motion_prompt[:60]}...'")

                block_prefix = f"{run_prefix}_s{scene_idx:03d}_b{block_idx:03d}"
                block_core = os.path.join(TEMP_DIR, f"{block_prefix}_core.mp4")

                if b_engine == "ltx":
                    if prefix_video_path and os.path.exists(prefix_video_path):
                        start_img = None
                        pref_vid = prefix_video_path
                    else:
                        start_img = anchor_img_path
                        pref_vid = None
                    ok, seg_core_path, seg_tail_path = generate_ltx_segment(
                        image_path=start_img,
                        prefix_video_path=pref_vid,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        core_frames=core_frames,
                        save_tail=save_tail,
                        fps=fps,
                        out_video_name=f"{block_prefix}_core.mp4",
                        progress_callback=progress_callback
                    )
                    if save_tail and seg_tail_path and os.path.exists(seg_tail_path):
                        prefix_video_path = seg_tail_path
                    else:
                        prefix_video_path = None
                    block_core_frames = core_frames
                else:
                    if b_idx == 0:
                        wan_num_frames = 57
                        wan_core_frames = 56
                        wan_pref_frames = 0
                        wan_start_img = anchor_img_path
                        wan_save_tail = (len(blocks) > 1)
                    else:
                        wan_num_frames = 57
                        wan_core_frames = 56
                        wan_pref_frames = 1
                        wan_start_img = next_end_frame_path or anchor_img_path
                        wan_save_tail = (not is_last)

                    res = generate_wan_segment(
                        image_path=wan_start_img,
                        prefix_frames=wan_pref_frames,
                        prompt=motion_prompt,
                        negative_prompt=b_neg,
                        width=width,
                        height=height,
                        num_frames=wan_num_frames,
                        core_frames=wan_core_frames,
                        save_tail=wan_save_tail,
                        fps=fps,
                        out_video_name=f"{block_prefix}_core.mp4",
                        progress_callback=progress_callback
                    )
                    ok = res[0]
                    seg_core_path = res[1]
                    seg_end_frame = res[2] if len(res) > 2 else None
                    if seg_end_frame:
                        next_end_frame_path = seg_end_frame
                        log(f"Chained end keyframe for next block: {next_end_frame_path}")
                    block_core_frames = wan_core_frames

                if not ok:
                    raise RuntimeError(f"Block generation failed [{scene_name} : {b_name}]: {seg_core_path}")

                processed_blocks += 1
                _PIPELINE_STATUS["processed_blocks"] = processed_blocks

                # Direct core output from ComfyUI: zero FFmpeg post-cutting!
                all_block_cores.append(block_core)
                actual_core_dur = block_core_frames / fps
                generated_block_metadata.append((block_core, actual_core_dur, overlap if not is_last else 0.0))

        # Clean up any leftover temporary tail videos
        if prefix_video_path and os.path.exists(prefix_video_path):
            try:
                os.remove(prefix_video_path)
            except Exception:
                pass

        _PIPELINE_STATUS["phase"] = "ffmpeg_final_assembly"
        notify_progress(
            f"All {total_scenes} scenes completed ({total_blocks} blocks). Assembling final cut with FFmpeg (-c copy)...",
            progress=total_blocks,
            total=total_blocks,
            cb=progress_callback
        )

        # Final movie assembly across all blocks in a single lossless Stream Copy (-c copy) pass
        if output_name:
            clean_out = os.path.splitext(os.path.basename(output_name))[0]
            final_name = f"{run_prefix}_{clean_out}.mp4"
        else:
            final_name = f"{run_prefix}_final.mp4"

        final_path = os.path.join(OUTPUT_VID_DIR, final_name)
        ffmpeg_engine.concat_videos(all_block_cores, final_path, smooth_transitions=False)
        total_duration = ffmpeg_engine.get_duration(final_path)

        # Poster handling: prioritize pristine anchor keyframe from Scene 1
        poster_filename = f"{os.path.splitext(final_name)[0]}_poster.png"
        poster_path = os.path.join(OUTPUT_IMG_DIR, poster_filename)
        if first_poster_path and os.path.exists(first_poster_path):
            try:
                shutil.copy2(first_poster_path, poster_path)
            except Exception as e:
                log(f"Warning: Failed to copy anchor image to poster: {e}")
            poster_b64 = first_poster_b64
            if not poster_b64:
                try:
                    with open(poster_path, "rb") as f:
                        poster_b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    pass
        else:
            # Fallback only when scenario has no anchor image
            poster_path, poster_b64 = ffmpeg_engine.extract_poster(final_path, poster_path)

        # Optional HUD generation
        hud_path = None
        if enable_hud and generated_block_metadata:
            hud_name = f"{os.path.splitext(final_name)[0]}_hud.mp4"
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

