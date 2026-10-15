#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Generators for media_mcp.
Polymorphic strategy implementations for Wan 2.2 Fast (4-step LoRA), Wan 2.2 Full (20-step UNet),
and LTX-2.3 22B.
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

from ..config import TEMP_DIR
from .. import comfy_client
from .base import BaseGenerator, VideoRequest, GenerationResult, snap_to_16, snap_to_32


def log(msg: str):
    sys.stderr.write(f"[video-generators] {msg}\n")
    sys.stderr.flush()


class BaseVideoGenerator(BaseGenerator):
    """Abstract base class for video generators."""

    kind = "video"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "settings": dict(self.settings)
        }

    def _execute_comfy_workflow(
        self,
        wf: Dict[str, Any],
        prefix_str: str,
        out_video_name: str,
        save_tail_latent: bool = False,
        save_tail_video: bool = False,
        progress_callback: Optional[Any] = None
    ) -> GenerationResult:
        """
        Submits workflow to ComfyUI, waits for completion, and extracts
        the core video, optional tail latent, and optional tail video.
        """
        log(f"Submitting {self.name} workflow (prefix={prefix_str})...")
        ok, prompt_id = comfy_client.queue_prompt(wf)
        if not ok:
            return GenerationResult(success=False, error=prompt_id)

        ok, history = comfy_client.wait_for_prompt(prompt_id, timeout=900, progress_callback=progress_callback)
        if not ok:
            return GenerationResult(success=False, error=str(history))

        outputs = history.get("outputs", {})

        # 1. Locate core video
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
            return GenerationResult(success=False, error=f"{self.name} completed but output mp4 file could not be located")

        # 2. Locate tail latent (if requested)
        out_tail_latent_path: Optional[str] = None
        if save_tail_latent:
            save_tail_out = outputs.get("save_tail_latent", {})
            saved_latents = save_tail_out.get("latents", [])
            if saved_latents:
                fn = saved_latents[0].get("filename")
                sub = saved_latents[0].get("subfolder", "")
                found = comfy_client.find_comfy_output_file(fn, sub)
                if found and os.path.exists(found):
                    out_tail_latent_path = os.path.join(TEMP_DIR, f"{prefix_str}_tail.latent")
                    shutil.copy2(found, out_tail_latent_path)
            if not out_tail_latent_path:
                for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
                    for f in files:
                        if f.startswith(f"{prefix_str}_tail") and f.endswith(".latent"):
                            cand = os.path.join(root, f)
                            out_tail_latent_path = os.path.join(TEMP_DIR, f"{prefix_str}_tail.latent")
                            shutil.copy2(cand, out_tail_latent_path)
                            break
                    if out_tail_latent_path:
                        break
            if out_tail_latent_path:
                log(f"Successfully captured tail latent: {os.path.basename(out_tail_latent_path)}")
            else:
                log(f"Warning: Tail latent output file not found for {prefix_str}")

        # 3. Locate tail video (if requested)
        out_tail_video_path: Optional[str] = None
        if save_tail_video:
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
                out_tail_video_path = os.path.join(TEMP_DIR, f"{prefix_str}_tail.mp4")
                shutil.copy2(found_tail_file, out_tail_video_path)
                log(f"Successfully captured tail video: {os.path.basename(out_tail_video_path)}")

        return GenerationResult(
            success=True,
            file_path=out_file,
            tail_latent_path=out_tail_latent_path,
            tail_video_path=out_tail_video_path,
            metadata={"name": self.name, "prefix": prefix_str}
        )


class BaseWanVideoGenerator(BaseVideoGenerator):
    """
    Common Wan 2.2 base implementing VAE, CLIP, conditioning, latent chaining,
    and core/tail slicing via native ComfyUI nodes.
    """

    def _build_wan_graph(
        self,
        request: VideoRequest,
        high_source: list,
        low_source: list,
        actual_steps: int,
        split_step: int,
        actual_cfg: float,
        prefix_str: str,
        out_video_name: str
    ) -> Dict[str, Any]:
        width = snap_to_16(request.width or self.settings.get("width", 832))
        height = snap_to_16(request.height or self.settings.get("height", 480))
        num_frames = request.num_frames
        fps = request.fps or self.settings.get("fps", 16)
        seed = random.randint(1, 1_000_000_000) if request.seed in (-1, None) else int(request.seed)

        wf: Dict[str, Any] = {
            "clip_loader": {
                "inputs": {
                    "clip_name": self.settings.get("text_encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    "type": "wan",
                    "device": "default"
                },
                "class_type": "CLIPLoader"
            },
            "vae_loader": {
                "inputs": {
                    "vae_name": self.settings.get("vae", "wan_2.1_vae.safetensors")
                },
                "class_type": "VAELoader"
            },
            "unet_high": {
                "inputs": {
                    "unet_name": self.settings.get("checkpoint", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "unet_low": {
                "inputs": {
                    "unet_name": self.settings.get("checkpoint_low", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "sampling_high": {
                "inputs": {
                    "model": high_source,
                    "shift": 5.0
                },
                "class_type": "ModelSamplingSD3"
            },
            "sampling_low": {
                "inputs": {
                    "model": low_source,
                    "shift": 5.0
                },
                "class_type": "ModelSamplingSD3"
            },
            "positive_cond": {
                "inputs": {
                    "clip": ["clip_loader", 0],
                    "text": request.prompt
                },
                "class_type": "CLIPTextEncode"
            },
            "negative_cond": {
                "inputs": {
                    "clip": ["clip_loader", 0],
                    "text": request.negative_prompt or "blurry, jitter, static, low quality, deformed, extra limbs, bad anatomy, text, watermark"
                },
                "class_type": "CLIPTextEncode"
            }
        }

        # Conditioning Setup
        if request.prefix_latent_path and os.path.exists(request.prefix_latent_path):
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_latent_path,
                custom_filename=f"wan_pref_{int(time.time()*1000)}.latent"
            )
            log(f"Configuring DIRECT LATENT PREFIX via WanLatentPrefixToVideo: {prefix_fn}")
            wf["load_prefix_latent"] = {
                "inputs": {"latent": prefix_fn},
                "class_type": "LoadLatent"
            }
            wf["wan_conditioning"] = {
                "inputs": {
                    "positive": ["positive_cond", 0],
                    "negative": ["negative_cond", 0],
                    "vae": ["vae_loader", 0],
                    "prefix_latent": ["load_prefix_latent", 0],
                    "width": width,
                    "height": height,
                    "length": num_frames,
                    "batch_size": 1
                },
                "class_type": "WanLatentPrefixToVideo"
            }
        elif request.end_image_path and os.path.exists(request.end_image_path):
            start_fn = comfy_client.upload_file_to_comfy(request.image_path)
            end_fn = comfy_client.upload_file_to_comfy(request.end_image_path)
            wf["start_image"] = {"inputs": {"image": start_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["end_image"] = {"inputs": {"image": end_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["wan_conditioning"] = {
                "inputs": {
                    "positive": ["positive_cond", 0],
                    "negative": ["negative_cond", 0],
                    "vae": ["vae_loader", 0],
                    "start_image": ["start_image", 0],
                    "end_image": ["end_image", 0],
                    "width": width,
                    "height": height,
                    "length": num_frames,
                    "batch_size": 1
                },
                "class_type": "WanFirstLastFrameToVideo"
            }
        else:
            start_img = request.image_path
            if isinstance(start_img, list) and len(start_img) > 0:
                curr_node = None
                for i, p in enumerate(start_img):
                    img_fn = comfy_client.upload_file_to_comfy(p)
                    node_key = f"prefix_frame_{i}"
                    wf[node_key] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
                    if curr_node is None:
                        curr_node = [node_key, 0]
                    else:
                        batch_key = f"batch_prefix_{i}"
                        wf[batch_key] = {"inputs": {"image1": curr_node, "image2": [node_key, 0]}, "class_type": "ImageBatch"}
                        curr_node = [batch_key, 0]
                start_image_node = curr_node
            elif isinstance(start_img, str) and os.path.exists(start_img):
                img_fn = comfy_client.upload_file_to_comfy(start_img)
                wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
                start_image_node = ["start_image", 0]
            else:
                raise ValueError(f"No valid start input provided for Wan video generator: {start_img}")

            wf["wan_conditioning"] = {
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

        cond_positive = ["wan_conditioning", 0]
        cond_negative = ["wan_conditioning", 1]
        latent_input = ["wan_conditioning", 2]

        wf.update({
            "sampler_stage1": {
                "inputs": {
                    "model": ["sampling_high", 0],
                    "positive": cond_positive,
                    "negative": cond_negative,
                    "latent_image": latent_input,
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": actual_steps,
                    "cfg": actual_cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": 0,
                    "end_at_step": split_step,
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
                    "steps": actual_steps,
                    "cfg": actual_cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "start_at_step": split_step,
                    "end_at_step": actual_steps,
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

        slice_batch_idx = int(request.prefix_frames) if request.prefix_frames > 0 else 0
        if request.core_frames is not None and request.core_frames < num_frames:
            wf["slice_core"] = {
                "inputs": {
                    "image": ["vae_decode", 0],
                    "batch_index": int(slice_batch_idx),
                    "length": int(request.core_frames)
                },
                "class_type": "ImageFromBatch"
            }
            video_images_input = ["slice_core", 0]
            log(f"Configured ComfyUI ImageFromBatch core: start={slice_batch_idx}, length={request.core_frames}")
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

        return wf


class WanFastVideoGenerator(BaseWanVideoGenerator):
    """
    Wan 2.2 Fast Mode Generator.
    Uses 4-step Lightx2v distillation LoRAs, CFG 1.0, 4 steps (split 2/2).
    Generates draft blocks quickly (~25s/block) with minimal motion.
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        actual_steps = int(request.steps or self.settings.get("steps", 4))
        actual_cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 1.0))
        split_step = max(1, actual_steps // 2)

        out_video_name = request.out_video_name or f"wan_fast_{int(time.time()*1000)}.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("_core"):
            clean_base = clean_base[:-5]
        prefix_str = clean_base

        log(f"Configuring Wan 2.2 FAST MODE: 4-steps distillation LoRA, CFG {actual_cfg}, {actual_steps} steps (split @ {split_step})")

        high_lora = {
            "inputs": {
                "model": ["unet_high", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        }
        low_lora = {
            "inputs": {
                "model": ["unet_low", 0],
                "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
                "strength_model": 1.0
            },
            "class_type": "LoraLoaderModelOnly"
        }

        wf = self._build_wan_graph(
            request=request,
            high_source=["lora_high", 0],
            low_source=["lora_low", 0],
            actual_steps=actual_steps,
            split_step=split_step,
            actual_cfg=actual_cfg,
            prefix_str=prefix_str,
            out_video_name=out_video_name
        )
        wf["lora_high"] = high_lora
        wf["lora_low"] = low_lora

        return self._execute_comfy_workflow(
            wf=wf,
            prefix_str=prefix_str,
            out_video_name=out_video_name,
            save_tail_latent=request.save_tail,
            progress_callback=request.progress_callback
        )


class WanFullVideoGenerator(BaseWanVideoGenerator):
    """
    Wan 2.2 Full Sampling Mode Generator.
    Uses pure High and Low Noise UNets with NO distillation LoRAs.
    Default 20 steps (split 10/10), CFG 5.0.
    Produces full macro-dynamics and high-quality motion.
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        actual_steps = int(request.steps or self.settings.get("steps", 20))
        actual_cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 3.5))
        split_step = max(1, int(round(actual_steps * 0.5)))

        out_video_name = request.out_video_name or f"wan_full_{int(time.time()*1000)}.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("_core"):
            clean_base = clean_base[:-5]
        prefix_str = clean_base

        log(f"Configuring Wan 2.2 FULL SAMPLING MODE: Direct High/Low Noise UNets, CFG {actual_cfg}, {actual_steps} steps (split @ {split_step})")

        wf = self._build_wan_graph(
            request=request,
            high_source=["unet_high", 0],
            low_source=["unet_low", 0],
            actual_steps=actual_steps,
            split_step=split_step,
            actual_cfg=actual_cfg,
            prefix_str=prefix_str,
            out_video_name=out_video_name
        )

        return self._execute_comfy_workflow(
            wf=wf,
            prefix_str=prefix_str,
            out_video_name=out_video_name,
            save_tail_latent=request.save_tail,
            progress_callback=request.progress_callback
        )


class LtxVideoGenerator(BaseVideoGenerator):
    """
    LTX-2.3 22B Video Generator via ComfyUI.
    Uses 20 steps, CFG 3.0, Distilled LoRA 1.1, and LTXVConditioning.
    Supports multi-frame forced prefix video conditioning.
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        width = snap_to_32(request.width or self.settings.get("width", 832))
        height = snap_to_32(request.height or self.settings.get("height", 480))
        fps = request.fps or self.settings.get("fps", 24)
        seed = random.randint(1, 1_000_000_000) if request.seed in (-1, None) else int(request.seed)
        cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 3.0))
        steps = int(request.steps or self.settings.get("steps", 20))
        num_frames = request.num_frames

        out_video_name = request.out_video_name or f"ltx_{int(time.time()*1000)}.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("_core"):
            clean_base = clean_base[:-5]
        prefix_str = f"ltx23_{int(time.time()*1000)}"

        wf: Dict[str, Any] = {
            "unet_loader": {
                "inputs": {
                    "unet_name": self.settings.get("checkpoint", "ltx-2.3-22b-dev-fp8.safetensors"),
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "lora_loader": {
                "inputs": {
                    "model": ["unet_loader", 0],
                    "lora_name": "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
                    "strength_model": 1.0
                },
                "class_type": "LoraLoaderModelOnly"
            },
            "clip_loader": {
                "inputs": {
                    "clip_name1": "ltx-2-3-22b-text_encoder.safetensors",
                    "clip_name2": "ltx-2-3-22b-text_encoder.safetensors",
                    "type": "ltxv"
                },
                "class_type": "DualCLIPLoader"
            },
            "vae_loader": {
                "inputs": {
                    "vae_name": "ltx-2-3-22b-VAE.safetensors"
                },
                "class_type": "VAELoader"
            },
            "positive_cond": {
                "inputs": {
                    "clip": ["clip_loader", 0],
                    "text": request.prompt
                },
                "class_type": "CLIPTextEncode"
            },
            "negative_cond": {
                "inputs": {
                    "clip": ["clip_loader", 0],
                    "text": request.negative_prompt or "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark"
                },
                "class_type": "CLIPTextEncode"
            },
            "ltxv_cond": {
                "inputs": {
                    "positive": ["positive_cond", 0],
                    "negative": ["negative_cond", 0],
                    "frame_rate": float(fps)
                },
                "class_type": "LTXVConditioning"
            },
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

        # Start conditioning (Prefix video or start image)
        if request.prefix_video_path and os.path.exists(request.prefix_video_path):
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_video_path,
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
                    "format": "AnimateDiff"
                },
                "class_type": "VHS_LoadVideoPath"
            }
            wf["ltxv_img_to_video"] = {
                "inputs": {
                    "image": ["prefix_video_loader", 0],
                    "vae": ["vae_loader", 0]
                },
                "class_type": "LTXVImageToVideo"
            }
        elif request.image_path and os.path.exists(str(request.image_path)):
            img_fn = comfy_client.upload_file_to_comfy(str(request.image_path))
            wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["ltxv_img_to_video"] = {
                "inputs": {
                    "image": ["start_image", 0],
                    "vae": ["vae_loader", 0]
                },
                "class_type": "LTXVImageToVideo"
            }
        else:
            raise ValueError(f"No valid start input provided for LTX video generator: {request.image_path}")

        wf.update({
            "guider": {
                "inputs": {
                    "model": ["lora_loader", 0],
                    "positive": ["ltxv_cond", 0],
                    "negative": ["ltxv_cond", 1],
                    "cfg": cfg
                },
                "class_type": "CFGGuider"
            },
            "noise": {
                "inputs": {
                    "noise_seed": seed
                },
                "class_type": "RandomNoise"
            },
            "sampler_select": {
                "inputs": {
                    "sampler_name": "euler"
                },
                "class_type": "KSamplerSelect"
            },
            "scheduler": {
                "inputs": {
                    "steps": steps,
                    "max_shift": 2.05,
                    "base_shift": 0.95,
                    "stretch": True,
                    "terminal": 0.1,
                    "latent": ["empty_latent", 0]
                },
                "class_type": "LTXVScheduler"
            },
            "sampler": {
                "inputs": {
                    "noise": ["noise", 0],
                    "guider": ["guider", 0],
                    "sampler": ["sampler_select", 0],
                    "sigmas": ["scheduler", 0],
                    "latent_image": ["ltxv_img_to_video", 0]
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

        if request.core_frames is not None and request.core_frames < num_frames:
            wf["slice_core"] = {
                "inputs": {
                    "image": ["vae_decode", 0],
                    "batch_index": 0,
                    "length": int(request.core_frames)
                },
                "class_type": "ImageFromBatch"
            }
            video_images_input = ["slice_core", 0]
            log(f"Configured ComfyUI ImageFromBatch to output {request.core_frames} core frames directly into MP4 (LTX)")
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

        if request.save_tail and request.core_frames is not None and request.core_frames < num_frames:
            tail_len = int(num_frames - request.core_frames)
            wf["slice_tail"] = {
                "inputs": {
                    "image": ["vae_decode", 0],
                    "batch_index": int(request.core_frames),
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
            log(f"Configured ComfyUI to export tail video ({tail_len} frames, start={request.core_frames}) directly for forced prefix chaining (LTX)")

        return self._execute_comfy_workflow(
            wf=wf,
            prefix_str=prefix_str,
            out_video_name=out_video_name,
            save_tail_video=(request.save_tail and request.core_frames is not None and request.core_frames < num_frames),
            progress_callback=request.progress_callback
        )
