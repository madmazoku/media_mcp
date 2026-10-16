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
import uuid
import datetime
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
        save_tail_image: bool = False,
        progress_callback: Optional[Any] = None
    ) -> GenerationResult:
        """
        Submits workflow to ComfyUI, waits for completion, and extracts
        the core video, optional tail latent, and optional tail image.
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
                    if f.startswith(prefix_str) and not f.startswith(f"{prefix_str}-tail") and f.endswith(".mp4"):
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
                    out_tail_latent_path = os.path.join(TEMP_DIR, f"{prefix_str}-tail.latent")
                    shutil.copy2(found, out_tail_latent_path)
            if not out_tail_latent_path:
                for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
                    for f in files:
                        if f.startswith(f"{prefix_str}-tail") and f.endswith(".latent"):
                            cand = os.path.join(root, f)
                            out_tail_latent_path = os.path.join(TEMP_DIR, f"{prefix_str}-tail.latent")
                            shutil.copy2(cand, out_tail_latent_path)
                            break
                    if out_tail_latent_path:
                        break
            if out_tail_latent_path:
                log(f"Successfully captured tail latent: {os.path.basename(out_tail_latent_path)}")
            else:
                log(f"Warning: Tail latent output file not found for {prefix_str}")

        # 3. Locate tail image (if requested)
        out_tail_image_path: Optional[str] = None
        if save_tail_image:
            save_tail_img_out = outputs.get("save_tail_image", {})
            saved_images = save_tail_img_out.get("images", [])
            found_tail_img = None
            if saved_images:
                fn = saved_images[0].get("filename")
                sub = saved_images[0].get("subfolder", "")
                found_tail_img = comfy_client.find_comfy_output_file(fn, sub)

            if not found_tail_img:
                for root, _, files in os.walk(comfy_client.COMFY_OUTPUT_DIR):
                    for f in files:
                        if f.startswith(f"{prefix_str}-tail") and f.endswith(".png"):
                            found_tail_img = os.path.join(root, f)
                            break
                    if found_tail_img:
                        break

            if found_tail_img and os.path.exists(found_tail_img):
                out_tail_image_path = os.path.join(TEMP_DIR, f"{prefix_str}-tail.png")
                shutil.copy2(found_tail_img, out_tail_image_path)
                log(f"Successfully captured pristine tail image from workflow: {os.path.basename(out_tail_image_path)}")
            else:
                log(f"Warning: Tail image output file not found for {prefix_str}")

        return GenerationResult(
            success=True,
            file_path=out_file,
            tail_latent_path=out_tail_latent_path,
            tail_image_path=out_tail_image_path,
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
                    "unet_name": self.settings.get("checkpoint") or self.settings.get("model") or "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "unet_low": {
                "inputs": {
                    "unet_name": self.settings.get("checkpoint_low") or self.settings.get("model_low") or "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
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
            pref_filename = os.path.basename(request.prefix_latent_path)
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_latent_path,
                custom_filename=pref_filename
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
        elif request.image_path and os.path.exists(request.image_path):
            img_fn = comfy_client.upload_file_to_comfy(request.image_path)
            wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["wan_conditioning"] = {
                "inputs": {
                    "positive": ["positive_cond", 0],
                    "negative": ["negative_cond", 0],
                    "vae": ["vae_loader", 0],
                    "start_image": ["start_image", 0],
                    "width": width,
                    "height": height,
                    "length": num_frames,
                    "batch_size": 1
                },
                "class_type": "WanImageToVideo"
            }
        else:
            raise ValueError(f"No valid start input provided for Wan video generator: {request.image_path}")

        cond_positive = ["wan_conditioning", 0]
        cond_negative = ["wan_conditioning", 1]
        latent_input = ["wan_conditioning", 2]

        model_high = ["sampling_high", 0]
        model_low = ["sampling_low", 0]
        latent_for_decode = ["sampler_stage2", 0]

        if request.prefix_latent_path and os.path.exists(request.prefix_latent_path):
            ramp_l = getattr(request, "ramp_latents", 3)
            ramp_l = 3 if ramp_l is None else int(ramp_l)
            min_c = getattr(request, "min_cfg", 1.0)
            min_c = 1.0 if min_c is None else float(min_c)

            log(f"Enabling WanLatentPrefixLock (Option B): ramp_latents={ramp_l}, min_cfg={min_c} + WanLatentPrefixRestore")
            wf["prefix_lock_stage1"] = {
                "inputs": {
                    "model": model_high,
                    "prefix_latent": ["load_prefix_latent", 0],
                    "ramp_latents": ramp_l,
                    "min_cfg": min_c,
                },
                "class_type": "WanLatentPrefixLock"
            }
            wf["prefix_lock_stage2"] = {
                "inputs": {
                    "model": model_low,
                    "prefix_latent": ["load_prefix_latent", 0],
                    "ramp_latents": ramp_l,
                    "min_cfg": min_c,
                },
                "class_type": "WanLatentPrefixLock"
            }
            model_high = ["prefix_lock_stage1", 0]
            model_low = ["prefix_lock_stage2", 0]

            wf["prefix_latent_restore"] = {
                "inputs": {
                    "samples": ["sampler_stage2", 0],
                    "prefix_latent": ["load_prefix_latent", 0],
                },
                "class_type": "WanLatentPrefixRestore"
            }
            latent_for_decode = ["prefix_latent_restore", 0]

        wf.update({
            "sampler_stage1": {
                "inputs": {
                    "model": model_high,
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
                    "model": model_low,
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
                    "samples": latent_for_decode,
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

        if request.save_tail and request.tail_frames > 0:
            wf["cut_tail_latent"] = {
                "inputs": {
                    "samples": ["sampler_stage2", 0],
                    "dim": "t",
                    "index": -int(request.tail_frames),
                    "amount": int(request.tail_frames)
                },
                "class_type": "LatentCut"
            }
            wf["save_tail_latent"] = {
                "inputs": {
                    "samples": ["cut_tail_latent", 0],
                    "filename_prefix": f"{prefix_str}-tail"
                },
                "class_type": "SaveLatent"
            }
            log(f"Configured native LatentCut (dim=t, amount={request.tail_frames}) + SaveLatent: prefix={prefix_str}-tail")

        if request.save_tail_image:
            wf["slice_tail_image"] = {
                "inputs": {
                    "image": ["vae_decode", 0],
                    "batch_index": int(num_frames - 1),
                    "length": 1
                },
                "class_type": "ImageFromBatch"
            }
            wf["save_tail_image"] = {
                "inputs": {
                    "images": ["slice_tail_image", 0],
                    "filename_prefix": f"{prefix_str}-tail"
                },
                "class_type": "SaveImage"
            }
            log(f"Configured ComfyUI ImageFromBatch (last frame {num_frames - 1}) + SaveImage: prefix={prefix_str}-tail")

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

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-wan-fast.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("-core"):
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
            save_tail_image=request.save_tail_image,
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

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-wan-full.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("-core"):
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
            save_tail_image=request.save_tail_image,
            progress_callback=request.progress_callback
        )


class LtxVideoGenerator(BaseVideoGenerator):
    """
    LTX-2.3 22B Video Generator via ComfyUI.
    Uses 20 steps, CFG 3.0, Distilled LoRA 1.1, and LTXVConditioning.
    Supports true tail latent prefix conditioning.
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        width = snap_to_32(request.width or self.settings.get("width", 832))
        height = snap_to_32(request.height or self.settings.get("height", 480))
        fps = request.fps or self.settings.get("fps", 24)
        seed = random.randint(1, 1_000_000_000) if request.seed in (-1, None) else int(request.seed)
        cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 3.0))
        steps = int(request.steps or self.settings.get("steps", 20))
        num_frames = request.num_frames

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-ltx.mp4"
        clean_base = os.path.splitext(os.path.basename(out_video_name))[0]
        if clean_base.endswith("-core"):
            clean_base = clean_base[:-5]
        prefix_str = clean_base

        model_setting = str(self.settings.get("model") or self.settings.get("checkpoint") or "ltx-2.3-22b-dev-fp8.safetensors").strip()
        m_lower = model_setting.lower()
        if "sulphur" in m_lower or "sulfur" in m_lower:
            if "bf16" in m_lower:
                unet_name = "sulphur_dev_bf16_model.safetensors"
            else:
                unet_name = "sulphur_dev_model_fp8mixed.safetensors"
        elif "dev" in m_lower or "ltx" in m_lower or "base" in m_lower:
            if "bf16" in m_lower:
                unet_name = "ltx-2.3-22b-dev.safetensors"
            else:
                unet_name = "ltx-2.3-22b-dev-fp8.safetensors"
        else:
            unet_name = model_setting if model_setting.endswith(".safetensors") else f"{model_setting}.safetensors"

        log(f"Configuring LTX generator: UNet={unet_name}, CFG={cfg}, steps={steps}")

        weight_dtype = str(self.settings.get("weight_dtype") or "default").strip()
        lora_setting = self.settings.get("lora_strength")
        lora_strength = float(lora_setting) if lora_setting is not None else 1.0

        wf: Dict[str, Any] = {
            "unet_loader": {
                "inputs": {
                    "unet_name": unet_name,
                    "weight_dtype": weight_dtype
                },
                "class_type": "UNETLoader"
            },
            "clip_loader": {
                "inputs": {
                    "clip_name1": self.settings.get("text_encoder", "gemma_3_12B_it.safetensors"),
                    "clip_name2": self.settings.get("text_encoder_proj", "ltx-2-3-22b-text_encoder.safetensors"),
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

        # Start conditioning (Prefix latent or start image)
        start_image_node = None
        if request.prefix_latent_path and os.path.exists(request.prefix_latent_path):
            pref_filename = os.path.basename(request.prefix_latent_path)
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_latent_path,
                custom_filename=pref_filename
            )
            log(f"Configuring TRUE TAIL LATENT PREFIX for LTX: {prefix_fn}")
            wf["load_prefix_latent"] = {
                "inputs": {"latent": prefix_fn},
                "class_type": "LoadLatent"
            }
            wf["prefix_latent_decode"] = {
                "inputs": {
                    "samples": ["load_prefix_latent", 0],
                    "vae": ["vae_loader", 0]
                },
                "class_type": "VAEDecode"
            }
            start_image_node = ["prefix_latent_decode", 0]
        elif request.image_path and os.path.exists(str(request.image_path)):
            img_fn = comfy_client.upload_file_to_comfy(str(request.image_path))
            wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
            start_image_node = ["start_image", 0]
        else:
            raise ValueError(f"No valid start input provided for LTX video generator: {request.image_path}")

        if start_image_node:
            wf["inplace_cond"] = {
                "inputs": {
                    "image": start_image_node,
                    "vae": ["vae_loader", 0],
                    "latent": ["empty_latent", 0],
                    "strength": 1.0,
                    "bypass": False
                },
                "class_type": "LTXVImgToVideoInplace"
            }
            active_latent_node = ["inplace_cond", 0]
        else:
            active_latent_node = ["empty_latent", 0]

        if lora_strength > 0:
            wf["lora_loader"] = {
                "inputs": {
                    "model": ["unet_loader", 0],
                    "lora_name": str(self.settings.get("lora_name") or "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"),
                    "strength_model": lora_strength
                },
                "class_type": "LoraLoaderModelOnly"
            }
            guider_model = ["lora_loader", 0]
            log(f"LTX LoRA enabled: strength={lora_strength}")
        else:
            guider_model = ["unet_loader", 0]
            log("LTX LoRA disabled (strength=0)")

        wf.update({
            "guider": {
                "inputs": {
                    "model": guider_model,
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
                    "latent": active_latent_node
                },
                "class_type": "LTXVScheduler"
            },
            "sampler": {
                "inputs": {
                    "noise": ["noise", 0],
                    "guider": ["guider", 0],
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

        if request.save_tail and request.tail_frames > 0:
            wf["cut_tail_latent"] = {
                "inputs": {
                    "samples": ["sampler", 0],
                    "dim": "t",
                    "index": -int(request.tail_frames),
                    "amount": int(request.tail_frames)
                },
                "class_type": "LatentCut"
            }
            wf["save_tail_latent"] = {
                "inputs": {
                    "samples": ["cut_tail_latent", 0],
                    "filename_prefix": f"{prefix_str}-tail"
                },
                "class_type": "SaveLatent"
            }
            log(f"Configured native LatentCut (dim=t, amount={request.tail_frames}) + SaveLatent for LTX: prefix={prefix_str}-tail")

        if request.save_tail_image:
            wf["slice_tail_image"] = {
                "inputs": {
                    "image": ["vae_decode", 0],
                    "batch_index": int(num_frames - 1),
                    "length": 1
                },
                "class_type": "ImageFromBatch"
            }
            wf["save_tail_image"] = {
                "inputs": {
                    "images": ["slice_tail_image", 0],
                    "filename_prefix": f"{prefix_str}-tail"
                },
                "class_type": "SaveImage"
            }
            log(f"Configured ComfyUI ImageFromBatch (last frame {num_frames - 1}) + SaveImage for LTX: prefix={prefix_str}-tail")

        return self._execute_comfy_workflow(
            wf=wf,
            prefix_str=prefix_str,
            out_video_name=out_video_name,
            save_tail_latent=(request.save_tail and request.tail_frames > 0),
            save_tail_image=request.save_tail_image,
            progress_callback=request.progress_callback
        )
