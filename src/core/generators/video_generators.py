#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Generators for media_mcp.
Polymorphic strategy implementations for Wan 2.2 Fast (4-step LoRA), Wan 2.2 Full (20-step UNet),
and LTX-2.3 22B with True 5D Latent Prefix Locking and Restoration.
"""

from __future__ import annotations
import sys
import uuid
import datetime
import shutil
import random
from pathlib import Path
from typing import Any, Dict, Optional, List

from .. import config
from .. import comfy_client
from .. import discovery
from .. import lora_inspector
from .base import BaseGenerator, VideoRequest, GenerationResult, snap_to_16, snap_to_32


def log(msg: str):
    sys.stderr.write(f"[video-generators] {msg}\n")
    sys.stderr.flush()


class BaseVideoGenerator(BaseGenerator):
    """Abstract base class for video generators."""

    kind = "video"
    temporal_stride: int = 4
    default_fps: int = 16

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "temporal_stride": self.temporal_stride,
            "default_fps": self.default_fps,
            "settings": dict(self.settings)
        }

    def _locate_direct_output(self, prefix_match: str, ext: str, exclude_prefix: Optional[str] = None) -> Optional[str]:
        """Checks top-level COMFY_OUTPUT_DIR directly without recursive os.walk."""
        out_dir = config.COMFY_OUTPUT_DIR
        if not out_dir or not Path(out_dir).is_dir():
            return None
        try:
            for entry in Path(out_dir).iterdir():
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith(prefix_match) and name.endswith(ext):
                    if exclude_prefix and name.startswith(exclude_prefix):
                        continue
                    return entry.resolve().as_posix()
        except Exception:
            pass
        return None

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
        temp_dir = config.get_temp_dir()

        # 1. Locate core video
        save_video_out = outputs.get("save_video", {})
        videos = save_video_out.get("videos", []) or save_video_out.get("gifs", []) or save_video_out.get("images", [])

        found_local_file = None
        if videos:
            fn = videos[0]["filename"]
            sub = videos[0].get("subfolder", "")
            found_local_file = comfy_client.find_comfy_output_file(fn, sub)

        if not found_local_file:
            found_local_file = self._locate_direct_output(
                prefix_match=prefix_str,
                ext=".mp4",
                exclude_prefix=f"{prefix_str}-tail"
            )

        out_file = (Path(temp_dir) / out_video_name).resolve().as_posix()
        if found_local_file and Path(found_local_file).is_file():
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
            found_latent = None
            if saved_latents:
                fn = saved_latents[0].get("filename")
                sub = saved_latents[0].get("subfolder", "")
                found_latent = comfy_client.find_comfy_output_file(fn, sub)
            if not found_latent:
                found_latent = self._locate_direct_output(prefix_match=f"{prefix_str}-tail", ext=".latent")
            if found_latent and Path(found_latent).is_file():
                out_tail_latent_path = (Path(temp_dir) / f"{prefix_str}-tail.latent").resolve().as_posix()
                shutil.copy2(found_latent, out_tail_latent_path)
                log(f"Successfully captured tail latent: {Path(out_tail_latent_path).name}")
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
                found_tail_img = self._locate_direct_output(prefix_match=f"{prefix_str}-tail", ext=".png")

            if found_tail_img and Path(found_tail_img).is_file():
                out_tail_image_path = (Path(temp_dir) / f"{prefix_str}-tail.png").resolve().as_posix()
                shutil.copy2(found_tail_img, out_tail_image_path)
                log(f"Successfully captured pristine tail image from workflow: {Path(out_tail_image_path).name}")
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
    user LoRA chaining, and core/tail slicing via native ComfyUI nodes.
    """

    temporal_stride = 4
    default_fps = 16

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

        model_bundle = str(
            request.model or self.settings.get("model") or self.settings.get("checkpoint") or ""
        ).strip()
        if not model_bundle:
            raise ValueError("Parameter 'model' is required for Wan video generation.")

        unet_high_name, unet_low_name = discovery.resolve_wan_model_bundle(model_bundle)
        if self.settings.get("checkpoint_low") or self.settings.get("model_low"):
            unet_low_name = str(self.settings.get("checkpoint_low") or self.settings.get("model_low"))

        weight_dtype = str(request.weight_dtype or self.settings.get("weight_dtype") or "default").strip()

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
                    "unet_name": unet_high_name,
                    "weight_dtype": weight_dtype
                },
                "class_type": "UNETLoader"
            },
            "unet_low": {
                "inputs": {
                    "unet_name": unet_low_name,
                    "weight_dtype": weight_dtype
                },
                "class_type": "UNETLoader"
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

        # Chain user-specified LoRAs onto both high_source and low_source branches
        loras_list: List[Dict[str, Any]] = request.loras if request.loras is not None else (self.settings.get("loras") or [])
        for idx, lora_entry in enumerate(loras_list):
            if not isinstance(lora_entry, dict):
                continue
            l_name = str(lora_entry.get("name") or "").strip()
            l_strength = float(lora_entry.get("strength", 0.8))
            if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                lora_fn = l_name if l_name.endswith(".safetensors") else f"{l_name}.safetensors"
                h_key = f"user_lora_high_{idx}"
                l_key = f"user_lora_low_{idx}"
                wf[h_key] = {
                    "inputs": {
                        "model": high_source,
                        "lora_name": lora_fn,
                        "strength_model": l_strength
                    },
                    "class_type": "LoraLoaderModelOnly"
                }
                wf[l_key] = {
                    "inputs": {
                        "model": low_source,
                        "lora_name": lora_fn,
                        "strength_model": l_strength
                    },
                    "class_type": "LoraLoaderModelOnly"
                }
                high_source = [h_key, 0]
                low_source = [l_key, 0]

        wf["sampling_high"] = {
            "inputs": {
                "model": high_source,
                "shift": 5.0
            },
            "class_type": "ModelSamplingSD3"
        }
        wf["sampling_low"] = {
            "inputs": {
                "model": low_source,
                "shift": 5.0
            },
            "class_type": "ModelSamplingSD3"
        }

        # Conditioning Setup
        end_img_node = None
        if request.end_image_path and Path(request.end_image_path).is_file():
            end_fn = comfy_client.upload_file_to_comfy(
                request.end_image_path,
                run_prefix=prefix_str,
                suffix="end"
            )
            wf["end_image"] = {"inputs": {"image": end_fn, "upload": "image"}, "class_type": "LoadImage"}
            end_img_node = ["end_image", 0]
            log(f"Configuring Wan end_image conditioning: {end_fn}")

        if request.prefix_latent_path and Path(request.prefix_latent_path).is_file():
            pref_filename = Path(request.prefix_latent_path).name
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_latent_path,
                custom_filename=pref_filename,
                run_prefix=prefix_str,
                suffix="prefix"
            )
            log(f"Configuring DIRECT LATENT PREFIX via WanLatentPrefixToVideo: {prefix_fn}")
            wf["load_prefix_latent"] = {
                "inputs": {"latent": prefix_fn},
                "class_type": "LoadLatent"
            }
            wan_cond_inputs = {
                "positive": ["positive_cond", 0],
                "negative": ["negative_cond", 0],
                "vae": ["vae_loader", 0],
                "prefix_latent": ["load_prefix_latent", 0],
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            }
            if end_img_node:
                wan_cond_inputs["end_image"] = end_img_node
            wf["wan_conditioning"] = {
                "inputs": wan_cond_inputs,
                "class_type": "WanLatentPrefixToVideo"
            }
        elif request.image_path and Path(request.image_path).is_file():
            img_fn = comfy_client.upload_file_to_comfy(
                request.image_path,
                run_prefix=prefix_str,
                suffix="anchor"
            )
            wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
            if end_img_node:
                wf["wan_conditioning"] = {
                    "inputs": {
                        "positive": ["positive_cond", 0],
                        "negative": ["negative_cond", 0],
                        "vae": ["vae_loader", 0],
                        "start_image": ["start_image", 0],
                        "end_image": end_img_node,
                        "width": width,
                        "height": height,
                        "length": num_frames,
                        "batch_size": 1
                    },
                    "class_type": "WanFirstLastFrameToVideo"
                }
            else:
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
        stage2_latent_input = ["sampler_stage1", 0]
        latent_for_decode = ["sampler_stage2", 0]

        if request.prefix_latent_path and Path(request.prefix_latent_path).is_file():
            ramp_l = getattr(request, "ramp_latents", 3)
            ramp_l = 3 if ramp_l is None else int(ramp_l)
            min_c = getattr(request, "min_cfg", 1.0)
            min_c = 1.0 if min_c is None else float(min_c)

            t_latents = ((num_frames - 1) // 4) + 1
            k_lat = lora_inspector.read_latent_temporal_frames(request.prefix_latent_path, default=0)
            if k_lat <= 0:
                pref_px = max(1, int(request.prefix_frames or 17))
                k_lat = max(1, ((pref_px - 1) // 4) + 1)
            k_lat = min(k_lat, t_latents)
            core_lat = t_latents - k_lat

            log(
                f"Enabling WanLatentPrefixLock (Option B): ramp_latents={ramp_l}, min_cfg={min_c}, "
                f"k_lat={k_lat}, core_lat={core_lat} + Stage1/Stage2 Latent AdaIN + WanLatentPrefixRestore"
            )
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

            if core_lat > 0:
                # 1. Mid-diffusion (Stage 1 -> Stage 2) per-channel AdaIN at noise level sigma(split_step)
                # Eliminates channel mean shift (-1.17/+1.01) injected by High-Noise Lightx2v LoRA before Low-Noise UNet runs
                wf["stage1_cut_prefix"] = {
                    "inputs": {
                        "samples": ["sampler_stage1", 0],
                        "dim": "t",
                        "index": 0,
                        "amount": int(k_lat)
                    },
                    "class_type": "LatentCut"
                }
                wf["stage1_cut_core"] = {
                    "inputs": {
                        "samples": ["sampler_stage1", 0],
                        "dim": "t",
                        "index": int(k_lat),
                        "amount": int(core_lat)
                    },
                    "class_type": "LatentCut"
                }
                wf["stage1_adain_core"] = {
                    "inputs": {
                        "latents": ["stage1_cut_core", 0],
                        "reference": ["stage1_cut_prefix", 0],
                        "factor": 1.0
                    },
                    "class_type": "LTXVAdainLatent"
                }
                wf["stage1_restored"] = {
                    "inputs": {
                        "samples1": ["stage1_cut_prefix", 0],
                        "samples2": ["stage1_adain_core", 0],
                        "dim": "t"
                    },
                    "class_type": "LatentConcat"
                }
                stage2_latent_input = ["stage1_restored", 0]

                # 2. Post-diffusion (Stage 2 -> VAEDecode) per-channel AdaIN against pristine prefix_latent
                wf["stage2_cut_core"] = {
                    "inputs": {
                        "samples": ["sampler_stage2", 0],
                        "dim": "t",
                        "index": int(k_lat),
                        "amount": int(core_lat)
                    },
                    "class_type": "LatentCut"
                }
                wf["stage2_adain_core"] = {
                    "inputs": {
                        "latents": ["stage2_cut_core", 0],
                        "reference": ["load_prefix_latent", 0],
                        "factor": 1.0
                    },
                    "class_type": "LTXVAdainLatent"
                }
                wf["stage2_concat"] = {
                    "inputs": {
                        "samples1": ["load_prefix_latent", 0],
                        "samples2": ["stage2_adain_core", 0],
                        "dim": "t"
                    },
                    "class_type": "LatentConcat"
                }
                restore_samples_input = ["stage2_concat", 0]
            else:
                restore_samples_input = ["sampler_stage2", 0]

            wf["prefix_latent_restore"] = {
                "inputs": {
                    "samples": restore_samples_input,
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
                    "latent_image": stage2_latent_input,
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
                    "samples": latent_for_decode,
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
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        actual_steps = int(request.steps or self.settings.get("steps", 4))
        actual_cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 1.0))
        split_step = max(1, actual_steps // 2)

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-wan-fast.mp4"
        clean_base = Path(out_video_name).stem
        if clean_base.endswith("-core"):
            clean_base = clean_base[:-5]
        prefix_str = request.run_prefix or clean_base

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
            save_tail_latent=(request.save_tail and request.tail_frames > 0),
            save_tail_image=request.save_tail_image,
            progress_callback=request.progress_callback
        )


class WanFullVideoGenerator(BaseWanVideoGenerator):
    """
    Wan 2.2 Full Sampling Mode Generator.
    Uses pure High and Low Noise UNets with NO distillation LoRAs.
    Default 20 steps (split 10/10), CFG 5.0.
    """

    def generate(self, request: VideoRequest) -> GenerationResult:
        actual_steps = int(request.steps or self.settings.get("steps", 20))
        actual_cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", 5.0))
        split_step = max(1, int(round(actual_steps * 0.5)))

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-wan-full.mp4"
        clean_base = Path(out_video_name).stem
        if clean_base.endswith("-core"):
            clean_base = clean_base[:-5]
        prefix_str = request.run_prefix or clean_base

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
            save_tail_latent=(request.save_tail and request.tail_frames > 0),
            save_tail_image=request.save_tail_image,
            progress_callback=request.progress_callback
        )


class LtxVideoGenerator(BaseVideoGenerator):
    """
    LTX-2.3 22B Video Generator via ComfyUI.
    Uses 20 steps, CFG 3.0, Distilled LoRA 1.1 (strength 0.2 for full 20-step CFG=3.0 mode, 1.0 for <=10-step CFG=1.0 mode),
    and LTXVConditioning.
    Supports pure 5D tail latent prefix conditioning via LTXVAddLatentGuide + LTXVCropGuides + LTXVAdainLatent
    without intermediate VAEDecode/VAEEncode degradation or t=0 causal slot corruption.
    """

    temporal_stride = 8
    default_fps = 24

    def generate(self, request: VideoRequest) -> GenerationResult:
        width = snap_to_32(request.width or self.settings.get("width", 832))
        height = snap_to_32(request.height or self.settings.get("height", 480))
        fps = request.fps or self.settings.get("fps", 24)
        seed = random.randint(1, 1_000_000_000) if request.seed in (-1, None) else int(request.seed)
        steps = int(request.steps or self.settings.get("steps", 25))
        default_cfg = 3.0
        cfg = float(request.cfg if request.cfg is not None else self.settings.get("cfg", default_cfg))
        num_frames = request.num_frames

        if request.out_video_name:
            out_video_name = request.out_video_name
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            req_hash = uuid.uuid4().hex[:6]
            out_video_name = f"{timestamp}-{req_hash}-ltx.mp4"
        clean_base = Path(out_video_name).stem
        if clean_base.endswith("-core"):
            clean_base = clean_base[:-5]
        prefix_str = request.run_prefix or clean_base

        model_setting = str(
            request.model or self.settings.get("model") or self.settings.get("checkpoint") or ""
        ).strip()
        if not model_setting:
            return GenerationResult(success=False, error="Parameter 'model' is required for LTX video generation.")

        m_lower = model_setting.lower()
        if "sulphur" in m_lower or "sulfur" in m_lower:
            if "bf16" in m_lower and not model_setting.endswith(".safetensors"):
                unet_name = "sulphur_dev_bf16_model.safetensors"
            elif not model_setting.endswith(".safetensors"):
                unet_name = "sulphur_dev_model_fp8mixed.safetensors"
            else:
                unet_name = model_setting
        elif model_setting.endswith(".safetensors"):
            unet_name = model_setting
        elif "bf16" in m_lower:
            unet_name = "ltx-2.3-22b-dev.safetensors"
        else:
            unet_name = "ltx-2.3-22b-dev-fp8.safetensors"

        log(f"Configuring LTX generator: UNet={unet_name}, CFG={cfg}, steps={steps}")

        weight_dtype = str(request.weight_dtype or self.settings.get("weight_dtype") or "default").strip()
        lora_setting = self.settings.get("lora_strength")
        if lora_setting is not None:
            lora_strength = float(lora_setting)
        else:
            # Distilled mode (cfg <= 1.5): strength = 0.7 per official audio_to_video / Lightricks benchmark.
            # Full CFG guidance mode (cfg > 1.5): strength = 0.2.
            lora_strength = 0.7 if cfg <= 1.5 else 0.2

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
            log(f"LTX base LoRA enabled: strength={lora_strength}")
        else:
            guider_model = ["unet_loader", 0]
            log("LTX base LoRA disabled (strength=0)")

        # Chain user-specified LoRAs onto guider_model
        loras_list: List[Dict[str, Any]] = request.loras if request.loras is not None else (self.settings.get("loras") or [])
        for idx, lora_entry in enumerate(loras_list):
            if not isinstance(lora_entry, dict):
                continue
            l_name = str(lora_entry.get("name") or "").strip()
            l_strength = float(lora_entry.get("strength", 0.8))
            if l_name and l_name.lower() not in ("none", "disabled", "false", ""):
                lora_fn = l_name if l_name.endswith(".safetensors") else f"{l_name}.safetensors"
                u_key = f"user_lora_{idx}"
                wf[u_key] = {
                    "inputs": {
                        "model": guider_model,
                        "lora_name": lora_fn,
                        "strength_model": l_strength
                    },
                    "class_type": "LoraLoaderModelOnly"
                }
                guider_model = [u_key, 0]

        # Conditioning Setup: Pure 5D Latent Prefix via LTXVAddLatentGuide/LTXVCropGuides/LTXVAdainLatent vs Image-to-Video
        active_positive = ["ltxv_cond", 0]
        active_negative = ["ltxv_cond", 1]
        scheduler_latent_node = ["empty_latent", 0]
        latent_for_decode = ["sampler", 0]
        has_guides = False
        has_prefix_latent = False
        has_end_image = bool(request.end_image_path and Path(str(request.end_image_path)).is_file())

        if request.prefix_latent_path and Path(request.prefix_latent_path).is_file():
            pref_filename = Path(request.prefix_latent_path).name
            prefix_fn = comfy_client.upload_file_to_comfy(
                request.prefix_latent_path,
                custom_filename=pref_filename,
                run_prefix=prefix_str,
                suffix="prefix"
            )

            log(f"Configuring NATIVE 5D LATENT PREFIX for LTX via LTXVAddLatentGuide + LTXVCropGuides + LTXVAdainLatent: {prefix_fn}")
            wf["load_prefix_latent"] = {
                "inputs": {"latent": prefix_fn},
                "class_type": "LoadLatent"
            }
            wf["ltx_latent_guide"] = {
                "inputs": {
                    "positive": ["ltxv_cond", 0],
                    "negative": ["ltxv_cond", 1],
                    "vae": ["vae_loader", 0],
                    "latent": ["empty_latent", 0],
                    "guiding_latent": ["load_prefix_latent", 0],
                    "latent_idx": 0,
                    "strength": 1.0
                },
                "class_type": "LTXVAddLatentGuide"
            }
            active_positive = ["ltx_latent_guide", 0]
            active_negative = ["ltx_latent_guide", 1]
            active_latent_node = ["ltx_latent_guide", 2]
            has_guides = True
            has_prefix_latent = True
        elif request.image_path and Path(str(request.image_path)).is_file():
            img_fn = comfy_client.upload_file_to_comfy(
                str(request.image_path),
                run_prefix=prefix_str,
                suffix="anchor"
            )
            log(f"Configuring LTX start_image guide (frame_idx=0, strength=0.7): {img_fn}")
            wf["start_image"] = {"inputs": {"image": img_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["preprocess_image"] = {
                "inputs": {
                    "image": ["start_image", 0],
                    "img_compression": 25
                },
                "class_type": "LTXVPreprocess"
            }
            wf["ltx_start_guide"] = {
                "inputs": {
                    "positive": active_positive,
                    "negative": active_negative,
                    "vae": ["vae_loader", 0],
                    "latent": ["empty_latent", 0],
                    "image": ["preprocess_image", 0],
                    "frame_idx": 0,
                    "strength": 0.7
                },
                "class_type": "LTXVAddGuide"
            }
            active_positive = ["ltx_start_guide", 0]
            active_negative = ["ltx_start_guide", 1]
            active_latent_node = ["ltx_start_guide", 2]
            scheduler_latent_node = ["empty_latent", 0]
            has_guides = True
        else:
            raise ValueError(f"No valid start input provided for LTX video generator: {request.image_path}")

        if has_end_image:
            end_img_fn = comfy_client.upload_file_to_comfy(
                str(request.end_image_path),
                run_prefix=prefix_str,
                suffix="end"
            )
            log(f"Configuring LTX end_image guide (frame_idx=-1, strength=0.7): {end_img_fn}")
            wf["end_image"] = {"inputs": {"image": end_img_fn, "upload": "image"}, "class_type": "LoadImage"}
            wf["preprocess_end_image"] = {
                "inputs": {
                    "image": ["end_image", 0],
                    "img_compression": 25
                },
                "class_type": "LTXVPreprocess"
            }
            wf["ltx_end_guide"] = {
                "inputs": {
                    "positive": active_positive,
                    "negative": active_negative,
                    "vae": ["vae_loader", 0],
                    "latent": active_latent_node,
                    "image": ["preprocess_end_image", 0],
                    "frame_idx": -1,
                    "strength": 0.7
                },
                "class_type": "LTXVAddGuide"
            }
            active_positive = ["ltx_end_guide", 0]
            active_negative = ["ltx_end_guide", 1]
            active_latent_node = ["ltx_end_guide", 2]
            has_guides = True

        if has_guides:
            wf["ltx_crop_guides"] = {
                "inputs": {
                    "positive": active_positive,
                    "negative": active_negative,
                    "latent": ["sampler", 0]
                },
                "class_type": "LTXVCropGuides"
            }
            if has_prefix_latent:
                wf["ltx_adain_latent"] = {
                    "inputs": {
                        "latents": ["ltx_crop_guides", 2],
                        "reference": ["load_prefix_latent", 0],
                        "factor": 1.0
                    },
                    "class_type": "LTXVAdainLatent"
                }
                latent_for_decode = ["ltx_adain_latent", 0]
            else:
                latent_for_decode = ["ltx_crop_guides", 2]

        default_sampler = "euler_ancestral"
        actual_sampler = str(getattr(request, "sampler", None) or self.settings.get("sampler") or default_sampler).strip()

        wf.update({
            "guider": {
                "inputs": {
                    "model": guider_model,
                    "positive": active_positive,
                    "negative": active_negative,
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
                    "sampler_name": actual_sampler
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
                    "latent": scheduler_latent_node
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
                    "samples": latent_for_decode,
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
