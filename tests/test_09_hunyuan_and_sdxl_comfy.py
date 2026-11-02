#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for HunyuanVideo 1.5 and SDXL ComfyUI generators.
Validates workflow graph construction, parameter resolution, LoRA chaining,
temporal/spatial stride snapping, discovery trees, and schema validation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.generators import (
    create_image_generator,
    create_video_generator,
    get_image_generator_class,
    get_video_generator_class,
    ImageRequest,
    VideoRequest,
    SdxlComfyImageGenerator,
    HunyuanVideoGenerator,
    IMAGE_GENERATOR_REGISTRY,
    VIDEO_GENERATOR_REGISTRY,
)
from core import discovery, schema_validator


class TestHunyuanAndSdxlComfyGenerators(unittest.TestCase):

    def test_factory_registrations(self):
        """Verifies canonical registration in generator registries and factory creation."""
        self.assertIn("sdxl-comfy", IMAGE_GENERATOR_REGISTRY)
        self.assertIn("hunyuan", VIDEO_GENERATOR_REGISTRY)

        img_gen = create_image_generator("sdxl-comfy")
        self.assertIsInstance(img_gen, SdxlComfyImageGenerator)
        self.assertEqual(img_gen.engine_id, "sdxl-comfy")
        self.assertEqual(img_gen.family, "sdxl")
        self.assertEqual(img_gen.service, "comfyui")
        self.assertEqual(img_gen.spatial_stride, 64)

        vid_gen = create_video_generator("hunyuan")
        self.assertIsInstance(vid_gen, HunyuanVideoGenerator)
        self.assertEqual(vid_gen.engine_id, "hunyuan")
        self.assertEqual(vid_gen.family, "hunyuan")
        self.assertEqual(vid_gen.service, "comfyui")
        self.assertEqual(vid_gen.temporal_stride, 4)
        self.assertEqual(vid_gen.spatial_stride, 16)

    def test_sdxl_comfy_resolution_and_workflow(self):
        """Verifies SDXL ComfyUI workflow generation and LoRA chaining."""
        gen = SdxlComfyImageGenerator(name="sdxl-comfy")

        req = ImageRequest(
            prompt="anime warrior girl in cyberpunk city",
            negative_prompt="blurry, bad hands",
            model="sd_xl_base_1.0.safetensors",
            width=1000,   # Not multiple of 64 -> should snap to 960 or 1024
            height=1000,
            steps=28,
            cfg=6.5,
            sampler="Euler a",
            loras=[{"name": "test_style_lora", "strength": 0.75}]
        )

        captured_wf = {}

        def mock_queue(wf):
            nonlocal captured_wf
            captured_wf = wf
            return True, "test_prompt_id"

        def mock_wait(pid, timeout=600, progress_callback=None):
            return True, {
                "outputs": {
                    "9": {
                        "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                    }
                }
            }

        with patch("core.comfy_client.queue_prompt", side_effect=mock_queue), \
             patch("core.comfy_client.wait_for_prompt", side_effect=mock_wait), \
             patch("core.comfy_client.download_output_file", return_value=b"fake_png"), \
             patch("core.discovery._locate_comfy_file", return_value=""):

            res = gen.generate(req)
            self.assertTrue(res.success)
            self.assertEqual(res.metadata["engine"], "sdxl-comfy")
            self.assertEqual(res.metadata["sampler"], "euler_ancestral")
            self.assertEqual(res.metadata["width"] % 64, 0)
            self.assertEqual(res.metadata["height"] % 64, 0)

            # Check workflow structure
            self.assertIn("1", captured_wf)
            self.assertEqual(captured_wf["1"]["class_type"], "CheckpointLoaderSimple")
            self.assertEqual(captured_wf["1"]["inputs"]["ckpt_name"], "sd_xl_base_1.0.safetensors")

            # Check LoRA loader node
            self.assertIn("20", captured_wf)
            self.assertEqual(captured_wf["20"]["class_type"], "LoraLoader")
            self.assertEqual(captured_wf["20"]["inputs"]["lora_name"], "test_style_lora.safetensors")
            self.assertEqual(captured_wf["20"]["inputs"]["strength_model"], 0.75)

            # Positive and negative prompt connected to LoRA clip
            self.assertEqual(captured_wf["2"]["inputs"]["clip"], ["20", 1])
            self.assertEqual(captured_wf["3"]["inputs"]["clip"], ["20", 1])

            # KSampler receives LoRA model
            self.assertEqual(captured_wf["7"]["inputs"]["model"], ["20", 0])
            self.assertEqual(captured_wf["7"]["inputs"]["sampler_name"], "euler_ancestral")
            self.assertEqual(captured_wf["7"]["inputs"]["steps"], 28)
            self.assertEqual(captured_wf["7"]["inputs"]["cfg"], 6.5)

    def test_hunyuan_stride_and_workflow(self):
        """Verifies HunyuanVideo 1.5 workflow, temporal 4k+1 snapping, and SigLIP conditioning."""
        gen = HunyuanVideoGenerator(name="hunyuan")

        # Frame snapping test: 80 frames -> snaps to 81 (4*20 + 1)
        req = VideoRequest(
            prompt="cinematic drone shot through mystical mountains",
            image_path="test_anchor.png",
            model="hunyuanvideo1.5_720p_i2v_fp16.safetensors",
            width=850,    # snap to 16 -> 848
            height=490,   # snap to 16 -> 480
            num_frames=80,
            fps=24,
            steps=35,
            cfg=6.0,
            save_tail=True,
            tail_frames=8,
            save_tail_image=True,
            loras=[{"name": "hunyuan_motion_lora", "strength": 0.8}]
        )

        captured_wf = {}

        def mock_queue(wf):
            nonlocal captured_wf
            captured_wf = wf
            return True, "test_hunyuan_pid"

        def mock_wait(pid, timeout=1800, progress_callback=None):
            return True, {
                "outputs": {
                    "save_video": {
                        "videos": [{"filename": "out.mp4", "subfolder": ""}]
                    }
                }
            }

        with patch("pathlib.Path.is_file", return_value=True), \
             patch("core.comfy_client.upload_file_to_comfy", return_value="uploaded_anchor.png"), \
             patch("core.comfy_client.queue_prompt", side_effect=mock_queue), \
             patch("core.comfy_client.wait_for_prompt", side_effect=mock_wait), \
             patch("core.comfy_client.find_comfy_output_file", return_value=""), \
             patch("core.comfy_client.download_output_file", return_value=b"fake_mp4"), \
             patch("core.discovery._locate_comfy_file", return_value=""):

            res = gen.generate(req)
            self.assertTrue(res.success)

            # Node checks
            self.assertEqual(captured_wf["unet_loader"]["class_type"], "UNETLoader")
            self.assertEqual(captured_wf["clip_loader"]["class_type"], "DualCLIPLoader")
            self.assertEqual(captured_wf["clip_loader"]["inputs"]["type"], "hunyuan_video_15")
            self.assertEqual(captured_wf["clip_vision_loader"]["class_type"], "CLIPVisionLoader")
            self.assertEqual(captured_wf["clip_vision_encode"]["class_type"], "CLIPVisionEncode")

            # HunyuanVideo15ImageToVideo conditioning
            i2v = captured_wf["hunyuan_i2v"]
            self.assertEqual(i2v["class_type"], "HunyuanVideo15ImageToVideo")
            self.assertEqual(i2v["inputs"]["length"], 81)  # 4 * 20 + 1
            self.assertEqual(i2v["inputs"]["width"] % 16, 0)
            self.assertEqual(i2v["inputs"]["height"] % 16, 0)

            # LoRA chaining
            self.assertIn("lora_1", captured_wf)
            self.assertEqual(captured_wf["lora_1"]["class_type"], "LoraLoaderModelOnly")

            # Model sampling shift = 7.0
            self.assertEqual(captured_wf["model_sampling"]["inputs"]["shift"], 7.0)

            # Tail latent and tail image nodes
            self.assertIn("cut_tail_latent", captured_wf)
            self.assertEqual(captured_wf["cut_tail_latent"]["inputs"]["amount"], 8)
            self.assertIn("save_tail_latent", captured_wf)
            self.assertIn("slice_tail_image", captured_wf)
            self.assertEqual(captured_wf["slice_tail_image"]["inputs"]["batch_index"], 80) # 81 - 1
            self.assertIn("save_tail_image", captured_wf)

    def test_schema_validations(self):
        """Verifies schema validation for both new engines."""
        # Valid sdxl-comfy payload
        valid_img = {
            "engine": "sdxl-comfy",
            "model": "sd_xl_base_1.0.safetensors",
            "prompt": "test prompt",
            "negative_prompt": "blurry",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "cfg": 7.0
        }
        schema_validator.validate_mcp_args("image_generate", valid_img)

        # Valid hunyuan scenario payload
        valid_vid = {
            "video": {
                "engine": "hunyuan",
                "model": "hunyuanvideo1.5_720p_i2v_fp16.safetensors",
                "prompt": "drone pan"
            },
            "scenes": [
                {
                    "anchor_image": {
                        "image_path": "test.png"
                    },
                    "blocks": [
                        {
                            "prompt": "motion",
                            "duration": 4.0
                        }
                    ]
                }
            ]
        }
        schema_validator.validate_mcp_args("video_generate", valid_vid)

    def test_image_engine_sdxl_comfy(self):
        """Verifies that unified image_engine.generate_image accepts and routes sdxl-comfy."""
        from core import image_engine

        mock_gen_result = MagicMock()
        mock_gen_result.success = True
        mock_gen_result.image_path = "mock_output.png"
        mock_gen_result.metadata = {"engine": "sdxl-comfy", "seed": 42}

        with patch("core.image_engine.create_image_generator") as mock_create, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("PIL.Image.open") as mock_open:
            mock_gen_instance = MagicMock()
            mock_gen_instance.generate.return_value = mock_gen_result
            mock_create.return_value = mock_gen_instance

            mock_im = MagicMock()
            mock_im.size = (1024, 1024)
            mock_open.return_value.__enter__.return_value = mock_im

            success, path, _, meta = image_engine.generate_image({
                "engine": "sdxl-comfy",
                "model": "sd_xl_base_1.0.safetensors",
                "prompt": "test prompt",
                "width": 1024,
                "height": 1024
            })
            self.assertTrue(success)
            self.assertEqual(meta["engine"], "sdxl-comfy")
            mock_create.assert_called_with("sdxl-comfy", settings=mock_create.call_args[1]["settings"])


if __name__ == "__main__":
    unittest.main()
