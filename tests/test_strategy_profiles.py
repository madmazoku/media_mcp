#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite: Strategy-driven model classification and auxiliary component profile resolution.
Verifies:
- Elimination of procedural magic numbers in architecture detection.
- Encapsulation of model signatures and profiles in Generator strategy classes.
- Dynamic text encoder resolution for FLUX.2 Dev (Mistral 3) vs FLUX.2 Klein 9B/4B (Qwen 3).
- Strict separation between FLUX.1 Dev and FLUX.2 models.
- Accurate workflow construction in Flux2ImageGenerator without hardcoded text encoders.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.generators.base import BaseGenerator, ImageRequest
from core.generators.image_generators import (
    SdxlImageGenerator,
    Flux1ImageGenerator,
    Flux2ImageGenerator,
)
from core.generators.video_generators import (
    BaseWanVideoGenerator,
    LtxVideoGenerator,
)
from core.generators.factory import get_model_strategies
from core import lora_inspector, discovery


class TestStrategyProfiles(unittest.TestCase):

    def test_01_strategy_registry(self):
        """Verify generator strategies are registered and queryable."""
        strategies = get_model_strategies()
        strategy_names = [s.__name__ for s in strategies]
        self.assertIn("Flux2ImageGenerator", strategy_names)
        self.assertIn("Flux1ImageGenerator", strategy_names)
        self.assertIn("BaseWanVideoGenerator", strategy_names)
        self.assertIn("LtxVideoGenerator", strategy_names)
        self.assertIn("SdxlImageGenerator", strategy_names)

    def test_02_flux2_profile_resolution_from_header(self):
        """Verify Flux2ImageGenerator resolves correct text encoder and hyperparameters from safetensors header."""
        # 1. FLUX.2 Dev (dim 15360 -> Mistral 3, 28 steps)
        h_dev = {
            "txt_in.weight": {"shape": [6144, 15360]},
            "img_in.weight": {"shape": [6144, 128]},
            "double_blocks.0.img_attn.qkv.weight": {"shape": [18432, 6144]}
        }
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("some_model.safetensors", header=h_dev), "dev")
        comps_dev = Flux2ImageGenerator.resolve_components("some_model.safetensors", header=h_dev)
        self.assertEqual(comps_dev["text_encoder"], "mistral_3_small_flux2_bf16.safetensors")
        self.assertEqual(comps_dev["vae"], "flux2-vae.safetensors")
        self.assertEqual(comps_dev["clip_type"], "flux2")
        self.assertEqual(comps_dev["steps"], 28)

        # 2. FLUX.2 Klein 9B (dim 12288 -> Qwen 3 8B, 4 steps)
        h_9b = {
            "txt_in.weight": {"shape": [4096, 12288]},
            "img_in.weight": {"shape": [4096, 128]},
            "double_blocks.0.img_attn.qkv.weight": {"shape": [12288, 4096]}
        }
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("some_model.safetensors", header=h_9b), "klein_9b")
        comps_9b = Flux2ImageGenerator.resolve_components("some_model.safetensors", header=h_9b)
        self.assertEqual(comps_9b["text_encoder"], "qwen_3_8b_fp8mixed.safetensors")
        self.assertEqual(comps_9b["steps"], 4)

        # 3. FLUX.2 Klein 4B (dim 7680 -> Qwen 3 4B, 4 steps)
        h_4b = {
            "txt_in.weight": {"shape": [3072, 7680]},
            "img_in.weight": {"shape": [3072, 128]},
            "double_blocks.0.img_attn.qkv.weight": {"shape": [9216, 3072]}
        }
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("some_model.safetensors", header=h_4b), "klein_4b")
        comps_4b = Flux2ImageGenerator.resolve_components("some_model.safetensors", header=h_4b)
        self.assertEqual(comps_4b["text_encoder"], "qwen_3_4b.safetensors")
        self.assertEqual(comps_4b["steps"], 4)

    def test_03_flux2_profile_resolution_from_filename(self):
        """Verify filename heuristics resolve appropriate profile when header is absent."""
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("flux2_dev_fp8mixed.safetensors"), "dev")
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("flux-2-klein-4b.safetensors"), "klein_4b")
        self.assertEqual(Flux2ImageGenerator.resolve_profile_key("flux-2-klein-9b-fp8.safetensors"), "klein_9b")

    def test_04_flux1_vs_flux2_classification(self):
        """Verify FLUX.1 Dev is NEVER classified as flux2 and vice versa."""
        # Real FLUX.1 Dev header: txt_in is 4096, img_in is 64, double_blocks.18 exists
        h_flux1 = {
            "txt_in.weight": {"shape": [3072, 4096]},
            "img_in.weight": {"shape": [3072, 64]},
            "double_blocks.0.img_attn.qkv.weight": {"shape": [9216, 3072]},
            "double_blocks.18.img_attn.qkv.weight": {"shape": [9216, 3072]}
        }
        self.assertTrue(Flux1ImageGenerator.matches_model(h_flux1, "flux1-dev.safetensors"))
        self.assertFalse(Flux2ImageGenerator.matches_model(h_flux1, "flux1-dev.safetensors"))
        self.assertEqual(lora_inspector.detect_model_architecture_from_header(h_flux1, "flux1-dev.safetensors"), "flux1")

        # Real FLUX.2 Dev header: txt_in is 15360, img_in is 128
        h_flux2_dev = {
            "txt_in.weight": {"shape": [6144, 15360]},
            "img_in.weight": {"shape": [6144, 128]},
            "double_blocks.0.img_attn.qkv.weight": {"shape": [18432, 6144]}
        }
        self.assertFalse(Flux1ImageGenerator.matches_model(h_flux2_dev, "flux2_dev_fp8mixed.safetensors"))
        self.assertTrue(Flux2ImageGenerator.matches_model(h_flux2_dev, "flux2_dev_fp8mixed.safetensors"))
        self.assertEqual(lora_inspector.detect_model_architecture_from_header(h_flux2_dev, "flux2_dev_fp8mixed.safetensors"), "flux2")

    def test_05_flux2_generate_injects_correct_text_encoder_workflow(self):
        """Verify Flux2ImageGenerator.generate injects the resolved Mistral 3 encoder for FLUX.2 Dev."""
        gen = Flux2ImageGenerator(name="flux2")

        queued_wfs = []
        def mock_queue_prompt(wf):
            queued_wfs.append(wf)
            return True, "test_prompt_id"

        mock_history = {
            "outputs": {
                "1011": {
                    "images": [{"filename": "out.png", "subfolder": "", "type": "output"}]
                }
            }
        }

        with patch("core.comfy_client.queue_prompt", side_effect=mock_queue_prompt), \
             patch("core.comfy_client.wait_for_prompt", return_value=(True, mock_history)), \
             patch("core.comfy_client.download_output_file", return_value=b"fake_png"), \
             patch("core.discovery._locate_comfy_file", return_value=""):

            # 1. Request with FLUX.2 Dev
            req_dev = ImageRequest(
                prompt="a beautiful landscape",
                model="flux2_dev_fp8mixed.safetensors"
            )
            res_dev = gen.generate(req_dev)
            self.assertTrue(res_dev.success)
            self.assertEqual(len(queued_wfs), 1)
            wf_dev = queued_wfs[0]
            # Node 1002 is CLIPLoader
            clip_loader_dev = wf_dev["1002"]
            self.assertEqual(clip_loader_dev["inputs"]["clip_name"], "mistral_3_small_flux2_bf16.safetensors")
            self.assertEqual(clip_loader_dev["inputs"]["type"], "flux2")

            # 2. Request with FLUX.2 Klein 4B
            queued_wfs.clear()
            req_4b = ImageRequest(
                prompt="cyberpunk city",
                model="flux-2-klein-4b.safetensors"
            )
            res_4b = gen.generate(req_4b)
            self.assertTrue(res_4b.success)
            self.assertEqual(len(queued_wfs), 1)
            wf_4b = queued_wfs[0]
            clip_loader_4b = wf_4b["1002"]
            self.assertEqual(clip_loader_4b["inputs"]["clip_name"], "qwen_3_4b.safetensors")

            # 3. Request with explicit settings override
            queued_wfs.clear()
            gen_custom = Flux2ImageGenerator(
                name="flux2",
                settings={"text_encoder": "custom_override_encoder.safetensors"}
            )
            req_custom = ImageRequest(
                prompt="test prompt",
                model="flux2_dev_fp8mixed.safetensors"
            )
            res_custom = gen_custom.generate(req_custom)
            self.assertTrue(res_custom.success)
            wf_custom = queued_wfs[0]
            self.assertEqual(wf_custom["1002"]["inputs"]["clip_name"], "custom_override_encoder.safetensors")


if __name__ == "__main__":
    unittest.main()
