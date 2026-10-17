#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite 4: Multi-Scene Video Pipeline, Pure 5D LTX Prefix Lock & Video LoRAs.
Covers:
- Scenario normalization, scenario_path loading, and local-first prompt combination (D7)
- Root-level technical parameters inheritance from .video into VideoRequest (D1)
- WanFast and WanFull workflow generation with bundle resolution and dynamic LoraLoaderModelOnly chaining (D5)
- LtxVideoGenerator pure 5D latent prefix lock (WanLatentPrefixLock + WanLatentPrefixRestore, no VAEDecode) (D5)
- Replayable *-spec.json in workflows/ validating strictly against schemas/scenario.json (D6)
- Timing validation respecting .video.max_duration and aspect ratio validation
"""

import sys
import struct
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, video_pipeline, schema_validator
from core.generators import create_video_generator, VideoRequest, GenerationResult
from core.tools import execute_tool


class TestVideoPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp_dir.name)
        cls._test_config = {
            "output_dir": str(cls.tmp_path / "media_mcp"),
            "comfy_dir": str(cls.tmp_path / "ComfyUI"),
            "comfy_url": "http://127.0.0.1:8188",
            "sd_webui_url": "http://127.0.0.1:7860",
            "media_mcp_port": 7865
        }
        config.load_config_dict(cls._test_config)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()
        config.cfg_json = {}

    def test_01_prompt_combination_and_lora_merge(self):
        """Verify local specificity comes first in combine_prompts (D7) and LoRAs merge properly."""
        combined = video_pipeline.combine_prompts("global style basis", "local block action")
        self.assertEqual(combined, "local block action\nglobal style basis")

        merged_loras = video_pipeline.merge_lora_lists(
            [{"name": "char_lora", "strength": 0.7}, {"name": "light_lora", "strength": 0.5}],
            [{"name": "char_lora", "strength": 0.95}, {"name": "motion_lora", "strength": 0.8}]
        )
        lora_map = {item["name"]: item["strength"] for item in merged_loras}
        self.assertEqual(lora_map, {"char_lora": 0.95, "light_lora": 0.5, "motion_lora": 0.8})

    def test_02_wan_fast_and_wan_full_workflow_and_loras(self):
        """Verify WanFast and WanFull resolve model bundles and chain user LoRAs on high/low branches."""
        start_img = self.tmp_path / "wan_start.png"
        Image.new("RGB", (832, 480), color=(20, 40, 60)).save(start_img)

        req = VideoRequest(
            prompt="she walks forward",
            image_path=start_img.as_posix(),
            model="wan2.2_i2v_14B_fp8_scaled.safetensors",
            width=832,
            height=480,
            fps=16,
            num_frames=65,
            loras=[{"name": "motion_boost", "strength": 0.85}],
            run_prefix="20261017-120000-abcdef-s001-b001"
        )

        captured_wfs = []

        def fake_exec(wf, **kwargs):
            captured_wfs.append(wf)
            return GenerationResult(success=True, file_path=start_img.as_posix())

        with patch("core.discovery.resolve_wan_model_bundle", return_value=("high_model.safetensors", "low_model.safetensors")), \
             patch("core.comfy_client.upload_file_to_comfy", return_value="input.png"):
            wan_fast = create_video_generator("wan-fast")
            with patch.object(wan_fast, "_execute_comfy_workflow", side_effect=fake_exec):
                wan_fast.generate(req)
            wf_fast = captured_wfs[-1]
            self.assertEqual(wf_fast["unet_high"]["inputs"]["unet_name"], "high_model.safetensors")
            self.assertEqual(wf_fast["unet_low"]["inputs"]["unet_name"], "low_model.safetensors")
            self.assertIn("lora_high", wf_fast)
            self.assertIn("lora_low", wf_fast)
            self.assertIn("user_lora_high_0", wf_fast)
            self.assertIn("user_lora_low_0", wf_fast)
            self.assertEqual(wf_fast["user_lora_high_0"]["inputs"]["strength_model"], 0.85)

            wan_full = create_video_generator("wan-full")
            with patch.object(wan_full, "_execute_comfy_workflow", side_effect=fake_exec):
                wan_full.generate(req)
            wf_full = captured_wfs[-1]
            self.assertNotIn("lora_high", wf_full)  # No 4-step LightX2V LoRA in wan-full
            self.assertIn("user_lora_high_0", wf_full)
            self.assertIn("user_lora_low_0", wf_full)

    def test_03_ltx_pure_5d_latent_prefix_lock_and_loras(self):
        """Verify LtxVideoGenerator uses pure 5D WanLatentPrefixLock + WanLatentPrefixRestore without VAEDecode."""
        dummy_prefix = self.tmp_path / "dummy-prefix.latent"
        dummy_prefix.write_bytes(b"mocklatent")

        req_prefix = VideoRequest(
            prompt="camera pans right",
            model="ltx-video-2b-v0.9.5.safetensors",
            width=832,
            height=480,
            fps=16,
            num_frames=65,
            prefix_frames=16,
            prefix_latent_path=dummy_prefix.as_posix(),
            loras=[{"name": "ltx_cinematic", "strength": 0.75}],
            run_prefix="20261017-120000-abcdef-s001-b002"
        )

        captured_wfs = []

        def fake_exec(wf, **kwargs):
            captured_wfs.append(wf)
            return GenerationResult(success=True, file_path=dummy_prefix.as_posix())

        with patch("core.comfy_client.upload_file_to_comfy", return_value="20261017-120000-abcdef-s001-b002-prefix.latent"):
            ltx_gen = create_video_generator("ltx")
            with patch.object(ltx_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                ltx_gen.generate(req_prefix)
            wf = captured_wfs[-1]

            # Must use LoadLatent + WanLatentPrefixLock + WanLatentPrefixRestore
            self.assertIn("load_prefix_latent", wf)
            self.assertEqual(wf["load_prefix_latent"]["class_type"], "LoadLatent")
            self.assertIn("prefix_lock", wf)
            self.assertEqual(wf["prefix_lock"]["class_type"], "WanLatentPrefixLock")
            self.assertIn("prefix_latent_restore", wf)
            self.assertEqual(wf["prefix_latent_restore"]["class_type"], "WanLatentPrefixRestore")

            # Must NOT contain the old buggy prefix_latent_decode -> LTXVImgToVideoInplace path
            self.assertNotIn("prefix_latent_decode", wf)
            self.assertNotIn("inplace_cond", wf)

            # User LoRA must be chained before prefix_lock
            self.assertIn("user_lora_0", wf)
            self.assertEqual(wf["prefix_lock"]["inputs"]["model"], ["user_lora_0", 0])
            self.assertEqual(wf["guider"]["inputs"]["model"], ["prefix_lock", 0])
            self.assertEqual(wf["vae_decode"]["inputs"]["samples"], ["prefix_latent_restore", 0])

    def test_04_video_validate_respects_scenario_path_max_duration(self):
        """Verify execute_video_validate resolves scenario_path BEFORE reading .video.max_duration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scen_file = Path(tmp_dir) / "custom_max_dur.json"
            scen_data = {
                "name": "Short Max Duration Scenario",
                "video": {
                    "engine": "wan-fast",
                    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors",
                    "max_duration": 6.0
                },
                "scenes": [
                    {
                        "name": "Scene 1",
                        "anchor_image": {"keyframe_path": "hero.png"},
                        "blocks": [
                            {"duration": 4.5, "overlap": 2.0, "motion_prompt": "b1"},
                            {"duration": 4.5, "overlap": 0.0, "motion_prompt": "b2"}  # 2.0 + 4.5 = 6.5 > 6.0!
                        ]
                    }
                ]
            }
            scen_file.write_text(json.dumps(scen_data), encoding="utf-8")

            res = execute_tool("video_validate", {"scenario_path": str(scen_file)})
            self.assertFalse(res["is_valid"])
            self.assertTrue(any("max_duration" in e for e in res["errors"]))

    def test_05_execute_scenario_pipeline_saves_replayable_spec_and_reads_5d_latent(self):
        """Verify execute_scenario_pipeline saves a schema-valid *-spec.json and reads 5D latent without torch."""
        src_img = self.tmp_path / "anchor_src.png"
        Image.new("RGB", (832, 480), color=(10, 20, 30)).save(src_img)

        # Create a mock 5D safetensors latent file with T=5 frames
        tail_latent = self.tmp_path / "mock-tail.latent"
        raw_header = json.dumps({
            "samples": {"dtype": "F32", "shape": [1, 16, 5, 60, 104], "data_offsets": [0, 0]}
        }).encode("utf-8")
        with open(tail_latent, "wb") as f:
            f.write(struct.pack("<Q", len(raw_header)))
            f.write(raw_header)

        scenario = {
            "name": "Replayable Spec Test",
            "image": {
                "engine": "sdxl",
                "model": "ponyDiffusionV6XL.safetensors"
            },
            "video": {
                "engine": "wan-fast",
                "model": "wan2.2_i2v_14B_fp8_scaled.safetensors",
                "width": 832,
                "height": 480,
                "fps": 16
            },
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {
                        "image_path": src_img.as_posix(),
                        "prompt": "refined anchor"
                    },
                    "blocks": [
                        {"motion_prompt": "block 1", "duration": 4.0, "overlap": 1.0},
                        {"motion_prompt": "block 2", "duration": 4.0, "overlap": 0.0}
                    ]
                }
            ]
        }

        with patch("core.image_engine.generate_image") as mock_gen_img, \
             patch("core.video_pipeline.create_video_generator") as mock_vid_factory, \
             patch("core.ffmpeg_engine.concat_videos"), \
             patch("core.ffmpeg_engine.get_duration", return_value=8.0):

            mock_gen_img.return_value = (
                True, src_img.as_posix(), "", {"run_prefix": "dummy", "spec_path": "dummy.json", "seed": 777}
            )
            mock_vg = mock_vid_factory.return_value
            mock_vg.name = "wan-fast"
            mock_vg.generate.side_effect = [
                GenerationResult(
                    success=True,
                    file_path=src_img.as_posix(),
                    tail_latent_path=tail_latent.as_posix()
                ),
                GenerationResult(
                    success=True,
                    file_path=src_img.as_posix()
                )
            ]

            res = video_pipeline.execute_scenario_pipeline(scenario)
            self.assertIn("video_path", res)
            self.assertIn("spec_path", res)
            self.assertNotIn("hud_video_path", res)

            # Verify saved *-spec.json is 100% valid against schemas/scenario.json (replayable!)
            saved_spec = json.loads(Path(res["spec_path"]).read_text(encoding="utf-8"))
            schema_validator.validate_scenario(saved_spec)
            self.assertIn("seed", saved_spec["video"])
            self.assertGreater(saved_spec["video"]["seed"], 0)
            self.assertIn("seed", saved_spec["image"])
            self.assertGreater(saved_spec["image"]["seed"], 0)


if __name__ == "__main__":
    unittest.main()
