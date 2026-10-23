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
        """Verify LtxVideoGenerator uses pure 5D LTXVAddLatentGuide + LTXVCropGuides + LTXVAdainLatent and Wan uses Stage1/Stage2 AdaIN."""
        dummy_prefix = self.tmp_path / "dummy-prefix.latent"
        dummy_prefix.write_bytes(b"mocklatent")

        req_prefix = VideoRequest(
            prompt="camera pans right",
            model="ltx-video-2b-v0.9.5.safetensors",
            width=832,
            height=480,
            fps=24,
            num_frames=65,
            prefix_frames=17,
            prefix_latent_path=dummy_prefix.as_posix(),
            loras=[{"name": "ltx_cinematic", "strength": 0.75}],
            run_prefix="20261017-120000-abcdef-s001-b002"
        )

        captured_wfs = []

        def fake_exec(wf, **kwargs):
            captured_wfs.append(wf)
            return GenerationResult(success=True, file_path=dummy_prefix.as_posix())

        with patch("core.comfy_client.upload_file_to_comfy", return_value="20261017-120000-abcdef-s001-b002-prefix.latent"), \
             patch("core.discovery.resolve_wan_model_bundle", return_value=("high_model.safetensors", "low_model.safetensors")):
            ltx_gen = create_video_generator("ltx")
            with patch.object(ltx_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                ltx_gen.generate(req_prefix)
                wf = captured_wfs[-1]

                # Must use LoadLatent + LTXVAddLatentGuide + LTXVCropGuides + LTXVAdainLatent (without WanLatentPrefixRestore overwriting t=0)
                self.assertIn("load_prefix_latent", wf)
                self.assertEqual(wf["load_prefix_latent"]["class_type"], "LoadLatent")
                self.assertIn("ltx_latent_guide", wf)
                self.assertEqual(wf["ltx_latent_guide"]["class_type"], "LTXVAddLatentGuide")
                self.assertEqual(wf["ltx_latent_guide"]["inputs"]["strength"], 1.0)
                self.assertIn("ltx_crop_guides", wf)
                self.assertEqual(wf["ltx_crop_guides"]["class_type"], "LTXVCropGuides")
                self.assertIn("ltx_adain_latent", wf)
                self.assertEqual(wf["ltx_adain_latent"]["class_type"], "LTXVAdainLatent")
                self.assertNotIn("prefix_latent_restore", wf)

                # Distilled LoRA must default to 0.7 in distilled mode (cfg <= 1.5)
                self.assertAlmostEqual(wf["lora_loader"]["inputs"]["strength_model"], 0.7)
                self.assertEqual(wf["sampler_select"]["inputs"]["sampler_name"], "euler_ancestral")

                # Verify full guidance mode (cfg > 1.5) uses strength 0.2
                req_cfg3 = VideoRequest(
                    prompt="camera pans right",
                    model="ltx-video-2b-v0.9.5.safetensors",
                    cfg=3.0,
                    width=832,
                    height=480,
                    fps=24,
                    num_frames=65,
                    prefix_frames=17,
                    prefix_latent_path=dummy_prefix.as_posix(),
                    run_prefix="20261017-120000-abcdef-s001-b002"
                )
                ltx_gen.generate(req_cfg3)
                wf_cfg3 = captured_wfs[-1]
                self.assertAlmostEqual(wf_cfg3["lora_loader"]["inputs"]["strength_model"], 0.2)
                self.assertEqual(wf_cfg3["sampler_select"]["inputs"]["sampler_name"], "euler")

                # Must NOT contain the old buggy prefix_latent_decode -> LTXVImgToVideoInplace path
                self.assertNotIn("prefix_latent_decode", wf)
                self.assertNotIn("inplace_cond", wf)

                # Guider and Sampler must receive guided conditioning, while LTXVScheduler receives empty_latent
                self.assertEqual(wf["guider"]["inputs"]["positive"], ["ltx_latent_guide", 0])
                self.assertEqual(wf["guider"]["inputs"]["negative"], ["ltx_latent_guide", 1])
                self.assertEqual(wf["scheduler"]["inputs"]["latent"], ["empty_latent", 0])
                self.assertEqual(wf["sampler"]["inputs"]["latent_image"], ["ltx_latent_guide", 2])
                self.assertEqual(wf["vae_decode"]["inputs"]["samples"], ["ltx_adain_latent", 0])

                # Verify LTX 1-frame image mode uses LTXVPreprocess (CRF=33) + LTXVImgToVideoInplace (strength=0.9)
                dummy_img = self.tmp_path / "dummy_start.png"
                dummy_img.write_bytes(b"mockimg")
                req_img = VideoRequest(
                    prompt="camera orbit",
                    model="ltx-video-2b-v0.9.5.safetensors",
                    image_path=dummy_img.as_posix(),
                    width=832,
                    height=480,
                    fps=24,
                    num_frames=65,
                    run_prefix="20261017-120000-abcdef-s001-b001"
                )
                with patch("core.comfy_client.upload_file_to_comfy", return_value="uploaded.png"):
                    ltx_gen.generate(req_img)
                wf_img = captured_wfs[-1]
                self.assertIn("preprocess_image", wf_img)
                self.assertEqual(wf_img["preprocess_image"]["class_type"], "LTXVPreprocess")
                self.assertEqual(wf_img["preprocess_image"]["inputs"]["img_compression"], 33)
                self.assertIn("inplace_cond", wf_img)
                self.assertEqual(wf_img["inplace_cond"]["class_type"], "LTXVImgToVideoInplace")
                self.assertEqual(wf_img["inplace_cond"]["inputs"]["strength"], 0.9)
                self.assertEqual(wf_img["sampler_select"]["inputs"]["sampler_name"], "euler_ancestral")

            # Also verify WanFast with prefix_latent_path wires Stage 1 and Stage 2 LTXVAdainLatent nodes
            wan_gen = create_video_generator("wan-fast")
            with patch.object(wan_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                wan_gen.generate(req_prefix)
            wf_wan = captured_wfs[-1]
            self.assertIn("stage1_adain_core", wf_wan)
            self.assertEqual(wf_wan["stage1_adain_core"]["class_type"], "LTXVAdainLatent")
            self.assertIn("stage2_adain_core", wf_wan)
            self.assertEqual(wf_wan["stage2_adain_core"]["class_type"], "LTXVAdainLatent")
            self.assertEqual(wf_wan["sampler_stage2"]["inputs"]["latent_image"], ["stage1_restored", 0])
            self.assertEqual(wf_wan["prefix_latent_restore"]["inputs"]["samples"], ["stage2_concat", 0])

    def test_04_video_generate_synchronous_validation_and_job_lifecycle(self):
        """Verify video_generate pre-flight validation rejects timing errors synchronously, supports dry_run, and queues valid jobs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            scen_file = Path(tmp_dir) / "custom_engine_dur.json"
            scen_data = {
                "name": "Engine Limits Scenario",
                "video": {
                    "engine": "wan-fast",
                    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"
                },
                "scenes": [
                    {
                        "name": "Scene 1",
                        "anchor_image": {"keyframe_path": "hero.png"},
                        "blocks": [
                            {"duration": 4.5, "overlap": 2.0, "motion_prompt": "b1"},
                            {"duration": 4.5, "overlap": 0.0, "motion_prompt": "b2"}  # 2.0 + 4.5 = 6.5 > 6.0 for Wan!
                        ]
                    }
                ]
            }
            scen_file.write_text(json.dumps(scen_data), encoding="utf-8")

            # Wan-fast: 6.5s >= 6.0s -> pre-flight synchronously fails with isError=True
            res = execute_tool("video_generate", {"scenario_path": str(scen_file)})
            self.assertTrue(res.get("isError"))
            self.assertIn("engine limit (6.0s for engine 'wan-fast')", res.get("error", ""))

            # LTX: 6.5s < 16.0s
            hero_path = Path(tmp_dir) / "hero.png"
            Image.new("RGB", (832, 480), color=(0, 0, 0)).save(hero_path)
            scen_data["video"]["engine"] = "ltx"
            scen_data["scenes"][0]["anchor_image"]["keyframe_path"] = hero_path.as_posix()
            scen_file.write_text(json.dumps(scen_data), encoding="utf-8")

            # Dry-run validation only (instant, no GPU job queued)
            res_dry = execute_tool("video_generate", {"scenario_path": str(scen_file), "dry_run": True})
            self.assertFalse(res_dry.get("isError", False))
            self.assertEqual(res_dry["status"], "validated")
            self.assertTrue(res_dry["is_valid"])
            self.assertEqual(res_dry["total_duration"], 7.0)
            self.assertEqual(res_dry["scenes_count"], 1)
            self.assertEqual(res_dry["blocks_count"], 2)

            def fake_exec(prepared_pipeline_data=None, progress_callback=None, **kw):
                return {
                    "video_path": "mock.mp4", "poster_path": "mock.png", "width": 832, "height": 480,
                    "duration": 6.5, "fps": 16, "scenes_count": 1, "blocks_count": 2, "anchor_metadata": {}
                }

            with patch("core.video_pipeline.execute_scenario_pipeline", side_effect=fake_exec), \
                 patch("core.comfy_client.interrupt", return_value=None):
                res_ltx = execute_tool("video_generate", {"scenario_path": str(scen_file)})
                self.assertFalse(res_ltx.get("isError", False))
                self.assertEqual(res_ltx["status"], "queued")
                self.assertTrue(bool(res_ltx["run_prefix"]))

                # Verify media_get_job retrieves queued job
                job_info = execute_tool("media_get_job", {"run_prefix": res_ltx["run_prefix"], "wait": 0})
                self.assertIn(job_info["status"], ("queued", "running", "completed"))
                self.assertEqual(job_info["run_prefix"], res_ltx["run_prefix"])

                # Verify media_remove_job purges or cancels the job
                del_info = execute_tool("media_remove_job", {"run_prefix": res_ltx["run_prefix"]})
                self.assertIn(del_info["status"], ("cancelled", "removed"))

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

    def test_06_wan_latent_prefix_restore_latent_adain(self):
        """Verify WanLatentPrefixRestore normalizes core frame channel stats via Latent AdaIN."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed in test environment")

        from unittest.mock import MagicMock
        for mod_name in ("comfy", "comfy.model_management", "comfy.utils", "comfy.latent_formats", "node_helpers"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = MagicMock()

        import importlib.util
        node_file = Path(__file__).resolve().parent.parent / "custom_nodes" / "ComfyUI-MediaMCP" / "wan_latent_nodes.py"
        spec = importlib.util.spec_from_file_location("wan_latent_nodes", node_file)
        wan_nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wan_nodes)

        node = wan_nodes.WanLatentPrefixRestore()
        # Create mock prefix latent with specific mean (+1.5) and std (0.5)
        prefix_tensor = torch.randn(1, 16, 5, 16, 16) * 0.5 + 1.5
        # Create core latent with drifted mean (-2.0) and blown-out std (3.0)
        core_tensor = torch.randn(1, 16, 17, 16, 16) * 3.0 - 2.0

        out_tuple = node.execute(
            samples={"samples": core_tensor},
            prefix_latent={"samples": prefix_tensor},
            color_lock_strength=1.0
        )
        out_latent = out_tuple[0]["samples"]

        # 1. Prefix frames (0..4) are restored bit-exact
        self.assertTrue(torch.allclose(out_latent[:, :, :5], prefix_tensor, atol=1e-5))

        # 2. Core frames (5..16) have their mean and std normalized to prefix stats
        core_out = out_latent[:, :, 5:]
        core_out_mean = core_out.mean(dim=(2, 3, 4))
        core_out_std = core_out.std(dim=(2, 3, 4))

        self.assertTrue(torch.allclose(core_out_mean, torch.full_like(core_out_mean, 1.5), atol=0.15))
        self.assertTrue(torch.allclose(core_out_std, torch.full_like(core_out_std, 0.5), atol=0.15))


if __name__ == "__main__":
    unittest.main()
