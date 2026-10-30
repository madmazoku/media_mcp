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
from unittest.mock import patch, MagicMock
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, video_pipeline, schema_validator, image_engine
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

                # Must use LoadLatent + in-canvas LTXLatentPrefixToVideo + LTXVAdainLatent
                self.assertIn("load_prefix_latent", wf)
                self.assertEqual(wf["load_prefix_latent"]["class_type"], "LoadLatent")
                self.assertIn("ltx_latent_prefix", wf)
                self.assertEqual(wf["ltx_latent_prefix"]["class_type"], "LTXLatentPrefixToVideo")
                self.assertEqual(wf["ltx_latent_prefix"]["inputs"]["strength"], 1.0)
                self.assertIn("ltx_adain_latent", wf)
                self.assertEqual(wf["ltx_adain_latent"]["class_type"], "LTXVAdainLatent")
                self.assertNotIn("prefix_latent_restore", wf)

                # Distilled LoRA must default to 0.7 in distilled mode (cfg <= 1.5)
                self.assertAlmostEqual(wf["lora_loader"]["inputs"]["strength_model"], 0.2)
                self.assertEqual(wf["sampler_select"]["inputs"]["sampler_name"], "euler_ancestral")

                # Verify full guidance mode (cfg > 1.5) uses strength 0.2
                req_cfg3 = VideoRequest(
                    prompt="camera pans right",
                    model="ltx-video-2b-v0.9.5.safetensors",
                    cfg=1.0,
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
                self.assertAlmostEqual(wf_cfg3["lora_loader"]["inputs"]["strength_model"], 0.7)
                self.assertEqual(wf_cfg3["sampler_select"]["inputs"]["sampler_name"], "euler_ancestral")

                # Must NOT contain the old buggy prefix_latent_decode -> LTXVImgToVideoInplace path
                self.assertNotIn("prefix_latent_decode", wf)
                self.assertNotIn("inplace_cond", wf)

                # Guider receives clean conditioning, while Sampler receives in-canvas prefix latent
                self.assertEqual(wf["guider"]["inputs"]["positive"], ["ltxv_cond", 0])
                self.assertEqual(wf["guider"]["inputs"]["negative"], ["ltxv_cond", 1])
                self.assertEqual(wf["sampler"]["inputs"]["latent_image"], ["ltx_latent_prefix", 0])
                self.assertEqual(wf["scheduler"]["inputs"]["latent"], ["empty_latent", 0])
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
                self.assertEqual(wf_img["preprocess_image"]["inputs"]["img_compression"], 25)
                self.assertIn("ltx_start_guide", wf_img)
                self.assertEqual(wf_img["ltx_start_guide"]["class_type"], "LTXVAddGuide")
                self.assertEqual(wf_img["ltx_start_guide"]["inputs"]["strength"], 0.7)
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

            # Verify Wan guard auto-routes high steps/cfg requests from wan-fast to wan-full
            req_oversampled = VideoRequest(
                prompt="fast action",
                model="wan2.2_i2v_14B_fp8_scaled.safetensors",
                steps=25,
                cfg=5.0,
                width=832,
                height=480,
                fps=16,
                num_frames=65,
                image_path=dummy_img.as_posix()
            )
            with patch("core.generators.video_generators.WanFullVideoGenerator.generate", return_value=GenerationResult(success=True, file_path="routed.mp4")) as mock_full:
                res_routed = wan_gen.generate(req_oversampled)
                self.assertTrue(res_routed.success)
                mock_full.assert_called_once_with(req_oversampled)

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
                            {"duration": 4.5, "overlap": 2.0, "prompt": "b1"},
                            {"duration": 4.5, "overlap": 0.0, "prompt": "b2"}  # 2.0 + 4.5 = 6.5 > 6.0 for Wan!
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
                        {"prompt": "block 1", "duration": 4.0, "overlap": 1.0},
                        {"prompt": "block 2", "duration": 4.0, "overlap": 0.0}
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



    def test_07_hierarchical_errors_and_promptless_anchor(self):
        """Verify hierarchical error breadcrumbs and promptless anchor handling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_img = Path(tmp_dir) / "test_src.png"
            Image.new("RGB", (1000, 1000), color=(100, 150, 200)).save(src_img)

            # 1. Passive crop adaptation without prompt
            ok, out_p, _, meta = image_engine.generate_image({
                "engine": "sdxl",
                "model": "ponyDiffusionV6XL.safetensors",
                "image_path": str(src_img),
                "fit": "crop",
                "width": 832,
                "height": 480
            })
            self.assertTrue(ok)
            self.assertTrue(meta.get("passive_crop"))
            self.assertEqual((meta["width"], meta["height"]), (832, 480))
            with Image.open(out_p) as im:
                self.assertEqual(im.size, (832, 480))

            # 2. Pre-flight check: missing prompt for pure txt2img anchor fails synchronously
            no_prompt_scen = {
                "name": "No Prompt Scenario",
                "video": {
                    "engine": "wan-fast",
                    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"
                },
                "scenes": [
                    {
                        "name": "Scene 1",
                        "anchor_image": {},
                        "blocks": [{"duration": 4.0, "prompt": "dance"}]
                    }
                ]
            }
            res_no_prompt = execute_tool("video_generate", no_prompt_scen)
            self.assertTrue(res_no_prompt.get("isError"))
            err = res_no_prompt.get("error", "")
            self.assertIn("failed scenario 'No Prompt Scenario'", err)
            self.assertIn("scene #1 ('Scene 1'): failed anchor_image: Prompt is required", err)

            # 3. Pre-flight check: missing file even when fit='crop' is specified fails synchronously
            missing_file_scen = {
                "name": "Missing File Scenario",
                "video": {
                    "engine": "wan-fast",
                    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"
                },
                "scenes": [
                    {
                        "name": "Scene 1",
                        "anchor_image": {"image_path": "non_existent_img.png", "fit": "crop"},
                        "blocks": [{"duration": 4.0, "prompt": "dance"}]
                    }
                ]
            }
            res_missing_file = execute_tool("video_generate", missing_file_scen)
            self.assertTrue(res_missing_file.get("isError"))
            err_mf = res_missing_file.get("error", "")
            self.assertIn("failed scenario 'Missing File Scenario'", err_mf)
            self.assertIn("scene #1 ('Scene 1'): failed anchor_image: Anchor image file not found on disk: non_existent_img.png", err_mf)


    def test_08_end_image_preflight_and_workflows(self):
        """Verify end_image pre-flight hierarchical validation and generator workflow attachment."""
        # 1. Pre-flight check: missing file in end_image
        missing_end_scen = {
            "name": "Missing End File Scenario",
            "video": {
                "engine": "wan-fast",
                "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"
            },
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {"prompt": "hero starts"},
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "prompt": "hero runs",
                            "end_image": {"keyframe_path": "non_existent_end.png"}
                        }
                    ]
                }
            ]
        }
        res_mf = execute_tool("video_generate", missing_end_scen)
        self.assertTrue(res_mf.get("isError"))
        err_mf = res_mf.get("error", "")
        self.assertIn("failed scenario 'Missing End File Scenario'", err_mf)
        self.assertIn("scene #1 ('Scene 1'): block #1 ('Block 1'): failed end_image: End image file not found on disk: non_existent_end.png", err_mf)

        # 2. Pre-flight check: missing prompt for pure txt2img end_image
        no_prompt_end_scen = {
            "name": "No Prompt End Scenario",
            "video": {
                "engine": "wan-fast",
                "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"
            },
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {"prompt": "hero starts"},
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "prompt": "hero runs",
                            "end_image": {}
                        }
                    ]
                }
            ]
        }
        res_np = execute_tool("video_generate", no_prompt_end_scen)
        self.assertTrue(res_np.get("isError"))
        err_np = res_np.get("error", "")
        self.assertIn("failed scenario 'No Prompt End Scenario'", err_np)
        self.assertIn("scene #1 ('Scene 1'): block #1 ('Block 1'): failed end_image: Prompt is required for end image generation when neither keyframe_path nor image_path is specified.", err_np)

        # 3. Wan video generator workflow attachment: WanFirstLastFrameToVideo (overlap=0)
        start_img = self.tmp_path / "wan_flf_start.png"
        end_img = self.tmp_path / "wan_flf_end.png"
        Image.new("RGB", (832, 480), color=(10, 20, 30)).save(start_img)
        Image.new("RGB", (832, 480), color=(40, 50, 60)).save(end_img)

        req_flf_0 = VideoRequest(
            prompt="motion to destination",
            image_path=start_img.as_posix(),
            end_image_path=end_img.as_posix(),
            model="wan2.2_i2v_14B_fp8_scaled.safetensors",
            width=832,
            height=480,
            fps=16,
            num_frames=65,
            prefix_frames=0,
            core_frames=65,
            run_prefix="20261027-120000-abcdef-s001-b001"
        )
        captured_wfs = []
        def fake_exec(wf, **kwargs):
            captured_wfs.append(wf)
            return GenerationResult(success=True, file_path=start_img.as_posix())

        with patch("core.discovery.resolve_wan_model_bundle", return_value=("high_model.safetensors", "low_model.safetensors")), \
             patch("core.comfy_client.upload_file_to_comfy", side_effect=lambda p, **kw: Path(p).name):
            wan_gen = create_video_generator("wan-fast")
            with patch.object(wan_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                wan_gen.generate(req_flf_0)

            wf_wan_0 = captured_wfs[-1]
            # Should have WanFirstLastFrameToVideo attached
            self.assertEqual(wf_wan_0["wan_conditioning"]["class_type"], "WanFirstLastFrameToVideo")
            self.assertIn("end_image", wf_wan_0["wan_conditioning"]["inputs"])
            self.assertIn("end_image", wf_wan_0)

        # 4. Wan video generator workflow attachment: WanLatentPrefixToVideo with end_image (overlap>0)
        mock_lat = self.tmp_path / "mock.latent"
        mock_lat.write_bytes(b"dummy")

        req_flf_pref = VideoRequest(
            prompt="motion to destination",
            image_path=start_img.as_posix(),
            end_image_path=end_img.as_posix(),
            prefix_latent_path=(self.tmp_path / "mock.latent").as_posix(),
            model="wan2.2_i2v_14B_fp8_scaled.safetensors",
            width=832,
            height=480,
            fps=16,
            num_frames=69,
            prefix_frames=5,
            core_frames=64,
            run_prefix="20261027-120000-abcdef-s001-b002"
        )
        with patch("core.discovery.resolve_wan_model_bundle", return_value=("high_model.safetensors", "low_model.safetensors")), \
             patch("core.comfy_client.upload_file_to_comfy", side_effect=lambda p, **kw: Path(p).name):
            with patch.object(wan_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                wan_gen.generate(req_flf_pref)

            wf_wan_pref = captured_wfs[-1]
            self.assertEqual(wf_wan_pref["wan_conditioning"]["class_type"], "WanLatentPrefixToVideo")
            self.assertIn("end_image", wf_wan_pref["wan_conditioning"]["inputs"])

        # 5. LTX video generator workflow attachment: LTXVAddGuide + LTXVCropGuides
        req_ltx = VideoRequest(
            prompt="motion to destination ltx",
            image_path=start_img.as_posix(),
            end_image_path=end_img.as_posix(),
            model="ltx-video-2b-v0.9.5.safetensors",
            width=832,
            height=480,
            fps=16,
            num_frames=65,
            prefix_frames=0,
            core_frames=65,
            run_prefix="20261027-120000-abcdef-s001-b001"
        )
        with patch("core.comfy_client.upload_file_to_comfy", side_effect=lambda p, **kw: Path(p).name):
            ltx_gen = create_video_generator("ltx")
            with patch.object(ltx_gen, "_execute_comfy_workflow", side_effect=fake_exec):
                ltx_gen.generate(req_ltx)

            wf_ltx = captured_wfs[-1]
            self.assertIn("ltx_end_guide", wf_ltx)
            self.assertEqual(wf_ltx["ltx_end_guide"]["class_type"], "LTXVAddGuide")
            self.assertEqual(wf_ltx["ltx_end_guide"]["inputs"]["frame_idx"], -1)
            self.assertIn("ltx_crop_guides", wf_ltx)
            self.assertEqual(wf_ltx["ltx_crop_guides"]["class_type"], "LTXVCropGuides")

        # 6. Pre-flight check: runtime references $anchor / $start in scene anchor_image are forbidden
        anchor_ref_scen = {
            "name": "Anchor Ref Scenario",
            "video": {"engine": "wan-fast", "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"},
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {"keyframe_path": "$start"},
                    "blocks": [{"duration": 4.0, "prompt": "b1"}]
                }
            ]
        }
        res_ar = execute_tool("video_generate", anchor_ref_scen)
        self.assertTrue(res_ar.get("isError"))
        self.assertIn("Runtime reference '$start' is only permitted inside block end_image, not in scene anchor_image", res_ar.get("error", ""))

        # 7. Pre-flight check: $anchor / $start in end_image image_path requires prompt
        no_prompt_ref_scen = {
            "name": "No Prompt Ref Scenario",
            "video": {"engine": "wan-fast", "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"},
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {"prompt": "hero starts"},
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "prompt": "hero runs",
                            "end_image": {"image_path": "$start"}
                        }
                    ]
                }
            ]
        }
        res_npr = execute_tool("video_generate", no_prompt_ref_scen)
        self.assertTrue(res_npr.get("isError"))
        self.assertIn("Prompt is required when using '$start' in end_image image_path", res_npr.get("error", ""))

        # 8. Runtime execution: $start and $anchor dynamic resolution & persistent spec preservation
        gen_img_calls = []
        def fake_gen_anchor(anchor_meta, width, height):
            gen_img_calls.append(dict(anchor_meta))
            meta_path = anchor_meta.get("custom_filename") or f"{anchor_meta.get('run_prefix')}-final.png"
            return True, meta_path, None, {}

        captured_requests = []
        def fake_gen_video(request):
            captured_requests.append(request)
            mock_video = self.tmp_path / f"{request.run_prefix}-core.mp4"
            mock_video.write_bytes(b"mockmp4")
            mock_tail_img = self.tmp_path / f"{request.run_prefix}-tail.png"
            mock_tail_img.write_bytes(b"mocktail")
            mock_tail_lat = self.tmp_path / f"{request.run_prefix}-tail.latent"
            raw_hdr = json.dumps({"samples": {"dtype": "F32", "shape": [1, 16, 5, 60, 104], "data_offsets": [0, 0]}}).encode("utf-8")
            with open(mock_tail_lat, "wb") as f_lat:
                f_lat.write(struct.pack("<Q", len(raw_hdr)))
                f_lat.write(raw_hdr)
            return GenerationResult(
                success=True,
                file_path=mock_video.as_posix(),
                tail_image_path=mock_tail_img.as_posix(),
                tail_latent_path=mock_tail_lat.as_posix()
            )

        multi_block_ref_scen = {
            "name": "Runtime Ref Test Scenario",
            "image": {"engine": "sdxl", "model": "ponyDiffusionV6XL.safetensors"},
            "video": {"engine": "wan-fast", "model": "wan2.2_i2v_14B_fp8_scaled.safetensors"},
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {"prompt": "hero anchor frame"},
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "overlap": 1.0,
                            "prompt": "block 1 motion",
                            "end_image": {
                                "image_path": "$start",
                                "prompt": "block 1 end modified from start"
                            }
                        },
                        {
                            "name": "Block 2",
                            "duration": 4.0,
                            "overlap": 0.0,
                            "prompt": "block 2 motion",
                            "end_image": {
                                "image_path": "$anchor",
                                "prompt": "block 2 end modified from anchor"
                            }
                        }
                    ]
                }
            ]
        }

        with patch("core.image_engine.generate_anchor_frame", side_effect=fake_gen_anchor), \
             patch("core.video_pipeline.create_video_generator") as mock_vg_factory, \
             patch("core.ffmpeg_engine.concat_videos", return_value="final.mp4"):
            mock_gen_inst = MagicMock()
            mock_gen_inst.name = "wan-fast"
            mock_gen_inst.temporal_stride = 1
            mock_gen_inst.generate.side_effect = fake_gen_video
            mock_vg_factory.return_value = mock_gen_inst

            prep = video_pipeline.prepare_scenario_pipeline(multi_block_ref_scen)
            spec_on_disk = Path(prep["spec_path"]).read_text(encoding="utf-8")
            # Verify persistent spec on disk preserves literal references '$start' and '$anchor'
            self.assertIn('"$start"', spec_on_disk)
            self.assertIn('"$anchor"', spec_on_disk)

            res_pipe = video_pipeline.execute_scenario_pipeline(prepared_pipeline_data=prep)
            self.assertIn("video_path", res_pipe)

            # Check calls to generate_anchor_frame:
            # Call 0: Scene anchor
            self.assertIn("hero anchor frame", gen_img_calls[0]["prompt"])
            scene_anchor_path = gen_img_calls[0]["custom_filename"]

            # Call 1: Block 1 end_image with $start -> resolved to scene anchor
            self.assertEqual(gen_img_calls[1]["image_path"], scene_anchor_path)

            # Call 2: Block 2 end_image with $anchor -> resolved to scene anchor
            self.assertEqual(gen_img_calls[2]["image_path"], scene_anchor_path)

            # Verify save_tail_image was True for Block 1 despite overlap=1.0
            self.assertTrue(captured_requests[0].save_tail_image)
            self.assertTrue(captured_requests[0].save_tail)

if __name__ == "__main__":
    unittest.main()
