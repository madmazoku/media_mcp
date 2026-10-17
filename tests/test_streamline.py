#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated test suite verifying the streamlined media_mcp architecture:
1. Model & LoRA Discovery with metadata trigger extraction and invariant tree shape.
2. Centralized JSON Schema validation and negative constraints.
3. Scenario normalization, scenario_path loading, and timing checks.
4. Unified MCP Server protocol dispatch, tools/list, and tool execution.
"""

import sys
import json
import unittest
from pathlib import Path

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import tempfile
from core import discovery, schema_validator, video_pipeline, config, schema_formatter
from mcp import media_mcp


class TestStreamlineArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmp_dir.name)
        cls._test_config = {
            "output_dir": str(tmp_path / "media_mcp"),
            "comfy_dir": str(tmp_path / "ComfyUI"),
            "comfy_url": "http://127.0.0.1:8188",
            "sd_webui_url": "http://127.0.0.1:7860",
            "media_mcp_port": 7865
        }
        config.load_config_dict(cls._test_config)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()
        config.cfg_json = {}

    def test_01_models_tree_invariant_shape(self):
        """Verify media_get_models returns invariant tree shape."""
        tree = discovery.get_models_tree()
        self.assertIn("image", tree)
        self.assertIn("video", tree)
        self.assertIsInstance(tree["image"], dict)
        self.assertIsInstance(tree["video"], dict)
        self.assertIn("sdxl", tree["image"])
        self.assertIn("flux1", tree["image"])
        self.assertIn("flux2", tree["image"])
        self.assertIn("wan-fast", tree["video"])

        # Filter by type="image"
        img_tree = discovery.get_models_tree(type_filter="image")
        self.assertIn("image", img_tree)
        self.assertNotIn("video", img_tree)

        # Filter by engine="sdxl"
        sdxl_tree = discovery.get_models_tree(engine_filter="sdxl")
        self.assertIn("image", sdxl_tree)
        self.assertIn("sdxl", sdxl_tree["image"])
        self.assertNotIn("flux1", sdxl_tree["image"])

    def test_02_samplers_list(self):
        """Verify media_get_samplers returns list of strings with recommended at index 0."""
        samplers = discovery.get_samplers_for_model("sdxl", "ponyDiffusionV6XL_v6StartWithThis.safetensors")
        self.assertIsInstance(samplers, list)
        self.assertTrue(len(samplers) > 0)
        self.assertEqual(samplers[0], "Euler a")

        flux_samplers = discovery.get_samplers_for_model("flux1", "persephoneFlux.safetensors")
        self.assertIsInstance(flux_samplers, list)
        self.assertEqual(flux_samplers[0], "euler")

    def test_03_loras_triggers(self):
        """Verify media_get_loras returns [{'name': ..., 'triggers': [...]}]"""
        loras = discovery.get_loras_for_model("sdxl", "test.safetensors")
        self.assertIsInstance(loras, list)
        for item in loras:
            self.assertIn("name", item)
            self.assertIn("triggers", item)
            self.assertIsInstance(item["triggers"], list)

    def test_04_schema_validation(self):
        """Verify all MCP tool schemas validate valid and invalid inputs."""
        # image_generate valid
        valid_img = {
            "prompt": "a beautiful forest",
            "engine": "sdxl",
            "cfg": 7.0,
            "loras": [{"name": "test_lora", "strength": 0.8}]
        }
        schema_validator.validate_mcp_args("image_generate", valid_img)

        # image_generate invalid: FLUX.2 cannot have loras
        invalid_flux2 = {
            "prompt": "a beautiful forest",
            "engine": "flux2",
            "loras": [{"name": "test_lora", "strength": 0.8}]
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_args("image_generate", invalid_flux2)

        # video_generate valid
        valid_vid = {
            "scenario_path": "examples/scenario.json",
            "enable_hud": False
        }
        schema_validator.validate_mcp_args("video_generate", valid_vid)

    def test_05_scenario_normalization_and_timing(self):
        """Verify scenario loading via scenario_path and timing validation."""
        req = {
            "scenario_path": "examples/scenario.json"
        }
        scenes = video_pipeline.normalize_scenario_structure(req)
        self.assertTrue(len(scenes) > 0)
        scene = scenes[0]
        self.assertIn("name", scene)
        self.assertIn("anchor_image", scene)
        self.assertIn("blocks", scene)
        self.assertTrue(len(scene["blocks"]) > 0)

        # Verify timing checks
        errors = video_pipeline.collect_scenario_timing_errors(scenes)
        self.assertEqual(len(errors), 0)

    def test_06_mcp_server_protocol(self):
        """Verify JSON-RPC 2.0 initialize, tools/list, and tool execution."""
        # Initialize
        init_res = media_mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_res["result"]["serverInfo"]["name"], "media-mcp")

        # Tools list
        tools_res = media_mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = tools_res["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        expected_tools = [
            "image_generate", "video_generate", "video_validate",
            "media_get_models", "media_get_samplers", "media_get_loras",
            "media_get_status", "media_interrupt", "media_reload_modules"
        ]
        for et in expected_tools:
            self.assertIn(et, tool_names)

        # Call video_validate
        call_val = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "video_validate",
                "arguments": {"scenario_path": "examples/scenario.json"}
            }
        })
        self.assertIn("result", call_val)
        self.assertTrue(call_val["result"]["is_valid"])

        # Call media_reload_modules dev tool
        call_reload = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "media_reload_modules",
                "arguments": {}
            }
        })
        self.assertIn("result", call_reload)
        self.assertIn("reloaded_modules", call_reload["result"])

    def test_07_scenario_timing_failure_cases(self):
        """Verify timing constraints: overlap < duration, prefix + duration < max_duration."""
        invalid_timing_scenes = [
            {
                "name": "Scene with invalid overlap",
                "anchor_image": {"prompt": "test prompt"},
                "blocks": [
                    {
                        "name": "Block 1",
                        "duration": 4.0,
                        "overlap": 5.0,  # overlap >= duration!
                        "motion_prompt": "motion"
                    },
                    {
                        "name": "Block 2",
                        "duration": 4.0,
                        "overlap": 0.0,
                        "motion_prompt": "motion"
                    }
                ]
            }
        ]
        errors = video_pipeline.collect_scenario_timing_errors(invalid_timing_scenes, max_duration=15.0)
        self.assertTrue(any("overlap_duration" in e for e in errors))

    def test_08_media_status_and_interrupt(self):
        """Verify media_get_status and media_interrupt JSON-RPC calls."""
        st_res = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "media_get_status",
                "arguments": {}
            }
        })
        self.assertIn("result", st_res)
        self.assertIn("pipeline", st_res["result"])

        int_res = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "media_interrupt",
                "arguments": {}
            }
        })
        self.assertIn("result", int_res)
        self.assertTrue(int_res["result"]["interrupted"])

    def test_09_scenario_path_resolution(self):
        """Verify video_validate handles scenario_path and deep merges overrides."""
        from core.tools import execute_tool

        res_scenario = execute_tool("video_validate", {"scenario_path": "examples/scenario.json"})
        self.assertFalse(res_scenario.get("isError", False))
        self.assertTrue(res_scenario["is_valid"])
        self.assertGreater(res_scenario["total_duration"], 0)
        self.assertGreater(len(res_scenario["scenes"]), 0)

    def test_10_image_spec_validation(self):
        """Verify image_generate validates strictly against image_spec.json schema."""
        from core import schema_validator

        valid_spec = {
            "prompt": "Hermione Granger in library",
            "engine": "sdxl",
            "seed": 12345
        }
        schema_validator.validate_mcp_args("image_generate", valid_spec)

        invalid_spec = {
            "engine": "sdxl"  # Missing required 'prompt'
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_args("image_generate", invalid_spec)

    def test_11_unified_execute_tool_dispatch(self):
        """Verify execute_tool provides unified transport-agnostic interface across all tools."""
        from core.tools import execute_tool, CANONICAL_TOOLS

        for t in CANONICAL_TOOLS:
            # Call with empty payload or minimal valid payload
            if t in ("video_validate", "video_generate"):
                payload = {"scenario_path": "examples/scenario.json"}
            elif t == "image_generate":
                payload = {"prompt": "Hermione in library"}
            elif t in ("media_get_samplers", "media_get_loras"):
                payload = {"engine": "sdxl", "model": "test.safetensors"}
            else:
                payload = {}

            if t in ("image_generate", "video_generate"):
                # Avoid triggering actual model diffusion during test suite
                continue

            res = execute_tool(t, payload)
            self.assertIsInstance(res, dict)
            self.assertIn("content", res)
            self.assertFalse(res.get("isError", False), f"Tool {t} failed unexpectedly: {res.get('error')}")

    def test_12_loras_pattern_regex_filtering(self):
        """Verify media_get_loras pattern regex filter against names and triggers."""
        from unittest.mock import patch
        from core.tools import execute_tool

        mock_loras = [
            "hermione_granger_v2.safetensors",
            "cyberpunk_neon_city.safetensors",
            "bikini_summer_beach.safetensors",
            "raw_sketch.safetensors"
        ]

        with patch("core.comfy_client.get_loras", return_value=mock_loras):
            # 1. Regex alternation: 'hermione|bikini'
            res = execute_tool("media_get_loras", {
                "engine": "flux1",
                "model": "flux_dev.safetensors",
                "pattern": "hermione|bikini"
            })
            self.assertFalse(res.get("isError", False))
            names = [l["name"] for l in res["loras"]]
            self.assertEqual(len(names), 2)
            self.assertIn("hermione_granger_v2", names)
            self.assertIn("bikini_summer_beach", names)

            # 2. Regex prefix: '^cyber.*'
            res_cyber = execute_tool("media_get_loras", {
                "engine": "flux1",
                "model": "flux_dev.safetensors",
                "pattern": r"^cyber.*"
            })
            self.assertFalse(res_cyber.get("isError", False))
            self.assertEqual(len(res_cyber["loras"]), 1)
            self.assertEqual(res_cyber["loras"][0]["name"], "cyberpunk_neon_city")

            # 3. Regex character class: '[pony]'
            # 'raw_sketch' does not contain p, o, n, or y
            res_char_class = execute_tool("media_get_loras", {
                "engine": "flux1",
                "model": "flux_dev.safetensors",
                "pattern": r"[pony]"
            })
            self.assertFalse(res_char_class.get("isError", False))
            names_class = [l["name"] for l in res_char_class["loras"]]
            self.assertNotIn("raw_sketch", names_class)

            # 4. Invalid regex error handling
            res_invalid = execute_tool("media_get_loras", {
                "engine": "flux1",
                "model": "flux_dev.safetensors",
                "pattern": "(unclosed_parenthesis"
            })
            self.assertTrue(res_invalid.get("isError", False))
            self.assertIn("Invalid regex pattern", res_invalid.get("error", ""))

    def test_13_img2img_spec_schema_and_generator_dispatch(self):
        """Verify image_spec schema and generator behavior for img2img."""
        import os
        from unittest.mock import patch
        from core import schema_validator
        from core.generators.base import ImageRequest
        from core.generators import create_image_generator
        import tempfile

        # 1. Valid img2img spec
        valid_img2img = {
            "prompt": "Hermione smiling portrait",
            "image_path": "path/to/test.png",
            "denoise": 0.6,
            "engine": "sdxl"
        }
        schema_validator.validate_mcp_args("image_generate", valid_img2img)

        # 2. Invalid denoise (out of bounds > 1.0)
        invalid_denoise = {
            "prompt": "test",
            "image_path": "test.png",
            "denoise": 1.5
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_args("image_generate", invalid_denoise)

        # 3. SdxlImageGenerator img2img payload verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "source.png")
            with open(tmp_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nfakeimagebytes")

            req = ImageRequest(
                prompt="test prompt",
                image_path=tmp_path,
                denoise=0.55
            )
            self.assertTrue(req.is_img2img)
            self.assertEqual(req.denoise, 0.55)

            sdxl_gen = create_image_generator("sdxl", settings={})
            with patch("core.generators.image_generators.OUTPUT_IMG_DIR", tmp_dir), \
                 patch("core.a1111_client.img2img") as mock_img2img:
                mock_img2img.return_value = {"images": ["data:image/png;base64,ZmFrZWltYWdl"]}
                res = sdxl_gen.generate(req)
                self.assertTrue(res.success, res.error)
                self.assertEqual(res.metadata.get("mode"), "img2img")
                self.assertEqual(res.metadata.get("denoise"), 0.55)
                # Verify payload sent to /sdapi/v1/img2img
                call_payload = mock_img2img.call_args[0][0]
                self.assertIn("init_images", call_payload)
                self.assertEqual(call_payload["denoising_strength"], 0.55)

            # 4. Flux1ImageGenerator img2img graph verification
            flux_gen = create_image_generator("flux1", settings={})
            with patch("core.generators.image_generators.OUTPUT_IMG_DIR", tmp_dir), \
                 patch("core.comfy_client.copy_to_comfy_input", return_value="comfy_input_test.png"), \
                 patch("core.comfy_client.queue_prompt") as mock_queue, \
                 patch("core.comfy_client.wait_for_prompt") as mock_wait, \
                 patch("core.comfy_client.download_output_file", return_value=b"fakeout"):
                mock_queue.return_value = (True, "prompt_123")
                mock_wait.return_value = (True, {"outputs": {"9": {"images": [{"filename": "out.png"}]}}})

                res_flux = flux_gen.generate(req)
                self.assertTrue(res_flux.success, res_flux.error)
                self.assertEqual(res_flux.metadata.get("mode"), "img2img")
                self.assertEqual(res_flux.metadata.get("denoise"), 0.55)
                # Verify graph queued to ComfyUI has LoadImage and VAEEncode
                graph = mock_queue.call_args[0][0]
                self.assertEqual(graph["6"]["class_type"], "LoadImage")
                self.assertEqual(graph["6"]["inputs"]["image"], "comfy_input_test.png")
                self.assertEqual(graph["10"]["class_type"], "VAEEncode")
                self.assertEqual(graph["7"]["inputs"]["denoise"], 0.55)

    def test_14_scenario_anchor_img2img_and_loras_merge(self):
        """Verify scenario anchor_image schema, img2img refinement, and LoRA list merging."""
        from core import video_pipeline, schema_validator

        scenario_raw = {
            "name": "LoRA and img2img Test Scenario",
            "image": {
                "engine": "sdxl",
                "denoise": 0.65,
                "loras": [
                    {"name": "hermione_v2", "strength": 0.8},
                    {"name": "cinematic_light", "strength": 0.4}
                ]
            },
            "video": {
                "engine": "wan-fast",
                "width": 832,
                "height": 480
            },
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {
                        "prompt": "reading book in cozy chair",
                        "image_path": "path/to/character_base.png",
                        "denoise": 0.45,  # Scene override
                        "loras": [
                            {"name": "hermione_v2", "strength": 0.95},  # Override strength
                            {"name": "vintage_library", "strength": 0.7}  # Add scene-specific LoRA
                        ]
                    },
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "motion_prompt": "turning pages"
                        }
                    ]
                }
            ]
        }

        # 1. Schema validation
        schema_validator.validate_scenario(scenario_raw)

        # 2. Normalization and LoRA merge check
        scenes = video_pipeline.normalize_scenario_structure(scenario_raw)
        self.assertEqual(len(scenes), 1)
        anchor = scenes[0]["anchor_image"]

        # Check denoise inheritance
        self.assertEqual(anchor["denoise"], 0.45)

        # Check LoRA merging
        lora_dict = {l["name"]: l["strength"] for l in anchor["loras"]}
        self.assertEqual(len(lora_dict), 3)
        self.assertEqual(lora_dict["hermione_v2"], 0.95)      # Updated from scene
        self.assertEqual(lora_dict["cinematic_light"], 0.4)    # Kept from global image
        self.assertEqual(lora_dict["vintage_library"], 0.7)    # Added from scene

        # 3. Schema rejection of engine/width/height in anchor_image
        invalid_scene_anchor = dict(scenario_raw)
        invalid_scene_anchor["scenes"] = [
            {
                "name": "Scene with invalid anchor props",
                "anchor_image": {
                    "prompt": "test",
                    "width": 1024  # Not allowed in anchor_image!
                },
                "blocks": [{"duration": 4.0, "motion_prompt": "motion"}]
            }
        ]
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_scene_anchor)

    def test_15_keyframe_path_isolation_and_denoise_normalization(self):
        """Verify keyframe_path isolation in scenario normalization and denoise normalization in ImageRequest."""
        from core import video_pipeline, schema_validator, image_engine
        from core.generators.base import ImageRequest
        import tempfile
        from pathlib import Path

        # 1. ImageRequest normalization
        txt_req = ImageRequest(prompt="pure text prompt", denoise=0.45)
        self.assertFalse(txt_req.is_img2img)
        self.assertEqual(txt_req.denoise, 1.0)  # Automatically normalized to 1.0

        img_req = ImageRequest(prompt="refine", image_path="some/file.png", denoise=0.45)
        self.assertTrue(img_req.is_img2img)
        self.assertEqual(img_req.denoise, 0.45)

        # 2. Scenario schema validation with ready keyframe_path
        valid_scenario = {
            "name": "Keyframe Isolation Scenario",
            "image": {
                "engine": "flux1",
                "prompt": "Global root style prompt that should not contaminate ready keyframe",
                "denoise": 0.65,
                "loras": [{"name": "global_lora", "strength": 0.7}]
            },
            "video": {"engine": "wan-fast"},
            "scenes": [
                {
                    "name": "Scene 1 Ready Keyframe",
                    "anchor_image": {
                        "keyframe_path": "already_rendered_hero.png"
                    },
                    "blocks": [{"motion_prompt": "camera pan"}]
                }
            ]
        }
        schema_validator.validate_scenario(valid_scenario)

        # 3. Validation failure if other properties are attached to keyframe_path
        invalid_keyframe_prompt = {
            "name": "Invalid Keyframe Scenario",
            "scenes": [
                {
                    "anchor_image": {
                        "keyframe_path": "hero.png",
                        "prompt": "should not be allowed"
                    },
                    "blocks": [{"motion_prompt": "zoom"}]
                }
            ]
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_keyframe_prompt)

        invalid_keyframe_denoise = {
            "name": "Invalid Keyframe Denoise Scenario",
            "scenes": [
                {
                    "anchor_image": {
                        "keyframe_path": "hero.png",
                        "denoise": 0.5
                    },
                    "blocks": [{"motion_prompt": "zoom"}]
                }
            ]
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_keyframe_denoise)

        # 4. Pipeline normalization check: ready keyframe must be completely isolated
        normalized_scenes = video_pipeline.normalize_scenario_structure(valid_scenario)
        self.assertEqual(len(normalized_scenes), 1)
        norm_anchor = normalized_scenes[0]["anchor_image"]
        self.assertEqual(norm_anchor, {"keyframe_path": "already_rendered_hero.png"})
        self.assertNotIn("prompt", norm_anchor)
        self.assertNotIn("loras", norm_anchor)
        self.assertNotIn("denoise", norm_anchor)

        # 5. Direct resolution via generate_anchor_frame
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_frame = Path(tmp_dir) / "frame0.png"
            test_frame.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")
            success, resolved_path, b64 = image_engine.generate_anchor_frame({"keyframe_path": str(test_frame)})
            self.assertTrue(success)
            self.assertEqual(resolved_path, str(test_frame.resolve()))
            self.assertIsNotNone(b64)

    def test_16_auto_resolution_and_aspect_ratio_validation(self):
        """Verify automatic native resolution calculation, schema rules, and aspect ratio validation."""
        from core import image_engine, schema_validator, video_pipeline
        from PIL import Image
        import tempfile

        # 1. calculate_native_dimensions verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 16:9 source (1920x1080)
            p16_9 = Path(tmp_dir) / "src_16_9.png"
            Image.new("RGB", (1920, 1080), color=(100, 100, 100)).save(p16_9)
            w16_9, h16_9 = image_engine.calculate_native_dimensions(p16_9)
            self.assertEqual(w16_9 % 16, 0)
            self.assertEqual(h16_9 % 16, 0)
            self.assertAlmostEqual(w16_9 / h16_9, 16 / 9, delta=0.02)
            self.assertTrue(900_000 <= w16_9 * h16_9 <= 1_150_000)

            # 9:16 source (1080x1920)
            p9_16 = Path(tmp_dir) / "src_9_16.png"
            Image.new("RGB", (1080, 1920), color=(100, 100, 100)).save(p9_16)
            w9_16, h9_16 = image_engine.calculate_native_dimensions(p9_16)
            self.assertEqual(w9_16 % 16, 0)
            self.assertEqual(h9_16 % 16, 0)
            self.assertAlmostEqual(w9_16 / h9_16, 9 / 16, delta=0.02)

            # 1:1 source (800x800)
            p1_1 = Path(tmp_dir) / "src_1_1.png"
            Image.new("RGB", (800, 800), color=(100, 100, 100)).save(p1_1)
            w1_1, h1_1 = image_engine.calculate_native_dimensions(p1_1)
            self.assertEqual((w1_1, h1_1), (1024, 1024))

        # 2. Schema validation: width/height forbidden when image_path is present
        valid_img2img = {
            "prompt": "portrait",
            "image_path": "path/to/img.png",
            "denoise": 0.6
        }
        schema_validator.validate_image_spec(valid_img2img)

        invalid_with_width = dict(valid_img2img)
        invalid_with_width["width"] = 1024
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(invalid_with_width)

        invalid_with_height = dict(valid_img2img)
        invalid_with_height["height"] = 1024
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(invalid_with_height)

        invalid_denoise_no_image = {
            "prompt": "txt2img with denoise",
            "denoise": 0.5
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(invalid_denoise_no_image)

        # 3. Video Pipeline Aspect Ratio Validation
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1:1 square frame
            sq_path = Path(tmp_dir) / "square.png"
            Image.new("RGB", (1024, 1024), color=(50, 50, 50)).save(sq_path)

            # 16:9 landscape frame
            ls_path = Path(tmp_dir) / "landscape.png"
            Image.new("RGB", (1280, 720), color=(50, 50, 50)).save(ls_path)

            # Mismatched scenario: 1:1 frame inside 832x480 (16:9) video
            scenes_mismatch = [
                {
                    "name": "Scene Mismatch",
                    "anchor_image": {"keyframe_path": str(sq_path)},
                    "blocks": [{"motion_prompt": "run"}]
                }
            ]
            ar_errors = video_pipeline.collect_scenario_aspect_ratio_errors(
                scenes_mismatch, video_width=832, video_height=480
            )
            self.assertEqual(len(ar_errors), 1)
            self.assertIn("Aspect ratio mismatch", ar_errors[0])
            self.assertIn("1024x1024", ar_errors[0])

            # Matched scenario: 16:9 frame inside 832x480 (16:9) video
            scenes_matched = [
                {
                    "name": "Scene Matched",
                    "anchor_image": {"keyframe_path": str(ls_path)},
                    "blocks": [{"motion_prompt": "run"}]
                }
            ]
            ar_errors_ok = video_pipeline.collect_scenario_aspect_ratio_errors(
                scenes_matched, video_width=832, video_height=480
            )
            self.assertEqual(len(ar_errors_ok), 0)

    def test_17_inpainting_schema_validation_and_generator_dispatch(self):
        """Verify inpainting schema, AR validation, continuous gradient mask preservation, and generator dispatch."""
        from core import image_engine, schema_validator
        from core.generators import create_image_generator, ImageRequest
        from unittest.mock import patch
        from PIL import Image
        import tempfile
        import os

        # 1. Schema validation
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_p = Path(tmp_dir) / "source.png"
            mask_p = Path(tmp_dir) / "mask.png"
            Image.new("RGB", (1280, 720), color=(100, 100, 100)).save(src_p)
            Image.new("L", (1280, 720), color=128).save(mask_p)

            valid_inpaint = {
                "prompt": "change jacket to blue silk",
                "image_path": str(src_p),
                "mask_path": str(mask_p),
                "denoise": 0.8
            }
            schema_validator.validate_image_spec(valid_inpaint)

            # mask_path without image_path must fail schema
            invalid_mask_only = {
                "prompt": "change jacket to blue silk",
                "mask_path": str(mask_p)
            }
            with self.assertRaises(schema_validator.SchemaValidationError):
                schema_validator.validate_image_spec(invalid_mask_only)

            # mask_path with manual width/height must fail schema
            invalid_mask_with_dim = dict(valid_inpaint)
            invalid_mask_with_dim["width"] = 1024
            with self.assertRaises(schema_validator.SchemaValidationError):
                schema_validator.validate_image_spec(invalid_mask_with_dim)

        # 2. Aspect ratio mismatch detection in image_engine.generate_image
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_16_9 = Path(tmp_dir) / "src_16_9.png"
            mask_1_1 = Path(tmp_dir) / "mask_1_1.png"
            Image.new("RGB", (1920, 1080), color=(100, 100, 100)).save(src_16_9)
            Image.new("L", (1000, 1000), color=255).save(mask_1_1)

            ok, out_p, _, meta = image_engine.generate_image({
                "prompt": "test",
                "image_path": str(src_16_9),
                "mask_path": str(mask_1_1)
            })
            self.assertFalse(ok)
            self.assertIn("Aspect ratio mismatch", meta.get("error", ""))

        # 3. Continuous gradient preservation (no binarization!)
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_ok = Path(tmp_dir) / "src.png"
            mask_gradient = Path(tmp_dir) / "gradient_mask.png"
            Image.new("RGB", (1280, 720), color=(100, 100, 100)).save(src_ok)
            # Create a 4-band gradient mask: 0 (black), 64 (dark gray), 128 (mid gray), 255 (white)
            m_img = Image.new("L", (1280, 720), color=0)
            for x in range(320, 640):
                for y in range(720):
                    m_img.putpixel((x, y), 64)
            for x in range(640, 960):
                for y in range(720):
                    m_img.putpixel((x, y), 128)
            for x in range(960, 1280):
                for y in range(720):
                    m_img.putpixel((x, y), 255)
            m_img.save(mask_gradient)

            with patch("core.image_engine.create_image_generator") as mock_create_gen:
                from core.generators.base import GenerationResult
                mock_gen_instance = mock_create_gen.return_value
                mock_gen_instance.generate.return_value = GenerationResult(
                    success=True,
                    image_path=str(src_ok),
                    metadata={"target_width": 1360, "target_height": 768}
                )
                ok, _, _, _ = image_engine.generate_image({
                    "prompt": "inpaint test",
                    "image_path": str(src_ok),
                    "mask_path": str(mask_gradient)
                })
                self.assertTrue(ok)
                # Check req passed to generator
                passed_req: ImageRequest = mock_gen_instance.generate.call_args[0][0]
                self.assertTrue(passed_req.is_inpaint)
                self.assertIsNotNone(passed_req.mask_path)
                # Verify that the prepared mask still contains intermediate gradient values (not binarized to 0/255)
                with Image.open(passed_req.mask_path) as prepared_m:
                    pixels = set(prepared_m.convert("L").getdata())
                    # Ensure intermediate values between 1 and 254 exist
                    intermediate = [v for v in pixels if 20 < v < 235]
                    self.assertTrue(len(intermediate) > 0, "Mask should preserve continuous gradients and not be binarized!")

        # 4. SdxlImageGenerator payload inpaint verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_f = Path(tmp_dir) / "s.png"
            mask_f = Path(tmp_dir) / "m.png"
            src_f.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
            mask_f.write_bytes(b"\x89PNG\r\n\x1a\nfakemask")
            inpaint_req = ImageRequest(
                prompt="inpaint prompt",
                image_path=str(src_f),
                mask_path=str(mask_f),
                denoise=0.75
            )
            self.assertTrue(inpaint_req.is_inpaint)

            sdxl_gen = create_image_generator("sdxl", settings={})
            with patch("core.generators.image_generators.OUTPUT_IMG_DIR", tmp_dir), \
                 patch("core.a1111_client.img2img") as mock_img2img:
                mock_img2img.return_value = {"images": ["data:image/png;base64,ZmFrZQ=="]}
                res = sdxl_gen.generate(inpaint_req)
                self.assertTrue(res.success)
                self.assertEqual(res.metadata.get("mode"), "inpaint")
                payload = mock_img2img.call_args[0][0]
                self.assertIn("mask", payload)
                self.assertIn("init_images", payload)
                self.assertEqual(payload["denoising_strength"], 0.75)
                self.assertFalse(payload["inpaint_full_res"])
                self.assertFalse(payload["mask_round"])
                self.assertFalse(payload["override_settings"]["overlay_inpaint"])

        # 5. Flux1ImageGenerator ComfyUI graph inpaint verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_f = Path(tmp_dir) / "s.png"
            mask_f = Path(tmp_dir) / "m.png"
            src_f.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
            mask_f.write_bytes(b"\x89PNG\r\n\x1a\nfakemask")
            inpaint_req = ImageRequest(
                prompt="flux inpaint",
                image_path=str(src_f),
                mask_path=str(mask_f),
                denoise=0.75
            )

            flux_gen = create_image_generator("flux1", settings={})
            with patch("core.generators.image_generators.OUTPUT_IMG_DIR", tmp_dir), \
                 patch("core.comfy_client.copy_to_comfy_input", side_effect=lambda p: os.path.basename(p)), \
                 patch("core.comfy_client.queue_prompt") as mock_queue, \
                 patch("core.comfy_client.wait_for_prompt") as mock_wait, \
                 patch("core.comfy_client.download_output_file", return_value=b"fakeout"):
                mock_queue.return_value = (True, "prompt_flux_inpaint")
                mock_wait.return_value = (True, {"outputs": {"9": {"images": [{"filename": "out.png"}]}}})

                res_flux = flux_gen.generate(inpaint_req)
                self.assertTrue(res_flux.success)
                self.assertEqual(res_flux.metadata.get("mode"), "inpaint")
                graph = mock_queue.call_args[0][0]
                # Node 12 must be LoadImageMask for mask
                self.assertEqual(graph["12"]["class_type"], "LoadImageMask")
                self.assertEqual(graph["12"]["inputs"]["channel"], "red")
                # Node 13 must be SetLatentNoiseMask
                self.assertEqual(graph["13"]["class_type"], "SetLatentNoiseMask")
                self.assertEqual(graph["13"]["inputs"]["mask"], ["12", 0])
                # Node 7 (KSampler) must consume Node 13 output
                self.assertEqual(graph["7"]["inputs"]["latent_image"], ["13", 0])

        # 6. Flux2ImageGenerator ComfyUI graph inpaint and SplitSigmasDenoise verification
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_f = Path(tmp_dir) / "s.png"
            mask_f = Path(tmp_dir) / "m.png"
            src_f.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
            mask_f.write_bytes(b"\x89PNG\r\n\x1a\nfakemask")
            inpaint_req2 = ImageRequest(
                prompt="flux2 inpaint",
                image_path=str(src_f),
                mask_path=str(mask_f),
                denoise=0.8
            )

            flux2_gen = create_image_generator("flux2", settings={})
            with patch("core.generators.image_generators.OUTPUT_IMG_DIR", tmp_dir), \
                 patch("core.comfy_client.copy_to_comfy_input", side_effect=lambda p: os.path.basename(p)), \
                 patch("core.comfy_client.queue_prompt") as mock_queue2, \
                 patch("core.comfy_client.wait_for_prompt") as mock_wait2, \
                 patch("core.comfy_client.download_output_file", return_value=b"fakeout"):
                mock_queue2.return_value = (True, "prompt_flux2_inpaint")
                mock_wait2.return_value = (True, {"outputs": {"1011": {"images": [{"filename": "out2.png"}]}}})

                res_flux2 = flux2_gen.generate(inpaint_req2)
                self.assertTrue(res_flux2.success)
                self.assertEqual(res_flux2.metadata.get("mode"), "inpaint")
                graph2 = mock_queue2.call_args[0][0]
                # SplitSigmasDenoise must be present
                self.assertEqual(graph2["1026"]["class_type"], "SplitSigmasDenoise")
                self.assertEqual(graph2["1026"]["inputs"]["denoise"], 0.8)
                self.assertEqual(graph2["1021"]["inputs"]["sigmas"], ["1026", 1])
                # LoadImageMask must be present
                self.assertEqual(graph2["1012"]["class_type"], "LoadImageMask")
                self.assertEqual(graph2["1012"]["inputs"]["channel"], "red")
                self.assertEqual(graph2["1013"]["inputs"]["mask"], ["1012", 0])
                self.assertEqual(graph2["1021"]["inputs"]["latent_image"], ["1013", 0])

    def test_18_fit_crop_and_outpaint_modes(self):
        """Verify fit modes ('crop', 'outpaint'), dimension rules, feathered masks, and AR bypass."""
        from core import schema_validator, image_engine, video_pipeline
        from PIL import Image
        import numpy as np
        import tempfile
        from unittest.mock import patch

        # 1. Schema validation rules for fit
        # Fit requires image_path, width, height
        spec_fit_no_img = {"prompt": "test", "fit": "crop", "width": 1024, "height": 1024}
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(spec_fit_no_img)

        # Fit requires width and height
        spec_fit_no_dims = {"prompt": "test", "image_path": "a.png", "fit": "crop"}
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(spec_fit_no_dims)

        # Image_path without fit bans width/height
        spec_img_manual_dims = {"prompt": "test", "image_path": "a.png", "width": 1024, "height": 1024}
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec(spec_img_manual_dims)

        # Valid crop spec
        spec_valid_crop = {"prompt": "test", "image_path": "a.png", "fit": "crop", "width": 1024, "height": 1024}
        schema_validator.validate_image_spec(spec_valid_crop)

        # Valid outpaint spec
        spec_valid_outpaint = {"prompt": "test", "image_path": "a.png", "fit": "outpaint", "width": 1360, "height": 768}
        schema_validator.validate_image_spec(spec_valid_outpaint)

        # 2. fit='crop' execution
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_16_9 = Path(tmp_dir) / "source_16_9.png"
            Image.new("RGB", (1920, 1080), color=(120, 130, 140)).save(src_16_9)

            with patch("core.image_engine.create_image_generator") as mock_gen_factory:
                from core.generators.base import GenerationResult
                mock_gen = mock_gen_factory.return_value
                mock_gen.generate.return_value = GenerationResult(
                    success=True,
                    image_path=str(src_16_9),
                    metadata={"width": 1024, "height": 1024}
                )

                ok, out_p, _, meta = image_engine.generate_image({
                    "prompt": "crop test",
                    "image_path": str(src_16_9),
                    "fit": "crop",
                    "width": 1024,
                    "height": 1024
                })
                self.assertTrue(ok)
                passed_req = mock_gen.generate.call_args[0][0]
                self.assertEqual(passed_req.width, 1024)
                self.assertEqual(passed_req.height, 1024)
                self.assertEqual(passed_req.fit, "crop")
                # Source image must be cropped to 1024x1024
                with Image.open(passed_req.image_path) as c_im:
                    self.assertEqual(c_im.size, (1024, 1024))

        # 3. fit='outpaint' execution with auto-generated feathered mask
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_square = Path(tmp_dir) / "source_1_1.png"
            Image.new("RGB", (1000, 1000), color=(80, 90, 100)).save(src_square)

            with patch("core.image_engine.create_image_generator") as mock_gen_factory:
                mock_gen = mock_gen_factory.return_value
                mock_gen.generate.return_value = GenerationResult(
                    success=True,
                    image_path=str(src_square),
                    metadata={"width": 1360, "height": 768}
                )

                ok, _, _, meta = image_engine.generate_image({
                    "prompt": "outpaint test",
                    "image_path": str(src_square),
                    "fit": "outpaint",
                    "width": 1360,
                    "height": 768
                })
                self.assertTrue(ok)
                passed_req = mock_gen.generate.call_args[0][0]
                self.assertEqual(passed_req.width, 1360)
                self.assertEqual(passed_req.height, 768)
                self.assertEqual(passed_req.fit, "outpaint")
                self.assertTrue(passed_req.is_inpaint)
                self.assertIsNotNone(passed_req.mask_path)

                # Check padded image dimensions
                with Image.open(passed_req.image_path) as p_im:
                    self.assertEqual(p_im.size, (1360, 768))

                # Check outpaint mask: dimensions, padding (255), center (0), and feathering
                with Image.open(passed_req.mask_path) as m_im:
                    self.assertEqual(m_im.size, (1360, 768))
                    m_arr = np.array(m_im)
                    # Outer left/right border must be 255 (new content)
                    self.assertEqual(m_arr[384, 0], 255)
                    self.assertEqual(m_arr[384, -1], 255)
                    # Inner center must be 0 (preserved content)
                    self.assertEqual(m_arr[384, 680], 0)
                    # Transition values must exist due to Gaussian feathering
                    unique_vals = set(m_arr.flatten())
                    intermediate = [v for v in unique_vals if 20 < v < 235]
                    self.assertTrue(len(intermediate) > 0, "Outpaint mask must have smooth feathered transitions")

        # 4. Aspect ratio validation bypass when fit is specified
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_mismatch = Path(tmp_dir) / "src_mismatch.png"
            Image.new("RGB", (1000, 1000), color=(100, 100, 100)).save(src_mismatch)

            # Scene without fit (1000x1000 vs 832x480) -> must report AR error
            scenes_no_fit = [{
                "scene_name": "Scene 1",
                "anchor_image": {"image_path": str(src_mismatch)}
            }]
            errors = video_pipeline.collect_scenario_aspect_ratio_errors(scenes_no_fit, video_width=832, video_height=480)
            self.assertEqual(len(errors), 1)

            # Scene with fit='outpaint' -> must bypass AR error
            scenes_with_fit = [{
                "scene_name": "Scene 1",
                "anchor_image": {"image_path": str(src_mismatch), "fit": "outpaint"}
            }]
            errors_bypassed = video_pipeline.collect_scenario_aspect_ratio_errors(scenes_with_fit, video_width=832, video_height=480)
            self.assertEqual(len(errors_bypassed), 0)

    def test_19_strict_config_enforcement(self):
        """Verify strict config check rejects operations when unconfigured."""
        saved_cfg = dict(config.cfg_json)
        try:
            config.cfg_json = {}
            self.assertFalse(config.is_configured())
            with self.assertRaises(RuntimeError):
                config.require_configured("Test operation")

            from core.tools import execute_tool
            res = execute_tool("video_validate", {"scenario_path": "examples/scenario.json"})
            self.assertTrue(res.get("isError"))
            self.assertIn("requires an environment configuration file", res.get("error", ""))
        finally:
            config.load_config_dict(saved_cfg)

    def test_20_response_schema_validation(self):
        """Verify response schemas validate valid payloads and reject invalid payloads."""
        # 1. Valid image_generate response (flat)
        valid_img_res = {
            "run_prefix": "20261017_123456_abcdef",
            "image_path": "G:/media_mcp/images/test.png",
            "spec_path": "G:/media_mcp/workflows/test-spec.json",
            "engine": "flux2",
            "model": "flux2-klein-9b.safetensors",
            "seed": 123456,
            "width": 1024,
            "height": 1024
        }
        schema_validator.validate_mcp_response("image_generate", valid_img_res)

        # 2. Invalid image_generate response: missing required properties
        invalid_img_res = {
            "run_prefix": "20261017_123456_abcdef"
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_response("image_generate", invalid_img_res)

        # 3. Invalid image_generate response: additional unexpected property
        invalid_extra_prop = dict(valid_img_res)
        invalid_extra_prop["unexpected_extra_field"] = "bad"
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_response("image_generate", invalid_extra_prop)

        # 4. Valid video_generate response (flat, without hud)
        valid_vid_res = {
            "run_prefix": "20261017_123456_abcdef",
            "video_path": "G:/media_mcp/videos/test.mp4",
            "poster_path": "G:/media_mcp/images/test_poster.png",
            "spec_path": "G:/media_mcp/workflows/test-spec.json",
            "duration": 10.5,
            "width": 832,
            "height": 480,
            "fps": 16,
            "scenes_count": 1,
            "blocks_count": 2,
            "hud_video_path": None
        }
        schema_validator.validate_mcp_response("video_generate", valid_vid_res)

        # 5. Valid video_generate response (flat, with hud string)
        valid_vid_with_hud = dict(valid_vid_res)
        valid_vid_with_hud["hud_video_path"] = "G:/media_mcp/videos/test_hud.mp4"
        schema_validator.validate_mcp_response("video_generate", valid_vid_with_hud)

        # 6. Invalid video_generate response: missing blocks_count
        invalid_vid = dict(valid_vid_res)
        del invalid_vid["blocks_count"]
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_response("video_generate", invalid_vid)

    def test_21_schema_formatter_help(self):
        """Verify schema_formatter outputs formatted terminal help."""
        # 1. General help
        gen_help = schema_formatter.format_general_help()
        self.assertIn("media_mcp", gen_help)
        self.assertIn("image_generate", gen_help)
        self.assertIn("video_generate", gen_help)
        self.assertIn("video_validate", gen_help)

        # 2. Tool help
        tool_help = schema_formatter.format_tool_help("image_generate")
        self.assertIn("image_generate", tool_help)
        self.assertIn("prompt", tool_help)
        self.assertIn("engine", tool_help)
        self.assertIn("sdxl", tool_help)
        self.assertIn("INPUT PARAMETERS", tool_help)
        self.assertIn("OUTPUT FORMAT", tool_help)

    def test_22_cli_interface(self):
        """Verify CLI entry point behavior and strict input resolution."""
        import subprocess
        # 1. Test cli --help
        res_help = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_help.returncode, 0)
        self.assertIn("AVAILABLE TOOLS:", res_help.stdout)

        # 2. Test cli <tool> --help
        res_tool_help = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "video_validate", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_tool_help.returncode, 0)
        self.assertIn("TOOL: video_validate", res_tool_help.stdout)

        # 3. Test cli execution with raw inline JSON when configured (must fail fast)
        res_raw_json = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "--config", "examples/config.json", "video_validate", '{"scenario_path": "x"}'],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_raw_json.returncode, 1)
        self.assertIn("Direct inline JSON strings are not accepted", res_raw_json.stderr)

    def test_23_request_and_response_schemas_symmetry(self):
        """Verify strict 1-to-1 symmetry between requests/<tool>.json and responses/<tool>.json for all tools."""
        from core.tools import CANONICAL_TOOLS
        for tool in CANONICAL_TOOLS:
            req_schema = schema_validator.get_mcp_tool_schema(tool)
            self.assertIsInstance(req_schema, dict)
            self.assertIn("type", req_schema)

            resp_schema = schema_validator.get_mcp_response_schema(tool)
            self.assertIsInstance(resp_schema, dict)
            self.assertIn("type", resp_schema)

    def test_24_video_pipeline_img2img_anchor_temp_dir(self):
        """Verify video_pipeline handles img2img anchor without NameError on TEMP_DIR."""
        from core import video_pipeline, config
        from PIL import Image
        from unittest.mock import patch
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            out_vid = tmp_p / "videos"
            out_img = tmp_p / "images"
            out_wf = tmp_p / "workflows"
            out_tmp = tmp_p / "temp"
            for d in [out_vid, out_img, out_wf, out_tmp]:
                d.mkdir(parents=True, exist_ok=True)

            orig_vid = config.OUTPUT_VID_DIR
            orig_img = config.OUTPUT_IMG_DIR
            orig_wf = config.WORKFLOWS_DIR
            orig_tmp = config.TEMP_DIR
            config.OUTPUT_VID_DIR = out_vid.resolve().as_posix()
            config.OUTPUT_IMG_DIR = out_img.resolve().as_posix()
            config.WORKFLOWS_DIR = out_wf.resolve().as_posix()
            config.TEMP_DIR = out_tmp.resolve().as_posix()

            try:
                src_img = tmp_p / "anchor_src.png"
                Image.new("RGB", (832, 480), color=(10, 20, 30)).save(src_img)

                scenario = {
                    "name": "Test Img2Img Anchor Scenario",
                    "video": {"engine": "wan-fast", "width": 832, "height": 480},
                    "scenes": [
                        {
                            "name": "Scene 1",
                            "anchor_image": {
                                "image_path": str(src_img),
                                "prompt": "refined anchor"
                            },
                            "blocks": [
                                {"motion_prompt": "slow pan", "duration": 2.0}
                            ]
                        }
                    ]
                }

                with patch("core.image_engine.generate_image") as mock_gen_img, \
                     patch("core.video_pipeline.create_video_generator") as mock_vid_factory, \
                     patch("core.ffmpeg_engine.concat_videos") as mock_concat, \
                     patch("core.ffmpeg_engine.get_duration") as mock_dur:

                    from core.generators.base import GenerationResult
                    mock_gen_img.return_value = (True, str(src_img), "base64dummy", {"run_prefix": "dummy", "spec_path": "dummy.json"})
                    mock_vg = mock_vid_factory.return_value
                    mock_vg.name = "wan-fast"
                    mock_vg.generate.return_value = GenerationResult(
                        success=True,
                        file_path=str(src_img)
                    )
                    mock_dur.return_value = 2.0

                    res = video_pipeline.execute_scenario_pipeline(scenario)
                    self.assertIn("video_path", res)
                    self.assertIn("poster_path", res)
            finally:
                config.OUTPUT_VID_DIR = orig_vid
                config.OUTPUT_IMG_DIR = orig_img
                config.WORKFLOWS_DIR = orig_wf
                config.TEMP_DIR = orig_tmp

    def test_25_comfy_client_queue_tracking_and_keep_alive(self):
        """Verify comfy_client active queue tracking, fast fail on dropped prompt, and Keep-Alive session."""
        from core import comfy_client, a1111_client
        from unittest.mock import patch

        # 1. Verify Keep-Alive session properties
        c_sess = comfy_client.get_session()
        self.assertFalse(c_sess.trust_env, "ComfyUI session must not read system proxy env vars")
        a_sess = a1111_client.get_session()
        self.assertFalse(a_sess.trust_env, "A1111 session must not read system proxy env vars")

        # 2. Test wait_for_prompt transitioning: queue_pending -> queue_running -> history
        call_seq = {"step": 0}

        def mock_make_req(endpoint, data=None, method="GET", raw=False, timeout=30):
            if endpoint.startswith("/history"):
                if call_seq["step"] >= 2:
                    return {"test_pid": {"status": {"completed": True}, "outputs": {"9": {"images": ["ok.png"]}}}}
                return {}
            elif endpoint == "/queue":
                if call_seq["step"] == 0:
                    call_seq["step"] = 1
                    return {"queue_running": [], "queue_pending": [[1, "test_pid"]]}
                elif call_seq["step"] == 1:
                    call_seq["step"] = 2
                    return {"queue_running": [[1, "test_pid"]], "queue_pending": []}
                return {"queue_running": [], "queue_pending": []}
            return {}

        notifs = []

        def mock_cb(msg, prog, tot):
            notifs.append(msg)

        with patch("core.comfy_client.make_request", side_effect=mock_make_req):
            ok, hist = comfy_client.wait_for_prompt("test_pid", timeout=10, poll_interval=0.01, progress_callback=mock_cb)
            self.assertTrue(ok)
            self.assertIn("outputs", hist)

        # 3. Test prompt dropped from queue without history (fail-fast)
        def mock_dropped_req(endpoint, data=None, method="GET", raw=False, timeout=30):
            if endpoint.startswith("/history") or endpoint == "/queue":
                return {}
            return {}

        with patch("core.comfy_client.make_request", side_effect=mock_dropped_req), \
             patch("time.time", side_effect=[0.0, 6.0, 7.0, 8.0, 9.0]):
            ok, err = comfy_client.wait_for_prompt("dropped_pid", timeout=10, poll_interval=0.01)
            self.assertFalse(ok)
            self.assertIn("disappeared from queue", err["error"])

    def test_26_gpu_serialization_lock(self):
        """Verify _GPU_LOCK serializes concurrent generations and reports status in media_get_status."""
        from core.tools import _GPU_LOCK, execute_tool

        # When GPU lock is free
        status_idle = execute_tool("media_get_status", {})
        self.assertIn("IDLE", status_idle["report"])

        # When GPU lock is held
        with _GPU_LOCK:
            status_busy = execute_tool("media_get_status", {})
            self.assertIn("BUSY", status_busy["report"])


if __name__ == "__main__":
    unittest.main()
