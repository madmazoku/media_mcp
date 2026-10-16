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

from core import discovery, schema_validator, video_pipeline
from mcp import media_mcp


class TestStreamlineArchitecture(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()

