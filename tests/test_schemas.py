#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite 1: JSON Schema Contracts, Example Validation & Terminal Help Formatting.
Covers:
- 1-to-1 symmetry between requests/<tool>.json and responses/<tool>.json
- Validation of all examples/*.json against canonical schemas
- Mandatory 'engine' and 'model' enforcement across image_generate and scenario.json (D1)
- Canonical video engine enum ['wan-fast', 'wan-full', 'ltx'] (D2)
- FLUX.2 LoRA support in image_generate schema (D3)
- Root-only technical parameters in scenario.json (.image and .video) (D1)
- Response schema validation (native_width, native_height, warnings, no HUD) (D4, D8)
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, schema_validator, schema_formatter
from core.tools import CANONICAL_TOOLS


class TestSchemasAndContracts(unittest.TestCase):

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

    def test_01_request_and_response_schemas_symmetry(self):
        """Verify strict 1-to-1 symmetry between requests/<tool>.json and responses/<tool>.json for all 9 tools."""
        self.assertEqual(len(CANONICAL_TOOLS), 9)
        for tool in CANONICAL_TOOLS:
            req_schema = schema_validator.get_mcp_tool_schema(tool)
            self.assertIsInstance(req_schema, dict)
            self.assertEqual(req_schema.get("type"), "object")

            resp_schema = schema_validator.get_mcp_response_schema(tool)
            self.assertIsInstance(resp_schema, dict)
            self.assertEqual(resp_schema.get("type"), "object")

    def test_02_all_repository_examples_validate(self):
        """Verify every example file in examples/ passes strict JSON Schema validation."""
        examples_dir = Path(config.REPO_DIR) / "examples"
        schema_validator.validate_config(
            json.loads((examples_dir / "config.json").read_text(encoding="utf-8"))
        )
        schema_validator.validate_scenario(
            json.loads((examples_dir / "scenario.json").read_text(encoding="utf-8"))
        )

        tool_examples = [
            ("image_generate", "mcp_image_generate.json"),
            ("image_generate", "mcp_image_generate_inpaint.json"),
            ("image_generate", "mcp_image_generate_outpaint.json"),
            ("video_generate", "mcp_video_generate.json"),
            ("video_validate", "mcp_video_validate.json"),
            ("media_get_models", "mcp_media_get_models.json"),
            ("media_get_samplers", "mcp_media_get_samplers.json"),
            ("media_get_loras", "mcp_media_get_loras.json"),
        ]
        for tool_name, filename in tool_examples:
            payload = json.loads((examples_dir / filename).read_text(encoding="utf-8"))
            schema_validator.validate_mcp_args(tool_name, payload)

    def test_03_image_generate_mandatory_engine_and_model(self):
        """Verify image_generate requires engine, model, and prompt, and allows loras on flux2."""
        valid_sdxl = {
            "engine": "sdxl",
            "model": "ponyDiffusionV6XL.safetensors",
            "prompt": "a beautiful forest",
            "cfg": 5.5,
            "loras": [{"name": "style_lora", "strength": 0.8}]
        }
        schema_validator.validate_mcp_args("image_generate", valid_sdxl)

        # FLUX.2 now supports loras (D3)
        valid_flux2_lora = {
            "engine": "flux2",
            "model": "aishaNSFWFlux2Klein9B.safetensors",
            "prompt": "portrait",
            "guidance": 3.5,
            "loras": [{"name": "flux2_detailer", "strength": 0.7}]
        }
        schema_validator.validate_mcp_args("image_generate", valid_flux2_lora)

        # Missing model must fail
        missing_model = {
            "engine": "sdxl",
            "prompt": "missing model"
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_args("image_generate", missing_model)

        # Missing engine must fail
        missing_engine = {
            "model": "ponyDiffusionV6XL.safetensors",
            "prompt": "missing engine"
        }
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_args("image_generate", missing_engine)

    def test_04_image_generate_conditional_constraints(self):
        """Verify conditional schema rules for image_path, mask_path, fit, width, height, and denoise."""
        base = {
            "engine": "flux1",
            "model": "flux1_dev.safetensors",
            "prompt": "test"
        }

        # 1. denoise without image_path is forbidden
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec({**base, "denoise": 0.5})

        # 2. mask_path without image_path is forbidden
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec({**base, "mask_path": "mask.png"})

        # 3. fit without image_path is forbidden
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec({**base, "fit": "crop", "width": 1024, "height": 1024})

        # 4. fit without width/height is forbidden
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec({**base, "image_path": "src.png", "fit": "crop"})

        # 5. image_path without fit forbids manual width/height
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_image_spec({**base, "image_path": "src.png", "width": 1024})

        # 6. Valid fit='crop' and fit='outpaint' with image_path + width + height
        schema_validator.validate_image_spec({
            **base, "image_path": "src.png", "fit": "crop", "width": 1024, "height": 1024
        })
        schema_validator.validate_image_spec({
            **base, "image_path": "src.png", "fit": "outpaint", "width": 1360, "height": 768
        })

    def test_05_scenario_schema_hierarchy_and_engine_enum(self):
        """Verify scenario.json enforces root .image/.video models and forbids technical params in blocks/anchors."""
        valid_scenario = {
            "name": "Valid Scenario",
            "image": {
                "engine": "sdxl",
                "model": "ponyDiffusionV6XL.safetensors",
                "steps": 28,
                "cfg": 5.5
            },
            "video": {
                "engine": "wan-full",
                "model": "wan2.2_i2v_14B_fp8_scaled.safetensors",
                "width": 832,
                "height": 480,
                "fps": 16,
                "steps": 20,
                "cfg": 3.5,
                "ramp_latents": 3
            },
            "scenes": [
                {
                    "name": "Scene 1",
                    "anchor_image": {
                        "prompt": "character standing"
                    },
                    "blocks": [
                        {
                            "name": "Block 1",
                            "duration": 4.0,
                            "overlap": 0.0,
                            "motion_prompt": "waves hand",
                            "loras": [{"name": "motion_lora", "strength": 0.8}]
                        }
                    ]
                }
            ]
        }
        schema_validator.validate_scenario(valid_scenario)

        # Legacy engine 'wan' must be rejected (replaced by 'wan-full')
        invalid_engine = json.loads(json.dumps(valid_scenario))
        invalid_engine["video"]["engine"] = "wan"
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_engine)

        # Missing video.model must be rejected
        missing_vid_model = json.loads(json.dumps(valid_scenario))
        del missing_vid_model["video"]["model"]
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(missing_vid_model)

        # Technical parameter (steps/cfg/seed) in block must be rejected
        invalid_block_tech = json.loads(json.dumps(valid_scenario))
        invalid_block_tech["scenes"][0]["blocks"][0]["steps"] = 20
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_block_tech)

        # Technical parameter (engine/model/width) in anchor_image must be rejected
        invalid_anchor_tech = json.loads(json.dumps(valid_scenario))
        invalid_anchor_tech["scenes"][0]["anchor_image"]["engine"] = "flux1"
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_scenario(invalid_anchor_tech)

    def test_06_response_schema_validation(self):
        """Verify response schemas validate refactored payloads and reject obsolete fields."""
        valid_img_res = {
            "run_prefix": "20261017-123456-abcdef",
            "image_path": "G:/media_mcp/images/20261017-123456-abcdef-final.png",
            "spec_path": "G:/media_mcp/workflows/20261017-123456-abcdef-spec.json",
            "engine": "flux2",
            "model": "flux2-klein-9b.safetensors",
            "seed": 123456,
            "width": 1920,
            "height": 1080,
            "native_width": 1360,
            "native_height": 768,
            "warnings": ["Requested resolution 1920x1080 exceeds optimal ~1.0 MP"]
        }
        schema_validator.validate_mcp_response("image_generate", valid_img_res)

        valid_vid_res = {
            "run_prefix": "20261017-123456-abcdef",
            "video_path": "G:/media_mcp/videos/20261017-123456-abcdef-final.mp4",
            "poster_path": "G:/media_mcp/images/20261017-123456-abcdef-poster.png",
            "spec_path": "G:/media_mcp/workflows/20261017-123456-abcdef-spec.json",
            "duration": 9.0,
            "width": 832,
            "height": 480,
            "fps": 16,
            "scenes_count": 1,
            "blocks_count": 2
        }
        schema_validator.validate_mcp_response("video_generate", valid_vid_res)

        # Obsolete hud_video_path must be rejected by strict response schema
        invalid_vid_hud = {**valid_vid_res, "hud_video_path": None}
        with self.assertRaises(schema_validator.SchemaValidationError):
            schema_validator.validate_mcp_response("video_generate", invalid_vid_hud)

    def test_07_schema_formatter_help(self):
        """Verify schema_formatter renders general help and per-tool help cleanly."""
        gen_help = schema_formatter.format_general_help()
        self.assertIn("media_mcp", gen_help)
        for tool in CANONICAL_TOOLS:
            self.assertIn(tool, gen_help)

        img_help = schema_formatter.format_tool_help("image_generate")
        self.assertIn("image_generate", img_help)
        self.assertIn("model", img_help)
        self.assertIn("INPUT PARAMETERS", img_help)
        self.assertIn("OUTPUT FORMAT", img_help)


if __name__ == "__main__":
    unittest.main()
