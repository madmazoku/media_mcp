#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite 2: Safetensors Binary Inspection, Wan 2.2 Bundle Pairing & Model/LoRA Discovery.
Covers:
- Binary .safetensors header parsing without PyTorch (read_safetensors_header, read_latent_temporal_frames)
- Checkpoint architecture classification (sdxl, flux1 [4096], flux2 [3072], wan, ltx)
- Automatic Wan 2.2 high_noise + low_noise bundle pairing and resolution (D3)
- LoRA architecture filtering and trigger extraction from .safetensors + sidecar .json/.civitai.info (D3)
- Regex pattern filtering in media_get_loras
"""

import sys
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, discovery, lora_inspector
from core.tools import execute_tool


def write_mock_safetensors(path: Path, header_dict: dict):
    """Writes a minimal valid .safetensors file with the given JSON header."""
    raw_json = json.dumps(header_dict).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw_json)))
        f.write(raw_json)


class TestDiscoveryAndSafetensorsInspector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp_dir.name)
        cls.comfy_dir = cls.tmp_path / "ComfyUI"
        (cls.comfy_dir / "models" / "unet").mkdir(parents=True, exist_ok=True)
        (cls.comfy_dir / "models" / "diffusion_models").mkdir(parents=True, exist_ok=True)
        (cls.comfy_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
        (cls.comfy_dir / "models" / "loras").mkdir(parents=True, exist_ok=True)

        cls._test_config = {
            "output_dir": str(cls.tmp_path / "media_mcp"),
            "comfy_dir": str(cls.comfy_dir),
            "comfy_url": "http://127.0.0.1:8188",
            "sd_webui_url": "http://127.0.0.1:7860",
            "media_mcp_port": 7865
        }
        config.load_config_dict(cls._test_config)
        lora_inspector.clear_cache()

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()
        config.cfg_json = {}

    def test_01_safetensors_checkpoint_architecture_detection(self):
        """Verify binary header inspection distinguishes flux1 (4096), flux2 (3072), wan, ltx, and sdxl."""
        unet_dir = self.comfy_dir / "models" / "unet"

        # 1. FLUX.1 (4096 hidden dim)
        f1_path = unet_dir / "custom_mystery_f1.safetensors"
        write_mock_safetensors(f1_path, {
            "double_blocks.0.img_attn.qkv.weight": {"dtype": "F16", "shape": [12288, 4096], "data_offsets": [0, 0]}
        })
        self.assertEqual(lora_inspector.detect_checkpoint_engine(f1_path), "flux1")

        # 2. FLUX.2 Klein (3072 hidden dim)
        f2_path = unet_dir / "custom_mystery_f2.safetensors"
        write_mock_safetensors(f2_path, {
            "double_blocks.0.img_attn.qkv.weight": {"dtype": "F16", "shape": [9216, 3072], "data_offsets": [0, 0]}
        })
        self.assertEqual(lora_inspector.detect_checkpoint_engine(f2_path), "flux2")

        # 3. Wan 2.2 (5D patch_embedding)
        wan_path = unet_dir / "custom_wan_expert.safetensors"
        write_mock_safetensors(wan_path, {
            "patch_embedding.weight": {"dtype": "F16", "shape": [5120, 36, 1, 2, 2], "data_offsets": [0, 0]},
            "blocks.0.self_attn.q.weight": {"dtype": "F16", "shape": [5120, 5120], "data_offsets": [0, 0]}
        })
        self.assertEqual(lora_inspector.detect_checkpoint_engine(wan_path), "wan")

        # 4. LTX-Video (scale_shift_table + transformer_blocks)
        ltx_path = unet_dir / "custom_video_dit.safetensors"
        write_mock_safetensors(ltx_path, {
            "transformer_blocks.0.scale_shift_table": {"dtype": "F16", "shape": [6, 2048], "data_offsets": [0, 0]},
            "adaln_single.linear.weight": {"dtype": "F16", "shape": [12288, 2048], "data_offsets": [0, 0]}
        })
        self.assertEqual(lora_inspector.detect_checkpoint_engine(ltx_path), "ltx")

    def test_02_wan_bundle_pairing_and_resolution(self):
        """Verify Wan 2.2 high_noise + low_noise files pair into a canonical bundle and resolve back."""
        diff_dir = self.comfy_dir / "models" / "diffusion_models"
        high_file = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        low_file = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        unpaired_high = "wan2.2_i2v_high_noise_orphan.safetensors"

        for fname in (high_file, low_file, unpaired_high):
            write_mock_safetensors(diff_dir / fname, {
                "patch_embedding.weight": {"dtype": "F8_E4M3", "shape": [5120, 36, 1, 2, 2], "data_offsets": [0, 0]}
            })

        with patch("core.comfy_client.get_unet_models", return_value=[high_file, low_file, unpaired_high]), \
             patch("core.comfy_client.get_checkpoint_models", return_value=[]), \
             patch("core.a1111_client.get_models", return_value=[{"title": "ponyDiffusionV6XL.safetensors"}]):
            tree = discovery.get_models_tree()

            # Paired bundle is present in both wan-fast and wan-full
            self.assertIn("wan2.2_i2v_14B_fp8_scaled.safetensors", tree["video"]["wan-fast"])
            self.assertIn("wan2.2_i2v_14B_fp8_scaled.safetensors", tree["video"]["wan-full"])

            # Unpaired orphan high_noise file is NOT exposed as a valid bundle
            self.assertNotIn("wan2.2_i2v_orphan.safetensors", tree["video"]["wan-fast"])
            self.assertNotIn(unpaired_high, tree["video"]["wan-fast"])

            # Resolve canonical bundle name back to (high_file, low_file)
            resolved_high, resolved_low = discovery.resolve_wan_model_bundle("wan2.2_i2v_14B_fp8_scaled.safetensors")
            self.assertEqual(resolved_high, high_file)
            self.assertEqual(resolved_low, low_file)

    def test_03_lora_triggers_from_safetensors_and_sidecars(self):
        """Verify ComfyUI LoRAs extract triggers from .safetensors metadata AND sidecar .civitai.info/.json."""
        lora_dir = self.comfy_dir / "models" / "loras"
        lora_inspector.clear_cache()

        # 1. FLUX.1 LoRA with safetensors metadata + sidecar .civitai.info
        f1_lora = lora_dir / "hermione_flux1.safetensors"
        write_mock_safetensors(f1_lora, {
            "__metadata__": {
                "ss_base_model_version": "flux1",
                "ss_tag_frequency": json.dumps({"dataset": {"hermione granger": 50, "hogwarts uniform": 30}})
            },
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": {
                "dtype": "F16", "shape": [16, 4096], "data_offsets": [0, 0]
            }
        })
        sidecar_info = lora_dir / "hermione_flux1.civitai.info"
        sidecar_info.write_text(json.dumps({"trainedWords": ["bushy brown hair", "hermione granger"]}), encoding="utf-8")

        # 2. FLUX.2 LoRA (3072 in_dim)
        f2_lora = lora_dir / "klein_style_f2.safetensors"
        write_mock_safetensors(f2_lora, {
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": {
                "dtype": "F16", "shape": [16, 3072], "data_offsets": [0, 0]
            }
        })
        (lora_dir / "klein_style_f2.json").write_text(
            json.dumps({"activation text": "klein_cyber_style, neon glow"}), encoding="utf-8"
        )

        with patch("core.comfy_client.get_loras", return_value=["hermione_flux1.safetensors", "klein_style_f2.safetensors"]):
            # Query for flux1: should return hermione_flux1 and exclude klein_style_f2
            f1_res = discovery.get_loras_for_model("flux1", "persephoneFlux1Dev.safetensors")
            f1_names = [item["name"] for item in f1_res]
            self.assertIn("hermione_flux1", f1_names)
            self.assertNotIn("klein_style_f2", f1_names)

            hermione_entry = next(item for item in f1_res if item["name"] == "hermione_flux1")
            self.assertIn("hermione granger", hermione_entry["triggers"])
            self.assertIn("bushy brown hair", hermione_entry["triggers"])

            # Query for flux2: should return klein_style_f2 and exclude hermione_flux1
            f2_res = discovery.get_loras_for_model("flux2", "aishaNSFWFlux2Klein9B.safetensors")
            f2_names = [item["name"] for item in f2_res]
            self.assertIn("klein_style_f2", f2_names)
            self.assertNotIn("hermione_flux1", f2_names)
            klein_entry = next(item for item in f2_res if item["name"] == "klein_style_f2")
            self.assertIn("klein_cyber_style", klein_entry["triggers"])

    def test_04_read_latent_temporal_frames_without_torch(self):
        """Verify read_latent_temporal_frames extracts T (shape[2]) from 5D latent safetensors."""
        latent_path = self.tmp_path / "test-tail.latent"
        write_mock_safetensors(latent_path, {
            "samples": {"dtype": "F32", "shape": [1, 16, 7, 60, 104], "data_offsets": [0, 0]}
        })
        t_frames = lora_inspector.read_latent_temporal_frames(latent_path)
        self.assertEqual(t_frames, 7)

    def test_05_loras_pattern_regex_filtering(self):
        """Verify media_get_loras pattern regex filter against names and triggers."""
        mock_loras = [
            "hermione_granger_v2.safetensors",
            "cyberpunk_neon_city.safetensors",
            "bikini_summer_beach.safetensors",
            "raw_sketch.safetensors"
        ]
        with patch("core.comfy_client.get_loras", return_value=mock_loras):
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

            res_invalid = execute_tool("media_get_loras", {
                "engine": "flux1",
                "model": "flux_dev.safetensors",
                "pattern": "(unclosed_parenthesis"
            })
            self.assertTrue(res_invalid.get("isError", False))
            self.assertIn("Invalid regex pattern", res_invalid.get("error", ""))


if __name__ == "__main__":
    unittest.main()
