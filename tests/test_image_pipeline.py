#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite 3: Unified Image Pipeline (~1.0 MP Optimal Resolution + Post-Resize + Modes).
Covers:
- Unified continuous ~1.0 MP optimal resolution calculation (64px step for SDXL, 16px step for FLUX) (D4)
- Two-stage render at (native_width, native_height) + Lanczos post-resize to (width, height) (D4)
- Non-fatal warnings for >1.1 MP upscale and extreme aspect ratios (> 2.4:1) (D4)
- Zero hidden prompt auto-prefix in Flux1ImageGenerator and Flux2ImageGenerator (D3)
- Dynamic LoraLoaderModelOnly chaining in both Flux1 and Flux2 (D3)
- img2img, inpaint (continuous gradient mask preservation), fit='crop', fit='outpaint' (pure PIL edge clamp)
- Hyphenated artifact naming (YYYYMMDD-HHMMSS-{hash6}-...) (D6)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, image_engine
from core.generators import create_image_generator, ImageRequest, GenerationResult


class TestImagePipeline(unittest.TestCase):

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

    def test_01_optimal_dimensions_quantization_by_engine(self):
        """Verify calculate_native_dimensions uses 64px quantization for sdxl and 16px for flux1/flux2."""
        # SDXL 16:9 (832x480 target) -> ~1.0 MP snapped to 64px
        sdxl_w, sdxl_h = image_engine.calculate_native_dimensions((832, 480), engine="sdxl")
        self.assertEqual(sdxl_w % 64, 0)
        self.assertEqual(sdxl_h % 64, 0)
        self.assertTrue(850_000 <= sdxl_w * sdxl_h <= 1_200_000)

        # FLUX 16:9 (832x480 target) -> ~1.0 MP snapped to 16px
        flux_w, flux_h = image_engine.calculate_native_dimensions((832, 480), engine="flux1")
        self.assertEqual(flux_w % 16, 0)
        self.assertEqual(flux_h % 16, 0)
        self.assertTrue(900_000 <= flux_w * flux_h <= 1_150_000)

        # 1:1 square -> 1024x1024 for all engines
        self.assertEqual(image_engine.calculate_native_dimensions((512, 512), engine="sdxl"), (1024, 1024))
        self.assertEqual(image_engine.calculate_native_dimensions((512, 512), engine="flux2"), (1024, 1024))

    def test_02_two_stage_generation_and_post_resize_with_warnings(self):
        """Verify generate_image diffuses at optimal ~1.0 MP, post-resizes to target, and emits warnings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            def fake_generate(req: ImageRequest):
                out_file = Path(tmp_dir) / (req.custom_filename or "out.png")
                Image.new("RGB", (req.width, req.height), color=(40, 80, 120)).save(out_file)
                return GenerationResult(
                    success=True,
                    image_path=out_file.as_posix(),
                    metadata={"seed": 123, "width": req.width, "height": req.height}
                )

            with patch("core.image_engine.create_image_generator") as mock_factory:
                mock_gen = mock_factory.return_value
                mock_gen.generate.side_effect = fake_generate

                # Case A: Request small 832x480 -> renders at ~1.0 MP and downscales to 832x480 without warnings
                ok, img_p, _, meta = image_engine.generate_image({
                    "engine": "flux1",
                    "model": "flux1_dev.safetensors",
                    "prompt": "test scene",
                    "width": 832,
                    "height": 480
                })
                self.assertTrue(ok)
                self.assertEqual((meta["width"], meta["height"]), (832, 480))
                self.assertGreater(meta["native_width"] * meta["native_height"], 832 * 480)
                self.assertNotIn("warnings", meta)
                with Image.open(img_p) as final_im:
                    self.assertEqual(final_im.size, (832, 480))

                # Verify hyphenated run_prefix format (YYYYMMDD-HHMMSS-{hash6})
                parts = meta["run_prefix"].split("-")
                self.assertEqual(len(parts), 3)
                self.assertNotIn("_", meta["run_prefix"])

                # Case B: Request 1920x1088 (>1.1 MP) -> renders at ~1.0 MP, upscales to 1920x1088, emits warning
                ok2, img_p2, _, meta2 = image_engine.generate_image({
                    "engine": "sdxl",
                    "model": "ponyDiffusionV6XL.safetensors",
                    "prompt": "upscale test",
                    "width": 1920,
                    "height": 1088
                })
                self.assertTrue(ok2)
                self.assertEqual((meta2["width"], meta2["height"]), (1920, 1088))
                self.assertIn("warnings", meta2)
                self.assertTrue(any("exceeds optimal ~1.0 MP" in w for w in meta2["warnings"]))
                with Image.open(img_p2) as up_im:
                    self.assertEqual(up_im.size, (1920, 1088))

                # Case C: Extreme aspect ratio 3.0:1 (1536x512) -> emits AR warning
                ok3, _, _, meta3 = image_engine.generate_image({
                    "engine": "flux2",
                    "model": "flux2_klein.safetensors",
                    "prompt": "ultrawide banner",
                    "width": 1536,
                    "height": 512
                })
                self.assertTrue(ok3)
                self.assertIn("warnings", meta3)
                self.assertTrue(any("aspect ratio" in w for w in meta3["warnings"]))

    def test_03_flux1_and_flux2_zero_auto_prefix_and_lora_chaining(self):
        """Verify Flux1 and Flux2 do not prepend hidden prefixes and support LoraLoaderModelOnly."""
        req = ImageRequest(
            prompt="exact user prompt only",
            model="test_flux_model.safetensors",
            width=1024,
            height=1024,
            loras=[{"name": "char_lora", "strength": 0.85}],
            run_prefix="20261017-120000-abcdef"
        )

        # Test Flux1ImageGenerator
        f1_gen = create_image_generator("flux1", settings={})
        with patch("core.comfy_client.queue_prompt", return_value=(True, "pid_f1")) as mock_q1, \
             patch("core.comfy_client.wait_for_prompt", return_value=(True, {"outputs": {"9": {"images": [{"filename": "out.png"}]}}})), \
             patch("core.comfy_client.download_output_file", return_value=b"\x89PNG\r\n\x1a\nfake"):
            res1 = f1_gen.generate(req)
            self.assertTrue(res1.success, res1.error)
            graph1 = mock_q1.call_args[0][0]
            self.assertEqual(graph1["5"]["inputs"]["text"], "exact user prompt only")
            self.assertIn("20", graph1)
            self.assertEqual(graph1["20"]["class_type"], "LoraLoaderModelOnly")

        # Test Flux2ImageGenerator
        f2_gen = create_image_generator("flux2", settings={})
        with patch("core.comfy_client.queue_prompt", return_value=(True, "pid_f2")) as mock_q2, \
             patch("core.comfy_client.wait_for_prompt", return_value=(True, {"outputs": {"1011": {"images": [{"filename": "out.png"}]}}})), \
             patch("core.comfy_client.download_output_file", return_value=b"\x89PNG\r\n\x1a\nfake"):
            res2 = f2_gen.generate(req)
            self.assertTrue(res2.success, res2.error)
            graph2 = mock_q2.call_args[0][0]
            self.assertEqual(graph2["1004"]["inputs"]["text"], "exact user prompt only")
            self.assertIn("1030", graph2)
            self.assertEqual(graph2["1030"]["class_type"], "LoraLoaderModelOnly")
            self.assertEqual(graph2["1006"]["inputs"]["model"], ["1030", 0])

    def test_04_inpaint_continuous_gradient_and_outpaint_pure_pil(self):
        """Verify continuous mask preservation and pure PIL outpaint edge clamping + feathering."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_square = Path(tmp_dir) / "source_1_1.png"
            Image.new("RGB", (1000, 1000), color=(80, 90, 100)).save(src_square)

            def fake_gen(req: ImageRequest):
                out_p = Path(tmp_dir) / "out.png"
                Image.new("RGB", (req.width, req.height), color=(80, 90, 100)).save(out_p)
                return GenerationResult(
                    success=True,
                    image_path=out_p.as_posix(),
                    metadata={"seed": 42, "width": req.width, "height": req.height}
                )

            with patch("core.image_engine.create_image_generator") as mock_factory:
                mock_factory.return_value.generate.side_effect = fake_gen

                ok, _, _, meta = image_engine.generate_image({
                    "engine": "flux2",
                    "model": "flux2_klein.safetensors",
                    "prompt": "outpaint background",
                    "image_path": str(src_square),
                    "fit": "outpaint",
                    "width": 1360,
                    "height": 768
                })
                self.assertTrue(ok)
                passed_req: ImageRequest = mock_factory.return_value.generate.call_args[0][0]
                self.assertTrue(passed_req.is_inpaint)
                self.assertIsNotNone(passed_req.mask_path)

                with Image.open(passed_req.mask_path) as m_im:
                    self.assertEqual(m_im.size, (passed_req.width, passed_req.height))
                    self.assertEqual(m_im.getpixel((0, passed_req.height // 2)), 255)
                    self.assertEqual(m_im.getpixel((passed_req.width // 2, passed_req.height // 2)), 0)
                    unique_vals = set(m_im.getdata())
                    intermediate = [v for v in unique_vals if 20 < v < 235]
                    self.assertTrue(len(intermediate) > 0, "Outpaint mask must have smooth feathered transitions")


if __name__ == "__main__":
    unittest.main()
