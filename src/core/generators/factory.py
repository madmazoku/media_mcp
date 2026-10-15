#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator Factory and Strategy Registry for media_mcp.
Instantiates the appropriate image or video generator based on engine name and configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Type

from .base import BaseGenerator
from .image_generators import BaseImageGenerator, SdxlImageGenerator, Flux1ImageGenerator, Flux2ImageGenerator
from .video_generators import BaseVideoGenerator, WanFastVideoGenerator, WanFullVideoGenerator, LtxVideoGenerator

IMAGE_GENERATOR_REGISTRY: Dict[str, Type[BaseImageGenerator]] = {
    # Automatic1111 / SDXL engines
    "sdxl": SdxlImageGenerator,
    "pony": SdxlImageGenerator,
    "illustrious": SdxlImageGenerator,
    "animagine": SdxlImageGenerator,
    "a1111": SdxlImageGenerator,
    "sd15": SdxlImageGenerator,
    # ComfyUI FLUX engines
    "flux1": Flux1ImageGenerator,
    "flux1_dev": Flux1ImageGenerator,
    "flux": Flux1ImageGenerator,
    "flux2": Flux2ImageGenerator,
    "klein": Flux2ImageGenerator,
}

VIDEO_GENERATOR_REGISTRY: Dict[str, Type[BaseVideoGenerator]] = {
    # Wan 2.2 Fast Mode (4-step LoRA, CFG 1.0)
    "wan-fast": WanFastVideoGenerator,
    "wan_fast": WanFastVideoGenerator,
    "wan-4step": WanFastVideoGenerator,
    "wan_4step": WanFastVideoGenerator,
    "fast": WanFastVideoGenerator,
    # Wan 2.2 Full Sampling Mode (20-step UNet, CFG 5.0)
    "wan": WanFullVideoGenerator,
    "wan-full": WanFullVideoGenerator,
    "wan_full": WanFullVideoGenerator,
    "wan2.2": WanFullVideoGenerator,
    "wan2": WanFullVideoGenerator,
    # LTX-2.3 22B
    "ltx": LtxVideoGenerator,
    "ltx-video": LtxVideoGenerator,
    "ltx23": LtxVideoGenerator,
    "ltx-2.3": LtxVideoGenerator,
}


def create_image_generator(engine_or_name: str = "sdxl", settings: Optional[Dict[str, Any]] = None) -> BaseImageGenerator:
    """
    Factory method to create an image generator matching engine_or_name.
    Defaults to SdxlImageGenerator if engine is unknown.
    """
    key = str(engine_or_name or "sdxl").strip().lower()
    cls = IMAGE_GENERATOR_REGISTRY.get(key)
    if cls is None:
        # Fallback check based on name hints
        if any(x in key for x in ["flux2", "klein"]):
            cls = Flux2ImageGenerator
        elif any(x in key for x in ["flux", "dev"]):
            cls = Flux1ImageGenerator
        else:
            cls = SdxlImageGenerator
    return cls(name=engine_or_name, settings=settings)


def create_video_generator(engine_or_name: str = "wan", settings: Optional[Dict[str, Any]] = None) -> BaseVideoGenerator:
    """
    Factory method to create a video generator matching engine_or_name.
    Defaults to WanFullVideoGenerator if engine is 'wan', or WanFastVideoGenerator if 'wan-fast'.
    """
    key = str(engine_or_name or "wan").strip().lower()
    cls = VIDEO_GENERATOR_REGISTRY.get(key)
    if cls is None:
        if any(x in key for x in ["ltx", "22b"]):
            cls = LtxVideoGenerator
        elif any(x in key for x in ["fast", "4step", "quick"]):
            cls = WanFastVideoGenerator
        else:
            cls = WanFullVideoGenerator
    return cls(name=engine_or_name, settings=settings)
