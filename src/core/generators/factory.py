#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator Factory and Strategy Registry for media_mcp.
Instantiates the appropriate image or video generator based on canonical engine names.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Type

from .image_generators import BaseImageGenerator, SdxlImageGenerator, Flux1ImageGenerator, Flux2ImageGenerator
from .video_generators import BaseVideoGenerator, WanFastVideoGenerator, WanFullVideoGenerator, LtxVideoGenerator

IMAGE_GENERATOR_REGISTRY: Dict[str, Type[BaseImageGenerator]] = {
    "sdxl": SdxlImageGenerator,
    "flux1": Flux1ImageGenerator,
    "flux2": Flux2ImageGenerator,
}

VIDEO_GENERATOR_REGISTRY: Dict[str, Type[BaseVideoGenerator]] = {
    "wan-fast": WanFastVideoGenerator,
    "wan": WanFullVideoGenerator,
    "ltx": LtxVideoGenerator,
}


def create_image_generator(engine_or_name: str = "sdxl", settings: Optional[Dict[str, Any]] = None) -> BaseImageGenerator:
    """
    Factory method to create an image generator matching canonical engine name ('sdxl', 'flux1', 'flux2').
    """
    key = str(engine_or_name or "sdxl").strip().lower()
    cls = IMAGE_GENERATOR_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported image engine '{engine_or_name}'. Allowed values: {list(IMAGE_GENERATOR_REGISTRY.keys())}"
        )
    return cls(name=key, settings=settings)


def create_video_generator(engine_or_name: str = "wan-fast", settings: Optional[Dict[str, Any]] = None) -> BaseVideoGenerator:
    """
    Factory method to create a video generator matching canonical engine name ('wan-fast', 'wan', 'ltx').
    """
    key = str(engine_or_name or "wan-fast").strip().lower()
    cls = VIDEO_GENERATOR_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unsupported video engine '{engine_or_name}'. Allowed values: {list(VIDEO_GENERATOR_REGISTRY.keys())}"
        )
    return cls(name=key, settings=settings)
