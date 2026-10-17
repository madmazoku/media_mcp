#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generators package for media_mcp.
Exports base dataclasses, concrete strategy generators, and factory functions.
"""

from .base import (
    GenerationRequest,
    ImageRequest,
    VideoRequest,
    GenerationResult,
    BaseGenerator,
    snap_to_step,
    snap_to_16,
    snap_to_32,
    snap_to_64,
)

from .image_generators import (
    BaseImageGenerator,
    SdxlImageGenerator,
    Flux1ImageGenerator,
    Flux2ImageGenerator,
    set_image_progress_callback,
)

from .video_generators import (
    BaseVideoGenerator,
    BaseWanVideoGenerator,
    WanFastVideoGenerator,
    WanFullVideoGenerator,
    LtxVideoGenerator,
)

from .factory import (
    IMAGE_GENERATOR_REGISTRY,
    VIDEO_GENERATOR_REGISTRY,
    create_image_generator,
    create_video_generator,
)

__all__ = [
    "GenerationRequest",
    "ImageRequest",
    "VideoRequest",
    "GenerationResult",
    "BaseGenerator",
    "snap_to_step",
    "snap_to_16",
    "snap_to_32",
    "snap_to_64",
    "BaseImageGenerator",
    "SdxlImageGenerator",
    "Flux1ImageGenerator",
    "Flux2ImageGenerator",
    "set_image_progress_callback",
    "BaseVideoGenerator",
    "BaseWanVideoGenerator",
    "WanFastVideoGenerator",
    "WanFullVideoGenerator",
    "LtxVideoGenerator",
    "IMAGE_GENERATOR_REGISTRY",
    "VIDEO_GENERATOR_REGISTRY",
    "create_image_generator",
    "create_video_generator",
]
