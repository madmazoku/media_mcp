#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base abstractions, request contracts, and result models for media generators.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def snap_to_step(dim: int, step: int = 16, default: int = 1024) -> int:
    try:
        val = int(dim)
        s = max(1, int(step))
        return max(64, (val // s) * s)
    except Exception:
        return default


def snap_to_16(dim: int, default: int = 1024) -> int:
    return snap_to_step(dim, step=16, default=default)


def snap_to_32(dim: int, default: int = 832) -> int:
    return snap_to_step(dim, step=32, default=default)


def snap_to_64(dim: int, default: int = 1024) -> int:
    return snap_to_step(dim, step=64, default=default)



@dataclass
class GenerationRequest:
    prompt: str
    seed: int = -1


@dataclass
class ImageRequest(GenerationRequest):
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None
    cfg: float = 7.0
    guidance: float = 3.5
    sampler: Optional[str] = None
    loras: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    custom_filename: Optional[str] = None
    run_prefix: Optional[str] = None
    image_path: Optional[str] = None
    mask_path: Optional[str] = None
    fit: Optional[str] = None
    denoise: float = 0.65
    progress_callback: Optional[Any] = None

    def __post_init__(self):
        if not self.image_path:
            self.denoise = 1.0
            self.mask_path = None
            self.fit = None

    @property
    def is_img2img(self) -> bool:
        return bool(self.image_path)

    @property
    def is_inpaint(self) -> bool:
        return bool(self.image_path and self.mask_path)

    # Compatibility aliases
    @property
    def cfg_scale(self) -> float:
        return self.cfg

    @cfg_scale.setter
    def cfg_scale(self, val: float):
        self.cfg = val

    @property
    def sampler_name(self) -> Optional[str]:
        return self.sampler

    @sampler_name.setter
    def sampler_name(self, val: Optional[str]):
        self.sampler = val

    @property
    def model_name(self) -> Optional[str]:
        return self.model

    @model_name.setter
    def model_name(self, val: Optional[str]):
        self.model = val


@dataclass
class VideoRequest(GenerationRequest):
    image_path: Optional[str] = None
    prefix_latent_path: Optional[str] = None
    prefix_frames: int = 0
    negative_prompt: str = ""
    width: int = 832
    height: int = 480
    num_frames: int = 81
    core_frames: Optional[int] = None
    tail_frames: int = 0
    save_tail: bool = True
    save_tail_image: bool = False
    fps: int = 16
    steps: Optional[int] = None
    cfg: Optional[float] = None
    ramp_latents: int = 3
    min_cfg: float = 1.0
    weight_dtype: Optional[str] = None
    model: Optional[str] = None
    loras: Optional[List[Dict[str, Any]]] = None
    out_video_name: Optional[str] = None
    run_prefix: Optional[str] = None
    progress_callback: Optional[Any] = None


@dataclass
class GenerationResult:
    success: bool
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tail_latent_path: Optional[str] = None
    tail_image_path: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        # Synchronize file_path with image_path or video_path
        if self.file_path and not self.image_path and not self.video_path:
            lower = self.file_path.lower()
            if any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                self.image_path = self.file_path
            elif any(lower.endswith(ext) for ext in (".mp4", ".mov", ".mkv", ".webm")):
                self.video_path = self.file_path
        elif self.image_path and not self.file_path:
            self.file_path = self.image_path
        elif self.video_path and not self.file_path:
            self.file_path = self.video_path


class BaseGenerator(ABC):
    """Abstract base class for all image and video generators."""

    def __init__(self, name: str, settings: Optional[Dict[str, Any]] = None):
        self.name = name
        self.settings = dict(settings or {})

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return generator metadata and configuration settings."""
        pass

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute generation and return structured result."""
        pass
