#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base abstractions, request contracts, and result models for media generators.
Following the audio_to_video Generator Strategy and Factory Pattern.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, List


def snap_to_16(dim: int, default: int = 1024) -> int:
    try:
        val = int(dim)
        return max(64, (val // 16) * 16)
    except Exception:
        return default


def snap_to_32(dim: int, default: int = 832) -> int:
    try:
        val = int(dim)
        return max(64, (val // 32) * 32)
    except Exception:
        return default


@dataclass
class GenerationRequest:
    prompt: str
    seed: int = -1
    sub_dir: str = ""
    debug_dir: Optional[Path] = None
    custom_filename: Optional[str] = None


@dataclass
class ImageRequest(GenerationRequest):
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None
    cfg_scale: float = 7.0
    guidance: float = 3.5
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    lora_name: Optional[str] = None
    lora_strength: float = 0.8
    model_name: Optional[str] = None
    highres_fix: bool = False


@dataclass
class VideoRequest(GenerationRequest):
    image_path: Optional[Union[str, List[str]]] = None
    prefix_video_path: Optional[str] = None
    prefix_latent_path: Optional[str] = None
    prefix_frames: int = 0
    end_image_path: Optional[str] = None
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
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    ramp_latents: int = 3
    min_cfg: float = 1.0
    out_video_name: Optional[str] = None
    progress_callback: Optional[Any] = None


@dataclass
class GenerationResult:
    success: bool
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tail_latent_path: Optional[str] = None
    tail_video_path: Optional[str] = None
    tail_image_path: Optional[str] = None
    prefix_video_path: Optional[str] = None
    error: Optional[str] = None



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
