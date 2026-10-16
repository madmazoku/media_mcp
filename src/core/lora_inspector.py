#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA and Model Inspector for media_mcp.
Fast safetensors header parser for extracting LoRA architectures and trigger words,
and model compatibility filters for Automatic1111 and ComfyUI.
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional


def log(msg: str):
    sys.stderr.write(f"[lora-inspector] {msg}\n")
    sys.stderr.flush()


def read_safetensors_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the embedded JSON __metadata__ dictionary from a .safetensors file header
    in sub-millisecond time without reading tensor bytes or loading models into RAM/GPU.
    """
    if not Path(filepath).is_file():
        return {}
    try:
        with open(filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return {}
            header_len = struct.unpack("<Q", header_bytes)[0]
            if header_len <= 0 or header_len > 50_000_000:
                return {}
            header_json = f.read(header_len).decode("utf-8", errors="replace")
            parsed = json.loads(header_json)
            return parsed.get("__metadata__", {})
    except Exception as ex:
        log(f"Warning: Failed to read safetensors metadata from {Path(filepath).name}: {ex}")
        return {}


def extract_triggers_from_metadata(meta: Dict[str, Any]) -> List[str]:
    """
    Extracts high-priority activation triggers / prompt tags from LoRA training metadata.
    Inspects ss_trained_words, ss_tag_frequency, dataset_trigger_words, and civitai info.
    """
    triggers: List[str] = []

    # 1. Direct trained words list
    for key in ("ss_trained_words", "dataset_trigger_words"):
        val = meta.get(key)
        if val:
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    val = [t.strip() for t in val.split(",") if t.strip()]
            if isinstance(val, list):
                for w in val:
                    ws = str(w).strip()
                    if ws and ws not in triggers:
                        triggers.append(ws)

    # 2. Tag frequency dictionary (Kohya-ss format: {"folder": {"tag": count}})
    raw_freq = meta.get("ss_tag_frequency")
    if raw_freq:
        freq_dict = raw_freq
        if isinstance(raw_freq, str):
            try:
                freq_dict = json.loads(raw_freq)
            except Exception:
                freq_dict = {}

        if isinstance(freq_dict, dict):
            tag_counts: Dict[str, int] = {}
            for bucket in freq_dict.values():
                if isinstance(bucket, dict):
                    for tag, count in bucket.items():
                        t_clean = str(tag).strip()
                        if t_clean:
                            try:
                                tag_counts[t_clean] = tag_counts.get(t_clean, 0) + int(count)
                            except Exception:
                                pass

            # Sort tags by frequency descending, take top 10 most prominent
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            for tag, _ in sorted_tags[:10]:
                if tag not in triggers:
                    triggers.append(tag)

    return triggers


def detect_lora_architecture(meta: Dict[str, Any], filepath: str = "") -> str:
    """
    Detects LoRA base model architecture: 'sd15', 'sdxl', 'flux', or 'unknown'.
    """
    base_ver = str(meta.get("ss_base_model_version") or "").lower()
    sd_model_name = str(meta.get("ss_sd_model_name") or "").lower()
    fn = Path(filepath).name.lower() if filepath else ""

    combined = f"{base_ver} {sd_model_name} {fn}"
    if any(x in combined for x in ["flux2"]):
        return "flux2"
    if any(x in combined for x in ["flux", "flux1"]):
        return "flux"
    if any(x in combined for x in ["xl", "sdxl", "pony", "illustrious", "animagine"]):
        return "sdxl"
    if any(x in combined for x in ["v1-5", "sd15", "sd_1_5", "1.5"]):
        return "sd15"

    return "sdxl" if "xl" in combined else "unknown"


def detect_checkpoint_architecture(model_name: str) -> str:
    """
    Infers architecture category of a checkpoint: 'pony', 'sdxl', 'flux1', 'flux2', 'wan', 'ltx', 'sd15'.
    """
    m = model_name.lower()
    if "pony" in m:
        return "pony"
    if "flux2" in m:
        return "flux2"
    if "flux" in m:
        return "flux1"
    if "wan" in m:
        return "wan"
    if "ltx" in m:
        return "ltx"
    if any(x in m for x in ["xl", "illustrious", "animagine", "cyberrealistic"]):
        return "sdxl"
    if any(x in m for x in ["v1-5", "1.5"]):
        return "sd15"
    return "sdxl"
