#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Discovery Engine for media_mcp.
Implements shape-invariant model tree discovery, model-specific sorted sampler queries,
and LoRA compatibility inspection with activation trigger extraction.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

from . import a1111_client
from . import comfy_client
from .lora_inspector import (
    read_safetensors_metadata,
    extract_triggers_from_metadata,
    detect_lora_architecture,
    detect_checkpoint_architecture
)


def log(msg: str):
    sys.stderr.write(f"[discovery] {msg}\n")
    sys.stderr.flush()


# Canonical engine declarations
IMAGE_ENGINES = ("sdxl", "flux1", "flux2")
VIDEO_ENGINES = ("wan-fast", "wan", "ltx")


def get_models_tree(type_filter: Optional[str] = None, engine_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns an invariant model tree:
    {
      "image"?: { "sdxl": [...], "flux1": [...], "flux2": [...] },
      "video"?: { "wan-fast": [...], "wan": [...], "ltx": [...] }
    }
    Filters prune non-matching keys without altering the hierarchical tree structure.
    """
    t_filter = (type_filter or "").strip().lower()
    e_filter = (engine_filter or "").strip().lower()

    tree: Dict[str, Any] = {}

    # 1. Image branch
    if not t_filter or t_filter == "image":
        img_branch: Dict[str, List[str]] = {}

        # SDXL / Pony (A1111)
        if not e_filter or e_filter == "sdxl":
            try:
                a_models = a1111_client.get_models()
                sdxl_list = [
                    m.get("title", m.get("model_name"))
                    for m in a_models
                    if m.get("title") or m.get("model_name")
                ]
                img_branch["sdxl"] = sdxl_list
            except Exception as ex:
                log(f"Notice: Failed to fetch A1111 models: {ex}")
                img_branch["sdxl"] = []

        # FLUX.1 Dev (ComfyUI)
        if not e_filter or e_filter == "flux1":
            try:
                c_unets = comfy_client.get_unet_models()
                flux1_list = [u for u in c_unets if "flux" in u.lower() and "flux2" not in u.lower() and "klein" not in u.lower()]
                img_branch["flux1"] = flux1_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI FLUX.1 models: {ex}")
                img_branch["flux1"] = []

        # FLUX.2 Klein (ComfyUI)
        if not e_filter or e_filter == "flux2":
            try:
                c_unets = comfy_client.get_unet_models()
                flux2_list = [u for u in c_unets if "flux2" in u.lower() or "klein" in u.lower()]
                img_branch["flux2"] = flux2_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI FLUX.2 models: {ex}")
                img_branch["flux2"] = []

        if img_branch:
            tree["image"] = img_branch

    # 2. Video branch
    if not t_filter or t_filter == "video":
        vid_branch: Dict[str, List[str]] = {}

        # Wan 2.2 Fast (ComfyUI)
        if not e_filter or e_filter == "wan-fast":
            try:
                c_unets = comfy_client.get_unet_models()
                wan_list = [u for u in c_unets if "wan" in u.lower()]
                vid_branch["wan-fast"] = wan_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI Wan Fast models: {ex}")
                vid_branch["wan-fast"] = []

        # Wan 2.2 Full (ComfyUI)
        if not e_filter or e_filter == "wan":
            try:
                c_unets = comfy_client.get_unet_models()
                wan_list = [u for u in c_unets if "wan" in u.lower()]
                vid_branch["wan"] = wan_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI Wan Full models: {ex}")
                vid_branch["wan"] = []

        # LTX-2.3 22B (ComfyUI)
        if not e_filter or e_filter == "ltx":
            try:
                c_ckpts = comfy_client.get_checkpoint_models()
                ltx_list = [c for c in c_ckpts if "ltx" in c.lower()]
                vid_branch["ltx"] = ltx_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI LTX models: {ex}")
                vid_branch["ltx"] = []

        if vid_branch:
            tree["video"] = vid_branch

    return tree


def get_samplers_for_model(engine: str, model: str) -> List[str]:
    """
    Returns a sorted list of valid samplers for the given engine and model.
    The most recommended sampler for the model architecture is placed FIRST at index 0.
    """
    eng = str(engine).strip().lower()
    mod = str(model).strip().lower()

    if eng in ("flux1", "flux2"):
        return ["euler"]
    if eng in ("wan-fast", "wan"):
        return ["uni_pc"]
    if eng == "ltx":
        return ["euler"]

    # For SDXL / Pony in A1111: query server
    try:
        raw_samplers = a1111_client.get_samplers()
        sampler_names = [s.get("name") for s in raw_samplers if s.get("name")]
    except Exception as ex:
        log(f"Notice: Failed to fetch A1111 samplers dynamically: {ex}")
        sampler_names = [
            "Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras",
            "DPM++ 2S a Karras", "Restart", "UniPC", "DDIM"
        ]

    # Prioritization: Pony models recommend "Euler a", standard SDXL recommends "Euler" or "DPM++ 2M Karras"
    is_pony = "pony" in mod or "score_" in mod
    preferred = "Euler a" if is_pony else "Euler"

    sorted_samplers = []
    if preferred in sampler_names:
        sorted_samplers.append(preferred)
    for s in sampler_names:
        if s not in sorted_samplers:
            sorted_samplers.append(s)

    return sorted_samplers


def get_loras_for_model(engine: str, model: str, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns a clean list of compatible LoRA objects for the specified model and engine:
    [
      {
        "name": "lora_name",
        "triggers": ["tag1", "tag2", ...]
      }
    ]
    Extracts trigger words from .safetensors embedded metadata.
    If 'pattern' is provided, filters case-insensitively using Python regex against name and triggers.
    """
    eng = str(engine).strip().lower()
    mod = str(model).strip()

    # FLUX.2 Klein is incompatible with FLUX.1 LoRAs
    if eng == "flux2":
        return []

    # 1. ComfyUI FLUX.1 LoRAs
    if eng == "flux1":
        try:
            raw_loras = comfy_client.get_loras()
            results = []
            for l_name in raw_loras:
                clean_name = os.path.splitext(l_name)[0]
                results.append({"name": clean_name, "triggers": []})
        except Exception as ex:
            log(f"Notice: Failed to fetch ComfyUI LoRAs: {ex}")
            results = []

    # 2. A1111 SDXL / Pony LoRAs
    else:
        target_arch = detect_checkpoint_architecture(mod)
        try:
            raw_a1111_loras = a1111_client.get_loras()
        except Exception as ex:
            log(f"Notice: Failed to fetch A1111 LoRAs: {ex}")
            raw_a1111_loras = []

        results = []
        for lora_obj in raw_a1111_loras:
            l_name = lora_obj.get("name") or ""
            l_path = lora_obj.get("path") or ""
            if not l_name:
                continue

            # Inspect metadata from .safetensors file directly if accessible
            meta = {}
            if l_path and os.path.exists(l_path):
                meta = read_safetensors_metadata(l_path)
            else:
                meta = lora_obj.get("metadata") or {}

            # Architecture compatibility check
            lora_arch = detect_lora_architecture(meta, filepath=l_path or l_name)
            if target_arch in ("pony", "sdxl"):
                if lora_arch == "sd15":
                    continue  # Skip SD 1.5 LoRAs for SDXL/Pony models
            elif target_arch == "sd15":
                if lora_arch in ("sdxl", "flux"):
                    continue

            # Extract triggers
            triggers = extract_triggers_from_metadata(meta)

            results.append({
                "name": l_name,
                "triggers": triggers
            })

    if pattern:
        try:
            rgx = re.compile(pattern, re.IGNORECASE)
        except re.error as ex:
            raise ValueError(f"Invalid regex pattern '{pattern}': {ex}")

        filtered = []
        for item in results:
            name_match = bool(rgx.search(item.get("name", "")))
            trigger_match = any(rgx.search(trg) for trg in item.get("triggers", []))
            if name_match or trigger_match:
                filtered.append(item)
        return filtered

    return results
