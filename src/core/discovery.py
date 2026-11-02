#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Discovery Engine for media_mcp.
Implements shape-invariant model tree discovery via safetensors header inspection,
automatic Wan 2.2 high/low noise bundle pairing, model-specific sorted sampler queries,
and LoRA compatibility inspection with activation trigger extraction.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from . import config
from . import a1111_client
from . import comfy_client
from .lora_inspector import (
    read_safetensors_header,
    read_safetensors_metadata,
    extract_triggers_from_metadata,
    detect_lora_architecture,
    detect_model_architecture_from_header,
    detect_checkpoint_architecture
)



def log(msg: str):
    sys.stderr.write(f"[discovery] {msg}\n")
    sys.stderr.flush()


# Canonical engine declarations
IMAGE_ENGINES = ("sdxl", "sdxl-comfy", "flux1", "flux2")
VIDEO_ENGINES = ("wan-fast", "wan-full", "ltx", "hunyuan")

_WAN_HIGH_LOW_RGX = re.compile(r"([_-]?)(high|low)(_noise)?(?=[_.-])", re.IGNORECASE)


def _locate_comfy_file(rel_name: str, subdirs: Tuple[str, ...]) -> str:
    """ Resolves a relative model/LoRA filename inside COMFY_DIR/models/<subdir>. """
    root = getattr(config, "COMFY_DIR", "") or getattr(config, "COMFY_ROOT", "")
    if not root:
        return ""
    base = Path(root) / "models"
    for sub in subdirs:
        candidate = base / sub / rel_name
        if candidate.is_file():
            return str(candidate)
    return ""


def _classify_comfy_model(rel_name: str, subdirs: Tuple[str, ...] = ("unet", "diffusion_models", "checkpoints")) -> str:
    """
    Classifies a ComfyUI model into ('flux1', 'flux2', 'wan', 'ltx', 'sdxl', 'sd15')
    by inspecting its .safetensors header on disk if accessible, falling back to filename heuristics.
    """
    full_path = _locate_comfy_file(rel_name, subdirs)
    if full_path:
        header = read_safetensors_header(full_path)
        if header:
            return detect_model_architecture_from_header(header, filepath=full_path)
    return detect_checkpoint_architecture(rel_name)


def _derive_wan_bundle_name(filename: str) -> Optional[Tuple[str, str]]:
    """
    If filename contains a high_noise / low_noise token, returns (bundle_name, role)
    where role is 'high' or 'low' and bundle_name has the token removed.
    Example: 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors'
          -> ('wan2.2_i2v_14B_fp8_scaled.safetensors', 'high')
    """
    m = _WAN_HIGH_LOW_RGX.search(filename)
    if not m:
        return None
    role = m.group(2).lower()
    bundle_name = _WAN_HIGH_LOW_RGX.sub("", filename, count=1)
    # Clean up double underscores or leading/trailing separators in stem
    bundle_name = re.sub(r"__+", "_", bundle_name)
    bundle_name = re.sub(r"--+", "-", bundle_name)
    return bundle_name, role


def _discover_wan_bundles(wan_models: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Pairs Wan 2.2 high_noise and low_noise UNET files into canonical bundle identifiers.
    Only returns bundles where BOTH high_noise and low_noise files are present.
    """
    high_map: Dict[str, str] = {}
    low_map: Dict[str, str] = {}
    standalone: List[str] = []

    for m_name in wan_models:
        parsed = _derive_wan_bundle_name(m_name)
        if parsed is None:
            standalone.append(m_name)
            continue
        bundle_name, role = parsed
        if role == "high":
            high_map[bundle_name] = m_name
        elif role == "low":
            low_map[bundle_name] = m_name

    bundles: Dict[str, Tuple[str, str]] = {}
    for b_name, high_file in high_map.items():
        if b_name in low_map:
            bundles[b_name] = (high_file, low_map[b_name])

    # If a Wan model is genuinely standalone (no high/low split), map it to itself
    for s_name in standalone:
        if s_name not in bundles:
            bundles[s_name] = (s_name, s_name)

    return bundles


def resolve_wan_model_bundle(model_or_bundle: str, available_unets: Optional[List[str]] = None) -> Tuple[str, str]:
    """
    Resolves a Wan 2.2 bundle identifier (e.g. 'wan2.2_i2v_14B_fp8_scaled.safetensors')
    or explicit high/low model name into a concrete (high_noise_model, low_noise_model) tuple.
    """
    target = str(model_or_bundle or "").strip()
    if not target:
        raise ValueError("Wan model identifier must not be empty.")

    if available_unets is None:
        try:
            available_unets = comfy_client.get_unet_models()
        except Exception:
            available_unets = []

    if available_unets:
        wan_models = [u for u in available_unets if _classify_comfy_model(u) == "wan"]
        bundles = _discover_wan_bundles(wan_models)
        if target in bundles:
            return bundles[target]
        # Also support if user passed exact high_noise or low_noise filename that belongs to a valid bundle
        for b_name, (h_file, l_file) in bundles.items():
            if target in (h_file, l_file):
                return h_file, l_file

    # Deterministic offline resolution (when ComfyUI is not queried or in unit tests)
    parsed = _derive_wan_bundle_name(target)
    if parsed is not None:
        _, role = parsed
        if role == "high":
            low_f = _WAN_HIGH_LOW_RGX.sub(lambda m: f"{m.group(1)}low{m.group(3) or ''}", target, count=1)
            return target, low_f
        else:
            high_f = _WAN_HIGH_LOW_RGX.sub(lambda m: f"{m.group(1)}high{m.group(3) or ''}", target, count=1)
            return high_f, target

    if "_i2v_" in target:
        high_f = target.replace("_i2v_", "_i2v_high_noise_", 1)
        low_f = target.replace("_i2v_", "_i2v_low_noise_", 1)
        return high_f, low_f

    p = Path(target)
    ext = p.suffix or ".safetensors"
    stem = p.stem if p.suffix else target
    return f"{stem}_high_noise{ext}", f"{stem}_low_noise{ext}"


def get_models_tree(type_filter: Optional[str] = None, engine_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns an invariant model tree:
    {
      "image"?: { "sdxl": [...], "flux1": [...], "flux2": [...] },
      "video"?: { "wan-fast": [...], "wan-full": [...], "ltx": [...] }
    }
    Filters prune non-matching keys without altering the hierarchical tree structure.
    """
    t_filter = (type_filter or "").strip().lower()
    e_filter = (engine_filter or "").strip().lower()

    tree: Dict[str, Any] = {}

    # Cache ComfyUI model lists so we only query once per call
    c_unets: Optional[List[str]] = None
    c_ckpts: Optional[List[str]] = None

    def _get_unets() -> List[str]:
        nonlocal c_unets
        if c_unets is None:
            c_unets = comfy_client.get_unet_models()
        return c_unets

    def _get_ckpts() -> List[str]:
        nonlocal c_ckpts
        if c_ckpts is None:
            c_ckpts = comfy_client.get_checkpoint_models()
        return c_ckpts

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

        # SDXL (ComfyUI)
        if not e_filter or e_filter == "sdxl-comfy":
            try:
                ckpts = _get_ckpts()
                sdxl_c_list = [c for c in ckpts if _classify_comfy_model(c) in ("sdxl", "pony")]
                img_branch["sdxl-comfy"] = sdxl_c_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI SDXL models: {ex}")
                img_branch["sdxl-comfy"] = []

        # FLUX.1 Dev (ComfyUI)
        if not e_filter or e_filter == "flux1":
            try:
                unets = _get_unets()
                flux1_list = [u for u in unets if _classify_comfy_model(u) == "flux1"]
                img_branch["flux1"] = flux1_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI FLUX.1 models: {ex}")
                img_branch["flux1"] = []

        # FLUX.2 Klein (ComfyUI)
        if not e_filter or e_filter == "flux2":
            try:
                unets = _get_unets()
                flux2_list = [u for u in unets if _classify_comfy_model(u) == "flux2"]
                img_branch["flux2"] = flux2_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI FLUX.2 models: {ex}")
                img_branch["flux2"] = []

        if img_branch:
            tree["image"] = img_branch

    # 2. Video branch
    if not t_filter or t_filter == "video":
        vid_branch: Dict[str, List[str]] = {}

        # Wan 2.2 Fast (ComfyUI paired bundles)
        if not e_filter or e_filter == "wan-fast":
            try:
                unets = _get_unets()
                wan_raw = [u for u in unets if _classify_comfy_model(u) == "wan"]
                bundles = _discover_wan_bundles(wan_raw)
                vid_branch["wan-fast"] = list(bundles.keys())
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI Wan Fast models: {ex}")
                vid_branch["wan-fast"] = []

        # Wan 2.2 Full (ComfyUI paired bundles)
        if not e_filter or e_filter == "wan-full":
            try:
                unets = _get_unets()
                wan_raw = [u for u in unets if _classify_comfy_model(u) == "wan"]
                bundles = _discover_wan_bundles(wan_raw)
                vid_branch["wan-full"] = list(bundles.keys())
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI Wan Full models: {ex}")
                vid_branch["wan-full"] = []

        # HunyuanVideo 1.5 (ComfyUI)
        if not e_filter or e_filter == "hunyuan":
            try:
                unets = _get_unets()
                hunyuan_list = [u for u in unets if _classify_comfy_model(u) == "hunyuan"]
                vid_branch["hunyuan"] = hunyuan_list
            except Exception as ex:
                log(f"Notice: Failed to fetch ComfyUI Hunyuan models: {ex}")
                vid_branch["hunyuan"] = []

        # LTX-2.3 22B / Sulphur (ComfyUI)
        if not e_filter or e_filter == "ltx":
            try:
                unets = _get_unets()
                ckpts = _get_ckpts()
                all_models = list(dict.fromkeys(unets + ckpts))
                ltx_list = [m for m in all_models if _classify_comfy_model(m) == "ltx"]
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
    from .generators.factory import IMAGE_GENERATOR_REGISTRY, VIDEO_GENERATOR_REGISTRY
    eng = str(engine).strip().lower()
    mod = str(model).strip()

    if eng in IMAGE_GENERATOR_REGISTRY:
        gen_cls = IMAGE_GENERATOR_REGISTRY[eng]
    elif eng in VIDEO_GENERATOR_REGISTRY:
        gen_cls = VIDEO_GENERATOR_REGISTRY[eng]
    else:
        raise ValueError(f"Unsupported engine '{engine}' for sampler discovery.")

    return gen_cls.get_samplers(mod)


def get_loras_for_model(engine: str, model: str, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns a clean list of compatible LoRA objects for the specified model and engine:
    [
      {
        "name": "lora_name",
        "triggers": ["tag1", "tag2", ...]
      }
    ]
    Extracts trigger words and architecture compatibility from .safetensors headers.
    If 'pattern' is provided, filters case-insensitively using Python regex against name and triggers.
    """
    from .generators.factory import IMAGE_GENERATOR_REGISTRY, VIDEO_GENERATOR_REGISTRY
    eng = str(engine).strip().lower()
    mod = str(model).strip()

    if eng in IMAGE_GENERATOR_REGISTRY:
        gen_cls = IMAGE_GENERATOR_REGISTRY[eng]
    elif eng in VIDEO_GENERATOR_REGISTRY:
        gen_cls = VIDEO_GENERATOR_REGISTRY[eng]
    else:
        raise ValueError(f"Unsupported engine '{engine}' for LoRA discovery.")

    results = []
    if gen_cls.service == "comfyui":
        try:
            raw_loras = comfy_client.get_loras()
        except Exception as ex:
            log(f"Notice: Failed to fetch ComfyUI LoRAs: {ex}")
            raw_loras = []

        for l_name in raw_loras:
            clean_name = Path(l_name).stem
            full_path = _locate_comfy_file(l_name, ("loras",))
            header = read_safetensors_header(full_path) if full_path else {}
            meta = header.get("__metadata__", {}) if isinstance(header, dict) else {}
            if not isinstance(meta, dict):
                meta = {}

            is_match = gen_cls.matches_lora(header=header, meta=meta, filename=full_path or l_name)
            if not is_match:
                arch = detect_lora_architecture(meta, filepath=full_path or l_name, header=header)
                if arch == "unknown" and getattr(gen_cls, "allow_unclassified_loras", False):
                    is_match = True
            if not is_match:
                continue

            triggers = extract_triggers_from_metadata(meta, filepath=full_path)
            results.append({
                "name": clean_name,
                "triggers": triggers
            })

    elif gen_cls.service == "automatic1111":
        try:
            raw_a1111_loras = a1111_client.get_loras()
        except Exception as ex:
            log(f"Notice: Failed to fetch A1111 LoRAs: {ex}")
            raw_a1111_loras = []

        for lora_obj in raw_a1111_loras:
            l_name = lora_obj.get("name") or ""
            l_path = lora_obj.get("path") or ""
            if not l_name:
                continue

            header = {}
            meta = {}
            if l_path and Path(l_path).is_file():
                header = read_safetensors_header(l_path)
                meta = header.get("__metadata__", {}) if isinstance(header, dict) else {}
            else:
                meta = lora_obj.get("metadata") or {}

            is_match = gen_cls.matches_lora(header=header, meta=meta, filename=l_path or l_name)
            if not is_match:
                arch = detect_lora_architecture(meta, filepath=l_path or l_name, header=header)
                if arch == "unknown" and getattr(gen_cls, "allow_unclassified_loras", False):
                    is_match = True
            if not is_match:
                continue

            triggers = extract_triggers_from_metadata(meta, filepath=l_path)
            results.append({
                "name": l_name,
                "triggers": triggers
            })
    else:
        raise ValueError(f"Unsupported service '{gen_cls.service}' for engine '{engine}'")

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
