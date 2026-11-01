#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA, Latent, and Model Inspector for media_mcp.
Fast safetensors binary header parser (<0.5ms) for extracting model/LoRA architectures,
activation trigger words, and 5D latent tensor dimensions without loading weights or requiring torch.
"""

import sys
import json
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional


def log(msg: str):
    sys.stderr.write(f"[lora-inspector] {msg}\n")
    sys.stderr.flush()


def read_safetensors_header(filepath: str) -> Dict[str, Any]:
    """
    Reads the complete embedded JSON header dictionary from a .safetensors or .latent file
    in sub-millisecond time without reading tensor weight bytes or loading models into RAM/GPU.
    Contains '__metadata__' as well as tensor descriptors {'name': {'dtype': ..., 'shape': [...]}}.
    """
    if not filepath or not Path(filepath).is_file():
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
            return parsed if isinstance(parsed, dict) else {}
    except Exception as ex:
        log(f"Warning: Failed to read safetensors header from {Path(filepath).name}: {ex}")
        return {}


def read_safetensors_metadata(filepath: str) -> Dict[str, Any]:
    """
    Reads the embedded JSON __metadata__ dictionary from a .safetensors file header.
    """
    header = read_safetensors_header(filepath)
    meta = header.get("__metadata__", {})
    return meta if isinstance(meta, dict) else {}


def read_latent_temporal_frames(filepath: str, default: int = 0) -> int:
    """
    Reads the temporal frame count (dim=2 of 5D tensor [B, C, T, H, W]) directly from a
    ComfyUI .latent (.safetensors) header using pure Python stdlib without torch.
    """
    header = read_safetensors_header(filepath)
    if not header:
        return default
    for key in ("latent_tensor", "samples"):
        tensor_info = header.get(key)
        if isinstance(tensor_info, dict):
            shape = tensor_info.get("shape")
            if isinstance(shape, list) and len(shape) >= 3:
                return int(shape[2])
    for key, tensor_info in header.items():
        if key == "__metadata__":
            continue
        if isinstance(tensor_info, dict):
            shape = tensor_info.get("shape")
            if isinstance(shape, list) and len(shape) >= 3:
                return int(shape[2])
    return default


def extract_triggers_from_metadata(meta: Dict[str, Any], filepath: str = "") -> List[str]:
    """
    Extracts high-priority activation triggers / prompt tags from LoRA training metadata
    and optional sidecar files (.json / .civitai.info).
    Inspects ss_trained_words, ss_tag_frequency, dataset_trigger_words, and civitai trainedWords.
    """
    triggers: List[str] = []

    if not isinstance(meta, dict):
        meta = {}

    # 1. Direct trained words list in safetensors metadata
    for key in ("ss_trained_words", "dataset_trigger_words", "trainedWords", "trigger_words"):
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

            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            for tag, _ in sorted_tags[:10]:
                if tag not in triggers:
                    triggers.append(tag)

    # 3. Companion sidecar metadata (.civitai.info or .json next to .safetensors)
    if filepath and Path(filepath).is_file():
        p = Path(filepath)
        for sidecar in (p.with_suffix(".civitai.info"), p.with_suffix(".json")):
            if sidecar.is_file():
                try:
                    side_data = json.loads(sidecar.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(side_data, dict):
                        for k in ("trainedWords", "triggerWords", "activation text", "triggers"):
                            s_val = side_data.get(k)
                            if isinstance(s_val, str):
                                s_val = [t.strip() for t in s_val.split(",") if t.strip()]
                            if isinstance(s_val, list):
                                for w in s_val:
                                    ws = str(w).strip()
                                    if ws and ws not in triggers:
                                        triggers.append(ws)
                except Exception:
                    pass

    return triggers


def detect_model_architecture_from_header(header: Dict[str, Any], filepath: str = "") -> str:
    """
    Determines model architecture ('flux2', 'flux1', 'wan', 'ltx', 'sdxl', 'pony', 'sd15')
    by querying polymorphic Generator strategies via BaseGenerator.matches_model().
    """
    if isinstance(header, dict) and header:
        from .generators.factory import get_model_strategies
        for strat in get_model_strategies():
            if strat.matches_model(header, filename=filepath):
                if strat.family == "sdxl" and "pony" in Path(filepath).name.lower():
                    return "pony"
                return strat.family

    return detect_checkpoint_architecture(filepath)


def detect_checkpoint_engine(filepath_or_name: Any) -> str:
    """
    Inspects a checkpoint/UNET file on disk (via safetensors header) or by filename
    and returns its canonical architecture ('sdxl', 'pony', 'flux1', 'flux2', 'wan', 'ltx', 'sd15').
    """
    p_str = str(filepath_or_name or "")
    if p_str and Path(p_str).is_file():
        header = read_safetensors_header(p_str)
        if header:
            return detect_model_architecture_from_header(header, filepath=p_str)
    return detect_checkpoint_architecture(p_str)


def clear_cache() -> None:
    """No-op cache clearer for test suite compatibility."""
    return None


def detect_lora_architecture(
    meta: Dict[str, Any],
    filepath: str = "",
    header: Optional[Dict[str, Any]] = None
) -> str:
    """
    Detects LoRA base model architecture: 'sd15', 'sdxl', 'flux1', 'flux2', 'wan', 'ltx', or 'unknown'
    by querying polymorphic Generator strategies via BaseGenerator.matches_lora().
    """
    if not isinstance(meta, dict):
        meta = {}

    from .generators.factory import get_model_strategies
    for strat in get_model_strategies():
        if strat.matches_lora(header=header, meta=meta, filename=filepath):
            return strat.family

    # SD 1.5 fallback
    base_ver = str(meta.get("ss_base_model_version") or "").lower()
    fn = Path(filepath).name.lower() if filepath else ""
    if any(x in f"{base_ver} {fn}" for x in ("v1-5", "sd15", "sd_1_5", "1.5")):
        return "sd15"
    if header and isinstance(header, dict):
        sample = " ".join([k for k in header.keys() if k != "__metadata__"][:120]).lower()
        if "lora_te_" in sample and "lora_te2_" not in sample:
            return "sd15"

    return "unknown"


def detect_checkpoint_architecture(model_name: str) -> str:
    """
    Infers architecture category of a checkpoint from its identifier:
    'pony', 'sdxl', 'flux1', 'flux2', 'wan', 'ltx', 'sd15'.
    """
    m = str(model_name or "").lower()
    from .generators.factory import get_model_strategies
    for strat in get_model_strategies():
        if strat.matches_model({}, filename=m):
            if strat.family == "sdxl" and "pony" in m:
                return "pony"
            return strat.family
    if "hunyuan" in m:
        return "hunyuan"
    if any(x in m for x in ("v1-5", "sd15", "sd_1_5", "1.5")):
        return "sd15"
    return "sdxl"
