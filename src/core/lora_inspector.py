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
    Determines model architecture ('sdxl', 'pony', 'flux1', 'flux2', 'wan', 'ltx', 'sd15')
    by inspecting tensor names/shapes in the .safetensors header first, falling back to metadata/filename.
    """
    meta = header.get("__metadata__", {}) if isinstance(header, dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    tensor_keys = [k for k in header.keys() if k != "__metadata__"] if isinstance(header, dict) else []
    fn_lower = Path(filepath).name.lower() if filepath else ""
    arch_meta = str(meta.get("modelspec.architecture") or meta.get("ss_base_model_version") or "").lower()

    if tensor_keys:
        sample_keys = " ".join(tensor_keys[:200]).lower()
        # Wan 2.1 / 2.2 DiT
        if "cross_attn.k_img" in sample_keys or ("patch_embedding.weight" in sample_keys and "blocks.0.self_attn" in sample_keys):
            return "wan"
        # LTX-Video DiT
        if "adaln_single" in sample_keys or "patchify_proj" in sample_keys or "scale_shift_table" in sample_keys or ("transformer_blocks.0.attn1" in sample_keys and "caption_projection" in sample_keys):
            return "ltx"
        # FLUX.1 vs FLUX.2 Klein DiT
        if "double_blocks." in sample_keys or "single_blocks." in sample_keys:
            if "flux2" in fn_lower or "klein" in fn_lower or "flux2" in arch_meta or "klein" in arch_meta:
                return "flux2"
            for k in tensor_keys:
                if "double_blocks.0.img_attn.qkv.weight" in k:
                    t_info = header.get(k)
                    if isinstance(t_info, dict) and isinstance(t_info.get("shape"), list) and len(t_info["shape"]) >= 2:
                        in_dim = int(t_info["shape"][1])
                        if in_dim == 3072:
                            return "flux2"
                        if in_dim == 4096:
                            return "flux1"
            img_in = header.get("img_in.weight") or header.get("model.diffusion_model.img_in.weight")
            if isinstance(img_in, dict) and isinstance(img_in.get("shape"), list) and len(img_in["shape"]) >= 1:
                if int(img_in["shape"][0]) == 3072:
                    return "flux2"
                if int(img_in["shape"][0]) == 4096:
                    return "flux1"
            has_block_18 = any("double_blocks.18." in k for k in tensor_keys)
            if not has_block_18 and any("double_blocks.0." in k for k in tensor_keys):
                return "flux2"
            return "flux1"
        # SDXL / Pony UNet
        if any("conditioner.embedders.1" in k or "label_emb.0.0.weight" in k for k in tensor_keys):
            return "pony" if "pony" in fn_lower else "sdxl"

    return detect_checkpoint_architecture(filepath or arch_meta)


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
    Detects LoRA base model architecture: 'sd15', 'sdxl', 'flux1', 'flux2', 'wan', 'ltx', or 'unknown'.
    Inspects metadata, safetensors tensor key patterns and shapes, and filename.
    """
    if not isinstance(meta, dict):
        meta = {}
    base_ver = str(meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or "").lower()
    sd_model_name = str(meta.get("ss_sd_model_name") or "").lower()
    fn = Path(filepath).name.lower() if filepath else ""

    tensor_keys: List[str] = []
    if isinstance(header, dict):
        tensor_keys = [k for k in header.keys() if k != "__metadata__"]

    if tensor_keys and isinstance(header, dict):
        sample = " ".join(tensor_keys[:120]).lower()
        if "cross_attn_k_img" in sample or "cross_attn.k_img" in sample or ("blocks_0_self_attn" in sample and "blocks_0_cross_attn" in sample):
            return "wan"
        if "adaln_single" in sample or "caption_projection" in sample or "transformer_blocks_0_attn1" in sample:
            return "ltx"
        if "double_blocks" in sample or "single_blocks" in sample:
            for k in tensor_keys:
                if "lora_down.weight" in k and ("double_blocks" in k or "single_blocks" in k):
                    t_info = header.get(k)
                    if isinstance(t_info, dict) and isinstance(t_info.get("shape"), list) and len(t_info["shape"]) >= 2:
                        in_dim = int(t_info["shape"][1])
                        if in_dim == 3072:
                            return "flux2"
                        if in_dim == 4096:
                            return "flux1"
            if "flux2" in base_ver or "klein" in base_ver or "flux2" in fn or "klein" in fn:
                return "flux2"
            return "flux1"
        if "lora_te1_" in sample or "lora_te2_" in sample or "conditioner_embedders" in sample:
            return "sdxl"
        if "lora_te_" in sample and "lora_te2_" not in sample:
            return "sd15"

    combined = f"{base_ver} {sd_model_name} {fn}"

    if any(x in combined for x in ("wan", "wan2.1", "wan2.2", "lightx2v")):
        return "wan"
    if any(x in combined for x in ("ltx", "ltxv", "sulphur", "sulfur")):
        return "ltx"
    if any(x in combined for x in ("flux2", "klein")):
        return "flux2"
    if any(x in combined for x in ("flux", "flux1")):
        return "flux1"
    if any(x in combined for x in ("xl", "sdxl", "pony", "illustrious", "animagine")):
        return "sdxl"
    if any(x in combined for x in ("v1-5", "sd15", "sd_1_5", "1.5")):
        return "sd15"

    return "unknown"


def detect_checkpoint_architecture(model_name: str) -> str:
    """
    Infers architecture category of a checkpoint from its identifier:
    'pony', 'sdxl', 'flux1', 'flux2', 'wan', 'ltx', 'sd15'.
    """
    m = str(model_name or "").lower()
    if "pony" in m:
        return "pony"
    if "flux2" in m or "klein" in m:
        return "flux2"
    if "flux" in m:
        return "flux1"
    if "wan" in m:
        return "wan"
    if any(x in m for x in ("ltx", "sulphur", "sulfur")):
        return "ltx"
    if any(x in m for x in ("xl", "illustrious", "animagine", "cyberrealistic")):
        return "sdxl"
    if any(x in m for x in ("v1-5", "1.5")):
        return "sd15"
    return "sdxl"
