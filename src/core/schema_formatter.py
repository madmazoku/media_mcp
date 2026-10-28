#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal-friendly documentation and help formatter for media_mcp tools.
Inspects JSON Schema input and response specifications and produces
clean, columnar, human-readable CLI help text without external dependencies.
"""

import sys
import json
import textwrap
from typing import Any, Dict, List, Optional

from core import schema_validator


def get_launcher_cmd() -> str:
    """Returns platform-native CLI launcher path."""
    if sys.platform == "win32":
        return r"bin\cli.cmd"
    return "bin/cli.sh"



TOOL_DESCRIPTIONS: Dict[str, str] = {
    "image_generate": "Generate high-quality images via SDXL/Pony (Automatic1111) or FLUX.1/FLUX.2 (ComfyUI) using a unified image_spec dictionary.",
    "video_generate": "Generate multi-scene or single-block videos asynchronously using Wan 2.2 Fast (wan-fast), Wan 2.2 Full (wan-full), or LTX-Video (ltx) with 5D Latent-Chaining. Synchronously validates and returns run_prefix immediately.",
    "media_get_job": "Poll or retrieve the status, progress, and completed video/poster outputs of an asynchronous video generation job by run_prefix. Supports blocking wait.",
    "media_remove_job": "Cancel a queued or active video generation job, or remove completed job metadata from the server registry.",
    "media_get_models": "Inspect installed checkpoint and UNET models across image and video engines. Returns invariant tree structure filtered by type or engine.",
    "media_get_samplers": "Get recommended and compatible samplers for a specific engine and model. Index 0 in returned list is the recommended default.",
    "media_get_loras": "Inspect compatible LoRA models and their trained activation triggers for a specific engine and model. Supports optional 'pattern' filter.",
    "media_get_status": "Check cluster health, GPU lock status, active job progress, and backend availability across ComfyUI and Automatic1111.",
    "media_interrupt": "Emergency stop: interrupt and cancel currently running image and video generations across all engines.",
    "media_reload_modules": "Developer tool: hot-reload all python modules under 'core' in memory on demand.",
}

TOOL_EXAMPLES_INPUT: Dict[str, Dict[str, Any]] = {
    "image_generate": {
        "engine": "flux2",
        "model": "flux2-klein-9b.safetensors",
        "prompt": "detailed 3D character render, cute blonde mechanic girl in workshop, 8k",
        "width": 1280,
        "height": 720
    },
    "video_generate": {
        "scenario_path": "examples/scenario.json"
    },
    "media_get_job": {
        "run_prefix": "20261023-205500-abc123",
        "wait": 10
    },
    "media_remove_job": {
        "run_prefix": "20261023-205500-abc123"
    },
    "media_get_models": {
        "type": "image",
        "engine": "flux2"
    },
    "media_get_samplers": {
        "engine": "sdxl",
        "model": "ponyDiffusionV6XL_v6StartWithThis.safetensors"
    },
    "media_get_loras": {
        "engine": "sdxl",
        "model": "ponyDiffusionV6XL_v6StartWithThis.safetensors",
        "pattern": "cyber.*"
    },
    "media_get_status": {},
    "media_interrupt": {},
    "media_reload_modules": {}
}

TOOL_EXAMPLES_OUTPUT: Dict[str, Dict[str, Any]] = {
    "image_generate": {
        "run_prefix": "20261017-002500-a1b2c3",
        "image_path": "./workspace/images/20261017-002500-a1b2c3-final.png",
        "spec_path": "./workspace/workflows/20261017-002500-a1b2c3-spec.json",
        "engine": "flux2",
        "model": "flux2-klein-9b.safetensors",
        "seed": 79192059900392,
        "width": 1280,
        "height": 720,
        "native_width": 1344,
        "native_height": 768
    },
    "video_generate": {
        "run_prefix": "20261023-205500-abc123",
        "status": "queued",
        "spec_path": "./workspace/workflows/20261023-205500-abc123-spec.json"
    },
    "media_get_job": {
        "run_prefix": "20261023-205500-abc123",
        "status": "completed",
        "spec_path": "./workspace/workflows/20261023-205500-abc123-spec.json",
        "timestamps": {
            "created_at": 1729700000.0,
            "started_at": 1729700001.0,
            "completed_at": 1729700065.0,
            "elapsed_seconds": 64.0
        },
        "progress": {
            "phase": "completed",
            "percent": 100,
            "scene_index": 1,
            "total_scenes": 1,
            "block_index": 2,
            "total_blocks": 2,
            "message": "Scenario video assembly completed successfully"
        },
        "video": {
            "video_path": "./workspace/videos/20261023-205500-abc123-final.mp4",
            "duration": 9.8,
            "width": 832,
            "height": 480,
            "fps": 16,
            "scenes_count": 1,
            "blocks_count": 2
        },
        "image": {
            "image_path": "./workspace/images/20261023-205500-abc123-poster.png",
            "seed": 79192059900392,
            "width": 832,
            "height": 480,
            "native_width": 1280,
            "native_height": 720
        }
    },
    "media_remove_job": {
        "run_prefix": "20261023-205500-abc123",
        "status": "removed",
        "message": "Finished job '20261023-205500-abc123' (status was: completed) was removed from registry."
    },
    "media_get_models": {
        "image": {
            "flux2": ["flux2-klein-9b.safetensors"]
        }
    },
    "media_get_samplers": {
        "samplers": ["Euler a", "Euler", "DPM++ 2M Karras"]
    },
    "media_get_loras": {
        "loras": [
            {"name": "cyber_suit_v1", "triggers": ["cyber suit", "glowing lines"]}
        ]
    },
    "media_get_status": {
        "pipeline": {"is_running": False},
        "comfy_online": True,
        "a1111_available": True,
        "report": "ComfyUI: ONLINE | A1111: ONLINE"
    },
    "media_interrupt": {
        "interrupted": True,
        "message": "Interrupt signals dispatched to ComfyUI, A1111 WebUI, and Active Video Pipelines."
    },
    "media_reload_modules": {
        "reloaded_modules": ["core.config", "core.tools"],
        "count": 2
    }
}

TOOL_CONSTRAINTS: Dict[str, List[str]] = {
    "image_generate": [
        "Parameters 'engine', 'model', and 'prompt' are REQUIRED.",
        "Parameter 'fit' ('crop' or 'outpaint') requires 'image_path', 'width', and 'height'.",
        "When 'image_path' is specified without 'fit', 'width' and 'height' must NOT be provided.",
        "Parameters 'denoise' and 'mask_path' are only allowed when 'image_path' is specified.",
        "Engine 'flux2' does NOT support 'loras'.",
        "Engines 'flux1' and 'flux2' do NOT accept 'cfg' or 'negative_prompt' (use 'guidance' instead)."
    ],
    "video_generate": [
        "Requires either 'scenario_path' or inline 'video' and 'scenes' (or both for overlay merges).",
        "Technical parameters ('engine', 'model', 'steps', 'cfg', 'seed', 'ramp_latents', 'min_cfg', 'weight_dtype') must be placed in root '.image' and '.video' sections.",
        "Multi-block scenes require sequential duration constraints (prefix_duration + clip_duration < max_video_duration)."
    ],
    "media_get_job": [
        "Parameter 'run_prefix' is REQUIRED.",
        "Parameter 'wait' is optional integer seconds (0-600) to block awaiting completion."
    ],
    "media_remove_job": [
        "Parameter 'run_prefix' is REQUIRED.",
        "Cancels in-flight GPU execution if active, or purges completed job metadata."
    ],
    "media_get_models": [
        "Filter 'type' allows: 'image' or 'video'.",
        "Filter 'engine' allows: 'sdxl', 'flux1', 'flux2', 'wan-fast', 'wan-full', 'ltx'."
    ],
    "media_get_samplers": [
        "Both 'engine' and 'model' parameters are REQUIRED."
    ],
    "media_get_loras": [
        "Both 'engine' and 'model' parameters are REQUIRED.",
        "Optional 'pattern' is a case-insensitive Python regular expression filtering by LoRA name or triggers."
    ]
}


def format_general_help() -> str:
    """Returns general CLI usage, tool index, and invocation rules."""
    cmd = get_launcher_cmd()
    lines = [
        "=" * 78,
        "media_mcp - Unified Image and Video CLI & MCP Runner",
        "=" * 78,
        "",
        "USAGE:",
        f"  {cmd} [--config <path>] <tool_name> [request.json | -]",
        f"  {cmd} <tool_name> --help",
        f"  {cmd} --help",
        f"  {cmd} --version (-v)",
        "",
        "CONFIGURATION:",
        "  Requires explicit configuration path via:",
        "    1. CLI argument: --config <path_to_config.json> (or -c)",
        "    2. Environment:  set MEDIA_CONFIG=<path_to_config.json>",
        "  Zero heuristic auto-discovery in CWD. Viewing help does not require configuration.",
        "",
        "AVAILABLE TOOLS:"
    ]

    for tool_name, desc in TOOL_DESCRIPTIONS.items():
        lines.append(f"  {tool_name:<22} {desc}")

    lines.extend([
        "",
        "EXECUTION MODES:",
        "  1. Read specification from file:",
        f"     {cmd} --config config.json image_generate request.json",
        "",
        "  2. Read specification from stdin (hyphen '-'):",
        f"     cat request.json | {cmd} --config config.json image_generate -",
        "",
        "  3. Tools without parameters (omit payload argument):",
        f"     {cmd} --config config.json media_get_status",
        "",
        "TOOL DOCUMENTATION:",
        f"  Run '{cmd} <tool_name> --help' to view parameters, constraints, and JSON examples.",
        "=" * 78
    ])
    return "\n".join(lines)

def _format_property_block(
    prop_name: str,
    prop_spec: Dict[str, Any],
    is_required: bool,
    indent_level: int = 1,
    is_response: bool = False,
    max_depth: int = 5
) -> List[str]:
    lines: List[str] = []
    base_indent = "  " * indent_level
    nested_indent = "  " * (indent_level + 1)

    # 1. Resolve type string
    raw_type = prop_spec.get("type")
    items_spec = prop_spec.get("items")

    if raw_type == "array":
        if isinstance(items_spec, dict):
            item_raw = items_spec.get("type")
            if item_raw == "object":
                type_str = "array of objects"
            elif item_raw:
                type_str = f"array of {item_raw}s"
            elif "$ref" in items_spec or "properties" in items_spec or "oneOf" in items_spec or "anyOf" in items_spec:
                type_str = "array of objects"
            else:
                type_str = "array"
        else:
            type_str = "array"
    elif raw_type == "object":
        type_str = "object"
    elif isinstance(raw_type, list):
        type_str = "/".join(raw_type)
    elif raw_type:
        type_str = str(raw_type)
    else:
        if "properties" in prop_spec or "oneOf" in prop_spec or "anyOf" in prop_spec:
            type_str = "object"
        else:
            type_str = "any"

    # 2. Attributes
    attrs = [f"[{type_str}]"]
    if is_response:
        if is_required:
            attrs.append("[required]")
    else:
        req_str = "REQUIRED" if is_required else "optional"
        attrs.append(f"({req_str})")

    if "default" in prop_spec:
        attrs.append(f"[default: {json.dumps(prop_spec['default'])}]")
    if "enum" in prop_spec:
        attrs.append(f"[choices: {', '.join(json.dumps(e) for e in prop_spec['enum'])}]")
    if "minimum" in prop_spec:
        attrs.append(f"[min: {prop_spec['minimum']}]")
    if "maximum" in prop_spec:
        attrs.append(f"[max: {prop_spec['maximum']}]")
    if "multipleOf" in prop_spec:
        attrs.append(f"[multipleOf: {prop_spec['multipleOf']}]")

    header_col_width = max(18 - (indent_level - 1) * 2, 8)
    lines.append(f"{base_indent}{prop_name:<{header_col_width}} {' '.join(attrs)}")

    # 3. Description
    p_desc = prop_spec.get("description", "").strip()
    if p_desc:
        for d_line in textwrap.wrap(p_desc, width=max(74 - (indent_level * 2), 40)):
            lines.append(f"{nested_indent}{d_line}")

    # 4. Nested structures
    if indent_level < max_depth:
        # A. Array of objects
        if raw_type == "array" and isinstance(items_spec, dict):
            sub_props = items_spec.get("properties")
            sub_req = set(items_spec.get("required", []))
            if sub_props:
                lines.append(f"{nested_indent}Item structure:")
                for sub_name, sub_s in sub_props.items():
                    sub_lines = _format_property_block(
                        sub_name,
                        sub_s,
                        is_required=(sub_name in sub_req),
                        indent_level=indent_level + 2,
                        is_response=is_response,
                        max_depth=max_depth
                    )
                    lines.extend(sub_lines)
            elif "oneOf" in items_spec or "anyOf" in items_spec:
                branches = items_spec.get("oneOf") or items_spec.get("anyOf") or []
                lines.append(f"{nested_indent}Item structure (variants):")
                for idx, branch in enumerate(branches):
                    b_desc = branch.get("description", f"Variant {idx + 1}")
                    lines.append(f"{nested_indent}  - Option {idx + 1}: {b_desc}")
                    b_props = branch.get("properties", {})
                    b_req = set(branch.get("required", []))
                    for b_name, b_s in b_props.items():
                        sub_lines = _format_property_block(
                            b_name,
                            b_s,
                            is_required=(b_name in b_req),
                            indent_level=indent_level + 2,
                            is_response=is_response,
                            max_depth=max_depth
                        )
                        lines.extend(sub_lines)

        # B. Object with properties
        elif (raw_type == "object" or "properties" in prop_spec) and prop_spec.get("properties"):
            sub_props = prop_spec.get("properties", {})
            sub_req = set(prop_spec.get("required", []))
            lines.append(f"{nested_indent}Nested fields:")
            for sub_name, sub_s in sub_props.items():
                sub_lines = _format_property_block(
                    sub_name,
                    sub_s,
                    is_required=(sub_name in sub_req),
                    indent_level=indent_level + 2,
                    is_response=is_response,
                    max_depth=max_depth
                )
                lines.extend(sub_lines)

    return lines


def format_tool_help(tool_name: str) -> str:
    """Formats comprehensive, columnar help for a specific tool."""
    if tool_name not in schema_validator.MCP_TOOL_SCHEMA_MAP:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_DESCRIPTIONS.keys())}"

    input_schema = schema_validator.get_mcp_tool_schema(tool_name)
    has_params = bool(input_schema.get("properties"))
    desc = input_schema.get("description") or TOOL_DESCRIPTIONS.get(tool_name, "")
    required_fields = set(input_schema.get("required", []))

    cmd = get_launcher_cmd()
    lines = [
        "=" * 78,
        f"TOOL: {tool_name}",
        "=" * 78,
        f"DESCRIPTION:",
        f"  {desc}",
        "",
        "USAGE:"
    ]

    if has_params:
        lines.append(f"  {cmd} [--config <path>] {tool_name} <request.json>")
        lines.append(f"  cat request.json | {cmd} [--config <path>] {tool_name} -")
    else:
        lines.append(f"  {cmd} [--config <path>] {tool_name}")
        lines.append("  (This tool takes no request payload)")

    # 1. Input Parameters Table
    lines.append("")
    lines.append("INPUT PARAMETERS:")
    if not has_params:
        lines.append("  (None)")
    else:
        props = input_schema.get("properties", {})
        for prop_name, prop_spec in props.items():
            lines.extend(
                _format_property_block(
                    prop_name=prop_name,
                    prop_spec=prop_spec,
                    is_required=(prop_name in required_fields),
                    indent_level=1,
                    is_response=False
                )
            )

    # 2. Conditions & Constraints
    constraints = TOOL_CONSTRAINTS.get(tool_name, [])
    if constraints:
        lines.append("")
        lines.append("CONDITIONS & CONSTRAINTS:")
        for c in constraints:
            for c_line in textwrap.wrap(f"- {c}", width=74, subsequent_indent="    "):
                lines.append(f"  {c_line}")

    # 3. Response Structure
    try:
        resp_schema = schema_validator.get_mcp_response_schema(tool_name)
        resp_props = resp_schema.get("properties", {})
        resp_req = set(resp_schema.get("required", []))
    except Exception:
        resp_props = {}
        resp_req = set()

    lines.append("")
    lines.append("OUTPUT FORMAT (RESPONSE JSON):")
    if not resp_props:
        lines.append("  Returns a JSON dictionary.")
    else:
        for r_name, r_spec in resp_props.items():
            lines.extend(
                _format_property_block(
                    prop_name=r_name,
                    prop_spec=r_spec,
                    is_required=(r_name in resp_req),
                    indent_level=1,
                    is_response=True
                )
            )

    # 4. Example Request
    example_in = TOOL_EXAMPLES_INPUT.get(tool_name)
    if example_in is not None and has_params:
        lines.append("")
        lines.append("EXAMPLE REQUEST (request.json):")
        lines.append(textwrap.indent(json.dumps(example_in, indent=2, ensure_ascii=False), "  "))

    # 5. Example Response
    example_out = TOOL_EXAMPLES_OUTPUT.get(tool_name)
    if example_out is not None:
        lines.append("")
        lines.append("EXAMPLE RESPONSE (stdout JSON):")
        lines.append(textwrap.indent(json.dumps(example_out, indent=2, ensure_ascii=False), "  "))

    lines.append("=" * 78)
    return "\n".join(lines)
