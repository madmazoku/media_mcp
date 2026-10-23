#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized JSON Schema loader and validator for media-mcp.
All schemas live in <repo_root>/schemas/*.json and use JSON Schema Draft 2020-12.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"

MCP_TOOL_SCHEMA_MAP: Dict[str, str] = {
    "image_generate": "requests/image_generate.json",
    "video_generate": "requests/video_generate.json",
    "media_get_job": "requests/media_get_job.json",
    "media_remove_job": "requests/media_remove_job.json",
    "media_get_models": "requests/media_get_models.json",
    "media_get_samplers": "requests/media_get_samplers.json",
    "media_get_loras": "requests/media_get_loras.json",
    "media_get_status": "requests/media_get_status.json",
    "media_interrupt": "requests/media_interrupt.json",
    "media_reload_modules": "requests/media_reload_modules.json",
}

MCP_RESPONSE_SCHEMA_MAP: Dict[str, str] = {
    "image_generate": "responses/image_generate.json",
    "video_generate": "responses/video_generate.json",
    "media_get_job": "responses/media_get_job.json",
    "media_remove_job": "responses/media_remove_job.json",
    "media_get_models": "responses/media_get_models.json",
    "media_get_samplers": "responses/media_get_samplers.json",
    "media_get_loras": "responses/media_get_loras.json",
    "media_get_status": "responses/media_get_status.json",
    "media_interrupt": "responses/media_interrupt.json",
    "media_reload_modules": "responses/media_reload_modules.json",
}

_RAW_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}
_INLINED_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


class SchemaValidationError(ValueError):
    """Raised when a JSON payload fails JSON Schema validation."""

    def __init__(self, context: str, errors: List[str]):
        self.context = context
        self.errors = errors
        joined = "; ".join(errors)
        super().__init__(f"[{context}] Schema validation failed: {joined}")


def _load_raw_schema(filename: str) -> Dict[str, Any]:
    if filename not in _RAW_SCHEMA_CACHE:
        schema_path = (SCHEMAS_DIR / filename).resolve()
        if not schema_path.is_file():
            candidates = [
                SCHEMAS_DIR / filename,
                SCHEMAS_DIR / "requests" / filename,
                SCHEMAS_DIR / "responses" / filename,
                SCHEMAS_DIR / Path(filename).name,
            ]
            found = False
            for cand in candidates:
                if cand.is_file():
                    schema_path = cand
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"JSON Schema file not found: {filename} (searched in {SCHEMAS_DIR})")
        with open(schema_path, "r", encoding="utf-8") as f:
            _RAW_SCHEMA_CACHE[filename] = json.load(f)
    return copy.deepcopy(_RAW_SCHEMA_CACHE[filename])


def _resolve_json_pointer(doc: Dict[str, Any], pointer: str) -> Any:
    """Resolve a JSON Pointer like '/$defs/image_config' within doc."""
    if not pointer or pointer == "/":
        return copy.deepcopy(doc)
    parts = pointer.lstrip("/").split("/")
    curr: Any = doc
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            raise KeyError(f"Cannot resolve JSON pointer '{pointer}' at '{part}'")
    return copy.deepcopy(curr)


def _inline_refs(node: Any, current_root: Dict[str, Any]) -> Any:
    """Recursively inline all $ref targets (both internal #/$defs/... and external file refs)."""
    if isinstance(node, list):
        return [_inline_refs(item, current_root) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref_str = node["$ref"]
        sibling_keys = {k: v for k, v in node.items() if k != "$ref"}

        if ref_str.startswith("#"):
            pointer = ref_str[1:]
            target = _resolve_json_pointer(current_root, pointer)
            resolved = _inline_refs(target, current_root)
        else:
            if "#" in ref_str:
                ext_file, frag = ref_str.split("#", 1)
            else:
                ext_file, frag = ref_str, ""
            ext_root = _load_raw_schema(ext_file)
            target = _resolve_json_pointer(ext_root, frag)
            # Strip root-only metadata when embedding a full external schema
            if isinstance(target, dict) and not frag:
                target = {
                    k: v
                    for k, v in target.items()
                    if k not in ("$schema", "$id", "$defs")
                }
            resolved = _inline_refs(target, ext_root)

        if isinstance(resolved, dict) and sibling_keys:
            merged = copy.deepcopy(resolved)
            merged.update(_inline_refs(sibling_keys, current_root))
            return merged
        return resolved

    result: Dict[str, Any] = {}
    for k, v in node.items():
        if k == "$defs":
            continue
        result[k] = _inline_refs(v, current_root)
    return result


def load_schema(filename: str) -> Dict[str, Any]:
    """Load and return a fully inlined JSON Schema from schemas/<filename>."""
    if filename not in _INLINED_SCHEMA_CACHE:
        raw = _load_raw_schema(filename)
        inlined = _inline_refs(raw, raw)
        Draft202012Validator.check_schema(inlined)
        _INLINED_SCHEMA_CACHE[filename] = inlined
    return copy.deepcopy(_INLINED_SCHEMA_CACHE[filename])


def get_mcp_tool_schema(tool_name: str) -> Dict[str, Any]:
    """Return an MCP-compatible inputSchema dict for the given tool name."""
    if tool_name not in MCP_TOOL_SCHEMA_MAP:
        raise KeyError(f"No JSON schema registered for MCP tool '{tool_name}'")
    schema = load_schema(MCP_TOOL_SCHEMA_MAP[tool_name])
    # Remove top-level JSON Schema metadata not needed in MCP inputSchema
    for meta_key in ("$schema", "$id", "title"):
        schema.pop(meta_key, None)
    return schema


def _format_path(error_path) -> str:
    parts: List[str] = []
    for elem in error_path:
        if isinstance(elem, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{elem}]"
            else:
                parts.append(f"[{elem}]")
        else:
            parts.append(str(elem))
    return ".".join(parts) if parts else "<root>"


def _format_validation_error(err) -> str:
    path_str = _format_path(err.absolute_path)
    if err.validator == "not" and isinstance(err.validator_value, dict):
        # Provide clearer message for engine-conditional forbidden properties
        not_schema = err.validator_value
        forbidden_props: List[str] = []
        if "required" in not_schema:
            forbidden_props.extend(not_schema["required"])
        elif "anyOf" in not_schema:
            for sub in not_schema["anyOf"]:
                if isinstance(sub, dict) and "required" in sub:
                    forbidden_props.extend(sub["required"])
        if forbidden_props and isinstance(err.instance, dict):
            present = [p for p in forbidden_props if p in err.instance]
            engine = err.instance.get("engine", "selected engine")
            if present:
                return (
                    f"{path_str}: Properties {present} are not allowed when engine='{engine}'"
                )
    return f"{path_str}: {err.message}"


def collect_schema_errors(schema_filename: str, instance: Any) -> List[str]:
    """Validate instance against schema_filename and return a list of formatted error strings."""
    schema = load_schema(schema_filename)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [_format_validation_error(e) for e in errors]


def validate_against_schema(schema_filename: str, instance: Any, context: str) -> None:
    """Validate instance against schema_filename or raise SchemaValidationError."""
    errors = collect_schema_errors(schema_filename, instance)
    if errors:
        raise SchemaValidationError(context, errors)


def validate_config(cfg_data: Any) -> None:
    """Validate config.json dictionary against schemas/config.json."""
    validate_against_schema("config.json", cfg_data, "config.json")


def validate_scenario(scenario_data: Any) -> None:
    """Validate scenario dictionary against schemas/scenario.json."""
    validate_against_schema("scenario.json", scenario_data, "scenario")


def validate_image_spec(image_spec_data: Any) -> None:
    """Validate image specification dictionary against schemas/requests/image_generate.json."""
    validate_against_schema("requests/image_generate.json", image_spec_data, "image_spec")


def validate_mcp_args(tool_name: str, tool_args: Any) -> None:
    """Validate MCP tool call arguments against the tool's schema."""
    if tool_name not in MCP_TOOL_SCHEMA_MAP:
        raise KeyError(f"Unknown MCP tool '{tool_name}'")
    schema_file = MCP_TOOL_SCHEMA_MAP[tool_name]
    validate_against_schema(schema_file, tool_args, f"MCP tool '{tool_name}'")


def get_mcp_response_schema(tool_name: str) -> Dict[str, Any]:
    """Return inlined JSON Schema dict for a tool's response structure."""
    if tool_name not in MCP_RESPONSE_SCHEMA_MAP:
        raise KeyError(f"No response schema registered for MCP tool '{tool_name}'")
    schema = load_schema(MCP_RESPONSE_SCHEMA_MAP[tool_name])
    for meta_key in ("$schema", "$id", "title"):
        schema.pop(meta_key, None)
    return schema


def validate_mcp_response(tool_name: str, response_data: Any) -> None:
    """Validate MCP tool return value against its response schema."""
    if tool_name not in MCP_RESPONSE_SCHEMA_MAP:
        return
    schema_file = MCP_RESPONSE_SCHEMA_MAP[tool_name]
    validate_against_schema(schema_file, response_data, f"MCP tool response '{tool_name}'")
