#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI runner for media_mcp tools.
Usage:
    cli.cmd [--config path] <tool_name> [request_path | -]
    echo '{"prompt": "..."}' | cli.cmd <tool_name> -
"""

import sys
import json
import os
from pathlib import Path

# Extract --config or -c before loading any config-dependent modules
config_path = None
filtered_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    if sys.argv[i] in ("--config", "-c") and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        i += 2
    else:
        filtered_argv.append(sys.argv[i])
        i += 1

if config_path:
    os.environ["MEDIA_CONFIG"] = config_path

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.tools import execute_tool, CANONICAL_TOOLS, load_spec_file

args = filtered_argv[1:]

if not args or args[0] in ("-h", "--help", "help"):
    print("Usage: cli.cmd [--config path] <tool_name> [request_path | -]")
    print("\nAvailable tools:")
    for t in CANONICAL_TOOLS:
        print(f"  - {t}")
    print("\nExamples:")
    print("  cli.cmd video_validate examples/scenario.json")
    print("  cli.cmd image_generate examples/mcp_image_generate.json")
    print("  cli.cmd --config my_config.json video_validate examples/scenario.json")
    print("  cli.cmd media_get_status")
    print("  echo '{\"prompt\": \"...\"}' | cli.cmd image_generate -")
    sys.exit(0)


def cli_progress(message: str, progress=None, total=None):
    """Prints progress notifications to stderr so stdout remains clean."""
    if progress is not None and total:
        pct = int((progress / total) * 100)
        sys.stderr.write(f"\r[{pct}%] {message}")
    else:
        sys.stderr.write(f"\n[*] {message}\n")
    sys.stderr.flush()


def main():
    tool_name = args[0]
    if tool_name not in CANONICAL_TOOLS:
        sys.stderr.write(f"Error: Unknown tool '{tool_name}'. Available: {CANONICAL_TOOLS}\n")
        return 1

    request_arg = args[1] if len(args) > 1 else None

    # Resolve payload from stdin, JSON string, or file on disk
    payload = {}
    if request_arg == "-":
        try:
            raw_input = sys.stdin.read().strip().lstrip("\ufeff")
            payload = json.loads(raw_input) if raw_input else {}
        except Exception as ex:
            sys.stderr.write(f"Error: Failed to parse JSON from stdin: {ex}\n")
            return 1
    elif request_arg:
        raw_str = request_arg.strip().lstrip("\ufeff")
        if raw_str.startswith("{") and raw_str.endswith("}"):
            try:
                payload = json.loads(raw_str)
            except Exception as ex:
                sys.stderr.write(f"Error: Failed to parse JSON argument: {ex}\n")
                return 1
        else:
            try:
                payload = load_spec_file(request_arg)
            except Exception as ex:
                sys.stderr.write(f"Error: {ex}\n")
                return 1
    else:
        payload = {}

    # Delegate execution to the unified core tool runner
    res = execute_tool(tool_name, payload, progress_callback=cli_progress)

    if res.get("isError"):
        err_msg = res.get("error") or "Execution failed"
        sys.stderr.write(f"\n[Error] {err_msg}\n")
        return 1

    content = res.get("content", [])
    for item in content:
        if item.get("type") == "text":
            print(item.get("text", ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
