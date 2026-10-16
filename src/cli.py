#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI runner for media_mcp tools.
Usage:
    bin/cli.sh [--config path] <tool_name> [request.json | -]   (Unix)
    bin\cli.cmd [--config path] <tool_name> [request.json | -]  (Windows)
    bin/cli.sh <tool_name> --help
    bin/cli.sh --help
"""

import sys
import json
import os
from pathlib import Path

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

from core import config, schema_formatter, schema_validator
from core.tools import execute_tool, CANONICAL_TOOLS, load_spec_file

args = filtered_argv[1:]


def cli_progress(message: str, progress=None, total=None):
    """Prints progress notifications to stderr so stdout remains clean JSON."""
    if progress is not None and total:
        pct = int((progress / total) * 100)
        sys.stderr.write(f"\r[{pct}%] {message}")
    else:
        sys.stderr.write(f"\n[*] {message}\n")
    sys.stderr.flush()


def main() -> int:
    # 1. Global help invocation (no config required)
    if not args or args[0] in ("-h", "--help", "help"):
        print(schema_formatter.format_general_help())
        return 0

    tool_name = args[0]

    # 2. Tool-specific help invocation (no config required)
    if len(args) > 1 and args[1] in ("-h", "--help", "help"):
        print(schema_formatter.format_tool_help(tool_name))
        return 0

    if tool_name not in CANONICAL_TOOLS:
        sys.stderr.write(f"Error: Unknown tool '{tool_name}'. Available tools: {CANONICAL_TOOLS}\n")
        sys.stderr.write("Run 'cmd\\cli.cmd --help' to view documentation.\n")
        return 1

    # 3. Explicit configuration check (Fail-Fast: required for all tool executions)
    if not config.is_configured():
        sys.stderr.write(
            f"Error: Tool '{tool_name}' requires a configuration file.\n"
            f"Specify --config <path_to_config.json> or set the MEDIA_CONFIG environment variable.\n"
            f"Run 'cmd\\cli.cmd {tool_name} --help' to view parameter requirements.\n"
        )
        return 1

    request_arg = args[1] if len(args) > 1 else None

    # 4. Strict payload resolution: only '-' (stdin) or specification file on disk
    payload = {}
    if request_arg == "-":
        try:
            raw_input = sys.stdin.read().strip().lstrip("\ufeff")
            payload = json.loads(raw_input) if raw_input else {}
        except Exception as ex:
            sys.stderr.write(f"Error: Failed to parse JSON from stdin: {ex}\n")
            return 1
    elif request_arg:
        p = Path(request_arg)
        if not p.is_file():
            sys.stderr.write(
                f"Error: Invalid argument '{request_arg}'.\n"
                f"Payload must be a path to an existing JSON specification file or '-' to read from stdin.\n"
                f"Direct inline JSON strings are not accepted on the command line.\n"
            )
            return 1
        try:
            payload = load_spec_file(request_arg)
        except Exception as ex:
            sys.stderr.write(f"Error loading specification file '{request_arg}': {ex}\n")
            return 1
    else:
        req_schema = schema_validator.get_mcp_tool_schema(tool_name)
        if req_schema.get("required"):
            sys.stderr.write(
                f"Error: Tool '{tool_name}' requires a specification file on disk or '-' to read from stdin.\n"
                f"Usage:\n"
                f"    bin/cli.sh {tool_name} <request.json | ->   (Unix)\n"
                f"    bin\\cli.cmd {tool_name} <request.json | ->  (Windows)\n\n"
                f"Tip: If piping JSON via pipe, pass '-' as the argument:\n"
                f"    echo '{{...}}' | bin\\cli.cmd {tool_name} -\n"
                f"Run 'bin\\cli.cmd {tool_name} --help' to view parameter requirements.\n"
            )
            return 1
        payload = {}

    # 5. Delegate execution to the unified core tool runner
    res = execute_tool(tool_name, payload, progress_callback=cli_progress)

    if res.get("isError"):
        err_msg = res.get("error") or "Execution failed"
        sys.stderr.write(f"\n[Error] {err_msg}\n")
        return 1

    # 6. Always emit clean, pure formatted JSON to stdout
    clean_output = {k: v for k, v in res.items() if k not in ("content", "isError")}
    print(json.dumps(clean_output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
