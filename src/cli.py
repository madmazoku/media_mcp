#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI runner for media_mcp tools.
Usage:
    bin/cli.sh [--config path] <tool_name> [request.json | -]   (Unix)
    bin\\cli.cmd [--config path] <tool_name> [request.json | -]  (Windows)
    bin/cli.sh <tool_name> --help
    bin/cli.sh --help
"""

from __future__ import annotations
import sys
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# Ensure standard I/O handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure package import from src/
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Extract --config or -c before loading any config-dependent modules
config_path = None
raw_filtered_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    if sys.argv[i] in ("--config", "-c") and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        i += 2
    else:
        raw_filtered_argv.append(sys.argv[i])
        i += 1

if config_path:
    os.environ["MEDIA_CONFIG"] = config_path

from core import config, schema_formatter
from core.tools import CORE_TOOLS, CoreTool, load_spec_file

args = raw_filtered_argv[1:]


def cli_progress(message: str, progress=None, total=None):
    """Prints progress notifications to stderr so stdout remains clean JSON."""
    if progress is not None and total:
        pct = int((progress / total) * 100)
        sys.stderr.write(f"\r[{pct}%] {message}")
    else:
        sys.stderr.write(f"\n[*] {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# CLI Execution Strategies (GoF Strategy Pattern)
# ---------------------------------------------------------------------------

class CliStrategy(ABC):
    @abstractmethod
    def execute(self, tool: CoreTool, payload: Dict[str, Any], progress_cb: Callable) -> int:
        """Executes the tool according to the strategy and returns process exit code."""
        pass


class DirectExecutionStrategy(CliStrategy):
    """Direct single-turn execution. Emits clean structured JSON to stdout."""

    def execute(self, tool: CoreTool, payload: Dict[str, Any], progress_cb: Callable) -> int:
        res = tool.execute(payload, progress_cb=progress_cb)

        if res.get("isError"):
            err_msg = res.get("error") or "Execution failed"
            sys.stderr.write(f"\n[Error] {err_msg}\n")
            return 1

        clean_output = {k: v for k, v in res.items() if k not in ("content", "isError")}
        print(json.dumps(clean_output, indent=2, ensure_ascii=False))
        return 0


class PollWaitJobStrategy(CliStrategy):
    """Asynchronous polling wait strategy with terminal progress display and clean exit."""

    def execute(self, tool: CoreTool, payload: Dict[str, Any], progress_cb: Callable) -> int:
        # 1. Trigger initial job submission or pre-flight validation
        res = tool.execute(payload, progress_cb=progress_cb)
        if res.get("isError"):
            err_msg = res.get("error") or "Execution failed"
            sys.stderr.write(f"\n[Error] {err_msg}\n")
            return 1

        run_prefix = res.get("run_prefix")
        status = res.get("status")

        # If not queued (e.g. validated on dry_run), output JSON and exit 0
        if status != "queued" or not run_prefix:
            clean_output = {k: v for k, v in res.items() if k not in ("content", "isError")}
            print(json.dumps(clean_output, indent=2, ensure_ascii=False))
            return 0

        sys.stderr.write(f"\n[*] Job submitted: {run_prefix}\n[*] Waiting for background GPU generation to complete...\n")
        sys.stderr.flush()

        # 2. Polling loop with media_get_job
        job_tool = CORE_TOOLS.get("media_get_job")
        remove_tool = CORE_TOOLS.get("media_remove_job")
        if not job_tool:
            sys.stderr.write("\n[Error] 'media_get_job' tool not found in core registry.\n")
            return 1

        last_phase = None
        last_pct = -1
        try:
            while True:
                job_res = job_tool.execute({"run_prefix": run_prefix, "wait": 1})
                if job_res.get("isError"):
                    sys.stderr.write(f"\n[Error] Failed to poll job: {job_res.get('error')}\n")
                    return 1

                st = job_res.get("status", "")
                st_str = st.upper() if st else "UNKNOWN"
                prog = job_res.get("progress", {})
                phase = prog.get("phase", "")
                pct = prog.get("percent", 0)
                msg = prog.get("message") or f"Phase: {phase}"

                if phase != last_phase or pct != last_pct:
                    progress_cb(f"[{st_str}] {msg}", pct, 100)
                    last_phase = phase
                    last_pct = pct

                if st == "completed":
                    sys.stderr.write(f"\n[+] Job '{run_prefix}' completed successfully.\n")
                    sys.stderr.flush()
                    # Clean up job from in-memory registry before exit
                    if remove_tool:
                        remove_tool.execute({"run_prefix": run_prefix})
                    clean_output = {k: v for k, v in job_res.items() if k not in ("content", "isError")}
                    print(json.dumps(clean_output, indent=2, ensure_ascii=False))
                    return 0
                elif st in ("failed", "cancelled"):
                    err_str = job_res.get("error") or f"Job ended with status '{st}'"
                    sys.stderr.write(f"\n[Error] Job '{run_prefix}' {st}: {err_str}\n")
                    # Clean up job from in-memory registry before exit
                    if remove_tool:
                        remove_tool.execute({"run_prefix": run_prefix})
                    return 1
                elif st == "not_found":
                    sys.stderr.write(f"\n[Error] Job '{run_prefix}' was not found in registry.\n")
                    return 1
        except KeyboardInterrupt:
            sys.stderr.write(f"\n\n[!] Interrupt received. Cancelling job '{run_prefix}' on GPU...\n")
            sys.stderr.flush()
            if remove_tool:
                remove_tool.execute({"run_prefix": run_prefix})
            sys.stderr.write(f"[!] Job '{run_prefix}' cancelled. Exiting.\n")
            sys.stderr.flush()
            return 130


# ---------------------------------------------------------------------------
# CliTool Metadata & Explicit Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CliTool:
    tool: CoreTool
    strategy: CliStrategy

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def requires_payload(self) -> bool:
        return bool(self.tool.input_schema.get("required"))

    def get_help(self) -> str:
        return schema_formatter.format_tool_help(self.tool.name)

    def execute(self, payload: Dict[str, Any], progress_cb: Callable) -> int:
        return self.strategy.execute(self.tool, payload, progress_cb=progress_cb)


DIRECT_STRATEGY = DirectExecutionStrategy()
POLL_WAIT_STRATEGY = PollWaitJobStrategy()

CLI_TOOLS: Dict[str, CliTool] = {
    "image_generate": CliTool(
        tool=CORE_TOOLS["image_generate"],
        strategy=DIRECT_STRATEGY,
    ),
    "video_generate": CliTool(
        tool=CORE_TOOLS["video_generate"],
        strategy=POLL_WAIT_STRATEGY,
    ),
    "media_get_job": CliTool(
        tool=CORE_TOOLS["media_get_job"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_remove_job": CliTool(
        tool=CORE_TOOLS["media_remove_job"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_get_models": CliTool(
        tool=CORE_TOOLS["media_get_models"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_get_samplers": CliTool(
        tool=CORE_TOOLS["media_get_samplers"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_get_loras": CliTool(
        tool=CORE_TOOLS["media_get_loras"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_get_status": CliTool(
        tool=CORE_TOOLS["media_get_status"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_interrupt": CliTool(
        tool=CORE_TOOLS["media_interrupt"],
        strategy=DIRECT_STRATEGY,
    ),
    "media_reload_modules": CliTool(
        tool=CORE_TOOLS["media_reload_modules"],
        strategy=DIRECT_STRATEGY,
    ),
}


def verify_cli_tools_integrity():
    """Asserts bidirectional parity between CoreTool and CliTool registries."""
    core_keys = set(CORE_TOOLS.keys())
    cli_keys = set(CLI_TOOLS.keys())
    missing_in_cli = core_keys - cli_keys
    extra_in_cli = cli_keys - core_keys
    if missing_in_cli or extra_in_cli:
        raise RuntimeError(
            f"CLI tools mismatch with Core tools!\n"
            f"Missing in CLI: {missing_in_cli}\n"
            f"Extra in CLI: {extra_in_cli}"
        )


verify_cli_tools_integrity()


# ---------------------------------------------------------------------------
# Main CLI Dispatcher
# ---------------------------------------------------------------------------

def main() -> int:
    # 0. Global version invocation (no config required)
    if args and args[0] in ("-v", "--version", "version"):
        from core import __version__
        print(f"media-mcp {__version__}")
        return 0

    # 1. Global help invocation (no config required)
    if not args or args[0] in ("-h", "--help", "help"):
        print(schema_formatter.format_general_help())
        return 0

    launcher = schema_formatter.get_launcher_cmd()
    tool_name = args[0]

    # 2. Tool-specific help invocation (no config required)
    if len(args) > 1 and args[1] in ("-h", "--help", "help"):
        cli_t = CLI_TOOLS.get(tool_name)
        if cli_t:
            print(cli_t.get_help())
        else:
            print(schema_formatter.format_tool_help(tool_name))
        return 0

    # 3. Known tool check via explicit CLI registry
    cli_tool = CLI_TOOLS.get(tool_name)
    if not cli_tool:
        sys.stderr.write(f"Error: Unknown tool '{tool_name}'. Available tools: {list(CLI_TOOLS.keys())}\n")
        sys.stderr.write(f"Run '{launcher} --help' to view documentation.\n")
        return 1

    # 4. Explicit configuration check (Fail-Fast: required for all tool executions)
    if not config.is_configured():
        sys.stderr.write(
            f"Error: Tool '{tool_name}' requires a configuration file.\n"
            f"Specify --config <path_to_config.json> or set the MEDIA_CONFIG environment variable.\n"
            f"Run '{launcher} {tool_name} --help' to view parameter requirements.\n"
        )
        return 1

    request_arg = args[1] if len(args) > 1 else None

    # 5. Strict payload resolution: only '-' (stdin) or specification file on disk
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
        if cli_tool.requires_payload:
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

    # 6. Delegate execution via polymorphic CliStrategy
    return cli_tool.execute(payload, progress_cb=cli_progress)


if __name__ == "__main__":
    sys.exit(main())
