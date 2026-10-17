#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite 5: MCP Server JSON-RPC 2.0 Transport, CLI Runner, Clients & Hot Reload.
Covers:
- Strict configuration enforcement (config.is_configured())
- ComfyUI & A1111 Keep-Alive sessions (trust_env == False) and active queue tracking
- Prefix-isolated upload_file_to_comfy preventing filename collisions (D6)
- Topological reload_core_modules() preserving active config and _GPU_LOCK
- JSON-RPC 2.0 protocol (initialize, tools/list, tools/call, ContextVar progress token, error result envelope)
- CLI interface (--help, <tool> --help, inline JSON rejection)
"""

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, comfy_client, a1111_client, tools
from core.tools import execute_tool, CANONICAL_TOOLS, reload_core_modules
from mcp import media_mcp


class TestMcpServerAndCli(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp_dir.name)
        (cls.tmp_path / "ComfyUI" / "input").mkdir(parents=True, exist_ok=True)
        (cls.tmp_path / "ComfyUI" / "output").mkdir(parents=True, exist_ok=True)
        cls._test_config = {
            "output_dir": str(cls.tmp_path / "media_mcp"),
            "comfy_dir": str(cls.tmp_path / "ComfyUI"),
            "comfy_url": "http://127.0.0.1:8188",
            "sd_webui_url": "http://127.0.0.1:7860",
            "media_mcp_port": 7865
        }
        config.load_config_dict(cls._test_config)

    @classmethod
    def tearDownClass(cls):
        cls._tmp_dir.cleanup()
        config.cfg_json = {}

    def test_01_strict_config_enforcement(self):
        """Verify strict config check rejects operations when unconfigured."""
        saved_cfg = dict(config.cfg_json)
        try:
            config.cfg_json = {}
            self.assertFalse(config.is_configured())
            with self.assertRaises(RuntimeError):
                config.require_configured("Test operation")

            res = execute_tool("video_validate", {"scenario_path": "examples/scenario.json"})
            self.assertTrue(res.get("isError"))
            self.assertIn("requires an environment configuration file", res.get("error", ""))
        finally:
            config.load_config_dict(saved_cfg)

    def test_02_comfy_client_queue_tracking_and_input_prefix_isolation(self):
        """Verify Keep-Alive sessions, active queue tracking, and prefix-isolated upload_file_to_comfy."""
        self.assertFalse(comfy_client.get_session().trust_env)
        self.assertFalse(a1111_client.get_session().trust_env)

        # Prefix-isolated upload_file_to_comfy
        src_file = self.tmp_path / "anchor.png"
        src_file.write_bytes(b"\x89PNG\r\n\x1a\ntest")
        copied_name = comfy_client.upload_file_to_comfy(
            str(src_file), run_prefix="20261017-120000-abcdef", suffix="input"
        )
        self.assertEqual(copied_name, "20261017-120000-abcdef-input.png")
        self.assertTrue((Path(config.COMFY_INPUT_DIR) / copied_name).is_file())

        # Queue tracking: queue_pending -> queue_running -> history
        call_seq = {"step": 0}

        def mock_make_req(endpoint, data=None, method="GET", raw=False, timeout=30):
            if endpoint.startswith("/history"):
                if call_seq["step"] >= 2:
                    return {"test_pid": {"status": {"completed": True}, "outputs": {"9": {"images": ["ok.png"]}}}}
                return {}
            elif endpoint == "/queue":
                if call_seq["step"] == 0:
                    call_seq["step"] = 1
                    return {"queue_running": [], "queue_pending": [[1, "test_pid"]]}
                elif call_seq["step"] == 1:
                    call_seq["step"] = 2
                    return {"queue_running": [[1, "test_pid"]], "queue_pending": []}
                return {"queue_running": [], "queue_pending": []}
            return {}

        with patch("core.comfy_client.make_request", side_effect=mock_make_req):
            ok, hist = comfy_client.wait_for_prompt("test_pid", timeout=10, poll_interval=0.01)
            self.assertTrue(ok)
            self.assertIn("outputs", hist)

    def test_03_topological_reload_preserves_config_and_gpu_lock(self):
        """Verify reload_core_modules reloads in topological order and preserves _GPU_LOCK and config."""
        lock_before = tools._GPU_LOCK
        reloaded = reload_core_modules()
        self.assertGreater(len(reloaded), 5)
        self.assertEqual(reloaded[0], "core.config")
        self.assertIs(tools._GPU_LOCK, lock_before)
        self.assertTrue(config.is_configured())

    def test_04_mcp_jsonrpc_protocol_and_error_envelope(self):
        """Verify JSON-RPC 2.0 initialize, tools/list, tools/call, and standard MCP result error envelope."""
        init_res = media_mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_res["result"]["serverInfo"]["name"], "media-mcp")

        tools_res = media_mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = [t["name"] for t in tools_res["result"]["tools"]]
        for et in CANONICAL_TOOLS:
            self.assertIn(et, tool_names)

        # Valid tools/call
        call_val = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "video_validate",
                "arguments": {"scenario_path": "examples/scenario.json"},
                "_meta": {"progressToken": "tok-123"}
            }
        })
        self.assertIn("result", call_val)
        self.assertTrue(call_val["result"]["is_valid"])
        self.assertIsNone(media_mcp._CURRENT_PROGRESS_TOKEN.get())

        # Invalid tools/call: isError must be inside result, NOT at top-level JSON-RPC envelope
        call_err = media_mcp.handle_jsonrpc({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "image_generate",
                "arguments": {"engine": "sdxl"}  # Missing model and prompt
            }
        })
        self.assertIn("result", call_err)
        self.assertNotIn("isError", call_err)
        self.assertTrue(call_err["result"]["isError"])

    def test_05_gpu_serialization_lock_and_status(self):
        """Verify _GPU_LOCK reports IDLE/BUSY in media_get_status and response matches schema."""
        status_idle = execute_tool("media_get_status", {})
        self.assertFalse(status_idle.get("isError", False))
        self.assertIn("IDLE", status_idle["report"])
        self.assertNotIn("comfyui_available", status_idle)
        self.assertIn("comfy_online", status_idle)

        with tools._GPU_LOCK:
            status_busy = execute_tool("media_get_status", {})
            self.assertIn("BUSY", status_busy["report"])

    def test_06_cli_interface(self):
        """Verify CLI entry point behavior and strict input resolution."""
        res_help = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_help.returncode, 0)
        self.assertIn("AVAILABLE TOOLS:", res_help.stdout)

        res_tool_help = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "video_validate", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_tool_help.returncode, 0)
        self.assertIn("TOOL: video_validate", res_tool_help.stdout)

        res_raw_json = subprocess.run(
            [sys.executable, str(SRC_DIR / "cli.py"), "--config", "examples/config.json", "video_validate", '{"scenario_path": "x"}'],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_raw_json.returncode, 1)
        self.assertIn("Direct inline JSON strings are not accepted", res_raw_json.stderr)


if __name__ == "__main__":
    unittest.main()
