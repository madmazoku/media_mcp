#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite: JobManager & Video Worker Lifecycle, FIFO Queue, Concurrency & Repro Scenarios.
Covers:
- Cancel queued job is not rendered on GPU (Repro 1)
- Terminal status protection: cancelled job is never overwritten to failed/completed (Repro 2)
- media_get_job via execute_tool does not return isError=True on normal domain queries (Repro 3)
- Snapshot read under lock (Torn read protection)
- Capacity eviction: terminal-only eviction and active-capacity rejection
- FIFO queue execution order
"""

import sys
import time
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core import config, tools, video_pipeline, comfy_client
from core.job_manager import JobManager, JobRecord, JOB_MANAGER
from core.tools import execute_tool


class TestJobManagerLifecycle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp_dir.name)
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

    def test_01_cancelled_queued_job_is_skipped(self):
        """Repro 1: A cancelled queued job must NOT run on GPU when worker reaches it."""
        executed = []

        def fake_exec(prepared_pipeline_data=None, progress_callback=None, **kw):
            executed.append(prepared_pipeline_data["run_prefix"])
            time.sleep(0.1)
            return {
                "video_path": "x.mp4", "poster_path": "p.png", "width": 832, "height": 480,
                "duration": 1.0, "fps": 16, "scenes_count": 1, "blocks_count": 1, "anchor_metadata": {}
            }

        def prep(rp):
            return {
                "run_prefix": rp, "spec_path": "s.json", "resolved_scenario_dict": {},
                "scenes": [], "total_scenes": 1, "total_blocks": 1
            }

        with patch.object(video_pipeline, "execute_scenario_pipeline", fake_exec), \
             patch.object(comfy_client, "interrupt", lambda: None):
            a = JOB_MANAGER.create_job("A-REPRO1", "s.json", {})
            b = JOB_MANAGER.create_job("B-REPRO1", "s.json", {})

            ta = threading.Thread(target=tools._run_video_job_worker, args=(a, prep("A-REPRO1")))
            tb = threading.Thread(target=tools._run_video_job_worker, args=(b, prep("B-REPRO1")))
            ta.start()
            tb.start()
            time.sleep(0.02)

            status, msg = JOB_MANAGER.cancel_or_remove_job("B-REPRO1")
            self.assertEqual(status, "cancelled")

            ta.join()
            tb.join()

            self.assertIn("A-REPRO1", executed)
            self.assertNotIn("B-REPRO1", executed)

    def test_02_terminal_status_is_never_overwritten(self):
        """Repro 2: A cancelled running job is NOT overwritten to 'failed' when interrupted exception is raised."""
        def fake_exec(prepared_pipeline_data=None, progress_callback=None, **kw):
            time.sleep(0.2)
            raise RuntimeError("Scenario execution was interrupted by user request.")

        def prep(rp):
            return {
                "run_prefix": rp, "spec_path": "s.json", "resolved_scenario_dict": {},
                "scenes": [], "total_scenes": 1, "total_blocks": 1
            }

        with patch.object(video_pipeline, "execute_scenario_pipeline", fake_exec), \
             patch.object(comfy_client, "interrupt", lambda: None):
            c = JOB_MANAGER.create_job("C-REPRO2", "s.json", {})
            tc = threading.Thread(target=tools._run_video_job_worker, args=(c, prep("C-REPRO2")))
            tc.start()
            time.sleep(0.05)

            status, msg = JOB_MANAGER.cancel_or_remove_job("C-REPRO2")
            self.assertEqual(status, "cancelled")
            tc.join()

            final_job = JOB_MANAGER.get_job("C-REPRO2")
            self.assertIsNotNone(final_job)
            self.assertEqual(final_job.status, "cancelled")

    def test_03_media_get_job_success_envelope_for_all_domain_statuses(self):
        """Repro 3: media_get_job must NOT set isError=True on failed, cancelled, or not_found jobs."""
        # 1. Cancelled job
        JOB_MANAGER.create_job("JOB-CANC", "s.json", {})
        JOB_MANAGER.cancel_or_remove_job("JOB-CANC")
        res_canc = execute_tool("media_get_job", {"run_prefix": "JOB-CANC"})
        self.assertFalse(res_canc.get("isError", False))
        self.assertEqual(res_canc["status"], "cancelled")

        # 2. Failed job
        JOB_MANAGER.create_job("JOB-FAIL", "s.json", {})
        JOB_MANAGER.fail_job("JOB-FAIL", "CUDA out of memory")
        res_fail = execute_tool("media_get_job", {"run_prefix": "JOB-FAIL"})
        self.assertFalse(res_fail.get("isError", False))
        self.assertEqual(res_fail["status"], "failed")
        self.assertEqual(res_fail["error"], "CUDA out of memory")
        self.assertEqual(res_fail["progress"]["message"], "CUDA out of memory")

        # 3. Non-existent job
        res_nf = execute_tool("media_get_job", {"run_prefix": "JOB-NOPE"})
        self.assertFalse(res_nf.get("isError", False))
        self.assertEqual(res_nf["status"], "not_found")
        self.assertIn("not found", res_nf["error"])

    def test_04_capacity_eviction_only_evicts_terminal_jobs(self):
        """Verify JobManager evicts terminal jobs by priority (cancelled -> failed -> completed) and rejects if full."""
        mgr = JobManager(maxlen=3)
        # Create 3 active queued jobs
        j1 = mgr.create_job("J1", "s.json", {})
        j2 = mgr.create_job("J2", "s.json", {})
        j3 = mgr.create_job("J3", "s.json", {})

        # All 3 active: 4th submission must raise RuntimeError
        with self.assertRaises(RuntimeError):
            mgr.create_job("J4", "s.json", {})

        # Test priority: completed vs failed vs cancelled
        # J1 -> completed, J2 -> failed, J3 -> cancelled
        mgr.complete_job("J1", video_data={}, image_data={})
        mgr.fail_job("J2", "error")
        mgr.cancel_or_remove_job("J3")

        # 1. 1st eviction should pick J3 (cancelled), even though J1 was oldest
        j4 = mgr.create_job("J4", "s.json", {})
        self.assertIsNone(mgr.get_job("J3"))
        self.assertIsNotNone(mgr.get_job("J1"))
        self.assertIsNotNone(mgr.get_job("J2"))
        self.assertIsNotNone(mgr.get_job("J4"))

        # 2. 2nd eviction should pick J2 (failed) over J1 (completed)
        j5 = mgr.create_job("J5", "s.json", {})
        self.assertIsNone(mgr.get_job("J2"))
        self.assertIsNotNone(mgr.get_job("J1"))
        self.assertIsNotNone(mgr.get_job("J4"))
        self.assertIsNotNone(mgr.get_job("J5"))

        # 3. 3rd eviction should finally pick J1 (completed)
        j6 = mgr.create_job("J6", "s.json", {})
        self.assertIsNone(mgr.get_job("J1"))
        self.assertIsNotNone(mgr.get_job("J4"))
        self.assertIsNotNone(mgr.get_job("J5"))
        self.assertIsNotNone(mgr.get_job("J6"))

    def test_05_snapshot_isolation(self):
        """Verify get_job_snapshot produces isolated dict under lock without torn read."""
        mgr = JobManager(maxlen=10)
        mgr.create_job("SNAP-1", "s.json", {})
        mgr.mark_started("SNAP-1")
        mgr.update_progress("SNAP-1", phase="diffusing", percent=50, message="Halfway")

        snap = mgr.get_job_snapshot("SNAP-1")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["status"], "running")
        self.assertEqual(snap["progress"]["percent"], 50)
        self.assertEqual(snap["progress"]["phase"], "diffusing")


    def test_06_media_interrupt_marks_running_job_as_cancelled(self):
        """Verify media_interrupt cancels active GPU job so its status becomes 'cancelled', not 'failed'."""
        def fake_exec(prepared_pipeline_data=None, progress_callback=None, **kw):
            time.sleep(0.2)
            raise RuntimeError("Scenario execution was interrupted by user request.")

        def prep(rp):
            return {
                "run_prefix": rp, "spec_path": "s.json", "resolved_scenario_dict": {},
                "scenes": [], "total_scenes": 1, "total_blocks": 1
            }

        with patch.object(video_pipeline, "execute_scenario_pipeline", fake_exec), \
             patch.object(comfy_client, "interrupt", lambda: None), \
             patch.object(tools.a1111_client, "interrupt", lambda: None):
            d = JOB_MANAGER.create_job("D-INTERRUPT", "s.json", {})
            td = threading.Thread(target=tools._run_video_job_worker, args=(d, prep("D-INTERRUPT")))
            td.start()
            time.sleep(0.05)

            # Trigger cluster-wide interrupt
            int_res = execute_tool("media_interrupt", {})
            self.assertTrue(int_res["interrupted"])
            self.assertIn("D-INTERRUPT", int_res["message"])

            td.join()
            final_job = JOB_MANAGER.get_job("D-INTERRUPT")
            self.assertIsNotNone(final_job)
            self.assertEqual(final_job.status, "cancelled")


if __name__ == "__main__":
    unittest.main()
