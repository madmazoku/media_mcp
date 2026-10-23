#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thread-safe in-memory Job Registry and Execution Manager for media_mcp.
Maintains bounded circular queue (maxlen=1000), threading.Event per-job synchronization,
and deterministic lifecycle transitions (queued -> running -> completed/failed/cancelled).
"""

from __future__ import annotations
import sys
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


def log(msg: str):
    sys.stderr.write(f"[job_manager] {msg}\n")
    sys.stderr.flush()


@dataclass
class JobRecord:
    run_prefix: str
    spec_path: str
    scenario_dict: Dict[str, Any]
    status: str = "queued"  # queued, running, completed, failed, cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    elapsed_seconds: float = 0.0
    done_event: threading.Event = field(default_factory=threading.Event)
    progress: Dict[str, Any] = field(default_factory=lambda: {
        "phase": "queued",
        "percent": 0,
        "scene_index": 1,
        "total_scenes": 1,
        "block_index": 0,
        "total_blocks": 0,
        "message": "Job queued for GPU execution"
    })
    video: Optional[Dict[str, Any]] = None
    image: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_response_dict(self) -> Dict[str, Any]:
        """Converts internal record to strictly typed media_get_job response contract."""
        cur_elapsed = self.elapsed_seconds
        if self.status == "running" and self.started_at:
            cur_elapsed = round(time.time() - self.started_at, 2)

        resp: Dict[str, Any] = {
            "run_prefix": self.run_prefix,
            "status": self.status,
            "spec_path": self.spec_path,
            "timestamps": {
                "created_at": round(self.created_at, 3),
                "started_at": round(self.started_at, 3) if self.started_at else None,
                "completed_at": round(self.completed_at, 3) if self.completed_at else None,
                "elapsed_seconds": cur_elapsed,
            },
            "progress": {
                "phase": self.progress.get("phase", "idle"),
                "percent": int(self.progress.get("percent", 0)),
                "scene_index": int(self.progress.get("scene_index", 1)),
                "total_scenes": int(self.progress.get("total_scenes", 1)),
                "block_index": int(self.progress.get("block_index", 0)),
                "total_blocks": int(self.progress.get("total_blocks", 0)),
                "message": str(self.progress.get("message", "")),
            },
            "video": self.video if self.status == "completed" else None,
            "image": self.image if self.status == "completed" else None,
        }
        if self.error:
            resp["error"] = self.error
        if self.warnings:
            resp["warnings"] = list(self.warnings)
        return resp


class JobManager:
    """Thread-safe bounded circular collection of video generation jobs."""

    def __init__(self, maxlen: int = 1000):
        self._maxlen = maxlen
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._lock = threading.RLock()
        self._active_run_prefix: Optional[str] = None

    def create_job(
        self,
        run_prefix: str,
        spec_path: str,
        scenario_dict: Dict[str, Any],
        warnings: Optional[List[str]] = None,
        total_scenes: int = 1,
        total_blocks: int = 0
    ) -> JobRecord:
        with self._lock:
            # Enforce capacity by evicting terminal jobs by priority: cancelled -> failed -> completed
            if len(self._jobs) >= self._maxlen:
                candidate_key = None
                for candidate_status in ("cancelled", "failed", "completed"):
                    for k, v in self._jobs.items():
                        if v.status == candidate_status:
                            candidate_key = k
                            break
                    if candidate_key is not None:
                        break

                if candidate_key is not None:
                    status_evicted = self._jobs[candidate_key].status
                    del self._jobs[candidate_key]
                    log(f"Evicted finished job record: {candidate_key} (status: {status_evicted})")
                else:
                    raise RuntimeError(
                        f"Job registry capacity ({self._maxlen}) reached with active (queued/running) jobs. "
                        "Cannot submit new job until existing jobs finish or are cancelled."
                    )

            job = JobRecord(
                run_prefix=run_prefix,
                spec_path=spec_path,
                scenario_dict=scenario_dict,
                warnings=list(warnings or []),
                progress={
                    "phase": "queued",
                    "percent": 0,
                    "scene_index": 1,
                    "total_scenes": total_scenes,
                    "block_index": 0,
                    "total_blocks": total_blocks,
                    "message": "Job queued for GPU execution"
                }
            )
            self._jobs[run_prefix] = job
            log(f"Created queued job '{run_prefix}' (total registered: {len(self._jobs)})")
            return job

    def get_job(self, run_prefix: str, wait: int = 0) -> Optional[JobRecord]:
        with self._lock:
            job = self._jobs.get(run_prefix)
            if not job:
                return None
            done_event = job.done_event
            status = job.status

        if wait > 0 and status in ("queued", "running"):
            done_event.wait(timeout=float(wait))

        with self._lock:
            return self._jobs.get(run_prefix)

    def get_job_snapshot(self, run_prefix: str, wait_seconds: int = 0) -> Optional[Dict[str, Any]]:
        """Atomically returns snapshot dictionary conforming to media_get_job response schema."""
        with self._lock:
            job = self._jobs.get(run_prefix)
            if not job:
                return None
            done_event = job.done_event
            status = job.status

        if wait_seconds > 0 and status in ("queued", "running"):
            done_event.wait(timeout=float(wait_seconds))

        with self._lock:
            job = self._jobs.get(run_prefix)
            if not job:
                return None
            return job.to_response_dict()

    def mark_started(self, run_prefix: str) -> bool:
        with self._lock:
            job = self._jobs.get(run_prefix)
            if job and job.status == "queued":
                job.status = "running"
                job.started_at = time.time()
                job.progress["phase"] = "initializing"
                job.progress["message"] = "GPU lock acquired, starting scenario pipeline"
                self._active_run_prefix = run_prefix
                log(f"Job '{run_prefix}' transitioned to RUNNING on GPU")
                return True
            return False

    def update_progress(
        self,
        run_prefix: str,
        phase: str,
        percent: Optional[int] = None,
        scene_index: int = 1,
        total_scenes: int = 1,
        block_index: int = 0,
        total_blocks: int = 0,
        message: Optional[str] = None
    ):
        with self._lock:
            job = self._jobs.get(run_prefix)
            if job and job.status == "running":
                job.progress["phase"] = phase
                if percent is not None:
                    job.progress["percent"] = max(0, min(100, int(percent)))
                job.progress["scene_index"] = scene_index
                job.progress["total_scenes"] = total_scenes
                job.progress["block_index"] = block_index
                job.progress["total_blocks"] = total_blocks
                if message:
                    job.progress["message"] = message

    def complete_job(
        self,
        run_prefix: str,
        video_data: Dict[str, Any],
        image_data: Dict[str, Any],
        warnings: Optional[List[str]] = None
    ):
        with self._lock:
            job = self._jobs.get(run_prefix)
            if not job or job.status in ("completed", "failed", "cancelled"):
                return

            now = time.time()
            job.status = "completed"
            job.completed_at = now
            if job.started_at:
                job.elapsed_seconds = round(now - job.started_at, 2)

            job.video = video_data
            job.image = image_data
            if warnings:
                for w in warnings:
                    if w not in job.warnings:
                        job.warnings.append(w)

            job.progress["percent"] = 100
            job.progress["phase"] = "completed"
            job.progress["message"] = "Scenario video assembly completed successfully"

            if self._active_run_prefix == run_prefix:
                self._active_run_prefix = None

            job.done_event.set()
            log(f"Job '{run_prefix}' COMPLETED in {job.elapsed_seconds}s")

    def fail_job(self, run_prefix: str, error_message: str):
        with self._lock:
            job = self._jobs.get(run_prefix)
            if not job or job.status in ("completed", "failed", "cancelled"):
                return

            now = time.time()
            job.status = "failed"
            job.completed_at = now
            if job.started_at:
                job.elapsed_seconds = round(now - job.started_at, 2)
            job.error = str(error_message)
            job.progress["phase"] = "failed"
            job.progress["message"] = f"Failed: {error_message}"

            if self._active_run_prefix == run_prefix:
                self._active_run_prefix = None

            job.done_event.set()
            log(f"Job '{run_prefix}' FAILED: {error_message}")

    def cancel_or_remove_job(self, run_prefix: str) -> Tuple[str, str]:
        need_interrupt = False
        with self._lock:
            if run_prefix not in self._jobs:
                return "not_found", f"Job '{run_prefix}' was not found in active jobs registry."

            job = self._jobs[run_prefix]

            if job.status == "queued":
                job.status = "cancelled"
                now = time.time()
                job.completed_at = now
                if job.started_at:
                    job.elapsed_seconds = round(now - job.started_at, 2)
                job.error = "Job cancelled while waiting in queue."
                job.progress["phase"] = "cancelled"
                job.progress["message"] = "Job cancelled before GPU execution"
                job.done_event.set()
                log(f"Cancelled queued job '{run_prefix}'")
                return "cancelled", f"Queued job '{run_prefix}' was cancelled."

            elif job.status == "running":
                need_interrupt = True

            else:
                # Terminal state (completed, failed, cancelled): purge from registry
                status_was = job.status
                del self._jobs[run_prefix]
                log(f"Purged finished job '{run_prefix}' (status was: {status_was})")
                return "removed", f"Finished job '{run_prefix}' (status: {status_was}) was removed from registry."

        if need_interrupt:
            # Dispatch interrupt to backend outside the lock
            try:
                from . import comfy_client, video_pipeline
                comfy_client.interrupt()
                video_pipeline.request_interrupt()
            except Exception as ex:
                log(f"Warning: Failed to dispatch interrupt to backend: {ex}")

            with self._lock:
                job = self._jobs.get(run_prefix)
                if job and job.status not in ("completed", "failed"):
                    job.status = "cancelled"
                    now = time.time()
                    job.completed_at = now
                    if job.started_at:
                        job.elapsed_seconds = round(now - job.started_at, 2)
                    job.error = "Job execution cancelled by user request."
                    job.progress["phase"] = "cancelled"
                    job.progress["message"] = "Job execution cancelled"
                    if self._active_run_prefix == run_prefix:
                        self._active_run_prefix = None
                    job.done_event.set()
                    log(f"Cancelled active GPU job '{run_prefix}'")
            return "cancelled", f"Active GPU job '{run_prefix}' was interrupted and marked cancelled."

    def cancel_active_job(self, reason: str = "Job execution cancelled by cluster interrupt request.") -> Optional[str]:
        with self._lock:
            if not self._active_run_prefix:
                return None
            run_prefix = self._active_run_prefix
            job = self._jobs.get(run_prefix)
            if job and job.status not in ("completed", "failed", "cancelled"):
                job.status = "cancelled"
                now = time.time()
                job.completed_at = now
                if job.started_at:
                    job.elapsed_seconds = round(now - job.started_at, 2)
                job.error = reason
                job.progress["phase"] = "cancelled"
                job.progress["message"] = reason
                self._active_run_prefix = None
                job.done_event.set()
                log(f"Marked active job '{run_prefix}' as CANCELLED via cluster interrupt")
                return run_prefix
            return None

    def get_queue_info(self) -> Dict[str, Any]:
        with self._lock:
            queued_count = sum(1 for j in self._jobs.values() if j.status == "queued")
            running_count = sum(1 for j in self._jobs.values() if j.status == "running")
            return {
                "active_job": self._active_run_prefix,
                "queued_jobs": queued_count,
                "running_jobs": running_count,
                "total_records": len(self._jobs)
            }


JOB_MANAGER = JobManager(maxlen=1000)
