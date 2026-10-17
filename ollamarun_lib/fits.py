"""Fit jobs for the serve API: asynchronous context probes.

A fit job runs one `ctxfit.fit_context(...)` in a worker thread and exposes
its lifecycle over the native agent API: queued → running → completed /
failed / cancelled, with progress, an optional applied result and a
`CancelToken`. Jobs are in-memory records (like turn records); terminal
jobs are swept after `serve_fit_ttl`. A job that carries a session id holds
that session for its duration -- the API refuses turns and edits meanwhile,
on top of this service's own one-active-job-per-session rule -- and a global
cap bounds how many fits (and therefore model loads) can churn at once.

The service never prints and never touches HTTP: `agent_api` validates the
request, builds the `FitRequest`, renders `fit_body()` and performs the
optional session write in its `on_success` hook.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .ctxfit import FitCancelled, FitOutcome, FitRequest, fit_context
from .helpers import CancelToken, now_iso

logger = logging.getLogger("ollamarun.fits")

#: Statuses that end a job; a terminal job only waits for its TTL.
TERMINAL_FIT_STATUSES = frozenset({"completed", "failed", "cancelled"})


class FitsClosed(Exception):
    """The service is shutting down and refuses new jobs."""


class TooManyFits(Exception):
    """The global active-fit cap is reached."""

    def __init__(self, limit: int):
        super().__init__(f"too many active fits (limit {limit})")
        self.limit = limit


class FitBusy(Exception):
    """The session already has an active fit job."""

    def __init__(self, session_id: str, active_fit_id: str):
        super().__init__(f"session {session_id!r} is fitting ({active_fit_id})")
        self.session_id = session_id
        self.active_fit_id = active_fit_id


@dataclass
class FitJob:
    """One fit's HTTP-side state."""

    id: str
    model: str
    mode: str                       # "gpu" | "cpu"
    probe: bool
    apply: bool
    session_id: Optional[str]
    req: FitRequest
    cancel: CancelToken
    percent: Optional[int] = None
    status: str = "queued"
    created_at: str = ""
    updated_at: str = ""
    updated_epoch: float = 0.0
    progress: dict[str, Any] = field(default_factory=dict)
    outcome: Optional[FitOutcome] = None
    error: Optional[str] = None
    applied: bool = False
    apply_error: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)


def fit_body(job: FitJob) -> dict[str, Any]:
    """The `GET /v1/agent/fits/{id}` shape for one job."""
    body: dict[str, Any] = {
        "id": job.id, "object": "fit", "status": job.status,
        "session_id": job.session_id, "model": job.model, "mode": job.mode,
        "probe": job.probe, "apply": job.apply,
        "created_at": job.created_at, "updated_at": job.updated_at,
        "progress": dict(job.progress),
        "applied": job.applied,
    }
    if job.outcome is not None:
        body["result"] = job.outcome.to_dict()
    if job.error is not None:
        body["error"] = job.error
    if job.apply_error is not None:
        body["apply_error"] = job.apply_error
    return body


def _failure_text(outcome: FitOutcome) -> str:
    """Why a finished fit produced no context (plain text, no markup)."""
    if outcome.probe_result is not None and outcome.probe_result.reason:
        return outcome.probe_result.reason
    fit = outcome.fit
    if fit is not None and getattr(fit, "error", ""):
        return str(fit.error)
    return "the fit produced no context"


class FitService:
    """The fit job registry: workers, progress, cancellation, caps, TTL."""

    def __init__(self, *, max_active: int = 2, ttl: float = 86400.0,
                 clock: Callable[[], float] = time.time):
        self._lock = threading.Lock()
        self._jobs: dict[str, FitJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._max_active = max(1, int(max_active))
        self._ttl = float(ttl)
        self._clock = clock
        self._closed = False

    # ── registry ────────────────────────────────────────────────────────────

    def submit(self, *, model: str, mode: str, req: FitRequest,
               percent: Optional[int] = None, probe: bool = True,
               apply: bool = False, session_id: Optional[str] = None,
               cancel: Optional[CancelToken] = None,
               on_success: Optional[Callable[[FitJob], None]] = None) -> FitJob:
        """Start one fit; raises `FitsClosed`/`FitBusy`/`TooManyFits`."""
        with self._lock:
            if self._closed:
                raise FitsClosed("the fit service is shutting down")
            self._sweep_locked(self._clock())
            if session_id is not None:
                active = self._active_for_locked(session_id)
                if active is not None:
                    raise FitBusy(session_id, active)
            if len(self._active_locked()) >= self._max_active:
                raise TooManyFits(self._max_active)
            token = cancel if cancel is not None else CancelToken()
            req.cancel = token
            job = FitJob(id=f"fit_{uuid.uuid4().hex[:12]}", model=model,
                         mode=mode, probe=probe, apply=apply,
                         session_id=session_id, req=req, cancel=token,
                         percent=percent, created_at=now_iso(),
                         updated_at=now_iso())
            req.on_progress = lambda data, job=job: self._progress(job, data)
            self._jobs[job.id] = job
            thread = threading.Thread(target=self._run, args=(job, on_success),
                                      daemon=True, name=f"fit-{job.id}")
            # Start under the lock so `shutdown()` can never miss an accepted
            # job whose worker had not started yet (the turns.py pattern).
            self._threads[job.id] = thread
            thread.start()
            return job

    def get(self, fit_id: str) -> Optional[FitJob]:
        with self._lock:
            self._sweep_locked(self._clock())
            return self._jobs.get(fit_id)

    def cancel(self, fit_id: str) -> Optional[FitJob]:
        """Request cancellation; returns the job (None when unknown/expired)."""
        with self._lock:
            self._sweep_locked(self._clock())
            job = self._jobs.get(fit_id)
            if job is None:
                return None
            if job.status not in TERMINAL_FIT_STATUSES:
                job.cancel.cancel()
                job.updated_at = now_iso()
            return job

    def active_for(self, session_id: str) -> Optional[str]:
        """The active job id holding one session, if any."""
        with self._lock:
            self._sweep_locked(self._clock())
            return self._active_for_locked(session_id)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active_locked())

    def wait(self, fit_id: str, timeout: Optional[float] = None) -> Optional[FitJob]:
        """Join one job's worker (tests and shutdown); None when unknown."""
        job = self.get(fit_id)
        if job is None:
            return None
        job.done.wait(timeout)
        return job

    def shutdown(self, *, cancel: bool = True, timeout: float = 5.0) -> None:
        """Stop accepting, cancel active jobs and join them briefly."""
        with self._lock:
            self._closed = True
            active = [job for job in self._jobs.values()
                      if job.status not in TERMINAL_FIT_STATUSES]
            if cancel:
                for job in active:
                    job.cancel.cancel()
            threads = list(self._threads.values())
        deadline = self._clock() + max(0.0, timeout)
        for thread in threads:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    # ── internals ───────────────────────────────────────────────────────────

    def _sweep_locked(self, now: float) -> int:
        """Drop terminal jobs past their TTL (under the lock)."""
        removed = 0
        for fit_id, job in list(self._jobs.items()):
            if job.status not in TERMINAL_FIT_STATUSES:
                continue
            if now - float(job.updated_epoch) > self._ttl:
                del self._jobs[fit_id]
                self._threads.pop(fit_id, None)
                removed += 1
        return removed

    def _active_locked(self) -> list[str]:
        return [job.id for job in self._jobs.values()
                if job.status not in TERMINAL_FIT_STATUSES]

    def _active_for_locked(self, session_id: str) -> Optional[str]:
        for job in self._jobs.values():
            if job.session_id == session_id \
                    and job.status not in TERMINAL_FIT_STATUSES:
                return job.id
        return None

    def _progress(self, job: FitJob, data: dict[str, Any]) -> None:
        with self._lock:
            if job.status in TERMINAL_FIT_STATUSES:
                return
            job.progress = dict(data)
            job.updated_at = now_iso()

    def _run(self, job: FitJob,
             on_success: Optional[Callable[[FitJob], None]]) -> None:
        try:
            with self._lock:
                job.status = "running"
                job.updated_at = now_iso()
            outcome = fit_context(job.req, mode=job.mode,
                                  percent=job.percent, probe=job.probe)
            with self._lock:
                job.outcome = outcome
                if outcome.num_ctx is None:
                    job.status = "failed"
                    job.error = outcome.error or _failure_text(outcome)
                else:
                    job.status = "completed"
                job.updated_at = now_iso()
                job.updated_epoch = self._clock()
            if job.status == "completed" and job.apply and on_success is not None:
                try:
                    on_success(job)
                    with self._lock:
                        job.applied = True
                except Exception as exc:  # completed, simply not applied
                    logger.debug("fit %s apply failed", job.id, exc_info=True)
                    with self._lock:
                        job.apply_error = str(exc)
        except FitCancelled:
            with self._lock:
                job.status = "cancelled"
                job.error = "cancelled"
                job.updated_at = now_iso()
                job.updated_epoch = self._clock()
        except Exception:  # a broken fit never kills the server
            logger.debug("fit %s crashed", job.id, exc_info=True)
            with self._lock:
                job.status = "failed"
                job.error = "the fit failed internally"
                job.updated_at = now_iso()
                job.updated_epoch = self._clock()
        finally:
            with self._lock:
                job.updated_epoch = self._clock()
            job.done.set()
