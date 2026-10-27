"""Which stage the "waiting for model" indicator is actually in.

A request that has not produced a token yet can be doing several very
different things: reading weights from disk, waiting because another model
(or another client's load) holds the machine, being evicted to make room,
reloading at a different context, or prefilling a model that is already
resident. They used to render as the same four words.

The evidence, strongest first:

1. **`/api/ps`** -- the residency truth. An entry exists only once the runner
   is loaded, and carries the window it was loaded with (`context_length`),
   so "resident at 4,096 while the session asks for 8,192" is exact.
2. **The runner process** -- `ollama` starts one `llama-server` per model.
   Matching it to ours by the blob paths `/api/show` reports in the generated
   Modelfile (`FROM …/blobs/sha256-…`) says whether a load has *started*, and
   its per-process VRAM (`nvidia-smi --query-compute-apps`) plus RSS give the
   progress, including the DRAM half of a model that does not fit in VRAM.
3. **Machine direction** -- free VRAM/RAM deltas and CPU busy are tiebreakers
   only: a GPU allocation happens in one burst, so sampling system-wide
   deltas misses it almost every time.

`classify_stage()` is pure and unit-testable; `StageTracker` does the sampling
on one daemon thread, only while a wait is active. Every probe is best-effort:
a machine without nvidia-smi, a broken ssh or an unreachable `/api/ps` only
narrows what the line can say.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .helpers import (
    HardwareSample,
    RunnerProcess,
    canonical_model_name,
    cpu_busy_fraction,
    local_hardware_sample,
    local_runner_processes,
)

_UNSET = object()

STAGE_READY = "ready"
STAGE_PREFILL = "prefill"
STAGE_RELOAD = "reload"
STAGE_RELOADING = "reloading"
STAGE_LOADING = "loading"
STAGE_EVICTING = "evicting"
STAGE_WAITING = "waiting"
STAGE_UNREACHABLE = "unreachable"

#: A runner younger than this was spawned for a wait that is happening now.
RUNNER_YOUNG_S = 30.0
#: Below this age (and with no footprint yet) the runner is still starting.
RUNNER_SPAWN_S = 1.5
#: Free-VRAM move that counts as "something is happening".
VRAM_STEP_MIB = 64
#: CPU busy share that reads as "weights are being read".
CPU_BUSY_MIN = 0.30


def _gib(count: Optional[float]) -> str:
    if not count:
        return "0"
    return f"{count / 2 ** 30:.1f} GiB"


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_show_blobs(modelfile: str) -> list[str]:
    """Blob names (`sha256-…`) from `/api/show`'s generated Modelfile.

    The shower emits one `FROM /…/blobs/sha256-…` per model file (a vision
    model has two); a `FROM some-name` line identifies nothing, so it is
    skipped. These names also appear on the runner's command line, which is
    what lets a `llama-server` process be matched to the session's model.
    """
    names: list[str] = []
    for line in (modelfile or "").splitlines():
        line = line.strip()
        if not line.startswith("FROM "):
            continue
        path = line[len("FROM "):].strip().strip('"\'')
        if "/blobs/" not in path:
            continue
        name = path.rsplit("/", 1)[-1]
        if name and name not in names:
            names.append(name)
    return names


@dataclass
class StageFacts:
    """Everything the classifier is allowed to know, gathered once per sample."""

    model: str = ""
    resident: bool = False
    resident_ctx: Optional[int] = None
    requested_ctx: Optional[int] = None
    inflight: bool = False              # a chat request is open, no token yet
    warming: bool = False               # the background preload is running
    reload_pending: bool = False        # loaded options differ from the next request
    pending_from: Optional[int] = None
    pending_to: Optional[int] = None
    others: tuple[str, ...] = ()        # other residents in /api/ps
    runner: Optional[RunnerProcess] = None   # our model's runner, if any
    other_runner: bool = False          # a different model's runner just spawned
    evicted: tuple[str, ...] = ()       # residents that vanished during this wait
    activity: bool = False              # hardware is moving, no runner identified
    gpu_used_mib: Optional[int] = None  # our runner's VRAM
    rss: Optional[int] = None           # our runner's RSS (bytes)
    model_size: Optional[int] = None    # model file size (bytes)
    context_used: Optional[int] = None  # tokens the request carries
    ps_failed: bool = False


@dataclass
class Stage:
    """The rendered state: a stable key, a detail line and when it started."""

    key: str = STAGE_READY
    detail: str = ""
    since: float = 0.0


def _ctx_transition(facts: StageFacts) -> str:
    old, new = facts.pending_from, facts.pending_to
    if old and new and old != new:
        return f" · {old:,} → {new:,}"
    return ""


def _loading_detail(facts: StageFacts) -> str:
    parts: list[str] = []
    if facts.gpu_used_mib:
        parts.append(f"GPU {_gib(facts.gpu_used_mib * 2 ** 20)}")
    if facts.rss:
        parts.append(f"RAM {_gib(facts.rss)}")
    detail = f"loading {facts.model}"
    if parts:
        detail += " · " + " + ".join(parts)
        if facts.model_size:
            loaded = ((facts.gpu_used_mib or 0) * 2 ** 20) + (facts.rss or 0)
            # RSS counts touched pages, not the allocation, so this is a floor;
            # the tilde keeps it honest.
            if 0 < loaded <= facts.model_size * 1.5:
                detail += f" of ~{_gib(facts.model_size)}"
    elif facts.runner is not None and facts.runner.age_s < RUNNER_SPAWN_S:
        detail = f"starting {facts.model} · runner spawned"
    return detail


def classify_stage(facts: StageFacts) -> tuple[str, str]:
    """(key, detail) for one set of facts -- no I/O, no clocks."""
    model = facts.model or "model"
    if facts.ps_failed:
        return STAGE_UNREACHABLE, f"waiting for {model} · host did not answer"
    if facts.resident:
        if facts.inflight:
            if facts.reload_pending:
                return STAGE_RELOADING, f"reloading {model}{_ctx_transition(facts)}"
            detail = "prefill"
            if facts.context_used:
                detail += f" · ~{facts.context_used:,} tok"
            return STAGE_PREFILL, detail
        if facts.reload_pending:
            old, new = facts.resident_ctx, facts.requested_ctx
            if old and new and old != new:
                return STAGE_RELOAD, (f"resident at {old:,} · "
                                      f"reloads at {new:,} on send")
            return STAGE_RELOAD, "resident with older settings · reloads on send"
        return STAGE_READY, ""
    # Not resident (yet): loading, or waiting for something else to make room.
    if facts.inflight and facts.reload_pending:
        return STAGE_RELOADING, f"reloading {model}{_ctx_transition(facts)}"
    if facts.runner is not None:
        return STAGE_LOADING, _loading_detail(facts)
    if facts.activity:
        return STAGE_LOADING, f"loading {model} · reading weights"
    if facts.evicted:
        return STAGE_EVICTING, f"making room · unloaded {', '.join(facts.evicted)}"
    if facts.other_runner:
        return STAGE_WAITING, "waiting · another model is loading"
    if facts.others:
        names = ", ".join(facts.others[:2])
        return STAGE_WAITING, f"waiting for room · {names} resident"
    if facts.warming:
        # A preload is not guesswork: this REPL called `client.load()` itself,
        # so the load is happening whether or not the hardware probes caught
        # it. The elapsed suffix from `render_stage_line()` is the progress
        # signal when a model's footprint lands in one burst.
        return STAGE_LOADING, _loading_detail(facts)
    return STAGE_WAITING, f"waiting for {model} · starting runner"


def render_stage_line(stage: Stage, elapsed: float, *, limit: int = 78
                      ) -> Optional[str]:
    """The one-line text for the status surfaces, or None when there is none."""
    if not stage.detail:
        return None
    text = stage.detail
    if elapsed >= 1.0:
        text += f" · {elapsed:.0f}s"
    if len(text) > limit:
        text = text[:max(1, limit - 1)] + "…"
    return text


def load_pending_line(pending: dict[str, Any]) -> Optional[str]:
    """The idle hint for a resident runner whose config no longer matches."""
    if not pending:
        return None
    kind = pending.get("kind")
    old, new = pending.get("from"), pending.get("to")
    if kind == "ctx" and old and new:
        return f"resident at {old:,} · reloads at {new:,} on send"
    if kind == "ctx" and new:
        return f"reloads at {new:,} on send"
    return "resident with older settings · reloads on send"


def _local_probe(with_apps: bool) -> tuple[HardwareSample, list[RunnerProcess]]:
    return (local_hardware_sample(with_apps=with_apps),
            local_runner_processes())


class StageTracker:
    """Samples the machine while a wait is active and publishes the stage.

    Every dependency is injectable and `sample_once()` is public, so tests can
    drive the whole state machine without threads, clocks or hardware.
    """

    def __init__(self, model: str, *,
                 ps: Optional[Callable[[], list[dict[str, Any]]]] = None,
                 probe: Optional[Callable[[bool],
                                         Optional[tuple[HardwareSample,
                                                         list[RunnerProcess]]]]] = None,
                 blobs: Optional[Callable[[], list[str]]] = None,
                 pending: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
                 requested_ctx: Optional[Callable[[], Optional[int]]] = None,
                 context_used: Optional[Callable[[], Optional[int]]] = None,
                 model_size: Optional[Callable[[], Optional[int]]] = None,
                 interval: float = 1.0,
                 on_change: Optional[Callable[[], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.model = model
        self.interval = max(0.1, float(interval or 1.0))
        self._ps = ps or (lambda: [])
        self._probe = probe or _local_probe
        self._blobs = blobs or (lambda: [])
        self._pending = pending or (lambda: None)
        self._requested_ctx = requested_ctx or (lambda: None)
        self._context_used = context_used or (lambda: None)
        self._model_size = model_size or (lambda: None)
        self.on_change = on_change
        self.clock = clock
        self._lock = threading.RLock()
        self._stage = Stage(since=clock())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = clock()
        self._inflight = False
        self._warming = False
        self._prev: Optional[HardwareSample] = None
        self._others_before: dict[str, str] = {}
        self._evicted: list[str] = []
        self._resident_last: Optional[bool] = None
        self._last_line: Optional[str] = None
        self._size_cache: Any = _UNSET

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self, *, inflight: bool = False, warming: bool = False) -> None:
        """Begin sampling (idempotent); marks a chat request or a preload."""
        seeded = False
        with self._lock:
            self._inflight = self._inflight or bool(inflight)
            self._warming = self._warming or bool(warming)
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stage.key == STAGE_READY and not self._stage.detail:
                self._stage.since = self.clock()
            if self._warming and not self._stage.detail:
                # The first sample is a second away (nvidia-smi, `/api/show`,
                # `/api/ps`) and the prompt is drawn before it answers. Since
                # `_preload_model()` initiated this load, the state is known
                # without probing, so publish it before the thread starts.
                self._stage = Stage(STAGE_LOADING, f"loading {self.model}",
                                    self.clock())
                self._last_line = None
                seeded = True
            self._stop_event.clear()
            thread = threading.Thread(target=self._run, daemon=True,
                                      name="ollamarun-stage")
            self._thread = thread
            thread.start()
        if seeded:
            self._notify()

    def stop(self, *, join: bool = True) -> None:
        """Stop sampling (idempotent). `join=False` never waits.

        A probe in flight can take seconds (an ssh round trip), so the caller
        that runs on the streaming thread -- the first token -- must not
        wait; `_run` discards the reading when it sees the stop flag.
        """
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if join and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.3)
        with self._lock:
            was_visible = bool(self._stage.detail)
            self._stage = Stage(since=self.clock())
            self._last_line = None
        if was_visible:
            self._notify()

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def set_inflight(self, inflight: bool) -> None:
        with self._lock:
            self._inflight = bool(inflight)
            self._warming = False

    # ── public reads ────────────────────────────────────────────────────────

    def stage(self) -> Stage:
        with self._lock:
            return self._stage

    def text(self) -> Optional[str]:
        """The current line (no spinner glyph), or None when nothing waits."""
        stage = self.stage()
        return render_stage_line(stage, self.clock() - stage.since)

    # ── sampling ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception:  # a probe must never kill the indicator
                pass
            if self._stop_event.is_set():
                # The wait ended while the probe was in flight: its reading
                # describes a moment that is already over, so it must not
                # leave a stage behind for the prompt to keep showing.
                with self._lock:
                    self._stage = Stage(since=self.clock())
                    self._last_line = None
                self._notify()
                break
            if self._stop_event.wait(self.interval):
                break

    def _model_size_value(self) -> Optional[int]:
        if self._size_cache is _UNSET:
            try:
                self._size_cache = self._model_size()
            except Exception:
                self._size_cache = None
        return self._size_cache

    def sample_once(self) -> Stage:
        """Take one reading, classify it and publish the new stage."""
        now = self.clock()
        with self._lock:
            inflight = self._inflight
            warming = self._warming
        resident_last = self._resident_last
        with_apps = resident_last is not True

        hardware, processes = HardwareSample(), []
        try:
            got = self._probe(with_apps)
        except Exception:
            got = None
        previous = self._prev
        if got is not None:
            hardware, processes = got
            self._prev = hardware
        cpu_busy = cpu_busy_fraction(previous.cpu, hardware.cpu) \
            if previous is not None else None
        gpu_delta = _gpu_free_delta(previous, hardware) \
            if previous is not None else None
        ram_delta = _ram_delta(previous, hardware) \
            if previous is not None else None

        entries: list[dict[str, Any]] = []
        ps_failed = False
        try:
            entries = list(self._ps() or [])
        except Exception:
            ps_failed = True

        target = canonical_model_name(self.model)
        resident_entry: Optional[dict[str, Any]] = None
        others: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("model") or "")
            if not name:
                continue
            if resident_entry is None and canonical_model_name(name) == target:
                resident_entry = entry
            else:
                others.append(name)
        resident = resident_entry is not None
        resident_ctx = (_int_or_none(resident_entry.get("context_length"))
                        if resident_entry else None)

        # A resident that disappeared while we wait was evicted for us.
        current = {canonical_model_name(name): name for name in others}
        if not resident and self._others_before:
            for key, name in self._others_before.items():
                if key not in current and name not in self._evicted:
                    self._evicted.append(name)
        if resident:
            self._evicted.clear()
        self._others_before = current

        blob_names: list[str] = []
        try:
            blob_names = list(self._blobs() or [])
        except Exception:
            blob_names = []
        mine: Optional[RunnerProcess] = None
        if blob_names:
            mine = next((proc for proc in processes if proc.holds(blob_names)), None)
        if mine is None and not resident:
            # No blob match (an unreadable Modelfile) still leaves a usable
            # signal: exactly one freshly spawned runner while we wait is ours.
            young = [proc for proc in processes if proc.age_s <= RUNNER_YOUNG_S]
            if len(young) == 1:
                mine = young[0]
        other_runner = any(proc is not mine and proc.age_s <= RUNNER_YOUNG_S
                           for proc in processes)

        pending_info: Optional[dict[str, Any]] = None
        try:
            pending_info = self._pending()
        except Exception:
            pending_info = None
        requested: Optional[int] = None
        try:
            requested = self._requested_ctx()
        except Exception:
            requested = None
        reload_pending = False
        if pending_info:
            kind = pending_info.get("kind")
            matches = (resident and requested and resident_ctx == requested)
            if not (matches and kind == "ctx"):
                reload_pending = True
        if resident and requested and resident_ctx is not None \
                and resident_ctx != requested:
            reload_pending = True

        facts = StageFacts(
            model=self.model,
            resident=resident,
            resident_ctx=resident_ctx,
            requested_ctx=requested,
            inflight=inflight,
            warming=warming,
            reload_pending=bool(reload_pending),
            pending_from=(pending_info or {}).get("from") or resident_ctx,
            pending_to=(pending_info or {}).get("to") or requested,
            others=tuple(others[:3]),
            runner=mine,
            other_runner=other_runner,
            evicted=tuple(self._evicted[-2:]),
            gpu_used_mib=(hardware.apps.get(mine.pid) if mine is not None else None),
            rss=(mine.rss_bytes or None) if mine is not None else None,
            model_size=self._model_size_value(),
            context_used=self._safe(self._context_used),
            ps_failed=ps_failed,
        )
        # Hardware direction is the last tiebreaker: a load still not visible
        # as a runner can only be guessed at from movement.
        if not resident and mine is None and not facts.others \
                and not facts.evicted and not ps_failed:
            if (gpu_delta is not None and gpu_delta < -VRAM_STEP_MIB) \
                    or (ram_delta is not None and ram_delta < -256 * 2 ** 20) \
                    or (cpu_busy is not None and cpu_busy >= CPU_BUSY_MIN):
                facts.activity = True

        key, detail = classify_stage(facts)
        with self._lock:
            if key == STAGE_EVICTING:
                # An eviction is a one-shot announcement: the next sample
                # describes whatever the machine does with the freed room.
                self._evicted = []
            previous_stage = self._stage
            since = previous_stage.since if key == previous_stage.key else now
            stage = Stage(key=key, detail=detail, since=since)
            line = render_stage_line(stage, now - since)
            # The rendered line is compared, not just the detail: the elapsed
            # suffix changes every second, so the attached terminal's prompt
            # hint keeps ticking during a long load instead of freezing.
            changed = (stage.key != previous_stage.key
                       or line != self._last_line)
            self._stage = stage
            self._last_line = line
            self._resident_last = resident
        if changed:
            self._notify()
        return stage

    def _safe(self, getter: Callable[[], Any]) -> Any:
        try:
            return getter()
        except Exception:
            return None

    def _notify(self) -> None:
        callback = self.on_change
        if callback is None:
            return
        try:
            callback()
        except Exception:
            pass



def _gpu_free_delta(prev: HardwareSample, now: HardwareSample) -> Optional[int]:
    if not prev.gpu or not now.gpu:
        return None
    return sum(item[1] for item in now.gpu) - sum(item[1] for item in prev.gpu)


def _ram_delta(prev: HardwareSample, now: HardwareSample) -> Optional[int]:
    if prev.ram is None or now.ram is None:
        return None
    return now.ram[1] - prev.ram[1]


# Re-exported for the REPL and tests: `StageTracker` only needs the helpers.
__all__ = [
    "Stage",
    "StageFacts",
    "StageTracker",
    "STAGE_EVICTING",
    "STAGE_LOADING",
    "STAGE_PREFILL",
    "STAGE_READY",
    "STAGE_RELOAD",
    "STAGE_RELOADING",
    "STAGE_UNREACHABLE",
    "STAGE_WAITING",
    "classify_stage",
    "load_pending_line",
    "parse_show_blobs",
    "render_stage_line",
]
