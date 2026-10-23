"""The guarded call path: run one tool, publish its files, cap its output."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from typing import Any

from .spec import (ToolCancelled, ToolContext, ToolError, ToolHandlerResult,
                   ToolOutcome, ToolSpec, truncate_output)


def _publish_artifacts(paths: list[str],
                       ctx: ToolContext) -> tuple[list[dict[str, Any]], list[str]]:
    """Register a handler's produced files through the turn's publisher.

    Regular files only (directories are skipped silently); a registration
    failure becomes a note in the tool text, never a failed tool -- the
    write itself succeeded and the refs that registered still attach.
    """
    publisher = getattr(ctx, "artifacts", None)
    refs: list[dict[str, Any]] = []
    notes: list[str] = []
    if publisher is None:
        return refs, notes
    for path in paths:
        try:
            if not path or not os.path.isfile(path):
                continue
            ref = publisher.publish_path(path)
        except Exception as exc:  # ApiError/OSError: a note, not a failure
            notes.append(f"[artifact not published: {path} ({exc})]")
            continue
        refs.append(ref.to_dict())
    return refs, notes


def execute_tool(spec: ToolSpec, args: dict[str, Any],
                 ctx: ToolContext) -> ToolOutcome:
    """Run one handler, converting everything into a model-readable result."""
    started = time.monotonic()
    outcome = ToolOutcome()
    ctx.title_note = ""
    cancel = getattr(ctx, "cancel", None)
    if cancel is not None and cancel.cancelled:
        outcome.text = f"{spec.name} not run: the turn was stopped"
        outcome.error = "cancelled"
        outcome.cancelled = True
        return outcome
    try:
        result = spec.handler(args, ctx)
        if isinstance(result, ToolHandlerResult):
            outcome.text = result.text
            outcome.artifacts, notes = _publish_artifacts(result.artifact_paths, ctx)
            if notes:
                outcome.text = f"{outcome.text}\n" + "\n".join(notes)
        else:
            outcome.text = str(result)
        outcome.ok = True
    except ToolCancelled as exc:
        outcome.text = str(exc)
        outcome.error = "cancelled"
        outcome.cancelled = True
    except ToolError as exc:
        outcome.text = str(exc)
        outcome.error = str(exc)
    except subprocess.TimeoutExpired as exc:
        outcome.text = f"{spec.name} timed out ({exc})"
        outcome.error = outcome.text
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # a broken tool must not kill the session
        outcome.text = f"{spec.name} failed: {exc}"
        outcome.error = outcome.text
    outcome.duration = time.monotonic() - started
    outcome.text = truncate_output(outcome.text, ctx.max_output)
    outcome.title_note = ctx.title_note
    return outcome


#: Tools whose handlers block on network I/O. The runner moves them to a
#: daemon worker so a stopped turn abandons the wait instead of hanging on it;
#: the result is simply discarded.
ABANDON_ON_CANCEL = frozenset({"fetch_url", "web_search"})


def _cancelled_outcome(name: str) -> ToolOutcome:
    return ToolOutcome(text=f"{name} cancelled: the turn was stopped; result discarded",
                       error="cancelled", cancelled=True)


class ToolRunner:
    """Runs one tool call, honouring the turn's cancel token.

    Subprocess tools kill their process group (through `run_cancellable()`);
    network tools run in a daemon worker and are *abandoned* on cancel -- the
    work finishes on its own and its result is dropped, nothing waits for it.
    """

    def __init__(self, poll: float = 0.05) -> None:
        self.poll = poll

    def run(self, spec: ToolSpec, args: dict[str, Any],
            ctx: ToolContext) -> ToolOutcome:
        cancel = getattr(ctx, "cancel", None)
        if cancel is not None and cancel.cancelled:
            return _cancelled_outcome(spec.name)
        if cancel is not None and spec.name in ABANDON_ON_CANCEL:
            return self._run_abandonable(spec, args, ctx)
        return execute_tool(spec, args, ctx)

    def _run_abandonable(self, spec: ToolSpec, args: dict[str, Any],
                         ctx: ToolContext) -> ToolOutcome:
        box: "queue.Queue[ToolOutcome]" = queue.Queue(maxsize=1)

        def work() -> None:
            try:
                box.put(execute_tool(spec, args, ctx))
            except BaseException:  # pragma: no cover - execute_tool never raises
                pass

        worker = threading.Thread(target=work, name=f"tool-{spec.name}", daemon=True)
        worker.start()
        while True:
            try:
                return box.get(timeout=self.poll)
            except queue.Empty:
                if ctx.cancel is not None and ctx.cancel.cancelled:
                    return _cancelled_outcome(spec.name)
