"""Regression tests for the ollamarun API layer.

Covers the agent runtime contract, the turn manager, the HTTP server, the
OpenAI-compatible surface, the native agent API, artifacts and context
fitting.  The canonical command is `python3 test_ollamarun.py`, which runs
this file through `run_all`; `python3 test_ollamarun_serve.py` runs this half
alone.
"""
import base64
import contextlib
import hashlib
import http.client
import io
import json
import os
import queue
import re
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ollamarun_lib as orun
from test_ollamarun import (CTX_SHOW, _BLANK_PDF, _MINIMAL_PDF,  # harness
                            _FakeClient, _check, _fit_check, _plain)


def test_runtime_events() -> int:
    """Phase 1: a headless turn through AgentRuntime emits the whole vocabulary."""
    failures = 0
    script = [
        [{"message": {"content": "checking", "tool_calls": [
            {"function": {"name": "run_shell",
                          "arguments": {"command": "echo hi"}}}]}}],
        [{"message": {"content": "done"}, "done": True,
         "prompt_eval_count": 7, "eval_count": 2,
         "prompt_eval_duration": 1_000_000_000, "eval_duration": 1_000_000_000}],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        repl.auto_approve = True
        events = []
        repl.event_sink = events.append
        with orun.console.capture() as cap:
            result = repl.send("check")
        out = _plain(cap.get())
        types = [event.type for event in events]
        failures = _check(failures, "headless turn returns the final answer",
                          result.content == "done", result.content)
        failures = _check(failures, "headless turn writes nothing to stdout",
                          out.strip() == "", repr(out[:120]))
        for wanted in ("turn.started", "message.delta", "message.completed",
                       "tool.call", "approval.resolved", "tool.result",
                       "turn.completed"):
            failures = _check(failures, f"event {wanted}", wanted in types, str(types))
        deltas = [event.data.get("text") for event in events
                  if event.type == "message.delta"]
        failures = _check(failures, "message.delta carries the streamed text",
                          "".join(deltas) == "checkingdone", str(deltas))
        roles = [message.get("role") for message in session.messages]
        failures = _check(failures, "the tool round is recorded in the session",
                          roles == ["user", "assistant", "tool", "assistant"]
                          and session.messages[2]["tool_name"] == "run_shell",
                          str(roles))
        failures = _check(failures, "turn.completed carries the stats",
                          any(event.type == "turn.completed"
                              and event.data.get("stats") for event in events))
    return failures


def test_headless_renderer() -> int:
    failures = 0
    events = []
    renderer = orun.HeadlessRenderer(on_event=events.append)
    with orun.console.capture() as cap:
        result = renderer.stream(iter([
            {"message": {"thinking": "hmm"}},
            {"message": {"content": "Hi"}},
            {"message": {"content": " there"}},
            {"done": True, "eval_count": 5, "eval_duration": 1_000_000_000,
             "prompt_eval_count": 3, "prompt_eval_duration": 1_000_000_000},
        ]))
    text = _plain(cap.get())
    failures = _check(failures, "headless renderer accumulates content",
                      result.content == "Hi there" and result.thinking == "hmm",
                      f"{result.content!r} {result.thinking!r}")
    failures = _check(failures, "headless renderer prints nothing",
                      text.strip() == "", repr(text[:120]))
    failures = _check(failures, "headless renderer emits channel deltas",
                      [event.data.get("channel") for event in events]
                      == ["thinking", "content", "content"]
                      and [event.data.get("text") for event in events]
                      == ["hmm", "Hi", " there"],
                      str([event.to_dict() for event in events]))
    failures = _check(failures, "headless renderer keeps the stats",
                      result.stats.get("prompt_tokens") == 3
                      and result.stats.get("completion_tokens") == 5,
                      str(result.stats))
    return failures


def test_session_store_lock() -> int:
    """Concurrent session saves (serve threads) must not lose index entries."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        sessions = [orun.Session.create(f"m{i}", "http://h", title=f"t{i}")
                    for i in range(8)]

        def save(session):
            for _ in range(5):
                store.save(session)

        threads = [threading.Thread(target=save, args=(session,))
                   for session in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        ids = {entry["id"] for entry in store.entries()}
        failures = _check(failures, "every concurrent save lands in the index",
                          ids == {session.id for session in sessions},
                          f"{len(ids)} of {len(sessions)}")
    return failures


def test_session_store_cross_process_lock() -> int:
    """Two processes saving through one index keep every entry (flock)."""
    failures = 0
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        repo = str(Path(__file__).parent)
        template = (
            "import sys; sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "import ollamarun_lib as orun\n"
            "store = orun.SessionStore(sessions_dir=Path({tmp!r}) / 's',"
            " index_path=Path({tmp!r}) / 'i.json')\n"
            "for i in range(5):\n"
            "    store.save(orun.Session.create('m{tag}-%d' % i, 'http://h'))\n"
        )
        procs = [subprocess.Popen(
                     [sys.executable, "-c",
                      template.format(repo=repo, tmp=tmp, tag=tag)],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for tag in ("a", "b")]
        results = [(proc.communicate(), proc.returncode) for proc in procs]
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        entries = store.entries()
        failures = _check(failures, "both processes saved cleanly",
                          all(code == 0 for _out, code in results),
                          str([(out[1].decode()[-160:], code)
                               for out, code in results]))
        failures = _check(failures, "every cross-process save reaches the index",
                          len(entries) == 10, f"{len(entries)} entries")
    return failures


def test_ui_interactive() -> int:
    failures = 0
    failures = _check(failures, "Ui defaults to non-interactive",
                      orun.Ui.interactive is False and not orun.Ui().interactive)
    console_ui = orun.ConsoleUi()
    failures = _check(failures, "ConsoleUi follows the console",
                      console_ui.interactive == orun.console.is_terminal)
    if orun.HAS_TEXTUAL:
        class _App:
            def call_from_thread(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        failures = _check(failures, "TuiUi is always interactive",
                          orun.TuiUi(_App()).interactive)
    return failures


def test_tool_cancellation() -> int:
    failures = 0
    ran = []
    spec = orun.ToolSpec(name="run_shell", description="", parameters={},
                         handler=lambda args, ctx: ran.append(1) or "ran")
    token = orun.CancelToken()
    token.cancel()
    ctx = orun.ToolContext(cwd=".", timeout=10, cancel=token)
    outcome = orun.ToolRunner().run(spec, {}, ctx)
    failures = _check(failures, "a cancelled turn never starts the tool",
                      outcome.cancelled and not ran, f"{outcome.cancelled} {ran}")

    slow = orun.ToolSpec(name="fetch_url", description="", parameters={},
                         handler=lambda args, ctx: (time.sleep(5), "late")[1])
    token2 = orun.CancelToken()
    ctx2 = orun.ToolContext(cwd=".", timeout=10, cancel=token2)
    timer = threading.Timer(0.2, token2.cancel)
    started = time.monotonic()
    timer.start()
    try:
        outcome = orun.ToolRunner(poll=0.02).run(slow, {}, ctx2)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    failures = _check(failures, "an abandoned network tool returns without waiting",
                      outcome.cancelled and elapsed < 2.0, f"{elapsed:.2f}s")

    token3 = orun.CancelToken()
    ctx3 = orun.ToolContext(cwd=".", timeout=30, cancel=token3)
    shell = orun.ToolSpec(name="run_shell", description="", parameters={},
                          handler=orun.HANDLERS["shell"])
    timer = threading.Timer(0.3, token3.cancel)
    started = time.monotonic()
    timer.start()
    try:
        outcome = orun.ToolRunner().run(shell, {"command": "sleep 30"}, ctx3)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    failures = _check(failures, "a cancelled shell command dies with its process group",
                      outcome.cancelled and elapsed < 3.0,
                      f"{elapsed:.2f}s {outcome.text!r}")
    return failures


def test_cancellation_coverage() -> int:
    """Nested subprocess callers refuse a cancelled turn without spawning."""
    failures = 0
    token = orun.CancelToken()
    token.cancel()

    try:
        orun.run_chem_inventory(sys.executable,
                                ["-c", "import time; time.sleep(30)"],
                                timeout=30, cancel=token)
        raised = None
    except Exception as exc:  # the test wants the exception type
        raised = exc
    failures = _check(failures, "a cancelled chem call refuses to start",
                      isinstance(raised, orun.CommandCancelled), repr(raised))

    try:
        orun.run_tts_say(sys.executable, "hi", timeout=30, cancel=token)
        raised = None
    except Exception as exc:
        raised = exc
    failures = _check(failures, "a cancelled speak call refuses to start",
                      isinstance(raised, orun.CommandCancelled), repr(raised))

    spec = orun.ToolSpec(name="list_voices", description="", parameters={},
                         handler=orun.HANDLERS["list_voices"])
    ctx = orun.ToolContext(cwd=".", timeout=5, cancel=token,
                           tts_remote=sys.executable)
    outcome = orun.execute_tool(spec, {}, ctx)
    failures = _check(failures, "the voices tool reports the cancellation",
                      outcome.cancelled and not outcome.ok, str(outcome))

    if shutil.which("ssh"):
        try:
            orun.ssh_free_disk("example.invalid", cancel=token)
            raised = None
        except Exception as exc:
            raised = exc
        failures = _check(failures, "a cancelled ssh probe refuses to start",
                          isinstance(raised, orun.CommandCancelled), repr(raised))
        for probe in (orun.ssh_gpu_probe, orun.ssh_ram_probe):
            try:
                probe("example.invalid", cancel=token)
                raised = None
            except Exception as exc:
                raised = exc
            failures = _check(failures,
                              f"a cancelled {probe.__name__} refuses to start",
                              isinstance(raised, orun.CommandCancelled), repr(raised))

        remote_ctx = orun.ToolContext(
            cwd=".", timeout=5, cancel=token, ssh_target="example.invalid",
            client=type("C", (), {"base": "http://192.0.2.1:11434"})())
        try:
            orun.HANDLERS["system_status"]({"target": "ollama"}, remote_ctx)
            raised = None
        except Exception as exc:
            raised = exc
        failures = _check(failures, "system_status threads the token into its probes",
                          isinstance(raised, orun.CommandCancelled), repr(raised))
    return failures


def test_file_mention_inlining() -> int:
    """`@path` really inlines the file into the stored (and sent) message."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        notes = Path(tmp) / "notes.txt"
        notes.write_text("SECRET-FILE-CONTENT\n", encoding="utf-8")
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[[{"message": {"content": "ok"},
                                            "done": True}]])
        config = dict(orun.DEFAULT_CONFIG)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, config, client, store, session)
        repl._build_prompt_session = lambda: None
        with orun.console.capture():
            repl.send(f"@{notes} explain")
        stored = session.messages[0].get("content", "")
        failures = _check(failures, "the REPL stores the expanded message",
                          "SECRET-FILE-CONTENT" in stored
                          and f'<file name="{notes}">' in stored,
                          stored[:200])
        sent = next((message for message in client.chat_payloads[-1].get("messages", [])
                     if message.get("role") == "user"), {})
        failures = _check(failures, "the REPL sends the inlined file to the model",
                          "SECRET-FILE-CONTENT" in sent.get("content", "")
                          and f'<file name="{notes}">' in sent.get("content", ""),
                          str(sent)[:200])
        failures = _check(failures, "the stored user message lists the file",
                          session.messages[0].get("files") == [str(notes)],
                          str(session.messages[0].get("files")))

        from ollamarun_lib.run import run_once
        session2 = orun.Session.create("m1", client.base)
        repl2 = orun.Repl(args, config, client, store, session2)
        repl2._build_prompt_session = lambda: None
        with orun.console.capture():
            code = run_once(args, config, client, store, session2,
                            f"@{notes} explain", repl2)
        stored2 = session2.messages[0].get("content", "")
        failures = _check(failures, "the one-shot path inlines too",
                          code == 0 and "SECRET-FILE-CONTENT" in stored2,
                          f"{code} {stored2[:200]}")
    return failures


def test_python310_syntax() -> int:
    """Every module must parse as Python 3.10 (the pinned interpreter)."""
    failures = 0
    import ast

    repo = Path(__file__).parent
    # rglob: a pipeline stage is allowed to be a package of submodules.
    paths = sorted((repo / "ollamarun_lib").rglob("*.py")) + [repo / "ollamarun"]
    bad = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"),
                      filename=str(path), feature_version=(3, 10))
        except SyntaxError as exc:
            bad.append(f"{path.name}: {exc}")
    failures = _check(failures, "every module parses as Python 3.10",
                      not bad, "; ".join(bad))
    return failures


def test_pyflakes_clean() -> int:
    """The pyflakes gate: the package, the entry script and both test files.

    The only tolerated findings are the deliberate re-export sites — the
    `constants.py` hub's imported-but-unused names and every package
    `__init__.py` (pure re-export hubs) — so any other unused import,
    undefined name or dead local fails the suite.  pyflakes is a required
    test dependency (like `openai`).
    """
    failures = 0
    try:
        from pyflakes import api as flakes_api
        from pyflakes import reporter as flakes_reporter
    except ImportError:
        print("FAIL  pyflakes is not installed "
              "(install the test dependency: pip install --user pyflakes)")
        return 1
    output = io.StringIO()
    reporter = flakes_reporter.Reporter(output, output)
    repo = Path(__file__).resolve().parent
    targets = sorted((repo / "ollamarun_lib").rglob("*.py"))
    targets += sorted((repo / "examples").glob("*.py"))
    targets += [repo / "ollamarun", repo / "test_ollamarun.py",
                repo / "test_ollamarun_serve.py"]
    for path in targets:
        flakes_api.checkPath(str(path), reporter)

    def tolerated(finding: str) -> bool:
        head, _, message = finding.partition(": ")
        try:
            rel = Path(head.rsplit(":", 2)[0]).relative_to(repo).as_posix()
        except ValueError:
            return False
        if rel == "ollamarun_lib/constants.py":
            return "imported but unused" in message
        if rel.startswith("ollamarun_lib/") and rel.endswith("__init__.py"):
            # Pure re-export hub: every name is deliberate (and the star
            # lines themselves always flag as re-exports).
            return True
        return False

    findings = [line for line in output.getvalue().splitlines() if line.strip()]
    real = [line for line in findings if not tolerated(line)]
    failures = _check(failures, "pyflakes: clean outside the deliberate "
                      "re-exports", not real, "\n".join(real[:6]))
    return failures


#: A module this large mixes too many responsibilities to review comfortably;
#: the staged refactor splits the named offenders into packages of submodules
#: and then removes the entries one by one.  A file that grows past the
#: budget without an entry fails the gate, and so does a stale entry, so the
#: allowlist can only shrink.
MODULE_SIZE_BUDGET = 800
SIZE_BUDGET_ALLOWLIST = {
    "ollamarun_lib/repl/core.py",
    "ollamarun_lib/agent_api.py",
    "ollamarun_lib/session_host.py",
    "ollamarun_lib/cli.py",
    "ollamarun_lib/mathrender.py",
    "ollamarun_lib/openai_api.py",
    "ollamarun_lib/helpers.py",
    "ollamarun_lib/interactive.py",
    "ollamarun_lib/host_client.py",
    "ollamarun_lib/tui.py",
    "ollamarun_lib/artifacts.py",
    "ollamarun_lib/renderer.py",
}


def test_module_size_budget() -> int:
    """No module slips past the size budget, and no allowlist entry goes stale.

    The budget is a review aid, not a hard design law: a genuinely cohesive
    module can stay listed, but adding an entry is a decision someone has to
    make in this test rather than a silent drift.
    """
    failures = 0
    repo = Path(__file__).resolve().parent
    sizes = {}
    for path in sorted((repo / "ollamarun_lib").rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        sizes[rel] = len(path.read_text(encoding="utf-8").splitlines())
    over = [f"{rel}: {count}" for rel, count in sizes.items()
            if count > MODULE_SIZE_BUDGET and rel not in SIZE_BUDGET_ALLOWLIST]
    stale = sorted(rel for rel in SIZE_BUDGET_ALLOWLIST
                   if sizes.get(rel, 0) <= MODULE_SIZE_BUDGET)
    failures = _check(failures, f"no module over {MODULE_SIZE_BUDGET} lines "
                      "outside the shrinking allowlist", not over,
                      "\n".join(over[:6]))
    failures = _check(failures, "the size-budget allowlist has no stale entries",
                      not stale, ", ".join(stale[:6]))
    return failures


def test_turn_result_mapping() -> int:
    """`to_turn_result` maps a GenerationResult onto the API-ready summary."""
    failures = 0
    hooks = orun.TurnHooks(
        client=None, store=None, session=orun.Session.create("m1", "http://h"),
        settings=orun.TurnSettings(),
        payload_for=lambda messages, with_tools: {},
        present_round=lambda payload, after_tools, cancel: orun.GenerationResult(),
        approver=lambda request: orun.ApprovalDecision(run=False))
    runtime = orun.AgentRuntime(hooks)
    runtime.tool_round = 2

    done = orun.GenerationResult(content="done", thinking="hmm",
                                 stats={"prompt_tokens": 5, "completion_tokens": 2})
    turn = runtime.to_turn_result(done, session_id="s1", turn_id="t1")
    failures = _check(failures, "a completed turn maps cleanly",
                      turn.id == "t1" and turn.session_id == "s1"
                      and turn.status == "completed"
                      and turn.final_message["content"] == "done"
                      and turn.final_message["thinking"] == "hmm"
                      and turn.usage["prompt_tokens"] == 5
                      and turn.rounds == 2 and turn.error is None,
                      str(turn))

    failed = runtime.to_turn_result(orun.GenerationResult(error="boom"))
    cancelled = runtime.to_turn_result(orun.GenerationResult(cancelled=True,
                                                             content="partial"))
    failures = _check(failures, "failure and cancellation get their status",
                      failed.status == "failed" and failed.error == "boom"
                      and cancelled.status == "cancelled"
                      and cancelled.final_message["content"] == "partial",
                      f"{failed} {cancelled}")

    request = orun.TurnRequest(messages=[{"role": "user", "content": "hi"}],
                               model="m1", session_id="s1")
    failures = _check(failures, "TurnRequest defaults match the API plan",
                      request.persist is True
                      and request.messages[0]["role"] == "user",
                      str(request))
    return failures


def test_turn_event_envelope() -> int:
    """TurnEvent ids, ordering and deterministic serialization."""
    failures = 0
    event = orun.TurnEvent(turn_id="turn_1", session_id="s1", seq=3,
                           ts="2026-10-17T10:00:00.000000-03:00",
                           type=orun.TOOL_RESULT, data={"ok": True})
    failures = _check(failures, "the envelope id is turn_id:seq",
                      event.id == "turn_1:3", event.id)
    payload = event.to_dict()
    failures = _check(failures, "the envelope serializes in fixed key order",
                      list(payload) == ["id", "seq", "type", "turn_id",
                                        "session_id", "ts", "data"],
                      str(list(payload)))
    failures = _check(failures, "every field round-trips through JSON",
                      json.loads(json.dumps(payload)) == payload, str(payload))
    failures = _check(failures, "stream.gap is part of the vocabulary",
                      orun.STREAM_GAP == "stream.gap"
                      and orun.STREAM_GAP in orun.EVENT_TYPES
                      and orun.STREAM_GAP not in orun.TERMINAL_EVENTS)
    return failures


def _collect(subscription, timeout: float = 0.2) -> list:
    events = []
    while True:
        event = subscription.get(timeout=timeout)
        if event is None:
            break
        events.append(event)
    return events


def test_turn_manager_lifecycle() -> int:
    """A turn runs on a worker, envelopes its events and stores a result."""
    failures = 0
    manager = orun.TurnManager()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {"model": "m1"}))
        sink(orun.RuntimeEvent(orun.MESSAGE_DELTA,
                               {"channel": "content", "text": "hi"}))
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {"content": "hi"}))
        return orun.TurnResult(id="x", session_id="s1", status="completed",
                               final_message={"role": "assistant", "content": "hi"},
                               usage={"completion_tokens": 1}, rounds=1)

    handle = manager.submit(session_id="s1", work=work)
    snapshot = manager.wait(handle.id, timeout=5)
    failures = _check(failures, "the turn completes and stores its result",
                      snapshot is not None and snapshot.status == "completed"
                      and snapshot.result is not None
                      and snapshot.result.final_message["content"] == "hi"
                      and snapshot.seq == 3 and snapshot.oldest_seq == 1,
                      str(snapshot))
    replay = _collect(manager.subscribe(handle.id, after=0))
    failures = _check(failures, "replay after=0 returns every envelope in order",
                      [event.seq for event in replay] == [1, 2, 3]
                      and [event.id for event in replay]
                      == [f"{handle.id}:1", f"{handle.id}:2", f"{handle.id}:3"],
                      str([(event.seq, event.type) for event in replay]))
    after = _collect(manager.subscribe(handle.id, after=2))
    failures = _check(failures, "replay after=k returns only the tail",
                      [event.seq for event in after] == [3], str(after))
    live = manager.subscribe(handle.id)
    failures = _check(failures, "a live-only subscriber on a done turn sees nothing",
                      live.get(timeout=0.05) is None and live.at_terminal)
    return failures


def test_turn_manager_synthesized_terminal() -> int:
    """A closure that dies (or skips the terminal) still gets one."""
    failures = 0
    manager = orun.TurnManager()

    def boom(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        raise RuntimeError("boom")

    handle = manager.submit(session_id="crash", work=boom)
    snapshot = manager.wait(handle.id, timeout=5)
    events = _collect(manager.subscribe(handle.id, after=0))
    failures = _check(failures, "a raising closure becomes a failed turn",
                      snapshot is not None and snapshot.status == "failed"
                      and snapshot.error == orun.INTERNAL_ERROR_MESSAGE
                      and snapshot.synthetic_terminal
                      and snapshot.result is not None
                      and snapshot.result.error == "RuntimeError: boom",
                      str(snapshot))
    failures = _check(failures, "the synthesized terminal is the last event",
                      [event.type for event in events]
                      == [orun.TURN_STARTED, orun.TURN_ERROR]
                      and events[-1].data.get("synthetic") is True
                      and events[-1].data.get("code") == "internal_error",
                      str([(event.type, event.data) for event in events]))
    payload = json.dumps([event.to_dict() for event in events])
    failures = _check(failures, "the raw crash text stays off the event stream",
                      "RuntimeError" not in payload and "boom" not in payload,
                      payload[:200])

    def quiet(sink, cancel):
        return orun.TurnResult(id="x", session_id="quiet", status="completed")

    handle2 = manager.submit(session_id="quiet", work=quiet)
    snapshot2 = manager.wait(handle2.id, timeout=5)
    failures = _check(failures, "a result without a terminal synthesizes one",
                      snapshot2 is not None and snapshot2.status == "completed"
                      and snapshot2.synthetic_terminal,
                      str(snapshot2))
    return failures


def test_turn_buffer_truncation() -> int:
    """Overflow drops the oldest events; replay says so with a gap."""
    failures = 0
    manager = orun.TurnManager(buffer_events=2)

    def work(sink, cancel):
        for index in range(4):
            sink(orun.RuntimeEvent(orun.MESSAGE_DELTA,
                                   {"channel": "content", "text": str(index)}))
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work)
    snapshot = manager.wait(handle.id, timeout=5)
    failures = _check(failures, "the record keeps only the newest events",
                      snapshot is not None and snapshot.truncated
                      and snapshot.oldest_seq == 4 and snapshot.latest_seq == 5,
                      str(snapshot))
    events = _collect(manager.subscribe(handle.id, after=0))
    gap = events[0] if events else None
    failures = _check(failures, "a stale replay starts with stream.gap",
                      gap is not None and gap.type == orun.STREAM_GAP
                      and gap.data.get("reason") == "truncated_buffer"
                      and gap.seq == 3 and gap.data.get("oldest_seq") == 4,
                      str(gap))
    failures = _check(failures, "the gap is followed by the surviving events",
                      [event.seq for event in events[1:]] == [4, 5],
                      str([event.seq for event in events[1:]]))

    # The same gap over the events endpoint, through the HTTP surface.
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chunk_count=6)
        server, _api, _store, thread = _start_agent_api(
            tmp, client, config_extra={"serve_turn_event_buffer": 3})
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("stream"))
            turn = json.loads(raw)
            _poll_turn(server, turn["id"])
            status, _headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn['id']}/events?after=0")
            payloads = [json.loads(item) for item in _sse_events(raw)]
            failures = _check(
                failures, "a truncated HTTP replay starts with stream.gap",
                status == 200 and payloads
                and payloads[0].get("type") == "stream.gap"
                and payloads[0].get("data", {}).get("reason") == "truncated_buffer"
                and payloads[-1].get("type") == "turn.completed",
                str(payloads[:2])[:240])
        finally:
            _stop_api(server, thread)
    return failures


def test_turn_expired() -> int:
    """A swept turn answers TurnExpired, not UnknownTurn; ttl 0 keeps it."""
    failures = 0
    manager = orun.TurnManager(completed_ttl=30.0)

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work)
    # Join the worker directly: wait()/get() would sweep the record first,
    # hiding whether subscribe() enforces retention on its own.
    manager._records[handle.id].thread.join(5)
    # Age the record past its retention so subscribe()'s own sweep expires it.
    manager._records[handle.id].finished_at -= 60.0
    try:
        manager.subscribe(handle.id)
        raised = None
    except orun.TurnExpired as exc:
        raised = exc
    failures = _check(failures, "a swept turn reports expired with its status",
                      raised is not None and raised.status == "completed"
                      and manager.get(handle.id) is None, repr(raised))

    # `completed_ttl: 0` is the "keep forever" convention: even a sweep long
    # after the turn finished leaves the record in place.
    keeper = orun.TurnManager(completed_ttl=0.0)

    def work2(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="y", session_id="s", status="completed")

    kept = keeper.submit(session_id="s", work=work2)
    keeper.wait(kept.id, timeout=5)
    keeper.sweep(now=time.monotonic() + 10_000_000)
    failures = _check(failures, "a zero retention keeps completed turns",
                      keeper.get(kept.id) is not None,
                      repr(keeper.get(kept.id)))
    return failures


def test_turn_manager_session_busy() -> int:
    """One active turn per session; a competing submit is rejected."""
    failures = 0
    manager = orun.TurnManager()
    gate = threading.Event()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        gate.wait(5)
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="a", work=work)
    try:
        manager.submit(session_id="a", work=work)
        raised = None
    except orun.SessionBusy as exc:
        raised = exc
    failures = _check(failures, "a busy session names its active turn",
                      raised is not None and raised.active_turn_id == handle.id,
                      repr(raised))
    other = manager.submit(session_id="b", work=work)
    failures = _check(failures, "a different session runs in parallel",
                      manager.active_for("b") == other.id)
    gate.set()
    manager.wait(handle.id, timeout=5)
    manager.wait(other.id, timeout=5)
    failures = _check(failures, "the session frees when the turn finishes",
                      manager.active_for("a") is None
                      and manager.active_for("b") is None)
    again = manager.submit(session_id="a", work=work)
    manager.wait(again.id, timeout=5)
    failures = _check(failures, "a new turn is accepted after the finish",
                      again.snapshot() is not None
                      and again.snapshot().status == "completed")
    return failures


def test_turn_manager_max_active() -> int:
    """The global active-turn cap rejects with TooManyTurns."""
    failures = 0
    manager = orun.TurnManager(max_active=1)
    gate = threading.Event()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        gate.wait(5)
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="a", work=work)
    try:
        manager.submit(session_id="b", work=work)
        raised = None
    except orun.TooManyTurns as exc:
        raised = exc
    failures = _check(failures, "the cap rejects with TooManyTurns",
                      raised is not None and raised.limit == 1, repr(raised))
    gate.set()
    manager.wait(handle.id, timeout=5)
    second = manager.submit(session_id="b", work=work)
    manager.wait(second.id, timeout=5)
    failures = _check(failures, "the cap frees after the turn finishes",
                      second.snapshot() is not None
                      and second.snapshot().status == "completed")
    return failures


def test_turn_manager_cancel() -> int:
    """cancel() reaches the turn's token and ends it as cancelled."""
    failures = 0
    manager = orun.TurnManager()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        for _ in range(200):
            if cancel.cancelled:
                sink(orun.RuntimeEvent(orun.TURN_CANCELLED, {"reason": "user"}))
                return orun.TurnResult(id="x", session_id="s", status="cancelled")
            time.sleep(0.005)
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work, cancel=orun.CancelToken())
    time.sleep(0.05)
    failures = _check(failures, "cancel() reports the request landed",
                      handle.cancel() is True)
    snapshot = manager.wait(handle.id, timeout=5)
    failures = _check(failures, "a cancelled turn ends cancelled",
                      snapshot is not None and snapshot.status == "cancelled",
                      str(snapshot))
    failures = _check(failures, "cancelling an unknown turn is a no-op",
                      manager.cancel("turn_nope") is False)
    return failures


def test_turn_subscriber_overflow() -> int:
    """A slow subscriber gets one gap and is closed; the turn finishes."""
    failures = 0
    manager = orun.TurnManager(subscriber_queue=2)
    gate = threading.Event()

    def work(sink, cancel):
        gate.wait(5)
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        for index in range(12):
            sink(orun.RuntimeEvent(orun.MESSAGE_DELTA,
                                   {"channel": "content", "text": str(index)}))
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work)
    subscription = manager.subscribe(handle.id)
    gate.set()
    snapshot = manager.wait(handle.id, timeout=5)
    events = _collect(subscription)
    gaps = [event for event in events if event.type == orun.STREAM_GAP]
    failures = _check(failures, "the slow subscriber is sent a gap and closed",
                      subscription.closed and subscription.gap_pending
                      and len(gaps) == 1
                      and gaps[0].data.get("reason") == "slow_subscriber",
                      str([(event.type, event.data) for event in events]))
    failures = _check(failures, "the turn still completes with no reader",
                      snapshot is not None and snapshot.status == "completed"
                      and snapshot.seq == 14, str(snapshot))
    return failures


def test_turn_manager_shutdown() -> int:
    """shutdown cancels active turns, joins, and refuses new ones."""
    failures = 0
    manager = orun.TurnManager()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        for _ in range(400):
            if cancel.cancelled:
                sink(orun.RuntimeEvent(orun.TURN_CANCELLED, {"reason": "user"}))
                return orun.TurnResult(id="x", session_id="s", status="cancelled")
            time.sleep(0.005)
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work, cancel=orun.CancelToken())
    time.sleep(0.03)
    manager.shutdown(timeout=5)
    snapshot = manager.get(handle.id)
    failures = _check(failures, "shutdown stops the active turn",
                      snapshot is not None and snapshot.status == "cancelled",
                      str(snapshot))
    try:
        manager.submit(session_id="s2", work=work)
        raised = None
    except orun.ManagerClosed as exc:
        raised = exc
    failures = _check(failures, "a shut-down manager refuses new turns",
                      raised is not None, repr(raised))
    manager.shutdown(timeout=0.1)  # idempotent
    return failures


def test_turn_manager_context() -> int:
    """The context block rides on turn.started/turn.completed and the snapshot."""
    failures = 0
    manager = orun.TurnManager()

    def work(sink, cancel):
        sink(orun.RuntimeEvent(orun.TURN_STARTED, {}))
        sink(orun.RuntimeEvent(orun.TURN_COMPLETED, {}))
        return orun.TurnResult(id="x", session_id="s", status="completed")

    handle = manager.submit(session_id="s", work=work,
                            context=lambda: {"used": 10, "total": 100,
                                             "source": "loaded runner"})
    snapshot = manager.wait(handle.id, timeout=5)
    events = _collect(manager.subscribe(handle.id, after=0))
    failures = _check(failures, "the snapshot keeps the context block",
                      snapshot is not None and snapshot.context is not None
                      and snapshot.context["used"] == 10
                      and snapshot.context["source"] == "loaded runner",
                      str(snapshot))
    decorated = {event.type: event.data.get("context") for event in events}
    failures = _check(failures, "started and completed carry the context",
                      decorated.get(orun.TURN_STARTED, {}).get("total") == 100
                      and decorated.get(orun.TURN_COMPLETED, {}).get("used") == 10,
                      str(decorated))

    broken = manager.submit(session_id="s2", work=work,
                            context=lambda: 1 / 0)
    snapshot2 = manager.wait(broken.id, timeout=5)
    failures = _check(failures, "a broken provider never blocks the turn",
                      snapshot2 is not None and snapshot2.status == "completed"
                      and snapshot2.context is None, str(snapshot2))
    return failures


def test_turn_manager_headless_turn() -> int:
    """The Phase 4 seam: a headless Repl turn enveloped by the manager."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[[{"message": {"content": "hello"},
                                            "done": True,
                                            "prompt_eval_count": 4,
                                            "eval_count": 2}]])
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        manager = orun.TurnManager()

        def work(sink, cancel):
            hooks = repl._turn_hooks()
            hooks.on_event = sink
            hooks.cancel = cancel
            renderer = orun.HeadlessRenderer(on_event=sink)

            def present(payload, after_tools, cancel_token):
                return renderer.stream(client.chat(payload, cancel=cancel_token),
                                       cancel=cancel_token)

            hooks.present_round = present
            runtime = orun.AgentRuntime(hooks)
            result = runtime.run()
            return runtime.to_turn_result(result, session_id=session.id)

        handle = manager.submit(session_id=session.id, work=work,
                                context=lambda: {"used": 3, "total": 100,
                                                 "source": "num_ctx"})
        snapshot = manager.wait(handle.id, timeout=5)
        events = _collect(manager.subscribe(handle.id, after=0))
        types = [event.type for event in events]
        failures = _check(failures, "the enveloped turn completes",
                          snapshot is not None and snapshot.status == "completed"
                          and snapshot.result is not None
                          and snapshot.result.final_message["content"] == "hello",
                          str(snapshot))
        failures = _check(failures, "the event stream is complete and ordered",
                          types[0] == orun.TURN_STARTED
                          and orun.MESSAGE_DELTA in types
                          and types[-1] == orun.TURN_COMPLETED
                          and [event.seq for event in events]
                          == list(range(1, len(events) + 1)),
                          str(types))
        started = [event for event in events
                   if event.type == orun.TURN_STARTED][0]
        failures = _check(failures, "the context block rides on turn.started",
                          started.data.get("context", {}).get("source") == "num_ctx",
                          str(started.data))
        failures = _check(failures, "the assistant message reached the session",
                          session.messages
                          and session.messages[-1].get("role") == "assistant"
                          and session.messages[-1].get("content") == "hello",
                          str(session.messages[-1] if session.messages else None))
    return failures


def test_tool_support_warning() -> int:
    """The capability warning fires once per model, after turn.started."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion"],
                             chat_script=[[{"message": {"content": "ok"},
                                            "done": True}]])
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        events = []
        repl.event_sink = events.append
        with orun.console.capture() as cap:
            repl.send("one")
        first = list(events)
        events.clear()
        with orun.console.capture():
            repl.send("two")
        second = list(events)
        warnings = [event for event in first if event.type == "turn.warning"]
        types = [event.type for event in first]
        failures = _check(failures, "a tool-less model warns once per model",
                          len(warnings) == 1
                          and not any(event.type == "turn.warning"
                                      for event in second),
                          f"{len(warnings)} {[event.type for event in second]}")
        failures = _check(failures, "turn.started precedes the warning",
                          "turn.started" in types and "turn.warning" in types
                          and types.index("turn.started") < types.index("turn.warning"),
                          str(types))
        failures = _check(failures, "the headless warning prints nothing",
                          warnings
                          and "does not advertise tool support"
                          in str(warnings[0].data.get("message"))
                          and _plain(cap.get()).strip() == "",
                          repr(_plain(cap.get())[:120]))
    return failures


def test_quiet_tools_live() -> int:
    """`/render raw` mid-turn silences tool panels from the next event on."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        hooks = repl._turn_hooks()
        failures = _check(failures, "quiet_tools follows the render mode",
                          hooks.settings.quiet_tools() is False)
        event = orun.RuntimeEvent(orun.TOOL_CALL,
                                  {"name": "run_shell", "detail": "echo hi"})
        with orun.console.capture() as cap:
            hooks.on_event(event)
        failures = _check(failures, "the tool panel prints in markdown mode",
                          "running" in _plain(cap.get()), repr(cap.get()[:120]))
        repl.render_mode = "raw"
        failures = _check(failures, "quiet_tools re-reads the mode live",
                          hooks.settings.quiet_tools() is True)
        with orun.console.capture() as cap:
            hooks.on_event(event)
        failures = _check(failures, "the tool panel is silenced after /render raw",
                          _plain(cap.get()).strip() == "", repr(cap.get()[:120]))
    return failures


def test_approval_event_contract() -> int:
    """The approver emits approval.required; the runtime emits the rest."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl.headless = True
        events = []
        repl.event_sink = events.append
        hooks = repl._turn_hooks()

        def approver(request):
            hooks.on_event(orun.approval_notice(request, reason="test"))
            return orun.ApprovalDecision(run=False, note="denied by test",
                                         action="deny")

        hooks.approver = approver
        calls = [{"function": {"name": "run_shell",
                               "arguments": {"command": "rm -rf /tmp/x"}}}]
        interrupted = orun.AgentRuntime(hooks).execute_tool_calls(
            calls, allow_prompt=True)
        types = [event.type for event in events]
        failures = _check(failures, "the approval contract order",
                          types == ["tool.call", "approval.required",
                                    "approval.resolved", "tool.result"]
                          and interrupted is False, str(types))
        results = [event for event in events if event.type == "tool.result"]
        failures = _check(failures, "a denied call is marked on the result",
                          bool(results) and results[0].data.get("ok") is False
                          and results[0].data.get("denied") is True,
                          str(results[0].data if results else None))
    return failures


def test_runtime_event_contract() -> int:
    """Phase 1 acceptance: one vocabulary, JSON-ready payloads, one terminal event."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        script = [
            [{"message": {"thinking": "hmm", "content": "checking", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": "echo hi"}}}]}},
             {"done": True}],
            [{"message": {"content": "done"}, "done": True,
             "prompt_eval_count": 7, "eval_count": 2,
             "prompt_eval_duration": 1_000_000_000, "eval_duration": 1_000_000_000}],
        ]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        repl.auto_approve = True
        events = []
        repl.event_sink = events.append
        with orun.console.capture():
            repl.send("check")
        types = [event.type for event in events]

        try:
            encoded = json.dumps([event.to_dict() for event in events])
            payload_ok = True
        except (TypeError, ValueError) as exc:
            encoded, payload_ok = str(exc), False
        failures = _check(failures, "every event payload is JSON-ready",
                          payload_ok, encoded[:200])
        unknown = sorted({event_type for event_type in types
                          if event_type not in orun.EVENT_TYPES})
        failures = _check(failures, "every emitted type is in the vocabulary",
                          not unknown, str(unknown))
        terminals = [event_type for event_type in types
                     if event_type in orun.TERMINAL_EVENTS]
        failures = _check(failures, "exactly one terminal event, and it ends the turn",
                          len(terminals) == 1 and types[-1] in orun.TERMINAL_EVENTS,
                          f"{terminals} {types[-3:]}")
        failures = _check(failures, "the turn opens with turn.started",
                          bool(types) and types[0] == "turn.started", str(types[:3]))

        started = events[0].data
        failures = _check(failures, "turn.started identifies the session and model",
                          started.get("model") == "m1"
                          and started.get("session_id") == session.id,
                          str(started))

        completed = [event.data for event in events
                     if event.type == "message.completed"]
        failures = _check(failures, "one message.completed per round, round-numbered",
                          [entry.get("round") for entry in completed] == [0, 1],
                          str([entry.get("round") for entry in completed]))
        failures = _check(failures, "message.completed carries content and thinking",
                          completed[0].get("thinking") == "hmm"
                          and completed[0].get("content") == "checking"
                          and completed[0].get("tool_calls") == 1
                          and completed[1].get("content") == "done",
                          str(completed))

        deltas = [(event.data.get("channel"), event.data.get("text"))
                  for event in events if event.type == "message.delta"]
        failures = _check(failures, "deltas stay in the content/thinking channels",
                          {channel for channel, _text in deltas} <= {"content", "thinking"}
                          and "".join(text for channel, text in deltas
                                      if channel == "content") == "checkingdone",
                          str(deltas))

        resolved = [event.data for event in events if event.type == "approval.resolved"]
        actions = {"run", "remember", "rule", "auto", "deny", "interrupt"}
        failures = _check(failures, "approval.resolved names how it was decided",
                          bool(resolved)
                          and all(entry.get("action") in actions
                                  for entry in resolved),
                          str(resolved))
        failures = _check(failures, "tool.call precedes approval.resolved precedes tool.result",
                          types.index("tool.call") < types.index("approval.resolved")
                          < types.index("tool.result"), str(types))

        final = [event.data for event in events if event.type == "turn.completed"]
        failures = _check(failures, "turn.completed carries the final round",
                          bool(final) and final[0].get("content") == "done"
                          and final[0].get("stats", {}).get("completion_tokens") == 2,
                          str(final))
        failures = _check(failures, "the session keeps the same final content",
                          session.messages[-1].get("role") == "assistant"
                          and session.messages[-1].get("content") == "done",
                          str(session.messages[-1])[:160])

        token = orun.CancelToken()
        token.cancel()
        hooks = orun.TurnHooks(
            client=client, store=store,
            session=orun.Session.create("m1", client.base),
            settings=orun.TurnSettings(no_save=True), cancel=token,
            payload_for=lambda messages, with_tools: {},
            present_round=lambda payload, after_tools, cancel: orun.GenerationResult(),
            approver=lambda request: orun.ApprovalDecision(run=False))
        cancel_events = []
        hooks.on_event = cancel_events.append
        cancelled = orun.AgentRuntime(hooks).run()
        failures = _check(failures, "a cancelled turn is terminal-cancelled",
                          cancelled.cancelled
                          and [event.type for event in cancel_events]
                          == ["turn.started", "turn.cancelled"],
                          str([event.type for event in cancel_events]))

        class _BrokenClient(_FakeClient):
            def chat(self, payload, timeout=None, cancel=None):
                raise orun.OllamaError("boom")

        broken = _BrokenClient(models=["m1"], capabilities=["completion", "tools"])
        broken_session = orun.Session.create("m1", broken.base)
        broken_repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), broken, store,
                                broken_session)
        broken_repl._build_prompt_session = lambda: None
        broken_repl.headless = True
        broken_events = []
        broken_repl.event_sink = broken_events.append
        with orun.console.capture():
            broken_result = broken_repl.send("check")
        broken_types = [event.type for event in broken_events]
        failures = _check(failures, "a failed round ends in turn.error",
                          broken_result.error == "boom"
                          and broken_types[-1] == "turn.error"
                          and "turn.completed" not in broken_types
                          and "turn.error" in orun.TERMINAL_EVENTS,
                          f"{broken_result.error!r} {broken_types}")
    return failures


def test_approval_runtime_contract() -> int:
    """The approval gate cannot be skipped, and the event order is fixed."""
    failures = 0
    ran = []
    spec = orun.ToolSpec(name="stub", description="", parameters={},
                         handler=lambda args, ctx: ran.append(dict(args)) or "STUB-RAN")
    call = [{"function": {"name": "stub", "arguments": {"x": 1}}}]

    def hooks_with(approver, token=None):
        events = []
        session = orun.Session.create("m1", "http://h")
        hooks = orun.TurnHooks(
            client=None, store=None, session=session,
            settings=orun.TurnSettings(no_save=True), cancel=token,
            registry={"stub": spec},
            tool_ctx=orun.ToolContext(cwd=".", timeout=5, max_output=1000),
            payload_for=lambda messages, with_tools: {},
            present_round=lambda payload, after_tools, cancel: orun.GenerationResult(),
            approver=approver)
        hooks.on_event = events.append
        return hooks, events

    hooks, events = hooks_with(None)

    def allow(request):
        hooks.on_event(orun.approval_notice(request, reason="needs approval"))
        return orun.ApprovalDecision(run=True, note="confirmed", action="run")

    hooks.approver = allow
    interrupted = orun.AgentRuntime(hooks).execute_tool_calls(call, allow_prompt=True)
    types = [event.type for event in events]
    failures = _check(failures, "an approved call runs in contract order",
                      types == ["tool.call", "approval.required",
                                "approval.resolved", "tool.result"]
                      and interrupted is False and ran == [{"x": 1}],
                      f"{types} {interrupted} {ran}")
    failures = _check(failures, "the approved result carries the handler output",
                      events[-1].data.get("ok") is True
                      and "STUB-RAN" in str(events[-1].data.get("text")),
                      str(events[-1].data))

    ran.clear()
    hooks, events = hooks_with(None)
    hooks.approver = lambda request: orun.ApprovalDecision(
        run=False, note="denied by test", action="deny")
    interrupted = orun.AgentRuntime(hooks).execute_tool_calls(call, allow_prompt=True)
    types = [event.type for event in events]
    failures = _check(failures, "the runtime never emits approval.required itself",
                      types == ["tool.call", "approval.resolved", "tool.result"]
                      and "approval.required" not in types and not ran,
                      f"{types} {ran}")
    failures = _check(failures, "a denial is recorded as a denied result",
                      events[-1].data.get("denied") is True
                      and "not run" in hooks.session.messages[-1].get("content", ""),
                      str(events[-1].data))

    ran.clear()
    hooks, events = hooks_with(None)

    def interrupt(request):
        raise KeyboardInterrupt

    hooks.approver = interrupt
    interrupted = orun.AgentRuntime(hooks).execute_tool_calls(call, allow_prompt=True)
    types = [event.type for event in events]
    failures = _check(failures, "Ctrl-C at the gate interrupts without running",
                      interrupted is True and not ran
                      and types == ["tool.call", "approval.resolved", "tool.result"]
                      and events[1].data.get("action") == "interrupt"
                      and events[-1].data.get("ok") is False
                      and "interrupted by user" in str(events[-1].data.get("text"))
                      and "interrupted by user"
                      in hooks.session.messages[-1].get("content", ""),
                      f"{interrupted} {types} {[event.data for event in events]}")

    token = orun.CancelToken()
    token.cancel()
    ran.clear()
    hooks, events = hooks_with(None, token)
    hooks.approver = lambda request: orun.ApprovalDecision(run=True)
    interrupted = orun.AgentRuntime(hooks).execute_tool_calls(call, allow_prompt=True)
    failures = _check(failures, "a stopped turn never starts the tool",
                      interrupted is True and not ran
                      and events[-1].data.get("cancelled") is True,
                      f"{interrupted} {ran} {[event.data for event in events]}")
    return failures


def test_unattended_turn_denial() -> int:
    """Phase 1 acceptance: a risky call cannot silently bypass the gate."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "owned.txt"
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        script = [
            [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": f"echo owned > {marker}"}}}]}},
             {"done": True}],
            [{"message": {"content": "understood"}, "done": True}],
        ]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        events = []
        repl.event_sink = events.append
        with orun.console.capture() as cap:
            result = repl.send("go")
        out = _plain(cap.get())
        tool_messages = [message for message in session.messages
                         if message.get("role") == "tool"]
        failures = _check(failures, "the risky call never ran",
                          not marker.exists() and bool(tool_messages)
                          and "not run" in tool_messages[0].get("content", ""),
                          f"{marker.exists()} {tool_messages}")
        failures = _check(failures, "the denial says the run is unattended",
                          "unattended" in tool_messages[0].get("content", ""),
                          str(tool_messages))
        types = [event.type for event in events]
        failures = _check(failures, "a non-interactive denial never parks for approval",
                          "approval.required" not in types
                          and any(event.type == "approval.resolved"
                                  and event.data.get("action") == "deny"
                                  for event in events),
                          str(types))
        failures = _check(failures, "the headless turn prints nothing and still answers",
                          out.strip() == "" and result.content == "understood",
                          f"{result.content!r} {out[:120]!r}")
    return failures


def test_cancel_reaches_production_tools() -> int:
    """Esc during a real turn's tool call must kill it, not wait for it."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "finished.txt"
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        script = [
            [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": f"sleep 4; touch {marker}"}}}]}},
             {"done": True}],
        ]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        repl.auto_approve = True
        repl.event_sink = lambda event: None

        box = {}

        def turn():
            started = time.monotonic()
            box["result"] = repl.send("run it")
            box["elapsed"] = time.monotonic() - started

        worker = threading.Thread(target=turn)
        worker.start()
        time.sleep(0.8)                  # the shell tool is now sleeping
        repl.cancel_token.cancel()       # what the stop key does
        worker.join(10)
        tool_messages = [message for message in session.messages
                         if message.get("role") == "tool"]
        failures = _check(failures, "the stopped turn returns without waiting for the tool",
                          not worker.is_alive() and box.get("elapsed", 99.0) < 2.5,
                          f"alive={worker.is_alive()} {box.get('elapsed', -1):.1f}s")
        failures = _check(failures, "the killed command never reaches its next step",
                          not marker.exists(), str(marker))
        failures = _check(failures, "the stopped tool is recorded as cancelled",
                          bool(tool_messages)
                          and "cancelled" in tool_messages[0].get("content", ""),
                          str(tool_messages))
    return failures


def test_quiet_tools_real_turn() -> int:
    """A /render raw flipped while the model streams silences that tool round."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(
            ["m1", "--render", "markdown", "--no-save", "-y"])
        script = [
            [{"message": {"content": "working"}},
             {"message": {"content": "", "tool_calls": [
                 {"function": {"name": "run_shell",
                               "arguments": {"command": "echo TOOL-OUTPUT"}}}]}},
             {"done": True}],
            [{"message": {"content": "final"}, "done": True}],
        ]

        class _FlipClient(_FakeClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.repl = None

            def chat(self, payload, timeout=None, cancel=None):
                self.chat_payloads.append(payload)
                chunks = self.chat_script[self.chat_calls % len(self.chat_script)]
                self.chat_calls += 1

                def gen():
                    yield chunks[0]
                    self.repl.render_mode = "raw"   # /render raw lands mid-stream
                    yield from chunks[1:]

                return gen()

        client = _FlipClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        client.repl = repl
        with orun.console.capture() as cap:
            result = repl.send("go")
        out = _plain(cap.get())
        tool_messages = [message for message in session.messages
                         if message.get("role") == "tool"]
        failures = _check(failures, "the tool still ran under /render raw",
                          result.content == "final" and bool(tool_messages)
                          and "TOOL-OUTPUT" in tool_messages[0].get("content", ""),
                          f"{result.content!r} {tool_messages}")
        # The assistant message's own call panel is renderer output; the
        # event panels are the "running …" line and the result panel (which
        # always carries the shell's "exit 0").
        failures = _check(failures, "the upcoming tool event panels are silenced",
                          "running" not in out and "exit 0" not in out,
                          repr(out[:200]))
    return failures


def test_session_store_process_gate() -> int:
    """The flock gates another process's index read-modify-write."""
    failures = 0
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX only
        return 0
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        repo = str(Path(__file__).parent)
        code = (
            "import sys; sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "import ollamarun_lib as orun\n"
            "store = orun.SessionStore(sessions_dir=Path({tmp!r}) / 's',"
            " index_path=Path({tmp!r}) / 'i.json')\n"
            "session = orun.Session.create('m1', 'http://h')\n"
            "store.save(session)\n"
            "print('saved', len(store.entries()))\n"
            "store.delete(session.id, use_trash=False)\n"
            "print('deleted', len(store.entries()))\n"
        ).format(repo=repo, tmp=tmp)
        guard = open(Path(tmp) / "i.json.lock", "a+")
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        proc = subprocess.Popen([sys.executable, "-c", code],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(0.7)
            blocked = proc.poll() is None
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
            guard.close()
        out, err = proc.communicate(timeout=20)
        failures = _check(failures, "the other process waits for the index lock",
                          blocked and proc.returncode == 0
                          and b"saved 1" in out and b"deleted 0" in out,
                          f"blocked={blocked} rc={proc.returncode} "
                          f"out={out[-120:]!r} err={err[-200:]!r}")
    return failures


def test_session_store_same_session_writers() -> int:
    """Two processes saving one session id cannot corrupt its file or index."""
    failures = 0
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        repo = str(Path(__file__).parent)
        template = (
            "import sys; sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "import ollamarun_lib as orun\n"
            "store = orun.SessionStore(sessions_dir=Path({tmp!r}) / 's',"
            " index_path=Path({tmp!r}) / 'i.json')\n"
            "for i in range(10):\n"
            "    session = orun.Session.create('m1', 'http://h', title='{tag}-%d' % i)\n"
            "    session.id = 'shared-session'\n"
            "    store.save(session)\n"
            "print('done')\n"
        )
        procs = [subprocess.Popen(
                     [sys.executable, "-c",
                      template.format(repo=repo, tmp=tmp, tag=tag)],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for tag in ("a", "b")]
        results = [(proc.communicate(), proc.returncode) for proc in procs]
        failures = _check(failures, "both same-session writers save cleanly",
                          all(code == 0 for _out, code in results),
                          str([(out[1].decode()[-160:], code)
                               for out, code in results]))
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        try:
            loaded = store.load("shared-session")
            parsed = loaded.id == "shared-session" and loaded.title.startswith(("a-", "b-"))
        except (OSError, ValueError, TypeError):
            parsed = False
        entries = store.entries()
        leftovers = sorted(str(path) for path in Path(tmp).rglob("*.tmp"))
        failures = _check(failures, "the session file stays valid JSON",
                          parsed, str(parsed))
        failures = _check(failures, "the index keeps exactly one entry",
                          len(entries) == 1
                          and entries[0].get("id") == "shared-session",
                          f"{len(entries)} {entries[:2]}")
        failures = _check(failures, "no temp files are left behind",
                          not leftovers, str(leftovers))
    return failures


def test_session_store_without_fcntl() -> int:
    """Without fcntl the store falls back to the in-process lock and still saves."""
    failures = 0
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        repo = str(Path(__file__).parent)
        code = (
            "import sys; sys.path.insert(0, {repo!r})\n"
            "from pathlib import Path\n"
            "import ollamarun_lib.session as session_module\n"
            "import ollamarun_lib as orun\n"
            "session_module.fcntl = None\n"
            "store = orun.SessionStore(sessions_dir=Path({tmp!r}) / 's',"
            " index_path=Path({tmp!r}) / 'i.json')\n"
            "store.save(orun.Session.create('m1', 'http://h'))\n"
            "print('entries', len(store.entries()))\n"
        ).format(repo=repo, tmp=tmp)
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=30)
        failures = _check(failures, "save works without fcntl (no-op fallback)",
                          proc.returncode == 0 and "entries 1" in proc.stdout,
                          f"rc={proc.returncode} out={proc.stdout[-120:]!r} "
                          f"err={proc.stderr[-200:]!r}")
    return failures


def test_artifact_store() -> int:
    """One opaque, indexed snapshot per artifact: bytes, hash, naming, jail."""
    failures = 0

    def _raises(call):
        try:
            call()
            return None
        except Exception as exc:  # the tests want the exception type
            return exc

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "artifacts"
        store = orun.ArtifactStore(root)

        ref = store.register(b"hello\n", filename="/etc/passwd",
                             media_type="Text/Plain", source="user")
        failures = _check(failures, "register returns the ref shape",
                          orun.ARTIFACT_ID_RE.fullmatch(ref.id) is not None
                          and ref.media_type == "text/plain"
                          and ref.filename == "passwd"
                          and ref.size == 6
                          and ref.sha256 == hashlib.sha256(b"hello\n").hexdigest()
                          and ref.source == "user" and ref.turn_id is None
                          and ref.session_id is None
                          and ref.url == f"/v1/artifacts/{ref.id}",
                          repr(ref))
        failures = _check(failures, "resolve returns the same record",
                          store.resolve(ref.id) == ref, repr(store.resolve(ref.id)))
        failures = _check(failures, "read returns the stored bytes",
                          store.read(ref.id) == (ref, b"hello\n"),
                          repr(store.read(ref.id)))
        failures = _check(failures, "the ref dict keeps the fixed key order",
                          list(ref.to_dict()) == [
                              "id", "media_type", "filename", "size", "sha256",
                              "source", "turn_id", "session_id", "created_at",
                              "url"],
                          str(list(ref.to_dict())))
        failures = _check(failures, "the stored object lives under uploads/",
                          (root / "uploads" / ref.id).is_file(),
                          str(sorted(p.name for p in root.rglob("*"))))
        manifest = json.loads((root / "meta" / f"{ref.id}.json").read_text())
        failures = _check(failures, "the manifest records the storage path",
                          manifest.get("path") == f"uploads/{ref.id}",
                          str(manifest.get("path")))
        index = json.loads((root / "index.json").read_text())
        entry = index["artifacts"][ref.id]
        failures = _check(failures, "the index entry is committed",
                          entry.get("pending") is False and entry.get("size") == 6,
                          str(entry))

        for bad in ("../../etc/passwd", "art_" + "0" * 31,
                    "art_" + "g" * 32, "art_" + "0" * 32 + "/../x", ""):
            failures = _check(failures,
                              f"a malformed id never resolves ({bad[:24]!r})",
                              store.resolve(bad) is None and store.read(bad) is None,
                              "resolved")

        path = Path(tmp) / "report.txt"
        path.write_text("first", encoding="utf-8")
        publisher = store.for_turn(turn_id="turn_1", session_id="ses_1")
        ref2 = publisher.publish_path(path)
        path.write_text("second", encoding="utf-8")
        failures = _check(failures, "a published artifact is an immutable snapshot",
                          store.read(ref2.id) == (ref2, b"first"),
                          repr(store.read(ref2.id)))
        failures = _check(failures, "a turn artifact carries its provenance",
                          ref2.source == "tool" and ref2.turn_id == "turn_1"
                          and ref2.session_id == "ses_1"
                          and ref2.media_type == "text/plain",
                          repr(ref2))
        failures = _check(failures, "a turn artifact is stored under the session/turn",
                          (root / "sessions" / "ses_1" / "turn_1"
                           / ref2.id).is_file(),
                          str(sorted(p.name for p in (root / "sessions").rglob("*"))))

        failures = _check(failures, "a path-shaped filename is reduced to its base",
                          store.register(b"x", filename="a/b/c.txt",
                                         media_type="text/plain").filename == "c.txt"
                          and store.register(b"x", filename="..\\evil",
                                             media_type="text/plain").filename == "evil"
                          and store.register(b"x", filename="\x01\x02",
                                             media_type="text/plain").filename == "file"
                          and store.register(b"x", filename="...",
                                             media_type="text/plain").filename == "file",
                          "sanitize")

        error = _raises(lambda: store.register(b"x", filename="f",
                                              media_type="not-a-type"))
        failures = _check(failures, "a bad media type is a 400",
                          isinstance(error, orun.ApiError) and error.status == 400,
                          repr(error))
        error = _raises(lambda: store.register(b"x", filename="f",
                                               media_type="text/plain",
                                               source="nope"))
        failures = _check(failures, "an unknown source is a 400",
                          isinstance(error, orun.ApiError) and error.status == 400,
                          repr(error))
        error = _raises(lambda: orun.ArtifactStore(
            Path(tmp) / "small", max_bytes=2).register(
                b"abc", filename="f", media_type="text/plain"))
        failures = _check(failures, "an oversized artifact is a 413",
                          isinstance(error, orun.ApiError)
                          and error.status == 413
                          and error.code == "artifact_too_large",
                          repr(error))
        error = _raises(lambda: publisher.publish_path(root))  # a directory
        failures = _check(failures, "publishing a directory is refused",
                          isinstance(error, orun.ApiError) and error.status == 400,
                          repr(error))

        blocked_root = Path(tmp) / "blocked"
        blocked_root.mkdir()
        (blocked_root / "uploads").write_text("not a dir", encoding="utf-8")
        blocked = orun.ArtifactStore(blocked_root)
        error = _raises(lambda: blocked.register(b"data", filename="f.txt",
                                                 media_type="text/plain"))
        index_after = (json.loads((blocked_root / "index.json").read_text())
                       if (blocked_root / "index.json").exists()
                       else {"artifacts": {}})
        failures = _check(failures,
                          "a failed write is reported and leaves no reservation",
                          error is not None and index_after.get("artifacts") == {},
                          f"{error!r} {index_after}")
    return failures


def test_artifact_quota_and_ttl() -> int:
    """Quota reservations hold under concurrency; TTL and reconciliation sweep."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "artifacts"
        store = orun.ArtifactStore(root, max_bytes=1024, quota_bytes=10)
        store.register(b"123456", filename="a.txt", media_type="text/plain")
        try:
            store.register(b"12345", filename="b.txt", media_type="text/plain")
            error = None
        except orun.ApiError as exc:
            error = exc
        failures = _check(failures, "a registration over the quota is a 507",
                          error is not None and error.status == 507
                          and error.code == "artifact_quota_exceeded",
                          repr(error))

        concurrent = orun.ArtifactStore(Path(tmp) / "conc", max_bytes=1024,
                                        quota_bytes=10)
        barrier = threading.Barrier(4)
        results: list = []

        def worker() -> None:
            barrier.wait(timeout=10)
            try:
                concurrent.register(b"1234", filename="w.txt",
                                    media_type="text/plain")
                results.append("ok")
            except orun.ApiError as exc:
                results.append(str(exc.code))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        failures = _check(failures,
                          "concurrent uploads cannot overshoot the quota",
                          results.count("ok") == 2
                          and results.count("artifact_quota_exceeded") == 2,
                          str(sorted(results)))

        now = [1000.0]
        ttl_store = orun.ArtifactStore(Path(tmp) / "ttl", ttl=50,
                                       clock=lambda: now[0], sweep_interval=0)
        old = ttl_store.register(b"old", filename="old.txt",
                                 media_type="text/plain")
        now[0] = 1030.0
        fresh = ttl_store.register(b"new", filename="new.txt",
                                   media_type="text/plain")
        now[0] = 1060.0
        removed = ttl_store.sweep()
        ttl_root = Path(tmp) / "ttl"
        failures = _check(failures, "the TTL sweep drops only expired artifacts",
                          removed == 1
                          and ttl_store.resolve(old.id) is None
                          and ttl_store.resolve(fresh.id) is not None
                          and not (ttl_root / "uploads" / old.id).exists()
                          and not (ttl_root / "meta" / f"{old.id}.json").exists(),
                          f"removed={removed}")

        rec_root = Path(tmp) / "recon"
        rec = orun.ArtifactStore(rec_root)
        ref = rec.register(b"payload", filename="r.txt", media_type="text/plain")
        index_path = rec_root / "index.json"
        data = json.loads(index_path.read_text())
        data["artifacts"][ref.id]["pending"] = True
        index_path.write_text(json.dumps(data), encoding="utf-8")
        second = orun.ArtifactStore(rec_root)
        second.sweep()
        refreshed = json.loads(index_path.read_text())["artifacts"][ref.id]
        failures = _check(failures, "a pending entry with a manifest is adopted",
                          refreshed.get("pending") is False
                          and second.resolve(ref.id) is not None,
                          str(refreshed))

        orphan = rec.register(b"orphan", filename="o.txt", media_type="text/plain")
        data = json.loads(index_path.read_text())
        data["artifacts"].pop(orphan.id, None)
        index_path.write_text(json.dumps(data), encoding="utf-8")
        rec.sweep()
        entry = json.loads(index_path.read_text())["artifacts"].get(orphan.id)
        failures = _check(failures, "an orphan manifest is re-adopted into the index",
                          isinstance(entry, dict) and entry.get("pending") is False,
                          str(entry))

        ghost = "art_" + "a" * 32
        data = json.loads(index_path.read_text())
        data["artifacts"][ghost] = {"path": f"uploads/{ghost}", "size": 3,
                                    "pending": True, "created_at": 0.0}
        index_path.write_text(json.dumps(data), encoding="utf-8")
        rec.sweep()
        failures = _check(failures, "a stale reservation without a manifest is reaped",
                          ghost not in json.loads(index_path.read_text())["artifacts"],
                          "still indexed")

        debris = rec_root / "uploads" / "art_dead.tmp"
        debris.write_text("x", encoding="utf-8")
        old_stamp = time.time() - 7200
        os.utime(debris, (old_stamp, old_stamp))
        rec.sweep()
        failures = _check(failures, "old temp debris is removed",
                          not debris.exists(), "still there")
    return failures


def test_tool_artifact_registration() -> int:
    """File tools publish their destination; publish_artifact is gated and jailed."""
    failures = 0

    def _spec(name: str) -> orun.ToolSpec:
        return orun.ToolSpec(name=name, description="", parameters={},
                             handler=orun.HANDLERS[name])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = orun.ArtifactStore(tmp / "artifacts")
        publisher = store.for_turn(turn_id="turn_9", session_id="ses_9")
        ctx = orun.ToolContext(cwd=str(tmp), artifacts=publisher)

        target = tmp / "out.txt"
        outcome = orun.execute_tool(_spec("write_file"),
                                    {"path": str(target), "content": "hi"}, ctx)
        refs = outcome.artifacts
        failures = _check(failures, "write_file publishes the file it wrote",
                          outcome.ok and len(refs) == 1
                          and refs[0]["filename"] == "out.txt"
                          and refs[0]["source"] == "tool"
                          and refs[0]["turn_id"] == "turn_9"
                          and refs[0]["session_id"] == "ses_9"
                          and store.read(refs[0]["id"])[1] == b"hi",
                          f"{outcome.ok} {refs}")

        target.write_text("changed on disk", encoding="utf-8")
        failures = _check(failures, "the published snapshot is immutable",
                          store.read(refs[0]["id"])[1] == b"hi",
                          repr(store.read(refs[0]["id"])))

        edited = orun.execute_tool(
            _spec("edit_file"),
            {"path": str(target), "old_string": "changed", "new_string": "edited"}, ctx)
        failures = _check(failures, "edit_file publishes the edited file",
                          edited.ok and len(edited.artifacts) == 1
                          and store.read(edited.artifacts[0]["id"])[1]
                          == b"edited on disk",
                          str(edited.artifacts))

        copied = tmp / "copy.txt"
        copy_outcome = orun.execute_tool(
            _spec("copy_file"),
            {"source": str(target), "destination": str(copied)}, ctx)
        failures = _check(failures, "copy_file publishes the destination",
                          copy_outcome.ok and len(copy_outcome.artifacts) == 1
                          and copy_outcome.artifacts[0]["filename"] == "copy.txt",
                          str(copy_outcome.artifacts))

        moved = tmp / "moved.txt"
        move_outcome = orun.execute_tool(
            _spec("move_file"),
            {"source": str(copied), "destination": str(moved)}, ctx)
        failures = _check(failures, "move_file publishes the destination",
                          move_outcome.ok and len(move_outcome.artifacts) == 1
                          and move_outcome.artifacts[0]["filename"] == "moved.txt"
                          and not copied.exists(),
                          str(move_outcome.artifacts))

        folder = tmp / "folder"
        folder.mkdir()
        (folder / "inner.txt").write_text("x", encoding="utf-8")
        dir_outcome = orun.execute_tool(
            _spec("copy_file"),
            {"source": str(folder), "destination": str(tmp / "folder-copy")}, ctx)
        failures = _check(failures, "a directory copy publishes nothing",
                          dir_outcome.ok and dir_outcome.artifacts == []
                          and "artifact not published" not in dir_outcome.text,
                          str(dir_outcome.artifacts))

        published = tmp / "report.bin"
        published.write_bytes(b"\x00\x01")
        publish_outcome = orun.execute_tool(
            _spec("publish_artifact"), {"path": "report.bin"}, ctx)
        failures = _check(failures, "publish_artifact publishes a relative path",
                          publish_outcome.ok and len(publish_outcome.artifacts) == 1
                          and publish_outcome.artifacts[0]["filename"] == "report.bin",
                          str(publish_outcome))
        outside = Path("/etc/hostname")
        if outside.is_file():
            refused = orun.execute_tool(
                _spec("publish_artifact"), {"path": str(outside)}, ctx)
            failures = _check(failures,
                              "publish_artifact refuses a path outside the cwd",
                              not refused.ok and refused.artifacts == []
                              and "outside the working directory" in refused.text,
                              str(refused))
        refused_dir = orun.execute_tool(
            _spec("publish_artifact"), {"path": "folder"}, ctx)
        failures = _check(failures, "publish_artifact refuses a directory",
                          not refused_dir.ok and refused_dir.artifacts == [],
                          str(refused_dir))
        no_store = orun.execute_tool(
            _spec("publish_artifact"), {"path": "report.bin"},
            orun.ToolContext(cwd=str(tmp)))
        failures = _check(failures, "publish_artifact needs the serve store",
                          not no_store.ok and no_store.artifacts == []
                          and "artifact store" in no_store.text,
                          str(no_store))
        failures = _check(failures, "publish_artifact is not auto-approved",
                          not orun.approval_decision(
                              _spec("publish_artifact"), {"path": "x"},
                              policy="allowlist").allowed,
                          "auto-approved")

        tiny = orun.ArtifactStore(tmp / "tiny", max_bytes=2)
        tiny_ctx = orun.ToolContext(
            cwd=str(tmp),
            artifacts=tiny.for_turn(turn_id="turn_t", session_id="ses_t"))
        note = orun.execute_tool(_spec("write_file"),
                                 {"path": str(tmp / "big.txt"), "content": "hello"},
                                 tiny_ctx)
        failures = _check(failures,
                          "a failed publication is a note, not a failed tool",
                          note.ok and note.artifacts == []
                          and "artifact not published" in note.text
                          and (tmp / "big.txt").is_file(),
                          str(note))
        refused_size = orun.execute_tool(
            _spec("publish_artifact"), {"path": "big.txt"}, tiny_ctx)
        failures = _check(failures, "publish_artifact refuses over the store limit",
                          not refused_size.ok
                          and "byte artifact limit" in refused_size.text,
                          str(refused_size))

        without = orun.default_registry(tts=False, search=False, chem=False)
        with_store = orun.default_registry(tts=False, search=False, chem=False,
                                           artifacts=True)
        failures = _check(failures, "publish_artifact appears only with a store",
                          "publish_artifact" not in without
                          and "publish_artifact" in with_store,
                          str(sorted(set(with_store) - set(without))))
    return failures


def test_artifact_events() -> int:
    """artifact.created rides the turn stream and reaches the TurnResult."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = orun.SessionStore(sessions_dir=root / "s",
                                  index_path=root / "i.json")
        artifacts = orun.ArtifactStore(root / "artifacts")
        args = orun.build_parser().parse_args(
            ["m1", "--render", "plain", "--no-save", "-y"])
        out = root / "report.txt"
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[
                                 [{"message": {"content": "", "tool_calls": [
                                     {"function": {"name": "write_file",
                                                   "arguments": {"path": str(out),
                                                                 "content": "payload"}}}]}},
                                  {"done": True}],
                                 [{"message": {"content": "done"}, "done": True}],
                             ])
        session = orun.Session.create("m1", client.base)
        publisher = artifacts.for_turn(turn_id="turn_test",
                                       session_id=session.id)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session,
                         artifacts=publisher)
        repl._build_prompt_session = lambda: None
        repl.headless = True
        manager = orun.TurnManager()

        def work(sink, cancel):
            hooks = repl._turn_hooks()
            hooks.on_event = sink
            hooks.cancel = cancel
            # The API seam: the runtime's own tool execution carries this
            # sink, the approver and the tool context into every round.
            hooks.execute_tools = None
            renderer = orun.HeadlessRenderer(on_event=sink)

            def present(payload, after_tools, cancel_token):
                return renderer.stream(client.chat(payload, cancel=cancel_token),
                                       cancel=cancel_token)

            hooks.present_round = present
            runtime = orun.AgentRuntime(hooks)
            result = runtime.run()
            return runtime.to_turn_result(result, session_id=session.id,
                                          turn_id="turn_test")

        handle = manager.submit(session_id=session.id, work=work)
        snapshot = manager.wait(handle.id, timeout=5)
        events = _collect(manager.subscribe(handle.id, after=0))
        types = [event.type for event in events]
        created = [event for event in events if event.type == orun.ARTIFACT_CREATED]
        results = [event for event in events if event.type == orun.TOOL_RESULT]
        failures = _check(failures, "artifact.created carries the full ref",
                          len(created) == 1
                          and created[0].data.get("source") == "tool"
                          and created[0].data.get("filename") == "report.txt"
                          and created[0].data.get("session_id") == session.id
                          and created[0].data.get("turn_id") == "turn_test"
                          and created[0].data.get("url", "").startswith("/v1/artifacts/"),
                          str(created[:1]))
        failures = _check(failures,
                          "artifact.created sits between the decision and the result",
                          orun.APPROVAL_RESOLVED in types
                          and types.index(orun.APPROVAL_RESOLVED)
                          < types.index(orun.ARTIFACT_CREATED)
                          < types.index(orun.TOOL_RESULT),
                          str(types))
        failures = _check(failures, "tool.result references the artifact ids",
                          bool(results)
                          and results[0].data.get("artifacts")
                          == [created[0].data["id"]],
                          str(results[:1]))
        failures = _check(failures, "the TurnResult carries the refs",
                          snapshot is not None and snapshot.result is not None
                          and [ref.get("id") for ref in snapshot.result.artifacts]
                          == [created[0].data["id"]],
                          str(snapshot.result.artifacts
                              if snapshot and snapshot.result else None))
        failures = _check(failures, "the session records the artifact",
                          [ref.get("id") for ref in session.artifacts]
                          == [created[0].data["id"]]
                          and any(message.get("artifacts")
                                  for message in session.messages
                                  if message.get("role") == "tool"),
                          str(session.artifacts))
        stored = artifacts.read(created[0].data["id"])
        failures = _check(failures, "the bytes are in the store",
                          stored is not None and stored[1] == b"payload",
                          repr(stored))
    return failures


def test_artifact_session_persistence() -> int:
    """Sessions accumulate refs without leaking them to the model."""
    failures = 0
    old = {"id": "s1", "model": "m1", "host": "h",
           "messages": [{"role": "user", "content": "hi", "ts": "t"}]}
    loaded = orun.Session.from_dict(old)
    failures = _check(failures, "old session files load without artifacts",
                      loaded.artifacts == [] and loaded.id == "s1",
                      repr(loaded.artifacts))
    ref = {"id": "art_" + "1" * 32, "media_type": "text/plain",
           "filename": "a.txt", "size": 1, "sha256": "x", "source": "tool",
           "turn_id": "t1", "session_id": "s1", "created_at": "now",
           "url": "/v1/artifacts/" + "art_" + "1" * 32}
    loaded.add_artifacts([ref])
    loaded.add_artifacts([dict(ref)])  # same id again: deduped
    failures = _check(failures, "artifact refs are deduped by id",
                      [item["id"] for item in loaded.artifacts] == [ref["id"]],
                      str(loaded.artifacts))
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        store.save(loaded)
        round_trip = store.load("s1")
        failures = _check(failures, "artifact refs survive the session round trip",
                          round_trip.artifacts == [ref],
                          str(round_trip.artifacts))
        payloads = round_trip.message_payloads(with_tools=True)
        failures = _check(failures, "artifact refs never reach the model payload",
                          all("artifacts" not in payload for payload in payloads),
                          str(payloads))
    return failures


def test_artifact_content_parts() -> int:
    """One resolver: file refs, text inlining, binary notes, vision parity."""
    failures = 0

    def _raises(call):
        try:
            call()
            return None
        except Exception as exc:
            return exc

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        store = orun.ArtifactStore(tmp / "artifacts")
        text_ref = store.register(b"# Title\nhello", filename="notes.md",
                                  media_type="text/markdown", source="user")
        json_ref = store.register(b'{"a": 1}', filename="data.json",
                                  media_type="application/json", source="user")
        bin_ref = store.register(b"PK\x03\x04not-a-pdf", filename="report.zip",
                                 media_type="application/zip", source="user")
        pdf_ref = store.register(_MINIMAL_PDF, filename="report.pdf",
                                 media_type="application/pdf", source="user")
        scan_ref = store.register(_BLANK_PDF, filename="scan.pdf",
                                  media_type="application/pdf", source="user")
        img_bytes = b"\x89PNG\r\n\x1a\nimage-bytes"
        img_ref = store.register(img_bytes, filename="pic.png",
                                 media_type="image/png", source="user")
        big_ref = store.register(b"x" * (orun.INLINE_TEXT_BYTES + 100),
                                 filename="big.txt", media_type="text/plain",
                                 source="user")

        resolved = orun.resolve_content_parts(
            [{"type": "text", "text": "look"},
             {"type": "file", "file_id": text_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "a text file part inlines as a <file> block",
                          resolved.text.startswith("look")
                          and '<file name="notes.md">' in resolved.text
                          and "# Title\nhello" in resolved.text
                          and resolved.images == []
                          and [ref.id for ref in resolved.artifacts]
                          == [text_ref.id],
                          resolved.text[:200])
        resolved_json = orun.resolve_content_parts(
            [{"type": "file", "file_id": json_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "application/json counts as text",
                          '{"a": 1}' in resolved_json.text,
                          resolved_json.text[:200])
        resolved_bin = orun.resolve_content_parts(
            [{"type": "file", "file_id": bin_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "a binary part is noted, never faked",
                          "contents were not extracted" in resolved_bin.text
                          and "report.zip" in resolved_bin.text,
                          resolved_bin.text[:200])
        resolved_pdf = orun.resolve_content_parts(
            [{"type": "file", "file_id": pdf_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "a PDF file part inlines its text layer",
                          '<file name="report.pdf">' in resolved_pdf.text
                          and "Hello PDF world" in resolved_pdf.text
                          and resolved_pdf.images == [],
                          resolved_pdf.text[:200])
        resolved_scan = orun.resolve_content_parts(
            [{"type": "file", "file_id": scan_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "a scanned PDF part is noted, never faked",
                          "no text layer could be extracted" in resolved_scan.text
                          and "scan.pdf" in resolved_scan.text
                          and resolved_scan.images == [],
                          resolved_scan.text[:200])
        resolved_img = orun.resolve_content_parts(
            [{"type": "file", "file_id": img_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "an image file part becomes an image payload",
                          len(resolved_img.images) == 1
                          and resolved_img.images[0].b64
                          == base64.b64encode(img_bytes).decode()
                          and resolved_img.images[0].artifact_id == img_ref.id,
                          str(resolved_img.images))
        resolved_big = orun.resolve_content_parts(
            [{"type": "file", "file_id": big_ref.id}],
            where="messages[0].content", store=store)
        failures = _check(failures, "an oversized text part is truncated, not dropped",
                          "truncated" in resolved_big.text
                          and resolved_big.text.count("x") >= orun.INLINE_TEXT_BYTES,
                          resolved_big.text[-120:])

        missing = _raises(lambda: orun.resolve_content_parts(
            [{"type": "file", "file_id": "art_" + "0" * 32}],
            where="messages[0].content", store=store))
        failures = _check(failures, "a missing artifact id is a structured 404",
                          isinstance(missing, orun.ApiError)
                          and missing.status == 404
                          and missing.code == "artifact_not_found"
                          and missing.param == "messages[0].content[0].file_id",
                          repr(missing))
        unknown = _raises(lambda: orun.resolve_content_parts(
            [{"type": "video", "url": "x"}],
            where="messages[0].content", store=store))
        failures = _check(failures, "an unknown part type is a 400",
                          isinstance(unknown, orun.ApiError)
                          and unknown.status == 400
                          and unknown.code == "unsupported_value",
                          repr(unknown))

        plan = orun.parse_chat_request({
            "model": "m1",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "check"},
                {"type": "file", "file_id": text_ref.id},
                {"type": "image_url", "image_url": {"url":
                    "data:image/png;base64,"
                    + base64.b64encode(b"png").decode()}}]}]}, store=store)
        failures = _check(failures, "the OpenAI surface resolves file parts",
                          plan.has_images
                          and '<file name="notes.md">' in plan.messages[0]["content"]
                          and plan.messages[0]["images"]
                          == [base64.b64encode(b"png").decode()],
                          str(plan.messages)[:200])

        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools", "vision"],
                              chat_script=[[{"message": {"content": "ok"},
                                             "done": True}]])
        server, api, sessions, thread = _start_agent_api(tmp, client,
                                                         artifacts=store)
        try:
            body = {"model": "m1", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "see"},
                {"type": "file", "file_id": text_ref.id}]}]}
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=body)
            turn = json.loads(raw)
            _poll_turn(server, turn["id"])
            session = sessions.load(turn["session_id"])
            user = session.messages[0]
            failures = _check(failures, "the native surface resolves file parts",
                              status == 202
                              and '<file name="notes.md">' in user["content"]
                              and user.get("artifacts", [{}])[0].get("id")
                              == text_ref.id
                              and session.artifacts
                              and session.artifacts[0]["id"] == text_ref.id,
                              str(user))
        finally:
            _stop_api(server, thread)

        blind = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[[{"message": {"content": "ok"},
                                            "done": True}]])
        server, api, sessions, thread = _start_agent_api(tmp, blind,
                                                         artifacts=store)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body={"model": "m1", "messages": [
                                              {"role": "user", "content": [
                                                  {"type": "file",
                                                   "file_id": img_ref.id}]}]})
            error = json.loads(raw).get("error", {})
            failures = _check(failures,
                              "the native surface refuses images without vision",
                              status == 400
                              and error.get("code")
                              == "model_does_not_support_images",
                              f"{status} {error}")
        finally:
            _stop_api(server, thread)
    return failures


def test_artifact_remote_fetch() -> int:
    """The shared fetch enforces scheme, redirect, size and timeout rules."""
    failures = 0
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the test output clean
            pass

        def _send(self, status, body=b"", content_type=None, location=None):
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if body:
                    self.wfile.write(body)
            except BrokenPipeError:
                # The client timed out on purpose (the /slow case); the
                # stub's write failing after that is expected.
                pass

        def do_GET(self):
            if self.path == "/small.png":
                self._send(200, b"\x89PNG\r\n\x1a\nbody", "image/png")
            elif self.path == "/redirect":
                self._send(302, location="/small.png")
            elif self.path == "/loop":
                self._send(302, location="/loop")
            elif self.path == "/noctype":
                self._send(200, b"GIF89a raw")
            elif self.path == "/slow":
                time.sleep(0.6)
                self._send(200, b"late", "text/plain")
            else:
                self._send(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        data, media = orun.fetch_remote_media(f"{base}/small.png", max_bytes=1024)
        failures = _check(failures, "a remote image comes back with its type",
                          data == b"\x89PNG\r\n\x1a\nbody"
                          and media == "image/png", f"{data!r} {media}")
        data, media = orun.fetch_remote_media(f"{base}/redirect", max_bytes=1024)
        failures = _check(failures, "a bounded redirect is followed",
                          data.startswith(b"\x89PNG") and media == "image/png",
                          f"{data!r} {media}")
        _data, media = orun.fetch_remote_media(f"{base}/noctype", max_bytes=1024)
        failures = _check(failures, "a missing content type is sniffed",
                          media == "image/gif", media)

        for name, call, code in (
            ("a redirect loop fails", lambda: orun.fetch_remote_media(
                f"{base}/loop", max_bytes=1024), "remote_fetch_failed"),
            ("an HTTP error fails", lambda: orun.fetch_remote_media(
                f"{base}/missing", max_bytes=1024), "remote_fetch_failed"),
            ("an unreachable host fails", lambda: orun.fetch_remote_media(
                "http://127.0.0.1:1/x", max_bytes=1024, timeout=0.5),
             "remote_fetch_failed"),
            ("a stalled fetch times out", lambda: orun.fetch_remote_media(
                f"{base}/slow", max_bytes=1024, timeout=0.2),
             "remote_fetch_failed"),
            ("an oversize body is a 413", lambda: orun.fetch_remote_media(
                f"{base}/small.png", max_bytes=3), "image_too_large"),
            ("a non-http scheme is refused", lambda: orun.fetch_remote_media(
                "ftp://example/x", max_bytes=1024), "unsupported_value"),
        ):
            try:
                call()
                error = None
            except orun.ApiError as exc:
                error = exc
            failures = _check(failures, name,
                              error is not None and error.status in (400, 413)
                              and error.code == code, repr(error))

        # The OpenAI surface resolves http(s) image URLs through the same
        # bounded fetch (a data URI still works; the mapper test covers the
        # scheme rejection).
        vision = _ServeClient(models=["m1"],
                              capabilities=["completion", "vision"])
        api_server, _vision, api_thread = _start_api(vision)
        try:
            status, _headers, raw = _http(
                api_server, "POST", "/v1/chat/completions", body={
                    "model": "m1",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url",
                         "image_url": {"url": f"{base}/small.png"}}]}]})
            sent = vision.chat_once_payloads[-1]
            images = sent["messages"][-1].get("images") or []
            failures = _check(
                failures, "the OpenAI surface fetches http(s) image URLs",
                status == 200 and images
                and images[0] == base64.b64encode(
                    b"\x89PNG\r\n\x1a\nbody").decode(),
                f"{status} {raw[:160]} {images[:1]}")
        finally:
            _stop_api(api_server, api_thread)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return failures


def test_artifact_upload_api() -> int:
    """POST /v1/files validates strictly and returns the full ref."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        artifact_store = orun.ArtifactStore(Path(tmp) / "artifacts", max_bytes=64)
        client = _ServeClient(models=["m1"])
        server, api, sessions, thread = _start_agent_api(
            tmp, client, artifacts=artifact_store)
        try:
            data = base64.b64encode(b"hello upload").decode()
            status, _headers, raw = _http(server, "POST", "/v1/files", body={
                "filename": "notes/../notes.txt", "media_type": "text/plain",
                "data_base64": data})
            ref = json.loads(raw)
            failures = _check(failures, "an upload returns the full ref",
                              status == 201 and ref.get("source") == "user"
                              and ref.get("filename") == "notes.txt"
                              and ref.get("size") == 12
                              and ref.get("sha256")
                              == hashlib.sha256(b"hello upload").hexdigest()
                              and ref.get("url")
                              == f"/v1/artifacts/{ref.get('id')}",
                              f"{status} {ref}")
            over = base64.b64encode(b"x" * 100).decode()
            for name, body, code in (
                ("an unknown field",
                 {"filename": "a", "media_type": "text/plain",
                  "data_base64": data, "x": 1}, "unknown_field"),
                ("a missing filename",
                 {"media_type": "text/plain", "data_base64": data}, None),
                ("a bad media type",
                 {"filename": "a", "media_type": "nope",
                  "data_base64": data}, "unsupported_value"),
                ("bad base64",
                 {"filename": "a", "media_type": "text/plain",
                  "data_base64": "!!!"}, None),
                ("oversize before decode",
                 {"filename": "a", "media_type": "text/plain",
                  "data_base64": over}, "artifact_too_large"),
            ):
                status, _headers, raw = _http(server, "POST", "/v1/files",
                                              body=body)
                error = json.loads(raw).get("error", {})
                failures = _check(failures, f"the upload API rejects {name}",
                                  status in (400, 413)
                                  and (code is None or error.get("code") == code),
                                  f"{status} {error}")
        finally:
            _stop_api(server, thread)

        tiny = orun.ArtifactStore(Path(tmp) / "quota", quota_bytes=5)
        server, api, sessions, thread = _start_agent_api(tmp, client,
                                                         artifacts=tiny)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/files", body={
                "filename": "a.txt", "media_type": "text/plain",
                "data_base64": base64.b64encode(b"123456").decode()})
            error = json.loads(raw).get("error", {})
            failures = _check(failures, "an upload over the quota is a 507",
                              status == 507
                              and error.get("code") == "artifact_quota_exceeded",
                              f"{status} {error}")
        finally:
            _stop_api(server, thread)
    return failures


def test_artifact_fetch_api() -> int:
    """Downloads serve the stored bytes with the recorded type; missing 404s."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        artifact_store = orun.ArtifactStore(Path(tmp) / "artifacts")
        payload = b"\x89PNG\r\n\x1a\nfetch-me"
        ref = artifact_store.register(payload, filename="pic.png",
                                      media_type="image/png", source="user")
        client = _ServeClient(models=["m1"])
        server, api, sessions, thread = _start_agent_api(
            tmp, client, artifacts=artifact_store)
        try:
            status, headers, raw = _http(server, "GET",
                                         f"/v1/artifacts/{ref.id}")
            failures = _check(failures, "a download carries the bytes and type",
                              status == 200 and raw == payload
                              and headers.get("Content-Type") == "image/png"
                              and 'filename="pic.png"'
                              in headers.get("Content-Disposition", ""),
                              f"{status} {headers}")
            status, headers, raw = _http(server, "GET",
                                         f"/v1/artifacts/{ref.id}/meta")
            meta = json.loads(raw)
            failures = _check(failures, "the meta route returns the full ref",
                              status == 200 and meta == ref.to_dict(),
                              f"{status} {meta}")
            for path, code in (
                (f"/v1/artifacts/art_{'0' * 32}", "artifact_not_found"),
                (f"/v1/artifacts/art_{'0' * 32}/meta", "artifact_not_found"),
                ("/v1/artifacts/../../etc/passwd", "not_found"),
            ):
                status, _headers, raw = _http(server, "GET", path)
                error = json.loads(raw).get("error", {})
                failures = _check(failures, f"a missing artifact 404s ({path[-28:]})",
                                  status == 404
                                  and error.get("code") == code,
                                  f"{status} {error}")
        finally:
            _stop_api(server, thread)

        server, api, sessions, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/files", body={
                "filename": "a.txt", "media_type": "text/plain",
                "data_base64": base64.b64encode(b"x").decode()})
            error = json.loads(raw).get("error", {})
            failures = _check(failures,
                              "an upload without a store is a structured 503",
                              status == 503
                              and error.get("code") == "artifacts_unavailable",
                              f"{status} {error}")
        finally:
            _stop_api(server, thread)
    return failures


def test_cancellation_destructive_guard() -> int:
    """A stopped turn must not trash anything, even inside the handler."""
    failures = 0
    token = orun.CancelToken()
    token.cancel()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        victim = tmp / "precious.txt"
        victim.write_text("keep me", encoding="utf-8")
        fake_bin = tmp / "bin"
        fake_bin.mkdir()
        shim = fake_bin / "trash-put"
        shim.write_text('#!/bin/sh\ntouch "$0.ran"\nexit 0\n', encoding="utf-8")
        shim.chmod(0o755)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(fake_bin) + os.pathsep + old_path
        try:
            ctx = orun.ToolContext(cwd=str(tmp), timeout=5, cancel=token)
            try:
                orun.HANDLERS["delete_file"]({"path": str(victim)}, ctx)
                raised = None
            except Exception as exc:  # the test wants the exception type
                raised = exc
            failures = _check(failures, "delete_file refuses after a stop",
                              isinstance(raised, orun.CommandCancelled), repr(raised))
            failures = _check(failures, "nothing was trashed and no child ran",
                              victim.exists()
                              and not Path(str(shim) + ".ran").exists(),
                              f"victim={victim.exists()} "
                              f"ran={Path(str(shim) + '.ran').exists()}")
            spec = orun.ToolSpec(name="delete_file", description="", parameters={},
                                 handler=orun.HANDLERS["delete_file"])
            outcome = orun.execute_tool(spec, {"path": str(victim)}, ctx)
            failures = _check(failures, "execute_tool reports the refusal as cancelled",
                              outcome.cancelled and not outcome.ok, str(outcome))
        finally:
            os.environ["PATH"] = old_path
    return failures


def test_run_once_json() -> int:
    """`--json` runs the tool loop through the runtime and prints one response."""
    failures = 0
    from ollamarun_lib.run import _run_once_json

    class _JsonClient(_FakeClient):
        def chat_once(self, payload, timeout=None):
            self.chat_once_payloads.append(payload)
            if len(self.chat_once_payloads) == 1:
                return {"message": {"content": "", "tool_calls": [
                            {"function": {"name": "run_shell",
                                          "arguments": {"command": "echo hi"}}}]},
                        "done": True, "prompt_eval_count": 5, "eval_count": 2}
            return {"message": {"content": "final"}, "done": True,
                    "prompt_eval_count": 9, "eval_count": 1}

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--json", "--yes", "--no-save"])
        client = _JsonClient(models=["m1"], capabilities=["completion", "tools"])
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        session.add_message("user", "hi")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = _run_once_json(args, client, store, session, repl)
        printed = out.getvalue().strip()
        failures = _check(failures, "the --json run exits 0 after the tool loop",
                          code == 0 and len(client.chat_once_payloads) == 2,
                          f"{code} {len(client.chat_once_payloads)}")
        roles = [message.get("role") for message in session.messages]
        failures = _check(failures, "the tool round reaches the session",
                          roles == ["user", "assistant", "tool", "assistant"],
                          str(roles))
        try:
            payload = json.loads(printed)
            ok = payload.get("message", {}).get("content") == "final"
        except ValueError:
            ok, payload = False, printed
        failures = _check(failures, "the final response is printed as JSON", ok,
                          str(payload)[:160])
    return failures


class _ServeClient(_FakeClient):
    """`_FakeClient` plus what the API server needs: abort, slow streams and
    embeddings (per-model capabilities included)."""

    def __init__(self, *args, chat_once_data=None, chunk_count=0,
                 chunk_delay=0.0, capabilities_by_model=None,
                 embed_data=None, embed_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat_once_data = chat_once_data
        self.chunk_count = chunk_count
        self.chunk_delay = chunk_delay
        self.abort_calls = 0
        self.capabilities_by_model = dict(capabilities_by_model or {})
        self.embed_data = embed_data
        self.embed_error = embed_error
        self.embed_payloads = []

    def abort_stream(self):
        self.abort_calls += 1
        return True

    def capabilities(self, model):
        if model in self.capabilities_by_model:
            return list(self.capabilities_by_model[model])
        return super().capabilities(model)

    def embed(self, model, inputs, *, truncate=True, dimensions=None,
              options=None):
        self.embed_payloads.append({"model": model, "input": list(inputs),
                                    "truncate": truncate,
                                    "dimensions": dimensions,
                                    "options": options})
        if self.embed_error:
            raise orun.OllamaError(self.embed_error)
        if self.embed_data is not None:
            return dict(self.embed_data)
        return {"embeddings": [[float(index), 0.5] for index in range(len(inputs))],
                "prompt_eval_count": len(inputs)}

    def chat_once(self, payload, timeout=None):
        self.chat_once_payloads.append(payload)
        if self.chat_once_data is not None:
            return dict(self.chat_once_data)
        return {"message": {"content": self.chat_once_reply}, "done": True,
                "prompt_eval_count": 4, "eval_count": 2, "done_reason": "stop"}

    def chat(self, payload, timeout=None, cancel=None):
        self.chat_payloads.append(payload)
        if self.chunk_count:
            def gen():
                for _ in range(self.chunk_count):
                    if cancel is not None and cancel.cancelled:
                        return
                    yield {"message": {"content": "x"}, "done": False}
                    time.sleep(self.chunk_delay)
                yield {"message": {"content": ""}, "done": True,
                       "prompt_eval_count": 1, "eval_count": self.chunk_count}
            return gen()
        return super().chat(payload, timeout=timeout, cancel=cancel)


def _start_api(client=None, *, factory=None, router_kwargs=None,
               server_kwargs=None):
    """An API server on an ephemeral loopback port with a fake backend."""
    client = client if client is not None else _ServeClient(models=["m1"])
    make_client = factory if factory is not None else (lambda: client)
    router = orun.build_router(make_client, **(router_kwargs or {}))
    server = orun.HttpServer(router, bind="127.0.0.1", port=0, ping_interval=0.2,
                             **(server_kwargs or {}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, client, thread


def _stop_api(server, thread):
    server.shutdown()
    server.close()
    thread.join(timeout=5)


def _http(server, method, path, *, body=None, raw=None, headers=None,
          timeout=10.0):
    conn = http.client.HTTPConnection("127.0.0.1", server.address[1],
                                      timeout=timeout)
    try:
        payload = raw if raw is not None else (
            json.dumps(body) if body is not None else None)
        if payload is not None and headers is None:
            headers = {"Content-Type": "application/json"}
        conn.request(method, path, body=payload, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def _api_error(callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
        return None
    except orun.ApiError as exc:
        return exc


def _sse_events(raw: bytes) -> list:
    """Parse an SSE body into its data payloads (comments are dropped)."""
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def test_openai_mapping() -> int:
    """Validation/mapping and response shaping of the OpenAI surface."""
    failures = 0
    user = [{"role": "user", "content": "x"}]
    plan = orun.parse_chat_request({
        "model": " m1 ",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5, "top_p": 0.9, "max_tokens": 64, "seed": 7,
        "stop": ["x", "y"], "presence_penalty": 0.25,
        "frequency_penalty": -0.5,
        "response_format": {"type": "json_object"},
    })
    failures = _check(failures, "options and format map onto the backend",
                      plan.model == "m1"
                      and plan.options == {"temperature": 0.5, "top_p": 0.9,
                                           "num_predict": 64, "seed": 7,
                                           "stop": ["x", "y"],
                                           "presence_penalty": 0.25,
                                           "frequency_penalty": -0.5}
                      and plan.fmt == "json" and plan.stream is False,
                      f"{plan.model} {plan.options} {plan.fmt}")

    tool = {"type": "function", "function": {"name": "get_weather"}}
    for name, body in (
        ("an unknown field", {"model": "m1", "messages": user, "foo": 1}),
        ("a boolean n", {"model": "m1", "messages": user, "n": True}),
        ("a float n", {"model": "m1", "messages": user, "n": 1.0}),
        ("a zero n", {"model": "m1", "messages": user, "n": 0}),
        ("a negative n", {"model": "m1", "messages": user, "n": -2}),
        ("a string n", {"model": "m1", "messages": user, "n": "2"}),
        ("an oversized n", {"model": "m1", "messages": user, "n": 9}),
        ("logprobs", {"model": "m1", "messages": user, "logprobs": True}),
        ("logit_bias", {"model": "m1", "messages": user,
                        "logit_bias": {"1": 2}}),
        ("audio", {"model": "m1", "messages": user,
                   "audio": {"format": "wav"}}),
        ("a non-text modality", {"model": "m1", "messages": user,
                                 "modalities": ["image"]}),
        ("a non-boolean parallel_tool_calls", {"model": "m1", "messages": user,
                                               "parallel_tool_calls": "yes"}),
        ("an out-of-range temperature", {"model": "m1", "messages": user,
                                         "temperature": 5}),
        ("empty messages", {"model": "m1", "messages": []}),
        ("a malformed message element", {"model": "m1", "messages": [None]}),
        ("a string message element", {"model": "m1", "messages": ["oops"]}),
        ("a numeric message element", {"model": "m1", "messages": [42]}),
        ("tool_choice required", {"model": "m1", "messages": user,
                                  "tools": [tool], "tool_choice": "required"}),
        ("tool_choice without tools", {"model": "m1", "messages": user,
                                       "tool_choice": "auto"}),
        ("required without tools", {"model": "m1", "messages": user,
                                    "tool_choice": "required"}),
        ("string include_usage", {"model": "m1", "messages": user,
                                  "stream_options": {"include_usage": "false"}}),
        ("numeric include_usage", {"model": "m1", "messages": user,
                                   "stream_options": {"include_usage": 1}}),
        ("strict tool schemas", {"model": "m1", "messages": user, "tools": [
            {"type": "function",
             "function": {"name": "f", "strict": True}}]}),
        ("an unknown tool field", {"model": "m1", "messages": user, "tools": [
            {"type": "function", "function": {"name": "f"}, "bogus": 1}]}),
        ("an unknown function field", {"model": "m1", "messages": user,
                                       "tools": [
            {"type": "function",
             "function": {"name": "f", "bogus": 1}}]}),
        ("a malformed json_schema wrapper", {"model": "m1", "messages": user,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "n", "schema": {"type": "object"}, "bogus": 1}}}),
        ("a non-boolean json_schema strict", {"model": "m1", "messages": user,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "n", "strict": "yes", "schema": {"type": "object"}}}}),
        ("an unknown role", {"model": "m1", "messages": [
            {"role": "wizard", "content": "x"}]}),
    ):
        exc = _api_error(orun.parse_chat_request, body)
        failures = _check(failures, f"rejects {name}",
                          exc is not None and exc.status == 400, repr(exc))
    exc = _api_error(orun.parse_chat_request,
                     {"model": "m1", "messages": user, "foo": 1})
    failures = _check(failures, "an unknown field names itself",
                      exc is not None and exc.param == "foo"
                      and exc.code == "unknown_field", repr(exc))
    exc = _api_error(orun.parse_chat_request,
                     {"model": "m1", "system": "auto", "messages": ["oops"]})
    failures = _check(failures, "system + a malformed first message is a 400",
                      exc is not None and exc.status == 400
                      and exc.param == "messages[0]", repr(exc))
    accepted = orun.parse_chat_request({
        "model": "m1", "n": 1, "messages": user,
        "stream_options": {"include_usage": False},
        "parallel_tool_calls": False,
        "tools": [{"type": "function",
                   "function": {"name": "f", "strict": False}}]})
    failures = _check(failures,
                      "the strict forms of n/strict/parallel_tool_calls still pass",
                      accepted.include_usage is False
                      and accepted.n == 1
                      and accepted.parallel_tool_calls is False
                      and bool(accepted.tools), str(accepted))
    multi = orun.parse_chat_request({"model": "m1", "n": 3, "messages": user})
    parallel = orun.parse_chat_request({"model": "m1",
                                        "parallel_tool_calls": True,
                                        "messages": user})
    default = orun.parse_chat_request({"model": "m1", "messages": user})
    failures = _check(failures, "n maps and parallel_tool_calls is three-state",
                      multi.n == 3 and parallel.parallel_tool_calls is True
                      and default.n == 1
                      and default.parallel_tool_calls is None,
                      f"{multi.n} {parallel.parallel_tool_calls} "
                      f"{default.parallel_tool_calls}")

    plain = orun.parse_chat_request({"model": "m1", "messages": user})
    auto = orun.parse_chat_request({"model": "m1", "system": "auto",
                                    "messages": user})
    custom = orun.parse_chat_request({"model": "m1", "system": "be terse",
                                      "messages": user})
    first_wins = orun.parse_chat_request({
        "model": "m1", "system": "ignored",
        "messages": [{"role": "system", "content": "kept"}] + user})
    failures = _check(failures, "system: absent means none",
                      [message["role"] for message in plain.messages] == ["user"],
                      str(plain.messages))
    failures = _check(failures, 'system: "auto" injects the built-in default',
                      auto.messages[0]["role"] == "system"
                      and bool(auto.messages[0]["content"])
                      and auto.messages[1]["role"] == "user",
                      str(auto.messages)[:160])
    failures = _check(failures, "system: text replaces it",
                      custom.messages[0] == {"role": "system",
                                             "content": "be terse"})
    failures = _check(failures, "a client system message wins over the extension",
                      first_wins.messages[0]["content"] == "kept"
                      and len(first_wins.messages) == 2,
                      str(first_wins.messages))

    history = orun.parse_chat_request({
        "model": "m1",
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": '{"city": "SP"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
    })
    roles = [message["role"] for message in history.messages]
    tool_message = [message for message in history.messages
                    if message["role"] == "tool"][0]
    assistant_call = [message for message in history.messages
                      if message.get("tool_calls")][0]
    failures = _check(failures, "tool history round-trips by name",
                      roles == ["user", "assistant", "tool", "assistant"]
                      and tool_message["tool_name"] == "get_weather"
                      and assistant_call["tool_calls"][0]["function"]["arguments"]
                      == {"city": "SP"},
                      str(history.messages)[:240])
    bad_ref = _api_error(orun.parse_chat_request, {
        "model": "m1",
        "messages": [{"role": "tool", "tool_call_id": "nope",
                      "content": "x"}]})
    failures = _check(failures, "an unknown tool_call_id is rejected",
                      bad_ref is not None and bad_ref.status == 400,
                      repr(bad_ref))

    chosen = orun.parse_chat_request({
        "model": "m1", "tools": [tool],
        "tool_choice": {"type": "function",
                        "function": {"name": "get_weather"}},
        "messages": user})
    dropped = orun.parse_chat_request({"model": "m1", "tools": [tool],
                                       "tool_choice": "none",
                                       "messages": user})
    failures = _check(failures, "tools pass through; tool_choice selects or drops",
                      bool(chosen.tools)
                      and chosen.tools[0]["function"]["name"] == "get_weather"
                      and dropped.tools is None,
                      f"{chosen.tools} {dropped.tools}")

    image = base64.b64encode(b"png-bytes").decode()
    with_image = orun.parse_chat_request({
        "model": "m1",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]}]})
    failures = _check(failures, "data-uri images map to Ollama images",
                      with_image.has_images
                      and with_image.messages[0]["images"] == [image]
                      and with_image.messages[0]["content"] == "what is this",
                      str(with_image.messages)[:200])
    remote = _api_error(orun.parse_chat_request, {
        "model": "m1",
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "ftp://example/x.png"}}]}]})
    failures = _check(failures, "a non-http image URL is rejected",
                      remote is not None and remote.status == 400
                      and remote.code == "unsupported_value",
                      repr(remote))

    import ollamarun_lib.openai_api as api_module
    wrapper = orun.parse_chat_request({
        "model": "m1", "messages": user,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "weather", "description": "d", "strict": True,
            "schema": {"type": "object"}}}})
    failures = _check(failures, "a full json_schema wrapper maps to its schema",
                      wrapper.fmt == {"type": "object"}, str(wrapper.fmt))
    original_cap = api_module.MAX_IMAGE_BYTES
    try:
        api_module.MAX_IMAGE_BYTES = 8
        small_b64 = base64.b64encode(b"1234").decode()
        small = orun.parse_chat_request({
            "model": "m1",
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{small_b64}"}}]}]})
        big_b64 = base64.b64encode(b"x" * 64).decode()
        big = _api_error(orun.parse_chat_request, {
            "model": "m1",
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{big_b64}"}}]}]})
        failures = _check(failures, "an oversized image is a 413",
                          small.has_images and big is not None
                          and big.status == 413
                          and big.code == "image_too_large", repr(big))
    finally:
        api_module.MAX_IMAGE_BYTES = original_cap

    plan = orun.parse_chat_request({"model": "m1", "messages": user})
    body = orun.completion_body(plan, [{
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "get_weather",
                          "arguments": {"city": "SP"}}}]},
        "done": True, "prompt_eval_count": 9, "eval_count": 3,
        "done_reason": "stop"}])
    choice = body["choices"][0]
    call = choice["message"]["tool_calls"][0]
    failures = _check(failures, "a tool-call completion maps with ids and usage",
                      body["object"] == "chat.completion"
                      and choice["finish_reason"] == "tool_calls"
                      and choice["message"]["content"] is None
                      and call["id"].startswith("call_")
                      and call["type"] == "function"
                      and json.loads(call["function"]["arguments"])
                      == {"city": "SP"}
                      and body["usage"] == {"prompt_tokens": 9,
                                            "completion_tokens": 3,
                                            "total_tokens": 12},
                      str(body)[:240])
    length = orun.completion_body(plan, [{"message": {"content": "cut"},
                                          "done": True, "eval_count": 5,
                                          "done_reason": "length"}])
    failures = _check(failures, "done_reason length becomes finish_reason length",
                      length["choices"][0]["finish_reason"] == "length",
                      str(length)[:160])

    # n > 1: one choice per emulated generation, usage summed.
    multi = orun.parse_chat_request({"model": "m1", "n": 2, "messages": user})
    multi_body = orun.completion_body(multi, [
        {"message": {"content": "one"}, "done": True,
         "prompt_eval_count": 5, "eval_count": 2, "done_reason": "stop"},
        {"message": {"content": "two"}, "done": True,
         "prompt_eval_count": 4, "eval_count": 3, "done_reason": "stop"},
    ])
    failures = _check(
        failures, "n > 1 maps to indexed choices with summed usage",
        [choice["index"] for choice in multi_body["choices"]] == [0, 1]
        and [choice["message"]["content"] for choice in multi_body["choices"]]
        == ["one", "two"]
        and multi_body["usage"] == {"prompt_tokens": 9,
                                    "completion_tokens": 5,
                                    "total_tokens": 14},
        str(multi_body)[:240])

    # parallel_tool_calls: false truncates the returned calls to the first.
    two_calls = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "a", "arguments": {}}},
        {"function": {"name": "b", "arguments": {}}}]},
        "done": True, "done_reason": "stop"}
    truncated_plan = orun.parse_chat_request(
        {"model": "m1", "messages": user, "parallel_tool_calls": False})
    kept_plan = orun.parse_chat_request(
        {"model": "m1", "messages": user, "parallel_tool_calls": True})
    truncated = orun.completion_body(truncated_plan, [two_calls])
    kept = orun.completion_body(kept_plan, [two_calls])
    failures = _check(
        failures, "parallel_tool_calls: false forwards only the first call",
        len(truncated["choices"][0]["message"]["tool_calls"]) == 1
        and truncated["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "a"
        and len(kept["choices"][0]["message"]["tool_calls"]) == 2,
        f"{truncated['choices'][0]['message']} {kept['choices'][0]['message']}")
    model = orun.model_object(
        {"name": "m1:latest",
         "modified_at": "2026-10-17T10:00:00.123456789-03:00"})
    failures = _check(failures, "model objects carry a parsed timestamp",
                      model["id"] == "m1:latest" and model["object"] == "model"
                      and isinstance(model["created"], int)
                      and model["created"] > 0, str(model))

    leaked = [name for name in ("AgentRuntime", "ToolRunner", "Session",
                                "TurnManager")
              if hasattr(api_module, name)]
    failures = _check(failures, "the OpenAI layer never reaches the agent loop",
                      not leaked, str(leaked))
    return failures


def test_embeddings_api() -> int:
    """`/v1/embeddings`: schema, encodings, gates and backend mapping."""
    failures = 0
    vectors = [[0.5, -1.25, 2.0], [1.0, 2.0, 3.0]]
    client = _ServeClient(
        models=["m1", "chat-only", "mystery"],
        capabilities=["embedding"],
        capabilities_by_model={"chat-only": ["completion"], "mystery": []},
        embed_data={"embeddings": vectors, "prompt_eval_count": 7})
    server, client, thread = _start_api(client)
    try:
        status, _headers, raw = _http(
            server, "POST", "/v1/embeddings",
            body={"model": "m1", "input": "hello"})
        body = json.loads(raw)
        failures = _check(
            failures, "a string input maps to the OpenAI embeddings schema",
            status == 200 and body["object"] == "list"
            and body["model"] == "m1"
            and body["data"] == [
                {"object": "embedding", "index": 0,
                 "embedding": [0.5, -1.25, 2.0]},
                {"object": "embedding", "index": 1,
                 "embedding": [1.0, 2.0, 3.0]}]
            and body["usage"] == {"prompt_tokens": 7, "total_tokens": 7},
            raw[:240])
        sent = client.embed_payloads[-1]
        failures = _check(failures, "the backend receives the inputs verbatim",
                          sent["input"] == ["hello"] and sent["truncate"] is True
                          and sent["dimensions"] is None, str(sent))

        status, _headers, raw = _http(
            server, "POST", "/v1/embeddings",
            body={"model": "m1", "input": ["a", "b"], "dimensions": 2,
                  "encoding_format": "float", "user": "someone"})
        body = json.loads(raw)
        failures = _check(failures, "a list input, dimensions and user map",
                          status == 200 and len(body["data"]) == 2
                          and client.embed_payloads[-1]["input"] == ["a", "b"]
                          and client.embed_payloads[-1]["dimensions"] == 2,
                          raw[:240])

        status, _headers, raw = _http(
            server, "POST", "/v1/embeddings",
            body={"model": "m1", "input": "hello",
                  "encoding_format": "base64"})
        body = json.loads(raw)
        packed = base64.b64decode(body["data"][0]["embedding"])
        values = list(struct.unpack(f"<{len(packed) // 4}f", packed))
        failures = _check(
            failures, "base64 is the SDK's little-endian float32 packing",
            status == 200 and values == [0.5, -1.25, 2.0], repr(values))

        status, _headers, raw = _http(
            server, "POST", "/v1/embeddings",
            body={"model": "chat-only", "input": "hello"})
        error = json.loads(raw)["error"]
        failures = _check(
            failures, "a completion-only model is refused",
            status == 400
            and error["code"] == "model_does_not_support_embeddings"
            and error["param"] == "model", f"{status} {raw[:200]}")
        status, _headers, raw = _http(
            server, "POST", "/v1/embeddings",
            body={"model": "mystery", "input": "hello"})
        failures = _check(failures, "an empty capability list passes",
                          status == 200, f"{status} {raw[:160]}")

        for label, body_ in (
            ("an unknown field", {"model": "m1", "input": "x", "foo": 1}),
            ("a missing input", {"model": "m1"}),
            ("an empty input", {"model": "m1", "input": ""}),
            ("an empty array", {"model": "m1", "input": []}),
            ("a token array", {"model": "m1", "input": [1, 2, 3]}),
            ("a nested array", {"model": "m1", "input": [[1, 2]]}),
            ("a zero dimensions", {"model": "m1", "input": "x",
                                   "dimensions": 0}),
            ("a boolean dimensions", {"model": "m1", "input": "x",
                                      "dimensions": True}),
            ("a bad encoding_format", {"model": "m1", "input": "x",
                                       "encoding_format": "hex"}),
        ):
            status, _headers, raw = _http(server, "POST", "/v1/embeddings",
                                          body=body_)
            failures = _check(failures, f"embeddings rejects {label}",
                              status == 400, f"{status} {raw[:160]}")

        # The serve_models allowlist applies like everywhere else.
        client2 = _ServeClient(models=["m1"], capabilities=["embedding"])
        server2, _client2, thread2 = _start_api(
            client2, router_kwargs={"config": {"serve_models": ["allowed"]}})
        try:
            status, _headers, raw = _http(server2, "POST", "/v1/embeddings",
                                          body={"model": "m1", "input": "x"})
            failures = _check(failures, "embeddings honors serve_models",
                              status == 404
                              and json.loads(raw)["error"]["code"]
                              == "model_not_found", f"{status} {raw[:160]}")
        finally:
            _stop_api(server2, thread2)

        # The resolved-input budget bounds embeddings inputs too.
        client3 = _ServeClient(models=["m1"], capabilities=["embedding"])
        server3, _client3, thread3 = _start_api(
            client3, router_kwargs={"config": {"serve_max_messages": 2}})
        try:
            status, _headers, raw = _http(
                server3, "POST", "/v1/embeddings",
                body={"model": "m1", "input": ["a", "b", "c"]})
            failures = _check(
                failures, "the input budget bounds the embeddings list",
                status == 413
                and json.loads(raw)["error"]["code"] == "input_too_large",
                f"{status} {raw[:160]}")
        finally:
            _stop_api(server3, thread3)

        client.embed_error = "HTTP 404 on /api/embed: model not found"
        status, _headers, raw = _http(server, "POST", "/v1/embeddings",
                                      body={"model": "m1", "input": "x"})
        failures = _check(failures, "an upstream failure is a sanitized 404",
                          status == 404
                          and json.loads(raw)["error"]["code"]
                          == "model_not_found", f"{status} {raw[:160]}")
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_http() -> int:
    """The OpenAI HTTP surface over a live loopback server, fake backend."""
    failures = 0
    script = [[
        {"message": {"thinking": "hmm", "content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 3,
         "eval_count": 2, "done_reason": "stop"},
    ]]
    client = _ServeClient(models=["m1", "m2"],
                          capabilities=["completion", "tools"],
                          chat_script=script)
    server, client, thread = _start_api(client)
    try:
        status, _headers, raw = _http(server, "GET", "/v1/models")
        payload = json.loads(raw)
        failures = _check(failures, "GET /v1/models lists the backend models",
                          status == 200 and payload["object"] == "list"
                          and payload["data"][0]["id"] == "m1"
                          and payload["data"][0]["object"] == "model"
                          and payload["data"][0]["owned_by"] == "ollamarun",
                          raw[:160])
        # The locked contract: LAN-accessible, no authentication at all.
        status, headers, raw = _http(server, "GET", "/v1/models")
        failures = _check(failures, "no credentials are required",
                          status == 200
                          and not any(key.lower() == "www-authenticate"
                                      for key in headers),
                          f"{status} {dict(headers)}")
        status, _headers, with_token = _http(
            server, "GET", "/v1/models",
            headers={"Authorization": "Bearer nope"})
        failures = _check(failures, "an arbitrary token changes nothing",
                          status == 200 and with_token == raw,
                          f"{status} {with_token[:80]!r}")
        status, _headers, raw = _http(server, "GET", "/v1/models/m1")
        failures = _check(failures, "GET /v1/models/{id} finds the model",
                          status == 200 and json.loads(raw)["id"] == "m1",
                          raw[:120])
        status, _headers, raw = _http(server, "GET", "/v1/models/nope")
        error = json.loads(raw)["error"]
        failures = _check(failures, "an unknown model 404s in the envelope",
                          status == 404 and error["code"] == "model_not_found"
                          and error["type"] == "invalid_request_error",
                          raw[:160])
        status, _headers, raw = _http(server, "POST", "/v1/models")
        failures = _check(failures, "a wrong method is a 405 envelope",
                          status == 405
                          and json.loads(raw)["error"]["code"] == "method_not_allowed",
                          raw[:160])
        status, _headers, raw = _http(server, "GET", "/v1/nope")
        failures = _check(failures, "an unknown endpoint is a 404 envelope",
                          status == 404 and "error" in json.loads(raw),
                          raw[:160])
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
                                      raw="{not json")
        failures = _check(failures, "an invalid JSON body is a 400",
                          status == 400
                          and json.loads(raw)["error"]["code"] == "invalid_json",
                          raw[:160])
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
                                      body={"model": "m1",
                                            "messages": [{"role": "user",
                                                          "content": "x"}],
                                            "foo": 1})
        body = json.loads(raw)
        failures = _check(failures, "an unknown request field is explicit",
                          status == 400 and body["error"]["param"] == "foo"
                          and body["error"]["code"] == "unknown_field",
                          raw[:160])
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
                                      body={"model": "missing",
                                            "messages": [{"role": "user",
                                                          "content": "x"}]})
        failures = _check(failures, "an unknown model 404s before inference",
                          status == 404 and not client.chat_payloads
                          and not client.chat_once_payloads, raw[:160])

        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
                                      body={"model": "m1", "temperature": 0.2,
                                            "messages": [{"role": "user",
                                                          "content": "hi"}]})
        completion = json.loads(raw)
        choice = completion["choices"][0]
        failures = _check(failures, "a non-streaming completion is OpenAI-shaped",
                          status == 200
                          and completion["object"] == "chat.completion"
                          and completion["model"] == "m1"
                          and choice["message"]["content"] == "ok"
                          and choice["finish_reason"] == "stop"
                          and completion["usage"] == {"prompt_tokens": 4,
                                                      "completion_tokens": 2,
                                                      "total_tokens": 6},
                          raw[:200])
        sent = client.chat_once_payloads[-1]
        failures = _check(failures, "mapped options reach the backend",
                          sent["options"] == {"temperature": 0.2}
                          and sent["stream"] is False
                          and sent["messages"][-1]["content"] == "hi",
                          str(sent)[:200])

        client.chat_once_data = {
            "message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": {"city": "SP"}}}]},
            "done": True, "prompt_eval_count": 8, "eval_count": 4,
            "done_reason": "stop"}
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
            body={"model": "m1",
                  "messages": [{"role": "user", "content": "weather?"}],
                  "tools": [{"type": "function", "function": {
                      "name": "get_weather", "description": "w",
                      "parameters": {"type": "object"}}}]})
        body = json.loads(raw)
        call = body["choices"][0]["message"]["tool_calls"][0]
        sent = client.chat_once_payloads[-1]
        failures = _check(failures, "client tools pass through as tool_calls",
                          body["choices"][0]["finish_reason"] == "tool_calls"
                          and call["function"]["name"] == "get_weather"
                          and json.loads(call["function"]["arguments"])
                          == {"city": "SP"}
                          and sent["tools"][0]["function"]["name"]
                          == "get_weather", raw[:200])

        status, headers, raw = _http(server, "POST", "/v1/chat/completions",
            body={"model": "m1",
                  "messages": [{"role": "user", "content": "hi"}],
                  "stream": True,
                  "stream_options": {"include_usage": True}})
        events = _sse_events(raw)
        chunks = [json.loads(item) for item in events if item != "[DONE]"]
        failures = _check(failures, "the SSE stream ends with [DONE]",
                          status == 200 and bool(events)
                          and events[-1] == "[DONE]"
                          and headers.get("Content-Type") == "text/event-stream",
                          f"{status} {events[-2:]}")
        failures = _check(failures, "the first chunk opens with the role",
                          chunks[0]["object"] == "chat.completion.chunk"
                          and chunks[0]["choices"][0]["delta"]
                          == {"role": "assistant", "content": ""},
                          str(chunks[0])[:160])
        data_chunks = [chunk for chunk in chunks if chunk["choices"]]
        deltas = [chunk["choices"][0]["delta"] for chunk in data_chunks]
        failures = _check(failures, "thinking and content stream as deltas",
                          deltas[1] == {"reasoning_content": "hmm"}
                          and deltas[2] == {"content": "Hello"}
                          and deltas[3] == {"content": " world"},
                          str(deltas))
        failures = _check(failures, "the finish chunk carries the reason",
                          data_chunks[-1]["choices"][0]["finish_reason"] == "stop",
                          str(data_chunks[-1])[:160])
        failures = _check(failures, "include_usage adds a final usage chunk",
                          chunks[-1]["choices"] == []
                          and chunks[-1]["usage"] == {"prompt_tokens": 3,
                                                      "completion_tokens": 2,
                                                      "total_tokens": 5},
                          str(chunks[-1])[:160])

        client.chat_script = [[
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": {"city": "SP"}}}]}},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 5,
             "eval_count": 3, "done_reason": "stop"},
        ]]
        client.chat_calls = 0
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
            body={"model": "m1", "stream": True,
                  "messages": [{"role": "user", "content": "w"}]})
        events = _sse_events(raw)
        chunks = [json.loads(item) for item in events if item != "[DONE]"]
        tool_delta = [chunk for chunk in chunks
                      if chunk["choices"][0]["delta"].get("tool_calls")]
        failures = _check(failures, "tool calls stream as one indexed delta array",
                          len(tool_delta) == 1
                          and tool_delta[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
                          and tool_delta[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
                          and chunks[-1]["choices"][0]["finish_reason"] == "tool_calls",
                          str(chunks)[:240])
        failures = _check(failures, "no usage chunk unless requested",
                          all("usage" not in json.loads(item)
                              for item in events if item != "[DONE]"),
                          str(events)[:200])

        # n > 1: emulated with sequential generations, one choice each.
        calls_before = len(client.chat_once_payloads)
        client.chat_once_data = None
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
                                      body={"model": "m1", "n": 3,
                                            "messages": [{"role": "user",
                                                          "content": "x"}]})
        completion = json.loads(raw)
        failures = _check(
            failures, "non-streaming n runs n generations and sums the usage",
            status == 200
            and [choice["index"] for choice in completion["choices"]] == [0, 1, 2]
            and all(choice["message"]["content"] == "ok"
                    for choice in completion["choices"])
            and len(client.chat_once_payloads) - calls_before == 3
            and completion["usage"] == {"prompt_tokens": 12,
                                        "completion_tokens": 6,
                                        "total_tokens": 18},
            raw[:240])

        client.chat_script = [
            [{"message": {"content": "one"}, "done": False},
             {"message": {"content": ""}, "done": True,
              "prompt_eval_count": 2, "eval_count": 1, "done_reason": "stop"}],
            [{"message": {"content": "two"}, "done": False},
             {"message": {"content": ""}, "done": True,
              "prompt_eval_count": 3, "eval_count": 2, "done_reason": "stop"}],
        ]
        client.chat_calls = 0
        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": "m1", "stream": True, "n": 2,
                  "stream_options": {"include_usage": True},
                  "messages": [{"role": "user", "content": "x"}]})
        events = _sse_events(raw)
        chunks = [json.loads(item) for item in events if item != "[DONE]"]
        data_chunks = [chunk for chunk in chunks if chunk["choices"]]
        contents = {chunk["choices"][0]["index"]:
                    chunk["choices"][0]["delta"].get("content")
                    for chunk in data_chunks
                    if chunk["choices"][0]["delta"].get("content")}
        finishes = [chunk["choices"][0]["finish_reason"] for chunk in data_chunks
                    if chunk["choices"][0]["finish_reason"]]
        failures = _check(
            failures, "streaming n multiplexes per-choice indexes",
            status == 200
            and [chunk["choices"][0]["index"] for chunk in data_chunks]
            == [0, 0, 0, 1, 1, 1]
            and contents == {0: "one", 1: "two"}
            and finishes == ["stop", "stop"]
            and chunks[-1]["choices"] == []
            and chunks[-1]["usage"] == {"prompt_tokens": 5,
                                        "completion_tokens": 3,
                                        "total_tokens": 8}
            and events[-1] == "[DONE]",
            str(chunks)[:240])

        # parallel_tool_calls: false truncates to the first returned call.
        two_calls = {"message": {"content": "", "tool_calls": [
            {"function": {"name": "a", "arguments": {}}},
            {"function": {"name": "b", "arguments": {}}}]},
            "done": True, "prompt_eval_count": 5, "eval_count": 2,
            "done_reason": "stop"}
        client.chat_once_data = two_calls
        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": "m1", "parallel_tool_calls": False,
                  "messages": [{"role": "user", "content": "x"}]})
        completion = json.loads(raw)
        failures = _check(
            failures, "non-streaming parallel_tool_calls: false keeps one call",
            status == 200 and len(completion["choices"][0]["message"]
                                  ["tool_calls"]) == 1
            and completion["choices"][0]["message"]["tool_calls"][0]
            ["function"]["name"] == "a",
            raw[:200])

        client.chat_script = [[
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": {}}},
                {"function": {"name": "b", "arguments": {}}}]}},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 5,
             "eval_count": 3, "done_reason": "stop"},
        ]]
        client.chat_calls = 0
        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": "m1", "stream": True,
                  "parallel_tool_calls": False,
                  "messages": [{"role": "user", "content": "x"}]})
        chunks = [json.loads(item) for item in _sse_events(raw)
                  if item != "[DONE]"]
        tool_delta = [chunk for chunk in chunks
                      if chunk["choices"][0]["delta"].get("tool_calls")]
        failures = _check(
            failures, "streaming parallel_tool_calls: false keeps one call",
            status == 200 and len(tool_delta) == 1
            and len(tool_delta[0]["choices"][0]["delta"]["tool_calls"]) == 1
            and tool_delta[0]["choices"][0]["delta"]["tool_calls"][0]
            ["function"]["name"] == "a",
            str(chunks)[:240])

        image = base64.b64encode(b"png").decode()
        image_part = [{"type": "text", "text": "?"},
                      {"type": "image_url",
                       "image_url": {"url": f"data:image/png;base64,{image}"}}]
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
            body={"model": "m1",
                  "messages": [{"role": "user", "content": image_part}]})
        failures = _check(failures, "images are refused without vision support",
                          status == 400
                          and json.loads(raw)["error"]["code"]
                          == "model_does_not_support_images", raw[:160])
        client.capability_list = ["completion", "vision"]
        status, _headers, raw = _http(server, "POST", "/v1/chat/completions",
            body={"model": "m2",
                  "messages": [{"role": "user", "content": image_part}]})
        failures = _check(failures, "a vision model receives the image",
                          status == 200
                          and client.chat_once_payloads[-1]["messages"][-1]["images"]
                          == [image], raw[:160])

        big_server, _big, big_thread = _start_api(
            _ServeClient(models=["m1"]),
            server_kwargs={"body_limit": 1024})
        try:
            status, _headers, raw = _http(big_server, "POST",
                                          "/v1/chat/completions",
                                          raw="x" * 2048)
            failures = _check(failures, "oversized bodies are rejected before parsing",
                              status == 413
                              and json.loads(raw)["error"]["code"] == "body_too_large",
                              raw[:160])
        finally:
            _stop_api(big_server, big_thread)
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_disconnect() -> int:
    """A disconnecting SSE client cancels only its own upstream generation."""
    failures = 0
    client = _ServeClient(models=["m1"], chunk_count=400, chunk_delay=0.005)
    server, client, thread = _start_api(client)
    try:
        body = json.dumps({"model": "m1", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                        timeout=10)
        head = (b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n")
        sock.sendall(head + body)
        first = b""
        deadline = time.monotonic() + 5.0
        while b"data: " not in first and time.monotonic() < deadline:
            piece = sock.recv(4096)
            if not piece:
                break
            first += piece
        # Hard-close (RST): the server's next write must fail immediately.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))
        sock.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and client.abort_calls == 0:
            time.sleep(0.02)
        failures = _check(failures, "a dropped client cancels upstream generation",
                          client.abort_calls >= 1,
                          f"abort_calls={client.abort_calls}")
        failures = _check(failures, "chunks streamed before the drop",
                          b"data: " in first, repr(first[:80]))
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_sockets() -> int:
    """Socket lifecycle: request timeout, long streams, shutdown, isolation."""
    failures = 0

    # A stalled request body hits the wired request timeout with a 408.
    server, _client, thread = _start_api(
        _ServeClient(models=["m1"]),
        server_kwargs={"request_timeout": 0.4})
    try:
        sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                        timeout=8)
        try:
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                         b"Host: 127.0.0.1\r\n"
                         b"Content-Type: application/json\r\n"
                         b"Content-Length: 100\r\n\r\n")
            data = b""
            while True:
                piece = sock.recv(4096)
                if not piece:
                    break
                data += piece
        finally:
            sock.close()
        status_line = data.split(b"\r\n", 1)[0].decode(errors="replace")
        body = data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in data else b""
        try:
            envelope = json.loads(body or b"{}")
        except ValueError:
            envelope = {}
        failures = _check(failures, "a stalled body times out with a 408 envelope",
                          "408" in status_line
                          and envelope.get("error", {}).get("code") == "timeout",
                          f"{status_line!r} {data[:160]!r}")
    finally:
        _stop_api(server, thread)

    # A stream that outlives request_timeout still completes: the timeout is
    # per socket operation, not a deadline on the response.
    slow_stream = _ServeClient(models=["m1"], chunk_count=8, chunk_delay=0.1)
    server, _client, thread = _start_api(
        slow_stream, server_kwargs={"request_timeout": 0.3})
    try:
        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": "m1", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=15)
        events = _sse_events(raw)
        content = [chunk for chunk in (json.loads(item) for item in events
                                       if item != "[DONE]")
                   if chunk["choices"]
                   and chunk["choices"][0]["delta"].get("content")]
        failures = _check(failures, "a stream longer than request_timeout completes",
                          status == 200 and events and events[-1] == "[DONE]"
                          and len(content) == 8,
                          f"{status} {len(events)} {len(content)}")
    finally:
        _stop_api(server, thread)

    # Shutdown while a stream is in flight cannot hang the accept loop.
    busy = _ServeClient(models=["m1"], chunk_count=150, chunk_delay=0.01)
    server, _client, thread = _start_api(busy)
    result: dict = {}

    def read_stream() -> None:
        result["out"] = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": "m1", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=20)

    reader = threading.Thread(target=read_stream)
    reader.start()
    time.sleep(0.25)
    started = time.monotonic()
    server.shutdown()
    elapsed = time.monotonic() - started
    server.close()
    thread.join(timeout=5)
    failures = _check(failures, "shutdown during a stream returns promptly",
                      elapsed < 2.0 and not thread.is_alive(), f"{elapsed:.2f}s")
    reader.join(timeout=10)
    failures = _check(failures, "the in-flight stream never deadlocks the turn",
                      not reader.is_alive(), "reader still alive")

    # Two concurrent streams: a hard client abort cancels only its own client.
    clients = []

    def factory():
        fresh = _ServeClient(models=["m1"], chunk_count=30, chunk_delay=0.02)
        clients.append(fresh)
        return fresh

    server, _client, thread = _start_api(factory=factory)
    try:
        body = json.dumps({"model": "m1", "stream": True,
                           "messages": [{"role": "user",
                                         "content": "hi"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                        timeout=8)
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                     b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode()
                     + b"\r\n\r\n" + body)
        sock.recv(256)  # the first stream is live

        second: dict = {}

        def read_second() -> None:
            second["out"] = _http(
                server, "POST", "/v1/chat/completions",
                body={"model": "m1", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]},
                timeout=20)

        reader = threading.Thread(target=read_second)
        reader.start()
        time.sleep(0.1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))
        sock.close()  # hard-abort the first stream
        reader.join(timeout=15)
        events = _sse_events(second["out"][2]) if second.get("out") else []
        content = [chunk for chunk in (json.loads(item) for item in events
                                       if item != "[DONE]")
                   if chunk["choices"]
                   and chunk["choices"][0]["delta"].get("content")]
        failures = _check(failures, "aborting one stream leaves the other intact",
                          bool(second.get("out")) and bool(events)
                          and events[-1] == "[DONE]" and len(content) == 30,
                          f"{len(events)} {len(content)}")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(c.abort_calls for c in clients):
            time.sleep(0.02)
        aborted = [c for c in clients if c.abort_calls]
        untouched = [c for c in clients if c not in aborted]
        failures = _check(failures, "only the aborted stream's client is cancelled",
                          len(aborted) == 1
                          and all(c.abort_calls == 0 for c in untouched),
                          str([c.abort_calls for c in clients]))
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_cache_concurrency() -> int:
    """Concurrent cold requests share one upstream model/capability fetch."""
    failures = 0

    class _CountingClient(_ServeClient):
        def __init__(self):
            super().__init__(models=["m1", "m2"],
                             capabilities=["completion", "vision"])
            self.tags_calls = 0
            self.capability_calls = 0

        def tags(self):
            self.tags_calls += 1
            time.sleep(0.05)
            return super().tags()

        def capabilities(self, model):
            self.capability_calls += 1
            time.sleep(0.05)
            return super().capabilities(model)

    client = _CountingClient()
    server, _client, thread = _start_api(factory=lambda: client)
    try:
        statuses: list = []

        def hit_models() -> None:
            statuses.append(_http(server, "GET", "/v1/models")[0])

        threads = [threading.Thread(target=hit_models) for _ in range(8)]
        for thread_ in threads:
            thread_.start()
        for thread_ in threads:
            thread_.join()
        failures = _check(failures, "concurrent model lists are all served",
                          statuses == [200] * 8 and client.tags_calls == 1,
                          f"{statuses} tags={client.tags_calls}")

        image = base64.b64encode(b"png").decode()
        image_part = [{"type": "image_url",
                       "image_url": {"url": f"data:image/png;base64,{image}"}}]
        statuses.clear()

        def hit_image() -> None:
            statuses.append(_http(
                server, "POST", "/v1/chat/completions",
                body={"model": "m2", "messages": [
                    {"role": "user", "content": image_part}]})[0])

        threads = [threading.Thread(target=hit_image) for _ in range(4)]
        for thread_ in threads:
            thread_.start()
        for thread_ in threads:
            thread_.join()
        failures = _check(failures, "concurrent vision checks share one probe",
                          statuses == [200] * 4
                          and client.capability_calls == 1,
                          f"{statuses} capabilities={client.capability_calls}")
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_config_matrix() -> int:
    """The validated serve_* matrix: defaults, zeros, invalid values, wiring."""
    failures = 0
    base = orun.serve_settings({})
    failures = _check(
        failures, "the serve matrix defaults come from the constants",
        base["serve_body_limit"] == orun.DEFAULT_SERVE_BODY_LIMIT
        and base["serve_request_timeout"] == orun.DEFAULT_SERVE_REQUEST_TIMEOUT
        and base["serve_sse_ping_interval"] == orun.DEFAULT_SERVE_PING_INTERVAL
        and base["serve_shutdown_timeout"] == orun.DEFAULT_SERVE_SHUTDOWN_TIMEOUT
        and base["serve_agent"] is True and base["serve_models"] == []
        and base["serve_workspace"] is None
        and base["serve_unload_on_exit"] is True
        and base["serve_max_messages"] == 16384
        and base["serve_max_content_parts"] == 4096
        and base["serve_max_message_bytes"] == 8 * 1024 * 1024
        and base["serve_max_text_bytes"] == 32 * 1024 * 1024
        and base["serve_max_images"] == 64
        and base["serve_max_image_bytes"] == 64 * 1024 * 1024
        and base["serve_max_active_turns"] == 4
        and base["serve_turn_event_buffer"] == 10000
        and base["serve_max_active_fits"] == 2,
        str({k: v for k, v in sorted(base.items())
             if k.startswith("serve_")}))

    zeros = orun.serve_settings({
        "serve_approval_timeout": 0, "serve_turn_timeout": 0,
        "serve_turn_event_ttl": 0, "serve_artifact_ttl": 0, "serve_fit_ttl": 0,
        "serve_artifact_max_bytes": 0, "serve_artifact_quota_bytes": 0})
    failures = _check(
        failures, "an explicit zero keeps its documented meaning",
        zeros["serve_approval_timeout"] == 0.0
        and zeros["serve_turn_timeout"] == 0.0
        and zeros["serve_turn_event_ttl"] == 0.0
        and zeros["serve_artifact_ttl"] == 0.0
        and zeros["serve_fit_ttl"] == 0.0
        and zeros["serve_artifact_max_bytes"] == 0
        and zeros["serve_artifact_quota_bytes"] == 0,
        str({k: zeros[k] for k in sorted(zeros)
             if "_ttl" in k or "timeout" in k or "artifact" in k}))

    bad = [
        ("serve_bind", ""), ("serve_port", "11440"), ("serve_port", 70000),
        ("serve_agent", "yes"), ("serve_models", "qwen3"),
        ("serve_models", ["qwen3", "  "]),
        ("serve_workspace", "/no/such/dir/anywhere"),
        ("serve_body_limit", 10), ("serve_body_limit", True),
        ("serve_request_timeout", 0), ("serve_request_timeout", float("nan")),
        ("serve_sse_ping_interval", 0), ("serve_shutdown_timeout", 0),
        ("serve_unload_on_exit", 1), ("serve_max_messages", 0),
        ("serve_max_content_parts", 0), ("serve_max_message_bytes", 0),
        ("serve_max_text_bytes", 0), ("serve_max_images", 0),
        ("serve_max_image_bytes", 0), ("serve_max_active_turns", 0),
        ("serve_turn_event_buffer", 0), ("serve_turn_event_ttl", -1),
        ("serve_approval_timeout", -1), ("serve_turn_timeout", -1),
        ("serve_artifact_root", ""), ("serve_artifact_max_bytes", -1),
        ("serve_artifact_quota_bytes", -1), ("serve_artifact_ttl", -1),
        ("serve_fit", "auto"), ("serve_max_active_fits", 0),
        ("serve_fit_ttl", -1),
    ]
    invalid = []
    for key, value in bad:
        try:
            orun.serve_settings({key: value})
            invalid.append(f"{key}={value!r}")
        except ValueError:
            pass
    failures = _check(failures, "invalid values raise and name the key",
                      not invalid, ", ".join(invalid))

    overridden = orun.serve_settings({"serve_bind": "127.0.0.1",
                                      "serve_port": 1},
                                     bind="0.0.0.0", port=11441)
    failures = _check(failures, "CLI --bind/--port win over the config",
                      overridden["serve_bind"] == "0.0.0.0"
                      and overridden["serve_port"] == 11441,
                      f"{overridden['serve_bind']} {overridden['serve_port']}")

    args = orun.build_serve_parser().parse_args(["--keep-loaded"])
    failures = _check(failures, "the serve parser takes --keep-loaded",
                      args.keep_loaded is True, str(args))

    with tempfile.TemporaryDirectory() as tmp:
        shaped = orun.serve_settings({
            "serve_workspace": tmp, "serve_models": [" qwen3 ", "m1:*"],
            "serve_port": None})
        failures = _check(
            failures, "workspace resolves, models strip, null means default",
            shaped["serve_workspace"] == str(Path(tmp).resolve())
            and shaped["serve_models"] == ["qwen3", "m1:*"]
            and shaped["serve_port"] == orun.DEFAULT_SERVE_PORT,
            f"{shaped['serve_workspace']} {shaped['serve_models']}")

        # The transport values actually reach HttpServer from serve_main().
        import ollamarun_lib.cli as cli
        captured = {}
        real_server, real_forever = cli.HttpServer, cli.run_forever
        real_store = cli.SessionStore

        class _FakeServer:
            def __init__(self, router, *, bind, port, body_limit,
                         ping_interval, request_timeout):
                captured.update(bind=bind, port=port, body_limit=body_limit,
                                ping_interval=ping_interval,
                                request_timeout=request_timeout)
                self.address = (bind, port)

            def drain(self, timeout):
                return True

            def close(self):
                pass

        try:
            cli.HttpServer = _FakeServer
            cli.run_forever = lambda server: None
            cli.SessionStore = lambda *a, **k: real_store(
                sessions_dir=Path(tmp) / "sessions",
                index_path=Path(tmp) / "sessions-index.json")
            cfg = Path(tmp) / "config.json"
            cfg.write_text(json.dumps({
                "serve_body_limit": 4 * 1024 * 1024,
                "serve_request_timeout": 45,
                "serve_sse_ping_interval": 7,
                "serve_artifact_root": str(Path(tmp) / "artifacts"),
            }), encoding="utf-8")
            with orun.console.capture():
                code = cli.serve_main(["--config", str(cfg), "--port", "0",
                                       "--bind", "127.0.0.1", "--host",
                                       "127.0.0.1:1"])
            failures = _check(
                failures, "serve_main builds the server from the matrix",
                code == 0 and captured.get("body_limit") == 4 * 1024 * 1024
                and captured.get("request_timeout") == 45.0
                and captured.get("ping_interval") == 7.0
                and captured.get("port") == 0
                and captured.get("bind") == "127.0.0.1",
                f"{code} {captured}")
        finally:
            cli.HttpServer = real_server
            cli.run_forever = real_forever
            cli.SessionStore = real_store
    return failures


def _raw_exchange(server, payload: bytes, *, shutdown_write: bool = False,
                  timeout: float = 10.0):
    """One raw HTTP/1.1 exchange; returns (status, body, full response)."""
    sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                    timeout=timeout)
    try:
        sock.sendall(payload)
        if shutdown_write:
            sock.shutdown(socket.SHUT_WR)
        collected = []
        while True:
            try:
                piece = sock.recv(65536)
            except socket.timeout:
                break
            if not piece:
                break
            collected.append(piece)
    finally:
        sock.close()
    raw = b"".join(collected)
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0]
    parts = status_line.split(b" ", 2)
    status = int(parts[1]) if len(parts) > 1 else 0
    return status, body, raw


def _sized_chat_body(size: int) -> bytes:
    """A `POST /v1/chat/completions` JSON body of exactly `size` bytes."""
    separators = (",", ":")
    payload = {"model": "m1", "stream": False,
               "messages": [{"role": "user", "content": ""}]}
    pad = size - len(json.dumps(payload, separators=separators).encode("utf-8"))
    payload["messages"][0]["content"] = "a" * pad
    raw = json.dumps(payload, separators=separators).encode("utf-8")
    assert len(raw) == size, (len(raw), size)
    return raw


def test_chunked_bodies() -> int:
    """Chunked decoding: same decoded cap, structured framing errors."""
    failures = 0
    body = json.dumps({
        "model": "m1", "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode("utf-8")
    client = _ServeClient(models=["m1"], capabilities=["completion"])
    server, client, thread = _start_api(client)
    base_headers = (b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
                    b"Connection: close\r\n")

    def request(headers: bytes, body_bytes: bytes) -> bytes:
        return (b"POST /v1/chat/completions HTTP/1.1\r\n" + base_headers
                + headers + b"\r\n" + body_bytes)

    def chunked(pieces, *, trailers: bytes = b"",
                extensions: bytes = b";ext=1") -> bytes:
        out = bytearray()
        for piece in pieces:
            out += f"{len(piece):x}".encode() + extensions + b"\r\n"
            out += piece + b"\r\n"
        out += b"0\r\n" + trailers + b"\r\n"
        return bytes(out)

    def error_code(body_bytes: bytes) -> str:
        try:
            return json.loads(body_bytes.decode("utf-8"))["error"]["code"]
        except (ValueError, KeyError, UnicodeDecodeError):
            return f"<{body_bytes[:80]!r}>"

    try:
        status, response, raw = _raw_exchange(server, request(
            b"Transfer-Encoding: chunked\r\n",
            chunked([body[:10], body[10:33], body[33:]],
                    trailers=b"X-Trace: 1\r\n")))
        failures = _check(
            failures, "a chunked request decodes and completes",
            status == 200 and len(client.chat_once_payloads) == 1
            and client.chat_once_payloads[-1]["messages"][0]["content"] == "hi",
            f"{status} {raw[:200]}")

        client.chat_once_payloads.clear()
        server2, client2, thread2 = _start_api(
            _ServeClient(models=["m1"], capabilities=["completion"]),
            server_kwargs={"body_limit": 256})
        try:
            exact = _sized_chat_body(256)
            status, _body, raw = _raw_exchange(server2, request(
                b"Transfer-Encoding: chunked\r\n",
                chunked([exact[:20], exact[20:]])))
            failures = _check(
                failures, "a chunked body at the exact cap is accepted",
                status == 200 and len(client2.chat_once_payloads) == 1,
                f"{status} {raw[:200]}")

            over = _sized_chat_body(257)
            status, response, raw = _raw_exchange(server2, request(
                b"Transfer-Encoding: chunked\r\n", chunked([over])))
            failures = _check(
                failures, "an over-cap chunked body is a 413",
                status == 413 and error_code(response) == "body_too_large",
                f"{status} {error_code(response)}")

            status, response, raw = _raw_exchange(server2, request(
                f"Content-Length: {len(over)}\r\n".encode(), over))
            failures = _check(
                failures, "the Content-Length path enforces the same cap",
                status == 413 and error_code(response) == "body_too_large",
                f"{status} {error_code(response)}")

            huge = request(b"Transfer-Encoding: chunked\r\n",
                           b"7fffffff\r\n")
            status, response, _raw = _raw_exchange(server2, huge)
            failures = _check(
                failures, "an oversized chunk size is refused before reading",
                status == 413 and error_code(response) == "body_too_large",
                f"{status} {error_code(response)}")
        finally:
            _stop_api(server2, thread2)

        status, response, _raw = _raw_exchange(server, request(
            b"Transfer-Encoding: chunked\r\n",
            chunked([body], extensions=b"")))
        failures = _check(failures, "chunk extensions are ignored",
                          status == 200, f"{status} {_raw[:120]}")

        client.chat_once_payloads.clear()
        cases = [
            ("an invalid chunk size is a 400",
             request(b"Transfer-Encoding: chunked\r\n", b"zz\r\n"),
             "invalid_body"),
            ("a missing chunk CRLF is a 400",
             request(b"Transfer-Encoding: chunked\r\n",
                     b"2\r\nhiXX0\r\n\r\n"),
             "invalid_body"),
            ("empty trailers with the terminal chunk succeed",
             request(b"Transfer-Encoding: chunked\r\n",
                     f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"),
             ""),
            ("Transfer-Encoding plus Content-Length is ambiguous",
             request(b"Transfer-Encoding: chunked\r\nContent-Length: 4\r\n",
                     b"0\r\n\r\n"),
             "invalid_header"),
            ("a non-chunked coding is rejected",
             request(b"Transfer-Encoding: gzip\r\n",
                     chunked([body])),
             "unsupported_value"),
            ("a chunked+gzip coding list is rejected",
             request(b"Transfer-Encoding: chunked, gzip\r\n",
                     chunked([body])),
             "unsupported_value"),
        ]
        for name, payload, expected in cases:
            status, response, _raw = _raw_exchange(server, payload)
            ok = (status == 200 if not expected
                  else status == 400 and error_code(response) == expected)
            failures = _check(failures, name, ok,
                              f"{status} {error_code(response)}")

        truncated = request(b"Transfer-Encoding: chunked\r\n", b"a\r\nabcd")
        for name, payload in [
            ("a missing terminal chunk is a 400",
             request(b"Transfer-Encoding: chunked\r\n", b"5\r\nhello\r\n")),
            ("a truncated chunk is a 400", truncated),
        ]:
            status, response, _raw = _raw_exchange(server, payload,
                                                   shutdown_write=True)
            failures = _check(failures, name,
                              status == 400
                              and error_code(response) == "invalid_body",
                              f"{status} {error_code(response)}")
    finally:
        _stop_api(server, thread)
    return failures


def test_input_limits() -> int:
    """The aggregate resolved-input budget: caps, both surfaces, fail-fast."""
    failures = 0
    limits = orun.InputLimits(max_messages=2, max_content_parts=3,
                              max_message_bytes=100, max_text_bytes=200,
                              max_images=2, max_image_bytes=1000)
    text_part = {"type": "text", "text": "hi"}

    def budget_error(limits_, content, *,
                     where="messages[0].content"):
        return _api_error(orun.resolve_content_parts, content, where=where,
                          budget=orun.InputBudget(limits_))

    exc = budget_error(limits, "a" * 101)
    failures = _check(
        failures, "a message past the byte cap is a 413",
        exc is not None and exc.status == 413
        and exc.code == "input_too_large"
        and exc.extra.get("limit_name") == "serve_max_message_bytes",
        repr(exc))

    budget = orun.InputBudget(limits)
    budget.note_messages(2)
    exc = _api_error(budget.note_messages, 1)
    failures = _check(failures, "message counts are charged cumulatively",
                      exc is not None
                      and exc.extra.get("limit_name") == "serve_max_messages",
                      repr(exc))

    exc = budget_error(limits, [text_part] * 4)
    failures = _check(
        failures, "a fourth content part is a 413 naming the part",
        exc is not None and exc.code == "input_too_large"
        and exc.param == "messages[0].content[3]"
        and exc.extra.get("limit_name") == "serve_max_content_parts",
        repr(exc))

    image = base64.b64encode(b"x" * 480).decode("ascii")
    image_part = {"type": "image_url",
                  "image_url": {"url": f"data:image/png;base64,{image}"}}
    exc = budget_error(limits, [image_part, image_part, image_part])
    failures = _check(failures, "image counts are charged cumulatively",
                      exc is not None
                      and exc.extra.get("limit_name") == "serve_max_images",
                      repr(exc))

    byte_limits = orun.InputLimits(max_content_parts=10, max_images=5,
                                   max_image_bytes=1000)
    budget = orun.InputBudget(byte_limits)
    exc = None
    for _ in range(3):
        exc = _api_error(orun.resolve_content_parts, [image_part],
                         where="messages[0].content", budget=budget)
    failures = _check(failures, "cumulative image bytes are capped",
                      exc is not None
                      and exc.extra.get("limit_name") == "serve_max_image_bytes",
                      repr(exc))

    text_limits = orun.InputLimits(max_message_bytes=1000,
                                   max_text_bytes=200)
    budget = orun.InputBudget(text_limits)
    exc = None
    for _ in range(3):
        exc = _api_error(orun.resolve_content_parts, "t" * 80,
                         where="messages[0].content", budget=budget)
    failures = _check(failures, "total resolved text is capped across messages",
                      exc is not None
                      and exc.extra.get("limit_name") == "serve_max_text_bytes",
                      repr(exc))

    fetched = []

    def fetch(url, *, max_bytes):
        fetched.append(url)
        return b"x" * 10, "image/png"

    exc = _api_error(
        orun.resolve_content_parts,
        [text_part, {"type": "image_url",
                     "image_url": {"url": "http://example.test/x.png"}}],
        where="messages[0].content",
        budget=orun.InputBudget(orun.InputLimits(max_content_parts=1)),
        fetch=fetch)
    failures = _check(
        failures, "counts fail before any fetch happens",
        exc is not None
        and exc.extra.get("limit_name") == "serve_max_content_parts"
        and not fetched, f"{exc!r} {fetched}")

    # Both surfaces enforce the same budget from their config.
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"])
        server, api, store, thread = _start_agent_api(
            tmp, client, config={"serve_max_messages": 1,
                                 "serve_max_text_bytes": 5})
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/chat/completions",
                body={"model": "m1",
                      "messages": [{"role": "user", "content": "hi"},
                                   {"role": "user", "content": "there"}]})
            error = json.loads(raw)["error"]
            failures = _check(
                failures, "the OpenAI surface enforces the config budget",
                status == 413 and error["code"] == "input_too_large"
                and error["param"] == "messages",
                f"{status} {raw[:160]}")

            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("too much text"))
            error = json.loads(raw)["error"]
            failures = _check(
                failures, "the agent surface enforces the same budget",
                status == 413 and error["code"] == "input_too_large"
                and error["param"] == "messages[0].content",
                f"{status} {raw[:160]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_native_parameters() -> int:
    """Native option validation: ranges, non-finite, bounds, pass-through."""
    failures = 0
    good = {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": -1,
        "num_predict": -1, "num_ctx": 32768, "stop": ["END"],
        "use_mmap": True, "mirostat": 2, "repeat_penalty": 1.1,
        "custom_flag": 5, "custom_list": ["a", 1, True],
    }
    try:
        validated = orun.validate_settings_parameters(good)
        ok = validated == good
    except orun.ApiError as exc:
        ok, validated = False, exc
    failures = _check(failures, "valid known and unknown options pass",
                      ok, repr(validated))

    bad = [
        ({"temperature": 3.0}, "parameters.temperature"),
        ({"temperature": True}, "parameters.temperature"),
        ({"temperature": "0.5"}, "parameters.temperature"),
        ({"top_k": 1.5}, "parameters.top_k"),
        ({"seed": "1"}, "parameters.seed"),
        ({"num_ctx": 0}, "parameters.num_ctx"),
        ({"num_predict": -2}, "parameters.num_predict"),
        ({"mirostat": 3}, "parameters.mirostat"),
        ({"mirostat": True}, "parameters.mirostat"),
        ({"repeat_penalty": 0}, "parameters.repeat_penalty"),
        ({"use_mmap": "yes"}, "parameters.use_mmap"),
        ({"stop": [1]}, "parameters.stop"),
        ({"custom": {"nested": 1}}, "parameters.custom"),
        ({"custom": [[1]]}, "parameters.custom"),
        ({"custom": float("nan")}, "parameters.custom"),
        ({"custom": "x" * 5000}, "parameters.custom"),
        ({"custom": list(range(65))}, "parameters.custom"),
        ({" ": 1}, "parameters"),
    ]
    invalid = []
    for parameters, where in bad:
        exc = _api_error(orun.validate_settings_parameters, parameters,
                         where="parameters")
        if exc is None or exc.status != 400 \
                or exc.code != "unsupported_value" or exc.param != where:
            invalid.append(f"{parameters!r} -> {exc!r}")
    failures = _check(failures, "bad option values 400 naming the path",
                      not invalid, "; ".join(invalid))

    exc = _api_error(orun.validate_settings_parameters,
                     {"big": "x" * (70 * 1024)}, where="parameters")
    failures = _check(failures, "an oversized option object is rejected",
                      exc is not None and exc.status == 400, repr(exc))

    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"])
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/sessions",
                body={"model": "m1", "parameters": {"temperature": 3.0}})
            error = json.loads(raw)["error"]
            failures = _check(
                failures, "session create validates parameters",
                status == 400 and error["code"] == "unsupported_value"
                and error["param"] == "parameters.temperature",
                f"{status} {raw[:160]}")

            status, _headers, raw = _http(
                server, "POST", "/v1/sessions",
                body={"model": "m1",
                      "parameters": {"custom": 5, "top_k": 20}})
            created = json.loads(raw)
            failures = _check(
                failures, "unknown options pass through on create",
                status == 201, f"{status} {raw[:200]}")
            session_id = created["id"]

            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"parameters": {"temperature": 0.5}})
            failures = _check(failures, "PATCH merges validated parameters",
                              status == 200, f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{session_id}")
            stored = json.loads(raw).get("parameters", {})
            failures = _check(
                failures, "the merged parameters are persisted",
                stored.get("temperature") == 0.5 and stored.get("custom") == 5
                and stored.get("top_k") == 20, str(stored))

            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"parameters": {"temperature": 9}})
            error = json.loads(raw)["error"]
            failures = _check(
                failures, "PATCH rejects out-of-range parameters",
                status == 400 and error["param"] == "parameters.temperature",
                f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{session_id}")
            stored = json.loads(raw).get("parameters", {})
            failures = _check(
                failures, "a rejected PATCH leaves the session untouched",
                stored.get("temperature") == 0.5, str(stored))

            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("hi",
                                   settings={"parameters": {"top_k": -1}}))
            error = json.loads(raw)["error"]
            failures = _check(
                failures, "turn settings validate parameters before starting",
                status == 400
                and error["param"] == "settings.parameters.top_k",
                f"{status} {raw[:160]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_workspace_jail() -> int:
    """The API workspace: relative paths resolve under it, escapes refused."""
    failures = 0
    registry = orun.default_registry(tts=False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        outside = Path(tmp) / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "inside.txt").write_text("in\n")
        (root / "sub").mkdir()
        (outside / "secret.txt").write_text("out\n")
        (root / "escape").symlink_to(outside)

        api_ctx = orun.ToolContext(cwd=str(root), workspace=str(root),
                                   timeout=30, max_output=4000)
        cli_ctx = orun.ToolContext(cwd=str(root), timeout=30, max_output=4000)

        def read(ctx, path):
            return orun.execute_tool(registry["read_file"], {"path": path}, ctx)

        outcome = read(api_ctx, "inside.txt")
        failures = _check(failures, "a relative path resolves under the root",
                          outcome.ok and "in" in outcome.text, outcome.text)
        for name, probe in [
            ("a parent-relative path", "../outside/secret.txt"),
            ("an absolute path outside", str(outside / "secret.txt")),
            ("a symlinked directory", "escape/secret.txt"),
            ("a traversal-shaped absolute",
             str(root / ".." / "outside" / "secret.txt")),
        ]:
            outcome = read(api_ctx, probe)
            failures = _check(
                failures, f"{name} is refused",
                not outcome.ok and "outside the workspace" in outcome.text,
                outcome.text)

        outcome = read(cli_ctx, str(outside / "secret.txt"))
        failures = _check(failures, "the CLI context keeps its behavior",
                          outcome.ok, outcome.text)

        write = orun.execute_tool(registry["write_file"],
                                  {"path": "sub/new.txt", "content": "x"},
                                  api_ctx)
        failures = _check(failures, "a write under the root lands there",
                          write.ok
                          and (root / "sub" / "new.txt").read_text() == "x",
                          write.text)
        write = orun.execute_tool(registry["write_file"],
                                  {"path": str(outside / "evil.txt"),
                                   "content": "x"}, api_ctx)
        failures = _check(failures, "an absolute write outside is refused",
                          not write.ok and not (outside / "evil.txt").exists(),
                          write.text)

        move = orun.execute_tool(registry["move_file"],
                                 {"source": "inside.txt",
                                  "destination": str(outside / "moved.txt")},
                                 api_ctx)
        failures = _check(failures, "a move destination outside is refused",
                          not move.ok
                          and not (outside / "moved.txt").exists()
                          and (root / "inside.txt").exists(), move.text)
        copy = orun.execute_tool(registry["copy_file"],
                                 {"source": str(outside / "secret.txt"),
                                  "destination": "copied.txt"}, api_ctx)
        failures = _check(failures, "a copy source outside is refused",
                          not copy.ok and not (root / "copied.txt").exists(),
                          copy.text)

        mkdir = orun.execute_tool(registry["make_dir"], {"path": "made"},
                                  api_ctx)
        failures = _check(failures, "make_dir resolves under the root",
                          mkdir.ok and (root / "made").is_dir(), mkdir.text)
        mkdir = orun.execute_tool(registry["make_dir"],
                                  {"path": str(outside / "made")}, api_ctx)
        failures = _check(failures, "make_dir outside is refused",
                          not mkdir.ok and not (outside / "made").exists(),
                          mkdir.text)

        delete = orun.execute_tool(registry["delete_file"],
                                   {"path": str(outside / "secret.txt")},
                                   api_ctx)
        failures = _check(failures, "delete_file outside is refused",
                          not delete.ok and (outside / "secret.txt").exists(),
                          delete.text)

        shell = orun.execute_tool(registry["run_shell"],
                                  {"command": "pwd", "cwd": "sub"}, api_ctx)
        failures = _check(failures, "run_shell resolves a relative cwd",
                          shell.ok and str(root / "sub") in shell.text,
                          shell.text)
        shell = orun.execute_tool(registry["run_shell"],
                                  {"command": "pwd", "cwd": str(outside)},
                                  api_ctx)
        failures = _check(failures, "run_shell refuses an outside cwd",
                          not shell.ok and "outside the workspace" in shell.text,
                          shell.text)
    return failures


def test_artifact_root_containment() -> int:
    """A symlinked dir under the root never redirects a read or a sweep."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "artifacts"
        outside = Path(tmp) / "outside"
        (outside / "uploads").mkdir(parents=True)
        store = orun.ArtifactStore(root, ttl=0.01)
        ref = store.register(b"payload", filename="a.txt",
                             media_type="text/plain")
        stored = store.read(ref.id)
        failures = _check(failures, "a normal artifact reads back",
                          stored is not None and stored[1] == b"payload",
                          repr(stored))

        # Replace the storage directory with a symlink out of the root and
        # move the stored bytes behind it: the resolved entry escapes.
        uploads = root / "uploads"
        target = outside / "uploads"
        (uploads / ref.id).rename(target / ref.id)
        uploads.rmdir()
        uploads.symlink_to(target)

        failures = _check(failures, "resolve refuses the escaping entry",
                          store.resolve(ref.id) is None,
                          "resolve returned a ref")
        failures = _check(failures, "read refuses the escaping entry",
                          store.read(ref.id) is None,
                          "read returned bytes")
        time.sleep(0.03)
        store.sweep()
        failures = _check(
            failures, "the sweep drops the entry without chasing it outside",
            store.resolve(ref.id) is None and (target / ref.id).exists(),
            f"outside file present: {(target / ref.id).exists()}")
    return failures


def test_serve_model_gate() -> int:
    """`serve_models` is one API-wide allowlist with tag-aware matching."""
    failures = 0
    patterns = ["m1", "qwen3*"]
    for model, expected in [("m1", True), ("m1:latest", True),
                            ("m1:8b", False), ("qwen3", True),
                            ("qwen3:8b", True), ("other", False), ("", False)]:
        got = orun.model_allowed(model, patterns)
        failures = _check(failures, f"model_allowed({model!r}) is {expected}",
                          got is expected, str(got))
    failures = _check(failures, "an empty allowlist serves everything",
                      orun.model_allowed("anything", [])
                      and orun.model_allowed("anything", None), "")

    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1", "other"],
                              capabilities=["completion", "tools"])
        server, api, store, thread = _start_agent_api(
            tmp, client, config={"serve_models": ["m1"]})
        try:
            status, _headers, raw = _http(server, "GET", "/v1/models")
            listed = [item["id"] for item in json.loads(raw)["data"]]
            failures = _check(failures, "the list filters to the allowlist",
                              status == 200 and listed == ["m1"], str(listed))

            for name, method, path, body in [
                ("lookup", "GET", "/v1/models/other", None),
                ("chat", "POST", "/v1/chat/completions",
                 {"model": "other",
                  "messages": [{"role": "user", "content": "hi"}]}),
                ("session create", "POST", "/v1/sessions",
                 {"model": "other"}),
                ("fit", "POST", "/v1/agent/fit",
                 {"model": "other", "mode": "gpu"}),
            ]:
                status, _headers, raw = _http(server, method, path, body=body)
                failures = _check(
                    failures, f"a forbidden model is 404 on {name}",
                    status == 404
                    and json.loads(raw)["error"]["code"] == "model_not_found",
                    f"{status} {raw[:140]}")

            stored = orun.Session.create("other", "")
            store.save(stored)
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body={"session_id": stored.id,
                      "messages": [{"role": "user", "content": "hi"}]})
            failures = _check(
                failures, "a stored session with a forbidden model is a 404",
                status == 404
                and json.loads(raw)["error"]["code"] == "model_not_found",
                f"{status} {raw[:140]}")

            status, _headers, raw = _http(server, "POST", "/v1/sessions",
                                          body={"model": "m1"})
            session_id = json.loads(raw)["id"]
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"model": "other"})
            failures = _check(
                failures, "a forbidden model is 404 on model PATCH",
                status == 404, f"{status} {raw[:140]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_serve_agent_switch() -> int:
    """serve_agent=false keeps inference and artifacts, drops agent routes."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        artifact_store = orun.ArtifactStore(Path(tmp) / "artifacts")
        client = _ServeClient(models=["m1"], capabilities=["completion"])
        server, api, store, thread = _start_agent_api(
            tmp, client, artifacts=artifact_store, agent_routes=False)
        try:
            status, _headers, raw = _http(server, "GET", "/v1/models")
            ok_models = status == 200 \
                and json.loads(raw)["data"][0]["id"] == "m1"
            status, _headers, raw = _http(
                server, "POST", "/v1/chat/completions",
                body={"model": "m1",
                      "messages": [{"role": "user", "content": "hi"}]})
            ok_chat = status == 200
            data = base64.b64encode(b"hello").decode()
            status, _headers, raw = _http(
                server, "POST", "/v1/files",
                body={"filename": "a.txt", "media_type": "text/plain",
                      "data_base64": data})
            ref = json.loads(raw)
            ok_upload = status == 201
            status, _headers, raw = _http(server, "GET",
                                          f"/v1/artifacts/{ref['id']}")
            ok_download = status == 200 and raw == b"hello"
            failures = _check(
                failures, "inference and artifacts stay available",
                ok_models and ok_chat and ok_upload and ok_download,
                f"{ok_models} {ok_chat} {ok_upload} {ok_download}")

            for method, path, body in [
                ("GET", "/v1/sessions", None),
                ("POST", "/v1/agent/turns",
                 {"model": "m1",
                  "messages": [{"role": "user", "content": "hi"}]}),
                ("POST", "/v1/agent/fit", {"model": "m1", "mode": "gpu"}),
            ]:
                status, _headers, raw = _http(server, method, path, body=body)
                failures = _check(failures, f"{method} {path} is a 404",
                                  status == 404, f"{status} {raw[:140]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_serve_shutdown() -> int:
    """begin_shutdown refuses new work; drain is bounded; sessions persist."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        shell = f"touch {Path(tmp) / 'never.txt'}"
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[
                                  [{"message": {"content": "", "tool_calls": [
                                      {"function": {"name": "run_shell",
                                                    "arguments": {
                                                        "command": shell}}}]}},
                                   {"done": True}],
                                  [{"message": {"content": "ok"}, "done": True}],
                              ])
        server, api, store, thread = _start_agent_api(
            tmp, client, manager=orun.TurnManager())
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("shut down",
                                   settings={"approval": "interactive"}))
            turn = json.loads(raw)
            _poll_turn(server, turn["id"], until={"waiting_approval"})

            api.begin_shutdown()
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns", body=_turn_request("later"))
            failures = _check(
                failures, "begin_shutdown refuses a new turn with 503",
                status == 503
                and json.loads(raw)["error"]["code"] == "server_shutting_down",
                f"{status} {raw[:140]}")
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/fits",
                body={"model": "m1", "mode": "gpu"})
            failures = _check(
                failures, "begin_shutdown refuses a new fit with 503",
                status == 503, f"{status} {raw[:140]}")

            started = time.monotonic()
            api.shutdown(timeout=3)
            elapsed = time.monotonic() - started
            snapshot = api.manager.get(turn["id"])
            failures = _check(failures, "shutdown cancels the parked turn",
                              elapsed < 4.0 and snapshot is not None
                              and snapshot.status == "cancelled",
                              f"{elapsed:.2f}s {snapshot}")

            events = _headless_events(api, turn["id"])
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            failures = _check(
                failures, "the parked approval resolved as cancelled",
                resolved and resolved[0].get("note")
                == "cancelled (server shutting down)", str(resolved[:1]))

            session_id = snapshot.session_id if snapshot else ""
            stored = store.resolve(session_id)
            contents = [message.get("content") for message in
                        (stored.messages if stored else [])]
            failures = _check(
                failures, "the cancelled turn's session is on disk",
                stored is not None and "shut down" in contents
                and "ok" not in contents, str(contents)[:200])
            failures = _check(failures, "the drain sees no stray handlers",
                              server.drain(2) is True,
                              f"{server.active_handlers}")
        finally:
            _stop_api(server, thread)
    return failures


def test_serve_unload_on_exit() -> int:
    """Only the models this server used are unloaded; embeddings stay."""
    failures = 0

    class _UnloadClient:
        def __init__(self):
            self.capabilities_map = {"m1": ["completion"],
                                     "embed": ["embedding"]}
            self.resident = {"m1", "embed", "untouched"}
            self.unloaded = []

        def capabilities(self, model):
            return list(self.capabilities_map.get(model, []))

        def is_resident(self, model):
            return model in self.resident

        def unload(self, model):
            self.unloaded.append(model)

    import ollamarun_lib.cli as cli
    fake = _UnloadClient()
    tracker = orun.ModelTracker()
    tracker.note("m1")
    tracker.note("m1")  # deduped
    tracker.note("embed")
    tracker.note("m2")  # not resident
    tracker.note("")
    failures = _check(failures, "the tracker dedupes and keeps order",
                      tracker.models() == ["embed", "m1", "m2"],
                      str(tracker.models()))
    cli._unload_served_models(lambda: fake, tracker)
    failures = _check(
        failures, "unload-on-exit touches only used, resident, non-embedding",
        fake.unloaded == ["m1"], str(fake.unloaded))

    with tempfile.TemporaryDirectory() as tmp:
        # Both surfaces note what they actually used.
        tracker = orun.ModelTracker()
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"])
        server, api, store, thread = _start_agent_api(tmp, client,
                                                      tracker=tracker)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/chat/completions",
                body={"model": "m1",
                      "messages": [{"role": "user", "content": "hi"}]})
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("hi"))
            turn = json.loads(raw)
            _poll_turn(server, turn["id"])
            failures = _check(failures, "chat and turns note their model",
                              tracker.models() == ["m1"],
                              str(tracker.models()))
        finally:
            _stop_api(server, thread)
    return failures


def test_serve_cli() -> int:
    """The `serve` subcommand: parser, config defaults and dispatch."""
    failures = 0
    parser = orun.build_serve_parser()
    args = parser.parse_args(["--bind", "127.0.0.1", "--port", "1234",
                              "--host", "example:1234"])
    failures = _check(failures, "the serve parser takes listener and upstream",
                      args.bind == "127.0.0.1" and args.port == 1234
                      and args.host == "example:1234",
                      str(args))
    config = orun.DEFAULT_CONFIG
    failures = _check(failures, "the LAN defaults are the locked contract",
                      config.get("serve_bind") == "0.0.0.0"
                      and config.get("serve_port") == 11440,
                      f"{config.get('serve_bind')} {config.get('serve_port')}")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            orun.main(["serve", "--help"])
            raised = None
        except SystemExit as exc:
            raised = exc
    failures = _check(failures, "`serve --help` exits 0",
                      raised is not None and raised.code == 0, repr(raised))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "config.json"
        cfg.write_text(json.dumps({"host": "127.0.0.1:1"}), encoding="utf-8")
        with orun.console.capture():
            code = orun.main(["serve", "--config", str(cfg),
                              "--port", "999999"])
        failures = _check(failures, "main() dispatches the serve subcommand",
                          code == 2, str(code))
    return failures


def test_openai_sdk() -> int:
    """The official OpenAI SDK against the local API (a required gate)."""
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        print("FAIL  the openai SDK is not installed (install the test "
              "dependency: pip install --user openai)")
        return 1
    print(f"      the openai SDK {openai.__version__} is under test")
    failures = 0
    script = [[
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " SDK"}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 6,
         "eval_count": 3, "done_reason": "stop"},
    ]]
    client = _ServeClient(models=["m1"],
                          capabilities=["completion", "tools", "embedding"],
                          chat_script=script)
    server, client, thread = _start_api(client)
    api = OpenAI(base_url=f"http://127.0.0.1:{server.address[1]}/v1",
                 api_key="test-key", timeout=10.0)
    try:
        models = api.models.list()
        failures = _check(failures, "the SDK lists models",
                          [model.id for model in models.data] == ["m1"],
                          str(models)[:160])

        embeddings = api.embeddings.create(model="m1", input="hello")
        failures = _check(
            failures, "the SDK embedding round-trips (its default base64)",
            embeddings.object == "list" and embeddings.model == "m1"
            and len(embeddings.data) == 1
            and embeddings.data[0].index == 0
            and embeddings.data[0].embedding == [0.0, 0.5]
            and embeddings.usage.prompt_tokens == 1
            and embeddings.usage.total_tokens == 1,
            str(embeddings)[:200])

        floats = api.embeddings.create(model="m1", input=["a", "b"],
                                       encoding_format="float", dimensions=2)
        failures = _check(
            failures, "the SDK embedding batches floats on request",
            [item.embedding for item in floats.data]
            == [[0.0, 0.5], [1.0, 0.5]],
            str(floats)[:200])

        response = api.chat.completions.create(
            model="m1", temperature=0,
            messages=[{"role": "user", "content": "hi"}])
        failures = _check(failures, "the SDK gets a typed completion",
                          response.choices[0].message.content == "ok"
                          and response.choices[0].finish_reason == "stop"
                          and response.usage.total_tokens == 6,
                          str(response)[:200])

        parts = []
        usage = None
        with api.chat.completions.create(
                model="m1", stream=True,
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"stream_options": {"include_usage": True}}) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                if chunk.usage is not None:
                    usage = chunk.usage
        failures = _check(failures, "the SDK streams deltas and usage",
                          "".join(parts) == "Hello SDK" and usage is not None
                          and usage.prompt_tokens == 6
                          and usage.completion_tokens == 3,
                          f"{parts!r} {usage}")

        client.chat_once_data = {
            "message": {"content": "", "tool_calls": [
                {"function": {"name": "get_weather",
                              "arguments": {"city": "SP"}}}]},
            "done": True, "prompt_eval_count": 8, "eval_count": 4,
            "done_reason": "stop"}
        tool_response = api.chat.completions.create(
            model="m1", temperature=0,
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "w",
                "parameters": {"type": "object"}}}])
        call = tool_response.choices[0].message.tool_calls[0]
        failures = _check(failures, "the SDK receives typed tool calls",
                          tool_response.choices[0].finish_reason == "tool_calls"
                          and call.type == "function"
                          and call.function.name == "get_weather"
                          and json.loads(call.function.arguments)
                          == {"city": "SP"},
                          str(tool_response)[:200])

        client.chat_once_data = None
        multi = api.chat.completions.create(
            model="m1", temperature=0, n=2,
            messages=[{"role": "user", "content": "hi"}])
        failures = _check(
            failures, "the SDK receives n choices with summed usage",
            [choice.index for choice in multi.choices] == [0, 1]
            and all(choice.message.content == "ok" for choice in multi.choices)
            and multi.usage.total_tokens == 12,
            str(multi)[:200])

        def sdk_failure(callable_):
            try:
                callable_()
                return None
            except Exception as exc:  # the test wants the SDK's own type
                return exc

        raised = sdk_failure(lambda: api.chat.completions.create(
            model="m1", n=9, messages=[{"role": "user", "content": "hi"}]))
        failures = _check(failures, "the SDK surfaces rejected fields as 400s",
                          raised is not None
                          and type(raised).__name__ == "BadRequestError"
                          and getattr(raised, "status_code", None) == 400
                          and "limited to 8" in str(raised),
                          repr(raised)[:200])

        raised = sdk_failure(lambda: api.chat.completions.create(
            model="m1", messages=[{"role": "user", "content": "hi"}],
            extra_body={"bogus": True}))
        failures = _check(failures, "the SDK surfaces unknown fields as 400s",
                          raised is not None
                          and type(raised).__name__ == "BadRequestError"
                          and "unknown field: bogus" in str(raised),
                          repr(raised)[:200])

        client.chat_script = [[{"error": "boom"}]]
        client.chat_calls = 0

        def drain() -> None:
            with api.chat.completions.create(
                    model="m1", stream=True,
                    messages=[{"role": "user", "content": "hi"}]) as stream:
                for _chunk in stream:
                    pass

        raised = sdk_failure(drain)
        failures = _check(failures, "a mid-stream backend failure raises APIError",
                          raised is not None
                          and type(raised).__name__ == "APIError"
                          and "the model backend failed" in str(raised),
                          repr(raised)[:200])
    finally:
        _stop_api(server, thread)
    return failures


def _smallest_chat_model(client, entries) -> str:
    """The smallest installed local model that advertises chat completion.

    Cloud-proxied models (`name:cloud`, ~zero on-disk size) are skipped
    unless nothing local exists: they can require paid usage and are not a
    meaningful smoke target.
    """
    def pick(candidates) -> str:
        for entry in candidates:
            name = entry.get("name") or entry.get("model") or ""
            if name and "completion" in client.capabilities(name):
                return str(name)
        return ""

    ordered = sorted((entry for entry in entries if isinstance(entry, dict)),
                     key=lambda entry: entry.get("size") or 0)
    local = [entry for entry in ordered
             if (entry.get("size") or 0) > 1024 * 1024
             and not str(entry.get("name") or "").endswith(":cloud")]
    return pick(local) or pick(ordered)


def test_serve_live() -> int:
    """Real-backend smoke test: only runs with `--live`."""
    failures = 0
    host = os.environ.get("OLLAMA_HOST") or orun.DEFAULT_HOST
    client = orun.OllamaClient(host, timeout=600)
    try:
        entries = client.tags()
    except orun.OllamaError as exc:
        print(f"FAIL  --live: Ollama at {host} is unreachable: {exc}")
        return 1
    model = os.environ.get("OLLAMARUN_TEST_MODEL") \
        or _smallest_chat_model(client, entries)
    if not model:
        print("FAIL  --live: no chat-capable model found on the backend")
        return 1
    print(f"      --live: using {model} on {host}")
    embed_model = os.environ.get("OLLAMARUN_TEST_EMBED_MODEL") or ""
    if not embed_model:
        for entry in entries:
            name = str(entry.get("name") or "")
            if name and "embedding" in client.capabilities(name):
                embed_model = name
                break
    server, _client, thread = _start_api(
        factory=lambda: orun.OllamaClient(host, timeout=600))
    try:
        status, _headers, raw = _http(server, "GET", "/v1/models", timeout=30)
        listed = {entry.get("id") for entry in json.loads(raw).get("data", [])}
        failures = _check(failures, "live: the model list includes the model",
                          status == 200 and model in listed, raw[:160])

        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": model, "temperature": 0, "max_tokens": 16,
                  "messages": [{"role": "user",
                                "content": "Reply with exactly: ok"}]},
            timeout=600)
        body = json.loads(raw)
        choices = body.get("choices") or []
        content = (choices[0]["message"].get("content") or "") if choices else ""
        usage = body.get("usage") or {}
        failures = _check(failures, "live: a completion returns content and usage",
                          status == 200 and bool(content.strip())
                          and (usage.get("total_tokens") or 0) > 0, raw[:300])

        status, _headers, raw = _http(
            server, "POST", "/v1/chat/completions",
            body={"model": model, "temperature": 0, "max_tokens": 16,
                  "stream": True,
                  "messages": [{"role": "user",
                                "content": "Reply with exactly: ok"}]},
            timeout=600)
        events = _sse_events(raw)
        streamed = [chunk for chunk in (json.loads(item) for item in events
                                        if item != "[DONE]")
                    if chunk.get("choices")
                    and chunk["choices"][0]["delta"].get("content")]
        failures = _check(failures, "live: streaming ends with [DONE]",
                          status == 200 and bool(events)
                          and events[-1] == "[DONE]" and bool(streamed),
                          str(events[-2:])[:200])

        if embed_model:
            status, _headers, raw = _http(
                server, "POST", "/v1/embeddings",
                body={"model": embed_model, "input": ["hello", "world"]},
                timeout=600)
            body = json.loads(raw)
            vectors = [item.get("embedding") for item in body.get("data", [])]
            dims = [len(vector) for vector in vectors
                    if isinstance(vector, list)]
            failures = _check(
                failures, "live: embeddings round-trip",
                status == 200 and len(vectors) == 2
                and all(isinstance(vector, list) and vector
                        for vector in vectors)
                and (body.get("usage") or {}).get("prompt_tokens", 0) > 0,
                raw[:240])
            print(f"      --live: {embed_model} embeddings dims={dims}")
        else:
            print("      --live: no embedding-capable model is installed; "
                  "skipping the embeddings check")

        if os.environ.get("OLLAMARUN_TEST_MODEL"):
            status, _headers, raw = _http(
                server, "POST", "/v1/chat/completions",
                body={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content":
                          "What is the weather in Sao Paulo? "
                          "Use the get_weather tool."}],
                      "tools": [{"type": "function", "function": {
                          "name": "get_weather", "description": "weather",
                          "parameters": {"type": "object", "properties": {
                              "city": {"type": "string"}},
                              "required": ["city"]}}}]},
                timeout=600)
            body = json.loads(raw)
            choices = body.get("choices") or []
            calls = (choices[0]["message"].get("tool_calls") or []) if choices else []
            failures = _check(failures, "live: the tool call round-trips",
                              status == 200 and bool(choices)
                              and choices[0].get("finish_reason") == "tool_calls"
                              and bool(calls)
                              and calls[0]["function"]["name"] == "get_weather",
                              raw[:300])

        # A full agent turn against the real backend, on a temp session store.
        # Tools ride along only when the model advertises them -- the backend
        # rejects tools outright for models without support -- while the live
        # tool pass-through above runs under OLLAMARUN_TEST_MODEL.
        with tempfile.TemporaryDirectory() as tmp:
            supports_tools = "tools" in client.capabilities(model)
            live_client = orun.OllamaClient(host, timeout=600)
            agent_server, agent_api, store, agent_thread = _start_agent_api(
                tmp, live_client)
            try:
                status, _headers, raw = _http(
                    agent_server, "POST", "/v1/agent/turns",
                    body={"model": model,
                          "settings": {"tools": supports_tools,
                                       "parameters": {"num_predict": 24}},
                          "messages": [{"role": "user",
                                        "content": "Reply with exactly: ok"}]},
                    timeout=600)
                turn = json.loads(raw)
                status, body = _poll_turn(agent_server, turn["id"], timeout=600)
                stored = store.resolve(turn["session_id"])
                roles = [message.get("role") for message in
                         (stored.messages if stored else [])]
                events = _headless_events(agent_api, turn["id"])
                types = [event.type for event in events]
                failures = _check(
                    failures, "live: a full agent turn completes and persists",
                    status == 200 and body.get("status") == "completed"
                    and bool(((body.get("result") or {}).get("content") or "").strip())
                    and types[-1:] == ["turn.completed"]
                    and roles[:1] == ["user"] and roles[-1:] == ["assistant"],
                    f"{body.get('status')} {types[:2]}..{types[-1:]} {roles}")
                print(f"      --live: agent turn {turn['id']} "
                      f"tools={supports_tools} "
                      f"rounds={(body.get('result') or {}).get('rounds')}")
            finally:
                _stop_api(agent_server, agent_thread)
    finally:
        _stop_api(server, thread)
    return failures


def test_serve_process() -> int:
    """The real entry point: serve, answer, drain a stream on SIGTERM, exit 0.

    Everything else in this file drives an in-process `HttpServer` over a
    fake backend; this is the one check that the shipped `ollamarun serve`
    binary wires config, banner, signal handling, drain and the tracked
    unload together — against a stub Ollama on a loopback port.
    """
    import select
    import signal
    import subprocess
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    failures = 0
    seen: list = []
    lock = threading.Lock()

    class Stub(BaseHTTPRequestHandler):
        chunks = 20
        chunk_delay = 0.05

        def log_message(self, *args):  # keep the test output quiet
            pass

        def _record(self, payload):
            with lock:
                seen.append((self.command, self.path, payload))

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw)
            except ValueError:
                return {}

        def _json(self, payload, status=200):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._record(None)
            if self.path == "/api/tags":
                self._json({"models": [{
                    "name": "m1", "modified_at": "2026-01-01T00:00:00Z"}]})
            elif self.path == "/api/ps":
                self._json({"models": [{"name": "m1"}]})
            else:
                self._json({"error": "not found"}, status=404)

        def do_POST(self):
            payload = self._body()
            self._record(payload)
            if self.path == "/api/show":
                self._json({"capabilities": ["completion", "tools"],
                            "model_info": {"llama.context_length": 8192}})
            elif self.path == "/api/generate":
                self._json({})
            elif self.path == "/api/chat" and payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for _ in range(self.chunks):
                    self.wfile.write(json.dumps(
                        {"message": {"content": "x"}, "done": False}).encode()
                        + b"\n")
                    self.wfile.flush()
                    time.sleep(self.chunk_delay)
                self.wfile.write(json.dumps(
                    {"done": True, "prompt_eval_count": 1,
                     "eval_count": self.chunks}).encode() + b"\n")
                self.wfile.flush()
            elif self.path == "/api/chat":
                self._json({"message": {"content": "ok"}, "done": True,
                            "prompt_eval_count": 1, "eval_count": 1,
                            "done_reason": "stop"})
            else:
                self._json({"error": "not found"}, status=404)

    stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    stub_thread = threading.Thread(target=stub.serve_forever, daemon=True)
    stub_thread.start()
    proc = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({
                "host": f"127.0.0.1:{stub.server_address[1]}",
                "serve_artifact_root": str(Path(tmp) / "artifacts"),
                "serve_workspace": tmp,
                "serve_unload_on_exit": True,
                "serve_shutdown_timeout": 5,
            }), encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable,
                 str(Path(__file__).resolve().parent / "ollamarun"),
                 "serve", "--config", str(config),
                 "--bind", "127.0.0.1", "--port", "0"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})

            port = None
            banner = []
            deadline = time.monotonic() + 30
            while port is None and time.monotonic() < deadline \
                    and proc.poll() is None:
                ready, _w, _x = select.select([proc.stdout], [], [], 0.5)
                if not ready:
                    continue
                line = proc.stdout.readline()
                if not line:
                    break
                banner.append(line)
                match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
                if match:
                    port = int(match.group(1))
            failures = _check(failures,
                              "the real serve process prints its address",
                              port is not None, "".join(banner)[:200])

            if port is not None:
                def api(method, path, body=None, timeout=30):
                    conn = http.client.HTTPConnection("127.0.0.1", port,
                                                      timeout=timeout)
                    try:
                        payload = json.dumps(body) if body is not None else None
                        headers = ({"Content-Type": "application/json"}
                                   if payload is not None else {})
                        conn.request(method, path, body=payload, headers=headers)
                        response = conn.getresponse()
                        return response.status, response.read()
                    finally:
                        conn.close()

                status, raw = api("GET", "/v1/models")
                failures = _check(failures, "the real process lists models",
                                  status == 200
                                  and json.loads(raw)["data"][0]["id"] == "m1",
                                  f"{status} {raw[:160]}")

                status, raw = api("POST", "/v1/chat/completions", {
                    "model": "m1", "temperature": 0,
                    "messages": [{"role": "user", "content": "hi"}]})
                body = json.loads(raw)
                content = (body.get("choices") or [{}])[0].get(
                    "message", {}).get("content")
                failures = _check(failures, "the real process answers a completion",
                                  status == 200 and content == "ok",
                                  f"{status} {raw[:160]}")

                # SIGTERM mid-stream: the handler must drain, so the client
                # still reads the whole SSE body and the process exits 0.
                streamed = {}

                def read_stream():
                    conn = http.client.HTTPConnection("127.0.0.1", port,
                                                      timeout=30)
                    try:
                        conn.request("POST", "/v1/chat/completions",
                                     body=json.dumps({
                                         "model": "m1", "stream": True,
                                         "messages": [{"role": "user",
                                                       "content": "hi"}]}),
                                     headers={"Content-Type": "application/json"})
                        response = conn.getresponse()
                        streamed["status"] = response.status
                        streamed["body"] = response.read()
                    except Exception as exc:  # the harness reports it
                        streamed["error"] = repr(exc)
                    finally:
                        conn.close()

                reader = threading.Thread(target=read_stream, daemon=True)
                reader.start()
                # Wait until the stream has actually started, then stop the
                # server a few chunks in: the drain must let it finish.
                deadline = time.monotonic() + 10
                while "status" not in streamed and "error" not in streamed \
                        and time.monotonic() < deadline:
                    time.sleep(0.02)
                time.sleep(0.2)
                proc.send_signal(signal.SIGTERM)
                reader.join(30)
                failures = _check(
                    failures, "the drain lets an in-flight stream finish",
                    streamed.get("status") == 200
                    and b"[DONE]" in streamed.get("body", b""),
                    str(streamed)[:200])

            out = "".join(banner) + (proc.stdout.read() or "")
            err = proc.stderr.read() or ""
            code = proc.wait(timeout=30)
            failures = _check(failures, "SIGTERM stops the process cleanly",
                              code == 0 and "Traceback" not in err
                              and "stopped" in out,
                              f"rc={code} err={err[-200:]!r}")
            unload = [payload for method, path, payload in seen
                      if path == "/api/generate"]
            failures = _check(
                failures, "the real process unloads only the used model",
                bool(unload) and unload[-1].get("model") == "m1"
                and unload[-1].get("keep_alive") == 0
                and "unloaded m1" in out,
                f"{unload[-1:]!r} {out[-200:]!r}")
            if port is not None:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=1)
                    released = False
                except OSError:
                    released = True
                failures = _check(failures, "the listener port is released",
                                  released, str(port))
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        stub.shutdown()
        stub.server_close()
        stub_thread.join(timeout=5)
    return failures


def _start_agent_api(tmp, client, *, config_extra=None, manager=None,
                     store=None, args_extra=(), config=None, artifacts=None,
                     agent_routes=True, tracker=None):
    """An API server with the native agent routes over a fake backend."""
    if store is None:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
    settings = dict(orun.DEFAULT_CONFIG)
    settings["tools"] = True
    settings.update(config or {})
    settings.update(config_extra or {})
    args = orun.build_parser().parse_args(
        ["m1", "--render", "plain", "--no-save", *args_extra])
    router = orun.build_router(lambda: client, config=settings,
                               tracker=tracker)
    api = orun.build_agent_api(args=args, config=settings, store=store,
                               client_factory=lambda: client,
                               manager=manager,
                               cwd=settings.get("serve_workspace"),
                               artifacts=artifacts, tracker=tracker)
    orun.add_artifact_routes(router, api)
    if agent_routes:
        orun.add_agent_routes(router, api)
    server = orun.HttpServer(router, bind="127.0.0.1", port=0, ping_interval=0.2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, api, store, thread


def _poll_turn(server, turn_id, *, until=(), timeout=20.0):
    """Poll the turn status endpoint until terminal, `until`, or the timeout."""
    deadline = time.monotonic() + timeout
    status, body = 0, {}
    while time.monotonic() < deadline:
        status, _headers, raw = _http(server, "GET", f"/v1/agent/turns/{turn_id}")
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"raw": raw[:200]}
        current = body.get("status")
        if status != 200 or current in until \
                or current in orun.TERMINAL_STATUSES:
            return status, body
        time.sleep(0.05)
    return status, body


def _turn_request(messages, *, session_id=None, model="m1", settings=None,
                  persist=True):
    body = {"model": model, "persist": persist,
            "messages": [{"role": "user", "content": messages}]
            if isinstance(messages, str) else messages}
    if session_id:
        body["session_id"] = session_id
    if settings is not None:
        body["settings"] = settings
    return body


def _headless_events(api, turn_id, after=0):
    """The buffered turn events, read straight from the manager."""
    return _collect(api.manager.subscribe(turn_id, after=after))


def test_examples() -> int:
    """Both reference clients run green against the fake stack (no rot)."""
    import subprocess

    failures = 0
    repo = Path(__file__).resolve().parent

    client = _ServeClient(models=["m1"],
                          capabilities=["completion", "tools", "embedding"],
                          chat_script=[[
                              {"message": {"content": "Hello"}, "done": False},
                              {"message": {"content": ""}, "done": True,
                               "prompt_eval_count": 3, "eval_count": 1,
                               "done_reason": "stop"}]])
    server, _client, thread = _start_api(client)
    try:
        result = subprocess.run(
            [sys.executable, str(repo / "examples" / "openai_client.py"),
             "--base-url", f"http://127.0.0.1:{server.address[1]}/v1",
             "--model", "m1"],
            capture_output=True, text=True, timeout=120, cwd=str(repo))
        out = _plain(result.stdout)
        failures = _check(
            failures, "the OpenAI reference client runs green",
            result.returncode == 0
            and "all checks passed" in out
            and "a non-streaming completion" in out
            and "streaming deltas" in out
            and "embeddings return one vector per input" in out
            and "n=2 returns two indexed choices" in out,
            f"rc={result.returncode} out={out[-300:]!r} "
            f"err={result.stderr[-200:]!r}")
    finally:
        _stop_api(server, thread)

    with tempfile.TemporaryDirectory() as tmp:
        agent_client = _ServeClient(
            models=["m1"], capabilities=["completion", "tools"],
            chat_script=[
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": "echo example"}}}]}},
                 {"message": {"content": ""}, "done": True}],
                [{"message": {"content": "done"}, "done": True}],
            ])
        server2, _api2, _store2, thread2 = _start_agent_api(tmp, agent_client)
        try:
            result = subprocess.run(
                [sys.executable, str(repo / "examples" / "agent_client.py"),
                 "--base-url", f"http://127.0.0.1:{server2.address[1]}",
                 "--model", "m1", "--prompt", "run it"],
                capture_output=True, text=True, timeout=120, cwd=str(repo))
            out = _plain(result.stdout)
            failures = _check(
                failures, "the agent reference client runs green",
                result.returncode == 0
                and "status completed" in out
                and "tool · run_shell" in out
                and "answer done" in out,
                f"rc={result.returncode} out={out[-300:]!r} "
                f"err={result.stderr[-200:]!r}")
        finally:
            _stop_api(server2, thread2)
    return failures


def test_turn_mode_isolation() -> int:
    """The same tool call: returned to the client vs executed by the agent."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "executed.txt"
        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"],
                              show_data={"model_info": {
                                  "llama.context_length": 8192}})
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            # A model turn: the call is translated, never executed.
            client.chat_once_data = {
                "message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": f"touch {marker}"}}}]},
                "done": True, "prompt_eval_count": 5, "eval_count": 2,
                "done_reason": "stop"}
            status, _headers, raw = _http(
                server, "POST", "/v1/chat/completions", body={
                    "model": "m1",
                    "messages": [{"role": "user", "content": "touch it"}],
                    "tools": [{"type": "function", "function": {
                        "name": "run_shell", "description": "shell",
                        "parameters": {"type": "object", "properties": {
                            "command": {"type": "string"}},
                            "required": ["command"]}}}]})
            body = json.loads(raw)
            choices = body.get("choices") or []
            calls = (choices[0]["message"].get("tool_calls") or []) \
                if choices else []
            failures = _check(
                failures, "a model turn returns the call and never runs it",
                status == 200 and bool(calls)
                and choices[0].get("finish_reason") == "tool_calls"
                and calls[0]["function"]["name"] == "run_shell"
                and not marker.exists()
                and len(client.chat_once_payloads) == 1
                and client.chat_calls == 0,
                f"{status} {raw[:200]} marker={marker.exists()}")

            # An agent turn: the same tool executes inside the loop.
            client.chat_once_data = None
            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": "echo mode-agent"}}}]}},
                 {"done": True}],
                [{"message": {"content": "ran it"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns", body=_turn_request("run it"))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"])
            results = [event.data for event in _headless_events(api, turn["id"])
                       if event.type == "tool.result"]
            stored = store.resolve(turn["session_id"])
            messages = stored.messages if stored else []
            roles = [message.get("role") for message in messages]
            failures = _check(
                failures, "the agent turn executes the call through the loop",
                status == 200 and body.get("status") == "completed"
                and (body.get("result") or {}).get("content") == "ran it"
                and client.chat_calls == 2
                and results and results[0].get("ok") is True
                and "mode-agent" in str(results[0].get("text"))
                and roles == ["user", "assistant", "tool", "assistant"]
                and any("mode-agent" in str(message.get("content"))
                        for message in messages),
                f"{body.get('status')} {results[:1]} {roles}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_turn_lifecycle() -> int:
    """POST a turn, watch it finish, replay its events and read the session."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        script = [[{"message": {"content": "hello"}, "done": True,
                    "prompt_eval_count": 5, "eval_count": 2,
                    "done_reason": "stop"}]]
        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"],
                              show_data={"model_info": {"llama.context_length": 8192},
                                         "parameters": "num_ctx 8192"},
                              chat_script=script)
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("hi"))
            started = json.loads(raw)
            failures = _check(failures, "a turn starts with 202 and its urls",
                              status == 202 and started["id"].startswith("turn_")
                              and started["session_id"]
                              and started["status_url"].endswith(started["id"])
                              and started["events_url"].endswith("/events")
                              and started["cancel_url"].endswith("/cancel"),
                              f"{status} {raw[:160]!r}")
            turn_id = started["id"]

            status, body = _poll_turn(server, turn_id)
            result = body.get("result") or {}
            failures = _check(failures, "the turn completes with the answer",
                              status == 200 and body["status"] == "completed"
                              and result.get("content") == "hello"
                              and result.get("rounds") == 0
                              and result.get("usage", {}).get("total_tokens") == 7
                              and body.get("error") is None,
                              str(body)[:240])
            failures = _check(failures, "the snapshot carries seq and context",
                              body.get("seq", 0) >= 2
                              and isinstance(body.get("context"), dict)
                              and body["context"].get("total") == 8192
                              and isinstance(body["context"].get("source"), str),
                              str(body.get("context")))

            events = _headless_events(api, turn_id)
            types = [event.type for event in events]
            failures = _check(failures, "the event stream is enveloped and ordered",
                              types[0] == "turn.started"
                              and "message.delta" in types
                              and types[-1] == "turn.completed"
                              and events[0].data.get("context") is not None
                              and [event.seq for event in events]
                              == list(range(1, len(events) + 1)),
                              str(types))

            status, headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn_id}/events?after=0")
            sse = _sse_events(raw)
            payloads = [json.loads(item) for item in sse]
            id_lines = [line for line in raw.decode().splitlines()
                        if line.startswith("id: ")]
            failures = _check(failures, "the SSE endpoint replays the typed stream",
                              status == 200 and payloads
                              and payloads[0]["type"] == "turn.started"
                              and payloads[-1]["type"] == "turn.completed"
                              and len(id_lines) == len(payloads)
                              and id_lines[0].endswith(":1"),
                              f"{status} {sse[:2]}")

            status, headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn_id}/events",
                headers={"Last-Event-ID": f"{turn_id}:1"})
            sse = _sse_events(raw)
            payloads = [json.loads(item) for item in sse]
            failures = _check(failures,
                              "Last-Event-ID resumes after the given seq",
                              status == 200 and payloads
                              and payloads[0]["seq"] == 2
                              and payloads[-1]["type"] == "turn.completed",
                              f"{status} {sse[:2]}")
            status, _headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn_id}/events",
                headers={"Last-Event-ID": "nonsense"})
            failures = _check(failures, "a malformed Last-Event-ID is a 400",
                              status == 400, f"{status} {raw[:160]}")

            status, _headers, raw = _http(server, "GET", "/v1/sessions")
            listed = json.loads(raw)["data"]
            failures = _check(failures, "the turn created a persisted session",
                              status == 200
                              and any(entry["id"] == started["session_id"]
                                      for entry in listed),
                              raw[:200])
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{started['session_id']}")
            session = json.loads(raw)
            roles = [message.get("role") for message in session.get("history", [])]
            failures = _check(failures, "the session history keeps the exchange",
                              status == 200 and roles == ["user", "assistant"]
                              and session["history"][1]["content"] == "hello"
                              and session["turns"] == 1,
                              str(session.get("history"))[:200])

            # A backend failure mid-turn: failed status, terminal turn.error
            # and a session that keeps only what the turn actually saw.
            client.chat_script = [[{"error": "boom"}]]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns", body=_turn_request("break"))
            broken = json.loads(raw)
            status, body = _poll_turn(server, broken["id"])
            failures = _check(failures, "a backend failure fails the turn",
                              status == 200 and body["status"] == "failed"
                              and body.get("error") == "boom"
                              and not (body.get("result") or {}).get("content"),
                              str(body)[:240])
            events = _headless_events(api, broken["id"])
            types = [event.type for event in events]
            failures = _check(failures, "the failure is a terminal turn.error",
                              types and types[-1] == "turn.error"
                              and events[-1].data.get("message") == "boom",
                              f"{types} {events[-1].data}")
            stored = store.resolve(broken["session_id"])
            contents = [message.get("content") for message in
                        (stored.messages if stored else [])]
            failures = _check(failures, "the failed session keeps the user turn only",
                              stored is not None and contents == ["break"],
                              str(contents)[:200])
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_turn_controls() -> int:
    """409 busy, 429 cap, cancel and disconnect policies."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chunk_count=400, chunk_delay=0.01)
        manager = orun.TurnManager(max_active=1)
        server, api, store, thread = _start_agent_api(tmp, client,
                                                      manager=manager)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("slow one"))
            first = json.loads(raw)
            session_id = first["session_id"]
            time.sleep(0.2)

            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("busy",
                                                             session_id=session_id))
            busy = json.loads(raw)["error"]
            failures = _check(failures, "a busy session is a 409 with the active id",
                              status == 409 and busy["code"] == "session_busy"
                              and busy["active_turn_id"] == first["id"]
                              and busy["type"] == "conflict_error",
                              f"{status} {raw[:200]}")
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("too many"))
            failures = _check(failures, "the global cap is a 429",
                              status == 429
                              and json.loads(raw)["error"]["code"] == "rate_limit_error",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "DELETE", f"/v1/sessions/{session_id}")
            failures = _check(failures, "a session with an active turn is protected",
                              status == 409, f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"title": "too early"})
            error = json.loads(raw).get("error", {})
            failures = _check(failures, "a session with an active turn is not editable",
                              status == 409
                              and error.get("code") == "session_busy"
                              and error.get("active_turn_id") == first["id"],
                              f"{status} {raw[:200]}")

            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/turns/{first['id']}/cancel")
            failures = _check(failures, "cancel is accepted with 202",
                              status == 202
                              and json.loads(raw)["cancel_requested"] is True,
                              f"{status} {raw[:160]}")
            status, body = _poll_turn(server, first["id"])
            failures = _check(failures, "a cancelled turn is terminal-cancelled",
                              status == 200 and body["status"] == "cancelled"
                              and (body.get("result") or {}).get("status") == "cancelled",
                              str(body)[:200])
            events = _headless_events(api, first["id"])
            cancelled = [event for event in events
                         if event.type == "turn.cancelled"]
            failures = _check(failures, "the cancellation reason is user",
                              cancelled and cancelled[0].data.get("reason") == "user",
                              str([event.data for event in cancelled]))
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{session_id}")
            history = json.loads(raw)["history"]
            partial = history[1].get("content", "") if len(history) > 1 else ""
            failures = _check(
                failures, "the cancelled turn's session keeps its shape",
                status == 200
                and [message.get("role") for message in history]
                == ["user", "assistant"]
                and history[0].get("content") == "slow one"
                and partial and set(partial) == {"x"}
                and history[1].get("cancelled") is True,
                str(history)[:200])
            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/turns/{first['id']}/cancel")
            failures = _check(failures, "cancelling a finished turn is a 409",
                              status == 409
                              and json.loads(raw)["error"]["code"] == "turn_finished",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"title": "after the turn"})
            failures = _check(failures, "editing works again once the turn is done",
                              status == 200
                              and json.loads(raw)["title"] == "after the turn",
                              f"{status} {raw[:160]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_approvals() -> int:
    """Interactive approvals: run, remember, deny and duplicate decisions."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "approved.txt"
        shell = f"touch {marker}"
        script = [
            [{"message": {"content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": shell}}}]}},
             {"done": True, "prompt_eval_count": 5, "eval_count": 2}],
            [{"message": {"content": "done"}, "done": True,
              "prompt_eval_count": 7, "eval_count": 1}],
        ]
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=script)
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("run it", settings={"approval": "interactive"}))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"],
                                      until={"waiting_approval"})
            failures = _check(failures, "the turn parks in waiting_approval",
                              status == 200 and body["status"] == "waiting_approval",
                              str(body)[:200])
            events = _headless_events(api, turn["id"])
            required = [event for event in events
                        if event.type == "approval.required"]
            notice = required[0].data if required else {}
            failures = _check(failures, "approval.required carries the choices",
                              bool(required)
                              and notice.get("approval_id", "").startswith("apr_")
                              and notice.get("name") == "run_shell"
                              and notice.get("detail")
                              and notice.get("reason")
                              and isinstance(notice.get("suggestions"), list),
                              str(notice)[:200])
            approval_id = notice.get("approval_id", "")

            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/approvals/{approval_id}",
                body={"action": "run"})
            decided = json.loads(raw)
            failures = _check(failures, "an approval resolves with run",
                              status == 200 and decided["run"] is True
                              and decided["action"] == "run",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/approvals/{approval_id}",
                body={"action": "deny"})
            failures = _check(failures, "a duplicate decision is rejected",
                              status == 409
                              and json.loads(raw)["error"]["code"] == "already_resolved",
                              f"{status} {raw[:160]}")
            status, body = _poll_turn(server, turn["id"])
            failures = _check(failures, "the approved tool ran and the turn finished",
                              status == 200 and body["status"] == "completed"
                              and marker.exists()
                              and (body.get("result") or {}).get("content") == "done",
                              f"{body.get('status')} marker={marker.exists()}")

            # Remember the exact call, then a second identical turn must not
            # park at all.
            marker2 = Path(tmp) / "remembered.txt"
            shell2 = f"touch {marker2}"
            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": shell2}}}]}},
                 {"done": True}],
                [{"message": {"content": "ok"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("again", settings={"approval": "interactive"}))
            turn2 = json.loads(raw)
            _poll_turn(server, turn2["id"], until={"waiting_approval"})
            events = _headless_events(api, turn2["id"])
            notice2 = [event.data for event in events
                       if event.type == "approval.required"][0]
            status, _headers, raw = _http(
                server, "POST",
                f"/v1/agent/approvals/{notice2['approval_id']}",
                body={"action": "remember"})
            status, body = _poll_turn(server, turn2["id"])
            failures = _check(failures, "remember executes this time",
                              body.get("status") == "completed" and marker2.exists(),
                              f"{body.get('status')} marker={marker2.exists()}")

            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": shell2}}}]}},
                 {"done": True}],
                [{"message": {"content": "ok"}, "done": True}],
            ]
            client.chat_calls = 0
            session_id = turn2["session_id"]
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("third", session_id=session_id,
                                   settings={"approval": "interactive"}))
            turn3 = json.loads(raw)
            status, body = _poll_turn(server, turn3["id"],
                                      until={"waiting_approval"})
            failures = _check(failures, "a remembered call never parks again",
                              body.get("status") == "completed"
                              and marker2.exists(),
                              str(body)[:200])

            # Deny path in a separate session.
            marker4 = Path(tmp) / "denied.txt"
            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": f"touch {marker4}"}}}]}},
                 {"done": True}],
                [{"message": {"content": "ok"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("deny it",
                                   settings={"approval": "interactive"}))
            turn4 = json.loads(raw)
            _poll_turn(server, turn4["id"], until={"waiting_approval"})
            events = _headless_events(api, turn4["id"])
            notice4 = [event.data for event in events
                       if event.type == "approval.required"][0]
            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/approvals/{notice4['approval_id']}",
                body={"action": "deny"})
            status, body = _poll_turn(server, turn4["id"])
            results = [event.data for event in _headless_events(api, turn4["id"])
                       if event.type == "tool.result"]
            failures = _check(failures, "deny reports a not-run tool result",
                              body.get("status") == "completed"
                              and not marker4.exists() and bool(results)
                              and results[0].get("ok") is False
                              and "tool not run" in str(results[0].get("text")),
                              f"{body.get('status')} {results[:1]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_approval_edge() -> int:
    """Approval timeout, cancellation while waiting and shutdown while waiting."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        shell = f"touch {Path(tmp) / 'never.txt'}"
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[
                                  [{"message": {"content": "", "tool_calls": [
                                      {"function": {"name": "run_shell",
                                                    "arguments": {"command": shell}}}]}},
                                   {"done": True}],
                                  [{"message": {"content": "ok"}, "done": True}],
                              ])
        server, api, store, thread = _start_agent_api(
            tmp, client, config_extra={"serve_approval_timeout": 0.4},
            manager=orun.TurnManager())
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("timeout me",
                                   settings={"approval": "interactive"}))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"], timeout=10)
            events = _headless_events(api, turn["id"])
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            results = [event.data for event in events
                       if event.type == "tool.result"]
            failures = _check(failures, "an unanswered approval times out as denied",
                              body.get("status") == "completed"
                              and resolved and resolved[0].get("action") == "timeout"
                              and results and results[0].get("ok") is False
                              and "approval timed out" in str(results[0].get("text")),
                              f"{body.get('status')} {resolved[:1]} {results[:1]}")

            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": shell}}}]}},
                 {"done": True}],
                [{"message": {"content": "ok"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("cancel me",
                                   settings={"approval": "interactive"}))
            turn2 = json.loads(raw)
            _poll_turn(server, turn2["id"], until={"waiting_approval"})
            status, _headers, raw = _http(
                server, "POST", f"/v1/agent/turns/{turn2['id']}/cancel")
            status, body = _poll_turn(server, turn2["id"])
            events = _headless_events(api, turn2["id"])
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            failures = _check(failures, "cancelling while waiting resolves the approval",
                              body.get("status") == "cancelled"
                              and resolved
                              and resolved[0].get("action") == "cancelled"
                              and resolved[0].get("note") == "cancelled by user",
                              f"{body.get('status')} {resolved[:1]}")

            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": shell}}}]}},
                 {"done": True}],
                [{"message": {"content": "ok"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("shut down",
                                   settings={"approval": "interactive"}))
            turn3 = json.loads(raw)
            _poll_turn(server, turn3["id"], until={"waiting_approval"})
            started = time.monotonic()
            api.shutdown(timeout=3)
            elapsed = time.monotonic() - started
            snapshot = api.manager.get(turn3["id"])
            failures = _check(failures, "shutdown while waiting stops the turn",
                              elapsed < 4.0 and snapshot is not None
                              and snapshot.status == "cancelled",
                              f"{elapsed:.2f}s {snapshot}")
            events = _headless_events(api, turn3["id"])
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            failures = _check(failures,
                              "shutdown says why the approval was cancelled",
                              resolved and resolved[0].get("note")
                              == "cancelled (server shutting down)",
                              str(resolved[:1]))
        finally:
            _stop_api(server, thread)
    return failures


def test_serve_approval_timeout_zero() -> int:
    """serve_approval_timeout: 0 disables the timeout (0 is not "unset")."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        shell = f"touch {Path(tmp) / 'never.txt'}"
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[
                                  [{"message": {"content": "", "tool_calls": [
                                      {"function": {"name": "run_shell",
                                                    "arguments": {"command": shell}}}]}},
                                   {"done": True}],
                                  [{"message": {"content": "ok"}, "done": True}],
                              ])
        server, api, store, thread = _start_agent_api(
            tmp, client, config_extra={"serve_approval_timeout": 0},
            manager=orun.TurnManager())
        try:
            failures = _check(failures,
                              "an explicit 0 disables the approval timeout",
                              api._approval_timeout == 0.0,
                              repr(api._approval_timeout))
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("wait forever",
                                   settings={"approval": "interactive"}))
            turn = json.loads(raw)
            _poll_turn(server, turn["id"], until={"waiting_approval"})
            time.sleep(1.0)
            events = _headless_events(api, turn["id"])
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            failures = _check(failures,
                              "a zero-timeout approval does not resolve itself",
                              not resolved, str(resolved[:1]))
            # Clean up: deny the parked call so the turn can finish.
            notice = [event.data for event in events
                      if event.type == "approval.required"][0]
            _http(server, "POST", f"/v1/agent/approvals/{notice['approval_id']}",
                  body={"action": "deny"})
            _poll_turn(server, turn["id"])
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_policy_mode() -> int:
    """The default mode runs allowlisted calls and denies the rest with a note."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "policy.txt"
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[
                                  [{"message": {"content": "", "tool_calls": [
                                      {"function": {"name": "run_shell",
                                                    "arguments": {"command": "echo safe"}}}]}},
                                   {"done": True}],
                                  [{"message": {"content": "safe"}, "done": True}],
                              ])
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("safe"))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"])
            results = [event.data for event in _headless_events(api, turn["id"])
                       if event.type == "tool.result"]
            failures = _check(failures, "an allowlisted call runs without parking",
                              body.get("status") == "completed"
                              and bool(results) and results[0].get("ok") is True,
                              f"{body.get('status')} {results[:1]}")

            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "run_shell",
                                  "arguments": {"command": f"touch {marker}"}}}]}},
                 {"done": True}],
                [{"message": {"content": "denied"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("risky"))
            turn2 = json.loads(raw)
            status, body = _poll_turn(server, turn2["id"])
            events = _headless_events(api, turn2["id"])
            types = [event.type for event in events]
            resolved = [event.data for event in events
                        if event.type == "approval.resolved"]
            results = [event.data for event in events
                       if event.type == "tool.result"]
            failures = _check(failures, "a risky call is denied by policy, never parked",
                              body.get("status") == "completed"
                              and "approval.required" not in types
                              and not marker.exists()
                              and resolved and resolved[0].get("action") == "deny"
                              and "not allowed by policy" in str(results[0].get("text")),
                              f"{types} {results[:1]} marker={marker.exists()}")

            # A tool that errors: an ok:false result, and the model continues.
            client.chat_script = [
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "read_file",
                                  "arguments": {"path": "definitely-missing.txt"}}}]}},
                 {"done": True}],
                [{"message": {"content": "handled"}, "done": True}],
            ]
            client.chat_calls = 0
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("missing"))
            turn3 = json.loads(raw)
            status, body = _poll_turn(server, turn3["id"])
            results = [event.data for event in _headless_events(api, turn3["id"])
                       if event.type == "tool.result"]
            failures = _check(
                failures, "a failing tool is ok:false and the model continues",
                body.get("status") == "completed"
                and (body.get("result") or {}).get("content") == "handled"
                and client.chat_calls == 2
                and results and results[0].get("ok") is False
                and "not a file" in str(results[0].get("text")),
                f"{body.get('status')} {results[:1]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_disconnect_policy() -> int:
    """A dropped event stream is an unsubscribe unless cancel_on_disconnect."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chunk_count=200, chunk_delay=0.01)
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("keep going"))
            turn = json.loads(raw)
            sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                            timeout=8)
            sock.sendall((f"GET /v1/agent/turns/{turn['id']}/events HTTP/1.1\r\n"
                          "Host: 127.0.0.1\r\n\r\n").encode())
            deadline = time.monotonic() + 5
            data = b""
            while b"data: " not in data and time.monotonic() < deadline:
                piece = sock.recv(4096)
                if not piece:
                    break
                data += piece
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                            struct.pack("ii", 1, 0))
            sock.close()
            status, body = _poll_turn(server, turn["id"])
            failures = _check(failures, "a disconnect leaves the turn running",
                              body.get("status") == "completed",
                              f"{body.get('status')}")

            client.chat_script = None
            client.chunk_count = 200
            client.chunk_delay = 0.01
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("cancel me on drop",
                                   settings={"cancel_on_disconnect": True}))
            turn2 = json.loads(raw)
            sock = socket.create_connection(("127.0.0.1", server.address[1]),
                                            timeout=8)
            sock.sendall((f"GET /v1/agent/turns/{turn2['id']}/events HTTP/1.1\r\n"
                          "Host: 127.0.0.1\r\n\r\n").encode())
            data = b""
            deadline = time.monotonic() + 5
            while b"data: " not in data and time.monotonic() < deadline:
                piece = sock.recv(4096)
                if not piece:
                    break
                data += piece
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                            struct.pack("ii", 1, 0))
            sock.close()
            status, body = _poll_turn(server, turn2["id"])
            failures = _check(failures, "cancel_on_disconnect cancels the turn",
                              body.get("status") == "cancelled",
                              f"{body.get('status')}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_turn_timeout() -> int:
    """serve_turn_timeout cancels a turn that runs too long."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chunk_count=300, chunk_delay=0.01)
        server, api, store, thread = _start_agent_api(
            tmp, client, config_extra={"serve_turn_timeout": 0.4},
            manager=orun.TurnManager())
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("too slow"))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"], timeout=15)
            events = _headless_events(api, turn["id"])
            cancelled = [event.data for event in events
                         if event.type == "turn.cancelled"]
            failures = _check(failures, "the wall-clock timeout cancels the turn",
                              body.get("status") == "cancelled"
                              and cancelled
                              and cancelled[0].get("reason") == "timeout",
                              f"{body.get('status')} {cancelled}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_sessions_api() -> int:
    """Session create/list/get/patch/delete over HTTP."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"])
        server, api, store, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/sessions",
                body={"title": "work", "model": "m1", "tags": ["a"],
                      "system": "be terse"})
            created = json.loads(raw)
            failures = _check(failures, "POST /v1/sessions creates a session",
                              status == 201 and created["id"]
                              and created["title"] == "work"
                              and created["model"] == "m1"
                              and created["system"] == "be terse",
                              f"{status} {raw[:200]}")
            session_id = created["id"]

            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"title": "renamed", "system": None, "tags": ["b"]})
            patched = json.loads(raw)
            failures = _check(failures, "PATCH updates the mutable fields",
                              status == 200 and patched["title"] == "renamed"
                              and patched["system"] is None
                              and patched["tags"] == ["b"],
                              f"{status} {raw[:200]}")
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{session_id}")
            fetched = json.loads(raw)
            failures = _check(failures, "GET returns metadata and empty history",
                              status == 200 and fetched["history"] == []
                              and fetched["model"] == "m1",
                              raw[:200])

            status, _headers, raw = _http(
                server, "POST", "/v1/sessions", body={"model": "missing"})
            failures = _check(failures, "an unknown create model is a 404",
                              status == 404
                              and json.loads(raw)["error"]["code"] == "model_not_found",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}", body={"bogus": 1})
            failures = _check(failures, "PATCH rejects unknown fields",
                              status == 400
                              and json.loads(raw)["error"]["code"] == "unknown_field",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(server, "GET", "/v1/sessions/nope-none")
            failures = _check(failures, "an unknown session is a 404",
                              status == 404, f"{status} {raw[:160]}")

            status, _headers, raw = _http(
                server, "DELETE", f"/v1/sessions/{session_id}")
            failures = _check(failures, "DELETE removes the session",
                              status == 200
                              and json.loads(raw).get("deleted") is True,
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(server, "GET", f"/v1/sessions/{session_id}")
            failures = _check(failures, "the deleted session is gone",
                              status == 404, f"{status}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_turn_validation() -> int:
    """The turn request schema rejects malformed bodies with exact envelopes."""
    failures = 0
    base = {"model": "m1", "messages": [{"role": "user", "content": "hi"}]}
    cases = (
        ("an unknown field", {**base, "foo": 1}),
        ("an unknown settings field", {**base, "settings": {"foo": 1}}),
        ("empty messages", {**base, "messages": []}),
        ("a tool role", {**base, "messages": [{"role": "tool", "content": "x"}]}),
        ("empty content", {**base, "messages": [{"role": "user", "content": " "}]}),
        ("a numeric content", {**base, "messages": [
            {"role": "user", "content": 123}]}),
        ("an empty parts array", {**base, "messages": [
            {"role": "user", "content": []}]}),
        ("a malformed part", {**base, "messages": [
            {"role": "user", "content": [42]}]}),
        ("a boolean max_rounds", {**base, "settings": {"max_rounds": True}}),
        ("max_rounds 0", {**base, "settings": {"max_rounds": 0}}),
        ("a bad approval mode", {**base, "settings": {"approval": "auto"}}),
        ("a bad system", {**base, "settings": {"system": 7}}),
        ("bad parameters", {**base, "settings": {"parameters": {"nested": {}}}}),
        ("a non-bool persist", {**base, "persist": "yes"}),
        ("a non-bool cancel_on_disconnect", {**base, "settings": {
            "cancel_on_disconnect": "yes"}}),
    )
    parsed = None
    for name, body in cases:
        try:
            parsed = orun.parse_agent_turn(body)
            error = None
        except orun.ApiError as exc:
            error = exc
        failures = _check(failures, f"turn schema rejects {name}",
                          error is not None and error.status == 400, repr(parsed))
    spec = orun.parse_agent_turn(base)
    failures = _check(failures, "the turn defaults are the safe contract",
                      spec.approval == "policy" and spec.persist is True
                      and spec.tools is True and spec.cancel_on_disconnect is False,
                      str(spec))

    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"])
        config = {"model": None}
        server, api, store, thread = _start_agent_api(tmp, client, config=config)
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body={"messages": [{"role": "user", "content": "hi"}]})
            failures = _check(failures, "a missing model is a 400",
                              status == 400, f"{status} {raw[:160]}")
            status, _headers, raw = _http(server, "GET", "/v1/agent/turns/nope")
            failures = _check(failures, "an unknown turn is a 404",
                              status == 404
                              and json.loads(raw)["error"]["code"] == "turn_not_found",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(server, "POST",
                                          "/v1/agent/approvals/nope",
                                          body={"action": "run"})
            failures = _check(failures, "an unknown approval is a 404",
                              status == 404, f"{status} {raw[:160]}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_turn_expiry() -> int:
    """Swept turn records answer 410; never-seen turns answer 404."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[[{"message": {"content": "ok"},
                                             "done": True}]])
        manager = orun.TurnManager(completed_ttl=0.3, expired_ttl=30.0,
                                   clock=time.monotonic)
        server, api, store, thread = _start_agent_api(tmp, client,
                                                      manager=manager)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/agent/turns",
                                          body=_turn_request("expire"))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"])
            failures = _check(failures, "the turn completes before expiry",
                              body.get("status") == "completed",
                              str(body)[:160])
            time.sleep(0.5)
            status, _headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn['id']}/events")
            failures = _check(failures, "an expired turn's events are a 410",
                              status == 410, f"{status} {raw[:160]}")
            status, _headers, raw = _http(
                server, "GET", f"/v1/agent/turns/{turn['id']}")
            failures = _check(failures, "an expired turn is a 410",
                              status == 410
                              and json.loads(raw)["error"]["code"] == "turn_expired",
                              f"{status} {raw[:160]}")
            status, _headers, raw = _http(server, "GET",
                                          "/v1/agent/turns/turn_never")
            failures = _check(failures, "an unknown turn stays a 404",
                              status == 404, f"{status}")
        finally:
            _stop_api(server, thread)
    return failures


def test_agent_context_warning() -> int:
    """The headless warning/compaction step (Phase 1b item 13)."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        session = orun.Session.create("m1", "http://fake:11434")
        for index in range(6):
            session.add_message("user" if index % 2 == 0 else "assistant",
                                f"message {index}")
        session.stats["last_prompt_tokens"] = 90
        session.stats["last_completion_tokens"] = 5
        store.save(session)
        client = _ServeClient(models=["m1"], capabilities=["completion", "tools"],
                              show_data={"model_info": {"llama.context_length": 100},
                                         "parameters": "num_ctx 100"},
                              chat_once_reply="Summary of the earlier talk",
                              chat_script=[[{"message": {"content": "ok"},
                                             "done": True}]])
        server, api, store, thread = _start_agent_api(
            tmp, client, store=store, config_extra={"auto_compact": "auto"})
        try:
            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("continue", session_id=session.id))
            turn = json.loads(raw)
            status, body = _poll_turn(server, turn["id"])
            events = _headless_events(api, turn["id"])
            warnings = [event.data.get("message") for event in events
                        if event.type == "turn.warning"]
            failures = _check(failures, "the context warning rides the event stream",
                              body.get("status") == "completed"
                              and any("context" in str(message)
                                      for message in warnings),
                              str(warnings))
            failures = _check(failures, "auto_compact compacts before the turn",
                              any("compacted" in str(message)
                                  for message in warnings)
                              and client.chat_once_payloads,
                              str(warnings))
            status, _headers, raw = _http(
                server, "GET", f"/v1/sessions/{session.id}")
            history = json.loads(raw)["history"]
            failures = _check(failures, "the compacted history keeps the summary",
                              any("Summary of earlier conversation" in str(
                                  message.get("content")) for message in history)
                              and len(history) < 7,
                              str(history)[:200])
        finally:
            _stop_api(server, thread)
    return failures





def test_session_context_fields() -> int:
    """PATCH and turn settings carry first-class num_ctx/gpu_layers."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        show_data = {"model_info": {"m1.context_length": 32768,
                                    "m1.block_count": 24}}
        client = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"],
                              resident=["m1"], show_data=show_data,
                              chat_script=[[{"message": {"content": "ok"},
                                             "done": True}]])
        server, api, sessions, thread = _start_agent_api(tmp, client)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/sessions",
                                          body={"model": "m1"})
            session_id = json.loads(raw)["id"]
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"num_ctx": 8192, "gpu_layers": "50%"})
            failures += _fit_check("PATCH stores num_ctx and a resolved spec",
                                  status == 200, f"{status} {raw[:120]!r}")
            saved = sessions.load(session_id)
            failures += _fit_check("the values persist on the session",
                                  saved.parameters.get("num_ctx") == 8192
                                  and saved.gpu_layers == "50%",
                                  f"{saved.parameters} {saved.gpu_layers}")
            failures += _fit_check("a placement change unloads the resident model",
                                  "m1" in client.unloaded, str(client.unloaded))
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"num_ctx": None, "gpu_layers": None})
            saved = sessions.load(session_id)
            failures += _fit_check("null clears the values",
                                  "num_ctx" not in saved.parameters
                                  and saved.gpu_layers is None,
                                  f"{saved.parameters} {saved.gpu_layers}")
            for name, body, param in (
                ("an over-max num_ctx", {"num_ctx": 65536}, "num_ctx"),
                ("a boolean num_ctx", {"num_ctx": True}, "num_ctx"),
                ("a zero num_ctx", {"num_ctx": 0}, "num_ctx"),
                ("a bad spec", {"gpu_layers": "nope"}, "gpu_layers"),
            ):
                status, _headers, raw = _http(
                    server, "PATCH", f"/v1/sessions/{session_id}", body=body)
                error = json.loads(raw).get("error", {})
                failures += _fit_check(f"PATCH rejects {name}",
                                      status == 400
                                      and error.get("param") == param,
                                      f"{status} {error}")

            status, _headers, raw = _http(
                server, "POST", "/v1/agent/turns",
                body=_turn_request("set it",
                                   settings={"num_ctx": 4096,
                                             "gpu_layers": "cpu"}))
            turn = json.loads(raw)
            _poll_turn(server, turn["id"])
            saved = sessions.load(turn["session_id"])
            failures += _fit_check("turn settings store the context fields",
                                  saved.parameters.get("num_ctx") == 4096
                                  and saved.gpu_layers == "cpu",
                                  f"{saved.parameters} {saved.gpu_layers}")
            for name, settings, param in (
                ("a negative num_ctx", {"num_ctx": -1}, "settings.num_ctx"),
                ("a numeric gpu_layers", {"gpu_layers": 5},
                 "settings.gpu_layers"),
            ):
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("bad", settings=settings))
                error = json.loads(raw).get("error", {})
                failures += _fit_check(f"turn settings reject {name}",
                                      status == 400
                                      and error.get("param") == param,
                                      f"{status} {error}")
        finally:
            _stop_api(server, thread)

        sparse = _ServeClient(models=["m1"],
                              capabilities=["completion", "tools"],
                              show_data={"model_info": {"m1.context_length": 32768}})
        server, api, sessions, thread = _start_agent_api(tmp, sparse)
        try:
            status, _headers, raw = _http(server, "POST", "/v1/sessions",
                                          body={"model": "m1"})
            session_id = json.loads(raw)["id"]
            status, _headers, raw = _http(
                server, "PATCH", f"/v1/sessions/{session_id}",
                body={"gpu_layers": "50%"})
            error = json.loads(raw).get("error", {})
            failures += _fit_check("a percent spec needs a resolvable layer count",
                                  status == 400
                                  and error.get("param") == "gpu_layers",
                                  f"{status} {error}")
        finally:
            _stop_api(server, thread)
    return failures




def test_fit_estimate_api() -> int:
    """POST /v1/agent/fit: estimate, apply, validation and policy gates."""
    failures = 0
    machine = orun.MachineGpu(devices=[(16384, 12000)], env={}, source="local")
    old_probe = orun.ctxfit.api.machine_state_for
    old_refresh = orun.agent_api.machine_state_for
    orun.ctxfit.api.machine_state_for = lambda *a, **k: machine
    orun.agent_api.machine_state_for = lambda *a, **k: machine
    try:
        with tempfile.TemporaryDirectory() as tmp:
            show = {"model_info": dict(
                CTX_SHOW["model_info"],
                **{"qwen35.context_length": 65536})}
            client = _ServeClient(
                models=["m1"], capabilities=["completion", "tools"],
                show_data=show,
                tag_data=[{"name": "m1", "size": 2_000_000_000,
                           "digest": "sha256:abc"}])
            # The CPU estimate reads the machine's own /proc/meminfo.
            client.base = "http://127.0.0.1:11434"
            server, api, sessions, thread = _start_agent_api(tmp, client)
            try:
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"model": "m1",
                                                    "mode": "gpu"})
                body = json.loads(raw)
                failures += _fit_check("an estimate returns the fit result",
                                      status == 200
                                      and body.get("object") == "fit"
                                      and body.get("bound") == "max_gpu"
                                      and (body.get("num_ctx") or 0) > 0
                                      and body.get("applied") is False
                                      and isinstance(body.get("report"), dict),
                                      f"{status} {str(body)[:200]}")
                status, _headers, raw = _http(server, "POST", "/v1/sessions",
                                              body={"model": "m1"})
                session_id = json.loads(raw)["id"]
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"session_id": session_id,
                                                    "mode": "gpu",
                                                    "apply": True})
                body = json.loads(raw)
                saved = sessions.load(session_id)
                failures += _fit_check("apply writes the fitted context",
                                      status == 200
                                      and body.get("applied") is True
                                      and saved.parameters.get("num_ctx")
                                      == body.get("num_ctx"),
                                      f"{status} {body.get('applied')} "
                                      f"{saved.parameters}")
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"model": "m1",
                                                    "mode": "cpu",
                                                    "cpu_percent": 70})
                failures += _fit_check("a cpu estimate reads the real machine",
                                      status == 200
                                      and json.loads(raw).get("bound")
                                      == "max_cpu",
                                      f"{status} {raw[:120]!r}")
                for name, body, param in (
                    ("an unknown field", {"model": "m1", "x": 1}, "x"),
                    ("a bad mode", {"model": "m1", "mode": "all"}, "mode"),
                    ("cpu without a percent",
                     {"model": "m1", "mode": "cpu"}, "cpu_percent"),
                    ("a bad percent", {"model": "m1", "mode": "cpu",
                                       "cpu_percent": 0}, "cpu_percent"),
                    ("a percent with gpu mode",
                     {"model": "m1", "cpu_percent": 50}, "cpu_percent"),
                    ("apply without a session",
                     {"model": "m1", "apply": True}, "session_id"),
                    ("a probe field on the sync route",
                     {"model": "m1", "probe": True}, "probe"),
                ):
                    status, _headers, raw = _http(server, "POST",
                                                  "/v1/agent/fit", body=body)
                    error = json.loads(raw).get("error", {})
                    failures += _fit_check(f"the fit API rejects {name}",
                                          status == 400
                                          and error.get("param") == param,
                                          f"{status} {error}")
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"model": "nope"})
                failures += _fit_check("an unknown fit model is a 404",
                                      status == 404, str(status))
            finally:
                _stop_api(server, thread)

            blind = _ServeClient(models=["m1"], capabilities=["completion"],
                                 show_data={})
            server, api, sessions, thread = _start_agent_api(tmp, blind)
            try:
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"model": "m1",
                                                    "mode": "gpu"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("an unsizeable fit is a structured 400",
                                      status == 400
                                      and error.get("code") == "fit_failed",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)

            server, api, sessions, thread = _start_agent_api(
                tmp, client, config_extra={"serve_fit": "off"})
            try:
                status, _headers, raw = _http(server, "POST", "/v1/agent/fit",
                                              body={"model": "m1"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("serve_fit off refuses the estimate",
                                      status == 403
                                      and error.get("code") == "fit_disabled",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)
    finally:
        orun.ctxfit.api.machine_state_for = old_probe
        orun.agent_api.machine_state_for = old_refresh
    return failures




def test_fit_job_api() -> int:
    """Async fit jobs: 202 + warnings, status/progress, cancel, gates."""
    failures = 0
    machine = orun.MachineGpu(devices=[(16384, 12000)], env={}, source="local")
    old_probe = orun.ctxfit.api.machine_state_for
    old_refresh = orun.agent_api.machine_state_for
    old_fit = orun.fits.fit_context
    orun.ctxfit.api.machine_state_for = lambda *a, **k: machine
    orun.agent_api.machine_state_for = lambda *a, **k: machine
    gate = threading.Event()
    gate.set()

    def slow_fit(req, *, mode, percent=None, probe=False):
        req.on_progress({"stage": "probe", "mode": mode, "ctx": 4096,
                         "attempt": 1})
        gate.wait(timeout=10)
        if req.cancel is not None and req.cancel.cancelled:
            raise orun.FitCancelled("fit cancelled")
        outcome = orun.FitOutcome(mode=mode, model=req.model, probe=probe)
        outcome.num_ctx = 8192
        return outcome

    orun.fits.fit_context = slow_fit

    def _poll_fit(server, fit_id, until, timeout=10.0, ready=None):
        deadline = time.monotonic() + timeout
        body: dict = {}
        while time.monotonic() < deadline:
            _status, _headers, raw = _http(server, "GET",
                                           f"/v1/agent/fits/{fit_id}")
            body = json.loads(raw)
            if body.get("status") in until and (ready is None or ready(body)):
                return body
            time.sleep(0.02)
        return body

    try:
        with tempfile.TemporaryDirectory() as tmp:
            show = {"model_info": dict(
                CTX_SHOW["model_info"],
                **{"qwen35.context_length": 65536})}
            client = _ServeClient(
                models=["m1"], capabilities=["completion", "tools"],
                resident=["m1"], show_data=show,
                tag_data=[{"name": "m1", "size": 2_000_000_000,
                           "digest": "sha256:abc"}],
                chat_script=[[{"message": {"content": "ok"},
                               "done": True}]])
            server, api, sessions, thread = _start_agent_api(tmp, client)
            try:
                status, _headers, raw = _http(server, "POST", "/v1/sessions",
                                              body={"model": "m1"})
                session_id = json.loads(raw)["id"]
                gate.clear()
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"session_id": session_id,
                                                    "mode": "gpu"})
                job = json.loads(raw)
                job_id = job.get("id")
                failures += _fit_check("a fit job starts with 202 and its urls",
                                      status == 202
                                      and job.get("status") in ("queued",
                                                                "running")
                                      and job.get("status_url", "").endswith(
                                          f"/v1/agent/fits/{job_id}")
                                      and isinstance(job.get("warnings"), list),
                                      f"{status} {str(job)[:200]}")
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("hi", session_id=session_id))
                error = json.loads(raw).get("error", {})
                failures += _fit_check("a fitting session refuses a turn",
                                      status == 409
                                      and error.get("active_fit_id") == job_id,
                                      f"{status} {error}")
                status, _headers, raw = _http(
                    server, "PATCH", f"/v1/sessions/{session_id}",
                    body={"title": "x"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("a fitting session refuses edits",
                                      status == 409
                                      and error.get("active_fit_id") == job_id,
                                      f"{status} {error}")
                status, _headers, raw = _http(server, "GET",
                                              f"/v1/agent/fits/{job_id}")
                live = json.loads(raw)
                failures += _fit_check("the status endpoint reports progress",
                                      status == 200
                                      and live.get("progress", {}).get("ctx")
                                      == 4096,
                                      f"{status} {str(live)[:160]}")
                status, _headers, raw = _http(
                    server, "POST", f"/v1/agent/fits/{job_id}/cancel")
                failures += _fit_check("cancel is accepted with 202",
                                      status == 202, str(status))
                gate.set()
                final = _poll_fit(server, job_id, until={"cancelled"},
                                  timeout=10)
                failures += _fit_check("the cancelled job reports cancelled",
                                      final.get("status") == "cancelled",
                                      str(final)[:160])

                gate.clear()
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/fits",
                    body={"session_id": session_id, "mode": "gpu",
                          "apply": True})
                applied_id = json.loads(raw)["id"]
                gate.set()
                applied = _poll_fit(server, applied_id, until={"completed"},
                                    ready=lambda body: body.get("applied"))
                saved = sessions.load(session_id)
                failures += _fit_check("an applied job writes the fitted context",
                                      applied.get("status") == "completed"
                                      and applied.get("applied") is True
                                      and saved.parameters.get("num_ctx") == 8192,
                                      f"{applied.get('status')} "
                                      f"{saved.parameters}")
                status, _headers, raw = _http(
                    server, "POST", f"/v1/agent/fits/{applied_id}/cancel")
                error = json.loads(raw).get("error", {})
                failures += _fit_check("cancelling a finished job is a 409",
                                      status == 409
                                      and error.get("code") == "fit_finished",
                                      f"{status} {error}")
                status, _headers, raw = _http(server, "GET",
                                              "/v1/agent/fits/fit_nope")
                error = json.loads(raw).get("error", {})
                failures += _fit_check("an unknown fit is a structured 404",
                                      status == 404
                                      and error.get("code") == "fit_not_found",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)

            # the global cap and the policy gates
            server, api, sessions, thread = _start_agent_api(
                tmp, client, config_extra={"serve_max_active_fits": 1})
            try:
                gate.clear()
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"model": "m1",
                                                    "mode": "gpu"})
                gated = json.loads(raw)["id"]
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"model": "m1",
                                                    "mode": "gpu"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("the fit cap answers 429",
                                      status == 429
                                      and error.get("code") == "rate_limit_error",
                                      f"{status} {error}")
                _http(server, "POST", f"/v1/agent/fits/{gated}/cancel")
                gate.set()
                _poll_fit(server, gated, until={"cancelled"})
            finally:
                _stop_api(server, thread)

            server, api, sessions, thread = _start_agent_api(
                tmp, client, config_extra={"serve_fit": "estimate"})
            try:
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"model": "m1",
                                                    "mode": "gpu"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("serve_fit estimate refuses probing",
                                      status == 403
                                      and error.get("code") == "probe_disabled",
                                      f"{status} {error}")
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"model": "m1",
                                                    "mode": "gpu",
                                                    "probe": False})
                failures += _fit_check("an explicit estimate-only job still runs",
                                      status == 202, f"{status} {raw[:80]!r}")
                _http(server, "POST", f"/v1/agent/fits/"
                                      f"{json.loads(raw).get('id')}/cancel")
            finally:
                _stop_api(server, thread)

            server, api, sessions, thread = _start_agent_api(
                tmp, client, config_extra={"serve_fit": "off"})
            try:
                status, _headers, raw = _http(server, "POST", "/v1/agent/fits",
                                              body={"model": "m1"})
                error = json.loads(raw).get("error", {})
                failures += _fit_check("serve_fit off refuses jobs",
                                      status == 403
                                      and error.get("code") == "fit_disabled",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)
    finally:
        gate.set()
        orun.ctxfit.api.machine_state_for = old_probe
        orun.agent_api.machine_state_for = old_refresh
        orun.fits.fit_context = old_fit
    return failures




def test_turn_start_fit() -> int:
    """settings.num_ctx "max-gpu" estimates before round one and reports it."""
    failures = 0
    machine = orun.MachineGpu(devices=[(16384, 12000)], env={}, source="local")
    old_probe = orun.ctxfit.api.machine_state_for
    old_refresh = orun.agent_api.machine_state_for
    orun.ctxfit.api.machine_state_for = lambda *a, **k: machine
    orun.agent_api.machine_state_for = lambda *a, **k: machine
    try:
        with tempfile.TemporaryDirectory() as tmp:
            show = {"model_info": dict(
                CTX_SHOW["model_info"],
                **{"qwen35.context_length": 65536})}
            client = _ServeClient(
                models=["m1"], capabilities=["completion", "tools"],
                show_data=show,
                tag_data=[{"name": "m1", "size": 2_000_000_000,
                           "digest": "sha256:abc"}],
                chat_script=[[{"message": {"content": "ok"},
                               "done": True}]])
            server, api, sessions, thread = _start_agent_api(tmp, client)
            try:
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("fit me",
                                       settings={"num_ctx": "max-gpu"}))
                turn = json.loads(raw)
                _status, body = _poll_turn(server, turn["id"])
                events = _headless_events(api, turn["id"])
                fits = [event.data for event in events
                        if event.type == orun.SESSION_FIT]
                failures += _fit_check("the turn emits one session.fit",
                                      len(fits) == 1
                                      and (fits[0].get("num_ctx") or 0) > 0
                                      and fits[0].get("applied") is True,
                                      str(fits[:1])[:200])
                failures += _fit_check("the fit result rides the turn status",
                                      body.get("result", {}).get("fit", {})
                                      .get("num_ctx") == fits[0].get("num_ctx"),
                                      str(body.get("result", {}).get("fit"))[:160])
                session = sessions.load(turn["session_id"])
                failures += _fit_check("the fitted context is persisted",
                                      session.parameters.get("num_ctx")
                                      == fits[0].get("num_ctx"),
                                      str(session.parameters))
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("bad",
                                       settings={"num_ctx": "max-cpu"}))
                error = json.loads(raw).get("error", {})
                failures += _fit_check('"max-cpu" without cpu_percent is a 400',
                                      status == 400
                                      and error.get("param")
                                      == "settings.cpu_percent",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)

            blind = _ServeClient(
                models=["m1"], capabilities=["completion", "tools"],
                show_data={},
                chat_script=[[{"message": {"content": "ok"},
                               "done": True}]])
            server, api, sessions, thread = _start_agent_api(tmp, blind)
            try:
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("fit fails",
                                       settings={"num_ctx": "max-gpu"}))
                turn = json.loads(raw)
                _status, body = _poll_turn(server, turn["id"])
                fits = [event.data for event in _headless_events(api, turn["id"])
                        if event.type == orun.SESSION_FIT]
                session = sessions.load(turn["session_id"])
                failures += _fit_check(
                    "a failed fit reports on the event and leaves ctx alone",
                    len(fits) == 1 and fits[0].get("num_ctx") is None
                    and fits[0].get("error")
                    and body.get("status") == "completed"
                    and "num_ctx" not in session.parameters,
                    f"{str(fits[:1])[:120]} {session.parameters}")
            finally:
                _stop_api(server, thread)

            server, api, sessions, thread = _start_agent_api(
                tmp, client, config_extra={"serve_fit": "off"})
            try:
                status, _headers, raw = _http(
                    server, "POST", "/v1/agent/turns",
                    body=_turn_request("off",
                                       settings={"num_ctx": "max-gpu"}))
                error = json.loads(raw).get("error", {})
                failures += _fit_check("serve_fit off refuses turn-start fits",
                                      status == 403
                                      and error.get("code") == "fit_disabled",
                                      f"{status} {error}")
            finally:
                _stop_api(server, thread)
    finally:
        orun.ctxfit.api.machine_state_for = old_probe
        orun.agent_api.machine_state_for = old_refresh
    return failures

class _ScriptedInput:
    """A blocking line source for the session client (no terminal needed)."""

    def __init__(self) -> None:
        self.lines: "queue.Queue[str]" = queue.Queue()

    def readline(self) -> str:
        return self.lines.get()

    def push(self, text: str) -> None:
        self.lines.put(text + "\n")


def _wait_until(predicate, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(0.02)
    return False


def _host_fixture(tmp: str, client, *, args=None):
    root = Path(tmp) / "hosts"
    store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                              index_path=Path(tmp) / "index.json")
    if args is None:
        args = orun.build_parser().parse_args(
            ["m1", "--render", "plain", "--no-tools", "--keep-loaded",
             "--no-math", "--no-tps"])
    session = orun.Session.create("m1", client.base)
    spec = orun.HostSpec(
        protocol=orun.HOST_PROTOCOL, version=orun.VERSION,
        session=session.to_dict(), args=vars(args), base=client.base,
        insecure=False, cwd=os.getcwd(), session_id=session.id,
        config_path=None, host_root=str(root))
    paths = orun.host_paths(session.id, root)
    server = orun.SessionHostServer(
        spec, paths, client_factory=lambda *a, **k: client, store=store)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    assert _wait_until(lambda: orun.ping_host(session.id, root), 10.0)
    return server, spec, paths, store, session, root


def _attach_client(spec, paths, *, buf=None):
    conn = orun.LineConnection.connect(paths.socket)
    out = io.StringIO()
    console_buf = buf if buf is not None else io.StringIO()
    inp = _ScriptedInput()
    client = orun.SessionClient(
        spec, conn, input_stream=inp, output_stream=out,
        console_obj=orun.Console(file=console_buf, width=100,
                                 force_terminal=False, theme=orun.THEME))
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()
    return client, inp, out, console_buf


def test_interrupt_guard() -> int:
    failures = 0
    now = [100.0]
    guard = orun.InterruptGuard(window=10.0, clock=lambda: now[0])
    failures = _check(failures, "the first press only arms the detach window",
                      guard.press() == "arm")
    now[0] = 105.0
    failures = _check(failures, "a second press inside 10s detaches",
                      guard.press() == "detach")
    now[0] = 200.0
    failures = _check(failures, "the guard starts a fresh window afterwards",
                      guard.press() == "arm")
    now[0] = 211.0
    failures = _check(failures, "an expired window must be armed again",
                      guard.press() == "arm")
    now[0] = 212.0
    failures = _check(failures, "the new window's second press detaches",
                      guard.press() == "detach")
    return failures


def test_session_host_protocol() -> int:
    """The real Repl runs in the host: prompt, command output, turn, close."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeClient(models=["m1"], capabilities=["completion"],
                             chat_script=[[
                                 {"message": {"content": "Hello from the host."}},
                                 {"done": True, "prompt_eval_count": 3,
                                  "eval_count": 4,
                                  "prompt_eval_duration": 1_000_000_000,
                                  "eval_duration": 1_000_000_000}]])
        server, spec, paths, store, session, _root = _host_fixture(tmp, client)
        try:
            term, inp, out, buf = _attach_client(spec, paths)
            failures = _check(
                failures, "the host prints its banner and prompt",
                _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0),
                out.getvalue()[-300:])
            inp.push("/help shortcuts")
            failures = _check(
                failures, "slash commands run in the host and stream output",
                _wait_until(lambda: "Ctrl+D" in out.getvalue(), 10.0),
                out.getvalue()[-500:])
            inp.push("hi")
            failures = _check(
                failures, "a turn streams and renders in the attached client",
                _wait_until(lambda: "Hello from the host." in buf.getvalue(),
                            20.0),
                buf.getvalue()[-500:])
            inp.push("/bye")
            failures = _check(failures, "/bye closes the host",
                              _wait_until(lambda: term.done, 20.0))
            saved = store.load(session.id)
            failures = _check(
                failures, "the turn is persisted before the host exits",
                [m.get("role") for m in saved.messages] == ["user", "assistant"]
                and saved.messages[1].get("content") == "Hello from the host.",
                str([(m.get("role"), m.get("content")) for m in saved.messages]))
        finally:
            server._finish()
    return failures


class _SlowFakeClient(_FakeClient):
    def __init__(self):
        super().__init__(models=["m1"], capabilities=["completion"])

    def chat(self, payload, timeout=None, cancel=None):
        def events():
            for text in ("Hello", " from", " the background."):
                if cancel is not None and cancel.cancelled:
                    return
                time.sleep(0.25)
                yield {"message": {"content": text}}
            yield {"done": True, "prompt_eval_count": 3, "eval_count": 4,
                   "prompt_eval_duration": 1_000_000_000,
                   "eval_duration": 1_000_000_000}
        return events()


def test_session_host_thinking_toggle() -> int:
    """Ctrl+T in the client sticks in the host and the next turn starts that way."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _SlowFakeClient()
        server, spec, paths, store, session, _root = _host_fixture(tmp, client)
        try:
            term, inp, out, buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            failures = _check(
                failures, "the host advertises the thinking op",
                "thinking" in term.features, str(term.features))
            inp.push("first")
            assert _wait_until(
                lambda: term.renderer is not None
                and "Hello" in buf.getvalue(), 15.0), buf.getvalue()
            first_renderer = term.renderer
            assert first_renderer is not None
            first_renderer._handle_key_bytes(bytes([orun.CTRL_T]))
            failures = _check(
                failures, "Ctrl+T reaches the host's Repl",
                _wait_until(lambda: server.repl is not None
                            and server.repl.thinking_mode == "expanded", 10.0),
                str(getattr(server.repl, "thinking_mode", None)))
            failures = _check(
                failures, "the host echoes the new mode to the client",
                _wait_until(lambda: term.settings.get("thinking") == "expanded",
                            10.0),
                str(term.settings.get("thinking")))
            assert _wait_until(
                lambda: term.renderer is None
                and "the background." in buf.getvalue(), 20.0), buf.getvalue()
            inp.push("second")
            failures = _check(
                failures, "the next turn's renderer starts expanded",
                _wait_until(lambda: term.renderer is not None
                            and term.renderer is not first_renderer
                            and term.renderer.thinking_mode == "expanded", 20.0),
                str(getattr(term.renderer, "thinking_mode", None)))
            inp.push("/bye")
            failures = _check(failures, "/bye closes the host",
                              _wait_until(lambda: term.done, 20.0))
        finally:
            server._finish()
    return failures


def test_session_host_detach_reattach() -> int:
    """Closing the terminal does not stop the turn; -c shows what was missed."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _SlowFakeClient()
        server, spec, paths, store, session, root = _host_fixture(tmp, client)
        try:
            term, inp, out, buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            inp.push("hi")
            assert _wait_until(lambda: "Hello" in buf.getvalue(), 15.0), \
                buf.getvalue()
            # Window closed: the socket goes away without a close op.
            term.conn.close()
            term.done = True
            def saved_messages():
                try:
                    return store.load(session.id).messages
                except Exception:
                    return []

            failures = _check(
                failures, "the host keeps the turn alive after a detach",
                _wait_until(lambda: bool(saved_messages()), 15.0))
            failures = _check(
                failures, "the detached turn finishes and is persisted",
                _wait_until(lambda: any(
                    m.get("content") == "Hello from the background."
                    for m in saved_messages()), 20.0),
                str([(m.get("role"), m.get("content"))
                     for m in saved_messages()]))
            term2, inp2, out2, buf2 = _attach_client(spec, paths)
            failures = _check(
                failures, "reattaching shows the turn completed while away",
                _wait_until(lambda: "Hello from the background." in buf2.getvalue(),
                            20.0),
                buf2.getvalue()[-600:])
            inp2.push("/bye")
            failures = _check(failures, "the reattached client closes cleanly",
                              _wait_until(lambda: term2.done, 20.0))
        finally:
            server._finish()
    return failures


class _ApprovalUi:
    interactive = True

    def __init__(self, action="deny"):
        self.action = action
        self.asked = []

    def approval(self, suggestions, *, name="", detail="", reason=""):
        self.asked.append((name, reason))
        return self.action, None

    def confirm(self, question, *, default=False):
        return default

    def ask(self, question, *, choices=None, default=None):
        return default

    def editor(self, initial=""):
        return None

    def copy(self, text):
        return True, "copied"

    def notify(self, event, **data):
        return


def test_session_host_approval() -> int:
    """An approval question round-trips through the attached terminal."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "denied.txt"
        client = _FakeClient(
            models=["m1"], capabilities=["completion", "tools"],
            chat_script=[
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "write_file",
                                  "arguments": {"path": str(target),
                                                "content": "x"}}}]}}],
                [{"message": {"content": "done"}, "done": True,
                 "prompt_eval_count": 1, "eval_count": 1,
                 "prompt_eval_duration": 1_000_000_000,
                 "eval_duration": 1_000_000_000}],
            ])
        host_args = orun.build_parser().parse_args(
            ["m1", "--render", "plain", "--keep-loaded", "--no-math",
             "--no-tps"])
        server, spec, paths, store, session, root = _host_fixture(
            tmp, client, args=host_args)
        try:
            term, inp, out, buf = _attach_client(spec, paths)
            ui = _ApprovalUi("deny")
            term.ui = ui
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            inp.push("write a file")
            failures = _check(
                failures, "the tool call parks for approval and the turn resumes",
                _wait_until(lambda: "done" in buf.getvalue(), 20.0),
                buf.getvalue()[-600:])
            failures = _check(
                failures, "the attached terminal was asked about the call",
                bool(ui.asked) and ui.asked[0][0] == "write_file",
                str(ui.asked))
            failures = _check(
                failures, "a denied call did not write its target",
                not target.exists())
            saved = store.load(session.id)
            failures = _check(
                failures, "the denial is recorded as a tool result",
                any(m.get("role") == "tool"
                    and "denied by the user" in (m.get("content") or "")
                    for m in saved.messages),
                str([(m.get("role"), m.get("content"))
                     for m in saved.messages]))
            inp.push("/bye")
            failures = _check(failures, "the approval session closes cleanly",
                              _wait_until(lambda: term.done, 20.0))
        finally:
            server._finish()
    return failures


def test_interrupt_hint() -> int:
    """The Ctrl+C warning names the close/stop keys, Ctrl+Q only when streaming."""
    failures = 0
    busy = "\n".join(orun.interrupt_lines(True))
    idle = "\n".join(orun.interrupt_lines(False))
    failures = _check(failures, "the Ctrl+C warning keeps the detach line",
                      "press Ctrl+C again within 10s" in busy
                      and "press Ctrl+C again within 10s" in idle)
    failures = _check(failures, "the warning always names Ctrl+D",
                      "Ctrl+D terminates the session" in busy
                      and "Ctrl+D terminates the session" in idle)
    failures = _check(failures, "Ctrl+Q is offered while generating",
                      "Ctrl+Q stops the current generation" in busy)
    failures = _check(failures, "Ctrl+Q is omitted when nothing streams",
                      "Ctrl+Q" not in idle, idle)
    return failures


def test_session_host_cancel() -> int:
    """Ctrl+Q cancels the turn, keeps the client attached and returns to prompt."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _SlowFakeClient()
        server, spec, paths, store, session, root = _host_fixture(tmp, client)
        try:
            term, inp, out, buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            inp.push("hi")
            assert _wait_until(lambda: "Hello" in buf.getvalue(), 15.0), \
                buf.getvalue()
            failures = _check(failures, "Ctrl+Q sends the cancel op",
                              term.cancel_turn())
            failures = _check(
                failures, "the cancelled turn returns to a live prompt",
                _wait_until(lambda: out.getvalue().count("m1 ›") >= 2, 20.0),
                out.getvalue()[-500:])
            failures = _check(failures, "the client stays attached after cancel",
                              not term.done)
            saved = store.load(session.id)
            failures = _check(
                failures, "the cancelled turn is persisted as cancelled",
                any(m.get("role") == "assistant" and m.get("cancelled")
                    for m in saved.messages),
                str([(m.get("role"), m.get("content"), m.get("cancelled"))
                     for m in saved.messages]))
            inp.push("/bye")
            failures = _check(failures, "the session closes after a cancel",
                              _wait_until(lambda: term.done, 20.0))
        finally:
            server._finish()
    return failures


class _BlockingApprovalUi(_ApprovalUi):
    def __init__(self):
        super().__init__("deny")
        self.asked = threading.Event()
        self.release = threading.Event()

    def approval(self, suggestions, *, name="", detail="", reason=""):
        self.asked.set()
        self.release.wait(timeout=15.0)
        return "deny", None


def test_session_host_cancel_approval() -> int:
    """Cancelling while an approval is parked unwinds the turn, not hangs."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "cancel-approval.txt"
        client = _FakeClient(
            models=["m1"], capabilities=["completion", "tools"],
            chat_script=[
                [{"message": {"content": "", "tool_calls": [
                    {"function": {"name": "write_file",
                                  "arguments": {"path": str(target),
                                                "content": "x"}}}]}}],
                [{"message": {"content": "after"}, "done": True}],
            ])
        host_args = orun.build_parser().parse_args(
            ["m1", "--render", "plain", "--keep-loaded", "--no-math",
             "--no-tps"])
        server, spec, paths, store, session, root = _host_fixture(
            tmp, client, args=host_args)
        try:
            term, inp, out, _buf = _attach_client(spec, paths)
            ui = _BlockingApprovalUi()
            term.ui = ui
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            inp.push("write a file")
            failures = _check(failures, "the approval prompt is waiting",
                              _wait_until(lambda: ui.asked.is_set(), 15.0))
            failures = _check(failures, "Ctrl+Q cancels the parked approval",
                              term.cancel_turn())
            ui.release.set()
            failures = _check(
                failures, "the prompt returns after cancelling the approval",
                _wait_until(lambda: out.getvalue().count("m1 ›") >= 2, 20.0),
                out.getvalue()[-500:])
            failures = _check(failures, "the turn did not write its tool result",
                              not target.exists())
            inp.push("/bye")
            failures = _check(failures, "the session closes after the cancel",
                              _wait_until(lambda: term.done, 20.0))
        finally:
            ui.release.set()
            server._finish()
    return failures


def test_session_host_stop() -> int:
    """--close-session (stop_host) closes a live host and waits for its exit."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeClient(models=["m1"], capabilities=["completion"])
        server, spec, paths, store, session, root = _host_fixture(tmp, client)
        try:
            term, inp, out, _buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            failures = _check(
                failures, "stop_host closes a live session host",
                orun.stop_host(session.id, root))
            failures = _check(
                failures, "the host socket is gone after stop_host",
                not paths.socket.exists()
                and not orun.host_is_running(session.id, root))
            # An embedded client (injected input) has no SIGINT wake-up;
            # one line lets its prompt observe the queued shutdown.
            inp.push("")
            failures = _check(
                failures, "the attached terminal is disconnected on close",
                _wait_until(lambda: term.done, 15.0))
        finally:
            server._finish()
    return failures


def test_session_host_takeover() -> int:
    """A second -c takes over; the old terminal is told and exits."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeClient(models=["m1"], capabilities=["completion"])
        server, spec, paths, store, session, root = _host_fixture(tmp, client)
        try:
            old, old_inp, old_out, _old_buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in old_out.getvalue(), 15.0)
            new, new_inp, new_out, _new_buf = _attach_client(spec, paths)
            failures = _check(
                failures, "the new terminal receives the same prompt",
                _wait_until(lambda: "m1 ›" in new_out.getvalue(), 15.0),
                new_out.getvalue()[-300:])
            failures = _check(
                failures, "the old terminal is told it was taken over",
                _wait_until(lambda: old.taken_over, 10.0))
            old_inp.push("")
            failures = _check(failures, "the old terminal exits cleanly",
                              _wait_until(lambda: old.done, 10.0)
                              and old.exit_code == 0)
            new_inp.push("/bye")
            failures = _check(failures, "the taking-over terminal closes",
                              _wait_until(lambda: new.done, 20.0))
        finally:
            server._finish()
    return failures


def test_session_host_rekey() -> int:
    """ /clear changes the session id; the host rekeys its socket with it."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeClient(models=["m1"], capabilities=["completion"])
        server, spec, paths, store, session, root = _host_fixture(tmp, client)
        try:
            term, inp, out, _buf = _attach_client(spec, paths)
            assert _wait_until(lambda: "m1 ›" in out.getvalue(), 15.0)
            old_id = server.session_id
            inp.push("/clear")
            failures = _check(
                failures, "/clear rekeys the background host",
                _wait_until(lambda: server.session_id != old_id, 15.0),
                f"{old_id} -> {server.session_id}")
            failures = _check(
                failures, "the new session id answers on its own socket",
                orun.ping_host(server.session_id, root))
            failures = _check(
                failures, "the client learned the new session id",
                _wait_until(lambda: term.session_id == server.session_id, 10.0),
                f"{term.session_id} != {server.session_id}")
            inp.push("/bye")
            failures = _check(failures, "the rekeyed host closes cleanly",
                              _wait_until(lambda: term.done, 20.0))
        finally:
            server._finish()
    return failures


def run_all(*, live: bool = False) -> int:
    """Run every API-layer test in file order; returns the failure count."""
    return (
        test_runtime_events() + test_headless_renderer()
        + test_session_store_lock() + test_ui_interactive()
        + test_session_store_cross_process_lock()
        + test_tool_cancellation() + test_run_once_json()
        + test_cancellation_coverage()
        + test_cancellation_destructive_guard()
        + test_file_mention_inlining()
        + test_tool_support_warning() + test_quiet_tools_live()
        + test_quiet_tools_real_turn()
        + test_approval_event_contract()
        + test_approval_runtime_contract()
        + test_unattended_turn_denial()
        + test_runtime_event_contract()
        + test_cancel_reaches_production_tools()
        + test_session_store_process_gate()
        + test_session_store_same_session_writers()
        + test_session_store_without_fcntl()
        + test_artifact_store() + test_artifact_quota_and_ttl()
        + test_tool_artifact_registration()
        + test_artifact_events() + test_artifact_session_persistence()
        + test_artifact_content_parts() + test_artifact_remote_fetch()
        + test_artifact_upload_api() + test_artifact_fetch_api()
        + test_python310_syntax() + test_pyflakes_clean()
        + test_module_size_budget()
        + test_turn_result_mapping()
        + test_turn_event_envelope()
        + test_turn_manager_lifecycle()
        + test_turn_manager_synthesized_terminal()
        + test_turn_buffer_truncation() + test_turn_expired()
        + test_turn_manager_session_busy() + test_turn_manager_max_active()
        + test_turn_manager_cancel() + test_turn_subscriber_overflow()
        + test_turn_manager_shutdown()
        + test_turn_manager_context() + test_turn_manager_headless_turn()
        + test_openai_mapping() + test_embeddings_api() + test_serve_http()
        + test_serve_disconnect() + test_serve_sockets()
        + test_serve_cache_concurrency()
        + test_serve_config_matrix()
        + test_chunked_bodies()
        + test_input_limits()
        + test_native_parameters()
        + test_workspace_jail()
        + test_artifact_root_containment()
        + test_serve_model_gate()
        + test_serve_agent_switch()
        + test_serve_shutdown()
        + test_serve_unload_on_exit()
        + test_serve_cli()
        + test_serve_process()
        + test_examples()
        + test_openai_sdk()
        + test_turn_mode_isolation()
        + test_agent_turn_lifecycle() + test_agent_turn_controls()
        + test_agent_approvals() + test_agent_approval_edge()
        + test_serve_approval_timeout_zero()
        + test_agent_policy_mode() + test_agent_disconnect_policy()
        + test_agent_turn_timeout() + test_agent_sessions_api()
        + test_agent_turn_validation() + test_agent_turn_expiry()
        + test_agent_context_warning()
        + test_session_context_fields()
        + test_fit_estimate_api() + test_fit_job_api()
        + test_turn_start_fit()
        + test_interrupt_guard()
        + test_session_host_protocol()
        + test_session_host_thinking_toggle()
        + test_session_host_detach_reattach()
        + test_session_host_approval()
        + test_session_host_cancel()
        + test_session_host_cancel_approval()
        + test_session_host_stop()
        + test_session_host_takeover()
        + test_session_host_rekey()
        + (test_serve_live() if live else 0))


if __name__ == "__main__":
    live = "--live" in sys.argv[1:]
    if live:
        print("--live: also running the real-backend smoke test")
    failures = run_all(live=live)
    if failures:
        print(f"\n{failures} failure(s)")
        sys.exit(1)
    print("\nAll tests passed.")
