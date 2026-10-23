"""Regression tests for ollamarun helpers, sessions and payloads.

The API-layer tests (agent runtime, turn manager, server, OpenAI and agent
surfaces, artifacts, context fitting) live in test_ollamarun_serve.py and run
from this file's canonical command.

Run: python3 test_ollamarun.py
"""
import base64
import contextlib
import csv
import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import requests
from rich.color import Color, ColorParseError
from rich.console import Console

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strips ANSI color codes so number/keyword highlighting doesn't split substrings."""
    return _ANSI_RE.sub("", text)

sys.path.insert(0, str(Path(__file__).parent))

import ollamarun_lib as orun

MATH_CASES = [
    ("inline compact",
     r"Energy is $E = mc^2$ exactly.",
     lambda out: "E = mc²" in out and "$" not in out),
    ("inline spaced dollars",
     r"Energy equals $ mc^2 $, where $ m $ is mass.",
     lambda out: "mc²" in out and "$" not in out),
    ("display dollars",
     "Thus\n$$a^2 + b^2 = c^2$$\nQED.",
     lambda out: "a² + b² = c²" in out and "$$" not in out),
    ("display brackets",
     r"Then \[ x_i^2 \le y \] holds.",
     lambda out: "xᵢ²" in out and "≤" in out and "\\[" not in out),
    ("inline parens",
     r"Where \( \alpha \le \beta \) inline.",
     lambda out: "α ≤ β" in out),
    ("code fence untouched",
     "Example:\n```latex\n$$not math$$\n```\nAnd $z_1^2$.",
     lambda out: "$$not math$$" in out and "z₁²" in out),
    ("inline code untouched",
     r"Use `x_i^2` code and $x_i^2$ math.",
     lambda out: "`x_i^2`" in out and "xᵢ²" in out),
    ("currency untouched",
     "It costs $5 and another $6 today.",
     lambda out: out == "It costs $5 and another $6 today."),
    ("bare boxed with nested group",
     r"Final Answer \boxed{\frac{1}{2}}",
     lambda out: out == "Final Answer [ 1/2 ]"),
    ("boxed inside dollars",
     r"value $\boxed{1}$ here",
     lambda out: out == "value [ 1 ] here"),
    ("boxed in code span untouched",
     r"code `\boxed{9}` done",
     lambda out: out == r"code `\boxed{9}` done"),
    ("fraction and greek",
     r"$$\int_0^\infty e^{-x^2}dx = \frac{\sqrt{\pi}}{2}$$",
     lambda out: "√" in out and "π" in out and "\\frac" not in out),
    ("math disabled",
     r"Keep $E = mc^2$.",
     None),
]


def test_math() -> int:
    failures = 0
    for name, text, check in MATH_CASES:
        if check is None:
            out = orun.convert_math(text, math=False)
            if out != text:
                print(f"FAIL  {name}: math=False changed the text: {out!r}")
                failures += 1
            else:
                print(f"PASS  {name}")
            continue
        out = orun.convert_math(text)
        if check(out):
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: got {out!r}")
            failures += 1
    return failures


def test_parsing_helpers() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    check("normalize_host default", orun.normalize_host(None), "http://127.0.0.1:11434")
    check("normalize_host named", orun.normalize_host("localhost"), "http://localhost:11434")
    check("normalize_host https port", orun.normalize_host("https://host:9999/x"),
          "https://host:9999")
    check("normalize_host ipv6", orun.normalize_host("[::1]:1234"),
          "http://[::1]:1234")
    check("parse_think high", orun.parse_think("high"), "high")
    check("parse_think bare", orun.parse_think("true"), True)
    check("parse_think off", orun.parse_think("false"), False)
    check("parse_think none", orun.parse_think(None), None)
    check("parse_bool fallback", orun.parse_bool("garbage", True), True)
    check("parse_bool false", orun.parse_bool("false", True), False)
    check("parse_format json", orun.parse_format("json"), "json")
    check("parse_format schema", orun.parse_format('{"type":"object"}'),
          {"type": "object"})
    check("parse_options", orun.parse_options(["temperature=0.7", "seed=42", "stop=END"]),
          {"temperature": 0.7, "seed": 42, "stop": "END"})
    check("coerce_value bool", orun.coerce_value("true"), True)
    check("shortcut", orun.shortcut("a" * 100, 10), "aaaaaaaaa…")
    check("human_bytes", orun.human_bytes(1536), "1.5 KiB")
    check("human_duration", orun.human_duration(0.25), "250 ms")

    try:
        orun.parse_think("sometimes")
        print("FAIL  parse_think rejects bad values")
        failures += 1
    except ValueError:
        print("PASS  parse_think rejects bad values")

    try:
        orun.parse_options(["temperature"])
        print("FAIL  parse_options requires key=value")
        failures += 1
    except ValueError:
        print("PASS  parse_options requires key=value")

    return failures


def test_stats_helpers() -> int:
    failures = 0
    final = {
        "done_reason": "stop",
        "total_duration": 6_000_000_000,
        "load_duration": 1_000_000_000,
        "prompt_eval_count": 30,
        "prompt_eval_duration": 300_000_000,
        "eval_count": 100,
        "eval_duration": 2_000_000_000,
    }
    stats = orun.stats_from_final(final)
    checks = [
        ("stats total", stats["total_duration"], 6.0),
        ("stats eval tps", round(stats["eval_tps"], 1), 50.0),
        ("stats prompt tps", round(stats["prompt_tps"]), 100),
        ("stats done reason", stats["done_reason"], "stop"),
    ]
    for name, got, expected in checks:
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1
    text = orun.format_stats(stats)
    if "50.0 tok/s" in text and "load 1.0 s" in text:
        print("PASS  format_stats")
    else:
        print(f"FAIL  format_stats: {text!r}")
        failures += 1
    return failures


def test_resolve_show_stats() -> int:
    failures = 0
    cases = [
        ("--verbose wins over the off-switch", (True, False, False), True),
        ("--verbose forces timings in piped output", (True, True, False), True),
        ("off-switch silences the terminal default", (False, False, True), False),
        ("terminal default shows timings", (False, True, True), True),
        ("piped output stays clean by default", (False, True, False), False),
    ]
    for name, (verbose, config_value, is_terminal), expected in cases:
        got = orun.resolve_show_stats(verbose, config_value, is_terminal)
        if got == expected:
            print(f"PASS  resolve_show_stats: {name}")
        else:
            print(f"FAIL  resolve_show_stats: {name}: {got!r} != {expected!r}")
            failures += 1
    return failures


def test_file_mentions() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "note.txt").write_text("hello file\n")
        text, included, warnings = orun.expand_file_mentions(
            "see @note.txt and @missing.txt", base=base, max_kb=64)
        if len(included) == 1 and "hello file" in text and warnings:
            print("PASS  expand_file_mentions")
        else:
            print(f"FAIL  expand_file_mentions: {included!r} {warnings!r}")
            failures += 1
        big = base / "big.txt"
        big.write_text("x" * 5000)
        _, included, warnings = orun.expand_file_mentions(
            "@big.txt", base=base, max_kb=1)
        if not included and warnings and "exceeds" in warnings[0]:
            print("PASS  expand_file_mentions size guard")
        else:
            print(f"FAIL  size guard: {included!r} {warnings!r}")
            failures += 1

        text, included, warnings = orun.expand_file_mentions(
            "Summarize @note.txt. One word.", base=base, max_kb=64)
        if len(included) == 1 and included[0].endswith("note.txt") and not warnings \
                and "hello file" in text:
            print("PASS  expand_file_mentions strips a sentence-final period")
        else:
            print(f"FAIL  sentence-final period: {included!r} {warnings!r}")
            failures += 1

        text, included, warnings = orun.expand_file_mentions(
            "See (@note.txt) and @note.txt, then @note.txt!", base=base, max_kb=64)
        if len(included) == 3 and not warnings \
                and all(path.endswith("note.txt") for path in included):
            print("PASS  expand_file_mentions strips closing punctuation")
        else:
            print(f"FAIL  closing punctuation: {included!r} {warnings!r}")
            failures += 1

        odd = base / "odd.txt."
        odd.write_text("punctuation is part of the name\n")
        _, included, warnings = orun.expand_file_mentions(
            "@odd.txt.", base=base, max_kb=64)
        if included == [str(odd)] and not warnings:
            print("PASS  a real trailing-punctuation filename resolves as typed")
        else:
            print(f"FAIL  raw-vs-stripped filename: {included!r} {warnings!r}")
            failures += 1

        _, included, warnings = orun.expand_file_mentions(
            "@missing.txt.", base=base, max_kb=64)
        if not included and warnings and warnings[0] == "@missing.txt.: no such file":
            print("PASS  a genuine miss still warns with the mention as typed")
        else:
            print(f"FAIL  genuine miss: {included!r} {warnings!r}")
            failures += 1
    return failures


def _tiny_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    raw = b"\x00" + b"\xff\x00\x00"
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def test_images() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "red.png"
        path.write_bytes(_tiny_png())
        record = orun.encode_image(path, "embed")
        if record["name"] == "red.png" and record["b64"] and record["sha1"]:
            print("PASS  encode_image embeds base64")
        else:
            print(f"FAIL  encode_image: {record}")
            failures += 1
        payload = orun.image_payload([record])
        if len(payload) == 1 and base64.b64decode(payload[0]) == path.read_bytes():
            print("PASS  image_payload round trip")
        else:
            print("FAIL  image_payload round trip")
            failures += 1
        missing = {"path": str(Path(tmp) / "nope.png"), "name": "nope.png"}
        if orun.image_payload([missing]) == []:
            print("PASS  image_payload skips missing files")
        else:
            print("FAIL  image_payload missing file")
            failures += 1
    return failures


def test_cmd_image() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        spaced = Path(tmp) / "my photo.png"
        spaced.write_bytes(_tiny_png())
        plain = Path(tmp) / "plain.png"
        plain.write_bytes(_tiny_png())

        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient()
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        with orun.console.capture() as cap:
            orun.cmd_image(repl, f'"{spaced}"')
        out = _plain(cap.get())
        if len(repl.pending_images) == 1 and "my photo.png" in out:
            print("PASS  /image attaches a quoted path containing spaces")
        else:
            print(f"FAIL  /image quoted path: {out!r} pending={repl.pending_images}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_image(repl, str(plain))
        if len(repl.pending_images) == 2:
            print("PASS  /image still accepts an unquoted plain path")
        else:
            print(f"FAIL  /image unquoted path: pending={repl.pending_images}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_image(repl, f'"{spaced}')
        out = _plain(cap.get())
        if len(repl.pending_images) == 2 and out.strip():
            print("PASS  /image reports an error on unbalanced quoting without crashing")
        else:
            print(f"FAIL  /image unbalanced quoting: {out!r} pending={repl.pending_images}")
            failures += 1
    return failures


def test_cli_image_errors() -> int:
    """A bad --image names the problem and the failed path (#40)."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        client = _FakeClient()
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None

        missing = str(Path(tmp) / "does-not-exist.png")
        with orun.console.capture() as cap:
            code = orun.attach_images(repl, [missing])
        out = _plain(cap.get())
        if code == 2 and "image not found" in out and missing in out \
                and not repl.pending_images:
            print("PASS  a missing --image names the problem and the path")
        else:
            print(f"FAIL  missing image: {code} {out!r}")
            failures += 1

        empty = Path(tmp) / "empty.png"
        empty.write_bytes(b"")
        with orun.console.capture() as cap:
            code = orun.attach_images(repl, [str(empty)])
        out = _plain(cap.get())
        if code == 2 and "cannot attach image" in out and "empty file" in out \
                and str(empty) in out and not repl.pending_images:
            print("PASS  an empty --image reports the path and the reason")
        else:
            print(f"FAIL  empty image: {code} {out!r}")
            failures += 1

        good = Path(tmp) / "red.png"
        good.write_bytes(_tiny_png())
        if orun.attach_images(repl, [str(good)]) is None \
                and len(repl.pending_images) == 1:
            print("PASS  a good --image attaches without an exit code")
        else:
            print(f"FAIL  good image: {repl.pending_images}")
            failures += 1
    return failures


def test_session_store() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                                  index_path=Path(tmp) / "index.json")
        session = orun.Session.create("m1", "http://127.0.0.1:11434", title="first")
        session.add_message("user", "hello world")
        session.add_message("assistant", "hi there", thinking="thought")
        session.record_turn({"prompt_tokens": 5, "completion_tokens": 7,
                             "total_duration": 1.5, "eval_tps": 12.0})
        store.save(session)

        loaded = store.load(session.id)
        if loaded.title == "first" and len(loaded.messages) == 2 and loaded.stats["turns"] == 1:
            print("PASS  session save/load round trip")
        else:
            print(f"FAIL  round trip: {loaded}")
            failures += 1

        if store.resolve(session.id[:12]) and store.resolve("first") \
                and store.resolve("FIRST").id == session.id:
            print("PASS  resolve by prefix and title")
        else:
            print("FAIL  resolve by prefix/title")
            failures += 1

        entries = store.entries()
        if len(entries) == 1 and entries[0]["messages"] == 2:
            print("PASS  index entry")
        else:
            print(f"FAIL  index entry: {entries}")
            failures += 1

        hits = store.search("HELLO")
        if len(hits) == 1 and "hello" in hits[0][1].lower():
            print("PASS  search finds message text")
        else:
            print(f"FAIL  search: {hits}")
            failures += 1

        with contextlib.redirect_stdout(io.StringIO()) as out:
            html = orun.export_html(loaded)
        if orun.export_markdown(loaded).startswith("# first") \
                and '"schema": 1' in orun.export_json(loaded) \
                and "<html" in html and out.getvalue() == "":
            print("PASS  exports md/json/html")
        else:
            print(f"FAIL  exports: stdout={out.getvalue()!r}")
            failures += 1

        other = orun.Session.create("m2", "http://127.0.0.1:11434", title="second")
        other.add_message("user", "another")
        store.save(other)
        if store.latest().id == other.id and store.latest(model="m1").id == session.id:
            print("PASS  latest and latest(model=...)")
        else:
            print("FAIL  latest")
            failures += 1

        store.delete(session.id, use_trash=False)
        if store.resolve(session.id) is None and len(store.entries()) == 1:
            print("PASS  delete removes file and index entry")
        else:
            print("FAIL  delete")
            failures += 1

        store.rebuild_index()
        if len(store.entries()) == 1:
            print("PASS  rebuild_index")
        else:
            print("FAIL  rebuild_index")
            failures += 1
    return failures


def test_session_payloads() -> int:
    failures = 0
    session = orun.Session.create("m", "http://h")
    session.add_message("user", "hello", images=[{"name": "x.png",
                                                  "b64": base64.b64encode(b"x").decode()}])
    session.add_message("assistant", "partial", cancelled=True)
    session.add_message("assistant", "", cancelled=True)
    payloads = session.message_payloads()
    if len(payloads) == 2 and payloads[0].get("images") == [base64.b64encode(b"x").decode()] \
            and "thinking" not in payloads[0]:
        print("PASS  message_payloads attaches images, drops empty cancelled")
    else:
        print(f"FAIL  message_payloads: {payloads}")
        failures += 1
    if session.last_user_index() == 0:
        print("PASS  last_user_index")
    else:
        print("FAIL  last_user_index")
        failures += 1
    return failures


def test_chat_payload() -> int:
    failures = 0
    payload = orun.build_chat_payload("m", [{"role": "user", "content": "hi"}],
                                      stream=True, fmt="json", think="high",
                                      keepalive="5m", options={"temperature": 0.2})
    expected = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                "stream": True, "format": "json", "think": "high",
                "keep_alive": "5m", "options": {"temperature": 0.2}}
    if payload == expected:
        print("PASS  build_chat_payload")
    else:
        print(f"FAIL  build_chat_payload: {payload}")
        failures += 1
    lean = orun.build_chat_payload("m", [], stream=False)
    if lean == {"model": "m", "messages": [], "stream": False}:
        print("PASS  build_chat_payload omits unset fields")
    else:
        print(f"FAIL  lean payload: {lean}")
        failures += 1
    return failures


def test_tool_definitions() -> int:
    failures = 0
    defs = orun.tool_definitions()
    names = sorted(definition.name for definition in defs)
    failures = _check(failures, "the shipped tool definitions load",
                      names == ["chem_container", "chem_locations", "chem_search",
                                "context_status", "copy_file", "delete_file",
                                "edit_file", "fetch_url", "list_dir",
                                "list_models", "list_voices", "make_dir",
                                "move_file", "publish_artifact", "read_file",
                                "read_pdf", "run_python", "run_shell",
                                "search_sessions", "session_info", "speak",
                                "system_status", "web_search", "write_file"],
                      str(names))
    failures = _check(failures, "every definition's handler resolves",
                      all(definition.handler in orun.HANDLERS
                          for definition in defs))
    web = next((definition for definition in defs
                if definition.name == "web_search"), None)
    failures = _check(failures, "web_search declares its capability and aliases",
                      web is not None and web.read_only
                      and web.requires == "search" and "google" in web.aliases)
    publish = next((definition for definition in defs
                    if definition.name == "publish_artifact"), None)
    failures = _check(failures, "publish_artifact needs the artifact store",
                      publish is not None and publish.requires == "artifacts"
                      and not publish.read_only
                      and not publish.always_safe, str(publish))
    run_python = next((definition for definition in defs
                       if definition.name == "run_python"), None)
    failures = _check(failures, "run_python prefers code over mental math",
                      run_python is not None
                      and "instead of doing math mentally" in run_python.description,
                      str(run_python))
    with_search = orun.default_registry(tts=False, search=True, chem=False)
    without_search = orun.default_registry(tts=False, search=False, chem=False)
    failures = _check(failures, "requires gates web_search and the speech tools",
                      "web_search" in with_search
                      and "web_search" not in without_search
                      and "speak" not in without_search
                      and "fetch_url" in without_search)
    with_chem = orun.default_registry(tts=False, search=False, chem=True)
    without_chem = orun.default_registry(tts=False, search=False, chem=False)
    failures = _check(failures, "requires gates the chem inventory tools",
                      "chem_search" in with_chem and "chem_container" in with_chem
                      and "chem_locations" in with_chem
                      and "chem_search" not in without_chem)
    failures = _check(failures, "the chem tools are read-only",
                      with_chem["chem_search"].read_only
                      and with_chem["chem_container"].read_only
                      and with_chem["chem_locations"].read_only)
    failures = _check(failures, "aliases are built from the definitions",
                      orun.resolve_tool_alias("google") == "web_search"
                      and orun.resolve_tool_alias("browse") == "fetch_url"
                      and orun.resolve_tool_alias("unheard_of") == "unheard_of")
    failures = _check(failures, "tool-call id prefixes are stripped on a miss",
                      orun.resolve_tool_alias("call_fetch_url") == "fetch_url"
                      and orun.resolve_tool_alias("Call_Google") == "web_search"
                      and orun.resolve_tool_alias("tool_curl") == "fetch_url"
                      and orun.resolve_tool_alias("tool_run_shell") == "run_shell"
                      and orun.resolve_tool_alias("functions.web_search") == "web_search"
                      and orun.resolve_tool_alias("call_whatever") == "call_whatever")

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "tools"
        override = Path(tmp) / "override"
        base.mkdir()
        override.mkdir()
        schema = {"type": "object", "properties": {}, "required": []}

        def write(folder: Path, filename: str, payload: dict) -> None:
            (folder / filename).write_text(json.dumps(payload), encoding="utf-8")

        write(base, "alpha.json", {"name": "alpha", "description": "first",
                                   "parameters": schema, "handler": "list_dir",
                                   "read_only": True})
        write(base, "beta.json", {"name": "alpha", "description": "duplicate",
                                  "parameters": schema, "handler": "list_dir"})
        (base / "broken.json").write_text("{not json", encoding="utf-8")
        write(base, "nosuch.json", {"name": "nosuch", "description": "x",
                                    "parameters": schema,
                                    "handler": "does_not_exist"})
        with orun.console.capture() as cap:
            loaded = orun.load_tool_definitions([base, override])
        out = _plain(cap.get())
        failures = _check(failures, "bad files and duplicate names are skipped",
                          [definition.name for definition in loaded] == ["alpha"],
                          str([definition.name for definition in loaded]))
        failures = _check(failures, "bad definitions warn and continue",
                          "ignoring tool definition" in out
                          and "duplicate name" in out and "unknown handler" in out,
                          out)

        write(override, "alpha.json", {"name": "alpha", "description": "overridden",
                                       "parameters": schema, "handler": "list_dir"})
        with orun.console.capture():
            loaded = orun.load_tool_definitions([base, override])
        alpha = next((definition for definition in loaded
                      if definition.name == "alpha"), None)
        failures = _check(failures, "a later layer overrides a file by name",
                          alpha is not None and alpha.description == "overridden",
                          str(alpha))
    return failures


def test_prompts() -> int:
    failures = 0
    base = orun.default_system_prompt(with_tools=False) or ""
    with_tools = orun.default_system_prompt(with_tools=True) or ""
    failures = _check(failures, "the built-in system prompt loads from its file",
                      bool(base) and "ollamarun" in base, base[:60])
    failures = _check(failures, "the tool rules are appended only with tools on",
                      with_tools.startswith(base) and "Web search" in with_tools
                      and with_tools != base, with_tools[:80])
    failures = _check(failures, "the tool rules push math into code",
                      "run_python" in with_tools
                      and "never work arithmetic out in your head" in with_tools
                      and "run_python" not in base, with_tools)
    failures = _check(failures, "the tool rules say a result is not a user turn",
                      "not user messages" in with_tools
                      and "fix the call and retry" in with_tools
                      and "never reply to the error" in with_tools, with_tools)
    title = orun.title_prompt("the first message")
    failures = _check(failures, "the title prompt substitutes its content",
                      title is not None and "the first message" in title
                      and "{{content}}" not in title, str(title))
    summary = orun.summarize_prompt("user: hi\n\nassistant: hello")
    failures = _check(failures, "the summarize prompt substitutes its transcript",
                      summary is not None and "user: hi" in summary
                      and "{{content}}" not in summary, str(summary))
    failures = _check(failures, "a marker-less template appends the content",
                      orun.render_prompt("just a title", "BODY")
                      == "just a title\n\nBODY")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "prompts"
        folder.mkdir()
        (folder / "system.md").write_text("OVERRIDE FIRST", encoding="utf-8")
        loaded = orun.load_prompt("system.md", directories=[folder])
        failures = _check(failures, "an explicit override layer is read first",
                          loaded == "OVERRIDE FIRST", str(loaded))
        with orun.console.capture() as cap:
            missing = orun.load_prompt("nope.md", directories=[folder])
        failures = _check(failures, "a missing prompt reads as None and warns",
                          missing is None and "not found" in _plain(cap.get()))
    return failures


def test_system_prompt() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        config = dict(orun.DEFAULT_CONFIG)
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"])
        repl = orun.Repl(args, config, client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None

        repl.session.add_message("user", "hi")
        with orun.console.capture():
            repl._generate()
        first = client.chat_payloads[0]["messages"]
        failures = _check(failures, "the built-in default leads the chat request",
                          first[0]["role"] == "system"
                          and "ollamarun" in first[0]["content"], str(first[:1]))
        failures = _check(failures, "the system prompt stays out of the history",
                          all(message.get("role") != "system"
                              for message in repl.session.messages))

        repl.session.system = "BE TERSE"
        with orun.console.capture():
            repl._generate()
        second = client.chat_payloads[1]["messages"]
        failures = _check(failures, "a session prompt replaces the built-in default",
                          second[0] == {"role": "system", "content": "BE TERSE"},
                          str(second[:1]))

        repl.session.system = ""
        with orun.console.capture():
            repl._generate()
        third = client.chat_payloads[2]["messages"]
        failures = _check(failures, "an empty session prompt sends no system message",
                          all(message.get("role") != "system" for message in third))

        with orun.console.capture() as cap:
            orun.cmd_system(repl, "clear")
        failures = _check(failures, "/system clear resets to the built-in default",
                          repl.session.system is None
                          and "built-in default" in _plain(cap.get()), cap.get())
        with orun.console.capture() as cap:
            orun.cmd_system(repl, "")
        failures = _check(failures, "/system shows the built-in prompt",
                          "ollamarun" in _plain(cap.get()), cap.get())
        with orun.console.capture():
            orun.cmd_system(repl, "off")
        failures = _check(failures, "/system off disables the prompt",
                          repl.session.system == "")
        with orun.console.capture() as cap:
            orun.cmd_system(repl, "")
        failures = _check(failures, "/system reports the disabled state",
                          "disabled" in _plain(cap.get()), cap.get())
        with orun.console.capture():
            orun.cmd_system(repl, "be nice")
        failures = _check(failures, "/system TEXT sets a custom prompt",
                          repl.session.system == "be nice")
        with orun.console.capture():
            orun.cmd_set(repl, "system off")
        failures = _check(failures, "/set system off disables it",
                          repl.session.system == "")
        with orun.console.capture():
            orun.cmd_set(repl, "system clear")
        failures = _check(failures, "/set system clear resets to the default",
                          repl.session.system is None)
        repl.session.system = "be nice"
        with orun.console.capture() as cap:
            orun.cmd_params(repl, "")
        failures = _check(failures, "the parameter table shows the system state",
                          "be nice" in _plain(cap.get()), cap.get())

        repl.auto_title = True
        with orun.console.capture():
            repl._maybe_title("plan the trip")
        title_text = client.chat_once_payloads[-1]["messages"][0]["content"]
        failures = _check(failures, "auto-title uses the prompt file",
                          "plan the trip" in title_text
                          and "starts with" in title_text, title_text[:80])

        repl.session.messages = [
            {"role": "user", "content": "one", "ts": ""},
            {"role": "assistant", "content": "two", "ts": ""},
            {"role": "user", "content": "three", "ts": ""},
            {"role": "assistant", "content": "four", "ts": ""},
            {"role": "user", "content": "five", "ts": ""},
        ]
        with orun.console.capture():
            orun.cmd_summarize(repl, "go")
        summary_text = client.chat_once_payloads[-1]["messages"][0]["content"]
        failures = _check(failures, "summarize uses the prompt file",
                          "Summarize the following conversation" in summary_text
                          and "user: one" in summary_text, summary_text[:80])
        failures = _check(failures, "the summary replaces the older messages",
                          repl.session.messages[0]["content"].startswith(
                              "[Summary of earlier conversation]"),
                          str(repl.session.messages[0])[:80])

        args_no = orun.build_parser().parse_args(["m1", "--no-system"])
        failures = _check(failures, "--no-system parses",
                          args_no.no_system is True
                          and orun.build_parser().parse_args(["m1"]).no_system is False)
        fresh = orun.resolve_session(args_no, dict(orun.DEFAULT_CONFIG), client, store)
        failures = _check(failures, "--no-system disables the prompt for a new session",
                          fresh.system == "", repr(fresh.system))
        args_sys = orun.build_parser().parse_args(["m1", "--system", "custom text"])
        fresh = orun.resolve_session(args_sys, dict(orun.DEFAULT_CONFIG), client, store)
        failures = _check(failures, "--system sets the session prompt",
                          fresh.system == "custom text", repr(fresh.system))
    return failures


def test_normalize_argv() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                                  index_path=Path(tmp) / "index.json")
        session = orun.Session.create("m", "h", title="named session")
        session.add_message("user", "x")
        store.save(session)

        cases = [
            ("bare -c with prompt", ["-c", "hello world", "--render", "plain"],
             ["--continue=__latest__", "hello world", "--render", "plain"]),
            ("-c with session title and prompt", ["-c", "named session", "hi"],
             ["-c", "named session", "hi"]),
            ("-c with a title equal to the whole prompt", ["-c", "named session"],
             ["--continue=__latest__", "named session"]),
            ("-c with a title substring", ["-c", "named"],
             ["--continue=__latest__", "named"]),
            ("-c with a title and only flags after it", ["-c", "named session", "--no-save"],
             ["--continue=__latest__", "named session", "--no-save"]),
            ("-c with a title, flags and a prompt",
             ["-c", "named session", "--render", "plain", "hi"],
             ["-c", "named session", "--render", "plain", "hi"]),
            ("-c with prefix", ["-c", session.id[:12], "hi"],
             ["-c", session.id[:12], "hi"]),
            ("-c with id-like miss", ["-c", "20260101-000000-dead", "hi"],
             ["-c", "20260101-000000-dead", "hi"]),
            ("-c followed by option", ["-c", "--render", "plain", "hi"],
             ["-c", "--render", "plain", "hi"]),
            ("--sort fuzzy TEXT folds into one token",
             ["--sort", "fuzzy", "qwen 3b", "--list"],
             ["--sort=fuzzy:qwen 3b", "--list"]),
            ("--sort=fuzzy TEXT folds the query in",
             ["--sort=fuzzy", "qwen3"], ["--sort=fuzzy:qwen3"]),
            ("--sort with a key is untouched", ["--sort", "size", "m1"],
             ["--sort", "size", "m1"]),
            ("--sort fuzzy without a query is left for the parser",
             ["--sort", "fuzzy"], ["--sort", "fuzzy"]),
        ]
        parser = orun.build_parser()
        for name, argv, expected in cases:
            got = orun.normalize_argv(argv, store, parser)
            if got == expected:
                print(f"PASS  normalize_argv {name}")
            else:
                print(f"FAIL  normalize_argv {name}: {got!r} != {expected!r}")
                failures += 1
    return failures


def test_renderer_plain() -> int:
    failures = 0
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    renderer = orun.Renderer(console, mode="plain", math=True, show_stats=True)

    def events():
        yield {"message": {"thinking": "hmm "}}
        yield {"message": {"content": "The value is "}}
        yield {"message": {"content": "$E = mc^2$."}}
        yield {"done": True, "eval_count": 4, "eval_duration": 1_000_000_000,
               "prompt_eval_count": 2, "prompt_eval_duration": 1_000_000_000,
               "total_duration": 2_000_000_000}

    result = renderer.stream(events())
    output = buffer.getvalue()
    if result.content == "The value is $E = mc^2$." and "hmm" in output \
            and "4.0 tok/s" in output:
        print("PASS  renderer plain streaming keeps raw content and stats")
    else:
        print(f"FAIL  renderer: {result.content!r} / {output!r}")
        failures += 1

    result = renderer.stream(iter([{"message": {"content": "hi"}}]))
    if not result.stats:
        print("PASS  renderer without final stats")
    else:
        print("FAIL  renderer without final stats")
        failures += 1

    quiet_buffer = io.StringIO()
    quiet = orun.Renderer(Console(file=quiet_buffer, width=100, force_terminal=False),
                          mode="plain", math=True)
    quiet.stream(events())
    if "tok/s" not in quiet_buffer.getvalue():
        print("PASS  renderer hides stats unless show_stats is set")
    else:
        print(f"FAIL  renderer stats default: {quiet_buffer.getvalue()!r}")
        failures += 1
    return failures


def test_renderer_feed_parity() -> int:
    """A remote `begin/feed/end` round renders exactly like `stream()`."""
    failures = 0
    stats = {"eval_count": 4, "eval_duration": 1.0,
             "prompt_eval_count": 2, "prompt_eval_duration": 1.0,
             "total_duration": 2.0}

    def build(buffer):
        return orun.Renderer(
            Console(file=buffer, width=100, force_terminal=False,
                    theme=orun.THEME),
            mode="plain", math=False, thinking_mode="collapsed",
            tps_footer=False, show_stats=False)

    def streamed():
        buffer = io.StringIO()
        result = build(buffer).stream(iter([
            {"message": {"thinking": "hmm "}},
            {"message": {"content": "The value is "}},
            {"message": {"content": "42."}},
            {"done": True, **stats},
        ]))
        return buffer.getvalue(), result

    def fed(stats_arg):
        buffer = io.StringIO()
        renderer = build(buffer)
        renderer.begin()
        renderer.feed("thinking", "hmm ")
        renderer.feed("content", "The value is ")
        renderer.feed("content", "42.")
        result = renderer.end(content="The value is 42.",
                              thinking="hmm ", stats=stats_arg)
        return buffer.getvalue(), result

    local_text, local_result = streamed()
    # The host forwards the already-normalized `message.completed.stats`.
    remote_text, remote_result = fed(local_result.stats)
    failures = _check(
        failures, "feed/end renders the same plain text as stream()",
        local_text == remote_text, f"{local_text!r} != {remote_text!r}")
    failures = _check(
        failures, "feed/end returns the same accumulated content",
        (local_result.content, local_result.thinking, local_result.stats)
        == (remote_result.content, remote_result.thinking, remote_result.stats),
        f"{local_result!r} != {remote_result!r}")
    # An explicit final hint wins over the after_tools heuristic.
    probe = orun.GenerationResult(content="x", final=True)
    renderer = build(io.StringIO())
    failures = _check(failures, "an explicit final flag controls the divider",
                      renderer.final_rule_wanted(probe) is True)
    probe.final = False
    failures = _check(failures, "a false final flag suppresses the divider",
                      renderer.final_rule_wanted(probe) is False)
    return failures


def test_renderer_stop_keys() -> int:
    """Ctrl+Q stops a stream, Ctrl+D requests a close, Ctrl+T still toggles."""
    failures = 0
    buffer = io.StringIO()
    renderer = orun.Renderer(
        Console(file=buffer, width=100, force_terminal=False, theme=orun.THEME),
        mode="plain", math=False, tps_footer=False, show_stats=False)
    failures = _check(failures, "Ctrl+Q sets the stop request",
                      renderer._handle_key_bytes(b"\x11")
                      and renderer.stop_requested
                      and not renderer.close_requested)
    renderer.begin()
    failures = _check(failures, "begin() resets the key requests",
                      not renderer.stop_requested
                      and not renderer.close_requested)
    failures = _check(failures, "Ctrl+D sets the close request",
                      renderer._handle_key_bytes(b"\x04")
                      and renderer.close_requested)
    renderer.begin()
    before = renderer.thinking_mode
    renderer._handle_key_bytes(b"\x14")
    failures = _check(failures, "Ctrl+T still toggles thinking",
                      renderer.thinking_mode != before)

    # A stream whose producer sets the flag mid-iteration stops there and
    # reports cancelled (the foreground Ctrl+Q path).
    def events():
        yield {"message": {"content": "first"}}
        renderer.stop_requested = True
        yield {"message": {"content": "second"}}
        yield {"message": {"content": "third"}}

    result = renderer.stream(events())
    failures = _check(
        failures, "a stop request cancels the stream and stops consuming",
        result.cancelled and result.content == "firstsecond"
        and "third" not in result.content, repr(result.content))
    return failures


def test_stream_error_reporting() -> int:
    """A mid-stream backend error is reported, not a TypeError crash (#38)."""
    failures = 0

    def check(name, condition, detail=""):
        nonlocal failures
        if condition:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {detail}")
            failures += 1

    err_buffer = io.StringIO()
    stderr_console = Console(file=err_buffer, width=100, stderr=True)
    out_buffer = io.StringIO()
    console = Console(file=out_buffer, width=100, force_terminal=False)
    renderer_err = orun.renderer.err_console
    repl_err = orun.repl.err_console
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                                  index_path=Path(tmp) / "index.json")
        try:
            orun.renderer.err_console = stderr_console
            raw = orun.Renderer(console, mode="raw")
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = raw.stream(iter([{"message": {"content": "partial "}},
                                          {"error": "mid-stream boom"}]))
            check("raw mid-stream error reports on stderr",
                  result.error == "mid-stream boom"
                  and "ollama error" in _plain(err_buffer.getvalue())
                  and "mid-stream boom" in err_buffer.getvalue(),
                  f"{result.error!r} / {err_buffer.getvalue()!r}")
            check("raw mode still writes the partial content to stdout",
                  stdout.getvalue() == "partial \n", repr(stdout.getvalue()))

            out_buffer.seek(0); out_buffer.truncate(0)
            err_buffer.seek(0); err_buffer.truncate(0)
            normal = orun.Renderer(console, mode="markdown", thinking_mode="hidden")
            result = normal.stream(iter([{"error": "normal-mode boom"}]))
            check("non-raw renderer leaves error reporting to the runtime event",
                  result.error == "normal-mode boom"
                  and not out_buffer.getvalue() and not err_buffer.getvalue(),
                  f"{out_buffer.getvalue()!r} / {err_buffer.getvalue()!r}")

            client = _FakeClient(chat_script=[[{"message": {"content": "partial "}},
                                               {"error": "mid-stream boom"}]])
            args = orun.build_parser().parse_args(["m1", "--render", "plain"])
            repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                             orun.Session.create("m1", client.base))
            repl._build_prompt_session = lambda: None
            with orun.console.capture() as cap:
                result = repl._generate()
            text = _plain(cap.get())
            check("a full turn reports the mid-stream error exactly once",
                  result.error == "mid-stream boom"
                  and text.count("mid-stream boom") == 1,
                  f"{result.error!r} / {text!r}")

            orun.repl.err_console = stderr_console
            client = _FakeClient(resident=["m1"], models=["m1"])

            def exploding_unload(model):
                raise orun.OllamaError("unload exploded")

            client.unload = exploding_unload
            args = orun.build_parser().parse_args(["m1", "--render", "plain"])
            repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                             orun.Session.create("m1", client.base))
            repl._build_prompt_session = lambda: None

            err_buffer.seek(0); err_buffer.truncate(0)
            ok = repl.unload_model("m1", quiet=True)
            check("quiet unload failure warns on stderr",
                  ok is False and "could not unload" in _plain(err_buffer.getvalue())
                  and "unload exploded" in err_buffer.getvalue(),
                  f"{err_buffer.getvalue()!r}")

            with orun.console.capture() as cap:
                ok = repl.unload_model("m1", quiet=False)
            check("loud unload failure warns on the console",
                  ok is False and "could not unload" in _plain(cap.get()),
                  cap.get())
        finally:
            orun.renderer.err_console = renderer_err
            orun.repl.err_console = repl_err
    return failures


def test_thinking_display() -> int:
    failures = 0

    def check(name, condition, detail=""):
        nonlocal failures
        if condition:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {detail}")
            failures += 1

    def render_text(renderable) -> str:
        buffer = io.StringIO()
        Console(file=buffer, width=100, force_terminal=False).print(renderable)
        return buffer.getvalue()

    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    renderer = orun.Renderer(console, mode="markdown", thinking_mode="collapsed")
    result = orun.GenerationResult()

    check("live view waiting spinner",
          "waiting for model" in render_text(renderer._live_view(result)))
    result.thinking = "deep reasoning " * 20
    collapsed = render_text(renderer._live_view(result))
    check("collapsed summary line",
          "Ctrl+T to expand" in collapsed and "waiting for model" not in collapsed)
    renderer._toggle_thinking()
    check("toggle collapsed -> expanded", renderer.thinking_mode == "expanded")
    expanded = render_text(renderer._live_view(result))
    check("expanded panel", "Ctrl+T to collapse" in expanded and "deep reasoning" in expanded)
    result.content = "The answer."
    with_content = render_text(renderer._live_view(result))
    check("content with expanded panel",
          "The answer." in with_content and "Ctrl+T to collapse" in with_content)
    renderer._toggle_thinking()
    check("toggle expanded -> collapsed", renderer.thinking_mode == "collapsed")
    renderer.thinking_mode = "hidden"
    hidden = render_text(renderer._live_view(result))
    check("hidden hides thinking",
          "deep reasoning" not in hidden and "Ctrl+T" not in hidden)
    renderer._toggle_thinking()
    check("hidden -> expanded on toggle", renderer.thinking_mode == "expanded")

    long_text = "\n".join(f"line {index}" for index in range(50))
    preview = render_text(orun.thinking_panel(long_text, preview_lines=5))
    check("thinking_panel tail preview",
          "line 49" in preview and "line 45" in preview and "line 0" not in preview)

    check("human_chars short", orun.human_chars(999) == "999 chars")
    check("human_chars kilo", orun.human_chars(1500) == "1.5k chars")
    check("non-tty renderer not interactive", renderer.interactive is False)

    read_fd, write_fd = os.pipe()
    try:
        renderer.thinking_mode = "collapsed"
        os.write(write_fd, b"x" + bytes([orun.CTRL_T]))
        os.close(write_fd)
        changed = renderer._poll_keys(read_fd)
        check("Ctrl+T poll expands inline (no pager)",
              changed and renderer.thinking_mode == "expanded"
              and not hasattr(renderer, "_pager_request"))
    finally:
        os.close(read_fd)

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, bytes([orun.CTRL_T]))
        os.close(write_fd)
        changed = renderer._poll_keys(read_fd)
        check("Ctrl+T poll collapses again",
              changed and renderer.thinking_mode == "collapsed")
    finally:
        os.close(read_fd)

    hidden_buffer = io.StringIO()
    hidden_renderer = orun.Renderer(Console(file=hidden_buffer, width=100),
                                    mode="plain", thinking_mode="hidden")
    hidden_renderer.stream(iter([{"message": {"thinking": "SECRET"}},
                                 {"message": {"content": "hi"}}, {"done": True}]))
    check("plain mode honors hidden", "SECRET" not in hidden_buffer.getvalue())

    visible_buffer = io.StringIO()
    visible_renderer = orun.Renderer(Console(file=visible_buffer, width=100),
                                     mode="plain", thinking_mode="collapsed")
    visible_renderer.stream(iter([{"message": {"thinking": "VISIBLE"}},
                                  {"message": {"content": "hi"}}, {"done": True}]))
    check("plain mode keeps visible thinking", "VISIBLE" in visible_buffer.getvalue())

    toggles = []
    callback_renderer = orun.Renderer(console, mode="plain",
                                      thinking_mode="collapsed",
                                      on_thinking_toggle=toggles.append)
    callback_renderer._handle_key_bytes(bytes([orun.CTRL_T]))
    callback_renderer._handle_key_bytes(bytes([orun.CTRL_T]))
    check("Ctrl+T reports each new mode to the front end",
          toggles == ["expanded", "collapsed"], str(toggles))

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        client = _FakeClient(models=["m1"])
        args = orun.build_parser().parse_args(
            ["m1", "--render", "plain", "--no-tools", "--no-tps",
             "--keep-loaded"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl.thinking_mode = "collapsed"
        live = repl._renderer()
        live._handle_key_bytes(bytes([orun.CTRL_T]))
        check("Ctrl+T updates the session's thinking mode",
              repl.thinking_mode == "expanded", repl.thinking_mode)
        check("the next round starts with the toggled mode",
              repl._renderer().thinking_mode == "expanded")
    return failures


def test_live_stream_final_frame_is_complete() -> int:
    """A throttled live refresh can leave the last live frame stale (missing
    the final chunk); stream() used to paper over that by printing the whole
    response a second time via render_final() once the live view stopped."""
    failures = 0
    import ollamarun_lib.renderer as renderer_mod

    updates = []

    class _RecordingLive(orun.Live):
        def update(self, renderable, *, refresh=False):
            updates.append(renderable)
            super().update(renderable, refresh=refresh)

    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=True)
    renderer = orun.Renderer(console, mode="markdown", thinking_mode="hidden")
    render_final_calls = []
    renderer.render_final = lambda content: render_final_calls.append(content)

    original_live = renderer_mod.Live
    renderer_mod.Live = _RecordingLive
    try:
        result = renderer.stream(iter([
            {"message": {"content": "Hello "}},
            {"message": {"content": "world."}},
            {"done": True},
        ]))
    finally:
        renderer_mod.Live = original_live

    def render_text(renderable) -> str:
        out_buffer = io.StringIO()
        Console(file=out_buffer, width=100, force_terminal=False).print(renderable)
        return out_buffer.getvalue()

    last_frame = render_text(updates[-1]) if updates else ""
    if (result.content == "Hello world." and "Hello world." in last_frame
            and render_final_calls == []):
        print("PASS  live stream freezes on the complete response and doesn't re-print it")
    else:
        print(f"FAIL  live stream final frame: last_frame={last_frame!r} "
              f"render_final_calls={render_final_calls!r}")
        failures += 1
    return failures


def test_tool_loop_separation() -> int:
    """The answer that ends a tool loop is separated from the processing
    above it by a bright centered `response` rule -- and only the committed
    final frame of a post-tools round carries it: streaming frames, interim
    rounds and tool-less turns keep today's rendering."""
    failures = 0
    import ollamarun_lib.renderer as renderer_mod

    def render_text(renderable) -> str:
        out_buffer = io.StringIO()
        Console(file=out_buffer, width=100, force_terminal=False).print(renderable)
        return out_buffer.getvalue()

    def run_stream(events, *, after_tools):
        updates = []

        class _RecordingLive(orun.Live):
            def update(self, renderable, *, refresh=False):
                updates.append(renderable)
                super().update(renderable, refresh=refresh)

        console = Console(file=io.StringIO(), width=100, force_terminal=True)
        renderer = orun.Renderer(console, mode="markdown", thinking_mode="hidden",
                                 after_tools=after_tools)
        original_live = renderer_mod.Live
        renderer_mod.Live = _RecordingLive
        try:
            result = renderer.stream(iter(events))
        finally:
            renderer_mod.Live = original_live
        return result, updates

    final_events = [{"message": {"content": "The answer is 42."}},
                    {"message": {"content": ""}, "done": True}]
    interim_events = [{"message": {"role": "assistant", "content": "checking the file",
                                   "tool_calls": [{"function": {"name": "run_shell"}}]}},
                      {"message": {"role": "assistant", "content": ""}, "done": True}]

    result, updates = run_stream(final_events, after_tools=True)
    committed = render_text(updates[-1])
    streamed = "".join(render_text(frame) for frame in updates[:-1])
    failures = _check(failures, "answer ending a tool loop gets the response rule",
                      "response" in committed and "The answer is 42." in committed,
                      committed)
    failures = _check(failures, "the rule only appears in the committed frame",
                      "response" not in streamed, streamed)

    result, updates = run_stream(interim_events, after_tools=True)
    committed = render_text(updates[-1])
    failures = _check(failures, "interim tool round is not marked",
                      "response" not in committed, committed)

    result, updates = run_stream(final_events, after_tools=False)
    committed = render_text(updates[-1])
    failures = _check(failures, "tool-less turn keeps today's rendering",
                      "response" not in committed, committed)

    renderer = orun.Renderer(Console(file=io.StringIO(), width=100), mode="markdown",
                             after_tools=True)
    cancelled = orun.GenerationResult(content="half an answer", cancelled=True)
    errored = orun.GenerationResult(error="connection lost")
    failures = _check(failures, "cancelled and errored rounds are never marked",
                      not renderer.final_rule_wanted(cancelled)
                      and not renderer.final_rule_wanted(errored)
                      and renderer.final_rule_wanted(orun.GenerationResult(content="done")))

    # _generate() hands each round's renderer the loop state: False while the
    # first round is still the only one, True for every round that follows a
    # round which requested tools.
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save", "-y"])
        script = [
            [{"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": "echo from-tool"}}}]},
              "done": False},
             {"message": {"role": "assistant", "content": ""}, "done": True}],
            [{"message": {"role": "assistant", "content": "all done"}, "done": True}],
        ]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=script)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        seen = []
        original_renderer = repl._renderer

        def recording_renderer(*, after_tools=False):
            seen.append(after_tools)
            return original_renderer(after_tools=after_tools)

        repl._renderer = recording_renderer
        repl.session.add_message("user", "run something")
        with orun.console.capture():
            repl._generate()
        failures = _check(failures, "_generate marks only post-tool rounds",
                          seen == [False, True], str(seen))
    return failures


MATH_LAYOUT_CASES = [
    (r"\frac{a}{b}", ["a", "─", "b"]),
    (r"\frac{a}{b} + c", ["a", "─ + c", "b"]),
    (r"\sqrt{x+1}", ["─────", "√x + 1"]),
    (r"\sum_{i=1}^{n} i", ["  n", "  ∑   i", "i = 1"]),
    (r"\boxed{42}", ["╭────╮", "│ 42 │", "╰────╯"]),
    (r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}", ["⎛ a  b ⎞", "⎝ c  d ⎠"]),
    (r"\begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}",
     ["⎧ 1  x > 0", "⎩ 0  x ≤ 0"]),
    (r"\binom{n}{k}", ["⎛n⎞", "⎜ ⎟", "⎝k⎠"]),
    (r"\left( \frac{1}{2} \right)", ["⎛1⎞", "⎜─⎟", "⎝2⎠"]),
    (r"\int_0^1 x\,dx", ["∫₀¹ x dx"]),
    (r"\begin{aligned} a &= b \\ c &= d \end{aligned}", ["a = b", "c = d"]),
    (r"\vec{F} = m\vec{a}", ["F⃗ = ma⃗"]),
    (r"\mathbb{R} \subset \mathcal{L}", ["ℝ ⊂ ℒ"]),
]

MATH_INLINE_CASES = [
    (r"E = mc^2", "E = mc²"),
    (r"x_i^2", "xᵢ²"),
    (r"z_1^2", "z₁²"),
    (r"a + b = c", "a + b = c"),
    (r"x = -b \pm \sqrt{b^2-4ac}", "x = -b ± √(b² - 4ac)"),
    (r"\int_0^\infty e^{-x^2}\,dx", "∫₀^∞e^(-x²) dx"),
    (r"\frac{a+b}{c}", "(a + b)/c"),
    (r"\binom{n}{k}", "(n/k)"),
]


def test_math_layout() -> int:
    failures = 0
    for tex, expected in MATH_LAYOUT_CASES:
        got = orun.render_display_math(tex)
        if got == expected:
            print(f"PASS  2D math {tex}")
        else:
            print(f"FAIL  2D math {tex}:\n  got      {got!r}\n  expected {expected!r}")
            failures += 1
    for tex, expected in MATH_INLINE_CASES:
        got = orun.render_inline_math(tex)
        if got == expected:
            print(f"PASS  inline math {tex}")
        else:
            print(f"FAIL  inline math {tex}: {got!r} != {expected!r}")
            failures += 1

    if orun.render_display_math(r"\begin{tikzpicture} x \end{tikzpicture}") is None:
        print("PASS  unknown environment falls back")
    else:
        print("FAIL  unknown environment should fall back")
        failures += 1
    if orun.render_display_math("a + b + c", width=3) is None:
        print("PASS  over-wide display math falls back")
    else:
        print("FAIL  over-wide display math should fall back")
        failures += 1
    huge = "x" * 500
    if orun.render_display_math(huge) is None:
        print("PASS  token budget falls back")
    else:
        print("FAIL  token budget should fall back")
        failures += 1
    return failures


def test_render_content() -> int:
    failures = 0

    def render(renderable, width=50) -> str:
        buffer = io.StringIO()
        Console(file=buffer, width=width, force_terminal=False).print(renderable)
        return buffer.getvalue()

    out = render(orun.render_content("before\n\n$$\\frac{a}{b}$$\n\nafter", width=46))
    if "before" in out and "after" in out and "─" in out and "$$" not in out:
        print("PASS  display math becomes a centered block")
    else:
        print(f"FAIL  display math block: {out!r}")
        failures += 1

    fenced = render(orun.render_content("```\n$$x = 1$$\n```"))
    if "$$x = 1$$" in fenced:
        print("PASS  display math inside code fence untouched")
    else:
        print(f"FAIL  code fence: {fenced!r}")
        failures += 1

    disabled = render(orun.render_content("value $E=mc^2$ and $$x$$", math=False))
    if "$$x$$" in disabled and "$E=mc^2$" in disabled:
        print("PASS  math=False leaves markup alone")
    else:
        print(f"FAIL  math=False: {disabled!r}")
        failures += 1

    table = render(orun.render_content(
        "| Option | Effect |\n| :--- | :--- |\n| `ssh -o BatchMode=yes` | no prompts |"),
        width=80)
    if ":---" not in table and "|" not in table and "─" in table \
            and "ssh -o BatchMode=yes" in table:
        print("PASS  table with inline code renders as a table")
    else:
        print(f"FAIL  table with inline code: {table!r}")
        failures += 1

    math_cell = render(orun.render_content(
        "| Sym | Formula |\n|-----|---------|\n| area | $$A = \\pi r^2$$ |"), width=80)
    if "$$" not in math_cell and "|" not in math_cell and "─" in math_cell \
            and "A = πr²" in math_cell:
        print("PASS  display math in a table cell stays inline")
    else:
        print(f"FAIL  table display math: {math_cell!r}")
        failures += 1

    sentence = render(orun.render_content("Use `a` and `b` in one sentence.", width=80))
    if "Use a and b in one sentence." in sentence:
        print("PASS  inline code does not split a paragraph")
    else:
        print(f"FAIL  inline code paragraph: {sentence!r}")
        failures += 1

    bullets = render(orun.render_content("- one `a`\n- two `b`", width=80))
    if bullets.count("•") == 2 and "one a" in bullets and "two b" in bullets:
        print("PASS  inline code does not split a list")
    else:
        print(f"FAIL  inline code list: {bullets!r}")
        failures += 1

    midline = render(orun.render_content("so $$x^2$$ wins", width=80))
    if "so x² wins" in midline and "$$" not in midline:
        print("PASS  mid-line display math renders inline")
    else:
        print(f"FAIL  mid-line display math: {midline!r}")
        failures += 1

    thinking_out = render(orun.render_thinking("reasoning $\\frac{a}{b}$ done"))
    if "reasoning" in thinking_out and "a/b" in thinking_out and "done" in thinking_out:
        print("PASS  render_thinking renders inline math")
    else:
        print(f"FAIL  render_thinking: {thinking_out!r}")
        failures += 1
    return failures


def test_tps_estimator() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    estimator = orun.TpsEstimator(window=5.0)
    check("tps empty", estimator.rate(now=0.0), None)
    for tick in range(10):
        estimator.add(1, now=float(tick))
    check("tps windowed rate", round(estimator.rate(now=9.0), 3), 1.0)
    check("tps windowed helper", round(estimator.windowed(now=9.0), 3), 1.0)
    check("tps average helper", round(estimator.average(now=9.0), 3), round(10 / 9, 3))
    check("tps token count", estimator.tokens, 10)

    early = orun.TpsEstimator(window=5.0)
    early.add(1, now=100.0)
    early.add(1, now=100.2)
    check("tps cumulative fallback", round(early.rate(now=100.2), 3), 10.0)
    check("tps window too fresh is None", early.windowed(now=100.2), None)
    check("tps average of a fresh window", round(early.average(now=100.2), 3), 10.0)

    stalling = orun.TpsEstimator(window=5.0)
    for tick in range(5):
        stalling.add(1, now=float(tick))
    check("tps stall falls back to cumulative",
          round(stalling.rate(now=30.0), 3), round(5 / 30, 3))
    check("tps stall window is None", stalling.windowed(now=30.0), None)
    return failures


class _FakeClient:
    def __init__(self, resident=(), models=(), ps_entries=(), show_data=None,
                 capabilities=("completion",), chat_script=None,
                 pull_events=None, pull_error=None, tag_data=None,
                 chat_once_reply="ok"):
        self.base = "http://fake:11434"
        self.resident = list(resident)
        self.models = list(models)
        self.ps_entries = list(ps_entries)
        self.show_data = show_data
        self.capability_list = list(capabilities)
        self.chat_script = [list(events) for events in (chat_script or [])]
        self.chat_calls = 0
        self.chat_once_reply = chat_once_reply
        self.chat_once_payloads = []
        self.unloaded = []
        self.loaded = []
        self.load_options = []
        self.chat_payloads = []
        self.pull_events = list(pull_events or [])
        self.pull_error = pull_error
        self.pulled = []
        self.tag_data = list(tag_data) if tag_data is not None else None
        self.show_calls = []

    def is_resident(self, model):
        return model in self.resident

    def unload(self, model):
        self.unloaded.append(model)

    def load(self, model, *, keepalive=None, options=None):
        self.loaded.append(model)
        self.load_options.append(dict(options or {}))

    def capabilities(self, model):
        return list(self.capability_list)

    def chat(self, payload, timeout=None, cancel=None):
        self.chat_payloads.append(payload)
        if self.chat_script:
            events = self.chat_script[self.chat_calls % len(self.chat_script)]
            self.chat_calls += 1
            return iter(events)
        return iter([{"message": {"content": "ok"}, "done": True}])

    def chat_once(self, payload, timeout=None):
        self.chat_once_payloads.append(payload)
        return {"message": {"content": self.chat_once_reply}, "done": True}

    def tags(self):
        if self.tag_data is not None:
            return list(self.tag_data)
        return [{"name": name} for name in self.models]

    def show(self, model):
        self.show_calls.append(model)
        return self.show_data if self.show_data is not None else {}

    def ps(self):
        return list(self.ps_entries)

    def model_exists(self, model):
        return model in self.models

    def pull_model(self, model, progress=None):
        self.pulled.append(model)
        if self.pull_error:
            raise orun.OllamaError(self.pull_error)
        for event in self.pull_events:
            if progress:
                progress(event)


def test_unload_housekeeping() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient(resident=["m1"], models=["m1", "m2"])
        session = orun.Session.create("m1", client.base, title="cur")
        repl = orun.Repl(args, config, client, store, session)
        repl._build_prompt_session = lambda: None

        repl.session.model = "m2"
        if repl.unload_model() is False and client.unloaded == []:
            print("PASS  unload skips non-resident models")
        else:
            print(f"FAIL  non-resident unload: {client.unloaded}")
            failures += 1

        repl.session.model = "m1"
        if repl.unload_model() is True and client.unloaded == ["m1"]:
            print("PASS  unload sends keep_alive=0")
        else:
            print(f"FAIL  resident unload: {client.unloaded}")
            failures += 1

        client.unloaded.clear()
        args_keep = orun.build_parser().parse_args(["m1", "--keep-loaded", "--render", "plain"])
        repl_keep = orun.Repl(args_keep, config, client, store,
                              orun.Session.create("m1", client.base))
        repl_keep._build_prompt_session = lambda: None
        repl_keep._read_line = lambda message: "/bye"
        repl_keep.run()
        if client.unloaded == []:
            print("PASS  --keep-loaded leaves the model loaded")
        else:
            print(f"FAIL  --keep-loaded unloaded: {client.unloaded}")
            failures += 1

        client.unloaded.clear()
        repl_exit = orun.Repl(args, config, client, store,
                              orun.Session.create("m1", client.base))
        repl_exit._build_prompt_session = lambda: None
        repl_exit._read_line = lambda message: "/bye"
        repl_exit.run()
        if client.unloaded == ["m1"]:
            print("PASS  /bye unloads the session model")
        else:
            print(f"FAIL  exit unload: {client.unloaded}")
            failures += 1

        client.unloaded.clear()
        current = orun.Session.create("m1", client.base, title="current")
        current.add_message("user", "hello")
        target = orun.Session.create("m2", client.base, title="target")
        target.add_message("user", "hi")
        store.save(target)
        repl.session = current
        orun.Prompt.ask = lambda *a, **k: "c"
        orun._switch_session(repl, target)
        if target.model == "m1" and client.unloaded == []:
            print("PASS  session switch can keep the current model")
        else:
            print(f"FAIL  switch keep-current: model={target.model} unloaded={client.unloaded}")
            failures += 1

        current2 = orun.Session.create("m1", client.base, title="current2")
        current2.add_message("user", "hello")
        target2 = orun.Session.create("m2", client.base, title="target2")
        target2.add_message("user", "hi")
        repl.session = current2
        orun.Prompt.ask = lambda *a, **k: "l"
        orun._switch_session(repl, target2)
        if target2.model == "m2" and client.unloaded == ["m1"]:
            print("PASS  session switch to last model unloads the other one")
        else:
            print(f"FAIL  switch last-model: model={target2.model} unloaded={client.unloaded}")
            failures += 1

        client.unloaded.clear()
        repl.session = orun.Session.create("m1", client.base)
        orun.cmd_model(repl, "m2")
        if repl.session.model == "m2" and client.unloaded == ["m1"]:
            print("PASS  /model unloads before switching")
        else:
            print(f"FAIL  /model unload: model={repl.session.model} unloaded={client.unloaded}")
            failures += 1
    return failures


def test_preload_model() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient(models=["m1"])
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        repl._read_line = lambda message: "/bye"

        repl.run()
        if repl._preload_thread is not None:
            repl._preload_thread.join(timeout=2)
        if client.loaded == ["m1"]:
            print("PASS  starting the REPL preloads the session's model")
        else:
            print(f"FAIL  preload on start: loaded={client.loaded}")
            failures += 1
    return failures


def test_preload_shows_ps_before_first_response() -> int:
    """`ollama ps` should show up once the background-preloaded model is
    resident, not only after the first response finishes generating.

    The preload runs on a background thread, and Console.capture() only
    buffers output on the thread that entered it (rich's buffer state is
    thread-local) -- so this swaps console.file directly instead, which is
    a plain shared attribute every thread's writes go through.
    """
    failures = 0
    original_force_terminal = orun.console._force_terminal
    original_file = orun.console.file
    orun.console._force_terminal = True
    buffer = io.StringIO()
    orun.console.file = buffer
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
            args = orun.build_parser().parse_args(["m1", "--render", "plain"])
            config = dict(orun.DEFAULT_CONFIG)
            entries = [{"name": "m1", "size": 100, "size_vram": 50, "details": {},
                        "expires_at": None}]
            client = _FakeClient(resident=["m1"], models=["m1"], ps_entries=entries)
            repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))
            repl._build_prompt_session = lambda: None
            repl._read_line = lambda message: "/bye"

            repl.run()
            if repl._preload_thread is not None:
                repl._preload_thread.join(timeout=2)
        out = _plain(buffer.getvalue())
        if "ollama ps" in out and client.chat_payloads == []:
            print("PASS  ollama ps shows up as soon as the preloaded model is "
                  "resident, without waiting for a response")
        else:
            print(f"FAIL  preload ps ordering: out={out!r} "
                  f"chat_payloads={client.chat_payloads!r}")
            failures += 1
    finally:
        orun.console._force_terminal = original_force_terminal
        orun.console.file = original_file
    return failures


def test_preload_loading_rprompt() -> int:
    """_preload_model() must not block the prompt from appearing. The
    "waiting for model" indicator is the prompt session's own animated
    rprompt (see _loading_rprompt()), driven by its refresh_interval,
    instead of a separate rich.Live view fighting prompt_toolkit for the
    terminal -- that fight is what caused the garbled startup output and
    vanishing spinner from a previous attempt at this. rprompt (not
    bottom_toolbar, which anchors to the terminal's bottom row and leaves a
    large blank gap -- confirmed with a pty-driven repro) keeps it on the
    prompt's own line."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)

        class _SlowLoadClient(_FakeClient):
            def load(self, model, *, keepalive=None, options=None):
                time.sleep(0.2)
                super().load(model, keepalive=keepalive, options=options)

        client = _SlowLoadClient(models=["m1"])
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        rprompt_before = repl._loading_rprompt()

        t0 = time.monotonic()
        repl._preload_model()
        elapsed = time.monotonic() - t0
        still_loading = repl._preload_thread is not None and repl._preload_thread.is_alive()
        rprompt_while_loading = repl._loading_rprompt()

        if repl._preload_thread is not None:
            repl._preload_thread.join(timeout=2)
        rprompt_after = repl._loading_rprompt()

        if (rprompt_before is None and elapsed < 0.1 and still_loading
                and rprompt_while_loading is not None and "waiting for m1" in rprompt_while_loading
                and rprompt_after is None):
            print("PASS  preload doesn't block, and the rprompt spinner "
                  "reflects loading state correctly")
        else:
            print(f"FAIL  preload rprompt: rprompt_before={rprompt_before!r} "
                  f"elapsed={elapsed:.3f} still_loading={still_loading} "
                  f"rprompt_while_loading={rprompt_while_loading!r} "
                  f"rprompt_after={rprompt_after!r}")
            failures += 1

    if orun.HAS_PROMPT_TOOLKIT:
        source = inspect.getsource(orun.Repl._read_line)
        if "patch_stdout" in source:
            print("PASS  _read_line wraps the prompt in patch_stdout so "
                  "background prints don't corrupt it")
        else:
            print("FAIL  _read_line no longer uses patch_stdout")
            failures += 1
    return failures


def test_generate_waits_for_pending_preload_before_showing_ps() -> int:
    """If a preload thread is still running when a message is sent (e.g. the
    non-terminal path, where _preload_model() doesn't block), _generate()
    should wait for that same load instead of firing a second, concurrent
    one via chat() -- and `ollama ps` should show up right then, not only
    after the response finishes generating."""
    failures = 0
    original_force_terminal = orun.console._force_terminal
    original_file = orun.console.file
    orun.console._force_terminal = True
    buffer = io.StringIO()
    orun.console.file = buffer

    class _SlowLoadClient(_FakeClient):
        def load(self, model, *, keepalive=None, options=None):
            time.sleep(0.15)
            self.resident.append(model)
            super().load(model, keepalive=keepalive, options=options)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
            args = orun.build_parser().parse_args(["m1", "--render", "plain"])
            config = dict(orun.DEFAULT_CONFIG)
            entries = [{"name": "m1", "size": 100, "size_vram": 50, "details": {},
                        "expires_at": None}]
            client = _SlowLoadClient(models=["m1"], ps_entries=entries)
            repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))
            repl._build_prompt_session = lambda: None

            thread = threading.Thread(target=lambda: client.load("m1"), daemon=True)
            repl._preload_thread = thread
            thread.start()
            still_loading = thread.is_alive()
            repl.send("hi")
        out = _plain(buffer.getvalue())
        ps_pos = out.find("ollama ps")
        marker_pos = out.find("m1 ›")
        if still_loading and ps_pos != -1 and marker_pos != -1 and ps_pos < marker_pos:
            print("PASS  ollama ps shows up before the response when the message "
                  "was sent while the model was still preloading")
        else:
            print(f"FAIL  generate/preload ordering: still_loading={still_loading} "
                  f"ps_pos={ps_pos} marker_pos={marker_pos} out={out!r}")
            failures += 1
    finally:
        orun.console._force_terminal = original_force_terminal
        orun.console.file = original_file
    return failures


def test_empty_enter_sends_pending_images() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "shot.png"
        image.write_bytes(_tiny_png())

        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient(models=["m1"])
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        orun.cmd_image(repl, str(image))

        lines = iter(["", "/bye"])
        repl._read_line = lambda message: next(lines)
        with orun.console.capture():
            repl.run()
        if (len(client.chat_payloads) == 1 and repl.pending_images == []
                and repl.session.messages
                and repl.session.messages[0].get("images")):
            print("PASS  Enter on an empty line sends pending images immediately")
        else:
            print(f"FAIL  empty-enter send: chats={len(client.chat_payloads)} "
                  f"pending={repl.pending_images} messages={repl.session.messages}")
            failures += 1

        client2 = _FakeClient(models=["m1"])
        repl2 = orun.Repl(args, config, client2, store, orun.Session.create("m1", client2.base))
        repl2._build_prompt_session = lambda: None
        lines2 = iter(["", "/bye"])
        repl2._read_line = lambda message: next(lines2)
        with orun.console.capture():
            repl2.run()
        if not client2.chat_payloads:
            print("PASS  Enter on an empty line with no pending images stays a no-op")
        else:
            print(f"FAIL  empty-enter without images: chats={len(client2.chat_payloads)}")
            failures += 1
    return failures


def test_context_length() -> int:
    failures = 0
    show = {"model_info": {"llama.context_length": 8192},
            "parameters": "num_ctx 4096"}
    if orun.model_context_length(show) == 8192:
        print("PASS  context length from model_info")
    else:
        print("FAIL  context length from model_info")
        failures += 1
    if orun.model_context_length(show, {"num_ctx": 2048}) == 2048:
        print("PASS  session num_ctx wins")
    else:
        print("FAIL  session num_ctx")
        failures += 1
    if orun.model_context_length({"parameters": "num_ctx 1024"}) == 1024:
        print("PASS  context length from parameters string")
    else:
        print("FAIL  parameters string")
        failures += 1
    return failures


def test_naive_title() -> int:
    failures = 0
    title = orun.naive_title("# Hello **world**\nsecond line")
    if title == "Hello **world**":
        print("PASS  naive_title strips heading marks")
    else:
        print(f"FAIL  naive_title: {title!r}")
        failures += 1
    if orun.naive_title("") == "untitled":
        print("PASS  naive_title fallback")
    else:
        print("FAIL  naive_title fallback")
        failures += 1
    return failures


def test_chat_bubbles() -> int:
    failures = 0
    console = Console(width=80, force_terminal=False)

    box_ = orun.user_message_box("hello", console)
    if isinstance(box_, orun.Align) and box_.align == "right":
        print("PASS  user_message_box right-aligns")
    else:
        print(f"FAIL  user_message_box alignment: {box_}")
        failures += 1

    panel = box_.renderable
    if isinstance(panel, orun.Panel) and panel.width == 12 and panel.border_style == "you" \
            and panel.title == "you":
        print("PASS  user_message_box floors narrow content to a minimum width")
    else:
        print(f"FAIL  user_message_box panel: width={panel.width} title={panel.title}")
        failures += 1

    wide_panel = orun.user_message_box("x" * 200, console).renderable
    if wide_panel.width == 72:
        print("PASS  user_message_box caps width at 72")
    else:
        print(f"FAIL  user_message_box width cap: {wide_panel.width}")
        failures += 1

    rule = orun.model_message_rule("llama3:latest")
    if (isinstance(rule, orun.Rule) and rule.style == "ai"
            and "llama3:latest" in str(rule.title)):
        print("PASS  model_message_rule is a Rule titled with the model name")
    else:
        print(f"FAIL  model_message_rule: {rule.title!r} style={rule.style}")
        failures += 1

    if "assistant" in str(orun.model_message_rule("").title):
        print("PASS  model_message_rule falls back to 'assistant' with no model")
    else:
        print("FAIL  model_message_rule fallback title")
        failures += 1
    return failures


def test_render_transcript_bubbles() -> int:
    failures = 0
    session = orun.Session.create("m1", "http://h", title="chat")
    session.add_message("user", "hello there")
    session.add_message("assistant", "hi, how can I help?")
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, theme=orun.THEME)
    orun.render_transcript(session, console, show_stats=False)
    out = buffer.getvalue()
    if "hello there" in out and "hi, how can I help" in out and "m1" in out:
        print("PASS  render_transcript renders user/model chat bubbles")
    else:
        print(f"FAIL  render_transcript bubbles: {out!r}")
        failures += 1
    return failures


def test_tool_colors() -> int:
    failures = 0
    orun.set_tool_colors({})  # known state: built-in colours, no memoised picks

    invalid = []
    for color in list(orun.TOOL_COLORS.values()) + list(orun.TOOL_COLOR_PALETTE):
        try:
            Color.parse(color)
        except ColorParseError:
            invalid.append(color)
    if not invalid:
        print("PASS  tool colours are valid rich colour names")
    else:
        print(f"FAIL  invalid tool colours: {invalid}")
        failures += 1

    if len(set(orun.TOOL_COLORS.values())) == len(orun.TOOL_COLORS):
        print("PASS  every built-in tool has a unique colour")
    else:
        print(f"FAIL  duplicate built-in tool colours: {orun.TOOL_COLORS}")
        failures += 1

    if not (set(orun.TOOL_COLORS.values()) & set(orun.TOOL_COLOR_PALETTE)):
        print("PASS  fallback palette is disjoint from the fixed colours")
    else:
        print("FAIL  palette overlaps fixed tool colours")
        failures += 1

    if orun.tool_color("run_shell") == orun.TOOL_COLORS["run_shell"]:
        print("PASS  a mapped tool keeps its fixed colour")
    else:
        print(f"FAIL  mapped tool colour: {orun.tool_color('run_shell')!r}")
        failures += 1

    first = orun.tool_color("mystery_one")
    again = orun.tool_color("mystery_one")
    second = orun.tool_color("mystery_two")
    if (first in orun.TOOL_COLOR_PALETTE and again == first and second != first
            and second in orun.TOOL_COLOR_PALETTE):
        print("PASS  unknown tools get stable, distinct random palette colours")
    else:
        print(f"FAIL  unknown tool colours: {first!r}, {again!r}, {second!r}")
        failures += 1
    if orun.tool_color("") == orun.tool_color("tool"):
        print("PASS  a nameless call uses the 'tool' fallback colour")
    else:
        print(f"FAIL  nameless fallback: {orun.tool_color('')!r} "
              f"vs {orun.tool_color('tool')!r}")
        failures += 1

    orun.set_tool_colors({"run_shell": "red", "mystery_one": "blue"})
    if orun.tool_color("run_shell") == "red" and orun.tool_color("mystery_one") == "blue":
        print("PASS  config tool_colors overrides win")
    else:
        print(f"FAIL  config overrides: {orun.tool_color('run_shell')!r} "
              f"{orun.tool_color('mystery_one')!r}")
        failures += 1

    with orun.console.capture() as cap:
        orun.set_tool_colors({"run_shell": "notacolor", "run_python": "bold green"})
    warned = _plain(cap.get())
    if (orun.tool_color("run_shell") == "cyan"
            and orun.tool_color("run_python") == "bold green"
            and "ignoring invalid tool colour" in warned):
        print("PASS  invalid tool_colors values are ignored with a warning")
    else:
        print(f"FAIL  invalid override: {orun.tool_color('run_shell')!r} "
              f"{orun.tool_color('run_python')!r} {warned!r}")
        failures += 1
    orun.set_tool_colors({})  # back to the built-ins for the rest of the suite

    panel = orun.tool_call_panel("run_shell", {"command": "ls -la"})
    if isinstance(panel, orun.Panel) and panel.border_style == "cyan" \
            and panel.title.plain == "tool · run_shell" \
            and str(panel.title.style) == "cyan":
        print("PASS  tool_call_panel uses the tool colour for border and title")
    else:
        print(f"FAIL  tool_call_panel: {panel.border_style!r} {panel.title!r}")
        failures += 1

    result = orun.tool_result_panel("run_shell", detail="ls", text="out",
                                    ok=False, duration=0.5)
    spans = {result.title.plain[span.start:span.end]: str(span.style)
             for span in result.title.spans}
    if (result.border_style == "cyan"
            and result.title.plain == "tool · run_shell · failed · 0.50s"
            and spans.get("run_shell") == "cyan" and spans.get("failed") == "red"):
        print("PASS  tool_result_panel wears the tool colour; failed is red in the title")
    else:
        print(f"FAIL  tool_result_panel: {result.border_style!r} "
              f"{result.title.plain!r} {spans!r}")
        failures += 1

    success = orun.tool_result_panel("run_shell", text="out", ok=True)
    success_spans = {success.title.plain[span.start:span.end]: str(span.style)
                     for span in success.title.spans}
    if success.border_style == "cyan" and success_spans.get("ok") == "green":
        print("PASS  tool_result_panel marks ok green in the title")
    else:
        print(f"FAIL  tool_result_panel ok: {success.border_style!r} "
              f"{success_spans!r}")
        failures += 1

    # the approval prompt and the /tools table name tools in their colours too
    def ansi(color: str) -> str:
        return "\x1b[" + ";".join(Color.parse(color).get_ansi_codes(True)) + "m"

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        session = orun.Session.create("m1", client.base)
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        recorded = io.StringIO()
        terminal_console = Console(file=recorded, force_terminal=True,
                                   color_system="truecolor", width=100,
                                   theme=orun.THEME)
        from ollamarun_lib.repl.commands import tools as tools_command
        real_console = orun.repl.console
        real_tools_console = tools_command.console
        # the approval prompt itself lives in ui.py now (the front-end seam)
        real_prompt = orun.ui._prompt_approval
        real_interactive = repl.ui.interactive
        orun.repl.console = terminal_console
        tools_command.console = terminal_console
        orun.ui._prompt_approval = lambda suggestions: ("deny", None)
        repl.ui.interactive = True  # the gate is the Ui's, not the console's
        try:
            orun.cmd_tools(repl, "")
            repl._approve_tool_call(repl.tool_registry["write_file"],
                                    {"path": "notes.txt", "content": "x"}, "fp")
        finally:
            orun.repl.console = real_console
            tools_command.console = real_tools_console
            orun.ui._prompt_approval = real_prompt
            repl.ui.interactive = real_interactive
        styled = recorded.getvalue()

    cyan, orange = ansi("cyan"), ansi("orange1")  # run_shell / write_file
    if (f"{cyan}run_shell" in styled and f"{orange}write_file" in styled
            and "needs confirmation" in _plain(styled)):
        print("PASS  the confirmation line and /tools table name tools in their colours")
    else:
        print(f"FAIL  approval/table colours: run_shell={f'{cyan}run_shell' in styled} "
              f"write_file={f'{orange}write_file' in styled}")
        failures += 1

    session = orun.Session.create("m1", "http://h", title="tools")
    session.add_message("assistant", "", tool_calls=[{
        "function": {"name": "run_python", "arguments": {"code": "print(1)"}}}])
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, theme=orun.THEME)
    orun.render_transcript(session, console, show_stats=False)
    out = _plain(buffer.getvalue())
    if "run_python" in out and "print(1)" in out:
        print("PASS  render_transcript renders the tool call panel")
    else:
        print(f"FAIL  transcript tool panel: {out!r}")
        failures += 1
    return failures


def test_ps_table_and_cmd_ps() -> int:
    failures = 0
    entries = [{"name": "llama3:latest", "size": 4_000_000_000, "size_vram": 3_500_000_000,
                "details": {"quantization_level": "Q4_0"}, "expires_at": "2026-01-01T00:00:00Z"}]
    table = orun.ps_table(entries)
    with orun.console.capture() as cap:
        orun.console.print(table)
    out = cap.get()
    if "llama3:latest" in out and "Q4_0" in out and "GiB" in out:
        print("PASS  ps_table renders model/quant/size columns")
    else:
        print(f"FAIL  ps_table: {out!r}")
        failures += 1

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)

        empty_client = _FakeClient()
        repl = orun.Repl(args, config, empty_client, store, orun.Session.create("m1", empty_client.base))
        with orun.console.capture() as cap:
            orun.cmd_ps(repl, "")
        if "no models currently loaded" in cap.get():
            print("PASS  cmd_ps reports an empty ps list")
        else:
            print(f"FAIL  cmd_ps empty: {cap.get()!r}")
            failures += 1

        full_client = _FakeClient(ps_entries=entries)
        repl2 = orun.Repl(args, config, full_client, store, orun.Session.create("m1", full_client.base))
        with orun.console.capture() as cap:
            orun.cmd_ps(repl2, "")
        if "llama3:latest" in cap.get():
            print("PASS  cmd_ps prints the ollama ps table")
        else:
            print(f"FAIL  cmd_ps table: {cap.get()!r}")
            failures += 1
    return failures


def test_maybe_show_ps() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        entries = [{"name": "m1", "size": 100, "size_vram": 50, "details": {}, "expires_at": None}]
        client = _FakeClient(resident=["m1"], models=["m1"], ps_entries=entries)
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        with orun.console.capture() as cap:
            repl._maybe_show_ps()
        first = cap.get()
        with orun.console.capture() as cap:
            repl._maybe_show_ps()
        second = cap.get()
        if "m1" in first and second == "":
            print("PASS  _maybe_show_ps prints ollama ps only once per model")
        else:
            print(f"FAIL  _maybe_show_ps: first={first!r} second={second!r}")
            failures += 1

        client2 = _FakeClient(resident=[], models=["m1"])
        repl2 = orun.Repl(args, config, client2, store, orun.Session.create("m1", client2.base))
        with orun.console.capture() as cap:
            repl2._maybe_show_ps()
        if cap.get() == "":
            print("PASS  _maybe_show_ps is silent when the model isn't resident")
        else:
            print(f"FAIL  _maybe_show_ps non-resident: {cap.get()!r}")
            failures += 1
    return failures


def test_context_usage() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        show = {"model_info": {"llama.context_length": 8192}}
        client = _FakeClient(models=["m1"], show_data=show)
        session = orun.Session.create("m1", client.base)
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")
        session.stats["last_prompt_tokens"] = 100
        session.stats["last_completion_tokens"] = 40
        repl = orun.Repl(args, config, client, store, session)

        if repl.context_used() == 140:
            print("PASS  context_used sums last prompt + completion tokens")
        else:
            print(f"FAIL  context_used: {repl.context_used()}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_context(repl, "")
        out = _plain(cap.get())
        if "last completion tokens" in out and "140" in out:
            print("PASS  /context reports prompt + completion output tokens")
        else:
            print(f"FAIL  /context rows: {out!r}")
            failures += 1

        # The warning must key off the combined usage: 100/150 alone is only
        # 67%, but 100 + 40 over a 150-token window is 93% and must warn.
        warn_client = _FakeClient(models=["m1"],
                                  show_data={"model_info": {"llama.context_length": 150}})
        warn_repl = orun.Repl(args, config, warn_client, store, session)
        with orun.console.capture() as cap:
            warn_repl._context_warning()
        if "140/150" in _plain(cap.get()):
            print("PASS  context warning counts the last output tokens")
        else:
            print(f"FAIL  context warning: {_plain(cap.get())!r}")
            failures += 1

        # The denominator is the window the runner actually loaded with, not
        # the model's architectural maximum: /api/ps reports context_length
        # (a qwen3.5 allows 262144 but may be loaded at 65536).
        loaded_client = _FakeClient(
            models=["m1"],
            show_data={"model_info": {"llama.context_length": 262144}},
            ps_entries=[{"name": "m1", "model": "m1", "size": 1,
                         "size_vram": 1, "context_length": 65536}])
        loaded_repl = orun.Repl(args, config, loaded_client, store, session)
        if loaded_repl.context_length() == 65536:
            print("PASS  loaded /api/ps context beats the model's arch max")
        else:
            print(f"FAIL  loaded context: {loaded_repl.context_length()}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_context(loaded_repl, "")
        if "65536" in _plain(cap.get()):
            print("PASS  /context reports the loaded window")
        else:
            print(f"FAIL  /context loaded window: {_plain(cap.get())!r}")
            failures += 1

        # A session num_ctx is what the next request asks for (and reloads
        # at), so it wins over whatever is loaded right now.
        session.parameters["num_ctx"] = 8192
        if loaded_repl.context_length() == 8192:
            print("PASS  session num_ctx beats the loaded context")
        else:
            print(f"FAIL  session num_ctx over loaded: {loaded_repl.context_length()}")
            failures += 1
        session.parameters.pop("num_ctx")

        # Before the model is loaded the show answer is provisional: once
        # /api/ps reports the runner, the next call picks the real window up.
        grow_client = _FakeClient(models=["m1"],
                                  show_data={"model_info": {"llama.context_length": 262144}})
        grow_repl = orun.Repl(args, config, grow_client, store, session)
        first = grow_repl.context_length()
        grow_client.ps_entries = [{"name": "m1:latest", "context_length": 65536}]
        second = grow_repl.context_length()
        if first == 262144 and second == 65536:
            print("PASS  a pre-load context is re-checked against /api/ps")
        else:
            print(f"FAIL  provisional context upgrade: {first} -> {second}")
            failures += 1

        # The warning measures against the loaded window too: 140 tokens over
        # the 262144 arch max is nothing, over the loaded 150-token runner it
        # is 93% and must warn.
        ps_warn_client = _FakeClient(
            models=["m1"],
            show_data={"model_info": {"llama.context_length": 262144}},
            ps_entries=[{"name": "m1", "context_length": 150}])
        ps_warn_repl = orun.Repl(args, config, ps_warn_client, store, session)
        with orun.console.capture() as cap:
            ps_warn_repl._context_warning()
        if "140/150" in _plain(cap.get()):
            print("PASS  context warning measures against the loaded window")
        else:
            print(f"FAIL  loaded-window warning: {_plain(cap.get())!r}")
            failures += 1
    return failures


def test_environment_tools() -> int:
    failures = 0
    registry = orun.default_registry(tts=False)
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        show = {"model_info": {"llama.context_length": 262144},
                "details": {"parameter_size": "8.0B",
                            "quantization_level": "Q4_K_M", "family": "llama"},
                "capabilities": ["completion", "tools", "thinking"]}
        client = _FakeClient(models=["m1"], show_data=show,
                             ps_entries=[{"name": "m1", "context_length": 65536}])
        session = orun.Session.create("m1", client.base, title="env test")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        session.add_message("tool", "result", tool_name="run_shell")
        session.stats["last_prompt_tokens"] = 11900
        session.stats["last_completion_tokens"] = 445
        repl = orun.Repl(args, config, client, store, session)
        repl._tool_round = 2

        report = repl.context_report()
        failures = _check(failures, "context_report resolves the loaded window",
                          report["window"] == 65536
                          and report["source"] == "loaded runner"
                          and report["model_max"] == 262144
                          and report["used"] == 12345
                          and report["messages"] == {"total": 3, "user": 1,
                                                     "assistant": 1, "tool": 1},
                          str(report))
        repl.context_report()
        failures = _check(failures, "the /api/show data is fetched once and cached",
                          len(client.show_calls) == 1, str(client.show_calls))

        outcome = orun.execute_tool(registry["context_status"], {}, repl.tool_ctx)
        failures = _check(failures, "context_status formats the context budget",
                          outcome.ok
                          and "context window: 65536 tokens (loaded runner; "
                              "model default 262144)" in outcome.text
                          and "context used: 12345 tokens = last prompt 11900 "
                              "+ last reply 445" in outcome.text
                          and "free: 53191 tokens (18.8% used)" in outcome.text
                          and "messages: 3 (assistant 1 · tool 1 · user 1)"
                          in outcome.text,
                          outcome.text)

        outcome = orun.execute_tool(registry["session_info"], {}, repl.tool_ctx)
        failures = _check(failures, "session_info reports identity and state",
                          outcome.ok
                          and 'session: ' in outcome.text
                          and '"env test"' in outcome.text
                          and "model: m1 · 8.0B · Q4_K_M · capabilities: "
                              "completion, tools, thinking" in outcome.text
                          and f"server: this machine ({client.base})" in outcome.text
                          and "tools run on: " in outcome.text
                          and "settings: keepalive default" in outcome.text
                          and "modes: thinking collapsed · render plain · "
                              "system prompt built-in default" in outcome.text
                          and "tools: on · policy allowlist · round 2 of 8 · "
                              "disabled: none" in outcome.text
                          and "context window" not in outcome.text,
                          outcome.text)

        # A session num_ctx is what the next request asks for, so it wins.
        ctx_session = orun.Session.create("m1", client.base, title="ctx")
        ctx_session.parameters["num_ctx"] = 32768
        ctx_repl = orun.Repl(args, config, client, store, ctx_session)
        report = ctx_repl.context_report()
        failures = _check(failures, "context_report labels a session num_ctx",
                          report["window"] == 32768 and report["source"] == "num_ctx",
                          str(report))

        # Unloaded and no num_ctx: the show default is provisional.
        bare_client = _FakeClient(models=["m1"], show_data=show)
        bare_session = orun.Session.create("m1", bare_client.base, title="bare")
        bare_repl = orun.Repl(args, config, bare_client, store, bare_session)
        report = bare_repl.context_report()
        failures = _check(failures, "context_report marks an unloaded default provisional",
                          report["window"] == 262144
                          and report["source"] == "model default (runner not loaded yet)",
                          str(report))

        # Nearly full: the result tells the model to wrap up.
        tight_client = _FakeClient(models=["m1"], show_data=show,
                                   ps_entries=[{"name": "m1", "context_length": 150}])
        tight_session = orun.Session.create("m1", tight_client.base, title="tight")
        tight_session.stats["last_prompt_tokens"] = 100
        tight_session.stats["last_completion_tokens"] = 40
        tight_repl = orun.Repl(args, config, tight_client, store, tight_session)
        outcome = orun.execute_tool(registry["context_status"], {}, tight_repl.tool_ctx)
        failures = _check(failures, "context_status warns when the window is nearly full",
                          outcome.ok and "nearly full" in outcome.text, outcome.text)

        # A fresh session has no measured request yet.
        fresh = orun.Session.create("m1", client.base, title="fresh")
        fresh_repl = orun.Repl(args, config, client, store, fresh)
        outcome = orun.execute_tool(registry["context_status"], {}, fresh_repl.tool_ctx)
        failures = _check(failures, "context_status handles a fresh session",
                          outcome.ok and "nothing yet" in outcome.text, outcome.text)

    bare_ctx = orun.ToolContext(cwd=".", timeout=10, max_output=4000)
    for name in ("context_status", "session_info"):
        outcome = orun.execute_tool(registry[name], {}, bare_ctx)
        failures = _check(failures, f"{name} errors without a REPL report",
                          not outcome.ok and "unavailable" in outcome.text, outcome.text)
    failures = _check(failures, "the awareness tools run unattended (read-only)",
                      orun.approval_decision(registry["context_status"], {},
                                             policy="allowlist").allowed
                      and orun.approval_decision(registry["session_info"], {},
                                                 policy="allowlist").allowed)
    failures = _check(failures, "context/whoami aliases resolve onto the new tools",
                      orun.resolve_tool_alias("context") == "context_status"
                      and orun.resolve_tool_alias("token_status") == "context_status"
                      and orun.resolve_tool_alias("whoami") == "session_info"
                      and orun.resolve_tool_alias("environment") == "session_info")
    return failures


def test_resolve_num_ctx() -> int:
    failures = 0
    client = _FakeClient(show_data={"model_info": {"llama.context_length": 8192}})
    applied, max_ctx = orun.resolve_num_ctx(client, "m1", 32000)
    if applied == 8192 and max_ctx == 8192:
        print("PASS  resolve_num_ctx caps an over-large request to the model max")
    else:
        print(f"FAIL  resolve_num_ctx cap: applied={applied} max_ctx={max_ctx}")
        failures += 1

    applied2, max_ctx2 = orun.resolve_num_ctx(client, "m1", 4096)
    if applied2 == 4096 and max_ctx2 == 8192:
        print("PASS  resolve_num_ctx leaves an in-range request unchanged")
    else:
        print(f"FAIL  resolve_num_ctx unchanged: applied={applied2} max_ctx={max_ctx2}")
        failures += 1

    class _ErrClient(_FakeClient):
        def show(self, model):
            raise orun.OllamaError("unreachable")

    applied3, max_ctx3 = orun.resolve_num_ctx(_ErrClient(), "m1", 999)
    if applied3 == 999 and max_ctx3 is None:
        print("PASS  resolve_num_ctx passes the request through when /api/show fails")
    else:
        print(f"FAIL  resolve_num_ctx unreachable server: {applied3}, {max_ctx3}")
        failures += 1
    return failures


def test_cmd_num_ctx() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient(show_data={"model_info": {"llama.context_length": 8192}})
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        with orun.console.capture() as cap:
            orun.cmd_num_ctx(repl, "32000")
        out = _plain(cap.get())
        if repl.session.parameters.get("num_ctx") == 8192 and "capped to 8192" in out:
            print("PASS  /num-ctx caps an over-large request with a warning")
        else:
            print(f"FAIL  /num-ctx capping: {out!r} params={repl.session.parameters}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_num_ctx(repl, "2048")
        if repl.session.parameters.get("num_ctx") == 2048:
            print("PASS  /num-ctx sets an in-range value without a warning")
        else:
            print(f"FAIL  /num-ctx in-range set: {repl.session.parameters}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_num_ctx(repl, "")
        out = _plain(cap.get())
        if "num_ctx = 2048" in out and "model max 8192" in out:
            print("PASS  /num-ctx with no argument reports the current value and max")
        else:
            print(f"FAIL  /num-ctx status: {out!r}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_num_ctx(repl, "0")
        if "positive integer" in cap.get():
            print("PASS  /num-ctx rejects a non-positive value")
        else:
            print(f"FAIL  /num-ctx non-positive: {cap.get()!r}")
            failures += 1

        with orun.console.capture() as cap:
            orun.cmd_num_ctx(repl, "abc")
        if "usage:" in cap.get():
            print("PASS  /num-ctx rejects a non-numeric value")
        else:
            print(f"FAIL  /num-ctx non-numeric: {cap.get()!r}")
            failures += 1
    return failures


def _fit_check(name: str, ok: bool, detail: str = "") -> int:
    if ok:
        print(f"PASS  {name}")
        return 0
    print(f"FAIL  {name}: {detail}")
    return 1


CTX_SHOW = {"model_info": {
    "qwen35.block_count": 24,
    "qwen35.attention.head_count": 8,
    "qwen35.attention.head_count_kv": [0, 0, 0, 2] * 6,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.context_length": 262144,
}}

# Dense attention (like qwen2.5), so the per-token KV is one scalar per
# block: 2 * 4 kv heads * (128 + 128) * 2 bytes * 28 blocks = 57,344 B.
CPU_SHOW = {"model_info": {
    "qwen2.block_count": 28,
    "qwen2.attention.head_count": 28,
    "qwen2.attention.head_count_kv": 4,
    "qwen2.attention.key_length": 128,
    "qwen2.attention.value_length": 128,
    "qwen2.context_length": 32768,
}}

# Real gemma3-270m metadata: 18 blocks, 3 full-attention layers and 15 sliding
# ones (every 6th full), the window is 512 tokens.
SWA_SHOW = {"model_info": {
    "gemma3.block_count": 18,
    "gemma3.attention.head_count": 4,
    "gemma3.attention.head_count_kv": 1,
    "gemma3.attention.key_length": 256,
    "gemma3.attention.value_length": 256,
    "gemma3.attention.sliding_window": 512,
    "gemma3.embedding_length": 640,
    "gemma3.context_length": 32768,
    "general.parameter_count": 268098176,
    "general.file_type": 7,
}}

# A qwen35-style hybrid: KV every 4th block, SSM state on the rest.
HYBRID_SHOW = {"model_info": dict(CTX_SHOW["model_info"], **{
    "qwen35.ssm.inner_size": 2048,
    "qwen35.ssm.state_size": 128,
    "qwen35.ssm.conv_kernel": 4,
})}

# DeepSeek-style MLA: one compressed latent plus a rope tail per layer.
MLA_SHOW = {"model_info": {
    "deepseek2.block_count": 61,
    "deepseek2.attention.head_count": 128,
    "deepseek2.attention.head_count_kv": 128,
    "deepseek2.attention.kv_lora_rank": 512,
    "deepseek2.attention.qk_rope_head_dim": 64,
    "deepseek2.embedding_length": 7168,
    "deepseek2.context_length": 163840,
}}

# A quantized model whose tag carries params and quant but no usable size.
PROFILE_SHOW = {"model_info": {
    "qwen2.block_count": 24,
    "qwen2.attention.head_count": 14,
    "qwen2.attention.head_count_kv": 2,
    "qwen2.embedding_length": 896,
    "qwen2.context_length": 32768,
    "qwen2.vocab_size": 151936,
    "general.parameter_count": 494032768,
    "general.file_type": 15,
}}
PROFILE_ENTRY = {"name": "m1", "size": 0,
                 "details": {"parameter_size": "494.03M",
                             "quantization_level": "Q4_K_M"}}


class _CtxFitClient(_FakeClient):
    """Fake Ollama whose loaded footprint scales with num_ctx (for --probe).

    The simulated GPU capacity is what the real fit would fill up: a load
    whose total (weights + KV) exceeds it reports a partial size_vram, and
    `fail_above` makes loads beyond a context raise like a CUDA OOM.
    """

    def __init__(self, *, weights=2_000_000_000, kv=12288,
                 capacity=int((3500 - 1024) * 2 ** 20), model_max=262144,
                 base="http://127.0.0.1:11434", fail_above=None):
        super().__init__(models=["m1"])
        self.base = base
        self.weights = weights
        self.kv = kv
        self.capacity = capacity
        self.model_max = model_max
        self.fail_above = fail_above
        self.current = None
        self.load_ctxs = []
        self.show_data = {"model_info": dict(CTX_SHOW["model_info"],
                                             **{"qwen35.context_length": model_max})}

    def tags(self):
        return [{"name": "m1", "size": self.weights}]

    def ps(self):
        if self.current is None:
            return []
        ctx, total, vram = self.current
        return [{"name": "m1", "model": "m1", "size": total,
                 "size_vram": vram, "context_length": ctx}]

    def unload(self, model):
        self.unloaded.append(model)
        self.current = None

    def load(self, model, *, keepalive=None, options=None):
        self.loaded.append(model)
        self.load_options.append(dict(options or {}))
        ctx = int((options or {}).get("num_ctx") or 0)
        if self.fail_above is not None and ctx > self.fail_above:
            raise orun.OllamaError("CUDA error: out of memory")
        total = self.weights + self.kv * ctx
        self.current = (ctx, total, min(total, self.capacity))
        self.load_ctxs.append(ctx)

    def truth(self) -> int:
        """Largest 256-aligned context the simulated GPU really fits."""
        return ((self.capacity - self.weights) // self.kv // 256) * 256


def test_ctxfit() -> int:
    failures = 0
    old_calibration_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
    calibration_tmp = tempfile.TemporaryDirectory()
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(calibration_tmp.name) / "ctxfit.json"

    devices = orun.parse_gpu_memory("4096, 3100\n24576, 12000\nnot-a-device\n")
    failures += _fit_check("parse_gpu_memory reads nvidia-smi CSV", devices == [(4096, 3100), (24576, 12000)],
                       repr(devices))
    env = orun.parse_service_env(
        'Environment=OLLAMA_NUM_PARALLEL=2 LLAMA_ARG_FIT_TARGET=512 '
        'OLLAMA_KV_CACHE_TYPE="q8_0" OLLAMA_HOST=100.1.2.3:11434')
    failures += _fit_check("parse_service_env reads the ollama unit environment",
                       env.get("LLAMA_ARG_FIT_TARGET") == "512"
                       and env.get("OLLAMA_NUM_PARALLEL") == "2"
                       and env.get("OLLAMA_KV_CACHE_TYPE") == "q8_0",
                       repr(env))

    if shutil.which("ssh"):
        class _Proc:
            returncode = 0
            stdout = ("4096, 3100\n" + orun.GPU_ENV_MARKER
                      + "Environment=LLAMA_ARG_FIT_TARGET=2048")
            stderr = ""

        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            return _Proc()

        got = orun.ssh_gpu_probe("somehost", user="u", runner=runner)
        failures += _fit_check("ssh_gpu_probe parses the combined GPU/env output",
                           got == ([(4096, 3100)], {"LLAMA_ARG_FIT_TARGET": "2048"})
                           and calls and calls[0][:2] == ["ssh", "-o"],
                           repr(got))
        class _Down(_Proc):
            returncode = 255
        failures += _fit_check("ssh_gpu_probe returns None when ssh fails",
                           orun.ssh_gpu_probe("somehost", runner=lambda *a, **k: _Down()) is None)
    else:
        print("SKIP  ssh_gpu_probe (ssh not installed)")

    # ── KV cache math ───────────────────────────────────────────────────────
    failures += _fit_check("kv_bytes_per_token handles per-layer kv-head lists",
                       orun.kv_bytes_per_token(CTX_SHOW) == 12288,
                       str(orun.kv_bytes_per_token(CTX_SHOW)))
    failures += _fit_check("kv_bytes_per_token scales with sequences",
                       orun.kv_bytes_per_token(CTX_SHOW, parallel=4) == 49152)
    failures += _fit_check("kv_bytes_per_token honours the cache element size",
                       orun.kv_bytes_per_token(CTX_SHOW,
                                               element_bytes=orun.kv_element_bytes("q8_0")) == 6528)
    llama_show = {"model_info": {"llama.block_count": 30, "llama.attention.head_count": 9,
                                 "llama.attention.head_count_kv": 3,
                                 "llama.embedding_length": 576, "llama.context_length": 8192}}
    failures += _fit_check("kv_bytes_per_token falls back to embedding/head_count",
                       orun.kv_bytes_per_token(llama_show) == 23040,
                       str(orun.kv_bytes_per_token(llama_show)))
    failures += _fit_check("kv_bytes_per_token without kv heads is None",
                       orun.kv_bytes_per_token({"model_info": {"x.block_count": 4}}) is None)
    failures += _fit_check("trained_context_length/model_total_layers read the metadata",
                       orun.trained_context_length(CTX_SHOW) == 262144
                       and orun.model_total_layers(CTX_SHOW) == 25)
    failures += _fit_check("_kv_fraction_on_gpu follows the attention blocks in the tail",
                        orun.ctxfit._kv_fraction_on_gpu(CTX_SHOW, 13) == 0.5
                        and orun.ctxfit._kv_fraction_on_gpu(CTX_SHOW, 25) == 1.0,
                        f"{orun.ctxfit._kv_fraction_on_gpu(CTX_SHOW, 13)}")

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.CtxCalibrationStore(Path(tmp) / "calibration.json")
        store.add("scope", 4096, 1_000_000)
        store.add("scope", 8192, 2_000_000)
        store.add("scope", 4096, 1_100_000)  # replace, don't duplicate a context
        samples = store.points("scope")
        line = orun.ctxfit._measured_memory_line(samples, 100, 200_000)
        failures += _fit_check("context calibration persists bounded measured points",
                               samples == [(4096, 1_100_000), (8192, 2_000_000)]
                               and line is not None and line[0] == 220
                               and line[1] == 200_000,
                               f"samples={samples} line={line}")
        for ctx in range(1, orun.ctxfit.CALIBRATION_MAX_SAMPLES + 5):
            store.add("bounded", ctx, ctx * 1000)
        bounded = store.points("bounded")
        store.add("stale", 4096, 1_000_000)
        cache_data = json.loads((Path(tmp) / "calibration.json").read_text())
        cache_data["entries"]["stale"]["samples"][0]["ts"] = (
            time.time() - orun.ctxfit.CALIBRATION_MAX_AGE - 1)
        (Path(tmp) / "calibration.json").write_text(json.dumps(cache_data))
        failures += _fit_check("calibration cache caps samples and expires stale points",
                               len(bounded) == orun.ctxfit.CALIBRATION_MAX_SAMPLES
                               and store.points("stale") == [],
                               f"bounded={len(bounded)} stale={store.points('stale')}")

    # ── estimate ────────────────────────────────────────────────────────────
    machine = orun.MachineGpu(devices=[(16384, 12000)], env={}, source="local")
    weights = 10_000_000_000
    fit = orun.estimate_max_gpu_ctx(CTX_SHOW, weights, machine)
    usable = (12000 - 1024) * 2 ** 20
    budget = usable - weights
    expected = ((budget // 12288) // 1024) * 1024
    failures += _fit_check("estimate_max_gpu_ctx computes the full-offload context",
                       fit.feasible and fit.ctx == expected and fit.kv_gpu_per_token == 12288,
                       f"ctx={fit.ctx} expected={expected}")
    failures += _fit_check("estimate reserves the fit margin per device",
                        fit.margin_bytes == 1024 * 2 ** 20
                        and fit.usable_bytes == (12000 - 1024) * 2 ** 20)

    empirical_gpu_points = [
        (ctx, weights + 256 * 2 ** 20 + 24576 * ctx)
        for ctx in (4096, 16384, 32768)
    ]
    calibrated_gpu = orun.estimate_max_gpu_ctx(CTX_SHOW, weights, machine,
                                               empirical_points=empirical_gpu_points)
    failures += _fit_check("GPU estimate uses the conservative measured byte slope",
                           calibrated_gpu.kv_gpu_per_token == 24576
                           and calibrated_gpu.ctx < fit.ctx
                           and any("calibrated from" in note for note in calibrated_gpu.notes),
                           f"slope={calibrated_gpu.kv_gpu_per_token} "
                           f"ctx={calibrated_gpu.ctx} baseline={fit.ctx}")

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "ctxfit.json"
        cached_client = _CtxFitClient()
        cached_machine = orun.MachineGpu(devices=[(4096, 3500)], env={}, source="local")
        cached_baseline = orun.estimate_max_gpu_ctx(
            cached_client.show_data, cached_client.weights, cached_machine)
        cached_req = orun.FitRequest(client=cached_client, model="m1",
                                     machine=cached_machine,
                                     calibration_path=str(cache_path))
        cache_key = orun.ctxfit._calibration_scope(
            cached_req, mode="gpu", model_digest="", weight_bytes=cached_client.weights,
            show_data=cached_client.show_data, machine=cached_machine,
            pinned_layers=None)
        cache = orun.CtxCalibrationStore(cache_path)
        for ctx in (4096, 16384):
            cache.add(cache_key, ctx, cached_client.weights + 256 * 2 ** 20 + 24576 * ctx)
        with orun.console.capture() as cap:
            chosen = orun.max_gpu_context(cached_req)
        failures += _fit_check("estimate-only GPU fit reuses stored measurements without loading",
                               chosen is not None and chosen < cached_baseline.ctx
                               and "calibrated from" in _plain(cap.get())
                               and not cached_client.load_ctxs,
                               f"chosen={chosen} baseline={cached_baseline.ctx} "
                               f"out={_plain(cap.get())[-180:]}")

    fit_env = orun.estimate_max_gpu_ctx(
        CTX_SHOW, weights, orun.MachineGpu(
            devices=[(16384, 12000)],
            env={"LLAMA_ARG_FIT_TARGET": "512", "OLLAMA_NUM_PARALLEL": "2",
                 "OLLAMA_KV_CACHE_TYPE": "q8_0"}))
    env_usable = (12000 - 512) * 2 ** 20
    env_budget = env_usable - weights
    env_kv = orun.kv_bytes_per_token(CTX_SHOW, parallel=2,
                                     element_bytes=orun.kv_element_bytes("q8_0"))
    env_expected = ((env_budget // env_kv) // 1024) * 1024
    failures += _fit_check("estimate honours the server env (fit target, parallel, kv type)",
                       fit_env.ctx == env_expected and fit_env.parallel == 2
                       and fit_env.margin_bytes == 512 * 2 ** 20,
                       f"ctx={fit_env.ctx} expected={env_expected}")

    pinned = orun.estimate_max_gpu_ctx(CTX_SHOW, 20_000_000_000, machine, pinned_layers=13)
    pinned_budget = usable - int(20_000_000_000 * 13 / 25)
    pinned_expected = ((pinned_budget // 6144) // 1024) * 1024
    failures += _fit_check("estimate scales weights and KV for pinned gpu-layers",
                       pinned.ctx == pinned_expected and pinned.kv_gpu_per_token == 6144
                       and pinned.weights_bytes == int(20_000_000_000 * 13 / 25),
                       f"ctx={pinned.ctx} expected={pinned_expected}")

    capped = orun.estimate_max_gpu_ctx(CTX_SHOW, 6_000_000_000,
                                       orun.MachineGpu(devices=[(16384, 12000)]))
    failures += _fit_check("estimate caps at the model's max context",
                           capped.capped_by_model and capped.ctx == 262144,
                           f"ctx={capped.ctx} capped={capped.capped_by_model}")

    infeasible = orun.estimate_max_gpu_ctx(CTX_SHOW, 3_000_000_000,
                                           orun.MachineGpu(devices=[(4096, 1000)]))
    failures += _fit_check("estimate reports an impossible placement",
                           not infeasible.feasible and "does not fit" in infeasible.error
                           and "0 B usable" in infeasible.error
                           and "?" not in infeasible.error,
                           infeasible.error)
    no_gpu = orun.estimate_max_gpu_ctx(CTX_SHOW, weights,
                                       orun.MachineGpu(error="no nvidia-smi GPU data"))
    failures += _fit_check("estimate needs GPU data",
                       not no_gpu.feasible and "nvidia-smi" in no_gpu.error)
    no_kv = orun.estimate_max_gpu_ctx({"model_info": {"x.block_count": 4}}, weights, machine)
    failures += _fit_check("estimate needs KV metadata",
                       not no_kv.feasible and "KV cache" in no_kv.error)

    credited = orun.estimate_max_gpu_ctx(CTX_SHOW, weights, machine,
                                         residents=[("m1", 2_000_000_000)])
    failures += _fit_check("estimate credits resident models' VRAM into the pool",
                           credited.credited_bytes == 2_000_000_000
                           and credited.usable_bytes == (12000 - 1024) * 2 ** 20
                           + 2_000_000_000
                           and any("crediting m1" in note for note in credited.notes),
                           f"usable={credited.usable_bytes} notes={credited.notes}")

    rescued = orun.estimate_max_gpu_ctx(CTX_SHOW, 2_000_000_000,
                                        orun.MachineGpu(devices=[(4096, 1000)]),
                                        residents=[("m1", 3_000_000_000)])
    failures += _fit_check("a resident model can make an infeasible placement fit",
                           rescued.feasible and rescued.ctx > 0,
                           f"ctx={rescued.ctx} error={rescued.error}")

    margin_only = orun.estimate_max_gpu_ctx(CTX_SHOW, 20_000_000,
                                            orun.MachineGpu(devices=[(4096, 1000)]))
    unc_budget = 1000 * 2 ** 20 - 20_000_000
    unc_expected = ((unc_budget // 12288) // 1024) * 1024
    failures += _fit_check("estimate reports the no-margin context as a hint",
                           not margin_only.feasible
                           and margin_only.ctx_no_margin == unc_expected,
                           f"ctx_no_margin={margin_only.ctx_no_margin} "
                           f"expected={unc_expected}")

    # ── probe loop ──────────────────────────────────────────────────────────
    def run_probe(client, free_mb=3500, **request_kwargs):
        previous_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
        probe_tmp = tempfile.TemporaryDirectory()
        orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(probe_tmp.name) / "ctxfit.json"
        req_machine = orun.MachineGpu(devices=[(4096, free_mb)], env={}, source="local")
        request = orun.FitRequest(client=client, model="m1",
                                  config={"max_gpu_probes": 10}, machine=req_machine,
                                  **request_kwargs)
        try:
            with orun.console.capture() as cap:
                chosen = orun.max_gpu_context(request, probe=True)
            return chosen, _plain(cap.get())
        finally:
            orun.ctxfit.CTXFIT_CALIBRATION_PATH = previous_path
            probe_tmp.cleanup()

    client = _CtxFitClient()
    chosen, out = run_probe(client)
    failures += _fit_check("--probe finds the true boundary exactly",
                       chosen == client.truth() == 48384 and len(client.load_ctxs) == 2
                       and client.load_ctxs[0] == 48128 and "100% GPU" in out,
                       f"chosen={chosen} truth={client.truth()} probes={client.load_ctxs}")

    resident_probe = _CtxFitClient(capacity=6_000_000_000)
    resident_probe.current = (65536, 2_800_000_000, 2_800_000_000)
    chosen, out = run_probe(resident_probe)
    failures += _fit_check("--probe starts from the estimate, not 4,096",
                           resident_probe.load_ctxs
                           and resident_probe.load_ctxs[0] == 262144,
                           f"probes={resident_probe.load_ctxs}")

    hidden = _CtxFitClient(capacity=int((3500 - 1024) * 2 ** 20) - 300 * 2 ** 20)
    chosen, _ = run_probe(hidden)
    failures += _fit_check("--probe corrects an optimistic estimate downward",
                           chosen == hidden.truth() and len(hidden.load_ctxs) <= 10,
                           f"chosen={chosen} truth={hidden.truth()} probes={hidden.load_ctxs}")

    tight = _CtxFitClient(capacity=int((3500 - 1024) * 2 ** 20) - 300 * 2 ** 20, kv=12288,
                          model_max=4096)
    chosen, out = run_probe(tight)
    failures += _fit_check("--probe stops at the model's max context",
                       chosen == 4096 and "capped by the model's max" in out)

    oom = _CtxFitClient(fail_above=40000)
    chosen, out = run_probe(oom)
    failures += _fit_check("--probe backs off from load failures",
                       chosen == 39936 and "load failed" in out,
                       f"chosen={chosen} probes={oom.load_ctxs}")

    class _BlindClient(_CtxFitClient):
        def ps(self):
            raise orun.OllamaError("no ps")

    chosen, out = run_probe(_BlindClient())
    failures += _fit_check("--probe reports when the model can't be measured",
                       chosen is None and "/api/ps" in out, out[-200:])

    stuck = _CtxFitClient(capacity=1000 * 2 ** 20)
    chosen, out = run_probe(stuck)
    failures += _fit_check("--probe reports when nothing fits",
                       chosen is None and "fully on the GPU" in out, out[-200:])

    # ── orchestration ───────────────────────────────────────────────────────
    request = orun.FitRequest(client=_CtxFitClient(weights=10_000_000_000), model="m1",
                              config={"max_gpu_probes": 4},
                              machine=orun.MachineGpu(devices=[(16384, 12000)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(request)
    out = _plain(cap.get())
    failures += _fit_check("max_gpu_context estimates without loading",
                           chosen == expected and "estimate only" in out
                           and request.client.load_ctxs == [],
                           f"chosen={chosen} expected={expected}")

    resident = _CtxFitClient()
    resident.current = (65536, 2_800_000_000, 2_800_000_000)
    request = orun.FitRequest(client=resident, model="m1", config={},
                              machine=orun.MachineGpu(devices=[(4096, 1000)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(request)
    out = _plain(cap.get())
    credited_budget = 2_800_000_000 - 2_000_000_000
    credited_expected = ((credited_budget // 12288) // 1024) * 1024
    failures += _fit_check("max_gpu_context credits a resident model's VRAM",
                           chosen == credited_expected and "resident" in out
                           and resident.load_ctxs == [],
                           f"chosen={chosen} expected={credited_expected}")

    class _TwoResidents(_CtxFitClient):
        def ps(self):
            return [{"name": "m1", "model": "m1", "size": 2_000_000_000,
                     "size_vram": 2_000_000_000, "context_length": 65536},
                    {"name": "other:latest", "model": "other:latest",
                     "size": 1_000_000_000, "size_vram": 1_000_000_000,
                     "context_length": 8192}]

    request = orun.FitRequest(client=_TwoResidents(), model="m1", config={},
                              machine=orun.MachineGpu(devices=[(4096, 1000)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(request)
    failures += _fit_check("max_gpu_context credits every resident by default",
                           chosen is not None and chosen > 0, f"chosen={chosen}")

    request = orun.FitRequest(client=_TwoResidents(), model="m1",
                              config={"max_gpu_credit_residents": False},
                              machine=orun.MachineGpu(devices=[(4096, 1000)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(request)
    out = _plain(cap.get())
    failures += _fit_check("max_gpu_credit_residents=false credits only the target",
                           chosen is None and "not credited" in out,
                           f"chosen={chosen} out={out[-200:]}")

    request = orun.FitRequest(client=_CtxFitClient(), model="m1",
                              config={"max_gpu_probes": 4},
                              machine=orun.MachineGpu(devices=[(4096, 1000)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(request, probe=True)
    failures += _fit_check("max_gpu_context probes even when the estimate says no",
                           chosen is not None and chosen >= 4096
                           and request.client.load_ctxs[:1] == [4096],
                           f"chosen={chosen} probes={request.client.load_ctxs}")

    class _NoModel(_CtxFitClient):
        def tags(self):
            return []

    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(orun.FitRequest(
            client=_NoModel(), model="m1",
            machine=orun.MachineGpu(devices=[(4096, 3500)])))
    failures += _fit_check("max_gpu_context needs the model in /api/tags",
                       chosen is None and "/api/tags" in cap.get(), cap.get())

    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(orun.FitRequest(
            client=_CtxFitClient(), model="m1",
            machine=orun.MachineGpu(error="no nvidia-smi GPU data")))
    failures += _fit_check("max_gpu_context reports unreadable VRAM",
                       chosen is None and "nvidia-smi" in cap.get(), cap.get())

    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(orun.FitRequest(
            client=_CtxFitClient(), model="m1", pinned_spec="cpu",
            machine=orun.MachineGpu(devices=[(4096, 3500)])))
    failures += _fit_check("max_gpu_context refuses a cpu-only placement",
                       chosen is None and "cpu" in cap.get(), cap.get())

    remote_like = _CtxFitClient(base="http://10.9.9.9:11434")
    with orun.console.capture() as cap:
        chosen = orun.max_gpu_context(orun.FitRequest(client=remote_like, model="m1"))
    failures += _fit_check("max_gpu_context needs --remote for non-local hosts",
                       chosen is None and "--remote" in cap.get(), cap.get())

    failures += _fit_check("parse_num_ctx_arg covers the CLI forms",
                       orun.parse_num_ctx_arg("max-gpu") == ("max-gpu", None)
                       and orun.parse_num_ctx_arg("--MAX-GPU") == ("max-gpu", None)
                       and orun.parse_num_ctx_arg("8192") == ("value", 8192)
                       and orun.parse_num_ctx_arg("nope") == ("error", None))
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = old_calibration_path
    calibration_tmp.cleanup()
    return failures


def test_ctxfit_outcome() -> int:
    """fit_context() returns the structured outcome the API serves."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _CtxFitClient()
        machine = orun.MachineGpu(devices=[(4096, 3500)], env={}, source="local")
        req = orun.FitRequest(client=client, model="m1", machine=machine,
                              config=dict(orun.DEFAULT_CONFIG),
                              calibration_path=str(Path(tmp) / "c.json"))
        outcome = orun.fit_context(req, mode="gpu")
        failures += _fit_check("fit_context returns the structured outcome",
                              outcome.ok and outcome.num_ctx is not None
                              and outcome.mode == "gpu"
                              and outcome.source == "estimate"
                              and outcome.bound == "max_gpu"
                              and outcome.num_ctx == outcome.fit.ctx,
                              repr(outcome.to_dict())[:200])
        payload = outcome.to_dict()
        failures += _fit_check("the outcome serializes as data",
                              payload["num_ctx"] == outcome.num_ctx
                              and payload["report"]["weights_total"] > 0
                              and payload["report"]["kv_per_token"] > 0
                              and isinstance(payload["report"]["devices"], list),
                              repr(payload)[:300])
        cpu = orun.fit_context(req, mode="cpu", percent=70)
        failures += _fit_check("the cpu outcome carries its percentage",
                              cpu.ok and cpu.percent == 70
                              and cpu.bound == "max_cpu",
                              repr(cpu.to_dict())[:200])
        bad = orun.fit_context(orun.FitRequest(client=client, model="m1",
                                               machine=machine),
                               mode="cpu", percent=0)
        failures += _fit_check("an invalid percentage is data, not a crash",
                              not bad.ok and bad.error is not None
                              and "percentage" in bad.error, repr(bad))
        missing = orun.fit_context(orun.FitRequest(client=client, model="",
                                                   machine=machine), mode="gpu")
        failures += _fit_check("a missing model is data, not a crash",
                              not missing.ok
                              and missing.error == "no model selected",
                              repr(missing))

        class _NoTags(_CtxFitClient):
            def tags(self):
                return []

            def show(self, model):
                return {"model_info": {"x.block_count": 4}}

        no_size = orun.fit_context(
            orun.FitRequest(client=_NoTags(), model="m1", machine=machine),
            mode="gpu")
        failures += _fit_check("an unsizeable model reports the tag error",
                              not no_size.ok and "no entry" in (no_size.error or ""),
                              repr(no_size))
    return failures


def test_ctxfit_cancel_progress() -> int:
    """A cancelled fit stops at the next checkpoint; progress is reported."""
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        client = _CtxFitClient(fail_above=65536)
        machine = orun.MachineGpu(devices=[(4096, 3500)], env={}, source="local")
        token = orun.CancelToken()
        token.cancel()
        req = orun.FitRequest(client=client, model="m1", machine=machine,
                              config=dict(orun.DEFAULT_CONFIG), cancel=token,
                              calibration_path=str(Path(tmp) / "c.json"))
        try:
            orun.fit_context(req, mode="gpu", probe=True)
            raised = None
        except orun.FitCancelled as exc:
            raised = exc
        failures += _fit_check("a pre-cancelled fit raises before any load",
                              raised is not None and client.load_ctxs == [],
                              repr(raised))

        token2 = orun.CancelToken()
        progress: list = []
        req2 = orun.FitRequest(client=client, model="m1", machine=machine,
                               config=dict(orun.DEFAULT_CONFIG), cancel=token2,
                               on_progress=progress.append,
                               calibration_path=str(Path(tmp) / "c2.json"))
        original_load = client.load

        def load_then_cancel(model, *, keepalive=None, options=None):
            original_load(model, keepalive=keepalive, options=options)
            if len(client.load_ctxs) == 1:
                token2.cancel()

        client.load = load_then_cancel
        try:
            orun.fit_context(req2, mode="gpu", probe=True)
            raised2 = None
        except orun.FitCancelled as exc:
            raised2 = exc
        failures += _fit_check("a mid-probe cancel stops before the next load",
                              raised2 is not None and len(client.load_ctxs) == 1,
                              f"{raised2!r} loads={client.load_ctxs}")
        stages = [item.get("stage") for item in progress]
        failures += _fit_check("progress fires for estimate and probe stages",
                              "estimate" in stages and "probe" in stages,
                              str(stages))
    return failures


def test_fit_service() -> int:
    """Fit jobs: lifecycle, progress, cancel, caps, TTL, apply and shutdown."""
    failures = 0
    gate = threading.Event()
    gate.set()

    def slow_fit(req, *, mode, percent=None, probe=False):
        req.on_progress({"stage": "probe", "mode": mode, "ctx": 2048,
                         "attempt": 1, "max": 2})
        gate.wait(timeout=5)
        if req.cancel is not None and req.cancel.cancelled:
            raise orun.FitCancelled("fit cancelled")
        outcome = orun.FitOutcome(mode=mode, model=req.model, probe=probe,
                                  bound="max_gpu" if mode == "gpu" else "max_cpu")
        outcome.num_ctx = 4096
        return outcome

    original = orun.fits.fit_context
    orun.fits.fit_context = slow_fit
    try:
        service = orun.FitService(max_active=1, ttl=3600)
        failures += _fit_check("cancelling an unknown fit is a no-op",
                              service.cancel("fit_nope") is None)
        gate.clear()
        job = service.submit(model="m1", mode="gpu", probe=True,
                             req=orun.FitRequest(client=object(), model="m1"),
                             session_id="s1")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not job.progress:
            time.sleep(0.01)
        failures += _fit_check("a submitted fit runs and reports progress",
                              job.status == "running"
                              and job.progress.get("ctx") == 2048,
                              f"{job.status} {job.progress}")
        try:
            service.submit(model="m1", mode="gpu",
                           req=orun.FitRequest(client=object(), model="m1"))
            cap_error = None
        except orun.TooManyFits as exc:
            cap_error = exc
        failures += _fit_check("the global cap refuses a second job",
                              cap_error is not None and cap_error.limit == 1,
                              repr(cap_error))
        try:
            service.submit(model="m1", mode="gpu", session_id="s1",
                           req=orun.FitRequest(client=object(), model="m1"))
            busy = None
        except orun.FitBusy as exc:
            busy = exc
        failures += _fit_check("a second fit for the same session is refused",
                              busy is not None and busy.active_fit_id == job.id,
                              repr(busy))
        failures += _fit_check("active_for names the holding job",
                              service.active_for("s1") == job.id
                              and service.active_count() == 1,
                              f"{service.active_for('s1')} "
                              f"{service.active_count()}")
        gate.set()
        service.wait(job.id, timeout=5)
        failures += _fit_check("a finished fit carries its outcome",
                              job.status == "completed" and job.outcome is not None
                              and job.outcome.num_ctx == 4096,
                              f"{job.status} {job.outcome}")
        body = orun.fit_body(job)
        failures += _fit_check("the fit body carries status and result",
                              body["status"] == "completed"
                              and body["result"]["num_ctx"] == 4096
                              and body["object"] == "fit"
                              and body["session_id"] == "s1",
                              str(body)[:200])

        gate.clear()
        cancel_job = service.submit(
            model="m1", mode="gpu", session_id="s2",
            req=orun.FitRequest(client=object(), model="m1"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and cancel_job.status != "running":
            time.sleep(0.01)
        service.cancel(cancel_job.id)
        gate.set()
        service.wait(cancel_job.id, timeout=5)
        failures += _fit_check("a cancelled job reports cancelled",
                              cancel_job.status == "cancelled"
                              and cancel_job.error == "cancelled",
                              f"{cancel_job.status} {cancel_job.error}")

        applied: list = []
        apply_job = service.submit(
            model="m1", mode="gpu", apply=True, session_id="s3",
            req=orun.FitRequest(client=object(), model="m1"),
            on_success=lambda finished: applied.append(finished.id))
        service.wait(apply_job.id, timeout=5)
        failures += _fit_check("apply runs after a successful fit",
                              apply_job.applied and applied == [apply_job.id],
                              f"{apply_job.applied} {applied}")

        def broken(finished):
            raise RuntimeError("nope")

        broken_job = service.submit(
            model="m1", mode="gpu", apply=True, session_id="s4",
            req=orun.FitRequest(client=object(), model="m1"),
            on_success=broken)
        service.wait(broken_job.id, timeout=5)
        failures += _fit_check("a failed apply is reported, not fatal",
                              broken_job.status == "completed"
                              and broken_job.applied is False
                              and broken_job.apply_error == "nope",
                              f"{broken_job.status} {broken_job.apply_error}")

        clock = [1000.0]
        ttl_service = orun.FitService(max_active=1, ttl=50,
                                      clock=lambda: clock[0])
        fast_job = ttl_service.submit(
            model="m1", mode="gpu",
            req=orun.FitRequest(client=object(), model="m1"))
        ttl_service.wait(fast_job.id, timeout=5)
        clock[0] += 100
        failures += _fit_check("a terminal job is swept after its TTL",
                              ttl_service.get(fast_job.id) is None,
                              "still there")
        ttl_service.shutdown()

        gate.clear()
        shut_job = service.submit(
            model="m1", mode="gpu",
            req=orun.FitRequest(client=object(), model="m1"))
        service.shutdown(cancel=True, timeout=0.2)
        gate.set()
        service.wait(shut_job.id, timeout=5)
        failures += _fit_check("shutdown cancels an active fit",
                              shut_job.status == "cancelled", shut_job.status)
        try:
            service.submit(model="m1", mode="gpu",
                           req=orun.FitRequest(client=object(), model="m1"))
            closed = None
        except orun.FitsClosed as exc:
            closed = exc
        failures += _fit_check("a shut-down service refuses new fits",
                              closed is not None, repr(closed))
    finally:
        gate.set()
        orun.fits.fit_context = original
    return failures


def test_model_profile() -> int:
    """First-principles modelling: GGUF geometry, quant/params and tags facts."""
    failures = 0
    old_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
    calibration_tmp = tempfile.TemporaryDirectory()
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(calibration_tmp.name) / "ctxfit.json"
    try:
        profile = orun.model_profile(SWA_SHOW)
        failures += _fit_check("model_profile reads the GGUF facts and file-type quant",
                               profile.params == 268098176
                               and profile.quantization == "Q8_0"
                               and profile.blocks == 18
                               and profile.embedding_length == 640
                               and profile.context_length == 32768,
                               f"params={profile.params} quant={profile.quantization} "
                               f"blocks={profile.blocks}")
        failures += _fit_check("sliding-window KV saturates past window + batch",
                               orun.kv_bytes_per_token(SWA_SHOW) == 18432
                               and profile.kv_slope() == 3072
                               and profile.kv_bytes(32768) == 124256256
                               and profile.kv_bytes(2048) == 29884416,
                               f"full={orun.kv_bytes_per_token(SWA_SHOW)} "
                               f"slope={profile.kv_slope()} "
                               f"@2048={profile.kv_bytes(2048)} "
                               f"@32768={profile.kv_bytes(32768)}")

        swa_list = {"model_info": {
            "gemma4.block_count": 6,
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": [8, 8, 8, 8, 8, 2],
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": [True, True, True, True,
                                                         True, False],
            "gemma4.embedding_length": 2816,
            "gemma4.context_length": 262144,
        }}
        gemma4 = orun.model_profile(swa_list)
        failures += _fit_check("a per-layer SWA pattern keeps its own key/value dims",
                               gemma4.kv_slope() == 4096
                               and gemma4.kv_bytes(8192) == 117440512,
                               f"slope={gemma4.kv_slope()} "
                               f"@8192={gemma4.kv_bytes(8192)}")

        mla = orun.model_profile(MLA_SHOW)
        expected_mla = 61 * (512 + 64) * 2
        failures += _fit_check("MLA KV uses the compressed latent plus rope tail",
                               orun.kv_bytes_per_token(MLA_SHOW) == expected_mla
                               and mla.kv_slope() == expected_mla,
                               f"kv={orun.kv_bytes_per_token(MLA_SHOW)} "
                               f"expected={expected_mla}")

        hybrid = orun.model_profile(HYBRID_SHOW)
        expected_ssm = 18 * ((2048 * 128 * 4) + (2048 * 4 * 4))
        failures += _fit_check("SSM state is sized from ssm.* per recurrent layer",
                               hybrid.recurrent_layers == 18
                               and hybrid.recurrent_bytes == expected_ssm
                               and "SSM state" in hybrid.recurrent_note,
                               f"layers={hybrid.recurrent_layers} "
                               f"bytes={hybrid.recurrent_bytes} expected={expected_ssm}")
        hybrid_fit = orun.estimate_max_gpu_ctx(
            HYBRID_SHOW, 2_000_000_000,
            orun.MachineGpu(devices=[(16384, 16000)], env={}))
        failures += _fit_check("the GPU estimate reserves the offloaded recurrent state",
                               hybrid_fit.recurrent_gpu_bytes == expected_ssm
                               and hybrid_fit.ctx > 0,
                               f"gpu_state={hybrid_fit.recurrent_gpu_bytes} "
                               f"ctx={hybrid_fit.ctx}")

        profiled = orun.model_profile(PROFILE_SHOW, PROFILE_ENTRY)
        expected_weights = int(494032768
                               * orun.quantization_bytes_per_param("Q4_K_M"))
        failures += _fit_check("weights fall back to params × quantization",
                               profiled.weights_bytes == expected_weights
                               and "params" in profiled.weights_source
                               and profiled.split_estimate,
                               f"weights={profiled.weights_bytes} "
                               f"expected={expected_weights} "
                               f"source={profiled.weights_source}")
        failures += _fit_check("weights split puts output first, embeddings host-side",
                               profiled.gpu_weight_share(1)[0] == profiled.output_bytes
                               and profiled.gpu_weight_share(
                                   profiled.total_layers)[0] == profiled.weights_bytes
                               and profiled.output_bytes > 0
                               and profiled.embedding_bytes > 0
                               and profiled.block_bytes > 0,
                               f"layer1={profiled.gpu_weight_share(1)} "
                               f"full={profiled.gpu_weight_share(profiled.total_layers)}")

        details = PROFILE_ENTRY["details"]
        tiny = orun.model_profile(PROFILE_SHOW, {"size": expected_weights // 4,
                                                 "details": details})
        huge = orun.model_profile(PROFILE_SHOW, {"size": expected_weights * 2,
                                                 "details": details})
        failures += _fit_check("params/quant cross-checks a suspicious tags size",
                               tiny.weights_bytes == expected_weights
                               and any("under" in note for note in tiny.notes)
                               and huge.weights_bytes == expected_weights * 2
                               and any("above" in note for note in huge.notes),
                               f"tiny={tiny.weights_bytes} huge={huge.weights_bytes}")
        cloud = orun.model_profile(PROFILE_SHOW, {
            "size": 317, "remote_model": "m1", "remote_host": "https://ollama.com",
            "details": details})
        failures += _fit_check("a remote/cloud tag is not sized from params",
                               cloud.weights_bytes == 0
                               and any("remote" in note for note in cloud.notes),
                               f"weights={cloud.weights_bytes} notes={cloud.notes}")

        vocab_show = {"model_info": {
            "qwen2.block_count": 24,
            "qwen2.attention.head_count": 14,
            "qwen2.attention.head_count_kv": 2,
            "qwen2.embedding_length": 896,
            "tokenizer.ggml.eos_token_id": 151645,
            "tokenizer.ggml.bos_token_id": 151643,
        }}
        failures += _fit_check("vocabulary falls back to the largest token id",
                               orun.model_profile(vocab_show).vocab_size == 151646,
                               str(orun.model_profile(vocab_show).vocab_size))
        sp_show = {"model_info": {
            "gemma3.block_count": 2,
            "gemma3.attention.head_count_kv": 1,
            "gemma3.embedding_length": 64,
            "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.ggml.eos_token_id": 1,
        }}
        failures += _fit_check("tiny sentencepiece ids do not fake a vocabulary",
                               orun.model_profile(sp_show).vocab_size is None)

        bert = orun.model_profile({"model_info": {
            "nomic-bert.block_count": 12,
            "nomic-bert.attention.head_count": 12,
            "nomic-bert.attention.causal": False,
            "nomic-bert.embedding_length": 768,
        }})
        failures += _fit_check("non-causal models report no KV cache",
                               not bert.kv_layers and "causal" in bert.error,
                               f"layers={bert.kv_layers} error={bert.error}")
    finally:
        orun.ctxfit.CTXFIT_CALIBRATION_PATH = old_path
        calibration_tmp.cleanup()
    return failures


def test_cmd_num_ctx_max_gpu() -> int:
    failures = 0
    old_calibration_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
    calibration_tmp = tempfile.TemporaryDirectory()
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(calibration_tmp.name) / "ctxfit.json"
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _CtxFitClient()
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        original = orun.ctxfit.probe_machine_gpu
        orun.ctxfit.probe_machine_gpu = lambda **kwargs: orun.MachineGpu(
            devices=[(4096, 3500)], env={}, source="local")
        try:
            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-gpu")
            out = _plain(cap.get())
            failures += _fit_check("/num-ctx --max-gpu estimates and sets num_ctx",
                               repl.session.parameters.get("num_ctx") == 48128
                               and "estimate only" in out and client.load_ctxs == [],
                               f"params={repl.session.parameters} out={out[-160:]}")

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-gpu --probe")
            out = _plain(cap.get())
            failures += _fit_check("/num-ctx --max-gpu --probe verifies by loading",
                               repl.session.parameters.get("num_ctx") == client.truth()
                               and "100% GPU" in out and "model is loaded" in out,
                               f"params={repl.session.parameters} probes={client.load_ctxs}")

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-gpu --bogus")
            failures += _fit_check("/num-ctx --max-gpu rejects unknown flags",
                               "usage:" in cap.get())

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--help")
            out = _plain(cap.get())
            failures += _fit_check("/num-ctx --help documents max-gpu and probe",
                               "--max-gpu" in out and "--probe" in out
                               and "max_gpu_margin_mb" in out)

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "")
            failures += _fit_check("/num-ctx status hints at --max-gpu",
                               "--max-gpu" in cap.get())
        finally:
            orun.ctxfit.probe_machine_gpu = original
            orun.ctxfit.CTXFIT_CALIBRATION_PATH = old_calibration_path
            calibration_tmp.cleanup()
    return failures


def test_ram_helpers() -> int:
    failures = 0
    values = orun.parse_meminfo(
        "MemTotal:       32717456 kB\n"
        "MemFree:        23318456 kB\n"
        "MemAvailable:   29306612 kB\n"
        "SwapTotal:       8388604 kB\n"
        "Bogus:           not a number\n")
    failures += _fit_check("parse_meminfo reads MemTotal/MemAvailable in bytes",
                           values == {"MemTotal": 32717456 * 1024,
                                      "MemFree": 23318456 * 1024,
                                      "MemAvailable": 29306612 * 1024},
                           repr(values))
    failures += _fit_check("parse_meminfo falls back to MemFree without MemAvailable",
                           orun.parse_meminfo("MemTotal: 8 kB\nMemFree: 3 kB\n")
                           == {"MemTotal": 8192, "MemFree": 3072})

    got = orun.local_ram_probe(reader=lambda: "MemTotal: 8 kB\nMemAvailable: 3 kB\n")
    failures += _fit_check("local_ram_probe parses an injected reader",
                           got == (8192, 3072), repr(got))

    def _missing():
        raise OSError("no /proc/meminfo here")

    failures += _fit_check("local_ram_probe survives a missing /proc/meminfo",
                           orun.local_ram_probe(reader=_missing) is None)

    class _Proc:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout

    got = orun.ssh_ram_probe("h", runner=lambda *a, **k: _Proc(
        0, "MemTotal: 16 kB\nMemAvailable: 6 kB\n"))
    failures += _fit_check("ssh_ram_probe parses the remote /proc/meminfo",
                           got == (16 * 1024, 6 * 1024), repr(got))
    failures += _fit_check("ssh_ram_probe survives a failed ssh",
                           orun.ssh_ram_probe("h", runner=lambda *a, **k: _Proc(255)) is None)
    failures += _fit_check("SSH_RAM_COMMAND reads /proc/meminfo",
                           orun.SSH_RAM_COMMAND == "cat /proc/meminfo")
    return failures


class _CpuFitClient(_FakeClient):
    """Fake Ollama whose /api/ps host bytes grow with num_ctx (for --max-cpu).

    `cpu_weights` and `cpu_kv` shape the reported host footprint -- the RAM
    the fit left on the CPU plus its per-token KV -- and `vram_weights` the
    reported size_vram; `fail_above` makes loads beyond a context raise like
    an out-of-memory late in a real one.
    """

    def __init__(self, *, weights=4_000_000_000, cpu_weights=2_000_000_000,
                 cpu_kv=40000, vram_weights=1_000_000_000, model_max=262144,
                 fail_above=None, base="http://127.0.0.1:11434", show_data=None):
        super().__init__(models=["m1"])
        self.base = base
        self.weights = weights
        self.cpu_weights = cpu_weights
        self.cpu_kv = cpu_kv
        self.vram_weights = vram_weights
        self.fail_above = fail_above
        self.current = None
        self.load_ctxs = []
        self.show_data = show_data if show_data is not None else {
            "model_info": dict(CPU_SHOW["model_info"],
                               **{"qwen2.context_length": model_max})}

    def tags(self):
        return [{"name": "m1", "size": self.weights}]

    def ps(self):
        if self.current is None:
            return []
        ctx, host, vram = self.current
        return [{"name": "m1", "model": "m1", "size": host + vram,
                 "size_vram": vram, "context_length": ctx}]

    def unload(self, model):
        self.unloaded.append(model)
        self.current = None

    def load(self, model, *, keepalive=None, options=None):
        self.loaded.append(model)
        self.load_options.append(dict(options or {}))
        ctx = int((options or {}).get("num_ctx") or 0)
        if self.fail_above is not None and ctx > self.fail_above:
            raise orun.OllamaError("CUDA error: out of memory")
        self.current = (ctx, self.cpu_weights + self.cpu_kv * ctx, self.vram_weights)
        self.load_ctxs.append(ctx)

    def truth(self, target: int) -> int:
        """Largest 256-aligned context whose reported host RAM is <= target."""
        room = target - self.cpu_weights
        if room < 0:
            return 0
        return (room // self.cpu_kv // 256) * 256


def test_cpu_fit() -> int:
    failures = 0
    old_calibration_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
    calibration_tmp = tempfile.TemporaryDirectory()
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(calibration_tmp.name) / "ctxfit.json"
    GIB = 2 ** 30
    weights = 4_000_000_000
    ram = orun.MachineRam(total_bytes=16 * GIB, available_bytes=16 * GIB, source="local")
    machine = orun.MachineGpu(devices=[(4096, 3300)], env={})

    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, ram, percent=30,
                                    machine=machine, pinned_layers=8)
    failures += _fit_check("estimate_max_cpu_ctx solves a pinned placement",
                           fit.ctx == 32768 and fit.capped_by_model
                           and fit.cpu_weights == 2896551724
                           and fit.cpu_kv_per_token == 43008
                           and fit.footprint_bytes == 4574273324,
                           f"ctx={fit.ctx} cpu_w={fit.cpu_weights} kv={fit.cpu_kv_per_token} "
                           f"fp={fit.footprint_bytes} capped={fit.capped_by_model}")

    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, ram, percent=30, machine=machine)
    failures += _fit_check("estimate_max_cpu_ctx fits the GPU tail when unpinned",
                           fit.gpu_layers == 11 and fit.cpu_weights == 2482758620
                           and fit.cpu_kv_per_token == 36864 and fit.capped_by_model,
                           f"layers={fit.gpu_layers} cpu_w={fit.cpu_weights} "
                           f"kv={fit.cpu_kv_per_token}")

    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, ram, percent=30, machine=None)
    failures += _fit_check("estimate_max_cpu_ctx assumes all-CPU without GPU data",
                           fit.ctx == 15360 and fit.gpu_layers == 0
                           and fit.cpu_weights == weights and fit.cpu_kv_per_token == 57344
                           and fit.footprint_bytes <= fit.target_bytes
                           and any("no GPU memory data" in note for note in fit.notes),
                           f"ctx={fit.ctx} fp={fit.footprint_bytes} target={fit.target_bytes}")

    empirical_cpu_points = [
        (ctx, weights + 256 * 2 ** 20 + 100000 * ctx)
        for ctx in (4096, 16384)
    ]
    calibrated_cpu = orun.estimate_max_cpu_ctx(
        CPU_SHOW, weights, ram, percent=30, machine=None,
        empirical_points=empirical_cpu_points)
    failures += _fit_check("CPU estimate uses the conservative measured byte slope",
                           calibrated_cpu.cpu_kv_per_token == 100000
                           and calibrated_cpu.ctx < fit.ctx
                           and any("calibrated from" in note for note in calibrated_cpu.notes),
                           f"slope={calibrated_cpu.cpu_kv_per_token} "
                           f"ctx={calibrated_cpu.ctx} baseline={fit.ctx}")

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "ctxfit.json"
        cached_client = _CpuFitClient()
        cached_machine = orun.MachineGpu()
        cached_ram = orun.MachineRam(total_bytes=16 * GIB, available_bytes=16 * GIB,
                                     source="local")
        base_cpu = orun.estimate_max_cpu_ctx(cached_client.show_data,
                                             cached_client.weights, cached_ram,
                                             percent=30, machine=cached_machine)
        cached_req = orun.FitRequest(client=cached_client, model="m1", config={},
                                     machine=cached_machine, ram=cached_ram,
                                     calibration_path=str(cache_path))
        cache_key = orun.ctxfit._calibration_scope(
            cached_req, mode="cpu", model_digest="", weight_bytes=cached_client.weights,
            show_data=cached_client.show_data, machine=cached_machine,
            pinned_layers=None, gpu_layers=base_cpu.gpu_layers)
        cache = orun.CtxCalibrationStore(cache_path)
        for ctx in (4096, 16384):
            cache.add(cache_key, ctx, 2_000_000_000 + 256 * 2 ** 20 + 80000 * ctx)
        with orun.console.capture() as cap:
            chosen = orun.max_cpu_context(cached_req, percent=30)
        failures += _fit_check("estimate-only CPU fit reuses stored measurements without loading",
                               chosen is not None and chosen < base_cpu.ctx
                               and "calibrated from" in _plain(cap.get())
                               and not cached_client.load_ctxs,
                               f"chosen={chosen} baseline={base_cpu.ctx} "
                               f"out={_plain(cap.get())[-180:]}")

    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, ram, percent=30,
                                    machine=machine, pinned_layers=29)
    failures += _fit_check("estimate_max_cpu_ctx flags a placement that cannot grow",
                           fit.cpu_kv_per_token == 0
                           and any("cannot grow" in note for note in fit.notes),
                           f"kv={fit.cpu_kv_per_token} notes={fit.notes}")

    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, ram, percent=30,
                                    resident_bytes=2 * GIB, machine=machine, pinned_layers=8)
    failures += _fit_check("estimate_max_cpu_ctx counts the resident model in the budget",
                           fit.budget_bytes == 18 * GIB
                           and fit.target_bytes == 18 * GIB * 30 // 100
                           and any("back into the budget" in note for note in fit.notes),
                           f"budget={fit.budget_bytes} notes={fit.notes}")

    tiny = orun.MachineRam(total_bytes=16 * GIB, available_bytes=2 * GIB, source="local")
    fit = orun.estimate_max_cpu_ctx(CPU_SHOW, weights, tiny, percent=5, machine=machine)
    failures += _fit_check("estimate_max_cpu_ctx refuses a target under the floor",
                           not fit.feasible and "past the" in fit.error, fit.error)

    # ── probe loop ──────────────────────────────────────────────────────────
    def run_cpu_probe(client, *, available=16 * GIB, percent=30, **request_kwargs):
        previous_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
        probe_tmp = tempfile.TemporaryDirectory()
        orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(probe_tmp.name) / "ctxfit.json"
        request = orun.FitRequest(
            client=client, model="m1", config={"max_cpu_probes": 10},
            ram=orun.MachineRam(total_bytes=16 * GIB, available_bytes=available,
                                source="local"),
            machine=orun.MachineGpu(devices=[(4096, 3300)], env={}),
            **request_kwargs)
        try:
            with orun.console.capture() as cap:
                chosen = orun.max_cpu_context(request, percent=percent, probe=True)
            return chosen, _plain(cap.get())
        finally:
            orun.ctxfit.CTXFIT_CALIBRATION_PATH = previous_path
            probe_tmp.cleanup()

    target = 16 * GIB * 30 // 100
    client = _CpuFitClient()
    chosen, out = run_cpu_probe(client)
    failures += _fit_check("--max-cpu --probe lands on the simulated boundary",
                           chosen == client.truth(target) == 78848
                           and client.load_ctxs == [54272, 79360, 78848, 79104]
                           and "host RAM" in out,
                           f"chosen={chosen} truth={client.truth(target)} "
                           f"probes={client.load_ctxs}")

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "ctxfit.json"
        cached_client = _CpuFitClient()
        cached_machine = orun.MachineGpu(devices=[(4096, 3300)], env={}, source="local")
        cached_ram = orun.MachineRam(total_bytes=16 * GIB, available_bytes=16 * GIB,
                                     source="local")
        baseline = orun.estimate_max_cpu_ctx(
            cached_client.show_data, cached_client.weights, cached_ram,
            percent=30, machine=cached_machine)
        cached_client.cpu_weights = baseline.cpu_weights
        cached_client.cpu_kv = baseline.cpu_kv_per_token
        cached_req = orun.FitRequest(
            client=cached_client, model="m1", config={"max_cpu_probes": 8},
            ram=cached_ram, machine=cached_machine, num_gpu=baseline.gpu_layers,
            calibration_path=str(cache_path))
        cache_key = orun.ctxfit._calibration_scope(
            cached_req, mode="cpu", model_digest="", weight_bytes=cached_client.weights,
            show_data=cached_client.show_data, machine=cached_machine,
            pinned_layers=baseline.gpu_layers, gpu_layers=baseline.gpu_layers)
        cache = orun.CtxCalibrationStore(cache_path)
        for ctx in (4096, 16384):
            cache.add(cache_key, ctx, cached_client.cpu_weights + cached_client.cpu_kv * ctx)
        with orun.console.capture() as cap:
            chosen = orun.max_cpu_context(cached_req, percent=30, probe=True)
        failures += _fit_check("cached CPU slope solves the probe boundary in fewer loads",
                               chosen == cached_client.truth(target)
                               and len(cached_client.load_ctxs) <= 2
                               and "calibrated from" in _plain(cap.get()),
                               f"chosen={chosen} truth={cached_client.truth(target)} "
                               f"probes={cached_client.load_ctxs}")

    capped = _CpuFitClient(model_max=4096)
    chosen, out = run_cpu_probe(capped)
    failures += _fit_check("--max-cpu --probe stops at the model's max context",
                           chosen == 4096 and "capped by the model's max" in out,
                           f"chosen={chosen} probes={capped.load_ctxs}")

    oom = _CpuFitClient(fail_above=50000)
    chosen, out = run_cpu_probe(oom)
    failures += _fit_check("--max-cpu --probe backs off from load failures",
                           chosen is not None and chosen <= 49920
                           and "load failed" in out,
                           f"chosen={chosen} probes={oom.load_ctxs}")

    over = _CpuFitClient(cpu_weights=6_000_000_000)
    chosen, out = run_cpu_probe(over)
    failures += _fit_check("--max-cpu --probe reports a target under the model's floor",
                           chosen is None and "past the target" in out, out[-200:])

    class _BlindCpu(_CpuFitClient):
        def ps(self):
            raise orun.OllamaError("no ps")

    chosen, out = run_cpu_probe(_BlindCpu())
    failures += _fit_check("--max-cpu --probe reports when the model can't be measured",
                           chosen is None and "/api/ps" in out, out[-200:])

    # ── orchestration ───────────────────────────────────────────────────────
    estimate_only = _CpuFitClient()
    request = orun.FitRequest(client=estimate_only, model="m1", config={},
                              ram=orun.MachineRam(total_bytes=16 * GIB,
                                                  available_bytes=16 * GIB, source="local"),
                              machine=orun.MachineGpu(devices=[(4096, 3300)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(request, percent=30)
    out = _plain(cap.get())
    failures += _fit_check("max_cpu_context estimates without loading",
                           chosen is not None and "estimate only" in out
                           and estimate_only.load_ctxs == [],
                           f"chosen={chosen}")

    resident = _CpuFitClient()
    resident.current = (65536, 3_000_000_000, 1_000_000_000)
    request = orun.FitRequest(client=resident, model="m1", config={},
                              ram=orun.MachineRam(total_bytes=16 * GIB,
                                                  available_bytes=16 * GIB, source="local"),
                              machine=orun.MachineGpu(devices=[(4096, 3300)], env={}))
    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(request, percent=30)
    out = _plain(cap.get())
    failures += _fit_check("max_cpu_context credits the model's own host RAM",
                           chosen is not None and "back on reload" in out
                           and resident.load_ctxs == [],
                           f"chosen={chosen} out={out[-200:]}")

    request = orun.FitRequest(client=_CpuFitClient(), model="m1",
                              ram=orun.MachineRam(error="cannot read /proc/meminfo"))
    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(request, percent=30)
    failures += _fit_check("max_cpu_context reports unreadable DRAM",
                           chosen is None and "cannot read free memory" in cap.get(),
                           cap.get()[-160:])

    remote_like = _CpuFitClient(base="http://10.9.9.9:11434")
    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(orun.FitRequest(client=remote_like, model="m1"),
                                      percent=30)
    failures += _fit_check("max_cpu_context needs --remote for non-local hosts",
                           chosen is None and "--remote" in cap.get(), cap.get()[-160:])

    request = orun.FitRequest(client=_CpuFitClient(), model="m1",
                              ram=orun.MachineRam(total_bytes=16 * GIB,
                                                  available_bytes=16 * GIB))
    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(request, percent=0)
    failures += _fit_check("max_cpu_context validates the percentage",
                           chosen is None and "1 and 100" in cap.get())

    class _NoTags(_CpuFitClient):
        def tags(self):
            return []

    with orun.console.capture() as cap:
        chosen = orun.max_cpu_context(orun.FitRequest(
            client=_NoTags(), model="m1",
            ram=orun.MachineRam(total_bytes=16 * GIB, available_bytes=16 * GIB)),
            percent=30)
    failures += _fit_check("max_cpu_context needs the model in /api/tags",
                           chosen is None and "/api/tags" in cap.get())
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = old_calibration_path
    calibration_tmp.cleanup()
    return failures


def test_cmd_num_ctx_max_cpu() -> int:
    failures = 0
    old_calibration_path = orun.ctxfit.CTXFIT_CALIBRATION_PATH
    calibration_tmp = tempfile.TemporaryDirectory()
    orun.ctxfit.CTXFIT_CALIBRATION_PATH = Path(calibration_tmp.name) / "ctxfit.json"
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _CpuFitClient()
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))

        original_gpu = orun.ctxfit.probe_machine_gpu
        original_ram = orun.ctxfit.probe_machine_ram
        orun.ctxfit.probe_machine_gpu = lambda **kwargs: orun.MachineGpu(
            devices=[(4096, 3300)], env={}, source="local")
        orun.ctxfit.probe_machine_ram = lambda **kwargs: orun.MachineRam(
            total_bytes=16 * 2 ** 30, available_bytes=16 * 2 ** 30, source="local")
        try:
            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu 30")
            out = _plain(cap.get())
            failures += _fit_check("/num-ctx --max-cpu estimates without loading",
                                   repl.session.parameters.get("num_ctx") is not None
                                   and "estimate only" in out
                                   and client.load_ctxs == [],
                                   f"params={repl.session.parameters} out={out[-160:]}")

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu 30 --probe")
            out = _plain(cap.get())
            target = 16 * 2 ** 30 * 30 // 100
            failures += _fit_check("/num-ctx --max-cpu N --probe lands on the boundary",
                                   repl.session.parameters.get("num_ctx") == client.truth(target)
                                   and "host RAM" in out and "model is loaded" in out,
                                   f"params={repl.session.parameters} "
                                   f"truth={client.truth(target)} probes={client.load_ctxs}")

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu")
            failures += _fit_check("/num-ctx --max-cpu needs a percentage",
                                   "usage:" in cap.get())

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu 0")
            failures += _fit_check("/num-ctx --max-cpu rejects 0",
                                   "1 and 100" in cap.get())

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu=101")
            failures += _fit_check("/num-ctx --max-cpu=101 rejects out of range",
                                   "1 and 100" in cap.get())

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--max-cpu 30 --max-gpu")
            failures += _fit_check("/num-ctx --max-cpu rejects mixing with --max-gpu",
                                   "usage:" in cap.get())

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "--help")
            out = _plain(cap.get())
            failures += _fit_check("/num-ctx --help documents max-cpu",
                                   "--max-cpu" in out and "max_cpu_overhead_mb" in out)

            with orun.console.capture() as cap:
                orun.cmd_num_ctx(repl, "")
            failures += _fit_check("/num-ctx status hints at --max-cpu",
                                   "--max-cpu" in cap.get())
        finally:
            orun.ctxfit.probe_machine_gpu = original_gpu
            orun.ctxfit.probe_machine_ram = original_ram
            orun.ctxfit.CTXFIT_CALIBRATION_PATH = old_calibration_path
            calibration_tmp.cleanup()
    return failures


def test_background_cli() -> int:
    """Background flags, gating and launch overrides."""
    failures = 0
    parser = orun.build_parser()
    args = parser.parse_args(["m1", "--no-background", "--close-session", "abc"])
    failures = _check(failures, "--no-background and --close-session parse",
                      args.no_background is True
                      and args.close_session == "abc")
    config = dict(orun.DEFAULT_CONFIG)
    failures = _check(
        failures, "--no-background disables the session host",
        orun.background_enabled(args, config) is False)
    plain = parser.parse_args(["m1"])
    failures = _check(
        failures, "a non-terminal run never starts a session host",
        orun.background_enabled(plain, config) is False)
    failures = _check(
        failures, "config background_sessions:false is honoured",
        orun.background_enabled(plain, {**config,
                                        "background_sessions": False}) is False)
    failures = _check(
        failures, "--no-save cannot start a resumable host",
        orun.background_enabled(parser.parse_args(["m1", "--no-save"]),
                                config) is False)
    session = orun.Session.create("m1", "http://fake")
    session.system = "custom"
    session.parameters = {"temperature": 0.2}
    session.gpu_layers = "10"
    session.title = "t"
    overrides = orun.host_overrides(
        parser.parse_args(["m1", "--system", "custom", "--option",
                           "temperature=0.2", "--gpu-layers", "10",
                           "--title", "t"]), session)
    failures = _check(
        failures, "explicit launch settings become host overrides",
        overrides.get("model") == "m1"
        and overrides.get("system") == "custom"
        and overrides.get("gpu_layers") == "10"
        and overrides.get("parameters") == {"temperature": 0.2}
        and overrides.get("title") == "t",
        str(overrides))
    return failures


def test_emit_and_cli_functions() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        session = orun.Session.create("m1", "http://h", title="alpha")
        session.add_message("user", "find me please")
        store.save(session)

        args_table = orun.build_parser().parse_args(["--list"])
        with orun.console.capture() as cap:
            code = orun.cli_list(args_table, store)
        if code == 0 and "alpha" in cap.get():
            print("PASS  cli_list renders a table")
        else:
            print(f"FAIL  cli_list table: {cap.get()!r}")
            failures += 1

        args_json = orun.build_parser().parse_args(["--list", "--json"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            orun.cli_list(args_json, store)
        try:
            data = json.loads(out.getvalue())
        except ValueError:
            data = None
        if isinstance(data, list) and data and data[0]["title"] == "alpha":
            print("PASS  cli_list --json emits parseable JSON")
        else:
            print(f"FAIL  cli_list --json: {out.getvalue()!r}")
            failures += 1

        args_csv = orun.build_parser().parse_args(["--list", "--csv"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            orun.cli_list(args_csv, store)
        text = out.getvalue()
        if text.splitlines()[0].split(",")[0] == "id" and "alpha" in text:
            print("PASS  cli_list --csv emits a CSV header")
        else:
            print(f"FAIL  cli_list --csv: {text!r}")
            failures += 1

        args_search = orun.build_parser().parse_args(["--search", "find", "--markdown"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = orun.cli_search(args_search, store)
        text = out.getvalue()
        if code == 0 and text.startswith("| id |") and "alpha" in text:
            print("PASS  cli_search --markdown emits a markdown table")
        else:
            print(f"FAIL  cli_search markdown: {text!r}")
            failures += 1

        args_export = orun.build_parser().parse_args(
            ["--export", session.id, "--export-format", "json"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = orun.cli_export(args_export, store)
        if code == 0 and '"title": "alpha"' in out.getvalue():
            print("PASS  cli_export --export-format json")
        else:
            print(f"FAIL  cli_export json: {out.getvalue()!r}")
            failures += 1

        args_missing = orun.build_parser().parse_args(["--show", "no-such-session"])
        with orun.console.capture() as cap:
            code = orun.cli_show(args_missing, store, dict(orun.DEFAULT_CONFIG))
        if code == 1 and "no such session" in cap.get():
            print("PASS  cli_show reports a missing session")
        else:
            print(f"FAIL  cli_show missing: {cap.get()!r}")
            failures += 1
    return failures


def test_machine_output_verbatim() -> int:
    """Machine-readable stdout must be byte-verbatim, never wrapped (#42)."""
    failures = 0

    def check(name, condition, detail=""):
        nonlocal failures
        if condition:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {detail}")
            failures += 1

    def machine(call):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = call()
        return code, out.getvalue()

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                                  index_path=Path(tmp) / "index.json")
        long_title = "a long session title " + "x" * 200
        session = orun.Session.create("m1", "http://h", title=long_title)
        session.add_message("user", "hello")
        store.save(session)

        args = orun.build_parser().parse_args(["--list", "--json"])
        _, out = machine(lambda: orun.cli_list(args, store))
        try:
            data = json.loads(out)
            ok = isinstance(data, list) and bool(data) and data[0]["title"] == long_title
        except ValueError:
            ok = False
        check("--list --json parses with a long title", ok, out[:120])

        args = orun.build_parser().parse_args(["--list", "--csv"])
        _, out = machine(lambda: orun.cli_list(args, store))
        rows = list(csv.reader(io.StringIO(out)))
        check("--list --csv keeps one row per session",
              len(rows) == 2 and rows[1][1] == long_title, out[:120])

        args = orun.build_parser().parse_args(["--list", "--markdown"])
        _, out = machine(lambda: orun.cli_list(args, store))
        rows = out.splitlines()
        check("--list --markdown keeps the row on one line",
              len(rows) == 3 and long_title in rows[2], out[:120])

        args = orun.build_parser().parse_args(["--show", session.id, "--json"])
        _, out = machine(lambda: orun.cli_show(args, store, dict(orun.DEFAULT_CONFIG)))
        try:
            ok = json.loads(out)["title"] == long_title
        except (ValueError, KeyError):
            ok = False
        check("--show --json is parseable", ok, out[:120])

        args = orun.build_parser().parse_args(
            ["--export", session.id, "--export-format", "json"])
        _, out = machine(lambda: orun.cli_export(args, store))
        try:
            ok = json.loads(out)["title"] == long_title
        except (ValueError, KeyError):
            ok = False
        check("--export json is parseable", ok, out[:120])

        args = orun.build_parser().parse_args(
            ["--export", session.id, "--export-format", "md"])
        _, out = machine(lambda: orun.cli_export(args, store))
        check("--export md keeps the long line intact", long_title in out, out[:120])

        class EmbedClient:
            def embed(self, model, inputs, *, truncate=True, dimensions=None,
                      options=None):
                return {"embeddings": [[0.1, 0.2]], "note": "y" * 300}

        code, out = machine(lambda: orun.run_embed(
            EmbedClient(), "m", "p", truncate=True, dimensions=None, json_out=True))
        try:
            payload = json.loads(out)
            ok = code == 0 and payload["note"] == "y" * 300
        except ValueError:
            ok = False
        check("embedding --json is verbatim", ok, out[:120])

        client = _FakeClient(chat_once_reply="z" * 300)
        args = orun.build_parser().parse_args(["m1", "--json", "--no-tools", "--no-save"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        code, out = machine(lambda: orun.run_once(
            args, dict(orun.DEFAULT_CONFIG), client, store, repl.session, "hello", repl))
        try:
            payload = json.loads(out)
            ok = code == 0 and payload["message"]["content"] == "z" * 300
        except (ValueError, KeyError):
            ok = False
        check("chat --json is verbatim", ok, out[:120])
    return failures


def test_build_parser_flags() -> int:
    failures = 0
    parser = orun.build_parser()

    args = parser.parse_args(["m1", "--num-ctx", "4096"])
    if args.num_ctx == "4096" and orun.parse_num_ctx_arg(args.num_ctx) == ("value", 4096):
        print("PASS  --num-ctx parses an integer")
    else:
        print(f"FAIL  --num-ctx: {args.num_ctx!r}")
        failures += 1

    probe_args = parser.parse_args(["m1", "--num-ctx", "max-gpu", "--max-gpu-probe"])
    if orun.parse_num_ctx_arg(probe_args.num_ctx) == ("max-gpu", None) \
            and probe_args.probe:
        print("PASS  --num-ctx max-gpu and --max-gpu-probe (alias) parse")
    else:
        print(f"FAIL  --num-ctx max-gpu: {probe_args.num_ctx!r} {probe_args.probe!r}")
        failures += 1

    cpu_args = parser.parse_args(["m1", "--num-ctx", "--max-cpu", "80", "--probe"])
    if cpu_args.num_ctx == "" and cpu_args.max_cpu == "80" and cpu_args.probe \
            and orun.parse_fill_percent(cpu_args.max_cpu) == 80:
        print("PASS  --num-ctx --max-cpu N and --probe parse")
    else:
        print(f"FAIL  --num-ctx --max-cpu: {cpu_args.num_ctx!r} {cpu_args.max_cpu!r} "
              f"{cpu_args.probe!r}")
        failures += 1

    if parser.parse_args(["m1", "--num-ctx"]).num_ctx == "" \
            and parser.parse_args(["m1", "--max-cpu", "80"]).max_cpu == "80":
        print("PASS  --num-ctx alone and --max-cpu alone parse (validated in main)")
    else:
        print("FAIL  --num-ctx/--max-cpu bare forms")
        failures += 1

    if parser.parse_args(["m1"]).num_ctx is None:
        print("PASS  --num-ctx defaults to None")
    else:
        print("FAIL  --num-ctx default")
        failures += 1

    args = parser.parse_args(["m1", "--ui", "tui"])
    if args.ui == "tui" and parser.parse_args(["m1"]).ui is None:
        print("PASS  --ui accepts tui and defaults to None (config decides)")
    else:
        print(f"FAIL  --ui: {getattr(args, 'ui', None)!r}")
        failures += 1

    verbose_args = parser.parse_args(["m1", "--verbose"])
    if verbose_args.verbose is True:
        print("PASS  --verbose still parses (obsolete, kept for ollama compatibility)")
    else:
        print(f"FAIL  --verbose: {getattr(verbose_args, 'verbose', None)!r}")
        failures += 1

    failures = _check(failures, "/queue is wired into the command tables",
                      orun.COMMAND_HANDLERS.get("/queue") is orun.cmd_queue
                      and "/queue" in orun.COMMAND_INFO)

    # front-end selection (config + flag + availability)
    class _Args:
        ui = None

    failures = _check(failures, "ui_mode defaults to the inline REPL",
                      orun.ui_mode(_Args(), {}) == "inline")
    failures = _check(failures, "ui_mode honours the config key",
                      orun.ui_mode(_Args(), {"ui": "tui"}) == "tui")
    args_auto = _Args()
    args_auto.ui = "auto"
    failures = _check(failures, "ui_mode auto resolves to the TUI",
                      orun.ui_mode(args_auto, {}) == "tui")
    failures = _check(failures, "ui_mode rejects nonsense",
                      orun.ui_mode(_Args(), {"ui": "wat"}) == "inline")
    failures = _check(failures, "wants_tui is False unless asked for",
                      orun.wants_tui(_Args(), {}) is False)
    import ollamarun_lib.tui as _tui_mod
    _args_tui = _Args()
    _args_tui.ui = "tui"
    if orun.HAS_TEXTUAL:
        failures = _check(failures, "wants_tui follows tui_available()",
                          orun.wants_tui(_args_tui, {}) == _tui_mod.tui_available()[0])
    else:
        failures = _check(failures, "wants_tui degrades without textual",
                          orun.wants_tui(_args_tui, {}) is False)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["m1", "--unload", "--keep-loaded"])
        print("FAIL  --unload/--keep-loaded should be mutually exclusive")
        failures += 1
    except SystemExit:
        print("PASS  --unload/--keep-loaded are mutually exclusive")

    if orun.COMMAND_HANDLERS.get("/ps") is orun.cmd_ps \
            and orun.COMMAND_HANDLERS.get("/num-ctx") is orun.cmd_num_ctx \
            and "/ps" in orun.COMMAND_INFO and "/num-ctx" in orun.COMMAND_INFO:
        print("PASS  /ps and /num-ctx are wired into the REPL command tables")
    else:
        print("FAIL  /ps or /num-ctx missing from the REPL command tables")
        failures += 1
    return failures


def test_parser_intermixed() -> int:
    """Flags may sit between MODEL and PROMPT (#37)."""
    failures = 0
    parser = orun.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "sessions",
                                  index_path=Path(tmp) / "index.json")
        cases = [
            ("a flag between model and prompt",
             ["qwen3:0.6b", "--no-tools", "hi"],
             lambda a: a.model == "qwen3:0.6b" and a.prompt == ["hi"] and a.no_tools),
            ("several flags between model and prompt",
             ["qwen3:0.6b", "--no-tools", "--no-save", "hi"],
             lambda a: a.prompt == ["hi"] and a.no_tools and a.no_save),
            ("a flag with a value between model and prompt",
             ["m1", "--num-ctx", "4096", "hi"],
             lambda a: a.prompt == ["hi"] and a.num_ctx == "4096"),
            ("a repeatable flag between model and prompt",
             ["m1", "--image", "a.png", "hi"],
             lambda a: a.prompt == ["hi"] and a.image == ["a.png"]),
            ("keepalive between model and prompt",
             ["m1", "--keepalive", "2h", "hi"],
             lambda a: a.prompt == ["hi"] and a.keepalive == "2h"),
            ("option between model and prompt",
             ["m1", "--option", "temperature=0.2", "hi"],
             lambda a: a.prompt == ["hi"] and a.option == ["temperature=0.2"]),
            ("format between model and prompt",
             ["m1", "--format", "json", "hi"],
             lambda a: a.prompt == ["hi"] and a.format == "json"),
            ("flags first still parse",
             ["--no-tools", "m1", "hi"],
             lambda a: a.model == "m1" and a.prompt == ["hi"] and a.no_tools),
            ("flags last still parse",
             ["m1", "hi", "--no-tools"],
             lambda a: a.model == "m1" and a.prompt == ["hi"] and a.no_tools),
            ("a flag value is not mistaken for a prompt",
             ["m1", "--render", "plain"],
             lambda a: a.model == "m1" and a.prompt == [] and a.render == "plain"),
            ("inspection flags with no positionals",
             ["--list", "--json"],
             lambda a: a.list and a.json and a.model is None),
            ("an optional-value flag keeps consuming its value",
             ["m1", "--think", "hi"],
             lambda a: a.model == "m1" and a.prompt == [] and a.think == "hi"),
            ("multi-word prompts collect",
             ["m1", "tell", "me", "more"],
             lambda a: a.prompt == ["tell", "me", "more"]),
            ("-c prompt followed by a flag",
             ["-c", "hello world", "--render", "plain"],
             lambda a: a.continue_ == "__latest__" and a.model == "hello world"
             and a.render == "plain"),
        ]
        for name, argv, ok in cases:
            try:
                args = orun.parse_cli_args(parser, argv, store)
                passed = bool(ok(args))
            except SystemExit:
                passed = False
            if passed:
                print(f"PASS  parse_cli_args {name}")
            else:
                print(f"FAIL  parse_cli_args {name}: {argv!r}")
                failures += 1
    return failures


def test_remote_hosts() -> int:
    failures = 0
    config = dict(orun.DEFAULT_CONFIG)
    ts = {"vinipi": "100.64.0.1", "kiddo": "100.64.0.2", "g3": "100.64.0.3",
          "optiplex7070": "100.64.0.4", "guri": "100.64.0.5"}

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    check("remote pi maps to vinipi", orun.resolve_remote("pi", config, ts),
          "http://100.64.0.1:11434")
    check("remote alias optiplex -> optiplex7070",
          orun.resolve_remote("optiplex", config, ts), "http://100.64.0.4:11434")
    check("remote name case-insensitive", orun.resolve_remote("KIDDO", config, ts),
          "http://100.64.0.2:11434")
    check("remote explicit port", orun.resolve_remote("kiddo:11435", config, ts),
          "http://100.64.0.2:11435")
    check("remote magicdns fallback", orun.resolve_remote("kiddo", config, {}),
          "http://kiddo:11434")
    check("remote arbitrary host", orun.resolve_remote("lab-box", config, ts),
          "http://lab-box:11434")
    check("remote url passthrough", orun.resolve_remote("https://box:9999/x", config, ts),
          "https://box:9999")
    check("remote config override",
          orun.resolve_remote("lab", {**config, "remote_hosts": {"lab": "kiddo:11435"}}, ts),
          "http://100.64.0.2:11435")
    check("remote override keeps explicit port",
          orun.resolve_remote("lab:9999", {**config, "remote_hosts": {"lab": "kiddo:11435"}}, ts),
          "http://100.64.0.2:9999")
    check("remote_node resolves alias", orun.remote_node("optiplex", config), "optiplex7070")
    check("is_local_host localhost", orun.is_local_host("localhost:11434"), True)
    check("is_local_host remote", orun.is_local_host("g3:11434"), False)
    check("is_local_host default", orun.is_local_host(orun.DEFAULT_HOST), True)
    try:
        orun.resolve_remote("", config, ts)
        print("FAIL  empty --remote value should raise")
        failures += 1
    except ValueError:
        print("PASS  empty --remote value raises")

    parser = orun.build_parser()
    args = parser.parse_args(["m1", "--remote", "kiddo"])
    check("--remote parses", args.remote, "kiddo")
    check("--remote defaults to None", parser.parse_args(["m1"]).remote, None)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["m1", "--remote", "g3", "--host", "127.0.0.1"])
        print("FAIL  --remote/--host should be mutually exclusive")
        failures += 1
    except SystemExit:
        print("PASS  --remote/--host are mutually exclusive")

    plain_args = parser.parse_args(["m1"])
    explicit = parser.parse_args(["m1", "--remote", "g3"])
    check("session reconnect keeps a remote host",
          orun.session_reconnect(plain_args, config, orun.Session.create(
              "m1", "http://100.64.0.2:11434")),
          "http://100.64.0.2:11434")
    check("session reconnect re-resolves the remote name",
          orun.session_reconnect(plain_args, config, orun.Session.create(
              "m1", "http://100.64.0.2:11434", remote="kiddo"), ts),
          "http://100.64.0.2:11434")
    check("session reconnect skips the local default",
          orun.session_reconnect(plain_args, config, orun.Session.create(
              "m1", "http://127.0.0.1:11434")),
          None)
    check("session reconnect yields to --remote",
          orun.session_reconnect(explicit, config, orun.Session.create(
              "m1", "http://100.64.0.2:11434")),
          None)

    class _DownClient:
        base = "http://100.64.0.9:11434"

        def version(self, timeout=None):
            raise orun.OllamaError(f"cannot reach Ollama at {self.base}: boom")

    class _UpClient:
        base = "http://100.64.0.9:11434"

        def version(self, timeout=None):
            return "0.34.4"

    saved_cache = orun.helpers._TAILSCALE_CACHE
    orun.helpers._TAILSCALE_CACHE = {
        "ghost": {"Online": False, "LastSeen": "2026-10-03T10:00:00Z"}}
    try:
        with orun.console.capture() as cap:
            ok = orun.remote_preflight(_DownClient(), "ghost", config)
        text = _plain(cap.get())
        check("remote_preflight fails fast", ok, False)
        check("remote_preflight names the remote", "cannot reach remote ghost" in text, True)
        check("remote_preflight reports the tailscale state", "offline" in text, True)
    finally:
        orun.helpers._TAILSCALE_CACHE = saved_cache
    with orun.console.capture() as cap:
        ok = orun.remote_preflight(_UpClient(), "ghost", config)
    check("remote_preflight accepts a live remote", ok, True)
    check("remote_preflight prints the version", "0.34.4" in _plain(cap.get()), True)
    return failures


def test_pull_and_disk_helpers() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    seen_requests = []

    def pull_handler(method, url, json, stream):
        seen_requests.append((method, url, json))
        return _FakeResponse(status=200, lines=[
            b'{"status":"pulling manifest"}',
            b'{"status":"pulling abc","digest":"sha256:abc","total":100,"completed":40}',
            b'{"status":"success"}',
        ])

    events = []
    client = orun.OllamaClient("127.0.0.1:11434")
    client.http = _FakeSession(pull_handler)
    client.pull_model("m1", events.append)
    check("pull_model forwards progress events",
          [event.get("status") for event in events],
          ["pulling manifest", "pulling abc", "success"])
    check("pull_model posts /api/pull",
          seen_requests[0][:2], ("POST", "http://127.0.0.1:11434/api/pull"))
    check("pull_model asks for a stream", seen_requests[0][2],
          {"model": "m1", "stream": True})

    client.http = _FakeSession(
        lambda method, url, json, stream: _FakeResponse(
            status=200, lines=[b'{"error":"boom"}']))
    try:
        client.pull_model("m1")
        print("FAIL  pull_model should raise on an error event")
        failures += 1
    except orun.OllamaError:
        print("PASS  pull_model raises on an error event")

    with tempfile.TemporaryDirectory() as tmp:
        usage = orun.local_free_disk([tmp])
    check("local_free_disk reports numbers",
          bool(usage and usage[0] > 0 and usage[1] > 0), True)
    check("local_free_disk reports the probed path", usage[2], tmp)
    usage = orun.local_free_disk(["/definitely-missing-dir-zz/x"])
    check("local_free_disk climbs to an existing ancestor",
          bool(usage and usage[2] == "/"), True)

    class _Proc:
        def __init__(self, returncode, stdout=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    check("ssh_free_disk parses df output",
          orun.ssh_free_disk("box", runner=lambda *a, **k: _Proc(
              0, "490577010688 17122672640 /\n")),
          (17122672640 * 1024, 490577010688 * 1024, "/"))
    check("ssh_free_disk tolerates a failing ssh",
          orun.ssh_free_disk("box", runner=lambda *a, **k: _Proc(1)), None)

    def _no_ssh(*args, **kwargs):
        raise OSError("no ssh")

    check("ssh_free_disk tolerates no ssh binary",
          orun.ssh_free_disk("box", runner=_no_ssh), None)
    check("ssh_free_disk tolerates garbage output",
          orun.ssh_free_disk("box", runner=lambda *a, **k: _Proc(0, "nope\n")), None)

    line = orun.disk_free_line(_FakeClient(), dict(orun.DEFAULT_CONFIG))
    check("disk_free_line: unknown without ssh", "unknown" in line, True)
    line = orun.disk_free_line(orun.OllamaClient(orun.DEFAULT_HOST),
                               dict(orun.DEFAULT_CONFIG))
    check("disk_free_line: local disk", "free disk space on this machine" in line, True)
    return failures


def test_ensure_model() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    config = dict(orun.DEFAULT_CONFIG)
    interactive = orun.interactive
    saved_confirm = interactive.Confirm
    saved_pick = interactive.pick_model

    installed = _FakeClient(models=["m1"])
    check("installed model passes through",
          orun.ensure_model(installed, "m1", config, interactive=False), "m1")
    check("installed model is not pulled", installed.pulled, [])

    missing = _FakeClient(pull_events=[{"status": "success"}])
    check("--pull pulls and returns the model",
          orun.ensure_model(missing, "m9", config, assume_yes=True, quiet=True,
                            interactive=False),
          "m9")
    check("--pull actually pulled", missing.pulled, ["m9"])

    class _Yes:
        @staticmethod
        def ask(*args, **kwargs):
            return True

    class _No:
        @staticmethod
        def ask(*args, **kwargs):
            return False

    interactive.Confirm = _Yes
    try:
        yes_client = _FakeClient(pull_events=[{"status": "success"}])
        with orun.console.capture() as cap:
            got = orun.ensure_model(yes_client, "m9", config, interactive=True)
        text = _plain(cap.get())
        check("interactive yes pulls", (got, yes_client.pulled), ("m9", ["m9"]))
        check("interactive prompt prints free space", "free disk space" in text, True)

        interactive.Confirm = _No
        no_client = _FakeClient()
        picker_kwargs: dict = {}
        interactive.pick_model = (
            lambda client, current, **kwargs: picker_kwargs.update(kwargs) or "other-model")
        with orun.console.capture() as cap:
            got = orun.ensure_model(no_client, "m9", config, interactive=True,
                                    pick_on_decline=True, sort="size")
        check("declining opens the picker", got, "other-model")
        check("declining does not pull", no_client.pulled, [])
        check("decline picker honours --sort", picker_kwargs.get("sort"), "size")

        interactive.pick_model = saved_pick
        with orun.console.capture() as cap:
            got = orun.ensure_model(no_client, "m9", config, interactive=True)
        text = _plain(cap.get())
        check("declining without picker stops", got, None)
        check("decline prints a hint", "nothing to run" in text, True)
    finally:
        interactive.Confirm = saved_confirm
        interactive.pick_model = saved_pick

    with orun.console.capture() as cap:
        got = orun.ensure_model(_FakeClient(), "m9", config, interactive=False)
    text = _plain(cap.get())
    check("non-interactive missing model stops", got, None)
    check("non-interactive warning", "not installed" in text, True)

    with orun.console.capture() as cap:
        got = orun.ensure_model(_FakeClient(pull_error="disk full"), "m9", config,
                                assume_yes=True, quiet=True, interactive=False)
    check("pull failure stops", got, None)
    check("pull failure message", "pull failed: disk full" in _plain(cap.get()), True)

    parser = orun.build_parser()
    check("--pull parses", parser.parse_args(["m1", "--pull"]).pull, True)
    check("--pull defaults off", parser.parse_args(["m1"]).pull, False)
    check("/pull is wired into the REPL command tables",
          orun.COMMAND_HANDLERS.get("/pull") is orun.cmd_pull
          and "/pull" in orun.COMMAND_INFO, True)

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        client = _FakeClient(models=["m1"], pull_events=[{"status": "success"}])
        repl = orun.Repl(args, config, client, store,
                         orun.Session.create("m1", client.base))
        with orun.console.capture() as cap:
            orun.cmd_pull(repl, "m9")
        check("cmd_pull pulls", client.pulled, ["m9"])
        check("cmd_pull reports success", "pulled" in _plain(cap.get()), True)
    return failures


class _ScriptedUi(orun.ConsoleUi):
    """ConsoleUi whose `ask` replays a fixed list of answers."""

    def __init__(self, answers):
        self.answers = list(answers)

    def ask(self, question, *, choices=None, default=None):
        return self.answers.pop(0) if self.answers else "q"


class _TagClient(_FakeClient):
    """_FakeClient with real tag entries and per-model /api/show data."""

    def __init__(self, tags, contexts=None):
        super().__init__(models=[tag["name"] for tag in tags])
        self.tag_data = list(tags)
        self.contexts = dict(contexts or {})

    def show(self, model):
        self.show_calls.append(model)
        length = self.contexts.get(model)
        return {"model_info": {"llama.context_length": length}} if length else {}


def test_fuzzy_sort() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    match = orun.fuzzy_match("qwen", "qwen3:8b")
    check("fuzzy_match: prefix hit", match is not None, True)
    check("fuzzy_match: positions", match.positions if match else None,
          (0, 1, 2, 3))
    check("fuzzy_match: case-insensitive",
          orun.fuzzy_match("QWEN", "qwen3:8b") is not None, True)
    check("fuzzy_match: miss", orun.fuzzy_match("qwen", "llama3.2"), None)
    check("fuzzy_match: out of order", orun.fuzzy_match("nq", "qwen"), None)
    check("fuzzy_match: empty term", orun.fuzzy_match("", "qwen"), None)

    boundary = orun.fuzzy_match("b", "a-b")
    middle = orun.fuzzy_match("b", "ab")
    check("fuzzy_match: boundary beats mid-word",
          bool(boundary and middle and boundary.score > middle.score), True)
    consecutive = orun.fuzzy_match("qwen", "qwen3")
    scattered = orun.fuzzy_match("qwen", "qxxwxxexxn")
    check("fuzzy_match: consecutive beats scattered",
          bool(consecutive and scattered and consecutive.score > scattered.score),
          True)
    camel = orun.fuzzy_match("c", "gpuCurator")
    plain = orun.fuzzy_match("c", "gpucurator")
    check("fuzzy_match: camel boundary beats plain",
          bool(camel and plain and camel.score > plain.score), True)
    check("fuzzy_match: multi-term AND hit",
          orun.fuzzy_match_query("qwen 3b", "qwen3:8b") is not None, True)
    check("fuzzy_match: multi-term AND miss",
          orun.fuzzy_match_query("qwen zz", "qwen3:8b"), None)
    check("fuzzy_match: highlight positions",
          orun.fuzzy_positions("qwen", "qwen3:8b"), (0, 1, 2, 3))

    check("parse_model_sort: implicit fuzzy",
          orun.parse_model_sort("qwen3"), "fuzzy:qwen3")
    check("parse_model_sort: explicit fuzzy",
          orun.parse_model_sort("fuzzy qwen 3b"), "fuzzy:qwen 3b")
    check("parse_model_sort: explicit fuzzy, colon",
          orun.parse_model_sort("fuzzy:qwen"), "fuzzy:qwen")
    check("parse_model_sort: keeps the typed case",
          orun.parse_model_sort("Qwen"), "fuzzy:Qwen")
    for bad in ("", "fuzzy", "fuzzy:"):
        try:
            orun.parse_model_sort(bad)
            check(f"parse_model_sort rejects {bad!r}", False, True)
        except ValueError as exc:
            check(f"parse_model_sort rejects {bad!r}",
                  "search string" in str(exc) or "empty sort" in str(exc), True)

    models = [
        {"name": "my-qwen-model"},
        {"name": "qwen2.5:14b"},
        {"name": "qwen3:8b"},
        {"name": "llama3.2:3b"},
    ]

    def names(items):
        return [item["name"] for item in items]

    check("fuzzy sort: strongest matches first",
          names(orun.sort_models(models, "fuzzy:qwen")),
          ["qwen3:8b", "qwen2.5:14b", "my-qwen-model", "llama3.2:3b"])
    check("fuzzy sort: non-matches stay, alphabetically",
          names(orun.sort_models(models, "fuzzy:zzz")),
          ["llama3.2:3b", "my-qwen-model", "qwen2.5:14b", "qwen3:8b"])
    check("fuzzy sort: leaves the input alone", names(models),
          ["my-qwen-model", "qwen2.5:14b", "qwen3:8b", "llama3.2:3b"])

    with orun.console.capture() as cap:
        orun.console.print(orun.model_table(
            orun.sort_models(models, "fuzzy:qwen"), sort="fuzzy:qwen"))
    text = _plain(cap.get())
    check("model_table names a fuzzy sort", "fuzzy 'qwen'" in text, True)
    check("model_table still shows the non-match", "llama3.2:3b" in text, True)
    return failures


def test_model_sort() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    check("parse_model_sort: canonical", orun.parse_model_sort("name"), "name")
    check("parse_model_sort: trims and lowercases",
          orun.parse_model_sort("  SIZE "), "size")
    check("parse_model_sort: params alias", orun.parse_model_sort("params"), "parameters")
    check("parse_model_sort: ctx alias", orun.parse_model_sort("ctx"), "context")
    check("parse_model_sort: recent alias", orun.parse_model_sort("recent"), "modified")

    check("parameter_count: 7.6B",
          orun.model_parameter_count({"details": {"parameter_size": "7.6B"}}), 7.6)
    check("parameter_count: 70B",
          orun.model_parameter_count({"details": {"parameter_size": "70B"}}), 70.0)
    check("parameter_count: 8x7B is the total",
          orun.model_parameter_count({"details": {"parameter_size": "8x7B"}}), 56.0)
    check("parameter_count: 134M",
          orun.model_parameter_count({"details": {"parameter_size": "134M"}}), 0.134)
    check("parameter_count: missing", orun.model_parameter_count({"details": {}}), None)
    check("parameter_count: garbage",
          orun.model_parameter_count({"details": {"parameter_size": "n/a"}}), None)

    models = [
        {"name": "z-small:latest", "size": 100, "modified_at": "2026-01-01",
         "details": {"parameter_size": "1B"}},
        {"name": "a-big:latest", "size": 900, "modified_at": "2025-01-01",
         "details": {"parameter_size": "70B"}},
        {"name": "m-mid:latest", "size": 500, "modified_at": "2026-05-01",
         "details": {}},
    ]

    def names(items):
        return [item["name"] for item in items]

    check("sort by name is A→Z", names(orun.sort_models(models, "name")),
          ["a-big:latest", "m-mid:latest", "z-small:latest"])
    check("sort by size lists the largest first",
          names(orun.sort_models(models, "size")),
          ["a-big:latest", "m-mid:latest", "z-small:latest"])
    check("sort by parameters puts missing last",
          names(orun.sort_models(models, "parameters")),
          ["a-big:latest", "z-small:latest", "m-mid:latest"])
    check("sort by modified is newest first",
          names(orun.sort_models(models, "modified")),
          ["m-mid:latest", "z-small:latest", "a-big:latest"])
    contexts = {"z-small:latest": 131072, "a-big:latest": 8192,
                "m-mid:latest": 32768}
    check("sort by context lists the largest first",
          names(orun.sort_models(models, "context", contexts=contexts)),
          ["z-small:latest", "m-mid:latest", "a-big:latest"])
    check("sort by context without data is deterministic",
          names(orun.sort_models(models, "context")),
          ["a-big:latest", "m-mid:latest", "z-small:latest"])
    check("sort by fuzzy ranks matches first",
          names(orun.sort_models(models, "fuzzy:big")),
          ["a-big:latest", "m-mid:latest", "z-small:latest"])
    check("sort by fuzzy keeps non-matches, alphabetically",
          names(orun.sort_models(models, "fuzzy:z")),
          ["z-small:latest", "a-big:latest", "m-mid:latest"])
    check("sort_models leaves the input alone", names(models),
          ["z-small:latest", "a-big:latest", "m-mid:latest"])

    with orun.console.capture() as cap:
        orun.console.print(orun.model_table(models, sort="context", contexts=contexts))
    text = _plain(cap.get())
    check("model_table names the sort", "by context" in text, True)
    check("model_table shows ctx values", "131k" in text, True)
    check("model_table shows params", "70B" in text, True)
    with orun.console.capture() as cap:
        orun.console.print(orun.model_table(models, sort="fuzzy:big"))
    check("model_table names a fuzzy sort",
          "fuzzy 'big'" in _plain(cap.get()), True)
    return failures


def test_model_picker_sort() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    tags = [
        {"name": "z-small:latest", "size": 100, "modified_at": "2026-01-01",
         "details": {"parameter_size": "1B", "family": "llama"}},
        {"name": "a-big:latest", "size": 900, "modified_at": "2025-01-01",
         "details": {"parameter_size": "70B", "family": "llama"}},
        {"name": "m-mid:latest", "size": 500, "modified_at": "2026-05-01",
         "details": {"parameter_size": "8B", "family": "qwen2"}},
    ]

    client = _TagClient(tags)
    ui = _ScriptedUi(["--sort size", "1"])
    with orun.console.capture() as cap:
        picked = orun.pick_model(client, None, ui=ui, sort="modified")
    text = _plain(cap.get())
    check("picker sorts then picks by number", picked, "a-big:latest")
    check("picker re-renders in the new order",
          text.rindex("a-big") < text.rindex("z-small"), True)
    check("picker names the sort", "by size" in text, True)
    check("picker still shows params", "70B" in text, True)

    seen = {}
    ui = _ScriptedUi(["-s name", "sort size", "q"])
    with orun.console.capture():
        orun.pick_model(client, None, ui=ui, sort="modified",
                        on_sort=lambda key: seen.update(sort=key))
    check("picker accepts -s and 'sort KEY' aliases", seen.get("sort"), "size")

    contexts = {"z-small:latest": 131072, "a-big:latest": 8192,
                "m-mid:latest": 32768}
    client = _TagClient(tags, contexts=contexts)
    orun.interactive._CONTEXT_CACHE.clear()
    ui = _ScriptedUi(["--sort context", "--sort context", "1"])
    with orun.console.capture() as cap:
        picked = orun.pick_model(client, None, ui=ui)
    text = _plain(cap.get())
    check("context sort picks the largest window", picked, "z-small:latest")
    check("context sort fetches each model once", len(client.show_calls), 3)
    check("context column is shown", "131k" in text, True)

    client = _TagClient(tags)
    ui = _ScriptedUi(["--sort mid", "1"])
    with orun.console.capture() as cap:
        picked = orun.pick_model(client, None, ui=ui, sort="size")
    text = _plain(cap.get())
    check("fuzzy sort picks the strongest match", picked, "m-mid:latest")
    check("fuzzy sort names the query", "fuzzy 'mid'" in text, True)

    seen = {}
    ui = _ScriptedUi(["--sort fuzzy qwen", "q"])
    with orun.console.capture():
        orun.pick_model(client, None, ui=ui, sort="modified",
                        on_sort=lambda key: seen.update(sort=key))
    check("picker accepts --sort fuzzy TEXT", seen.get("sort"), "fuzzy:qwen")

    client = _TagClient(tags)
    ui = _ScriptedUi(["--sort nothing-matches", "1"])
    with orun.console.capture() as cap:
        picked = orun.pick_model(client, None, ui=ui, sort="modified")
    text = _plain(cap.get())
    check("fuzzy sort with no match warns", "no model matches" in text, True)
    check("no-match fuzzy keeps every model, alphabetically",
          picked, "a-big:latest")

    ui = _ScriptedUi(["--sort fuzzy", "q"])
    with orun.console.capture() as cap:
        orun.pick_model(client, None, ui=ui, sort="modified")
    check("bare fuzzy shows the usage",
          "needs a search string" in _plain(cap.get()), True)
    return failures


def test_model_command_sort() -> int:
    failures = 0
    import ollamarun_lib.repl.commands.model as model_commands

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    tags = [
        {"name": "z-small:latest", "size": 100, "modified_at": "2026-01-01",
         "details": {"parameter_size": "1B"}},
        {"name": "a-big:latest", "size": 900, "modified_at": "2025-01-01",
         "details": {"parameter_size": "70B"}},
    ]
    repl, _client, _tmp = _test_repl(models=("a-big:latest", "z-small:latest"),
                                     tag_data=tags)

    with orun.console.capture() as cap:
        orun.cmd_model(repl, "--list --sort size")
    text = _plain(cap.get())
    check("/model --list --sort size lists largest first",
          text.index("a-big") < text.index("z-small"), True)
    check("--list keeps the sort for the session", repl.model_sort, "size")

    seen = {}
    saved_pick = model_commands.pick_model

    def fake_pick(client, current=None, *, ui=None, sort="modified", on_sort=None):
        seen["sort"] = sort
        if on_sort:
            on_sort("parameters")
        return None

    model_commands.pick_model = fake_pick
    try:
        with orun.console.capture():
            orun.cmd_model(repl, "")
    finally:
        model_commands.pick_model = saved_pick
    check("picker starts with the session sort", seen.get("sort"), "size")
    check("picker sort sticks for later pickers", repl.model_sort, "parameters")

    check("/models dispatches to cmd_model",
          orun.COMMAND_HANDLERS.get("/models") is orun.cmd_model, True)
    check("/models is advertised", "/model, /models" in orun.COMMAND_INFO, True)

    with orun.console.capture() as cap:
        orun.cmd_model(repl, "--list --sort fuzzy small")
    text = _plain(cap.get())
    check("/model --list --sort fuzzy TEXT ranks the match first",
          text.index("z-small") < text.index("a-big"), True)
    check("--list keeps the fuzzy spec", repl.model_sort, "fuzzy:small")

    with orun.console.capture():
        orun.cmd_model(repl, "--list --sort big")
    check("--list accepts an implicit fuzzy query", repl.model_sort, "fuzzy:big")

    saved_pick = model_commands.pick_model
    model_commands.pick_model = lambda *args, **kwargs: check("bare fuzzy --sort must not pick", False, True)
    try:
        with orun.console.capture() as cap:
            orun.cmd_model(repl, "--sort fuzzy")
        check("bare fuzzy --sort errors",
              "needs a search string" in _plain(cap.get()), True)
    finally:
        model_commands.pick_model = saved_pick

    saved_ensure = model_commands.ensure_model
    model_commands.ensure_model = lambda client, name, config, **kwargs: name
    try:
        with orun.console.capture():
            orun.cmd_model(repl, "--sort name z-small:latest")
    finally:
        model_commands.ensure_model = saved_ensure
    check("--sort with a name sets the sort", repl.model_sort, "name")
    check("--sort with a name still switches", repl.session.model, "z-small:latest")
    return failures


def test_sort_flag() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    parser = orun.build_parser()
    check("--sort parses", parser.parse_args(["--sort", "size"]).sort, "size")
    check("--sort defaults to None", parser.parse_args(["m1"]).sort, None)

    with orun.console.capture() as cap:
        rc = orun.main(["--sort", "fuzzy"])
    text = _plain(cap.get())
    check("main rejects a bare fuzzy --sort", rc, 2)
    check("bare fuzzy --sort prints the usage", "search string" in text, True)

    seen = {}
    saved_pick = orun.cli.pick_model
    saved_terminal = orun.console._force_terminal
    orun.cli.pick_model = lambda client, current=None, **kwargs: seen.update(kwargs) or None
    orun.console._force_terminal = True
    try:
        with orun.console.capture():
            orun.resolve_model(parser.parse_args(["--sort", "context"]),
                               dict(orun.DEFAULT_CONFIG), _FakeClient(), None)
        check("resolve_model forwards --sort", seen.get("sort"), "context")
        seen.clear()
        with orun.console.capture():
            orun.resolve_model(parser.parse_args([]), dict(orun.DEFAULT_CONFIG),
                               _FakeClient(), None)
        check("resolve_model defaults to modified", seen.get("sort"), "modified")
    finally:
        orun.cli.pick_model = saved_pick
        orun.console._force_terminal = saved_terminal

    args = parser.parse_args(["--sort", "ctx", "--list"])
    check("--sort aliases survive parse", args.sort, "ctx")
    return failures


def test_gpu_layers() -> int:
    failures = 0

    def check(name, got, expected):
        nonlocal failures
        if got == expected:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: {got!r} != {expected!r}")
            failures += 1

    check("block_count parsed",
          orun.model_block_count({"model_info": {"llama.block_count": 32}}), 32)
    check("block_count absent", orun.model_block_count({"model_info": {}}), None)

    check("parse absolute", orun.parse_gpu_layers_spec("20"), ("abs", 20))
    check("parse percent", orun.parse_gpu_layers_spec("50%"), ("pct", 50))
    check("parse plus delta", orun.parse_gpu_layers_spec("+4"), ("delta", 4))
    check("parse minus delta", orun.parse_gpu_layers_spec("-4"), ("delta", -4))
    check("parse all", orun.parse_gpu_layers_spec("all"), ("all", None))
    check("parse cpu", orun.parse_gpu_layers_spec("cpu"), ("cpu", None))
    check("parse auto", orun.parse_gpu_layers_spec("auto"), ("auto", None))
    for bad in ("abc", "-50%", "150%", "1.5", ""):
        try:
            orun.parse_gpu_layers_spec(bad)
            print(f"FAIL  parse {bad!r} should raise")
            failures += 1
        except ValueError:
            print(f"PASS  parse {bad!r} raises")

    check("resolve absolute", orun.resolve_gpu_layers("20", 32), 20)
    check("resolve percent rounds", orun.resolve_gpu_layers("50%", 32), 17)
    check("resolve 100% is a full offload", orun.resolve_gpu_layers("100%", 36), 37)
    check("resolve all", orun.resolve_gpu_layers("all", 32), 33)
    check("resolve all without layer count", orun.resolve_gpu_layers("all", None), 999)
    check("resolve cpu", orun.resolve_gpu_layers("cpu", None), 0)
    check("resolve auto", orun.resolve_gpu_layers("auto", 32), None)
    check("resolve delta from current", orun.resolve_gpu_layers("+4", 32, 20), 24)
    check("resolve delta from full", orun.resolve_gpu_layers("-4", 32), 29)
    check("resolve delta clamps at zero", orun.resolve_gpu_layers("-99", 32, 3), 0)
    try:
        orun.resolve_gpu_layers("50%", None)
        print("FAIL  percent without a layer count should raise")
        failures += 1
    except RuntimeError:
        print("PASS  percent without a layer count raises")
    try:
        orun.resolve_gpu_layers("-4", None)
        print("FAIL  delta without a layer count should raise")
        failures += 1
    except RuntimeError:
        print("PASS  delta without a layer count raises")

    check("label percent", orun.gpu_layers_label("50%", 36), "50% → 19 of 37 layers")
    check("label cpu", orun.gpu_layers_label("cpu", 36), "cpu → cpu only")
    check("label auto", orun.gpu_layers_label(None, None), "auto (ollama decides)")
    check("label all without layer count",
          orun.gpu_layers_label("all", None), "all → all layers")

    parser = orun.build_parser()
    check("--gpu-layers parses",
          parser.parse_args(["m1", "--gpu-layers", "50%"]).gpu_layers, "50%")
    check("--gpu-layers defaults off", parser.parse_args(["m1"]).gpu_layers, None)
    try:
        help_text = parser.format_help()
        check("--help formats (percent signs escaped)", "gpu-layers" in help_text, True)
    except (TypeError, ValueError) as exc:
        check("--help formats (percent signs escaped)", f"{exc}", "no error")
    check("/gpu-layers is wired into the REPL command tables",
          orun.COMMAND_HANDLERS.get("/gpu-layers") is orun.cmd_gpu_layers
          and "/gpu-layers" in orun.COMMAND_INFO, True)

    session = orun.Session.create("m1", "http://h:11434", gpu_layers="50%")
    check("session round-trips gpu_layers",
          orun.Session.from_dict(session.to_dict()).gpu_layers, "50%")

    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        client = _FakeClient(models=["m1"],
                             show_data={"model_info": {"llama.block_count": 32}})
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base, gpu_layers="50%"))
        check("payload resolves the percent",
              repl._payload([]).get("options", {}).get("num_gpu"), 17)
        repl.session.parameters["num_gpu"] = 3
        check("dedicated setting beats /set parameter",
              repl._payload([])["options"]["num_gpu"], 17)
        repl.session.gpu_layers = "cpu"
        check("cpu resolves to zero", repl._payload([])["options"]["num_gpu"], 0)
        repl.session.gpu_layers = None
        check("auto leaves an explicit parameter alone",
              repl._payload([])["options"]["num_gpu"], 3)

        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "20")
        check("cmd sets an absolute value", repl.session.gpu_layers, "20")
        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "+4")
        check("cmd normalizes a delta", repl.session.gpu_layers, "24")
        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "-100")
        check("cmd clamps a delta", repl.session.gpu_layers, "0")
        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "auto")
        check("cmd auto clears the spec", repl.session.gpu_layers, None)
        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "")
        check("cmd status shows auto", "auto" in _plain(cap.get()), True)
        with orun.console.capture() as cap:
            orun.cmd_gpu_layers(repl, "bogus")
        check("cmd rejects garbage", "invalid gpu layers" in _plain(cap.get()), True)
        with orun.console.capture() as cap:
            orun.cmd_params(repl, "")
        check("/params shows the gpu layers row", "gpu layers" in _plain(cap.get()), True)

    seen: dict = {}

    def embed_handler(method, url, json, stream):
        seen.update(json or {})
        return _FakeResponse(status=200, payload={"embeddings": [[0.1]]})

    client = orun.OllamaClient("127.0.0.1:11434")
    client.http = _FakeSession(embed_handler)
    client.embed("m1", ["x"], options={"num_gpu": 5})
    check("embed forwards options", seen.get("options"), {"num_gpu": 5})
    return failures


def test_slash_completer() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        config = dict(orun.DEFAULT_CONFIG)
        client = _FakeClient(models=["m1", "m2"])
        repl = orun.Repl(args, config, client, store, orun.Session.create("m1", client.base))
        repl.model_names = ["m1", "m2"]
        completer = orun.SlashCompleter(repl)

        class _Doc:
            def __init__(self, text):
                self.text_before_cursor = text

        names = {c.text for c in completer.get_completions(_Doc("/nu"), None)}
        if names == {"/num-ctx"}:
            print("PASS  SlashCompleter completes slash commands")
        else:
            print(f"FAIL  SlashCompleter commands: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/model m"), None)}
        if names == {"m1", "m2"}:
            print("PASS  SlashCompleter completes model names after /model")
        else:
            print(f"FAIL  SlashCompleter models: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/models m"), None)}
        if names == {"m1", "m2"}:
            print("PASS  SlashCompleter completes model names after /models")
        else:
            print(f"FAIL  SlashCompleter /models: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/model -"), None)}
        if names == {"--list", "--sort"}:
            print("PASS  SlashCompleter completes /model flags")
        else:
            print(f"FAIL  SlashCompleter /model flags: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/model --sort s"), None)}
        if names == {"size"}:
            print("PASS  SlashCompleter completes sort keys")
        else:
            print(f"FAIL  SlashCompleter sort keys: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/model --sort "), None)}
        if names == set(orun.MODEL_SORT_KEYS) | {"fuzzy"}:
            print("PASS  SlashCompleter offers every sort key and fuzzy")
        else:
            print(f"FAIL  SlashCompleter all sort keys: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/model --sort f"), None)}
        if names == {"fuzzy"}:
            print("PASS  SlashCompleter completes the fuzzy keyword")
        else:
            print(f"FAIL  SlashCompleter fuzzy keyword: {names}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools "), None)}
        if {"web_search", "run_shell", "policy", "help"} <= names and "call" not in names:
            print("PASS  /tools completes subcommands and tool names")
        else:
            print(f"FAIL  /tools first token: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools we"), None)}
        if "web_search" in names and "web" in names:
            print("PASS  /tools completes with tool aliases")
        else:
            print(f"FAIL  /tools alias completion: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools web_search "), None)}
        if names == {"on", "off", "help", "call"}:
            print("PASS  /tools NAME completes its actions")
        else:
            print(f"FAIL  /tools NAME actions: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools search o"), None)}
        if names == {"on", "off"}:
            print("PASS  /tools ALIAS completes its actions")
        else:
            print(f"FAIL  /tools ALIAS actions: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools enable we"), None)}
        if "web_search" in names:
            print("PASS  /tools enable completes tool names")
        else:
            print(f"FAIL  /tools enable: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools policy "), None)}
        if names == {"auto", "allowlist", "ask"}:
            print("PASS  /tools policy completes the modes")
        else:
            print(f"FAIL  /tools policy: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools rules "), None)}
        if names == {"add", "rm"}:
            print("PASS  /tools rules completes its subcommands")
        else:
            print(f"FAIL  /tools rules: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools rules add "), None)}
        if names == {"shell", "path", "python", "call"}:
            print("PASS  /tools rules add completes the rule kinds")
        else:
            print(f"FAIL  /tools rules add: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools help re"), None)}
        if "read_file" in names and "read_url" in names:
            print("PASS  /tools help completes tool names and aliases")
        else:
            print(f"FAIL  /tools help: {sorted(names)}")
            failures += 1

        names = {c.text for c in completer.get_completions(_Doc("/tools web_search call "), None)}
        if names == set():
            print("PASS  /tools NAME call stops completing (free-form arguments)")
        else:
            print(f"FAIL  /tools call arguments: {sorted(names)}")
            failures += 1
    return failures


def test_prompt_toolkit_fallback() -> int:
    """Importing without prompt_toolkit must degrade, not crash.

    `sys.modules['prompt_toolkit'] = None` makes any import of it raise, so
    a subprocess pins the fallback path: `constants.py` must bind stand-in
    classes (a `None` binding once crashed `repl.py`'s module-level
    `SlashCompleter(Completer)` definition with `TypeError: NoneType takes
    no arguments`, which killed every command, `--version` included).
    """
    import subprocess
    failures = 0
    repo = Path(__file__).parent

    code = (
        "import sys\n"
        "sys.modules['prompt_toolkit'] = None\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "import ollamarun_lib as orun\n"
        "assert orun.HAS_PROMPT_TOOLKIT is False\n"
        "assert isinstance(orun.SlashCompleter, type)\n"
        "print(orun.VERSION)\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() == orun.VERSION:
        print("PASS  package imports with prompt_toolkit unavailable")
    else:
        print(f"FAIL  fallback import: rc={result.returncode} "
              f"stderr={result.stderr.strip().splitlines()[-1:]}")
        failures += 1

    code = (
        "import sys, runpy\n"
        "sys.modules['prompt_toolkit'] = None\n"
        "sys.argv = ['ollamarun', '--version']\n"
        f"runpy.run_path({str(repo / 'ollamarun')!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0 and orun.VERSION in result.stdout:
        print("PASS  CLI starts without prompt_toolkit")
    else:
        print(f"FAIL  fallback --version: rc={result.returncode} "
              f"stderr={result.stderr.strip().splitlines()[-1:]}")
        failures += 1
    return failures


def test_textual_optional() -> int:
    """Importing without textual must degrade, not crash.

    `tui.py` imports the Textual classes at module level, so the package
    re-export is conditional on `HAS_TEXTUAL`; without that guard a machine
    with no `textual` package could not run any command (`--version`
    included), even though the inline REPL is the documented fallback.
    """
    import subprocess
    failures = 0
    repo = Path(__file__).parent

    code = (
        "import sys\n"
        "sys.modules['textual'] = None\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "import ollamarun_lib as orun\n"
        "assert orun.HAS_TEXTUAL is False\n"
        "assert not hasattr(orun, 'OllamarunTui')\n"
        "assert callable(orun.main)\n"
        "print(orun.VERSION)\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip() == orun.VERSION:
        print("PASS  package imports with textual unavailable")
    else:
        print(f"FAIL  textual fallback import: rc={result.returncode} "
              f"stderr={result.stderr.strip().splitlines()[-1:]}")
        failures += 1

    code = (
        "import sys, runpy\n"
        "sys.modules['textual'] = None\n"
        "sys.argv = ['ollamarun', '--version']\n"
        f"runpy.run_path({str(repo / 'ollamarun')!r}, run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0 and orun.VERSION in result.stdout:
        print("PASS  CLI starts without textual")
    else:
        print(f"FAIL  textual fallback --version: rc={result.returncode} "
              f"stderr={result.stderr.strip().splitlines()[-1:]}")
        failures += 1
    return failures


class _FakeResponse:
    def __init__(self, status=200, payload=None, text="", lines=None):
        self.status_code = status
        self._payload = payload
        self.text = text
        self._lines = lines or []

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def close(self):
        pass

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


class _FakeSession:
    def __init__(self, handler):
        self.handler = handler

    def request(self, method, url, *, json=None, stream=False, timeout=None, verify=None):
        return self.handler(method, url, json, stream)


def _check(failures, name, cond, detail=""):
    if cond:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name} {detail}")
        return failures + 1
    return failures


def _test_repl(resident=(), models=("m1",), args_extra=(), config_extra=None,
               tag_data=None):
    """A throwaway Repl wired to `_FakeClient` and a temp session store."""
    tmp = tempfile.mkdtemp()
    store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
    args = orun.build_parser().parse_args(
        ["m1", "--render", "plain", "--no-save", *args_extra])
    config = dict(orun.DEFAULT_CONFIG)
    config.update(config_extra or {})
    client = _FakeClient(resident=resident, models=models,
                         show_data={"model_info": {"llama.context_length": 8192}},
                         tag_data=tag_data)
    session = orun.Session.create("m1", client.base)
    repl = orun.Repl(args, config, client, store, session)
    repl._preload_model = lambda: None
    return repl, client, tmp


def test_tools_policy() -> int:
    failures = 0
    safe_shell = [
        "ls -la", "cat README.md", "head -n 5 file.txt", "tail file.txt",
        "grep -rn TODO .", "rg --files", "git log --oneline -5", "git status",
        "git diff --stat", "git config --get user.name", "find . -name '*.py'",
        "ls | grep foo", "ls && pwd", "sort file.txt", "du -sh .", "df -h",
        "tree -L 2", "date", "which python3", "uname -a", "cut -d: -f1 /etc/passwd",
    ]
    risky_shell = [
        "rm -rf /", "ls; rm x", "ls && rm x", "cat $(pwd)/secret",
        "cat `pwd`/secret", "echo hi > out.txt", "echo hi >> out.txt",
        "python3 -c 'print(1)'", "bash -c 'rm x'", "sh -c 'x'", "sudo ls",
        "git -c core.pager=less log", "git config user.name admin",
        "git branch -D main", "git remote add origin x", "find . -exec rm {} ;",
        "find . -delete", "find . -fprint out", "rg --pre 'sh -c' needle",
        "sort -o out.txt in.txt", "curl http://example.com", "wget x",
        "FOO=1 ls", "ls $(echo hi)", "xargs rm", "env ls", "nice ls",
        "diff <(ls) <(ls)", "ls > log",
    ]
    for command in safe_shell:
        reason = orun.shell_safety(command)
        failures = _check(failures, f"shell safe: {command}", reason == "",
                          f"reason={reason!r}")
    for command in risky_shell:
        reason = orun.shell_safety(command)
        failures = _check(failures, f"shell needs approval: {command}",
                          reason != "", "unexpectedly allowed")

    safe_python = [
        "print(2 + 2)", "import math\nprint(math.pi)",
        "from collections import Counter\nprint(Counter('aab'))",
        "print(len('x' * 10))",
        "import json\nprint(json.dumps({'a': 1}))",
    ]
    risky_python = [
        "import os\nos.system('ls')", "open('/etc/passwd').read()",
        "__import__('os')", "exec('print(1)')", "eval('1+1')",
        "from subprocess import run", "import pathlib\n"
        "pathlib.Path('x').write_text('y')", "import socket",
        "f = open\nf('/etc/passwd')", "print(open)",
        "from math import *", "import sys\nsys.exit()",
    ]
    for code in safe_python:
        reason = orun.python_safety(code)
        failures = _check(failures, f"python safe: {code.splitlines()[0]}",
                          reason == "", f"reason={reason!r}")
    for code in risky_python:
        reason = orun.python_safety(code)
        failures = _check(failures, f"python needs approval: {code.splitlines()[0]}",
                          reason != "", "unexpectedly allowed")

    registry = orun.default_registry(tts=False)
    read_only = registry["read_file"]
    shell = registry["run_shell"]
    write = registry["write_file"]
    decisions = [
        ("read-only always auto",
         orun.approval_decision(read_only, {}, policy="ask").allowed, True),
        ("safe shell auto under allowlist",
         orun.approval_decision(shell, {"command": "ls -la"}, policy="allowlist").allowed, True),
        ("risky shell asks under allowlist",
         orun.approval_decision(shell, {"command": "rm -rf /"}, policy="allowlist").allowed, False),
        ("safe shell asks under policy=ask",
         orun.approval_decision(shell, {"command": "ls"}, policy="ask").allowed, False),
        ("write asks under allowlist",
         orun.approval_decision(write, {"path": "x", "content": "y"},
                                policy="allowlist").allowed, False),
        ("write auto under auto",
         orun.approval_decision(write, {"path": "x", "content": "y"},
                                policy="auto").allowed, True),
    ]
    for name, got, expected in decisions:
        failures = _check(failures, name, got is expected, f"got={got}")

    normalized = [
        ({"command": "ls"}, {"command": "ls"}),
        ('{"command": "ls"}', {"command": "ls"}),
        (None, {}),
        ("", {}),
    ]
    for raw, expected in normalized:
        got = orun.normalize_arguments(raw)
        failures = _check(failures, f"normalize_arguments({raw!r})", got == expected,
                          f"got={got!r}")
    failures = _check(failures, "normalize_arguments rejects junk",
                      orun.normalize_arguments("{not json") is None)
    failures = _check(failures, "normalize_arguments rejects non-objects",
                      orun.normalize_arguments("[1, 2]") is None)
    failures = _check(failures, "fingerprint is stable",
                      orun.call_fingerprint("run_shell", {"a": 1})
                      == orun.call_fingerprint("run_shell", {"a": 1}))
    failures = _check(failures, "call_detail summarizes shell",
                      orun.call_detail("run_shell", {"command": "ls -la"}) == "ls -la")
    return failures


def test_tool_execution() -> int:
    failures = 0
    registry = orun.default_registry(tts=False)
    ctx = orun.ToolContext(cwd=os.getcwd(), timeout=30, max_output=2000)

    outcome = orun.execute_tool(registry["run_shell"], {"command": "echo from-shell"}, ctx)
    failures = _check(failures, "run_shell echoes and reports exit 0",
                      outcome.ok and outcome.text.startswith("exit 0")
                      and "from-shell" in outcome.text, outcome.text)

    outcome = orun.execute_tool(registry["run_shell"], {"command": "exit 3"}, ctx)
    failures = _check(failures, "run_shell reports a failing exit code",
                      outcome.ok and outcome.text.startswith("exit 3"), outcome.text)

    outcome = orun.execute_tool(registry["run_python"], {"code": "print(6 * 7)"}, ctx)
    failures = _check(failures, "run_python prints", outcome.ok and "42" in outcome.text,
                      outcome.text)

    outcome = orun.execute_tool(registry["run_python"], {"code": "raise ValueError('boom')"},
                                ctx)
    failures = _check(failures, "run_python captures tracebacks",
                      "boom" in outcome.text and "exit 1" in outcome.text, outcome.text)

    tiny = orun.ToolContext(cwd=os.getcwd(), timeout=30, max_output=200)
    outcome = orun.execute_tool(registry["run_python"], {"code": "print('x' * 2000)"}, tiny)
    failures = _check(failures, "output is truncated to the cap",
                      "truncated" in outcome.text and len(outcome.text) < 1000,
                      f"len={len(outcome.text)}")

    outcome = orun.execute_tool(registry["run_shell"], {"command": "printf 'raw\\n'"}, ctx)
    failures = _check(failures, "executor runs whatever it is given (policy is separate)",
                      outcome.ok and outcome.text.startswith("exit 0"), outcome.text)

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "notes.txt"
        write = orun.execute_tool(registry["write_file"],
                                  {"path": str(target), "content": "hello\nthere"}, ctx)
        failures = _check(failures, "write_file writes", target.read_text() == "hello\nthere",
                          write.text)
        append = orun.execute_tool(registry["write_file"],
                                   {"path": str(target), "content": "\nmore", "append": True},
                                   ctx)
        failures = _check(failures, "write_file appends",
                          target.read_text().endswith("more"), append.text)
        read = orun.execute_tool(registry["read_file"], {"path": str(target)}, ctx)
        failures = _check(failures, "read_file numbers lines",
                          "1: hello" in read.text and "3: more" in read.text, read.text)
        listing = orun.execute_tool(registry["list_dir"], {"path": tmp}, ctx)
        failures = _check(failures, "list_dir lists entries",
                          "notes.txt" in listing.text, listing.text)
        missing = orun.execute_tool(registry["read_file"], {"path": str(Path(tmp) / "nope")},
                                    ctx)
        failures = _check(failures, "read_file reports missing files",
                          not missing.ok and "not a file" in missing.text, missing.text)

    bad = orun.execute_tool(registry["run_shell"], {"command": ""}, ctx)
    failures = _check(failures, "missing command is an error", not bad.ok, bad.text)
    return failures


def test_file_tools() -> int:
    failures = 0
    registry = orun.default_registry(tts=False)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = orun.ToolContext(cwd=tmp, timeout=30, max_output=4000)
        source = Path(tmp) / "notes.txt"
        source.write_text("alpha\nbeta\ngamma\n")

        # edit_file: unique replacement, in the result and on disk
        outcome = orun.execute_tool(
            registry["edit_file"],
            {"path": str(source), "old_string": "beta", "new_string": "BETA"}, ctx)
        failures = _check(failures, "edit_file replaces a unique fragment",
                          outcome.ok and "-beta" in outcome.text
                          and "+BETA" in outcome.text
                          and source.read_text() == "alpha\nBETA\ngamma\n", outcome.text)

        # edit_file refuses a missing or ambiguous fragment
        miss = orun.execute_tool(
            registry["edit_file"],
            {"path": str(source), "old_string": "nope", "new_string": "x"}, ctx)
        failures = _check(failures, "edit_file reports a missing fragment",
                          not miss.ok and "not found" in miss.text, miss.text)
        source.write_text("x\nx\n")
        ambiguous = orun.execute_tool(
            registry["edit_file"],
            {"path": str(source), "old_string": "x", "new_string": "y"}, ctx)
        failures = _check(failures, "edit_file refuses an ambiguous fragment",
                          not ambiguous.ok and "2 times" in ambiguous.text, ambiguous.text)
        everything = orun.execute_tool(
            registry["edit_file"],
            {"path": str(source), "old_string": "x", "new_string": "y",
             "replace_all": True}, ctx)
        failures = _check(failures, "edit_file replace_all replaces every match",
                          everything.ok and source.read_text() == "y\ny\n",
                          everything.text)

        # edit_file leaves no temp file behind and keeps the mode
        leftovers = [entry.name for entry in Path(tmp).iterdir()
                     if entry.name.endswith(".tmp")]
        failures = _check(failures, "edit_file leaves no temp file behind",
                          leftovers == [], str(leftovers))

        # the approval/call panel renders the diff for edit_file
        with orun.console.capture() as cap:
            orun.console.print(orun.tool_call_panel(
                "edit_file", {"path": "a.py", "old_string": "old line",
                              "new_string": "new line"}))
        panel = _plain(cap.get())
        failures = _check(failures, "the edit call panel previews the diff",
                          "-old line" in panel and "+new line" in panel
                          and "old_string" not in panel, panel)
        failures = _check(failures, "call_detail summarizes an edit",
                          orun.call_detail("edit_file", {"path": "a.py",
                                                         "old_string": "a\nb",
                                                         "new_string": "c"})
                          == "a.py (2 → 1 lines)",
                          orun.call_detail("edit_file", {"path": "a.py",
                                                         "old_string": "a\nb",
                                                         "new_string": "c"}))

        # make_dir creates parents and tolerates an existing directory
        nested = Path(tmp) / "one" / "two"
        made = orun.execute_tool(registry["make_dir"], {"path": str(nested)}, ctx)
        again = orun.execute_tool(registry["make_dir"], {"path": str(nested)}, ctx)
        failures = _check(failures, "make_dir creates parent directories",
                          made.ok and nested.is_dir() and "already exists" in again.text,
                          f"{made.text} / {again.text}")

        # move/copy refuse an existing destination unless overwrite=true
        target = Path(tmp) / "moved.txt"
        moved = orun.execute_tool(registry["move_file"],
                                  {"source": str(source), "destination": str(target)}, ctx)
        refused = orun.execute_tool(registry["move_file"],
                                    {"source": str(target), "destination": str(target)}, ctx)
        failures = _check(failures, "move_file moves and refuses an existing destination",
                          moved.ok and target.is_file() and not source.exists()
                          and not refused.ok and "destination exists" in refused.text,
                          f"{moved.text} / {refused.text}")
        copied = orun.execute_tool(registry["copy_file"],
                                   {"source": str(target),
                                    "destination": str(Path(tmp) / "copy.txt")}, ctx)
        forced = orun.execute_tool(registry["copy_file"],
                                   {"source": str(target),
                                    "destination": str(Path(tmp) / "copy.txt"),
                                    "overwrite": True}, ctx)
        failures = _check(failures, "copy_file copies and overwrite=true is accepted",
                          copied.ok and forced.ok, f"{copied.text} / {forced.text}")

        # delete_file goes through trash-put (a fake one on PATH)
        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        trash = bindir / "trash-put"
        trash.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$1\" > {tmp}/trashed.txt\n")
        trash.chmod(0o755)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bindir}{os.pathsep}{old_path}"
        try:
            outcome = orun.execute_tool(registry["delete_file"],
                                        {"path": str(target)}, ctx)
        finally:
            os.environ["PATH"] = old_path
        failures = _check(failures, "delete_file moves the path to the trash",
                          outcome.ok and "moved to trash" in outcome.text
                          and (Path(tmp) / "trashed.txt").read_text().strip() == str(target),
                          outcome.text)

        # path rules cover the new file tools
        edit_spec = registry["edit_file"]
        failures = _check(failures, "edit_file is a path-rule tool",
                          edit_spec.name in orun.PATH_RULE_TOOLS
                          and "make_dir" in orun.PATH_RULE_TOOLS
                          and "delete_file" in orun.PATH_RULE_TOOLS)
        suggestion = orun.suggest_rules(
            "edit_file", {"path": str(Path(tmp) / "x.py")}, cwd=tmp)
        failures = _check(failures, "edit_file suggests a path rule",
                          any(rule.kind == "path" for rule in suggestion),
                          str(suggestion))
    failures = _check(failures, "file tools need confirmation under allowlist",
                      not orun.approval_decision(
                          registry["edit_file"], {"path": "x", "old_string": "a",
                                                  "new_string": "b"},
                          policy="allowlist").allowed)
    return failures


def test_session_model_tools() -> int:
    failures = 0
    registry = orun.default_registry(tts=False)

    # search_sessions reads through the store the REPL injects
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        first = orun.Session.create("m1", "http://fake:11434",
                                    title="jupyter kernels")
        first.add_message("user", "fix the jupyter kernel for the llm-jukebox venv")
        first.add_message("assistant", "reinstalled ipykernel inside the venv")
        store.save(first)
        second = orun.Session.create("m2", "http://fake:11434", title="gardening")
        second.add_message("user", "when to water tomatoes")
        store.save(second)
        ctx = orun.ToolContext(cwd=tmp, timeout=10, max_output=4000, store=store)
        outcome = orun.execute_tool(registry["search_sessions"],
                                    {"query": "ipykernel"}, ctx)
        failures = _check(failures, "search_sessions finds a matching message",
                          outcome.ok and first.id in outcome.text
                          and "jupyter kernels" in outcome.text
                          and "reinstalled ipykernel" in outcome.text, outcome.text)
        empty = orun.execute_tool(registry["search_sessions"],
                                  {"query": "zzz-nothing"}, ctx)
        failures = _check(failures, "search_sessions reports no matches",
                          empty.ok and "no saved session" in empty.text, empty.text)
        bare = orun.ToolContext(cwd=tmp, timeout=10, max_output=4000)
        none = orun.execute_tool(registry["search_sessions"], {"query": "x"}, bare)
        failures = _check(failures, "search_sessions errors without a store",
                          not none.ok and "unavailable" in none.text, none.text)

    # list_models marks residents from /api/ps
    client = _FakeClient(
        tag_data=[
            {"name": "m1", "size": 4 * 1024**3,
             "modified_at": "2026-10-01T10:00:00-03:00",
             "details": {"parameter_size": "8.0B", "quantization_level": "Q4_K_M"}},
            {"name": "m2", "size": 1 * 1024**3,
             "modified_at": "2026-09-01T10:00:00-03:00",
             "details": {"parameter_size": "1.0B", "quantization_level": "Q8_0"}},
        ],
        ps_entries=[{"name": "m1", "size": 4 * 1024**3, "size_vram": 3 * 1024**3,
                     "expires_at": "2026-10-16T23:00:00-03:00"}])
    ctx = orun.ToolContext(cwd=".", timeout=10, max_output=4000, client=client)
    outcome = orun.execute_tool(registry["list_models"], {}, ctx)
    failures = _check(failures, "list_models lists installed models",
                      outcome.ok and "m1" in outcome.text and "m2" in outcome.text
                      and "8.0B" in outcome.text and "Q4_K_M" in outcome.text,
                      outcome.text)
    failures = _check(failures, "list_models marks the loaded model",
                      "loaded (3.0 GiB VRAM of 4.0 GiB, unloads 2026-10-16 23:00:00)"
                      in outcome.text, outcome.text)
    loaded = orun.execute_tool(registry["list_models"], {"loaded_only": True}, ctx)
    failures = _check(failures, "list_models loaded_only hides installed-only models",
                      loaded.ok and "m1" in loaded.text and "m2" not in loaded.text,
                      loaded.text)

    class _DeadClient(_FakeClient):
        def tags(self):
            raise orun.OllamaError("cannot reach Ollama at http://fake:11434: boom")

    dead = orun.ToolContext(cwd=".", timeout=10, max_output=4000,
                            client=_DeadClient())
    outcome = orun.execute_tool(registry["list_models"], {}, dead)
    failures = _check(failures, "list_models surfaces server errors",
                      not outcome.ok and "cannot reach Ollama" in outcome.text,
                      outcome.text)
    bare = orun.ToolContext(cwd=".", timeout=10, max_output=4000)
    outcome = orun.execute_tool(registry["list_models"], {}, bare)
    failures = _check(failures, "list_models errors without a client",
                      not outcome.ok and "connection" in outcome.text, outcome.text)

    failures = _check(failures, "recall and ps aliases resolve onto the new tools",
                      orun.resolve_tool_alias("recall") == "search_sessions"
                      and orun.resolve_tool_alias("ps") == "list_models"
                      and orun.resolve_tool_alias("models") == "list_models")
    return failures


def test_system_status_tool() -> int:
    failures = 0
    registry = orun.default_registry(tts=False)
    from ollamarun_lib.tools import system_status as tools_mod
    probe_names = ("local_ram_probe", "local_gpu_probe", "local_free_disk",
                   "ssh_ram_probe", "ssh_gpu_probe", "ssh_free_disk")
    saved = {name: getattr(tools_mod, name) for name in probe_names}
    ssh_calls = []
    try:
        tools_mod.local_ram_probe = lambda: (16 * 1024**3, 8 * 1024**3)
        tools_mod.local_gpu_probe = lambda: ([(12288, 6144)], {})
        tools_mod.local_free_disk = lambda: (500 * 1024**3, 1000 * 1024**3, "/home")

        def fake_ssh_ram(target, *, user=None, **kw):
            ssh_calls.append(("ram", target, user))
            return (32 * 1024**3, 4 * 1024**3)

        def fake_ssh_gpu(target, *, user=None, **kw):
            ssh_calls.append(("gpu", target, user))
            return ([(24576, 1024)], {})

        def fake_ssh_disk(target, *, user=None, **kw):
            ssh_calls.append(("disk", target, user))
            return (100 * 1024**3, 1000 * 1024**3, "/mnt")

        tools_mod.ssh_ram_probe = fake_ssh_ram
        tools_mod.ssh_gpu_probe = fake_ssh_gpu
        tools_mod.ssh_free_disk = fake_ssh_disk

        client = _FakeClient()
        client.base = "http://127.0.0.1:11434"
        ctx = orun.ToolContext(cwd=".", timeout=10, max_output=4000, client=client)
        outcome = orun.execute_tool(registry["system_status"], {}, ctx)
        failures = _check(failures, "system_status reports the local ollama host",
                          outcome.ok
                          and "this machine (ollama at http://127.0.0.1:11434)" in outcome.text
                          and "RAM: 8.0 GiB free of 16.0 GiB" in outcome.text
                          and "GPU 0: 6.0 GiB free of 12.0 GiB" in outcome.text
                          and "disk (models): 500.0 GiB free of 1000.0 GiB at /home"
                          in outcome.text,
                          outcome.text)

        remote = _FakeClient()
        remote.base = "http://10.9.9.9:11434"
        ctx = orun.ToolContext(cwd=".", timeout=10, max_output=4000, client=remote,
                               ssh_target="kiddo", ssh_user="vinibretas")
        outcome = orun.execute_tool(registry["system_status"], {}, ctx)
        failures = _check(failures, "system_status ssh-probes a remote ollama host",
                          outcome.ok
                          and "kiddo (ollama at http://10.9.9.9:11434)" in outcome.text
                          and "RAM: 4.0 GiB free of 32.0 GiB" in outcome.text
                          and "GPU 0: 1.0 GiB free of 24.0 GiB" in outcome.text
                          and "disk (models): 100.0 GiB free of 1000.0 GiB at /mnt"
                          in outcome.text
                          and ssh_calls == [("ram", "kiddo", "vinibretas"),
                                            ("gpu", "kiddo", "vinibretas"),
                                            ("disk", "kiddo", "vinibretas")],
                          f"{outcome.text} / {ssh_calls}")

        outcome = orun.execute_tool(registry["system_status"],
                                    {"target": "local"}, ctx)
        failures = _check(failures, "system_status target=local stays on this machine",
                          outcome.ok and "this machine" in outcome.text,
                          outcome.text)

        no_ssh = orun.ToolContext(cwd=".", timeout=10, max_output=4000, client=remote)
        outcome = orun.execute_tool(registry["system_status"], {}, no_ssh)
        failures = _check(failures, "system_status explains a remote host without --remote",
                          not outcome.ok and "--remote" in outcome.text, outcome.text)

        outcome = orun.execute_tool(registry["system_status"],
                                    {"target": "elsewhere"}, ctx)
        failures = _check(failures, "system_status rejects a bad target",
                          not outcome.ok and "'ollama' or 'local'" in outcome.text,
                          outcome.text)
    finally:
        for name, probe in saved.items():
            setattr(tools_mod, name, probe)
    return failures


def _fake_chem(directory: str) -> Path:
    """A stand-in chemInventoryAPI answering the read commands with canned JSON."""
    script = Path(directory) / "chemInventoryAPI"
    script.write_text(
        "#! /usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['search']:\n"
        "    if len(args) > 1 and args[1] == 'boom':\n"
        "        print('error: boom', file=sys.stderr)\n"
        "        sys.exit(2)\n"
        "    payload = {'containers': [\n"
        "        {'id': 13368670, 'name': 'Acetone', 'cas': '67-64-1', 'size': '500',\n"
        "         'unit': 'mL', 'locationtext': 'Shelf 2', 'barcode': 'B1',\n"
        "         'supplier': 'Sigma'},\n"
        "        {'id': 7, 'name': 'Acetone-d6', 'cas': '666-52-4', 'size': '10',\n"
        "         'unit': 'g', 'locationtext': 'Cabinet A'}],\n"
        "        'orders': [{'id': 3}]}\n"
        "elif args[:2] == ['container', 'info']:\n"
        "    payload = {'id': int(args[2]), 'containername': 'Acetone', 'cas': '67-64-1',\n"
        "               'locationString': 'Shelf 2', 'size': '500', 'unit': 'mL',\n"
        "               'supplier': 'Sigma', 'molecularformula': 'C3H6O',\n"
        "               'molecularweight': '58.08', 'cf-1': '2026-12-31',\n"
        "               'ghs': [{'hazardstatements': 'H225|H319', 'source': 'Sigma'}]}\n"
        "elif args[:2] == ['location', 'list']:\n"
        "    payload = [{'id': 1, 'name': 'Room A', 'numbercontainers': 3, 'parent': 0},\n"
        "               {'id': 2, 'name': 'Shelf 2', 'numbercontainers': 1, 'parent': 1}]\n"
        "elif args[:2] == ['location', 'containers']:\n"
        "    payload = {'containers': [{'id': 5, 'name': 'NaOH', 'cas': '1310-73-2',\n"
        "                               'size': '1', 'unit': 'kg',\n"
        "                               'locationtext': 'Shelf 2'}], 'orders': []}\n"
        "else:\n"
        "    print('error: unknown command', file=sys.stderr)\n"
        "    sys.exit(2)\n"
        "print(json.dumps(payload))\n")
    script.chmod(0o755)
    return script


def test_chem_tools() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        fake = _fake_chem(tmp)
        registry = orun.default_registry(tts=False, chem=True)
        ctx = orun.ToolContext(cwd=tmp, timeout=10, max_output=4000,
                               chem_inventory=str(fake))
        outcome = orun.execute_tool(registry["chem_search"], {"query": "acetone"}, ctx)
        failures = _check(failures, "chem_search lists matching containers",
                          outcome.ok and "#13368670" in outcome.text
                          and "Acetone" in outcome.text and "CAS 67-64-1" in outcome.text
                          and "500 mL" in outcome.text and "at Shelf 2" in outcome.text
                          and "1 matching order request(s)" in outcome.text,
                          outcome.text)
        outcome = orun.execute_tool(registry["chem_container"], {"id": 13368670}, ctx)
        failures = _check(failures, "chem_container formats the full record",
                          outcome.ok and "container #13368670" in outcome.text
                          and "formula: C3H6O" in outcome.text
                          and "GHS: H225, H319" in outcome.text
                          and "custom fields: 1=2026-12-31" in outcome.text,
                          outcome.text)
        outcome = orun.execute_tool(registry["chem_locations"], {}, ctx)
        failures = _check(failures, "chem_locations renders the location tree",
                          outcome.ok and "Room A (#1, 3)" in outcome.text
                          and "  Shelf 2 (#2, 1)" in outcome.text, outcome.text)
        outcome = orun.execute_tool(registry["chem_locations"],
                                    {"location": "Shelf 2"}, ctx)
        failures = _check(failures, "chem_locations lists one location's containers",
                          outcome.ok and "NaOH" in outcome.text
                          and "1310-73-2" in outcome.text, outcome.text)
        outcome = orun.execute_tool(registry["chem_search"], {"query": "boom"}, ctx)
        failures = _check(failures, "chem_search surfaces CLI errors",
                          not outcome.ok and "error: boom" in outcome.text, outcome.text)
        missing = orun.execute_tool(registry["chem_container"], {}, ctx)
        failures = _check(failures, "chem_container requires an id",
                          not missing.ok and "missing 'id'" in missing.text, missing.text)
        failures = _check(failures, "the chem tools run unattended (read-only)",
                          orun.approval_decision(registry["chem_search"], {"query": "x"},
                                                 policy="allowlist").allowed
                          and orun.approval_decision(registry["chem_locations"], {},
                                                     policy="allowlist").allowed)
        failures = _check(failures, "find_chem_inventory honors the configured path",
                          orun.find_chem_inventory(ctx) == str(fake))
        failures = _check(failures, "the chem alias resolves onto chem_search",
                          orun.resolve_tool_alias("inventory_search") == "chem_search")
    return failures


_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 52 >> stream
BT /F1 12 Tf 10 100 Td (Hello PDF world) Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
trailer << /Root 1 0 R >>
%%EOF
"""

_BLANK_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj
trailer << /Root 1 0 R >>
%%EOF
"""


def test_web_tools() -> int:
    failures = 0
    from ollamarun_lib.tools import fetch_url as fetch_tools
    from ollamarun_lib.tools import net as net_tools
    from ollamarun_lib.tools import web_search as search_tools

    # ── HTML turned into readable text
    html = ("<html><head><title>Widget &amp; Co</title>"
            "<style>body{color:red}</style><script>var x=1;</script></head>"
            "<body><nav><a href='/nav'>Nav link</a></nav>"
            "<main><h1>Widgets</h1><p>Hello&nbsp;world.</p>"
            "<ul><li>One</li><li>Two</li></ul>"
            "<a href='/docs/one'>Docs</a></main>"
            "<footer>Junk footer</footer></body></html>")
    title, text, links = orun.extract_html_text(html, "https://example.com/base/page")
    failures = _check(failures, "HTML title is decoded",
                      title == "Widget & Co", title)
    failures = _check(failures, "HTML keeps headings, lists and text",
                      "# Widgets" in text and "- One" in text
                      and "Hello world." in text, text)
    failures = _check(failures, "HTML drops scripts, styles and chrome",
                      "color:red" not in text and "var x=1" not in text
                      and "Nav link" not in text and "Junk footer" not in text
                      and "Widget & Co" not in text, text)
    failures = _check(failures, "HTML links are absolute and deduplicated",
                      links == [("Docs", "https://example.com/docs/one")], str(links))

    _, _, sketchy = orun.extract_html_text(
        "<div><a href='javascript:x'>js</a><a href='#frag'>frag</a>"
        "<a href='mailto:a@b.test'>mail</a><a href='https://other.test/a'>A</a>"
        "<a href='https://other.test/a'>A again</a></div>", "https://example.com/")
    failures = _check(failures, "link schemes/fragments/duplicates are filtered",
                      sketchy == [("A", "https://other.test/a")], str(sketchy))

    page = ("<body><div>nav " * 10 + "</div>" +
            "<article>" + "body text " * 50 + "</article></body>")
    _, article_text, _ = orun.extract_html_text(page, "https://x.test/")
    failures = _check(failures, "the article region wins over page chrome",
                      "body text" in article_text and "nav nav" not in article_text,
                      article_text[:80])

    # ── fetch responses (pure formatter)
    not_found = orun.format_fetch_response(
        url="https://x.test/missing", final_url="https://x.test/missing", status=404,
        content_type="text/html", body="<html><h1>Not Found</h1></html>", size=34)
    failures = _check(failures, "404 explains itself and points at web_search",
                      "HTTP 404" in not_found and "does not exist" in not_found
                      and "web_search" in not_found and "search again" in not_found
                      and "<h1>" not in not_found,
                      not_found)
    blocked = orun.format_fetch_response(
        url="https://x.test/", final_url="https://x.test/", status=403,
        content_type="text/html", body="<title>Just a moment...</title>", size=31)
    failures = _check(failures, "Cloudflare 403 is named as a bot wall",
                      "bot-protection" in blocked and "Do not retry" in blocked
                      and "Just a moment" not in blocked, blocked)
    wiki_404 = orun.format_fetch_response(
        url="https://x.test/nope", final_url="https://x.test/nope", status=404,
        content_type="text/html", size=120,
        body='<html><script>{"captcha":"hcaptcha"}</script></html>')
    failures = _check(failures, "a 404 that merely mentions captcha stays a 404",
                      "does not exist" in wiki_404 and "bot-protection" not in wiki_404,
                      wiki_404)
    payload = orun.format_fetch_response(
        url="https://api.test/x", final_url="https://api.test/x", status=200,
        content_type="application/json", body='{"z": 1, "a": [1, 2]}', size=22)
    failures = _check(failures, "JSON is pretty-printed",
                      '"a": [' in payload and "\n" in payload, payload)
    shell = orun.format_fetch_response(
        url="https://x.test/app", final_url="https://x.test/app", status=200,
        content_type="text/html", size=5000,
        body="<html><body><noscript>enablejs please</noscript></body></html>")
    failures = _check(failures, "JavaScript shells are called out",
                      "JavaScript" in shell and "web_search" in shell, shell)
    empty = orun.format_fetch_response(
        url="https://x.test/empty", final_url="https://x.test/empty", status=200,
        content_type="text/html", body="", size=0)
    failures = _check(failures, "empty bodies say so",
                      "(empty response body)" in empty, empty)
    binary = orun.format_fetch_response(
        url="https://x.test/blob", final_url="https://x.test/blob", status=200,
        content_type="application/octet-stream", body="", size=1024)
    failures = _check(failures, "binary payloads are described, not dumped",
                      "binary content" in binary and "1024 bytes" in binary, binary)
    with_links = orun.format_fetch_response(
        url="https://example.com/page", final_url="https://example.com/final",
        status=200, content_type="text/html", body=html, size=len(html))
    failures = _check(failures, "HTML pages come back with their links",
                      "Links on this page:" in with_links
                      and "https://example.com/docs/one" in with_links
                      and "(asked for https://example.com/page)" in with_links,
                      with_links)

    failures = _check(failures, "DNS failures get a short explanation",
                      "could not be resolved" in net_tools._request_error(
                          requests.exceptions.ConnectionError(
                              "HTTPSConnectionPool: Failed to resolve 'nope.invalid'")))

    # ── the handlers, against a fake network
    registry = orun.default_registry(tts=False, search=True)
    failures = _check(failures, "web_search pushes a re-search over a verdict",
                      "starting point, not a verdict" in registry["web_search"].description
                      and "search again" in registry["web_search"].description,
                      registry["web_search"].description)
    failures = _check(failures, "fetch_url points back at a re-search too",
                      "search again" in registry["fetch_url"].description)
    ctx = orun.ToolContext(cwd=os.getcwd(), timeout=10, max_output=4000)
    real_get = requests.get
    calls: list[str] = []
    sent_headers: list[dict] = []

    class _FakeResponse:
        def __init__(self, url="https://example.com/page", status=200,
                     content_type="text/html; charset=utf-8",
                     body=b"<html><body><h1>Hi there</h1></body></html>"):
            self.status_code = status
            self.url = url
            self.headers = {"Content-Type": content_type}
            self.content = body
            self.encoding = None
            self.text = body.decode("utf-8", "replace")

        def close(self):
            pass

    def fake_get(url, **kwargs):
        calls.append(url)
        sent_headers.append(kwargs.get("headers") or {})
        if url.endswith(".pdf"):
            return _FakeResponse(url=url, content_type="application/pdf",
                                 body=_MINIMAL_PDF)
        if url.endswith("/utf8"):
            return _FakeResponse(url=url, content_type="text/html",
                                 body=("<html><body><p>café — ok</p></body>"
                                       "</html>").encode("utf-8"))
        return _FakeResponse(url="https://example.com/final")

    requests.get = fake_get
    try:
        outcome = orun.execute_tool(registry["fetch_url"],
                                    {"url": "https://example.com/page"}, ctx)
        failures = _check(failures, "fetch_url returns extracted text",
                          outcome.ok and "Hi there" in outcome.text
                          and "HTTP 200" in outcome.text, outcome.text)
        failures = _check(failures, "fetch_url sends browser-like headers",
                          "Mozilla" in sent_headers[0].get("User-Agent", ""),
                          str(sent_headers[:1]))
        utf8 = orun.execute_tool(registry["fetch_url"],
                                 {"url": "https://example.com/utf8"}, ctx)
        failures = _check(failures, "UTF-8 pages without a charset decode right",
                          outcome.ok and "café — ok" in utf8.text, utf8.text)
        before = len(calls)
        again = orun.execute_tool(registry["fetch_url"],
                                  {"url": "https://example.com/page"}, ctx)
        failures = _check(failures, "a repeated fetch is answered from the cache",
                          "already fetched" in again.text and len(calls) == before,
                          f"{again.text[:80]!r} calls={calls}")

        pdf_out = orun.execute_tool(registry["fetch_url"],
                                    {"url": "https://example.com/doc.pdf"},
                                    orun.ToolContext(cwd=os.getcwd(), timeout=10,
                                                     max_output=4000))
        failures = _check(failures, "PDFs are turned into text",
                          pdf_out.ok and "Hello PDF world" in pdf_out.text,
                          pdf_out.text[:120])

        real_extract = fetch_tools.extract_pdf_text
        fetch_tools.extract_pdf_text = lambda data, max_chars=1, pages=None: None
        try:
            no_pdf = orun.execute_tool(registry["fetch_url"],
                                       {"url": "https://example.com/nopdf.pdf"},
                                       ctx)
        finally:
            fetch_tools.extract_pdf_text = real_extract
        failures = _check(failures, "a failed PDF extraction points at read_pdf",
                          "its text could not be extracted" in no_pdf.text
                          and "read_pdf" in no_pdf.text, no_pdf.text[:200])

        class _FakeDDGS:
            queries: list[tuple] = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def text(self, query, **kwargs):
                _FakeDDGS.queries.append((query, kwargs.get("max_results")))
                if query == "boom":
                    raise RuntimeError("backend down")
                if query == "nothing":
                    return []
                return [{"title": "Raspberry Pi AI Kit",
                         "href": "https://www.raspberrypi.com/products/ai-kit/",
                         "body": "The AI Kit pairs the Pi 5 with a Hailo chip."},
                        {"title": "Second",
                         "url": "https://docs.test/ai-hat",
                         "snippet": "More detail"}]

        original_ddgs = search_tools.DDGS
        search_tools.DDGS = _FakeDDGS
        try:
            search = orun.execute_tool(registry["web_search"],
                                       {"query": "raspberry pi ai hat"}, ctx)
            failures = _check(failures, "web_search formats results",
                              search.ok and "1. Raspberry Pi AI Kit" in search.text
                              and "https://www.raspberrypi.com/products/ai-kit/" in search.text
                              and "2. Second" in search.text
                              and "docs.test/ai-hat" in search.text, search.text)
            failures = _check(failures, "web_search nudges the model to fetch",
                              "fetch_url" in search.text
                              and "search again" in search.text, search.text)
            count = len(_FakeDDGS.queries)
            cached = orun.execute_tool(registry["web_search"],
                                       {"query": "Raspberry Pi AI Hat"}, ctx)
            failures = _check(failures, "a repeated search is answered from the cache",
                              "cached result" in cached.text
                              and len(_FakeDDGS.queries) == count, cached.text[:120])
            orun.execute_tool(registry["web_search"],
                              {"query": "clamp", "max_results": 99}, ctx)
            failures = _check(failures, "max_results is clamped",
                              _FakeDDGS.queries[-1] == ("clamp", 10),
                              str(_FakeDDGS.queries[-1:]))
            none = orun.execute_tool(registry["web_search"], {"query": "nothing"}, ctx)
            failures = _check(failures, "no results tells the model to search again",
                              none.ok and "no results" in none.text
                              and "not proof" in none.text
                              and "search again" in none.text
                              and "do not guess URLs" in none.text, none.text)
            broken = orun.execute_tool(registry["web_search"], {"query": "boom"}, ctx)
            failures = _check(failures, "backend failures surface as errors",
                              not broken.ok and "web search failed" in broken.text,
                              broken.text)

            routed = orun.execute_tool(
                registry["fetch_url"],
                {"url": "https://www.google.com/search?q=pi+ai+hat"}, ctx)
            failures = _check(failures, "search-engine URLs are rerouted to web_search",
                              routed.ok and "ran web_search for 'pi ai hat'" in routed.text
                              and "Raspberry Pi AI Kit" in routed.text, routed.text)

            failures = _check(failures, "web_search is read-only (auto-approved)",
                              orun.approval_decision(registry["web_search"],
                                                     {"query": "x"},
                                                     policy="ask").allowed)
            failures = _check(failures, "web_search is absent without a backend",
                              "web_search" not in orun.default_registry(tts=False,
                                                                        search=False))
            failures = _check(failures, "fetch_url survives without a backend",
                              "fetch_url" in orun.default_registry(tts=False,
                                                                   search=False))
            failures = _check(failures, "search_available follows the backend",
                              orun.search_available() is True)
        finally:
            search_tools.DDGS = original_ddgs
    finally:
        requests.get = real_get

    failures = _check(failures, "invented tool names are resolved",
                      orun.resolve_tool_alias("search") == "web_search"
                      and orun.resolve_tool_alias("Google") == "web_search"
                      and orun.resolve_tool_alias("curl") == "fetch_url"
                      and orun.resolve_tool_alias("run_shell") == "run_shell")
    failures = _check(failures, "search-engine URLs are recognised",
                      orun.search_engine_query(
                          "https://www.google.com/search?q=pi+ai+hat") == "pi ai hat"
                      and orun.search_engine_query(
                          "https://lite.duckduckgo.com/lite/?q=test") == "test"
                      and orun.search_engine_query(
                          "https://www.raspberrypi.com/products/") == "")
    failures = _check(failures, "PDFs are text-extracted",
                      "Hello PDF world" in (orun.extract_pdf_text(_MINIMAL_PDF) or ""),
                      "needs pypdf or pdftotext")
    failures = _check(failures, "scan-like PDFs extract to nothing",
                      orun.extract_pdf_text(_BLANK_PDF) == "",
                      repr(orun.extract_pdf_text(_BLANK_PDF)))

    # ── the REPL dispatcher resolves aliases too
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain",
                                               "--no-save", "-y"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        repl.tool_registry["web_search"] = orun.ToolSpec(
            "web_search", "test double",
            {"type": "object", "properties": {}, "required": []},
            lambda a, c: f"searched {a.get('query')}", read_only=True)
        with orun.console.capture():
            repl._execute_tool_calls([{"function": {"name": "search",
                                                    "arguments": {"query": "pi"}}}],
                                     quiet=True)
        message = repl.session.messages[-1]
        failures = _check(failures, "the REPL runs an aliased call as the real tool",
                          message.get("tool_name") == "web_search"
                          and "is called 'web_search' here" in message.get("content", "")
                          and "searched pi" in message.get("content", ""),
                          str(message))
        with orun.console.capture():
            repl._execute_tool_calls([{"function": {"name": "call_web_search",
                                                    "arguments": {"query": "pi"}}}],
                                     quiet=True)
        message = repl.session.messages[-1]
        failures = _check(failures, "a call_ prefixed call runs the aliased tool",
                          message.get("tool_name") == "web_search"
                          and "'call_web_search' is called 'web_search' here"
                          in message.get("content", "")
                          and "searched pi" in message.get("content", ""),
                          str(message))
        with orun.console.capture():
            repl._execute_tool_calls([{"function": {"name": "call_whatever",
                                                    "arguments": {}}}],
                                     quiet=True)
        message = repl.session.messages[-1]
        failures = _check(failures, "an unknown call_ name stays unknown",
                          "unknown or disabled tool: call_whatever"
                          in message.get("content", ""), str(message))
        repl.session.tools_disabled = ["web_search"]
        with orun.console.capture():
            repl._execute_tool_calls([{"function": {"name": "search",
                                                    "arguments": {"query": "pi"}}}],
                                     quiet=True)
        message = repl.session.messages[-1]
        failures = _check(failures, "aliases respect a disabled target",
                          "unknown or disabled tool: search" in message.get("content", ""),
                          str(message))
    return failures


class _PdfClient:
    """A per-model-capability client double for the read_pdf tests."""

    def __init__(self, capabilities=None, models=None, resident=(), replies=None):
        self.base = "http://fake:11434"
        self.capability_map = dict(capabilities or {})
        self.models = [dict(entry) for entry in (models or [])]
        self.resident = list(resident)
        self.replies = list(replies or [])
        self.payloads = []
        self.unloaded = []
        self.on_chat = None

    def tags(self):
        return [dict(entry) for entry in self.models]

    def show(self, model):
        return {"capabilities": list(self.capability_map.get(model,
                                                             ["completion"]))}

    def model_exists(self, model):
        return any(entry.get("name") == model for entry in self.models)

    def is_resident(self, model):
        return model in self.resident

    def unload(self, model):
        self.unloaded.append(model)

    def chat(self, payload, timeout=None, cancel=None):
        self.payloads.append(payload)
        if self.on_chat is not None:
            self.on_chat()
        reply = self.replies.pop(0) if self.replies else "recovered page text"
        return iter([{"message": {"content": reply}, "done": True}])


def test_pdf_tool() -> int:
    failures = 0
    from ollamarun_lib.tools import read_pdf as pdf_tools

    registry = orun.default_registry(tts=False)
    spec = registry.get("read_pdf")

    def context(client=None, **kwargs):
        return orun.ToolContext(cwd=os.getcwd(), timeout=10, max_output=40000,
                                client=client, vision_prompt="", **kwargs)

    def fake_render(data, page, directory, dpi=150):
        path = Path(directory) / f"page-{page + 1}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        return path

    real_render = pdf_tools.render_pdf_page
    pdf_tools.render_pdf_page = fake_render
    try:
        failures = _check(failures, "read_pdf is a read-only tool",
                          spec is not None and spec.read_only
                          and orun.resolve_tool_alias("pdf") == "read_pdf"
                          and orun.resolve_tool_alias("extract_pdf") == "read_pdf"
                          and orun.approval_decision(spec, {"path": "x"},
                                                     policy="ask").allowed,
                          str(spec))
        failures = _check(failures, "page specs parse and clamp",
                          orun.parse_page_spec("all") is None
                          and orun.parse_page_spec("1-3,7", total=10) == [0, 1, 2, 6]
                          and orun.parse_page_spec("2", total=5, cap=10) == [1]
                          and orun.select_pdf_pages(None, total=30, cap=10)
                          == list(range(10))
                          and orun.pdf_page_count(_MINIMAL_PDF) == 1)
        for bad in ("zero", "0-2", "3-1", "x"):
            try:
                orun.parse_page_spec(bad, total=5)
                parsed = "no error"
            except orun.ToolError as exc:
                parsed = str(exc)
            failures = _check(failures, f"a bad page spec is refused ({bad})",
                              parsed != "no error" and "pages" in parsed, parsed)

        with tempfile.TemporaryDirectory() as tmp:
            text_pdf = Path(tmp) / "text.pdf"
            text_pdf.write_bytes(_MINIMAL_PDF)
            scan_pdf = Path(tmp) / "scan.pdf"
            scan_pdf.write_bytes(_BLANK_PDF)

            # ── text layer: no model call, and a second read is cached
            client = _PdfClient()
            ctx = context(client)
            out = orun.execute_tool(spec, {"path": str(text_pdf)}, ctx)
            failures = _check(failures, "read_pdf returns a text layer",
                              out.ok and "Hello PDF world" in out.text
                              and "text layer" in out.text
                              and client.payloads == [], out.text[:200])
            again = orun.execute_tool(spec, {"path": str(text_pdf)}, ctx)
            failures = _check(failures, "a repeated read is answered from the session",
                              "not reading" in again.text
                              and "Hello PDF world" in again.text
                              and client.payloads == [], again.text[:160])
            no_source = orun.execute_tool(spec, {}, context(client))
            failures = _check(failures, "read_pdf without path or url is refused",
                              not no_source.ok and "path" in no_source.text
                              and "url" in no_source.text,
                              no_source.text[:160])

            # ── a scan read by a vision-capable session model: no swap
            client = _PdfClient(
                capabilities={"vlm:7b": ["completion", "vision", "thinking"]},
                models=[{"name": "vlm:7b", "size": 4}],
                resident=["vlm:7b"])
            ctx = context(client, current_model=lambda: "vlm:7b")
            out = orun.execute_tool(spec, {"path": str(scan_pdf)}, ctx)
            failures = _check(failures, "a vision session model reads a scan itself",
                              out.ok and "--- page 1 ---" in out.text
                              and "recovered page text" in out.text
                              and "read with vlm:7b" in out.text
                              and client.unloaded == []
                              and client.payloads
                              and client.payloads[0]["model"] == "vlm:7b"
                              and client.payloads[0].get("think") is False,
                              f"{out.text[:160]} calls={client.payloads}")
            failures = _check(failures, "the panel notes the vision model",
                              out.title_note == "vision: vlm:7b", out.title_note)

            # ── a non-vision session model: pick, unload, read, unload
            client = _PdfClient(
                capabilities={"text:8b": ["completion"],
                              "moondream:1.8b": ["completion", "vision"],
                              "qwen3-vl:8b": ["completion", "vision"],
                              "qwen3-vl:2b": ["completion", "vision"]},
                models=[{"name": "text:8b", "size": 9},
                        {"name": "moondream:1.8b", "size": 1},
                        {"name": "qwen3-vl:8b", "size": 3},
                        {"name": "qwen3-vl:2b", "size": 2}],
                resident=["text:8b"])
            ctx = context(client, current_model=lambda: "text:8b")
            out = orun.execute_tool(spec, {"path": str(scan_pdf)}, ctx)
            failures = _check(failures, "the smallest tag of the best family is picked",
                              out.ok and "read with qwen3-vl:2b" in out.text
                              and client.payloads
                              and client.payloads[0]["model"] == "qwen3-vl:2b",
                              f"{out.text[:160]} calls={client.payloads}")
            failures = _check(failures, "the session model is unloaded and restored",
                              client.unloaded == ["text:8b", "qwen3-vl:2b"],
                              str(client.unloaded))
            failures = _check(failures, "the result explains the swap",
                              "cannot read images" in out.text
                              and "reloads on the next round" in out.text,
                              out.text[:240])
            failures = _check(failures, "the panel names the swap",
                              out.title_note == "text:8b → qwen3-vl:2b",
                              out.title_note)
            failures = _check(failures, "the model ranking prefers the family",
                              orun.pick_vision_model(client, current="text:8b")[0]
                              == "qwen3-vl:2b")

            # ── pinned / unfindable / disabled models
            pinned_missing = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(client, current_model=lambda: "text:8b",
                        pdf_vision_model="not-installed:1b"))
            failures = _check(failures, "a pinned model must be installed",
                              not pinned_missing.ok
                              and "pdf_vision_model" in pinned_missing.text,
                              pinned_missing.text[:200])
            pinned_client = _PdfClient(
                capabilities={"text:8b": ["completion"],
                              "moondream:1.8b": ["completion", "vision"]},
                models=[{"name": "text:8b", "size": 9},
                        {"name": "moondream:1.8b", "size": 1}],
                resident=["text:8b"])
            pinned = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(pinned_client, current_model=lambda: "text:8b",
                        pdf_vision_model="moondream:1.8b"))
            failures = _check(failures, "a pinned model wins the choice",
                              pinned.ok
                              and "read with moondream:1.8b" in pinned.text
                              and pinned_client.payloads
                              and pinned_client.payloads[0]["model"]
                              == "moondream:1.8b",
                              pinned.text[:200])

            no_vision = _PdfClient(
                capabilities={"text:8b": ["completion"]},
                models=[{"name": "text:8b", "size": 9}])
            missing = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(no_vision, current_model=lambda: "text:8b"))
            failures = _check(failures, "no installed vision model is a clear failure",
                              not missing.ok and "no vision model is installed" in missing.text
                              and no_vision.unloaded == [] and not no_vision.payloads,
                              missing.text[:200])

            off = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(None, current_model=lambda: "text:8b", pdf_vision=False))
            failures = _check(failures, "pdf_vision false stays text-only",
                              off.ok and "pdf_vision: false" in off.text,
                              off.text[:200])

            locked = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(client, current_model=lambda: "text:8b",
                        pdf_vision_switch=False))
            failures = _check(failures, "pdf_vision_switch false refuses the swap",
                              not locked.ok and "pdf_vision_switch" in locked.text,
                              locked.text[:200])

            text_mode = orun.execute_tool(
                spec, {"path": str(scan_pdf), "mode": "text"},
                context(client, current_model=lambda: "text:8b"))
            failures = _check(failures, "mode=text never calls a model",
                              text_mode.ok and "no text layer" in text_mode.text,
                              text_mode.text[:200])
            forced = orun.execute_tool(
                spec, {"path": str(text_pdf), "mode": "vision"},
                context(_PdfClient(
                    capabilities={"vlm:7b": ["completion", "vision"]},
                    models=[{"name": "vlm:7b", "size": 4}]),
                    current_model=lambda: "vlm:7b"))
            failures = _check(failures, "mode=vision forces page images",
                              forced.ok and "--- page 1 ---" in forced.text,
                              forced.text[:200])
            empty_pages = orun.execute_tool(
                spec, {"path": str(scan_pdf), "pages": "2"},
                context(client, current_model=lambda: "text:8b"))
            failures = _check(failures, "pages past the end are refused",
                              not empty_pages.ok and "selects no page" in empty_pages.text,
                              empty_pages.text[:200])

            # ── cancellation stops the pass and still unloads the vision model
            token = orun.CancelToken()
            cancelled_client = _PdfClient(
                capabilities={"text:8b": ["completion"],
                              "qwen3-vl:2b": ["completion", "vision"]},
                models=[{"name": "text:8b", "size": 9},
                        {"name": "qwen3-vl:2b", "size": 2}],
                resident=["text:8b"])
            cancelled_client.on_chat = token.cancel
            cancelled = orun.execute_tool(
                spec, {"path": str(scan_pdf)},
                context(cancelled_client, current_model=lambda: "text:8b",
                        cancel=token))
            failures = _check(failures, "a cancelled vision pass is reported",
                              cancelled.cancelled and not cancelled.ok,
                              cancelled.text[:160])
            failures = _check(failures, "a cancelled pass still unloads the vision model",
                              cancelled_client.unloaded == ["text:8b", "qwen3-vl:2b"],
                              str(cancelled_client.unloaded))

            # ── read_file points a PDF at read_pdf
            refused = orun.execute_tool(registry["read_file"],
                                        {"path": str(text_pdf)}, context())
            failures = _check(failures, "read_file refuses a PDF",
                              not refused.ok and "read_pdf" in refused.text,
                              refused.text[:200])

        # ── URL reads
        class _FakePdfResponse:
            status_code = 200

            def iter_content(self, size):
                yield _MINIMAL_PDF

            def close(self):
                pass

        real_get = pdf_tools.requests.get
        pdf_tools.requests.get = lambda *a, **k: _FakePdfResponse()
        try:
            url_out = orun.execute_tool(
                spec, {"url": "https://example.com/doc.pdf"}, context())
        finally:
            pdf_tools.requests.get = real_get
        failures = _check(failures, "read_pdf fetches a URL",
                          url_out.ok and "Hello PDF world" in url_out.text,
                          url_out.text[:200])
    finally:
        pdf_tools.render_pdf_page = real_render
    return failures


def test_tool_loop() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        config = dict(orun.DEFAULT_CONFIG)
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save", "-y"])
        first = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": "echo from-tool"}}}]},
             "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        second = [
            {"message": {"role": "assistant", "content": "all done"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[first, second])
        repl = orun.Repl(args, config, client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        repl.session.add_message("user", "run something")
        with orun.console.capture() as cap:
            result = repl._generate()
        out = _plain(cap.get())

        failures = _check(failures, "tool loop returns the final answer",
                          result.content == "all done", result.content)
        failures = _check(failures, "tool loop makes one call per round",
                          len(client.chat_payloads) == 2, str(len(client.chat_payloads)))
        failures = _check(failures, "tools are sent with the request",
                          "run_shell" in json.dumps(client.chat_payloads[0].get("tools", [])))
        roles = [message["role"] for message in repl.session.messages]
        failures = _check(failures, "session records assistant/tool/assistant",
                          roles == ["user", "assistant", "tool", "assistant"], str(roles))
        tool_message = repl.session.messages[2]
        failures = _check(failures, "tool result carries the name and output",
                          tool_message.get("tool_name") == "run_shell"
                          and "from-tool" in tool_message.get("content", ""),
                          str(tool_message))
        failures = _check(failures, "second request replays tool history",
                          any(m.get("role") == "tool"
                              for m in client.chat_payloads[1]["messages"])
                          and any(m.get("tool_calls")
                                  for m in client.chat_payloads[1]["messages"]))
        failures = _check(failures, "tool round does not inflate the turn count",
                          repl.session.stats.get("turns") == 1,
                          str(repl.session.stats.get("turns")))
        failures = _check(failures, "tool result panel is rendered",
                          "tool · run_shell" in out and "ok" in out, out)

        # Non-interactive + no -y: a risky call is denied with a note.
        args2 = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        script2 = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": "rm -rf /tmp/definitely-not"}}}]},
             "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        script3 = [{"message": {"role": "assistant", "content": "understood"}, "done": True}]
        client2 = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[script2, script3])
        repl2 = orun.Repl(args2, dict(orun.DEFAULT_CONFIG), client2, store,
                          orun.Session.create("m1", client2.base))
        repl2._build_prompt_session = lambda: None
        repl2.session.add_message("user", "do it")
        with orun.console.capture():
            result2 = repl2._generate()
        denial = repl2.session.messages[2].get("content", "")
        failures = _check(failures, "unattended risky call is denied, model continues",
                          denial.startswith("tool not run:") and result2.content == "understood",
                          denial)

        # A remembered fingerprint runs even without -y.
        client3 = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[script2, script3])
        repl3 = orun.Repl(args2, dict(orun.DEFAULT_CONFIG), client3, store,
                          orun.Session.create("m1", client3.base))
        repl3._build_prompt_session = lambda: None
        repl3.session.tool_allow.append(orun.call_fingerprint(
            "run_shell", {"command": "rm -rf /tmp/definitely-not"}))
        repl3.session.add_message("user", "do it")
        with orun.console.capture():
            repl3._generate()
        allowed_text = repl3.session.messages[2].get("content", "")
        failures = _check(failures, "remembered fingerprints run unattended",
                          allowed_text.startswith("exit"), allowed_text)

        # A model that leaks the call id into the function name (issue #26):
        # the prefixed call must run the real tool, not die as unknown.
        script4 = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "call_run_shell",
                              "arguments": {"command": "echo from-prefix"}}}]},
             "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        script5 = [{"message": {"role": "assistant", "content": "recovered"},
                    "done": True}]
        client4 = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[script4, script5])
        repl4 = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client4, store,
                          orun.Session.create("m1", client4.base))
        repl4._build_prompt_session = lambda: None
        repl4.session.add_message("user", "run something")
        with orun.console.capture():
            result4 = repl4._generate()
        recovered = repl4.session.messages[2]
        failures = _check(failures, "a call_ prefixed name runs the real tool",
                          recovered.get("tool_name") == "run_shell"
                          and "from-prefix" in recovered.get("content", "")
                          and "'call_run_shell' is called 'run_shell' here"
                          in recovered.get("content", "")
                          and "unknown" not in recovered.get("content", ""),
                          str(recovered))
        failures = _check(failures, "the prefixed round reaches the model again",
                          result4.content == "recovered"
                          and any(m.get("role") == "tool"
                                  for m in client4.chat_payloads[1]["messages"]),
                          result4.content)
    return failures


def test_tool_commands() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None

        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "")
        out = _plain(cap.get())
        failures = _check(failures, "/tools lists the registry",
                          "run_shell" in out and "run_python" in out
                          and "policy" in out, out)
        failures = _check(failures, "/tools reports the active policy",
                          "allowlist" in out, out)

        with orun.console.capture():
            orun.cmd_tools(repl, "disable run_shell")
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tools disable drops the schema",
                          "run_shell" not in names and "run_python" in names, str(names))
        with orun.console.capture():
            orun.cmd_tools(repl, "enable run_shell")
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tools enable restores the schema",
                          "run_shell" in names, str(names))

        with orun.console.capture():
            orun.cmd_tools(repl, "policy auto")
        failures = _check(failures, "/tools policy sets the mode",
                          repl.tools_policy == "auto", repl.tools_policy)
        with orun.console.capture():
            orun.cmd_tools(repl, "off")
        failures = _check(failures, "/tools off stops sending schemas",
                          repl.active_tool_schemas() == [])
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "allow")
        failures = _check(failures, "/tools allow shows the shell allowlist",
                          "ls" in _plain(cap.get()) and "git" in _plain(cap.get()))

        # ── name-first per-tool control, help pages and direct calls ───────
        with orun.console.capture():
            orun.cmd_tools(repl, "on")
        with orun.console.capture():
            orun.cmd_tools(repl, "web_search off")
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tools NAME off drops just that schema",
                          "web_search" not in names and "fetch_url" in names, str(names))
        with orun.console.capture():
            orun.cmd_tools(repl, "search on")  # an invented alias resolves
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tools ALIAS on restores the schema",
                          "web_search" in names, str(names))

        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "web_search help")
        out = _plain(cap.get())
        failures = _check(failures, "/tools NAME help states the parameters",
                          "model calls it with" in out and "web_search(query" in out
                          and "required" in out and "optional" in out, out)
        failures = _check(failures, "/tools NAME help lists aliases and approval",
                          "aliases: search" in out and "read-only" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "web_search")
        failures = _check(failures, "/tools NAME alone shows the same page",
                          "model calls it with" in _plain(cap.get()))

        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "help")
        out = _plain(cap.get())
        failures = _check(failures, "/tools help documents every form",
                          "NAME call ARGS" in out and "policy" in out
                          and "rules" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "help web_search")
        failures = _check(failures, "/tools help NAME is the per-tool page",
                          "max_results" in _plain(cap.get()))
        with orun.console.capture() as cap:
            orun.cmd_help(repl, "tools")
        failures = _check(failures, "/help tools reaches the same reference",
                          "NAME call ARGS" in _plain(cap.get()))

        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "not_a_tool help")
        out = _plain(cap.get())
        failures = _check(failures, "an unknown tool is refused",
                          "unknown tool" in out and "usage: /tools" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "web_serch help")
        out = _plain(cap.get())
        failures = _check(failures, "a near-miss tool name gets a suggestion",
                          "did you mean" in out and "web_search" in out, out)

        # a definition whose `requires` is unmet still gets a help page
        spec = repl.tool_registry.pop("web_search")
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "web_search help")
        out = _plain(cap.get())
        failures = _check(failures, "a not-installed tool still explains itself",
                          "not installed" in out and "ddgs" in out
                          and "max_results" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "web_search off")
        failures = _check(failures, "a not-installed tool cannot be toggled",
                          "not installed" in _plain(cap.get()))
        repl.tool_registry["web_search"] = spec

        # ── /tools NAME call: JSON, key=value, bare value, validation ──────
        with orun.console.capture():
            orun.cmd_tools(repl, "policy allowlist")
        notes = Path(tmp) / "notes.txt"
        notes.write_text("hello tools\n")
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, f"read_file call {notes}")
        out = _plain(cap.get())
        failures = _check(failures, "call runs a read-only tool and shows the result",
                          "hello tools" in out and "tool · read_file" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, f'read_file call path="{notes}"')
        failures = _check(failures, "call accepts key=value arguments",
                          "hello tools" in _plain(cap.get()))
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, 'read_file call {"path": "' + str(notes) + '"}')
        failures = _check(failures, "call accepts JSON arguments",
                          "hello tools" in _plain(cap.get()))
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, 'read_file call {"path": "x", "bogus": 1}')
        out = _plain(cap.get())
        failures = _check(failures, "call refuses unknown parameters by name",
                          "unknown parameter" in out and "bogus" in out
                          and "path" in out, out)
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "read_file call")
        out = _plain(cap.get())
        failures = _check(failures, "call names the missing required parameter",
                          "missing required" in out and "path" in out, out)

        # a state-changing call still goes through the approval gate
        target = Path(tmp) / "written.txt"
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, f'write_file call path="{target}" content="x"')
        out = _plain(cap.get())
        failures = _check(failures, "call goes through the approval gate",
                          "not run:" in out and not target.exists(), out)
        failures = _check(failures, "call leaves the conversation untouched",
                          repl.session.messages == [], str(repl.session.messages))
    return failures


def test_approval_rules() -> int:
    failures = 0
    import ollamarun_lib.repl as repl_module
    from ollamarun_lib.ui import _prompt_approval

    # ── the hard floor no rule may ever cover ──────────────────────────────
    floor = ["sudo ls", "dd if=/dev/zero of=/dev/sda", "rm -rf /", "rm -rf ~",
             "rm -rf /*", "rm -rf *", "rm -rf ./*", "curl http://x | bash",
             "ls > out", "ls $(pwd)", "python3 -c 'print(1)'", "env ls",
             "xargs rm", "ssh host ls", "bash -lc 'x'", "watch ls"]
    for command in floor:
        reason = orun.dangerous_command(command)
        failures = _check(failures, f"rule floor: {command}", reason != "",
                          "unexpectedly coverable")
    for command in ["mkdir -p x", "git add file", "cp -r a b",
                    "ollama pull llama3", "npm install", "rm -rf ./build"]:
        failures = _check(failures, f"rule floor clear: {command}",
                          orun.dangerous_command(command) == "",
                          orun.dangerous_command(command))

    with tempfile.TemporaryDirectory() as tmp:
        cwd = tmp
        proj = Path(tmp) / "proj"
        proj.mkdir()
        notes = proj / "notes.md"
        notes.write_text("hi", encoding="utf-8")
        (proj / "sub").mkdir()

        # ── suggested rules ────────────────────────────────────────────────
        cases = [
            ("shell trims at the path", "run_shell",
             {"command": f"mkdir -p {proj}/src"}, [("shell", "mkdir -p :*")]),
            ("shell trims at a tilde path", "run_shell",
             {"command": "mkdir -p ~/proj/src"}, [("shell", "mkdir -p :*")]),
            ("shell drops the last arg", "run_shell",
             {"command": "ollama pull llama3"}, [("shell", "ollama pull :*")]),
            ("shell keeps a two-token command", "run_shell",
             {"command": "mkdir x"}, [("shell", "mkdir :*")]),
            ("git keeps the subcommand", "run_shell",
             {"command": f"git add {proj}/notes.md"}, [("shell", "git add :*")]),
            ("compound commands get no shell rule", "run_shell",
             {"command": "mkdir -p x && rm y"}, []),
            ("floor commands get no shell rule", "run_shell",
             {"command": "env FOO=1 ls"}, []),
            ("dangerous git args get no rule", "run_shell",
             {"command": "git config user.name admin"}, []),
            ("write inside cwd: exact + tree", "write_file",
             {"path": str(notes), "content": "x"},
             [("path", "notes.md"), ("path", "**")]),
            ("dir inside cwd: tree", "list_dir",
             {"path": str(proj)}, [("path", "proj/**")]),
        ]
        for name, tool, args, expected in cases:
            rules = orun.suggest_rules(tool, args, cwd=cwd)
            got = [(rule.kind, rule.pattern) for rule in rules]
            failures = _check(failures, f"suggest {name}", got == expected, str(got))

        outside = tempfile.mkdtemp()
        outside_file = Path(outside) / "x.txt"
        outside_file.write_text("z", encoding="utf-8")
        for tool in ("write_file", "read_file"):
            args = {"path": str(outside_file)}
            if tool == "write_file":
                args["content"] = "x"
            rules = orun.suggest_rules(tool, args, cwd=cwd)
            got = [(rule.kind, rule.pattern) for rule in rules]
            failures = _check(failures, f"outside cwd: exact only ({tool})",
                              got == [("path", "x.txt")], str(got))
        rules = orun.suggest_rules("run_python",
                                   {"code": "import os\nimport sys\n"
                                            "print(os.path, sys.argv)"}, cwd=cwd)
        failures = _check(failures, "python rule lists the module roots",
                          [(r.kind, r.pattern) for r in rules] == [("python", "os,sys")],
                          str(rules))
        rules = orun.suggest_rules("run_python",
                                   {"code": "import os\nos.system('ls')"}, cwd=cwd)
        failures = _check(failures, "python rule refused for .system", rules == [], str(rules))

        # ── shell rules ────────────────────────────────────────────────────
        shell_rule = orun.ToolRule(kind="shell", pattern="mkdir -p :*")
        shell_cases = [
            ({"command": f"mkdir -p {proj}/a/b"}, True),
            ({"command": "mkdir /tmp/x"}, False),
            ({"command": "mkdir -p / && sudo rm -rf /"}, False),
            ({"command": f"mkdir -p {proj}/a && ls"}, True),
            ({"command": f"mkdir -p {proj}/a && rm {proj}/x"}, False),
        ]
        for args, expected in shell_cases:
            got = orun.rule_matches(shell_rule, "run_shell", args, cwd=cwd)
            failures = _check(failures, f"shell rule {'covers' if expected else 'rejects'}"
                              f" {args['command']}", got is expected, f"got={got}")
        failures = _check(failures, "shell rule is tool-scoped",
                          orun.rule_matches(shell_rule, "run_python",
                                            {"code": "print(1)"}, cwd=cwd) is False)
        exact_rule = orun.ToolRule(kind="shell", pattern="mkdir -p")
        failures = _check(failures, "a pattern without :* is exact",
                          orun.rule_matches(exact_rule, "run_shell",
                                            {"command": "mkdir -p /x"}, cwd=cwd) is False
                          and orun.rule_matches(exact_rule, "run_shell",
                                                {"command": "mkdir -p"}, cwd=cwd) is True)
        git_rule = orun.ToolRule(kind="shell", pattern="git add :*")
        failures = _check(failures, "git subcommand rules cover git add",
                          orun.rule_matches(git_rule, "run_shell",
                                            {"command": f"git add {proj}/x"}, cwd=cwd) is True)
        failures = _check(failures, "bare git prefixes stay refused",
                          orun.rule_matches(orun.ToolRule(kind="shell", pattern="git :*"),
                                            "run_shell",
                                            {"command": f"git add {proj}/x"}, cwd=cwd) is False)
        failures = _check(failures, "git structural hazards keep their veto",
                          orun.rule_matches(git_rule, "run_shell",
                                            {"command": f"git -c core.pager=x add {proj}/x"},
                                            cwd=cwd) is False)
        failures = _check(failures, "rule labels read well",
                          orun.rule_label(shell_rule) == "shell: mkdir -p :*",
                          orun.rule_label(shell_rule))

        # ── path rules ─────────────────────────────────────────────────────
        tree_rule = orun.ToolRule(kind="path", pattern="**", base=str(proj))
        path_cases = [
            ("deep write", "write_file", str(proj / "sub" / "f.txt"), True),
            ("relative write", "write_file", "proj/sub/f.txt", True),
            ("escape attempt", "write_file", str(proj / ".." / "escape.txt"), False),
            ("outside the base", "write_file", "/etc/passwd-x", False),
        ]
        for name, tool, path, expected in path_cases:
            args = {"path": path, "content": "x"}
            got = orun.rule_matches(tree_rule, tool, args, cwd=cwd)
            failures = _check(failures, f"path rule {name}", got is expected, f"got={got}")
        link = proj / "link.txt"
        os.symlink("/etc/hostname", str(link))
        failures = _check(failures, "path rules resolve symlinks out of the tree",
                          orun.rule_matches(tree_rule, "read_file",
                                            {"path": str(link)}, cwd=cwd) is False)
        dir_rule = orun.ToolRule(kind="path", pattern="proj/**", base=tmp)
        failures = _check(failures, "dir tree covers the dir and children",
                          orun.rule_matches(dir_rule, "list_dir",
                                            {"path": str(proj)}, cwd=cwd) is True
                          and orun.rule_matches(dir_rule, "list_dir",
                                                {"path": str(proj / "sub")}, cwd=cwd) is True)
        failures = _check(failures, "dir tree does not cover siblings",
                          orun.rule_matches(dir_rule, "list_dir",
                                            {"path": tmp}, cwd=cwd) is False)
        file_rule = orun.ToolRule(kind="path", pattern="notes.md", base=str(proj))
        failures = _check(failures, "exact file rule matches only that file",
                          orun.rule_matches(file_rule, "read_file",
                                            {"path": str(notes)}, cwd=cwd) is True
                          and orun.rule_matches(file_rule, "read_file",
                                                {"path": str(proj / "other.md")},
                                                cwd=cwd) is False)
        failures = _check(failures, "path rules are tool-scoped",
                          orun.rule_matches(file_rule, "run_shell",
                                            {"command": f"cat {notes}"}, cwd=cwd) is False)

        # ── python rules ───────────────────────────────────────────────────
        py_rule = orun.ToolRule(kind="python", pattern="os")
        py_cases = [
            ("plain module use", "import os\nprint(os.path.join('a', 'b'))", True),
            ("os.system call", "import os\nos.system('ls')", False),
            ("imported os.system", "from os import system\nsystem('ls')", False),
            ("open()", "open('/etc/passwd')", False),
            ("wildcard import", "from os import *\nprint(os)", False),
            ("name without import", "print(os)", False),
        ]
        for name, code, expected in py_cases:
            got = orun.rule_matches(py_rule, "run_python", {"code": code}, cwd=cwd)
            failures = _check(failures, f"python rule {name}", got is expected, f"got={got}")

        # ── stored rules through the approval flow, unattended included ────
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        script = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_shell",
                              "arguments": {"command": "rm -rf /tmp/definitely-not"}}}]},
             "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True},
        ]
        final = [{"message": {"role": "assistant", "content": "cleaned"}, "done": True}]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[script, final])
        session = orun.Session.create("m1", client.base)
        session.tool_rules = [orun.ToolRule(kind="shell", pattern="rm -rf :*",
                                            created="t").to_dict()]
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG), client, store, session)
        repl._build_prompt_session = lambda: None
        repl.session.add_message("user", "clean it")
        with orun.console.capture() as cap:
            result = repl._generate()
        out = _plain(cap.get())
        tool_text = repl.session.messages[2].get("content", "")
        failures = _check(failures, "stored rules run unattended",
                          tool_text.startswith("exit") and result.content == "cleaned",
                          tool_text)
        failures = _check(failures, "rule matches are announced",
                          "allowed by rule" in out, out)

        # ── /tools rules listing, add and rm ───────────────────────────────
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "rules")
        out = _plain(cap.get())
        failures = _check(failures, "/tools rules lists the stored rule",
                          "rm -rf :*" in out and "shell" in out, out)
        with orun.console.capture():
            orun.cmd_tools(repl, "rules add python os")
        failures = _check(failures, "/tools rules add appends",
                          [entry.get("pattern") for entry in repl.session.tool_rules]
                          == ["rm -rf :*", "os"], str(repl.session.tool_rules))
        with orun.console.capture() as cap:
            orun.cmd_tools(repl, "rules rm 2")
        failures = _check(failures, "/tools rules rm N removes",
                          [entry.get("pattern") for entry in repl.session.tool_rules]
                          == ["rm -rf :*"], str(repl.session.tool_rules))
        with orun.console.capture():
            orun.cmd_tools(repl, "rules rm 99")
        failures = _check(failures, "out-of-range rm changes nothing",
                          len(repl.session.tool_rules) == 1)
        status = orun.Repl._session_rules(repl)
        failures = _check(failures, "session rules parse back into ToolRule",
                          len(status) == 1 and status[0].pattern == "rm -rf :*")

        # ── session persistence ────────────────────────────────────────────
        restored = orun.Session.from_dict(session.to_dict())
        failures = _check(failures, "tool_rules survive the session round trip",
                          restored.tool_rules == session.tool_rules,
                          str(restored.tool_rules))
        legacy = session.to_dict()
        legacy.pop("tool_rules")
        failures = _check(failures, "old sessions without tool_rules load",
                          orun.Session.from_dict(legacy).tool_rules == [])
        failures = _check(failures, "rule round trip",
                          orun.ToolRule.from_dict(shell_rule.to_dict()) == shell_rule)
        failures = _check(failures, "junk rules are rejected",
                          orun.ToolRule.from_dict({"kind": "nope", "pattern": "x"}) is None
                          and orun.ToolRule.from_dict("x") is None)

    # ── the numbered approval prompt ───────────────────────────────────────
    import ollamarun_lib.ui as ui_module  # the prompt lives here now

    class _StubPrompt:
        answer = "1"
        seen: list = []

        @classmethod
        def ask(cls, *args, **kwargs):
            cls.seen.append({"args": args, **kwargs})
            return cls.answer

    original_prompt = ui_module.Prompt
    ui_module.Prompt = _StubPrompt
    try:
        suggestions = [
            orun.ToolRule(kind="shell", pattern="mkdir -p :*", created="t"),
            orun.ToolRule(kind="path", pattern="**", base="/tmp/x", created="t"),
        ]
        _StubPrompt.answer = "1"
        with orun.console.capture() as cap:
            action, rule = _prompt_approval(suggestions)
        out = _plain(cap.get())
        failures = _check(failures, "prompt default answer runs once",
                          (action, rule) == ("run", None), f"{action} {rule}")
        failures = _check(failures, "prompt numbers every row with letters "
                          "for the fixed actions",
                          "1/y" in out and "2/a" in out and "5/A" in out
                          and "6/n" in out and "3 " in out and "4 " in out, out)
        failures = _check(failures, "prompt lists the suggested rules",
                          "mkdir -p :*" in out and "/tmp/x/**" in out, out)
        kwargs = _StubPrompt.seen[-1]
        failures = _check(failures, "prompt offers every number and letter",
                          all(choice in kwargs.get("choices", [])
                              for choice in ["1", "2", "3", "4", "5", "6",
                                             "y", "a", "A", "n"])
                          and kwargs.get("default") == "1", str(kwargs))
        expected = [
            ("1", "run", None), ("y", "run", None),
            ("2", "remember", None), ("a", "remember", None),
            ("3", "rule", "shell"), ("4", "rule", "path"),
            ("5", "auto", None), ("A", "auto", None),
            ("6", "deny", None), ("n", "deny", None),
        ]
        for answer, want_action, want_kind in expected:
            _StubPrompt.answer = answer
            with orun.console.capture():
                action, rule = _prompt_approval(suggestions)
            ok = action == want_action and ((rule is None) if want_kind is None
                                            else (rule is not None
                                                  and rule.kind == want_kind))
            failures = _check(failures, f"prompt answer {answer!r}",
                              ok, f"{action} {rule}")

        # ── interactive flow through _approve_tool_call ─────────────────────
        class _TerminalConsole:
            is_terminal = True

            @staticmethod
            def print(*args, **kwargs):
                orun.console.print(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp2:
            store2 = orun.SessionStore(sessions_dir=Path(tmp2) / "s",
                                       index_path=Path(tmp2) / "i.json")
            args2 = orun.build_parser().parse_args(
                ["m1", "--render", "plain", "--no-save"])
            client2 = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
            repl2 = orun.Repl(args2, dict(orun.DEFAULT_CONFIG), client2, store2,
                              orun.Session.create("m1", client2.base))
            repl2._build_prompt_session = lambda: None
            shell_spec = repl2.tool_registry["run_shell"]
            original_console = repl_module.console
            repl_module.console = _TerminalConsole()
            repl2.ui.interactive = True  # the gate is the Ui's, not the console's
            try:
                deny_args = {"command": "rm /tmp/approval-b/x"}
                _StubPrompt.answer = "n"
                with orun.console.capture():
                    run, note = repl2._approve_tool_call(
                        shell_spec, deny_args,
                        orun.call_fingerprint("run_shell", deny_args))
                failures = _check(failures, "typing n denies the call",
                                  run is False and note == "denied by the user", note)
                store_args = {"command": "mkdir -p /tmp/approval-a/x"}
                _StubPrompt.answer = "3"
                with orun.console.capture() as cap:
                    run, note = repl2._approve_tool_call(
                        shell_spec, store_args,
                        orun.call_fingerprint("run_shell", store_args))
                failures = _check(failures, "typing a rule number stores it and runs",
                                  run is True and "stored rule" in _plain(cap.get())
                                  and [entry.get("pattern")
                                       for entry in repl2.session.tool_rules]
                                  == ["mkdir -p :*"],
                                  str(repl2.session.tool_rules))
                failures = _check(failures, "the stored rule matches the call again",
                                  orun.matching_rule(repl2._session_rules(),
                                                     "run_shell", store_args,
                                                     cwd=repl2.tool_ctx.cwd,
                                                     shell_allow=repl2.shell_allow)
                                  is not None)
                remember_args = {"command": "cp /tmp/approval-c/x /tmp/approval-c/y"}
                _StubPrompt.answer = "2"
                with orun.console.capture():
                    run, note = repl2._approve_tool_call(
                        shell_spec, remember_args,
                        orun.call_fingerprint("run_shell", remember_args))
                failures = _check(failures, "typing 2 remembers the exact call",
                                  run is True
                                  and orun.call_fingerprint("run_shell", remember_args)
                                  in repl2.session.tool_allow)
                auto_args = {"command": "rm /tmp/approval-d/x"}
                _StubPrompt.answer = "A"
                with orun.console.capture():
                    run, note = repl2._approve_tool_call(
                        shell_spec, auto_args,
                        orun.call_fingerprint("run_shell", auto_args))
                failures = _check(failures, "typing A switches to auto-allow",
                                  run is True and repl2.auto_approve is True)
            finally:
                repl_module.console = original_console
    finally:
        ui_module.Prompt = original_prompt
    return failures


def test_message_queue() -> int:
    """The queue is what makes typing while the model works possible."""
    failures = 0
    queue = orun.MessageQueue()
    failures = _check(failures, "queue starts empty", len(queue) == 0 and not queue)
    queue.add_message("first")
    queue.add_command("/clear")
    queue.add_message("second")
    failures = _check(failures, "queue keeps order", [item.text for item in queue.items]
                      == ["first", "/clear", "second"], queue.describe())
    failures = _check(failures, "has_messages looks at the head", queue.has_messages())
    failures = _check(failures, "take_messages stops at a command",
                      queue.take_messages() == ["first"], queue.describe())
    failures = _check(failures, "a command at the head stops deliveries",
                      not queue.has_messages() and queue.take_messages() == [])
    failures = _check(failures, "pop_next returns the command",
                      queue.pop_next().kind == "command")
    failures = _check(failures, "remove counts from 1", queue.remove(1).text == "second"
                      and len(queue) == 0)
    failures = _check(failures, "remove out of range is a no-op", queue.remove(3) is None)
    queue.add_command("/one")
    queue.add_command("/two")
    failures = _check(failures, "clear reports how much it dropped", queue.clear() == 2
                      and not queue)

    # /queue command
    repl, client, tmp = _test_repl(resident=[])
    with orun.console.capture() as cap:
        orun.cmd_queue(repl, "")
    failures = _check(failures, "/queue says when it is empty",
                      "queue is empty" in _plain(cap.get()))
    repl.queue.add_message("hello there")
    repl.queue.add_command("/clear")
    with orun.console.capture() as cap:
        orun.cmd_queue(repl, "list")
    out = _plain(cap.get())
    failures = _check(failures, "/queue lists messages and commands",
                      "hello there" in out and "/clear" in out and "cmd" in out, out)
    with orun.console.capture():
        orun.cmd_queue(repl, "rm 1")
    failures = _check(failures, "/queue rm drops one item",
                      [item.text for item in repl.queue.items] == ["/clear"])
    with orun.console.capture():
        orun.cmd_queue(repl, "rm 9")
    failures = _check(failures, "/queue rm out of range is reported",
                      [item.text for item in repl.queue.items] == ["/clear"])
    with orun.console.capture():
        orun.cmd_queue(repl, "clear")
    failures = _check(failures, "/queue clear empties it", not repl.queue)
    with orun.console.capture() as cap:
        orun.cmd_queue(repl, "nonsense")
    failures = _check(failures, "/queue rejects unknown arguments",
                      "usage" in _plain(cap.get()))
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


def test_status_format() -> int:
    """The status line fits itself to the terminal and shows the right bits."""
    failures = 0
    snapshot = orun.StatusSnapshot(state="generating", detail="", model="qwen3:8b",
                                   remote=None, thinking="collapsed",
                                   tools="allowlist", render="markdown",
                                   context_used=12400, context_total=32768,
                                   queued=2, tps=41.2, tps_avg=38.5, elapsed=3.2)
    wide = orun.format_status(snapshot, frame="⠋", width=155)
    plain = orun.strip_markup(wide)
    failures = _check(failures, "status shows state, model, mode and context",
                      "⠋ generating" in plain and "qwen3:8b" in plain
                      and "think:collapsed" in plain and "tools:allowlist" in plain
                      and "12.400k/32.768k (37.842%)" in plain, plain)
    failures = _check(failures, "status shows the queue, speed and its average",
                      "2 queued" in plain and "41.2 tok/s (avg 38.5)" in plain, plain)
    failures = _check(failures, "status fits the terminal width",
                      len(plain) <= 155, str(len(plain)))
    narrow = orun.strip_markup(orun.format_status(snapshot, frame="⠋", width=46))
    failures = _check(failures, "narrow status keeps state, model and context",
                      "generating" in narrow and "qwen3:8b" in narrow
                      and "ctx" in narrow, narrow)
    failures = _check(failures, "narrow status drops the mode summary first",
                      "think:" not in narrow, narrow)
    tiny = orun.strip_markup(orun.format_status(snapshot, frame="⠋", width=18))
    failures = _check(failures, "tiny statuses are clipped, not wrapped",
                      len(tiny) <= 18, tiny)

    idle = orun.StatusSnapshot(state="ready", model="m", thinking="hidden",
                               tools="off", render="plain", context_used=None,
                               context_total=8192)
    plain = orun.strip_markup(orun.format_status(idle, frame="●", width=100))
    failures = _check(failures, "idle status shows an empty context as 0",
                      "ctx 0/8.192k" in plain and "tools:off" in plain, plain)
    done = orun.StatusSnapshot(state="done", detail="done in 3.2s", model="m",
                               render="markdown", tps=41.2)
    plain = orun.strip_markup(orun.format_status(done, frame="✔", width=80))
    failures = _check(failures, "done status keeps its summary",
                      "✔ done" in plain and "done in 3.2s" in plain, plain)
    failures = _check(failures, "done status has no average to show",
                      "41.2 tok/s" in plain and "avg" not in plain, plain)

    failures = _check(failures, "spinner animates while busy",
                      orun.spinner_frame("generating", 0.0) != orun.spinner_frame(
                          "generating", 0.15))
    failures = _check(failures, "spinner is a fixed icon when idle",
                      orun.spinner_frame("ready", 0.0) == orun.STATE_GLYPH["ready"]
                      and orun.spinner_frame("done", 0.5) == "✔")
    failures = _check(failures, "loading counts as busy",
                      orun.spinner_frame("loading", 0.05) in orun.SPINNER_FRAMES)
    return failures


def test_context_counter() -> int:
    """The ctx counter: dynamic token formatting and the 3-decimal share."""
    failures = 0
    failures = _check(failures, "counts read raw below 1k",
                      orun.human_tokens(845) == "845"
                      and orun.human_tokens(0) == "0")
    failures = _check(failures, "counts go 3-decimal k from 1k on",
                      orun.human_tokens(1000) == "1.000k"
                      and orun.human_tokens(12345) == "12.345k"
                      and orun.human_tokens(32768) == "32.768k")
    failures = _check(failures, "an unknown count reads ?",
                      orun.human_tokens(None) == "?")
    failures = _check(failures, "the ctx piece carries the 3-decimal share",
                      orun.format_context(12345, 32768)
                      == "ctx 12.345k/32.768k (37.674%)",
                      orun.format_context(12345, 32768))
    failures = _check(failures, "a small count stays raw inside the ctx piece",
                      orun.format_context(845, 32768) == "ctx 845/32.768k (2.579%)",
                      orun.format_context(845, 32768))
    failures = _check(failures, "missing numbers degrade like the status bar",
                      orun.format_context(None, None) == "ctx ?"
                      and orun.format_context(0, 8192) == "ctx 0/8.192k"
                      and orun.format_context(12345, None) == "ctx 12.345k")
    return failures


def test_tps_footer() -> int:
    """The streaming footer and the snapshot: rate, average, tokens, live ctx."""
    failures = 0
    buffer = io.StringIO()
    console = Console(file=buffer, width=200, force_terminal=False)

    renderer = orun.Renderer(console, mode="plain", context_used=12000,
                             context_total=32768)
    failures = _check(failures, "no footer before any token",
                      renderer._tps_text(now=0.0) is None)
    for tick in range(10):
        renderer.estimator.add(1, now=100.0 + tick)
    text = renderer._tps_text(now=109.0).plain
    failures = _check(failures, "footer shows the rate, its average and the count",
                      "~1.0 tok/s" in text and "(avg 1.1)" in text
                      and "10 tok" in text, text)
    failures = _check(failures, "footer counts the live ctx with 3 decimals",
                      "ctx 12.010k/32.768k (36.652%)" in text, text)

    bare = orun.Renderer(console, mode="plain")
    bare.estimator.add(1, now=0.0)
    plain = bare._tps_text(now=0.5).plain
    failures = _check(failures, "no ctx piece without a base to count from",
                      "ctx" not in plain, plain)

    quiet = orun.Renderer(console, mode="plain", tps_footer=False)
    quiet.estimator.add(1, now=0.0)
    failures = _check(failures, "the footer honours /tps off",
                      quiet._tps_text(now=1.0) is None)

    repl, client, _ = _test_repl()
    repl.session.stats["last_prompt_tokens"] = 12000
    repl.session.stats["last_completion_tokens"] = 0
    repl._context_totals[("m1", None)] = 32768
    streaming = orun.Renderer(console, mode="plain")
    streaming.estimator.add(1, now=time.monotonic() - 0.5)
    repl.current_renderer = streaming
    snapshot = repl.status_snapshot()
    failures = _check(failures, "the status bar adds the live stream to ctx",
                      snapshot.context_used == 12001
                      and snapshot.context_total == 32768
                      and snapshot.context_text() == "ctx 12.001k/32.768k (36.624%)",
                      snapshot.context_text())
    failures = _check(failures, "the status bar carries the windowed average",
                      snapshot.tps_avg is not None and snapshot.tps_avg > 1.0,
                      str(snapshot.tps_avg))
    repl.current_renderer = None
    failures = _check(failures, "idle status falls back to the stored ctx",
                      repl.status_snapshot().context_used == 12000
                      and repl.status_snapshot().tps_avg is None)
    return failures


def test_cancel_token() -> int:
    failures = 0
    token = orun.CancelToken()
    failures = _check(failures, "token starts clean", not token.cancelled)
    token.cancel()
    failures = _check(failures, "cancel() sets it", token.cancelled and token.is_set())
    token.reset()
    failures = _check(failures, "reset() clears it", not token.cancelled)
    return failures


def test_stream_cancel() -> int:
    """Cancelling must stop a stream even while it is blocked on the socket."""
    failures = 0
    release = threading.Event()
    closed = threading.Event()

    class BlockingResponse(_FakeResponse):
        def iter_lines(self, decode_unicode=False):
            yield json.dumps({"message": {"content": "partial "}}).encode()
            release.wait(10)          # simulates the model taking its time
            yield json.dumps({"message": {"content": "rest"}, "done": True}).encode()

        def close(self):
            closed.set()
            release.set()             # closing unblocks the reader

    response = BlockingResponse()
    client = orun.OllamaClient()
    client.http = _FakeSession(lambda *a, **k: response)
    token = orun.CancelToken()
    seen = []

    def consume():
        for event in client.chat({"model": "m"}, cancel=token):
            seen.append(event)

    reader = threading.Thread(target=consume)
    reader.start()
    time.sleep(0.2)
    failures = _check(failures, "stream is blocked mid-response", len(seen) == 1)
    failures = _check(failures, "abort_stream closes the response",
                      client.abort_stream() and closed.is_set())
    token.cancel()
    reader.join(5)
    failures = _check(failures, "cancelled stream ends without raising",
                      not reader.is_alive() and len(seen) == 1, str(seen))
    failures = _check(failures, "abort_stream is a no-op when idle",
                      client.abort_stream() is False)

    # A cancel that lands between chunks stops without closing anything.
    client2 = orun.OllamaClient()
    client2.http = _FakeSession(lambda *a, **k: _FakeResponse(
        lines=[json.dumps(event).encode() for event in
               [{"message": {"content": "a"}}, {"message": {"content": "b"}},
                {"done": True}]]))
    token2 = orun.CancelToken()
    got = []
    for event in client2.chat({"model": "m"}, cancel=token2):
        got.append(event)
        token2.cancel()
    failures = _check(failures, "cancel between chunks stops the stream",
                      len(got) == 1, str(got))
    return failures


def test_boundary_delivery() -> int:
    """A message typed while a tool round runs is delivered at the boundary,
    inside the same turn."""
    failures = 0
    repl, client, tmp = _test_repl()
    repl.tools_enabled = True
    repl.tools_policy = "auto"
    repl.auto_approve = True
    queued = {"done": False}

    class ScriptedClient(_FakeClient):
        def chat(self, payload, timeout=None, cancel=None):
            self.chat_payloads.append(payload)
            self.chat_calls += 1
            round_number = self.chat_calls
            if round_number == 1:
                return iter([
                    {"message": {"tool_calls": [{"function": {
                        "name": "list_dir", "arguments": {"path": "."}}}]}},
                    {"done": True, "eval_count": 3, "eval_duration": 1e9},
                ])
            if not queued["done"] and round_number == 2:
                queued["done"] = True
            return iter([{"message": {"content": "final answer"}}, {"done": True}])

        def is_resident(self, model):
            return False

    scripted = ScriptedClient()
    scripted.base = client.base
    repl.client = scripted

    original_execute = repl._execute_tool_calls

    def execute_with_queue(calls, **kwargs):
        # the user types while the tool is running -- the queue fills mid-turn
        repl.queue_message("and also check the parent directory")
        return original_execute(calls, **kwargs)

    repl._execute_tool_calls = execute_with_queue
    result = repl.send("look around")
    failures = _check(failures, "turn finished normally", not result.cancelled
                      and not result.error, str(result))
    failures = _check(failures, "the queued message was sent in the same turn",
                      scripted.chat_calls == 2, str(scripted.chat_calls))
    second = scripted.chat_payloads[1]["messages"]
    roles = [message["role"] for message in second]
    failures = _check(failures, "the system prompt leads the request",
                      roles[:1] == ["system"]
                      and second[0]["content"].strip() != "", str(roles))
    failures = _check(failures, "second round carries tool result, then the message",
                      roles[1:] == ["user", "assistant", "tool", "user"], str(roles))
    failures = _check(failures, "the queued text is the last user message",
                      second[-1]["content"] == "and also check the parent directory",
                      str(second[-1]))
    failures = _check(failures, "the queue is empty afterwards",
                      len(repl.queue) == 0, repl.queue.describe())
    failures = _check(failures, "the model's final answer was recorded",
                      any(message.get("role") == "assistant"
                          and message.get("content") == "final answer"
                          for message in repl.session.messages))

    # A queued message also keeps the turn alive when no tool was called.
    repl2, client2, tmp2 = _test_repl()

    class TailClient(_FakeClient):
        def chat(self, payload, timeout=None, cancel=None):
            self.chat_payloads.append(payload)
            self.chat_calls += 1
            if self.chat_calls == 1:
                repl2.queue_message("one more thing")
            return iter([{"message": {"content": f"answer {self.chat_calls}"}},
                         {"done": True}])

    repl2.client = TailClient()
    result = repl2.send("hello")
    failures = _check(failures, "a queued message triggers a follow-up round",
                      repl2.client.chat_calls == 2 and not result.cancelled,
                      str(repl2.client.chat_calls))
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    return failures


class _TuiClient(_FakeClient):
    """Fake client with a paced stream, cancellation and abort tracking."""

    def __init__(self, scripts=None, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scripts = [list(events) for events in (scripts or [])]
        self.delay = delay
        self.aborts = 0
        self.cancel_tokens = []
        self.executed = []

    def chat(self, payload, timeout=None, cancel=None):
        self.chat_payloads.append(payload)
        self.cancel_tokens.append(cancel)
        events = (self.scripts[self.chat_calls % len(self.scripts)] if self.scripts
                  else [{"message": {"content": "ok"}, "done": True}])
        self.chat_calls += 1

        def gen():
            for event in events:
                if cancel is not None and cancel.cancelled:
                    return
                if self.delay:
                    time.sleep(self.delay)
                yield event

        return gen()

    def abort_stream(self):
        self.aborts += 1
        return True


def _tui_setup(scripts=None, delay=0.0, tools=False, **kwargs):
    """A Repl ready for the Textual front end (fake client, temp store)."""
    tmp = tempfile.mkdtemp()
    store = orun.SessionStore(sessions_dir=Path(tmp) / "s", index_path=Path(tmp) / "i.json")
    args = orun.build_parser().parse_args(
        ["m1", "--render", "markdown", "--no-save"] + (["--tools"] if tools else []))
    config = dict(orun.DEFAULT_CONFIG)
    config["ui"] = "tui"
    config["tools"] = tools
    client = _TuiClient(scripts=scripts, delay=delay,
                        show_data={"model_info": {"llama.context_length": 8192},
                                   "capabilities": ["completion", "tools"]})
    session = orun.Session.create("m1", client.base)
    repl = orun.Repl(args, config, client, store, session)
    repl._preload_model = lambda: None
    return repl, client, tmp


def _transcript_text(app) -> str:
    """Plain text of every transcript widget (the app's visible conversation)."""
    from rich.console import Console

    out = []
    for child in app.query_one("#transcript").children:
        visual = child.render()
        renderable = getattr(visual, "_renderable", None)
        if renderable is None:
            out.append(str(visual).strip())
            continue
        buffer = io.StringIO()
        try:
            Console(file=buffer, width=110, force_terminal=False).print(renderable)
        except Exception as exc:  # a widget that cannot render is a test failure
            out.append(f"<render error: {exc}>")
            continue
        out.append(_plain(buffer.getvalue()))
    return "\n".join(out)


def _status_text(app) -> str:
    return orun.strip_markup(str(app.query_one("#status").render()))


async def _wait_until(pilot, predicate, timeout=8.0, step=0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(step)
    return bool(predicate())


def test_tui_app() -> int:
    """The Textual front end: panes, queue, stop, send-now, approvals, quit."""
    if importlib.util.find_spec("textual") is None:  # pragma: no cover
        print("SKIP  TUI tests (textual is not installed)")
        return 0
    import asyncio

    from ollamarun_lib import tui as tui_mod

    saved_history = os.environ.get("OLLAMA_NOHISTORY")
    os.environ["OLLAMA_NOHISTORY"] = "1"
    failures = 0
    try:
        failures += asyncio.run(_tui_basics(tui_mod))
        failures += asyncio.run(_tui_queue(tui_mod))
        failures += asyncio.run(_tui_stop_and_send_now(tui_mod))
        failures += asyncio.run(_tui_approval(tui_mod))
        failures += asyncio.run(_tui_quit(tui_mod))
    finally:
        if saved_history is None:
            os.environ.pop("OLLAMA_NOHISTORY", None)
        else:
            os.environ["OLLAMA_NOHISTORY"] = saved_history
    return failures


async def _tui_basics(tui_mod) -> int:
    from textual.widgets import Input

    from ollamarun_lib.tui import PromptInput

    failures = 0
    repl, client, tmp = _tui_setup(scripts=[[
        {"message": {"thinking": "let me think "}},
        {"message": {"content": "Hello "}},
        {"message": {"content": "**world**"}},
        {"done": True, "eval_count": 10, "eval_duration": 2_000_000_000,
         "prompt_eval_count": 5, "prompt_eval_duration": 1_000_000_000,
         "total_duration": 3_000_000_000},
    ]], delay=0.02)
    app = tui_mod.OllamarunTui(repl)
    repl.ui = tui_mod.TuiUi(app)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.3)
        status = _status_text(app)
        transcript = _transcript_text(app)
        failures = _check(failures, "banner lands in the transcript pane",
                          "ollamarun" in transcript and "m1" in transcript,
                          transcript[:200])
        failures = _check(failures, "status bar starts ready with the model",
                          "ready" in status and "m1" in status and "ctx" in status,
                          status)
        failures = _check(failures, "status bar shows the mode",
                          "think:collapsed" in status and "render:markdown" in status,
                          status)
        failures = _check(failures, "input box is always there",
                          app.query_one("#prompt").is_mounted
                          and app.query_one("#prompt").has_focus)
        failures = _check(failures, "input box sits next to the status bar",
                          app.query_one("#status").region.y
                          <= app.query_one("#prompt").region.y)

        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        ok = await _wait_until(pilot, lambda: not app._busy, timeout=8)
        failures = _check(failures, "an idle Enter starts a turn and finishes it", ok)
        roles = [message["role"] for message in repl.session.messages]
        failures = _check(failures, "the turn reached the session",
                          roles == ["user", "assistant"], str(roles))
        failures = _check(failures, "the answer was recorded",
                          repl.session.messages[-1]["content"] == "Hello **world**",
                          repl.session.messages[-1].get("content", ""))
        transcript = _transcript_text(app)
        failures = _check(failures, "the answer shows in the transcript",
                          "world" in transcript and "hi" in transcript,
                          transcript[-300:])
        failures = _check(failures, "thinking stays collapsed by default",
                          "thinking" in transcript and "let me think" not in transcript,
                          transcript[-300:])
        status = _status_text(app)
        failures = _check(failures, "status reports the finished turn",
                          await _wait_until(pilot, lambda: "done" in _status_text(app)
                                            and "tok/s" in _status_text(app), timeout=3),
                          _status_text(app))

        # Tab completion and history in the input box
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/think"
        prompt.cursor_position = len(prompt.value)
        await pilot.press("tab")
        failures = _check(failures, "Tab completes commands",
                          prompt.value == "/thinking", prompt.value)
        prompt.value = ""
        await pilot.press("up")
        failures = _check(failures, "Up recalls the last message",
                          prompt.value == "hi", prompt.value)
        await pilot.press("down")
        failures = _check(failures, "Down clears the recalled line", prompt.value == "")

        # modals: the REPL asks from a worker thread, the app shows a screen
        answers: dict[str, Any] = {}

        def ask_from_worker() -> None:
            answers["confirm"] = repl.ui.confirm("keep the context?", default=False)
            answers["ask"] = repl.ui.ask("model number, name or 'q' to quit",
                                         default="q")
            answers["approval"] = repl.ui.approval(
                [], name="run_shell", detail="$ rm -rf /",
                reason="writes outside the workspace")[0]

        worker = threading.Thread(target=ask_from_worker)
        worker.start()
        shown = await _wait_until(pilot, lambda: isinstance(app.screen,
                                                            tui_mod.ConfirmScreen),
                                  timeout=3)
        failures = _check(failures, "confirm() opens a modal from a worker", shown,
                          type(app.screen).__name__)
        await pilot.press("y")
        shown = await _wait_until(pilot, lambda: isinstance(app.screen, tui_mod.AskScreen),
                                  timeout=3)
        failures = _check(failures, "ask() opens a modal from a worker", shown)
        app.screen.query_one("#modal-input", Input).value = "42"
        await pilot.press("enter")
        shown = await _wait_until(pilot, lambda: isinstance(app.screen,
                                                            tui_mod.ApprovalScreen),
                                  timeout=3)
        failures = _check(failures, "approval() opens a modal from a worker", shown)
        await pilot.press("A")
        worker.join(5)
        failures = _check(failures, "the modal answers reach the caller",
                          answers == {"confirm": True, "ask": "42", "approval": "auto"},
                          str(answers))
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


async def _tui_queue(tui_mod) -> int:
    failures = 0
    repl, client, tmp = _tui_setup(scripts=[
        [{"message": {"content": "first answer"}}, {"done": True}],
        [{"message": {"content": "second answer"}}, {"done": True}],
        [{"message": {"content": "third answer"}}, {"done": True}],
    ], delay=0.25)
    app = tui_mod.OllamarunTui(repl)
    repl.ui = tui_mod.TuiUi(app)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.2)
        app.query_one("#prompt").value = "first"
        await pilot.press("enter")
        failures = _check(failures, "the turn is running",
                          await _wait_until(pilot, lambda: app._busy, timeout=3))

        app.query_one("#prompt").value = "second"
        await pilot.press("enter")
        await pilot.pause(0.05)
        failures = _check(failures, "Enter while busy queues the message",
                          [item.text for item in repl.queue.items] == ["second"],
                          repl.queue.describe())
        failures = _check(failures, "the status bar counts the queue",
                          "1 queued" in _status_text(app), _status_text(app))
        failures = _check(failures, "the input box is emptied after queueing",
                          app.query_one("#prompt").value == "")

        # immediate command: runs while the model is still working
        app.query_one("#prompt").value = "/thinking expanded"
        await pilot.press("enter")
        failures = _check(failures, "an immediate command runs while busy",
                          await _wait_until(pilot,
                                            lambda: repl.thinking_mode == "expanded",
                                            timeout=3), repl.thinking_mode)
        # a session-rewriting command waits for the turn to end
        app.query_one("#prompt").value = "/clear"
        await pilot.press("enter")
        await pilot.pause(0.05)
        failures = _check(failures, "other commands queue instead of running",
                          "/clear" in [item.text for item in repl.queue.items]
                          and len(repl.session.messages) >= 1,
                          f"{repl.queue.describe()} {len(repl.session.messages)}")
        failures = _check(failures, "the session still has the running turn",
                          len(repl.session.messages) >= 1, str(repl.session.messages))

        ok = await _wait_until(pilot, lambda: not app._busy and not repl.queue,
                               timeout=10)
        failures = _check(failures, "the queue drains and the turn ends", ok,
                          repl.queue.describe())
        payloads = client.chat_payloads
        failures = _check(failures, "the queued message was delivered mid-turn",
                          len(payloads) == 2
                          and any(message["role"] == "user"
                                  and message.get("content") == "second"
                                  for message in payloads[1]["messages"]),
                          str([[m.get("content") for m in p["messages"]] for p in payloads]))
        failures = _check(failures, "the assistant answer from round 1 stays in context",
                          any(message["role"] == "assistant"
                              and message.get("content") == "first answer"
                              for message in payloads[1]["messages"]))
        failures = _check(failures, "the queued command ran after the turn",
                          repl.session.messages == [], str(repl.session.messages))
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


async def _tui_stop_and_send_now(tui_mod) -> int:
    failures = 0
    long_script = [{"message": {"content": f"word{index} "}} for index in range(30)] \
        + [{"done": True}]
    repl, client, tmp = _tui_setup(scripts=[long_script, long_script,
                                            [{"message": {"content": "fine"}},
                                             {"done": True}]], delay=0.2)
    app = tui_mod.OllamarunTui(repl)
    repl.ui = tui_mod.TuiUi(app)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.2)
        app.query_one("#prompt").value = "write a long essay"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: app._busy, timeout=3)

        await pilot.press("escape")
        await pilot.pause(0.3)
        failures = _check(failures, "the first Esc only arms the stop",
                          "Esc again" in app._hint and app._busy, app._hint)
        await pilot.press("escape")
        await pilot.pause(0.3)
        failures = _check(failures, "Esc Esc cancels the turn",
                          repl.cancel_token.cancelled and client.aborts >= 1,
                          f"{repl.cancel_token.cancelled} {client.aborts}")
        ok = await _wait_until(pilot, lambda: not app._busy, timeout=5)
        failures = _check(failures, "the cancelled turn ends", ok)
        assistant = [message for message in repl.session.messages
                     if message["role"] == "assistant"]
        failures = _check(failures, "the partial answer is kept and marked cancelled",
                          assistant and assistant[-1].get("cancelled") is True
                          and assistant[-1]["content"].startswith("word0"),
                          str(assistant[-1])[:160] if assistant else "none")
        failures = _check(failures, "the status bar says it stopped",
                          "stopped" in _status_text(app), _status_text(app))

        # Ctrl+B: send the typed text now, cutting the streaming answer short
        await _wait_until(pilot, lambda: repl.state in {"done", "ready"}, timeout=4)
        app.query_one("#prompt").value = "write another long essay"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: app._busy, timeout=3)
        aborts_before = client.aborts
        app.query_one("#prompt").value = "actually, do this instead"
        await pilot.press("ctrl+b")
        await pilot.pause(0.2)
        failures = _check(failures, "Ctrl+B queues the message and cancels now",
                          client.aborts > aborts_before
                          and [item.text for item in repl.queue.items]
                          in (["actually, do this instead"], []),
                          f"{client.aborts} {repl.queue.describe()}")
        await _wait_until(pilot, lambda: not app._busy and not repl.queue, timeout=10)
        delivered = [message for message in repl.session.messages
                     if message["role"] == "user"
                     and message.get("content") == "actually, do this instead"]
        failures = _check(failures, "Ctrl+B delivers the message", bool(delivered),
                          str([m.get("content") for m in repl.session.messages])[:200])
        failures = _check(failures, "the message arrived as a running turn",
                          any(message["role"] == "assistant"
                              and message.get("content") == "fine"
                              for message in repl.session.messages),
                          str(repl.session.messages[-1])[:160])
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


async def _tui_approval(tui_mod) -> int:
    failures = 0
    marker = Path(tempfile.mkdtemp()) / "touched.txt"
    unsafe = f"touch {marker}"
    scripts = [
        [{"message": {"tool_calls": [{"function": {
            "name": "run_shell", "arguments": {"command": unsafe}}}]}},
         {"done": True}],
        [{"message": {"content": "done"}}, {"done": True}],
        [{"message": {"tool_calls": [{"function": {
            "name": "run_shell", "arguments": {"command": unsafe}}}]}},
         {"done": True}],
        [{"message": {"content": "done"}}, {"done": True}],
    ]
    repl, client, tmp = _tui_setup(scripts=scripts, delay=0.02, tools=True)
    app = tui_mod.OllamarunTui(repl)
    repl.ui = tui_mod.TuiUi(app)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.2)
        app.query_one("#prompt").value = "run the command"
        await pilot.press("enter")
        shown = await _wait_until(pilot, lambda: isinstance(app.screen,
                                                            tui_mod.ApprovalScreen),
                                  timeout=5)
        failures = _check(failures, "a risky tool call opens the approval modal", shown,
                          type(app.screen).__name__)
        failures = _check(failures, "the status bar shows the pending approval",
                          await _wait_until(pilot,
                                            lambda: "approval" in _status_text(app),
                                            timeout=2),
                          _status_text(app))
        await pilot.press("n")
        await _wait_until(pilot, lambda: not app._busy, timeout=8)
        failures = _check(failures, "denying leaves the command unrun", not marker.exists())
        denied = [message for message in repl.session.messages
                  if message["role"] == "tool"
                  and "denied by the user" in message.get("content", "")]
        failures = _check(failures, "the model is told the call was denied",
                          bool(denied), str(repl.session.messages)[-300:])

        # the same call, approved this time
        app.query_one("#prompt").value = "run it again"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, tui_mod.ApprovalScreen),
                          timeout=5)
        await pilot.press("enter")
        await _wait_until(pilot, lambda: not app._busy, timeout=8)
        failures = _check(failures, "approving runs the tool", marker.exists())
        results = [message for message in repl.session.messages
                   if message["role"] == "tool" and "exit 0" in message.get("content", "")]
        failures = _check(failures, "the tool result is fed back to the model",
                          bool(results), str(repl.session.messages)[-300:])
        failures = _check(failures, "the answer after the tool loop gets the divider",
                          "response" in _transcript_text(app),
                          _transcript_text(app)[-200:])
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(str(marker.parent), ignore_errors=True)
    return failures


async def _tui_quit(tui_mod) -> int:
    failures = 0
    long_script = [{"message": {"content": f"w{index} "}} for index in range(40)] \
        + [{"done": True}]
    repl, client, tmp = _tui_setup(scripts=[long_script], delay=0.2)
    app = tui_mod.OllamarunTui(repl)
    repl.ui = tui_mod.TuiUi(app)
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.2)
        app.query_one("#prompt").value = "long one"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: app._busy, timeout=3)
        await pilot.press("ctrl+q")
        await pilot.pause(0.2)
        failures = _check(failures, "Ctrl+Q asks before dropping a running turn",
                          app.is_running and "Ctrl+Q again" in app._hint, app._hint)
        await pilot.press("ctrl+q")
        await pilot.pause(0.4)
        failures = _check(failures, "a second Ctrl+Q quits", not app.is_running)
    shutil.rmtree(tmp, ignore_errors=True)
    return failures


def test_cancel_stops_turn() -> int:
    """Esc Esc: the stream stops, the partial answer is kept, the queue runs."""
    failures = 0
    repl, client, tmp = _test_repl()

    class CancelClient(_FakeClient):
        def chat(self, payload, timeout=None, cancel=None):
            self.chat_payloads.append(payload)
            self.chat_calls += 1

            def gen():
                yield {"message": {"content": "partial answer"}}
                for _ in range(200):
                    if cancel is not None and cancel.cancelled:
                        return
                    time.sleep(0.01)
                    yield {"message": {"content": " more"}}
                yield {"done": True}

            return gen()

    repl.client = CancelClient()
    threaded = threading.Thread(target=lambda: repl.send("write an essay"))
    threaded.start()
    time.sleep(0.2)
    failures = _check(failures, "turn is running", threaded.is_alive())
    repl.queue_message("never mind")
    repl.cancel_token.cancel()
    threaded.join(10)
    failures = _check(failures, "cancel ends the turn", not threaded.is_alive())
    assistants = [message for message in repl.session.messages
                  if message.get("role") == "assistant"]
    last_assistant = assistants[-1] if assistants else {}
    failures = _check(failures, "the partial answer is kept and marked cancelled",
                      last_assistant.get("cancelled") is True
                      and last_assistant.get("content", "").startswith("partial answer"),
                      json.dumps(repl.session.messages)[-200:])
    failures = _check(failures, "the queued message survives the stop",
                      len(repl.queue) == 1, repl.queue.describe())
    return failures


def test_tool_payloads() -> int:
    failures = 0
    session = orun.Session.create("m", "http://h")
    session.add_message("user", "hi")
    session.add_message("assistant", "", tool_calls=[
        {"function": {"name": "run_shell", "arguments": {"command": "ls"}}}])
    session.add_message("tool", "exit 0\nfile.txt", tool_name="run_shell")
    session.add_message("assistant", "there is one file")

    without = session.message_payloads()
    roles = [message["role"] for message in without]
    failures = _check(failures, "tool history is hidden when tools are off",
                      roles == ["user", "assistant"], str(roles))
    with_tools = session.message_payloads(with_tools=True)
    roles = [message["role"] for message in with_tools]
    failures = _check(failures, "tool history is replayed when tools are on",
                      roles == ["user", "assistant", "tool", "assistant"], str(roles))
    failures = _check(failures, "assistant tool_calls survive the round trip",
                      with_tools[1].get("tool_calls")
                      and with_tools[2].get("tool_name") == "run_shell",
                      json.dumps(with_tools))

    stats = {"prompt_tokens": 10, "completion_tokens": 3, "total_duration": 1.5}
    session.record_turn(stats, count_turn=False)
    session.record_turn(stats)
    failures = _check(failures, "record_turn can skip the turn counter",
                      session.stats["turns"] == 1
                      and session.stats["prompt_tokens"] == 20,
                      json.dumps(session.stats))

    payload = orun.build_chat_payload("m", [], tools=[{"type": "function"}])
    failures = _check(failures, "build_chat_payload passes tools through",
                      payload.get("tools") == [{"type": "function"}], json.dumps(payload))
    return failures


def test_format_tools_veto() -> int:
    """A response format steps the whole harness aside, not just the schemas."""
    failures = 0
    repl, _client, _tmp = _test_repl()
    repl.tools_enabled = True
    base_system = orun.default_system_prompt(with_tools=False) or ""
    tools_system = orun.default_system_prompt(with_tools=True) or ""
    failures = _check(failures, "tools on: the tool rules ride the system prompt",
                      repl.tools_active()
                      and "Web search" in tools_system
                      and repl.effective_system() == tools_system)

    repl.session.format = "json"  # what --format (or a resumed session) sets
    failures = _check(failures, "a format vetoes the active tools",
                      repl.tools_active() is False
                      and repl.active_tool_schemas() == [],
                      str(repl.active_tool_schemas()))
    failures = _check(failures, "a format drops the tool rules from the system prompt",
                      repl.effective_system() == base_system
                      and "run_python" not in (repl.effective_system() or ""),
                      (repl.effective_system() or "")[:160])
    failures = _check(failures, "the runtime sees tools off under a format",
                      repl._turn_hooks().settings.tools_enabled() is False)
    failures = _check(failures, "session_info reports tools off under a format",
                      repl.session_report()["tools"]["enabled"] is False)
    failures = _check(failures, "the status bar shows tools off under a format",
                      repl.status_snapshot().tools == "off",
                      repl.status_snapshot().tools)

    with orun.console.capture() as cap:
        payload = repl._payload(repl.session.message_payloads(with_tools=True),
                                with_tools=True)
    out = _plain(cap.get())
    failures = _check(failures, "a format leaves the tools key out",
                      "tools" not in payload and payload.get("format") == "json",
                      json.dumps(payload)[:200])
    failures = _check(failures, "a format leaves the base system message",
                      payload["messages"][0]["role"] == "system"
                      and payload["messages"][0]["content"] == base_system,
                      str(payload["messages"][0])[:160])
    failures = _check(failures, "the format/tools notice prints once",
                      "cannot be combined" in out, out)
    with orun.console.capture() as cap:
        repl._payload(repl.session.message_payloads(with_tools=True), with_tools=True)
    failures = _check(failures, "the format/tools notice does not repeat",
                      "cannot be combined" not in _plain(cap.get()))

    repl._format_tools_warned = False
    repl.args.json = True
    with orun.console.capture() as cap:
        repl._payload(repl.session.message_payloads(with_tools=True), with_tools=True)
    failures = _check(failures, "--json output stays free of the notice",
                      "cannot be combined" not in _plain(cap.get()))
    repl.args.json = False
    repl._format_tools_warned = True

    with orun.console.capture() as cap:
        orun.cmd_tools(repl, "")
    out = _plain(cap.get())
    failures = _check(failures, "/tools shows every tool off and says why",
                      "run_shell" in out and "off" in out
                      and "a response format is set" in out, out)
    with orun.console.capture() as cap:
        orun.cmd_tools(repl, "on")
    out = _plain(cap.get())
    failures = _check(failures, "/tools on explains the format wins",
                      "tools on" in out and "a response format is set" in out, out)
    with orun.console.capture() as cap:
        orun.cmd_tools(repl, "run_shell")
    failures = _check(failures, "the per-tool page shows off (format)",
                      "off (format)" in _plain(cap.get()))
    with orun.console.capture():
        orun.cmd_tools(repl, "enable run_shell")
    failures = _check(failures, "a per-tool enable notes the format veto",
                      repl._turn_hooks().settings.tools_enabled() is False)

    repl.session.format = None
    failures = _check(failures, "clearing the format restores the harness",
                      repl.tools_active()
                      and repl.active_tool_schemas() != []
                      and repl.effective_system() == tools_system
                      and repl.session_report()["tools"]["enabled"] is True
                      and repl.status_snapshot().tools == "allowlist"
                      and repl._turn_hooks().settings.tools_enabled() is True)
    return failures


def test_tool_cli_flags() -> int:
    failures = 0
    parser = orun.build_parser()
    args = parser.parse_args(["m1", "--tools"])
    failures = _check(failures, "--tools parses", args.tools is True
                      and args.no_tools is False)
    args = parser.parse_args(["m1", "--no-tools"])
    failures = _check(failures, "--no-tools parses", args.no_tools is True
                      and args.tools is False)
    try:
        parser.parse_args(["m1", "--tools", "--no-tools"])
        failures = _check(failures, "--tools/--no-tools are exclusive", False)
    except SystemExit:
        failures = _check(failures, "--tools/--no-tools are exclusive", True)
    return failures


def _fake_tts(directory: str, report: "dict | None" = None) -> Path:
    """A stand-in tts-remote that records argv/stdin, answers voices and
    `say --json` (the report defaults to a render on optiplex, playing on g3)."""
    base = Path(directory)
    script = base / "tts-remote"
    args_file = base / "tts-args.json"
    stdin_file = base / "tts-stdin.txt"
    plan = report if report is not None else {
        "ok": True, "source": "render", "host": "optiplex", "play_host": "g3"}
    script.write_text(
        "#! /usr/bin/env python3\n"
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        f"open({str(args_file)!r}, 'w').write(json.dumps(argv))\n"
        "if argv and argv[0] == 'voices':\n"
        "    print(json.dumps({'default_voice': 'nova', 'styles': ['narration'],\n"
        "                      'voices': [{'slug': 'nova', 'style': 'friendly',\n"
        "                                  'lang': 'en', 'name': 'Nova'}]}))\n"
        "    sys.exit(0)\n"
        f"open({str(stdin_file)!r}, 'w').write(sys.stdin.read())\n"
        "if '--json' in argv:\n"
        f"    print(json.dumps({plan!r}))\n")
    script.chmod(0o755)
    return script


def _fake_old_tts(directory: str) -> Path:
    """A tts-remote predating the report: rejects --json like argparse does."""
    base = Path(directory)
    script = base / "tts-remote"
    args_file = base / "tts-args.json"
    stdin_file = base / "tts-stdin.txt"
    script.write_text(
        "#! /usr/bin/env python3\n"
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        f"open({str(args_file)!r}, 'w').write(json.dumps(argv))\n"
        "if '--json' in argv:\n"
        "    sys.stderr.write('tts-remote: error: unrecognized arguments: --json\\n')\n"
        "    sys.exit(2)\n"
        f"open({str(stdin_file)!r}, 'w').write(sys.stdin.read())\n")
    script.chmod(0o755)
    return script


def test_tts_tools() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        fake = _fake_tts(tmp)
        registry = orun.default_registry(tts=True)
        ctx = orun.ToolContext(cwd=tmp, timeout=10, max_output=2000,
                               tts_remote=str(fake))
        outcome = orun.execute_tool(registry["speak"],
                                    {"text": "hello world", "voice": "v1", "lang": "pt"}, ctx)
        failures = _check(failures, "speak queues text",
                          outcome.ok and "queued" in outcome.text, outcome.text)
        recorded = json.loads((Path(tmp) / "tts-args.json").read_text())
        stdin = (Path(tmp) / "tts-stdin.txt").read_text()
        failures = _check(failures, "speak asks for the report, passes voice/lang and pipes stdin",
                          recorded == ["say", "-q", "--json", "--voice", "v1", "--lang", "pt"]
                          and stdin == "hello world",
                          f"args={recorded} stdin={stdin!r}")
        failures = _check(failures, "speak notes where the voice comes from and plays",
                          outcome.title_note == "optiplex → g3", outcome.title_note)
        outcome = orun.execute_tool(registry["speak"], {"text": "again"}, ctx)
        failures = _check(failures, "a report without voice/lang still notes the hosts",
                          outcome.title_note == "optiplex → g3", outcome.title_note)
        _fake_tts(tmp, report={"ok": True, "source": "cache",
                               "host": "kiddo", "play_host": "g3"})
        outcome = orun.execute_tool(registry["speak"], {"text": "again"}, ctx)
        failures = _check(failures, "cached speech marks the origin host",
                          outcome.title_note == "kiddo (cached) → g3", outcome.title_note)
        outcome = orun.execute_tool(registry["list_voices"], {}, ctx)
        failures = _check(failures, "list_voices formats the catalog",
                          outcome.ok and "nova" in outcome.text
                          and "friendly" in outcome.text, outcome.text)
        _fake_old_tts(tmp)
        outcome = orun.execute_tool(registry["speak"], {"text": "older tts-remote"}, ctx)
        recorded = json.loads((Path(tmp) / "tts-args.json").read_text())
        failures = _check(failures, "an older tts-remote is retried without --json",
                          outcome.ok and outcome.title_note == ""
                          and recorded == ["say", "-q"],
                          f"ok={outcome.ok} note={outcome.title_note!r} args={recorded}")
        failures = _check(failures, "tts schemas are advertised",
                          {schema["function"]["name"] for schema in
                           orun.tool_schemas(registry)} >= {"speak", "list_voices"})
        failures = _check(failures, "tts schemas can be disabled",
                          all(schema["function"]["name"] not in {"speak", "list_voices"}
                              for schema in orun.tool_schemas(registry,
                                                              {"speak", "list_voices"})))
        failures = _check(failures, "tts tools are skipped entirely when off",
                          "speak" not in orun.default_registry(tts=False))
    return failures


def test_speech_hosts_note() -> int:
    """The `X → Y` note: formats, the box title, the transcript, the payload."""
    failures = 0
    failures = _check(failures, "a render report reads host → play host",
                      orun.speech_hosts('{"ok": true, "source": "render", '
                                        '"host": "optiplex", "play_host": "g3"}')
                      == "optiplex → g3")
    failures = _check(failures, "a disk hit marks the origin as (cached)",
                      orun.speech_hosts('{"source": "cache", "host": "kiddo", '
                                        '"play_host": "g3"}') == "kiddo (cached) → g3")
    failures = _check(failures, "a report without both hosts yields no note",
                      orun.speech_hosts('{"source": "render", "host": null, '
                                        '"play_host": "g3"}') == ""
                      and orun.speech_hosts('{"source": "cache", "host": "g3"}') == "")
    failures = _check(failures, "garbage or empty output yields no note",
                      orun.speech_hosts("queued\n") == "" and orun.speech_hosts("") == "")

    with orun.console.capture() as cap:
        orun.console.print(orun.tool_result_panel("speak", text="queued 5 chars for speech",
                                                  note="optiplex → g3", duration=0.4))
    out = _plain(cap.get())
    failures = _check(failures, "the result box title carries the note",
                      "tool · speak · ok · optiplex → g3 · 0.40s" in out, out)
    with orun.console.capture() as cap:
        orun.console.print(orun.tool_result_panel("speak", text="queued 5 chars for speech",
                                                  duration=0.4))
    out = _plain(cap.get())
    failures = _check(failures, "no note, no title change",
                      "tool · speak · ok · 0.40s" in out, out)

    session = orun.Session.create("m", "http://h")
    session.add_message("assistant", "", tool_calls=[
        {"function": {"name": "speak", "arguments": {"text": "hi"}}}])
    session.add_message("tool", "queued 2 chars for speech",
                        tool_name="speak", tool_note="optiplex → g3")
    stored = orun.Session.from_dict(json.loads(json.dumps(session.to_dict())))
    failures = _check(failures, "the note survives a session round trip",
                      stored.messages[1].get("tool_note") == "optiplex → g3")
    payloads = session.message_payloads(with_tools=True)
    tool_payload = [payload for payload in payloads if payload["role"] == "tool"][0]
    failures = _check(failures, "the note never reaches the model",
                      "tool_note" not in tool_payload, json.dumps(tool_payload))
    with orun.console.capture() as cap:
        orun.render_transcript(session, orun.console)
    out = _plain(cap.get())
    failures = _check(failures, "the transcript replays the titled box",
                      "optiplex → g3" in out, out)

    # End to end through the REPL tool loop: the model calls speak and the
    # live box (and the stored message) carry the note.
    with tempfile.TemporaryDirectory() as tmp:
        fake = _fake_tts(tmp)
        repl, client, _ = _test_repl(config_extra={"tts_remote": str(fake)})
        repl._build_prompt_session = lambda: None
        client.capability_list = ["completion", "tools"]
        client.chat_script = [
            [{"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "speak", "arguments": {"text": "hello there"}}}]},
              "done": False},
             {"message": {"role": "assistant", "content": ""}, "done": True}],
            [{"message": {"role": "assistant", "content": "spoken"}, "done": False},
             {"message": {"role": "assistant", "content": ""}, "done": True}],
        ]
        repl.session.add_message("user", "say something")
        with orun.console.capture() as cap:
            repl._generate()
        out = _plain(cap.get())
        tool_message = [message for message in repl.session.messages
                        if message.get("role") == "tool"][0]
        failures = _check(failures, "the REPL stores the note on the tool message",
                          tool_message.get("tool_note") == "optiplex → g3",
                          json.dumps(tool_message))
        failures = _check(failures, "the live box title carries the note",
                          "tool · speak · ok · optiplex → g3" in out, out)
    return failures


def test_tts_command() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        fake = _fake_tts(tmp)
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain"])
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"])
        session = orun.Session.create("m1", client.base)
        config = dict(orun.DEFAULT_CONFIG, tts_remote=str(fake))
        repl = orun.Repl(args, config, client, store, session)
        repl._build_prompt_session = lambda: None
        repl.tool_ctx.tts_remote = str(fake)

        with orun.console.capture() as cap:
            orun.cmd_tts(repl, "")
        out = _plain(cap.get())
        failures = _check(failures, "/tts status reports settings",
                          "auto-speak" in out and "voice" in out
                          and "speak" in out, out)

        with orun.console.capture():
            orun.cmd_tts(repl, "on")
        failures = _check(failures, "/tts on enables auto-speak",
                          repl.tts_auto is True and session.tts_auto is True)
        with orun.console.capture():
            orun.cmd_tts(repl, "off")
        failures = _check(failures, "/tts off disables auto-speak",
                          repl.tts_auto is False and session.tts_auto is False)

        with orun.console.capture():
            orun.cmd_tts(repl, "voice narrator")
        with orun.console.capture():
            orun.cmd_tts(repl, "lang en")
        failures = _check(failures, "/tts voice/lang persist in the session",
                          repl.tts_voice == "narrator" and session.tts_voice == "narrator"
                          and session.tts_lang == "en")

        with orun.console.capture():
            orun.cmd_tts(repl, "tools off")
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tts tools off hides speak/list_voices",
                          "speak" not in names and "list_voices" not in names,
                          str(names))
        with orun.console.capture():
            orun.cmd_tts(repl, "tools on")
        names = [schema["function"]["name"] for schema in repl.active_tool_schemas()]
        failures = _check(failures, "/tts tools on restores them",
                          "speak" in names and "list_voices" in names, str(names))

        with orun.console.capture() as cap:
            orun.cmd_tts(repl, "voices")
        failures = _check(failures, "/tts voices lists the catalog",
                          "nova" in _plain(cap.get()))
    return failures


def test_tts_auto_speak() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        fake = _fake_tts(tmp)
        store = orun.SessionStore(sessions_dir=Path(tmp) / "s",
                                  index_path=Path(tmp) / "i.json")
        args = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save",
                                               "--speak"])
        events = [{"message": {"role": "assistant", "content": "spoken answer"},
                   "done": True}]
        client = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                             chat_script=[events])
        repl = orun.Repl(args, dict(orun.DEFAULT_CONFIG, tts_remote=str(fake)), client, store,
                         orun.Session.create("m1", client.base))
        repl._build_prompt_session = lambda: None
        repl.tool_ctx.tts_remote = str(fake)
        repl.session.add_message("user", "say something")
        with orun.console.capture():
            repl._generate()
        recorded = json.loads((Path(tmp) / "tts-args.json").read_text())
        stdin = (Path(tmp) / "tts-stdin.txt").read_text()
        failures = _check(failures, "--speak reads the final answer aloud",
                          recorded == ["say", "-q", "--json", "--markdown"]
                          and stdin == "spoken answer", f"{recorded} {stdin!r}")

        quiet = orun.build_parser().parse_args(["m1", "--render", "plain", "--no-save"])
        client2 = _FakeClient(models=["m1"], capabilities=["completion", "tools"],
                              chat_script=[events])
        repl2 = orun.Repl(quiet, dict(orun.DEFAULT_CONFIG), client2, store,
                          orun.Session.create("m1", client2.base))
        repl2._build_prompt_session = lambda: None
        repl2.tool_ctx.tts_remote = str(fake)
        (Path(tmp) / "tts-args.json").unlink()
        with orun.console.capture():
            repl2._generate()
        failures = _check(failures, "no auto-speak when tts is off",
                          not (Path(tmp) / "tts-args.json").exists())

        failures = _check(failures, "tts config defaults exist",
                          orun.DEFAULT_CONFIG.get("tts_auto") is False
                          and orun.DEFAULT_CONFIG.get("tts_remote") == "tts-remote"
                          and orun.DEFAULT_CONFIG.get("tts_tools") is True)
    return failures


def test_client_http() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name} {detail}")
            failures += 1

    client = orun.OllamaClient("127.0.0.1:11434")

    def ok_handler(method, url, payload, stream):
        if url.endswith("/api/version"):
            return _FakeResponse(200, {"version": "0.9.0"})
        if url.endswith("/api/tags"):
            return _FakeResponse(200, {"models": [{"name": "m1"}, {"name": "m2:latest"}]})
        if url.endswith("/api/ps"):
            return _FakeResponse(200, {"models": [{"name": "m1"}]})
        if url.endswith("/api/show"):
            return _FakeResponse(200, {"capabilities": ["completion"]})
        raise AssertionError(f"unexpected url {url}")

    client.http = _FakeSession(ok_handler)
    check("version()", client.version() == "0.9.0")
    check("tags()", [m["name"] for m in client.tags()] == ["m1", "m2:latest"])
    check("resident_models()", client.resident_models() == ["m1"])
    check("is_resident() canonicalizes tags", client.is_resident("m1:latest") is True)
    check("is_resident() false for an absent model", client.is_resident("other") is False)
    check("capabilities()", client.capabilities("m1") == ["completion"])
    check("model_exists() true", client.model_exists("m1") is True)

    def error_handler(method, url, payload, stream):
        return _FakeResponse(404, payload={"error": "model not found"}, text="not found")

    client.http = _FakeSession(error_handler)
    try:
        client.show("missing")
        check("HTTP error raises OllamaError", False)
    except orun.OllamaError as exc:
        check("HTTP error raises OllamaError", "model not found" in str(exc))
    check("model_exists() false on error", client.model_exists("missing") is False)

    def bad_json_handler(method, url, payload, stream):
        return _FakeResponse(200, payload=None, text="not json")

    client.http = _FakeSession(bad_json_handler)
    try:
        client.version()
        check("invalid JSON raises OllamaError", False)
    except orun.OllamaError:
        check("invalid JSON raises OllamaError", True)

    class _ExplodingSession:
        def request(self, *a, **k):
            raise requests.RequestException("boom")

    client.http = _ExplodingSession()
    try:
        client.version()
        check("network failure raises OllamaError", False)
    except orun.OllamaError as exc:
        check("network failure raises OllamaError", "cannot reach Ollama" in str(exc))

    def stream_handler(method, url, payload, stream):
        check("chat() streams with stream=True", payload.get("stream") is True)
        return _FakeResponse(200, lines=[b'{"message": {"content": "a"}}', b"",
                                         b'{"done": true}'])

    client.http = _FakeSession(stream_handler)
    events = list(client.chat({"model": "m1", "messages": []}))
    check("chat() parses NDJSON and skips blank lines",
          events == [{"message": {"content": "a"}}, {"done": True}])

    def bad_chunk_handler(method, url, payload, stream):
        return _FakeResponse(200, lines=[b"not-json"])

    client.http = _FakeSession(bad_chunk_handler)
    try:
        list(client.stream_ndjson("/api/chat", {}))
        check("bad NDJSON chunk raises OllamaError", False)
    except orun.OllamaError:
        check("bad NDJSON chunk raises OllamaError", True)

    def unload_handler(method, url, payload, stream):
        check("unload() posts keep_alive=0", payload == {"model": "m1", "keep_alive": 0})
        return _FakeResponse(200, {})

    client.http = _FakeSession(unload_handler)
    client.unload("m1")
    return failures


if __name__ == "__main__":
    live = "--live" in sys.argv[1:]
    if live:
        print("--live: also running the real-backend smoke test")
    # The API-layer tests live in test_ollamarun_serve.py and import this
    # module's shared harness; registering the running module under its
    # import name keeps that file from re-parsing (and re-executing) this
    # one when the canonical command pulls both halves together.
    sys.modules.setdefault("test_ollamarun", sys.modules[__name__])
    import test_ollamarun_serve
    total = (test_math() + test_parsing_helpers() + test_stats_helpers()
             + test_resolve_show_stats()
             + test_file_mentions() + test_images() + test_session_store()
             + test_session_payloads() + test_chat_payload()
             + test_tool_definitions() + test_prompts() + test_system_prompt()
             + test_normalize_argv() + test_renderer_plain()
             + test_renderer_feed_parity()
             + test_renderer_stop_keys()
             + test_stream_error_reporting()
             + test_thinking_display() + test_math_layout()
             + test_render_content() + test_tps_estimator()
             + test_unload_housekeeping() + test_preload_model()
             + test_preload_shows_ps_before_first_response()
             + test_preload_loading_rprompt()
             + test_generate_waits_for_pending_preload_before_showing_ps()
             + test_empty_enter_sends_pending_images() + test_context_length()
             + test_naive_title() + test_chat_bubbles()
             + test_render_transcript_bubbles() + test_tool_colors()
             + test_ps_table_and_cmd_ps()
             + test_maybe_show_ps() + test_resolve_num_ctx()
             + test_context_usage()
             + test_cmd_num_ctx() + test_ctxfit()
             + test_ctxfit_outcome() + test_ctxfit_cancel_progress()
             + test_fit_service()
             + test_cmd_num_ctx_max_gpu()
             + test_ram_helpers() + test_cpu_fit() + test_cmd_num_ctx_max_cpu()
             + test_model_profile()
             + test_emit_and_cli_functions()
             + test_machine_output_verbatim()
             + test_build_parser_flags() + test_parser_intermixed()
             + test_background_cli()
             + test_remote_hosts()
             + test_pull_and_disk_helpers() + test_ensure_model()
             + test_model_sort() + test_fuzzy_sort() + test_model_picker_sort()
             + test_model_command_sort() + test_sort_flag()
             + test_gpu_layers() + test_slash_completer()
             + test_prompt_toolkit_fallback() + test_textual_optional()
             + test_tools_policy() + test_tool_execution() + test_web_tools()
             + test_pdf_tool()
             + test_file_tools() + test_session_model_tools()
             + test_system_status_tool() + test_chem_tools()
             + test_environment_tools()
             + test_tool_loop() + test_tool_loop_separation()
             + test_tool_commands() + test_approval_rules()
             + test_tool_payloads() + test_format_tools_veto() + test_tool_cli_flags()
             + test_tts_tools() + test_speech_hosts_note()
             + test_tts_command() + test_tts_auto_speak()
             + test_client_http() + test_cmd_image()
             + test_cli_image_errors()
             + test_message_queue() + test_status_format() + test_cancel_token()
             + test_context_counter() + test_tps_footer()
             + test_stream_cancel() + test_boundary_delivery()
             + test_cancel_stops_turn() + test_tui_app()
             + test_live_stream_final_frame_is_complete()
             + test_ollamarun_serve.run_all(live=live))
    if total:
        print(f"\n{total} failure(s)")
        sys.exit(1)
    print("\nAll tests passed.")
