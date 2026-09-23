"""Regression tests for ollamarun helpers, sessions and payloads.

Run: python3 test_ollamarun.py
"""
import base64
import contextlib
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

import requests
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

        if orun.export_markdown(loaded).startswith("# first") \
                and '"schema": 1' in orun.export_json(loaded) \
                and "<html" in orun.export_html(loaded):
            print("PASS  exports md/json/html")
        else:
            print("FAIL  exports")
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
            ("-c with session title", ["-c", "named session", "hi"],
             ["-c", "named session", "hi"]),
            ("-c with prefix", ["-c", session.id[:12], "hi"],
             ["-c", session.id[:12], "hi"]),
            ("-c with id-like miss", ["-c", "20260101-000000-dead", "hi"],
             ["-c", "20260101-000000-dead", "hi"]),
            ("-c followed by option", ["-c", "--render", "plain", "hi"],
             ["-c", "--render", "plain", "hi"]),
        ]
        for name, argv, expected in cases:
            got = orun.normalize_argv(argv, store)
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
    check("expanded panel", "Ctrl+T to re-read" in expanded and "deep reasoning" in expanded)
    result.content = "The answer."
    with_content = render_text(renderer._live_view(result))
    check("content with expanded panel",
          "The answer." in with_content and "Ctrl+T to re-read" in with_content)
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
        renderer._pager_request = False
        os.write(write_fd, b"x" + bytes([orun.CTRL_T]))
        os.close(write_fd)
        changed = renderer._poll_keys(read_fd)
        check("Ctrl+T poll expands and requests pager",
              changed and renderer.thinking_mode == "expanded"
              and renderer._pager_request is True)
    finally:
        os.close(read_fd)

    read_fd, write_fd = os.pipe()
    try:
        renderer._pager_request = False
        os.write(write_fd, bytes([orun.CTRL_T]))
        os.close(write_fd)
        changed = renderer._poll_keys(read_fd)
        check("Ctrl+T poll collapses without pager",
              changed and renderer.thinking_mode == "collapsed"
              and renderer._pager_request is False)
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
    check("tps token count", estimator.tokens, 10)

    early = orun.TpsEstimator(window=5.0)
    early.add(1, now=100.0)
    early.add(1, now=100.2)
    check("tps cumulative fallback", round(early.rate(now=100.2), 3), 10.0)

    stalling = orun.TpsEstimator(window=5.0)
    for tick in range(5):
        stalling.add(1, now=float(tick))
    check("tps stall falls back to cumulative",
          round(stalling.rate(now=30.0), 3), round(5 / 30, 3))
    return failures


class _FakeClient:
    def __init__(self, resident=(), models=(), ps_entries=(), show_data=None):
        self.base = "http://fake:11434"
        self.resident = list(resident)
        self.models = list(models)
        self.ps_entries = list(ps_entries)
        self.show_data = show_data
        self.unloaded = []
        self.loaded = []
        self.chat_payloads = []

    def is_resident(self, model):
        return model in self.resident

    def unload(self, model):
        self.unloaded.append(model)

    def load(self, model, *, keepalive=None):
        self.loaded.append(model)

    def chat(self, payload, timeout=None):
        self.chat_payloads.append(payload)
        return iter([{"message": {"content": "ok"}, "done": True}])

    def tags(self):
        return [{"name": name} for name in self.models]

    def show(self, model):
        return self.show_data if self.show_data is not None else {}

    def ps(self):
        return list(self.ps_entries)

    def model_exists(self, model):
        return model in self.models


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


def test_generate_waits_for_preload_before_showing_ps() -> int:
    """If the user sends a message before the background preload finishes,
    _generate() should wait for that same load instead of firing a second,
    concurrent one via chat() -- and `ollama ps` should show up right then,
    not only after the response finishes generating."""
    failures = 0
    original_force_terminal = orun.console._force_terminal
    original_file = orun.console.file
    orun.console._force_terminal = True
    buffer = io.StringIO()
    orun.console.file = buffer

    class _SlowLoadClient(_FakeClient):
        def load(self, model, *, keepalive=None):
            time.sleep(0.15)
            self.resident.append(model)
            super().load(model, keepalive=keepalive)

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

            repl._preload_model()
            still_loading = (repl._preload_thread is not None
                              and repl._preload_thread.is_alive())
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
        with orun.console.capture() as cap:
            orun.cli_list(args_json, store)
        try:
            data = json.loads(cap.get())
        except ValueError:
            data = None
        if isinstance(data, list) and data and data[0]["title"] == "alpha":
            print("PASS  cli_list --json emits parseable JSON")
        else:
            print(f"FAIL  cli_list --json: {cap.get()!r}")
            failures += 1

        args_csv = orun.build_parser().parse_args(["--list", "--csv"])
        with orun.console.capture() as cap:
            orun.cli_list(args_csv, store)
        out = cap.get()
        if out.splitlines()[0].split(",")[0] == "id" and "alpha" in out:
            print("PASS  cli_list --csv emits a CSV header")
        else:
            print(f"FAIL  cli_list --csv: {out!r}")
            failures += 1

        args_search = orun.build_parser().parse_args(["--search", "find", "--markdown"])
        with orun.console.capture() as cap:
            code = orun.cli_search(args_search, store)
        out = cap.get()
        if code == 0 and out.startswith("| id |") and "alpha" in out:
            print("PASS  cli_search --markdown emits a markdown table")
        else:
            print(f"FAIL  cli_search markdown: {out!r}")
            failures += 1

        args_export = orun.build_parser().parse_args(
            ["--export", session.id, "--export-format", "json"])
        with orun.console.capture() as cap:
            code = orun.cli_export(args_export, store)
        if code == 0 and '"title": "alpha"' in cap.get():
            print("PASS  cli_export --export-format json")
        else:
            print(f"FAIL  cli_export json: {cap.get()!r}")
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


def test_build_parser_flags() -> int:
    failures = 0
    parser = orun.build_parser()

    args = parser.parse_args(["m1", "--num-ctx", "4096"])
    if args.num_ctx == 4096:
        print("PASS  --num-ctx parses to an int")
    else:
        print(f"FAIL  --num-ctx: {args.num_ctx!r}")
        failures += 1

    if parser.parse_args(["m1"]).num_ctx is None:
        print("PASS  --num-ctx defaults to None")
    else:
        print("FAIL  --num-ctx default")
        failures += 1

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
    total = (test_math() + test_parsing_helpers() + test_stats_helpers()
             + test_file_mentions() + test_images() + test_session_store()
             + test_session_payloads() + test_chat_payload()
             + test_normalize_argv() + test_renderer_plain()
             + test_thinking_display() + test_math_layout()
             + test_render_content() + test_tps_estimator()
             + test_unload_housekeeping() + test_preload_model()
             + test_preload_shows_ps_before_first_response()
             + test_generate_waits_for_preload_before_showing_ps()
             + test_empty_enter_sends_pending_images() + test_context_length()
             + test_naive_title() + test_chat_bubbles()
             + test_render_transcript_bubbles() + test_ps_table_and_cmd_ps()
             + test_maybe_show_ps() + test_resolve_num_ctx()
             + test_cmd_num_ctx() + test_emit_and_cli_functions()
             + test_build_parser_flags() + test_slash_completer()
             + test_client_http() + test_cmd_image()
             + test_live_stream_final_frame_is_complete())
    if total:
        print(f"\n{total} failure(s)")
        sys.exit(1)
    print("\nAll tests passed.")
