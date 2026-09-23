"""Regression tests for ollamarun helpers, sessions and payloads.

Run: python3 test_ollamarun.py
"""
import base64
import importlib.machinery
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from rich.console import Console

MODULE = Path(__file__).parent / "ollamarun"

loader = importlib.machinery.SourceFileLoader("ollamarun", str(MODULE))
spec = importlib.util.spec_from_loader("ollamarun", loader)
orun = importlib.util.module_from_spec(spec)
sys.modules["ollamarun"] = orun
loader.exec_module(orun)

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
    def __init__(self, resident=(), models=()):
        self.base = "http://fake:11434"
        self.resident = list(resident)
        self.models = list(models)
        self.unloaded = []

    def is_resident(self, model):
        return model in self.resident

    def unload(self, model):
        self.unloaded.append(model)

    def tags(self):
        return [{"name": name} for name in self.models]

    def show(self, model):
        return {}

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


if __name__ == "__main__":
    total = (test_math() + test_parsing_helpers() + test_stats_helpers()
             + test_file_mentions() + test_images() + test_session_store()
             + test_session_payloads() + test_chat_payload()
             + test_normalize_argv() + test_renderer_plain()
             + test_thinking_display() + test_math_layout()
             + test_render_content() + test_tps_estimator()
             + test_unload_housekeeping() + test_context_length()
             + test_naive_title())
    if total:
        print(f"\n{total} failure(s)")
        sys.exit(1)
    print("\nAll tests passed.")
