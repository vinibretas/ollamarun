"""CLI: `build_parser()` (the full argparse surface, mirroring every
`ollama run` flag), `resolve_model`/`resolve_session` (figuring out what
model/session a given invocation means, including the `-c`/`-r`/
`--session` disambiguation in `normalize_argv`), and `main()` as the entry
point.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from rich.markup import escape

from .client import OllamaClient, OllamaError
from .constants import (
    CONFIG_PATH,
    DEFAULT_HOST,
    DEFAULT_TIMEOUT,
    EMBED_MODEL_HINTS,
    LOG_PATH,
    REMOTE_PROBE_TIMEOUT,
    VERSION,
    console,
    set_tool_colors,
)
from .emit import cli_delete, cli_export, cli_list, cli_search, cli_show
from .helpers import (
    is_local_host,
    load_config,
    parse_bool,
    parse_format,
    parse_options,
    parse_think,
    remote_node,
    resolve_remote,
    setup_logging,
    shortcut,
    tailscale_peer_state,
)
from .images import encode_image
from .ctxfit import (
    FitRequest,
    machine_state_for,
    max_gpu_context,
    parse_num_ctx_arg,
)
from .interactive import (
    ensure_model,
    gpu_layers_label,
    model_block_count,
    parse_gpu_layers_spec,
    pick_model,
    pick_session,
    resolve_gpu_layers,
)
from .repl import Repl, resolve_num_ctx
from .run import run_embed, run_once
from .session import Session, SessionStore


def build_parser() -> argparse.ArgumentParser:
    epilog = """\
interactive commands:
  /help                 all commands (/? shortcuts for key bindings)
  /set parameter NAME V session parameter (temperature, num_ctx, seed, …)
  /set system TEXT      system prompt
  /show [topic]         model info|license|modelfile|parameters|system|template
  /load [MODEL|SESSION] switch model or resume a saved session
  /save NAME            create an Ollama model from this session
  /pull [MODEL]         pull a model onto this host (updates if present)
  /clear /new           start fresh context or a brand new session
  /undo /retry /edit    fix or regenerate the last exchange
  /summarize /context   manage the context window
  /tools [on|off]       tool harness, approval rules and per-tool enable/disable
                        (`/tools policy auto|allowlist|ask`, `/tools rules`)
  /tts [on|off]         auto-speak answers via tts-remote; voice/lang/voices,
                        and `/tts tools on|off` for the model's tts access
  /thinking [MODE]      thinking display: show|collapse|hide|toggle|full
  /tps [on|off]         live tokens/s footer
  /ps                   resident models (ollama ps)
  /num-ctx [N | --max-gpu [--probe]]
                        context window: N, or the largest that keeps the GPU
                        placement (--probe verifies by loading the model)
  /gpu-layers [SPEC]    GPU layers: N, P%, +N/-N, all, cpu or auto
  /image PATH           attach an image to the next message
  /export [FILE]        export mid-session as md/json/html

while a response streams:
  Ctrl+T                expand the thinking panel (press again to collapse)

model housekeeping:
  /model and /load unload the previous model before switching; leaving the
  REPL unloads the session model (--keep-loaded to keep it, --unload to also
  unload after a one-shot run).

environment:
  OLLAMA_HOST           server address (default 127.0.0.1:11434)
  OLLAMA_EDITOR         editor for Ctrl+G / /edit (falls back to $EDITOR)
  OLLAMA_NOHISTORY      do not keep prompt input history

remote:
  --remote NAME         run on a tailnet machine: pi, kiddo, g3, optiplex7070
                        (alias optiplex) or guri, or any host[:port]. The name
                        is resolved through tailscale; models load and unload
                        there while sessions, files and tools stay local.
                        Resuming a session reconnects to the machine it was
                        created on unless --host/--remote override it.

sessions live in ~/etc/data/ollamarun; config in ~/etc/config_files/ollamarun.json
"""
    parser = argparse.ArgumentParser(
        prog="ollamarun", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a model (full ollama run wrapper with rendering and saved sessions)",
        epilog=epilog)
    parser.add_argument("model", nargs="?", help="model name (omit to pick interactively)")
    parser.add_argument("prompt", nargs="*", help="prompt for a one-shot run "
                        "(otherwise reads stdin or starts a REPL)")

    ollama = parser.add_argument_group("ollama-compatible flags")
    ollama.add_argument("--dimensions", type=int, metavar="int",
                        help="truncate output embeddings to specified dimension "
                             "(embedding models only)")
    ollama.add_argument("--format", metavar="string",
                        help="response format (e.g. json, or an inline JSON schema)")
    ollama.add_argument("--hidethinking", action="store_true",
                        help="hide thinking output (if provided)")
    ollama.add_argument("--insecure", action="store_true",
                        help="use an insecure registry (skip TLS verification)")
    ollama.add_argument("--keepalive", metavar="string",
                        help="duration to keep a model loaded (e.g. 5m)")
    ollama.add_argument("--nowordwrap", action="store_true",
                        help="don't wrap words to the next line automatically")
    ollama.add_argument("--think", nargs="?", const="true", metavar="string",
                        help='enable thinking mode: true/false or high/medium/low '
                             'for supported models')
    ollama.add_argument("--truncate", nargs="?", const="true", metavar="bool",
                        help="for embedding models: truncate inputs exceeding context "
                             "length (default: true). Set --truncate=false to error instead")
    ollama.add_argument("--verbose", action="store_true",
                        help="show timings for response")

    wrapper = parser.add_argument_group("wrapper flags")
    host_group = wrapper.add_mutually_exclusive_group()
    host_group.add_argument("--host", metavar="HOST",
                            help=f"ollama server, e.g. 127.0.0.1:11434 (default: "
                                 f"OLLAMA_HOST or {DEFAULT_HOST})")
    host_group.add_argument("--remote", metavar="NAME",
                            help="run on a tailnet machine (pi, kiddo, g3, "
                                 "optiplex7070 or optiplex, guri) or any host[:port]; "
                                 "models load and unload there")
    wrapper.add_argument("--pull", action="store_true",
                         help="pull the model if it is not installed, without asking "
                              "(shows free disk space and progress on a terminal)")
    wrapper.add_argument("-m", "--model", dest="model_flag", metavar="MODEL",
                         help="model name (same as the positional MODEL)")
    wrapper.add_argument("--system", metavar="TEXT", help="system prompt")
    wrapper.add_argument("--system-file", metavar="FILE",
                         help="read the system prompt from a file")
    wrapper.add_argument("--option", action="append", metavar="KEY=VALUE",
                         help="model option, repeatable (e.g. -o temperature=0.7)")
    wrapper.add_argument("--num-ctx", metavar="N|max-gpu",
                         help="context window size; capped to the model's max "
                              "(from /api/show) with a warning. 'max-gpu' picks "
                              "the largest context that still keeps the model "
                              "on the GPU (see /num-ctx --help)")
    wrapper.add_argument("--max-gpu-probe", action="store_true",
                         help="verify --num-ctx max-gpu by loading the model at "
                              "candidate contexts and reading /api/ps back")
    wrapper.add_argument("--gpu-layers", metavar="SPEC",
                         help="GPU layers for the model: N (absolute), P%% of its "
                              "layers, +N/-N (adjust the current value), all, cpu "
                              "or auto (default)")
    wrapper.add_argument("--options-json", metavar="JSON",
                         help="model options as a JSON object")
    wrapper.add_argument("--image", action="append", metavar="PATH",
                         help="attach an image (vision models), repeatable")
    wrapper.add_argument("--render", choices=["auto", "markdown", "plain", "raw"],
                         help="response rendering (default: markdown on a tty)")
    wrapper.add_argument("--show-thinking", action="store_true",
                         help="start with the thinking panel expanded (Ctrl+T toggles)")
    wrapper.add_argument("--no-math", action="store_true",
                         help="disable LaTeX→Unicode math conversion")
    wrapper.add_argument("--no-tps", action="store_true",
                         help="hide the live tokens/s footer")
    unload_group = wrapper.add_mutually_exclusive_group()
    unload_group.add_argument("--unload", action="store_true",
                              help="unload the model when the program ends "
                                   "(one-shot included)")
    unload_group.add_argument("--keep-loaded", action="store_true",
                              help="never unload the model on exit (overrides "
                                   "the unload_on_exit config default)")
    wrapper.add_argument("--pager", action="store_true",
                         help="page long transcripts")
    tools_group = wrapper.add_mutually_exclusive_group()
    tools_group.add_argument("--tools", action="store_true",
                             help="enable the tool harness (default: on; the model "
                                  "needs tool-calling support)")
    tools_group.add_argument("--no-tools", action="store_true",
                             help="disable the tool harness for this run")
    speak_group = wrapper.add_mutually_exclusive_group()
    speak_group.add_argument("--speak", action="store_true",
                             help="speak every final answer through tts-remote "
                                  "(fire-and-forget)")
    speak_group.add_argument("--no-speak", action="store_true",
                             help="never auto-speak answers (config tts_auto)")
    wrapper.add_argument("--no-save", action="store_true",
                         help="do not persist this session")
    wrapper.add_argument("--raw", action="store_true",
                         help="write only the response text to stdout")
    wrapper.add_argument("--timeout", type=float, metavar="SECONDS",
                         help=f"request timeout (default: {DEFAULT_TIMEOUT})")
    wrapper.add_argument("--config", metavar="FILE",
                         help=f"config file (default: {CONFIG_PATH})")
    wrapper.add_argument("--debug", action="store_true",
                         help=f"log to {LOG_PATH}")
    wrapper.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

    sessions = parser.add_argument_group("session flags")
    sessions.add_argument("-c", "--continue", dest="continue_", nargs="?",
                          const="__latest__", metavar="SESSION_ID",
                          help="continue the most recent session, or a given one")
    sessions.add_argument("-r", "--resume", action="store_true",
                          help="browse and resume a saved session")
    sessions.add_argument("--session", metavar="NAME",
                          help="continue or create a named session")
    sessions.add_argument("--new", action="store_true",
                          help="force a new session even with --continue/--session")
    sessions.add_argument("--title", metavar="TEXT", help="title for a new session")
    sessions.add_argument("--tag", action="append", metavar="TAG",
                          help="tag for the session, repeatable")
    sessions.add_argument("--auto-title", action="store_true",
                          help="ask the model for a title after the first turn")

    inspect = parser.add_argument_group("session inspection")
    inspect.add_argument("--list", action="store_true", help="list saved sessions")
    inspect.add_argument("--search", metavar="TEXT",
                         help="search messages across saved sessions")
    inspect.add_argument("--show", metavar="SESSION_ID",
                         help="render a saved transcript")
    inspect.add_argument("--export", metavar="SESSION_ID",
                         help="export a saved session (see --export-format/--out)")
    inspect.add_argument("--export-format", choices=["md", "json", "html"], default="md",
                         help="export format (default: md)")
    inspect.add_argument("--delete", metavar="SESSION_ID",
                         help="move a session to the trash")
    inspect.add_argument("-y", "--yes", action="store_true",
                         help="skip confirmations (--delete, tool runs)")
    inspect.add_argument("--reindex", action="store_true",
                         help="rebuild the session index")
    inspect.add_argument("--json", action="store_true",
                         help="JSON output with --list/--search, or the final API "
                              "response as JSON when chatting")
    inspect.add_argument("--csv", action="store_true",
                         help="CSV output with --list/--search")
    inspect.add_argument("--markdown", action="store_true",
                         help="markdown table output with --list/--search")
    inspect.add_argument("--out", metavar="FILE",
                         help="write --list/--search/--export output to FILE")
    return parser


def resolve_model(args: argparse.Namespace, config: dict[str, Any],
                  client: OllamaClient, session: Optional[Session]) -> Optional[str]:
    explicit = args.model_flag or args.model
    if explicit and session is not None:
        session.model = explicit
    model = explicit or (session.model if session else None) or config.get("model")
    if not model:
        if console.is_terminal:
            model = pick_model(client, None)
        if not model:
            console.print("[err]no model given[/err] — pass MODEL or configure one")
            return None
    return model


class SessionError(RuntimeError):
    pass


def resolve_session(args: argparse.Namespace, config: dict[str, Any],
                    client: OllamaClient, store: SessionStore) -> Session:
    session: Optional[Session] = None
    if args.session:
        session = store.resolve(args.session)
        if session is None:
            session = Session.create("", client.base, title=args.session)
    elif args.continue_ is not None:
        if args.continue_ == "__latest__":
            session = store.latest()
        else:
            session = store.resolve(args.continue_)
        if session is None:
            raise SessionError(f"no such session: {args.continue_}")
    elif args.resume and console.is_terminal:
        session = pick_session(store)
        if session is None:
            console.print("[dim]starting a new session[/dim]")
    if session is not None and args.new:
        session = None
    if session is None:
        options = dict(config.get("options") or {})
        try:
            options.update(parse_options(args.option))
            if args.options_json:
                options.update(json.loads(args.options_json))
            think = parse_think(args.think if args.think is not None else config.get("think"))
            fmt = parse_format(args.format)
        except (ValueError, json.JSONDecodeError) as exc:
            raise SessionError(str(exc)) from exc
        return Session.create(
            "", client.base,
            system=args.system if args.system is not None else config.get("system"),
            parameters=options, think=think,
            keepalive=args.keepalive or config.get("keepalive"),
            format=fmt, tags=list(args.tag or []), title=args.title or "",
            gpu_layers=config.get("gpu_layers"))
    try:
        if args.system is not None:
            session.system = args.system
        if args.think is not None:
            session.think = parse_think(args.think)
        if args.format:
            session.format = parse_format(args.format)
    except ValueError as exc:
        raise SessionError(str(exc)) from exc
    if args.keepalive:
        session.keepalive = args.keepalive
    if args.tag:
        session.tags = sorted(set(session.tags) | set(args.tag))
    return session


def looks_like_session(store: SessionStore, ref: str) -> bool:
    """True when ref is an exact id, an id prefix, or a session title."""
    if not ref or ref.startswith("-"):
        return False
    try:
        if store._path(ref).exists():
            return True
        if re.fullmatch(r"[\w.-]+", ref) and any(store.dir.glob(f"{ref}*.json")):
            return True
    except OSError:
        return False
    lowered = ref.lower()
    return any(lowered in (entry.get("title") or "").lower() for entry in store.entries())


def normalize_argv(argv: Optional[list[str]], store: SessionStore) -> Optional[list[str]]:
    """Keep `-c/--continue` from swallowing a following prompt.

    `-c PROMPT` should mean "continue the latest session with PROMPT", while
    `-c SESSION_ID PROMPT` should continue that session. argparse's nargs='?'
    cannot tell them apart, so a value is kept only when it resolves to a
    saved session (or looks like a session id), and a bare `-c` followed by
    anything else is rewritten to an explicit `--continue=__latest__`.
    """
    if argv is None:
        argv = sys.argv[1:]
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-c", "--continue"} and index + 1 < len(argv) \
                and not argv[index + 1].startswith("-"):
            candidate = argv[index + 1]
            id_like = re.fullmatch(r"\d{4,8}-\d{3,6}[-\w]*", candidate) is not None
            if looks_like_session(store, candidate) or id_like:
                out += [token, candidate]
                index += 2
                continue
            out.append("--continue=__latest__")
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def remote_preflight(client: OllamaClient, label: str, config: dict[str, Any],
                     *, quiet: bool = False) -> bool:
    """Probe a remote Ollama with a short timeout; False (and a message) when down."""
    try:
        version = client.version(timeout=REMOTE_PROBE_TIMEOUT)
    except OllamaError as exc:
        detail = str(exc)
        prefix = f"cannot reach Ollama at {client.base}: "
        if detail.startswith(prefix):
            detail = detail[len(prefix):]
        node = remote_node(label, config)
        state = tailscale_peer_state(node)
        hint = f" — tailscale reports {node} {state}" if state.startswith("offline") else ""
        console.print(f"[err]cannot reach remote {escape(label)} at {client.base}: "
                      f"{escape(detail)}{escape(hint)}[/err]")
        return False
    if not quiet:
        console.print(f"[dim]remote {escape(label)} · {client.base} · ollama {version}[/dim]")
    return True


def session_reconnect(args: argparse.Namespace, config: dict[str, Any],
                      session: Session,
                      ts_map: Optional[dict[str, str]] = None) -> Optional[str]:
    """Base URL a resumed session should reconnect to, or None for the usual chain.

    A session that last ran somewhere other than the local default host is
    sticky -- unless --host/--remote explicitly says where to go instead.
    """
    if args.host or args.remote or not session.host or is_local_host(session.host):
        return None
    return resolve_remote(session.remote, config, ts_map) if session.remote else session.host


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    store = SessionStore()
    args = parser.parse_args(normalize_argv(argv, store))
    resuming = args.continue_ is not None or bool(args.session) or args.resume
    if resuming and args.model and not args.prompt and not args.model_flag:
        args.prompt, args.model = [args.model], None
    config = load_config(Path(args.config) if args.config else None)
    set_tool_colors(config.get("tool_colors"))
    setup_logging(args.debug)

    if args.reindex:
        rows = store.rebuild_index()
        console.print(f"[ok]indexed[/ok] {len(rows)} sessions")
        return 0
    if args.list:
        return cli_list(args, store)
    if args.search:
        return cli_search(args, store)
    if args.show:
        return cli_show(args, store, config)
    if args.export:
        return cli_export(args, store)
    if args.delete:
        return cli_delete(args, store)

    if args.json and args.raw:
        console.print("[err]--json and --raw are mutually exclusive[/err]")
        return 2
    if args.dimensions is not None and args.dimensions <= 0:
        console.print("[err]--dimensions must be a positive integer[/err]")
        return 2
    if args.system_file and not args.system:
        try:
            args.system = Path(args.system_file).read_text(encoding="utf-8").rstrip("\n")
        except OSError as exc:
            console.print(f"[err]--system-file: {exc}[/err]")
            return 2

    timeout = args.timeout or config.get("timeout") or DEFAULT_TIMEOUT
    quiet = not console.is_terminal or args.json or args.raw
    remote_name: Optional[str] = None
    try:
        if args.remote:
            host = resolve_remote(args.remote, config)
            remote_name = args.remote
        elif args.host:
            host = args.host
        elif config.get("remote"):
            remote_name = str(config["remote"])
            host = resolve_remote(remote_name, config)
        else:
            host = config.get("host") or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
        client = OllamaClient(host, insecure=args.insecure, timeout=timeout)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return 2

    try:
        session = resolve_session(args, config, client, store)
    except SessionError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return 1

    # Reconnect where the session last ran: a session whose host is not the
    # local default resumes there unless --host/--remote says otherwise.
    try:
        reconnect = session_reconnect(args, config, session)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return 2
    adopted = reconnect is not None
    if reconnect and reconnect != client.base:
        client = OllamaClient(reconnect, insecure=args.insecure, timeout=timeout)
        console.print(f"[dim]resuming session on {escape(session.remote or reconnect)} "
                      f"({client.base})[/dim]")
    if not adopted and (args.host or args.remote) and session.host \
            and not is_local_host(session.host) and session.host != client.base:
        console.print(f"[dim]session was created on {escape(session.remote or session.host)}"
                      f"; using {escape(args.remote or args.host or '')} instead[/dim]")

    probe_label = (session.remote or session.host) if adopted else remote_name
    if probe_label and not remote_preflight(client, probe_label, config, quiet=quiet):
        return 1

    if adopted:
        host_label = session.remote or session.host
        ssh_target = remote_node(session.remote, config) if session.remote else None
    elif remote_name:
        host_label = remote_name
        ssh_target = remote_node(remote_name, config)
    elif not is_local_host(client.base):
        host_label, ssh_target = args.host or client.base, None
    else:
        host_label, ssh_target = None, None

    if args.gpu_layers is not None:
        try:
            parse_gpu_layers_spec(args.gpu_layers)
        except ValueError as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
            return 2
        session.gpu_layers = args.gpu_layers

    model = resolve_model(args, config, client, session)
    if not model:
        return 2
    chosen = ensure_model(client, model, config, ssh_target=ssh_target,
                          host_label=host_label, assume_yes=args.pull,
                          pick_on_decline=True, quiet=quiet)
    if not chosen:
        return 2
    model = chosen
    session.model = model
    session.host = client.base
    if remote_name and not adopted:
        session.remote = remote_name
    elif args.host or is_local_host(client.base):
        session.remote = None

    gpu_layers_value: Optional[int] = None
    if session.gpu_layers:
        block_count: Optional[int] = None
        try:
            block_count = model_block_count(client.show(model))
        except OllamaError:
            pass
        if not quiet:
            console.print("[dim]gpu layers: "
                          f"{escape(gpu_layers_label(session.gpu_layers, block_count))}[/dim]")
        try:
            gpu_layers_value = resolve_gpu_layers(session.gpu_layers, block_count)
        except (ValueError, RuntimeError):
            gpu_layers_value = None

    if args.max_gpu_probe and (args.num_ctx or "").strip().lower() not in {"max-gpu", "--max-gpu"}:
        console.print("[err]--max-gpu-probe requires --num-ctx max-gpu[/err]")
        return 2
    if args.num_ctx is not None:
        kind, value = parse_num_ctx_arg(args.num_ctx)
        if kind == "error":
            console.print("[err]--num-ctx must be a positive integer or max-gpu[/err]")
            return 2
        if kind == "max-gpu":
            request = FitRequest(
                client=client, model=model, config=config,
                host_label=host_label, ssh_target=ssh_target,
                ssh_user=config.get("remote_ssh_user"),
                pinned_spec=session.gpu_layers,
                num_gpu=(session.parameters or {}).get("num_gpu"),
                base_options={key: val
                              for key, val in (session.parameters or {}).items()
                              if key != "num_ctx"},
                keepalive=session.keepalive or args.keepalive,
                refresh_gpu=lambda: machine_state_for(client.base, ssh_target=ssh_target,
                                                      ssh_user=config.get("remote_ssh_user")),
            )
            ctx = max_gpu_context(request, probe=args.max_gpu_probe)
            if ctx is None:
                return 2
            session.parameters["num_ctx"] = ctx
            note = " [dim]· model is loaded at that context[/dim]" if args.max_gpu_probe else ""
            console.print(f"[ok]num_ctx set to {ctx}[/ok]{note}")
        else:
            if int(value or 0) <= 0:
                console.print("[err]--num-ctx must be a positive integer[/err]")
                return 2
            applied, max_ctx = resolve_num_ctx(client, model, int(value or 0))
            if max_ctx and int(value or 0) > max_ctx:
                console.print(f"[warn]--num-ctx {value} exceeds {escape(model)}'s max "
                              f"context ({max_ctx}) — capped to {max_ctx}[/warn]")
            session.parameters["num_ctx"] = applied

    try:
        prompt_text = " ".join(args.prompt).strip()
        piped = not sys.stdin.isatty()
        if not prompt_text and piped:
            prompt_text = sys.stdin.read().strip()

        capabilities = client.capabilities(model)
        is_embedding = "embedding" in capabilities or (
            not capabilities
            and any(hint in model.lower() for hint in EMBED_MODEL_HINTS))
        has_embed_flags = args.dimensions is not None or args.truncate is not None
        if is_embedding:
            if not prompt_text:
                console.print("[err]embedding models need a prompt (or stdin)[/err]")
                return 2
            truncate = parse_bool(args.truncate, True)
            embed_options = {"num_gpu": gpu_layers_value} if gpu_layers_value is not None else None
            return run_embed(client, model, prompt_text, truncate=truncate,
                             dimensions=args.dimensions, json_out=args.json,
                             options=embed_options)
        if has_embed_flags and "embedding" not in capabilities:
            console.print("[warn]--dimensions/--truncate only apply to embedding models; "
                          "ignored[/warn]")

        repl = Repl(args, config, client, store, session)
        if args.json and not args.tools:
            repl.tools_enabled = False
        for image in args.image or []:
            try:
                repl.pending_images.append(encode_image(Path(os.path.expanduser(image)),
                                                        repl.image_storage))
            except (OSError, ValueError) as exc:
                console.print(f"[err]{escape(str(exc))}[/err]")
                return 2
        if session.system:
            console.print(f"[dim]system prompt: {shortcut(session.system, 80)}[/dim]")

        if prompt_text:
            return run_once(args, config, client, store, session, prompt_text, repl)
        if not console.is_terminal:
            console.print("[err]no prompt given on a non-interactive stdin[/err]")
            return 2
        return repl.run()
    except KeyboardInterrupt:
        console.print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
