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

from .artifacts import ArtifactStore
from .client import OllamaClient, OllamaError
from .constants import (
    CONFIG_PATH,
    DEFAULT_HOST,
    DEFAULT_SERVE_BIND,
    DEFAULT_SERVE_PORT,
    DEFAULT_TIMEOUT,
    EMBED_MODEL_HINTS,
    HAS_TEXTUAL,
    LOG_PATH,
    REMOTE_PROBE_TIMEOUT,
    TEXTUAL_MIN_VERSION,
    UI_MODES,
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
    serve_settings,
    setup_logging,
    shortcut,
    tailscale_peer_state,
)
from .images import encode_image
from .agent_api import (add_agent_routes, add_artifact_routes,
                        build_agent_api)
from .openai_api import build_router
from .server import HttpServer, run_forever
from .ctxfit import (
    FitRequest,
    machine_state_for,
    max_cpu_context,
    max_gpu_context,
    parse_fill_percent,
    parse_num_ctx_arg,
)
from .interactive import (
    MODEL_SORT_KEYS,
    ensure_model,
    gpu_layers_label,
    model_block_count,
    parse_gpu_layers_spec,
    parse_model_sort,
    pick_model,
    pick_session,
    resolve_gpu_layers,
)
from .repl import Repl, resolve_num_ctx
from .run import run_embed, run_once
from .session import Session, SessionStore


def ui_mode(args: argparse.Namespace, config: dict[str, Any]) -> str:
    """Which interactive front end to start: inline, tui, or the auto choice."""
    wanted = (getattr(args, "ui", None) or config.get("ui") or "inline").lower()
    if wanted not in UI_MODES:
        wanted = "inline"
    if wanted == "auto":
        wanted = "tui"
    return wanted


def wants_tui(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    """True when the Textual front end should be used for this run."""
    if ui_mode(args, config) != "tui":
        return False
    if not HAS_TEXTUAL:
        console.print(f"[warn]--ui tui needs the textual package "
                      f"({TEXTUAL_MIN_VERSION}+) — using the inline REPL[/warn]")
        return False
    from .tui import tui_available

    ok, reason = tui_available()
    if not ok:
        console.print(f"[warn]{escape(reason)} — using the inline REPL[/warn]")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    epilog = """\
interactive commands:
  /help                 all commands (/? shortcuts for key bindings)
  /set parameter NAME V session parameter (temperature, num_ctx, seed, …)
  /set system TEXT      system prompt (unset: the built-in default)
  /show [topic]         model info|license|modelfile|parameters|system|template
  /model [NAME|--list|--sort KEY]
                        switch model, or list/sort installed models
                        (KEY: name|size|parameters|context|modified; the
                        picker prompt takes --sort KEY too)
  /load [MODEL|SESSION] switch model or resume a saved session
  /save NAME            create an Ollama model from this session
  /pull [MODEL]         pull a model onto this host (updates if present)
  /clear /new           start fresh context or a brand new session
  /undo /retry /edit    fix or regenerate the last exchange
  /summarize /context   manage the context window
  /tools [SPEC]         tool harness: on|off, policy MODE, enable|disable
                        NAME…, allow, rules; per-tool: NAME on|off|help|call
                        ARGS — help comes from the tool's definition
  /tts [on|off]         auto-speak answers via tts-remote; voice/lang/voices,
                        and `/tts tools on|off` for the model's tts access
  /thinking [MODE]      thinking display: show|collapse|hide|toggle|full
  /tps [on|off]         live tokens/s footer
  /ps                   resident models (ollama ps)
  /num-ctx [N | --max-gpu [--probe] | --max-cpu N [--probe]]
                        context window: N, the largest that keeps the GPU
                        placement, or fill N% of the available DRAM
                        (--probe verifies by loading the model)
  /gpu-layers [SPEC]    GPU layers: N, P%, +N/-N, all, cpu or auto
  /image PATH           attach an image to the next message
  /export [FILE]        export mid-session as md/json/html
  /queue [list|clear|rm N]  what is waiting while a turn is running (--ui tui)

while a response streams:
  Ctrl+T                expand the thinking panel (press again to collapse)
  --ui tui: Enter queues a message, Ctrl+B sends what you typed right now,
            Esc Esc stops the turn, /queue shows what is waiting

front end:
  --ui inline|tui|auto  inline (default) keeps the scrolling prompt_toolkit
                        REPL; tui splits the screen into transcript, status bar
                        and an always-available input box (needs the optional
                        textual package); auto picks tui when it is installed.
                        Config key: "ui".

model housekeeping:
  /model and /load unload the previous model before switching; leaving the
  REPL unloads the session model (--keep-loaded to keep it, --unload to also
  unload after a one-shot run).

serve:
  ollamarun serve [--bind IP] [--port N] [--host HOST|--remote NAME]
                        OpenAI-compatible HTTP API in front of Ollama:
                        GET /v1/models, GET /v1/models/{id} and
                        POST /v1/chat/completions (stream or not). Model
                        turns only — client tools are passed through and
                        returned, never executed. LAN access without
                        authentication: trusted-network-only, never expose
                        the port to the internet. See `ollamarun serve --help`.

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
                        help="obsolete: response timings are always shown on a "
                             "terminal; pass this to also get them when the output "
                             "is piped or redirected (--raw/--json never print them; "
                             "config show_stats: false disables)")

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
    wrapper.add_argument("--sort", metavar="KEY",
                         help="order for the model picker and lists: "
                              + ", ".join(MODEL_SORT_KEYS)
                              + " (default: modified)")
    wrapper.add_argument("-m", "--model", dest="model_flag", metavar="MODEL",
                         help="model name (same as the positional MODEL)")
    wrapper.add_argument("--system", metavar="TEXT",
                         help="system prompt (replaces the built-in default)")
    wrapper.add_argument("--system-file", metavar="FILE",
                         help="read the system prompt from a file")
    wrapper.add_argument("--no-system", action="store_true",
                         help="send no system prompt at all (the built-in default "
                              "is used otherwise)")
    wrapper.add_argument("--option", action="append", metavar="KEY=VALUE",
                         help="model option, repeatable (e.g. -o temperature=0.7)")
    wrapper.add_argument("--num-ctx", nargs="?", const="", default=None,
                         metavar="N|max-gpu",
                         help="context window size; capped to the model's max "
                              "(from /api/show) with a warning. 'max-gpu' picks "
                              "the largest context that still keeps the model on "
                              "the GPU; with --max-cpu N the context is raised "
                              "until the model's host RAM fills N%% of the "
                              "available DRAM (see /num-ctx --help)")
    wrapper.add_argument("--max-cpu", metavar="N",
                         help="with --num-ctx: fill N%% (1-100) of the DRAM "
                              "available to the model")
    wrapper.add_argument("--probe", dest="probe", action="store_true",
                         help="with --num-ctx max-gpu or --num-ctx --max-cpu N: "
                              "verify by loading the model at candidate contexts "
                              "and reading /api/ps back")
    wrapper.add_argument("--max-gpu-probe", dest="probe", action="store_true",
                         help=argparse.SUPPRESS)
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
    wrapper.add_argument("--ui", choices=["inline", "tui", "auto"], metavar="MODE",
                         help="interactive front end: inline (default prompt_toolkit "
                              "REPL), tui (Transcript / status bar / always-on input "
                              "box, Textual) or auto (tui when Textual is installed)")
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


def build_serve_parser() -> argparse.ArgumentParser:
    epilog = """\
endpoints:
  GET  /v1/models           installed models (OpenAI model objects)
  GET  /v1/models/{id}      one model, 404 when unknown
  POST /v1/chat/completions OpenAI-compatible model turn, streaming or not

  POST /v1/agent/turns            start a full agent turn (tool loop, 202)
  GET  /v1/agent/turns/{id}       turn snapshot: status, seq, result, error
  GET  /v1/agent/turns/{id}/events SSE turn events (replay: ?after= or
                                  Last-Event-ID; a disconnect only unsubscribes)
  POST /v1/agent/turns/{id}/cancel request cancellation
  POST /v1/agent/approvals/{id}   resolve a pending approval (run/remember/
                                  rule/auto/deny; first decision wins)
  GET/POST /v1/sessions           list/create sessions
  GET/PATCH/DELETE /v1/sessions/{id}
                                  inspect, edit or trash a session

  Model turns go through one inference call: client `tools` are passed to
  the backend and returned as standard `tool_calls` (`finish_reason:
  "tool_calls"`) but never executed here — the agent endpoints run the full
  tool loop with the serve policy and API-parked approvals. A top-level
  "system" field is accepted: "auto" uses the built-in default prompt, text
  replaces it, absent means none. Unsupported features (n>1, logprobs,
  logit_bias, audio, non-text modalities, unknown fields) fail with a
  structured 400.

security:
  the API is trusted-network-only. It has no authentication and its bind
  address is the only boundary: keep it on the LAN/tailnet and never expose
  the port to the public internet. The listener (--bind/--port) is separate
  from the upstream Ollama address (--host/--remote).

config: ~/etc/config_files/ollamarun.json — the validated serve_* matrix
        (bind, port, body/input limits, timeouts, serve_agent, serve_models,
        serve_workspace, artifact and fit keys; see the README). The startup
        banner prints the bound address, upstream and the warning.
"""
    parser = argparse.ArgumentParser(
        prog="ollamarun serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Serve the OpenAI-compatible and native agent HTTP API in "
                    "front of Ollama (LAN access, no authentication)",
        epilog=epilog)
    host_group = parser.add_mutually_exclusive_group()
    host_group.add_argument("--host", metavar="HOST",
                            help=f"upstream ollama server, e.g. 127.0.0.1:11434 "
                                 f"(default: OLLAMA_HOST or {DEFAULT_HOST})")
    host_group.add_argument("--remote", metavar="NAME",
                            help="run inference on a tailnet machine (pi, kiddo, "
                                 "g3, optiplex7070 or optiplex, guri) or any "
                                 "host[:port]")
    parser.add_argument("--bind", metavar="IP",
                        help=f"API listener address (default: {DEFAULT_SERVE_BIND}, "
                             f"all interfaces)")
    parser.add_argument("--port", type=int, metavar="N",
                        help=f"API listener port (default: {DEFAULT_SERVE_PORT})")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help=f"upstream request timeout (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--insecure", action="store_true",
                        help="skip TLS verification for the upstream")
    parser.add_argument("--keep-loaded", action="store_true",
                        help="do not unload the models this server used when "
                             "it stops (overrides serve_unload_on_exit)")
    parser.add_argument("--config", metavar="FILE",
                        help=f"config file (default: {CONFIG_PATH})")
    parser.add_argument("--debug", action="store_true",
                        help=f"log requests and tracebacks to {LOG_PATH}")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {VERSION}")
    return parser


def serve_main(argv: Optional[list[str]] = None) -> int:
    """`ollamarun serve`: run the OpenAI-compatible API until a signal."""
    parser = build_serve_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config) if args.config else None)
    setup_logging(args.debug)
    timeout = args.timeout or config.get("timeout") or DEFAULT_TIMEOUT

    remote_name: Optional[str] = None
    try:
        if args.remote:
            remote_name = args.remote
            host = resolve_remote(remote_name, config)
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
    if remote_name and not remote_preflight(client, remote_name, config,
                                            quiet=not console.is_terminal):
        return 1

    try:
        settings = serve_settings(config, bind=args.bind, port=args.port)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return 2
    if args.keep_loaded:
        settings["serve_unload_on_exit"] = False
    bind = settings["serve_bind"]
    port = settings["serve_port"]

    artifact_store = ArtifactStore(
        Path(settings["serve_artifact_root"]),
        max_bytes=settings["serve_artifact_max_bytes"],
        quota_bytes=settings["serve_artifact_quota_bytes"],
        ttl=settings["serve_artifact_ttl"])
    try:
        artifact_store.sweep()
    except OSError:
        console.print("[warn]artifact store sweep failed — continuing[/warn]")

    def client_factory() -> OllamaClient:
        return OllamaClient(host, insecure=args.insecure, timeout=timeout)

    router = build_router(client_factory, config=settings, timeout=timeout,
                          keepalive=settings.get("keepalive"),
                          artifacts=artifact_store)
    agent_api = build_agent_api(
        args=build_parser().parse_args([]), config=settings,
        store=SessionStore(), client_factory=client_factory,
        cwd=settings["serve_workspace"], artifacts=artifact_store)
    add_artifact_routes(router, agent_api)
    if settings["serve_agent"]:
        add_agent_routes(router, agent_api)
    try:
        server = HttpServer(router, bind=bind, port=port,
                            body_limit=settings["serve_body_limit"],
                            request_timeout=settings["serve_request_timeout"],
                            ping_interval=settings["serve_sse_ping_interval"])
    except OSError as exc:
        console.print(f"[err]cannot listen on {escape(bind)}:{port}: "
                      f"{escape(str(exc))}[/err]")
        return 1

    bound_host, bound_port = server.address
    scope = " (all interfaces)" if bound_host == "0.0.0.0" else ""
    console.print()
    console.print(f"[key]ollamarun serve[/key] [dim]· {VERSION} · "
                  f"OpenAI-compatible + agent API[/dim]")
    console.print(f"  listening [ok]http://{escape(bound_host)}:{bound_port}[/ok]"
                  f"[dim]{scope}[/dim]")
    console.print(f"  upstream  [dim]{escape(client.base)}[/dim]")
    workspace = settings["serve_workspace"] or os.getcwd()
    console.print(f"  workspace [dim]{escape(workspace)}[/dim]")
    if settings["serve_agent"]:
        console.print("[dim]  models /v1/models · chat /v1/chat/completions · "
                      "agent turns /v1/agent/turns · sessions /v1/sessions"
                      "[/dim]")
    else:
        console.print("[dim]  models /v1/models · chat /v1/chat/completions · "
                      "files /v1/files (agent routes disabled)[/dim]")
    console.print("[warn]trusted-network-only: no authentication — never expose "
                  "this port to the internet[/warn]")
    console.print("[dim]Ctrl-C stops the server (active turns are cancelled)[/dim]")
    try:
        run_forever(server)
    finally:
        server.close()
        agent_api.shutdown()
    console.print("[dim]stopped[/dim]")
    return 0


def resolve_model(args: argparse.Namespace, config: dict[str, Any],
                  client: OllamaClient, session: Optional[Session]) -> Optional[str]:
    explicit = args.model_flag or args.model
    if explicit and session is not None:
        session.model = explicit
    model = explicit or (session.model if session else None) or config.get("model")
    if not model:
        if console.is_terminal:
            model = pick_model(client, None, sort=args.sort or "modified")
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
        if args.no_system:
            system: Optional[str] = ""
        elif args.system is not None:
            system = args.system
        else:
            system = config.get("system")
        return Session.create(
            "", client.base,
            system=system,
            parameters=options, think=think,
            keepalive=args.keepalive or config.get("keepalive"),
            format=fmt, tags=list(args.tag or []), title=args.title or "",
            gpu_layers=config.get("gpu_layers"))
    try:
        if args.no_system:
            session.system = ""
        elif args.system is not None:
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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        return serve_main(argv[1:])
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
    if args.sort is not None:
        try:
            args.sort = parse_model_sort(args.sort)
        except ValueError as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
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
                          pick_on_decline=True, quiet=quiet,
                          sort=args.sort or "modified")
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

    num_ctx_arg = (args.num_ctx or "").strip()
    if args.max_cpu is not None and args.num_ctx is None:
        console.print("[err]--max-cpu requires --num-ctx[/err]")
        return 2
    if args.max_cpu is not None and num_ctx_arg:
        console.print("[err]--max-cpu takes the mode: use --num-ctx --max-cpu N[/err]")
        return 2
    if args.max_cpu is None and args.num_ctx is not None and not num_ctx_arg:
        console.print("[err]--num-ctx needs a value or --max-cpu N[/err]")
        return 2
    if num_ctx_arg.lower() in {"max-cpu", "--max-cpu"}:
        console.print("[err]--num-ctx max-cpu takes a percentage: use "
                      "--num-ctx --max-cpu N[/err]")
        return 2
    max_mode = bool(args.max_cpu is not None) or num_ctx_arg.lower() in {"max-gpu", "--max-gpu"}
    if args.probe and not max_mode:
        console.print("[err]--probe requires --num-ctx max-gpu or --num-ctx --max-cpu N[/err]")
        return 2
    if args.max_cpu is not None:
        percent = parse_fill_percent(args.max_cpu)
        if percent is None:
            console.print("[err]--max-cpu takes a percentage between 1 and 100[/err]")
            return 2
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
        )
        ctx = max_cpu_context(request, percent=percent, probe=args.probe)
        if ctx is None:
            return 2
        session.parameters["num_ctx"] = ctx
        note = " [dim]· model is loaded at that context[/dim]" if args.probe else ""
        console.print(f"[ok]num_ctx set to {ctx}[/ok]{note}")
    elif args.num_ctx is not None:
        kind, value = parse_num_ctx_arg(args.num_ctx)
        if kind == "error":
            console.print("[err]--num-ctx must be a positive integer, max-gpu, "
                          "or --max-cpu N[/err]")
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
            ctx = max_gpu_context(request, probe=args.probe)
            if ctx is None:
                return 2
            session.parameters["num_ctx"] = ctx
            note = " [dim]· model is loaded at that context[/dim]" if args.probe else ""
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
        elif session.system == "":
            console.print("[dim]system prompt disabled[/dim]")
        elif console.is_terminal:
            console.print("[dim]system prompt: built-in default[/dim]")

        if prompt_text:
            return run_once(args, config, client, store, session, prompt_text, repl)
        if not console.is_terminal:
            console.print("[err]no prompt given on a non-interactive stdin[/err]")
            return 2
        if wants_tui(args, config):
            from .tui import run_tui

            return run_tui(repl)
        return repl.run()
    except KeyboardInterrupt:
        console.print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
