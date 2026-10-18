#! /bin/python3.10
"""agent_client - reference native-agent client and integration check for ollamarun serve

Version: 1.0

Summary of changes in this version:
  - First release with Phase 8: submits a full agent turn, follows the typed
    SSE event stream, resolves a parked approval (interactive mode) and
    prints the terminal summary — the reference flow every agent client
    repeats.

Against a local server:

    ollamarun serve &
    python3 examples/agent_client.py --prompt "List the files in /tmp"

`--mode policy` (the default) applies the configured tool policy
server-side; `--mode interactive` parks undecided calls and answers them
with `--approve`. `--session-id` continues an existing session; `--json`
prints the raw event envelopes instead of the transcript.
"""
from __future__ import annotations

import argparse
import json
import sys

import requests
from rich.console import Console

console = Console()

TERMINALS = ("turn.completed", "turn.error", "turn.cancelled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference native-agent client for `ollamarun serve`.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11440",
                        help="the server's base URL (default: %(default)s)")
    parser.add_argument("--model", default="",
                        help="model id (default: the first listed)")
    parser.add_argument("--prompt", default="Reply with one short sentence.",
                        help="the user message for the turn")
    parser.add_argument("--session-id", default="",
                        help="continue an existing session instead of a new one")
    parser.add_argument("--mode", choices=["policy", "interactive"],
                        default="policy",
                        help="the turn's approval mode (default: %(default)s)")
    parser.add_argument("--approve",
                        choices=["run", "remember", "rule", "auto", "deny"],
                        default="run",
                        help="the decision sent for a parked approval")
    parser.add_argument("--no-persist", dest="persist", action="store_false",
                        help="run the turn ephemerally (no session write)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="print raw event envelopes instead of the transcript")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="client timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")
    timeout = args.timeout

    model = args.model
    if not model:
        try:
            listed = requests.get(f"{base}/v1/models", timeout=10).json()
        except requests.RequestException as exc:
            console.print(f"[red]cannot reach {base}: {exc}[/red]")
            return 1
        models = listed.get("data") or []
        if models:
            model = models[0].get("id") or ""
    if not model:
        console.print("[red]no model to run; pass --model[/red]")
        return 1

    payload = {
        "model": model,
        "persist": args.persist,
        "messages": [{"role": "user", "content": args.prompt}],
        "settings": {"approval": args.mode},
    }
    if args.session_id:
        payload["session_id"] = args.session_id
    try:
        created = requests.post(f"{base}/v1/agent/turns", json=payload,
                                timeout=timeout)
    except requests.RequestException as exc:
        console.print(f"[red]cannot reach {base}: {exc}[/red]")
        return 1
    if created.status_code != 202:
        console.print(f"[red]turn rejected ({created.status_code}): "
                      f"{created.text[:300]}[/red]")
        return 1
    turn = created.json()
    console.print(f"[cyan]turn {turn['id']}[/cyan] "
                  f"[dim]· session {turn['session_id']} · model {model} · "
                  f"approval {args.mode}[/dim]")

    status: dict = {}
    content_open = False
    try:
        with requests.get(f"{base}/v1/agent/turns/{turn['id']}/events",
                          params={"after": 0}, stream=True,
                          timeout=timeout) as stream:
            for raw in stream.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue  # `: ping` heartbeats and blank separators
                try:
                    event = json.loads(raw[6:])
                except ValueError:
                    continue
                kind = event.get("type", "")
                data = event.get("data") or {}
                if args.as_json:
                    console.print(json.dumps(event), markup=False)
                if kind == "message.delta":
                    if not args.as_json:
                        if data.get("channel") == "thinking":
                            console.print(f"[dim]{data.get('text', '')}[/dim]",
                                          end="")
                        else:
                            console.print(data.get("text", ""), end="")
                            content_open = True
                elif kind == "tool.call":
                    if content_open:
                        console.print()
                        content_open = False
                    console.print(f"[magenta]tool · {data.get('name')}[/magenta] "
                                  f"[dim]{json.dumps(data.get('args'))[:120]}[/dim]")
                elif kind == "tool.result":
                    ok = data.get("ok")
                    colour = "green" if ok else "red"
                    console.print(f"[{colour}]  → {'ok' if ok else 'error'}"
                                  f"[/{colour}] [dim]"
                                  f"{str(data.get('text', '')).splitlines()[:1]}"
                                  f"[/dim]")
                elif kind == "approval.required":
                    console.print(f"[yellow]approval required[/yellow] "
                                  f"[dim]{data.get('reason', '')}[/dim]")
                    approval_id = data.get("approval_id")
                    if args.mode == "interactive" and approval_id:
                        resolved = requests.post(
                            f"{base}/v1/agent/approvals/{approval_id}",
                            json={"action": args.approve}, timeout=30)
                        console.print(f"      resolved as [bold]{args.approve}"
                                      f"[/bold] ({resolved.status_code})")
                elif kind == "turn.warning":
                    console.print(f"[yellow]warning[/yellow] "
                                  f"[dim]{data.get('message', '')}[/dim]")
                elif kind in TERMINALS:
                    break
    except requests.RequestException as exc:
        console.print(f"[red]event stream failed: {exc}[/red]")
        return 1
    if content_open and not args.as_json:
        console.print()

    try:
        status = requests.get(f"{base}/v1/agent/turns/{turn['id']}",
                              timeout=30).json()
    except requests.RequestException as exc:
        console.print(f"[red]status read failed: {exc}[/red]")
        return 1
    result = status.get("result") or {}
    usage = result.get("usage") or {}
    console.print(f"[cyan]status[/cyan] [bold]{status.get('status')}[/bold] · "
                  f"rounds {result.get('rounds')} · "
                  f"tokens {usage.get('total_tokens')}")
    if status.get("error"):
        console.print(f"[red]error: {status['error']}[/red]")
    if not args.as_json and result.get("content"):
        console.print(f"[bold]answer[/bold] {result['content']}")
    return 0 if status.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
