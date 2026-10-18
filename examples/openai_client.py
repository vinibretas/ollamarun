#! /bin/python3.10
"""openai_client - reference OpenAI client and integration check for ollamarun serve

Version: 1.0

Summary of changes in this version:
  - First release with Phase 8: the official OpenAI SDK against
    `ollamarun serve` — the model list, a non-streaming completion, streaming
    with usage, embeddings (the SDK's default base64 encoding), `n > 1` and
    a tools request — printed as PASS/SKIP/FAIL lines, exiting non-zero on
    a failure.

Needs the official SDK (`pip install --user openai`). Against a local server:

    ollamarun serve &
    python3 examples/openai_client.py --model qwen3:8b

`--model` should name a chat model; an embedding-only model turns the chat
checks into failures, and a model without tool support skips the tools check.
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference OpenAI client for `ollamarun serve`.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11440/v1",
                        help="the server's OpenAI base URL (default: %(default)s)")
    parser.add_argument("--api-key", default="unused",
                        help="sent as-is; the server ignores credentials")
    parser.add_argument("--model", default="",
                        help="model id (default: the first listed)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="client timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        console.print("[red]the official SDK is not installed "
                      "(pip install --user openai)[/red]")
        return 2
    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=args.timeout)
    try:
        models = client.models.list()
    except Exception as exc:  # a wrong base URL is the common mistake
        console.print(f"[red]cannot reach {args.base_url}: {exc}[/red]")
        return 1
    console.print(f"[cyan]openai_client[/cyan] [dim]· SDK {openai.__version__} · "
                  f"{args.base_url}[/dim]")
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if ok:
            console.print(f"[green]PASS[/green]  {name}")
        else:
            failures += 1
            console.print(f"[red]FAIL[/red]  {name} {detail[:200]}")

    def skip(name: str, detail: str) -> None:
        console.print(f"[yellow]SKIP[/yellow]  {name} — {detail}")

    model = args.model
    check("the model list answers", bool(models.data),
          f"{len(models.data)} model(s)")
    if not model and models.data:
        model = models.data[0].id
    if not model:
        console.print("[red]no model to exercise[/red]")
        return 1
    console.print(f"      using model [bold]{model}[/bold]")

    try:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}])
        choice = response.choices[0]
        check("a non-streaming completion returns content and usage",
              bool((choice.message.content or "").strip())
              and (response.usage.total_tokens or 0) > 0,
              str(response)[:200])
    except Exception as exc:
        check("a non-streaming completion returns content and usage",
              False, repr(exc))

    try:
        parts: list[str] = []
        usage = None
        with client.chat.completions.create(
                model=model, temperature=0, max_tokens=32, stream=True,
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                extra_body={"stream_options": {"include_usage": True}}) as stream:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
                if chunk.usage is not None:
                    usage = chunk.usage
        check("streaming deltas arrive and usage closes the stream",
              bool(parts) and usage is not None
              and (usage.total_tokens or 0) > 0, f"{parts!r} {usage}")
    except Exception as exc:
        check("streaming deltas arrive and usage closes the stream",
              False, repr(exc))

    try:
        # The SDK sends encoding_format="base64" by default and decodes it.
        embeddings = client.embeddings.create(model=model,
                                              input=["hello", "world"])
        vectors = [item.embedding for item in embeddings.data]
        check("embeddings return one vector per input",
              len(vectors) == 2 and all(vectors)
              and len(vectors[0]) == len(vectors[1]),
              str(embeddings)[:200])
    except Exception as exc:
        if "does_not_support_embeddings" in repr(exc):
            skip("embeddings return one vector per input",
                 f"{model} is not an embedding model")
        else:
            check("embeddings return one vector per input", False, repr(exc))

    try:
        multi = client.chat.completions.create(
            model=model, temperature=0, max_tokens=16, n=2,
            messages=[{"role": "user", "content": "Say hi"}])
        check("n=2 returns two indexed choices",
              [choice.index for choice in multi.choices] == [0, 1],
              str(multi)[:200])
    except Exception as exc:
        check("n=2 returns two indexed choices", False, repr(exc))

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    try:
        tooled = client.chat.completions.create(
            model=model, temperature=0, max_tokens=64,
            messages=[{"role": "user",
                       "content": "Use get_weather for Sao Paulo."}],
            tools=tools)
        calls = tooled.choices[0].message.tool_calls or []
        check("a tools request is accepted (the server never executes it)",
              tooled.choices[0].finish_reason in ("tool_calls", "stop"),
              str(tooled)[:200])
        if calls:
            console.print(f"      the model asked for "
                          f"[bold]{calls[0].function.name}"
                          f"({calls[0].function.arguments})[/bold]")
        else:
            console.print("      the model answered without a tool call")
    except Exception as exc:
        if "does not support tools" in repr(exc):
            skip("a tools request is accepted (the server never executes it)",
                 f"{model} does not support tools")
        else:
            check("a tools request is accepted (the server never executes it)",
                  False, repr(exc))

    if failures:
        console.print(f"\n[red]{failures} check(s) failed[/red]")
        return 1
    console.print("\n[green]all checks passed[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
