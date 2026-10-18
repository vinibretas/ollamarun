# ollamarun

> A terminal agent harness for Ollama — a personal project about how far local
> models go when you hand them tools.

ollamarun began as a nicer `ollama run`: streamed markdown, 2D Unicode math,
sessions that actually persist. It has since grown into a full harness — an
approval-gated tool loop, two front ends, remote inference, context/GPU
placement fitting, and an HTTP API layer (`ollamarun serve`) that speaks both
the OpenAI API and its own native agent API.

Every `ollama run` flag (ollama 0.34) is accepted and extended, so it still
doubles as a drop-in replacement for the CLI it started as.

## Quickstart

Needs Linux, Python 3.10+ and a reachable Ollama server (default
`127.0.0.1:11434`).

```console
$ pip install --user rich requests prompt_toolkit
$ ./ollamarun qwen3:8b "explain this repo"    # one-shot
$ ./ollamarun qwen3:8b                        # interactive REPL
```

## Explore

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting started**

    ---

    Install, first run, and the API-layer reference clients.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-console:{ .lg .middle } **CLI**

    ---

    The terminal harness: the REPL, tools and approvals, rendering, sessions,
    models, context and placement, remote inference.

    [:octicons-arrow-right-24: CLI overview](cli/index.md)

-   :material-api:{ .lg .middle } **API**

    ---

    `ollamarun serve` — the OpenAI-compatible surface and the native agent
    API (turns, events, approvals, sessions, artifacts, fits).

    [:octicons-arrow-right-24: API overview](api/index.md)

-   :material-monitor:{ .lg .middle } **TUI**

    ---

    The Textual front end, deprecated ahead of the API-client rewrite.

    [:octicons-arrow-right-24: Textual TUI](tui/index.md)

-   :material-web:{ .lg .middle } **Web UI**

    ---

    Planned: a browser client built entirely on the native agent API.

    [:octicons-arrow-right-24: Web UI](web-ui/index.md)

-   :material-book-open-variant:{ .lg .middle } **Development**

    ---

    Architecture, where to change what, the test suite and this docs site.

    [:octicons-arrow-right-24: Contributing](development/contributing.md)

</div>

## Status

Early, single-developer and moving. Built against one Linux box and a small
Tailscale fleet; Python 3.10; a few tools are wired to my own side projects
and degrade gracefully when those are absent. Rough edges are part of the
deal — expect the structure to keep changing as the learning continues.

More background in [Project overview](project/overview.md);
[Apache-2.0](project/license.md) ·
[Acknowledgements](project/acknowledgements.md).
