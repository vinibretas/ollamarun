# Overview

ollamarun began as a nicer `ollama run`: streamed markdown, 2D Unicode math,
sessions that actually persist. It has since grown into a full harness — an
approval-gated tool loop, two front ends, remote inference, context/GPU
placement fitting, and an HTTP API layer (`ollamarun serve`) that speaks both
the OpenAI API and its own native agent API.

This is not a product. It is a one-person workbench for learning how local LLM
stacks fit together, and how much of a real agent harness you can build around
one. Every `ollama run` flag (ollama 0.34) is accepted and extended, so it
still doubles as a drop-in replacement for the CLI it started as.

## What I'm using it to learn

- **Local LLM orchestration** — talking to Ollama over raw HTTP, streaming
  responses by hand, model lifecycle (`/api/ps`, load/unload), GPU placement.
- **Harness engineering** — the unglamorous middle of agents: tool schemas,
  approvals that do not get in the way, cancellation, output truncation, and
  keeping a model on the rails across multi-round tool loops.
- **API compatibility and support** — making unmodified OpenAI clients work
  against local models (strict validation, SSE, embeddings, emulated `n > 1`),
  plus a native agent API (turns, events, parked approvals, sessions, fits)
  for clients that want the whole loop.
- **Terminal UX** — live markdown re-rendering, a self-contained
  LaTeX → 2D Unicode math layout engine, a collapsible thinking panel,
  tok/s + context footers.
- **Systems math** — sizing context windows from GGUF metadata and KV-cache
  geometry, VRAM budgets, and calibrating estimates with measured probes.
- **Data-driven prompting** — tool definitions and system prompts as editable
  files, not Python strings.

## The path so far

1. **`ollama run`, but nicer** — richer rendering, 2D math, persistent
   sessions, slash commands.
2. **Tools and approvals** — Ollama function calling with a read-only
   allowlist, numbered approval prompts with remembered rules, and sessions
   that keep the whole tool loop.
3. **A real terminal app** — a Textual TUI, remote inference over Tailscale,
   GPU/context fitting, image input, artifacts.
4. **An API layer** — `ollamarun serve`: an OpenAI-compatible `/v1` surface
   plus a native agent API for full turns, with reference clients and a
   compatibility test suite.

## Status

Early, single-developer and moving. Built against one Linux box and a small
Tailscale fleet; Python 3.10; a few tools are wired to my own side projects
and degrade gracefully when those are absent. Rough edges are part of the
deal — expect the structure to keep changing as the learning continues.

## Roadmap

The main thing on the roadmap is a [web interface](../web-ui/index.md) on top
of `ollamarun serve` as its backend. The native agent API already has the
shape a browser client needs — submit a turn, follow its event stream, answer
parked approvals, manage sessions and files — because that is the same loop
the terminal drives.

The existing Textual [TUI](../tui/index.md) is deprecated in the same light:
it will be rewritten as a pure client of that API instead of reaching into the
runtime in-process, so every front end speaks one interface.
