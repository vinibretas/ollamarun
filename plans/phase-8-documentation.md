# Phase 8 — Documentation and examples

- README: “OpenAI-compatible API” (endpoints, fields, base URL, SDK example, streaming/tool behavior, compatibility gaps) and “Agent API” (turns, events, sessions, approvals, artifacts, deployment, and the deferred auth hooks).
- `examples/openai_client.py` and `examples/agent_client.py` — practical integration tests and reference clients.
- AGENTS.md/CLAUDE.md: architecture entries for `runtime.py`, `events.py`, `turns.py`, `server.py`, `openai_api.py`, `artifacts.py`; the new `serve` command; config keys; test notes. Keep the two files identical.
- Top-level `ollamarun` docstring changelog + `VERSION` bump per landed phase (Phase 2 landed as 1.46; Phase 3 onward increments from there).
- `__init__.py` exports; `build_parser()` epilog mentions `serve`.
