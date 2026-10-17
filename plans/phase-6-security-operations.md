# Phase 6 — Security and operations

Network exposure is a privileged control surface. **Locked: LAN-accessible, no authentication in this stage.**

- Default listener `0.0.0.0:11440` (all IPv4 interfaces), configurable with `--bind IP [--port N]`; keep it separate from Ollama’s `--host`. Document prominently — in the README and in the `serve` startup banner — that the API is trusted-network-only and must not be exposed directly to the public internet.
- No authentication requirement for LAN requests. Do not implement token checks; keep the route structure easy to wrap with auth later, so adding it does not reshape routes. When it lands, authentication must never implicitly authorize tool execution.
- **Tool authorization is not network authorization.** Unauthenticated API access must not silently change the configured tool execution policy. Serve still defaults to `tools_policy=allowlist` even if the CLI config says `auto`; the agent API uses interactive approvals only when the client asks for them; `--yes` is the only auto path.
- `@file` expansion off; no implicit filesystem references from remote input.
- Request size caps, message length caps, image/file caps, model parameter ranges, `serve_max_active_turns`, one turn per session, turn/approval/SSE timeouts.
- Chunked request bodies (`Transfer-Encoding: chunked`) are decoded with the
  byte cap enforced during the read, replacing the current 411, so proxies
  and streaming uploaders interoperate.
- Path jail for artifacts and tool cwd; opaque IDs; no filesystem paths in responses.
- Sanitized errors; no message content in logs by default; `--debug` opts into detail.
- `serve_models` allowlist (optional), `serve_agent=false` for a pure inference server.
- Cancellation through `OllamaClient.abort_stream` plus cancellable tool subprocesses; graceful shutdown drains and cancels.
- Signal handling: SIGTERM/SIGINT stop accepting, cancel turns, flush sessions, optional unload.
- Fold Phase 10's context-control policy keys (`serve_fit`, `serve_max_active_fits`, `serve_fit_ttl`, shipped with that phase) into the consolidated policy matrix: values, caps and their tests alongside `serve_agent`/`serve_models`.

**Acceptance:** the service is LAN-accessible by default with no credentials, documented as trusted-network-only; tool safeguards survive HTTP unchanged; oversized/invalid requests fail safely; cancellation leaves sessions consistent.
