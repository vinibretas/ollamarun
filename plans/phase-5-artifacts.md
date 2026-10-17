# Phase 5 — Files, images, and artifacts

Adopt the provenance split and keep binary data out of message bodies.

Status: tightened after the v1.53 review; the contracts below are the
implementation target and supersede the first draft of this file. No code
has landed yet. Items marked **locked** are decisions, not options.

## A. One canonical artifact model

`POST /v1/files` and `ArtifactRef` are one resource behind one
`ArtifactStore` (`ollamarun_lib/artifacts.py`). There is no separate file
registry to keep in sync: the upload is sugar over `register(...)`, and a
`file_id` in a message part is resolved by the same lookup as any other
artifact id.

- Uploads register with `source: "user"` and return the full ref; tool and
  script outputs register with `source: "tool"`. `source: "model"` is
  reserved for a future model-generated-image path; nothing produces it in
  this phase.
- Ref shape (values illustrative):

  ```json
  {
    "id": "art_…", "media_type": "text/plain", "filename": "report.txt",
    "size": 1842, "sha256": "…", "source": "tool",
    "turn_id": "turn_…", "session_id": "ses_…",
    "created_at": "2026-…", "url": "/v1/artifacts/art_…"
  }
  ```

- Lifecycle properties (**locked**):
  - **Immutable snapshots.** Registration copies the bytes, so editing the
    original workspace file never changes a published artifact. Metadata is
    immutable too: `session_id`/`turn_id` are creation provenance, and a
    later association (an upload referenced by a turn) is recorded on the
    session, never by rewriting the ref.
  - **Hash and size come from the stored bytes** (sha256, decoded byte
    length), never from what the request declared.
  - **`url` is a relative API path** (`/v1/artifacts/<id>`), matching the
    existing relative `status_url`/`events_url` convention — never
    assembled from the incoming request host.
- Opaque ids: `art_` + `secrets.token_hex(16)`, validated against
  `^art_[0-9a-f]{32}$` before any lookup. Lookups go through the index
  (section E), so a client-supplied id never constructs a storage path.

## B. One input contract, two request formats

Both surfaces resolve message content through one shared resolver in
`artifacts.py`, while keeping their own request schemas:

- OpenAI surface (existing behavior preserved): `text` and `image_url`
  parts validated exactly as today, plus a new `file` part
  (`{"type": "file", "file_id": "art_…"}`). `image_url` gains http(s)
  support (section F).
- Native agent turns: `messages[].content` becomes a string **or** a parts
  array (`text`, `image_url`, `file`) — the current string-only rule
  (`agent_api.py`) cannot express a file reference. Unknown-field,
  unknown-part and unknown-role strictness is unchanged.
- Resolution table — one implementation, identical behavior and error
  codes on both surfaces:

| Input | Behavior |
|---|---|
| Image: data URI, remote URL, or an uploaded `image/*` file part | Capability check against `/api/show`; absent `vision` → 400 `model_does_not_support_images`, before inference. Bytes pass to Ollama as `images`. |
| Text-ish file: `text/*`, or the bounded head decodes as UTF-8 with no NUL | Bounded decode (constant, 32 KiB) inlined as a `<file name="…">` block — the same shape `@file` expansions use — with truncation noted. |
| Unsupported binary (pdf, docx, zip, …) | Contents are never invented. The model sees a short `[attached: <name> (<media_type>, N bytes) as artifact art_…, contents not extracted]` note; the artifact stays downloadable. PDF extraction is explicit later work. |
| Missing / unknown / malformed / expired `file_id` | Structured 404 `artifact_not_found` naming the part's param path, before inference starts. |

- Resolved user-message content is stored expanded (same as `@file`
  today), the referenced refs are appended to the session's artifact list,
  and the message records their ids for provenance.

## C. Tool result contract

- `ToolHandlerResult(text: str, artifact_paths: list[str] = [])` in
  `tools.py`; `HANDLERS` accepts `str | ToolHandlerResult` and
  `execute_tool()` unpacks either, so every existing string handler is
  untouched.
- Registration happens in `execute_tool()` through `ctx.artifacts`, a
  **turn-bound facade** (`store.for_turn(turn_id, session_id)`) built in
  `agent_api._make_repl()` — the turn id is known before `submit()`
  (`agent_api.py:463`). That matches the existing live-handle pattern
  (`client`, `store`, `cancel`), so `tools.py` never imports
  `artifacts.py`.
- **Regular files only**: each returned path that is a regular file is
  snapshotted and registered with `source: "tool"`; directories are
  skipped (no recursive bundles) and nothing else on the filesystem is
  scanned.
- Registration problems (oversize, quota, I/O) never fail the tool: a
  short note is appended to the result text and the refs that succeeded
  still attach.
- `write_file` / `edit_file` return the resulting file; `copy_file` /
  `move_file` return the **destination**; `delete_file` / `make_dir` are
  unchanged.
- `publish_artifact` (new): `defs/tools/publish_artifact.json` with one
  `path` parameter and a handler in `HANDLERS`. Approval-gated by the
  normal gate (neither `read_only` nor `always_safe`), cwd-jailed
  (`Path.resolve()` inside `ctx.cwd`), refuses directories and files over
  `serve_artifact_max_bytes`, and joins `PATH_RULE_TOOLS`
  (`tools.py:139`) so `/tools rules` can grant a directory. The scripts
  (`run_python`/`run_shell`) publish explicitly — no output scanning.
- Requirement wiring: `"artifacts"` joins `REQUIREMENTS` /
  `REQUIREMENT_HINTS`; `default_registry(artifacts=…)` gates the tool;
  `Repl(…, artifacts=None)` sets `tool_ctx.artifacts` before building the
  registry; the CLI passes `None`, so the tool is absent and no
  auto-registration happens there (zero CLI behavior change).

## D. Events and persistent state

- `ARTIFACT_CREATED` already exists in the vocabulary
  (`events.py:25`) and is never emitted yet.
- Emission point: `AgentRuntime.execute_tool_calls`, after the outcome and
  before `tool.result`. One `artifact.created` per ref whose payload is
  the complete `ArtifactRef`; the `tool.result` event then carries
  `artifacts: [<id>…]`. Contract order:
  `tool.call → approval.required → approval.resolved → artifact.created
  ×N → tool.result`.
- **Ids only on `tool.result`** (locked): the full record streams
  immediately before it, a client that missed it recovers through
  `GET /v1/artifacts/{id}/meta` — the same recovery pattern as the turn
  status endpoint after `stream.gap` — and the 10k-event ring buffers stay
  lean. Full refs are recorded here as the rejected alternative.
- The runtime accumulates refs; `TurnResult.artifacts` is filled from the
  same accumulator, and `turn_body()` exposes them under `result.artifacts`
  so a status read is self-contained.
- `Session.artifacts: list[dict]` accumulates full refs (deduped by id)
  from tool results and from uploads referenced by a turn; tool messages
  carry their refs and user messages their referenced ids. Old session
  files load unchanged (verified: `Session.from_dict` filters to known
  dataclass fields, `session.py:67-69`). `GET /v1/sessions/{id}` gains the
  artifact list.
- `persist: false`: artifacts are still registered with their turn/session
  provenance and stay fetchable; nothing is written to the session store.
  Retention is TTL-only — the sweep never keys off session existence.

## E. Storage, limits and retention

| Concern | Contract |
|---|---|
| Storage writes | Temp file → fsync → `os.replace`; then the per-artifact manifest, then the index update under RLock + `flock` (the `SessionStore` pattern) |
| Lookup | `index.json` + `meta/<id>.json` only; id charset validated first; no path is ever built from client input |
| Filename | Sanitized display name lives in metadata only; the on-disk name is the bare id |
| Media type | Validated at register (`type/subtype`), stored, served verbatim on download with `Content-Disposition` |
| Per-file maximum | `serve_artifact_max_bytes`, default 16 MiB; startup validation against the derivable upload ceiling (32 MiB body cap ⇒ ≈24 MiB binary minus the JSON envelope; 16 MiB leaves headroom) |
| Total quota | `serve_artifact_quota_bytes`, default 2 GiB; reserved under the index lock so concurrent uploads cannot overshoot; over → 507 `artifact_quota_exceeded`; stale reservations are reaped by the sweep |
| Retention | `serve_artifact_ttl`, default 7 d; sweep on startup (also reconciling the index from manifests and reaping orphan temp files) and opportunistically, rate-limited, on register; expiry unlinks bytes, manifest and index entry |
| Session deletion | **Locked**: artifacts are not deleted with the session; they stay fetchable until their own TTL expires (`DELETE /v1/sessions/{id}` and CLI `--delete` stay conversation-only). An opt-in purge can come later |
| Root | `serve_artifact_root`, default `~/etc/data/ollamarun/artifacts/` (new `ARTIFACTS_DIR` constant) |

Layout:

```
~/etc/data/ollamarun/artifacts/
  index.json                    # id -> {path, size, media_type, created_at,
                                #        session_id, turn_id, pending}
  index.lock                    # flock for index updates + quota reservations
  meta/<id>.json                # authoritative manifest: full ref + storage facts
  sessions/<sid>/<turn>/<id>    # turn-created artifacts
  uploads/<id>                  # uploads (no turn yet)
```

Deletes by expiry use a plain unlink (this is cache-style retention, not a
user delete); `trash-put` stays the rule for user-initiated deletions.
Missing and expired ids both answer 404 `artifact_not_found` (no
tombstones).

## F. Remote URL fetching and capability consistency

- One helper, `fetch_remote_media(url)`, in `artifacts.py`: http/https
  only, at most 3 redirects, explicit connect/read timeouts, the size cap
  enforced while reading (abort at cap + 1), media type from the response
  header with an image-signature fallback. Used by both surfaces.
- Errors: 400 `remote_fetch_failed` (DNS/connect/HTTP/redirect limit),
  413 `image_too_large` (past the cap), 400 `unsupported_value`
  (non-http scheme) — identical on both surfaces.
- One shared `require_vision(capabilities)` check → the same 400
  `model_does_not_support_images` on both surfaces.
- SSRF caveat documented (trusted-network contract); no IP filtering in
  this phase — Phase 6 owns any hardening.
- Unchanged deferrals: `/v1/embeddings`, `n > 1` and `parallel_tool_calls`
  stay in Phase 9, which also verifies the OpenAI surface uses these
  remote-image rules (they land here).

## G. Housekeeping that ships first

- Fix the `serve_approval_timeout` zero bug (`agent_api.py:361-362`):
  absent → 300, explicit `0` → disabled. Regression test for 0.
- When the phase index is next touched, reconcile the stale
  Last-Event-ID note in the phase-4 done plan (coverage landed in v1.53).

## Endpoints

Registered on the native surface (`agent_api.add_agent_routes()`):

| Endpoint | Success | Errors |
|---|---|---|
| `POST /v1/files` — `{filename, media_type, data_base64}`, unknown fields rejected | 201 + full ref | 400 validation (bad base64, bad filename/media type), 413 `artifact_too_large` (checked on the encoded length before decoding), 507 `artifact_quota_exceeded` |
| `GET /v1/artifacts/{id}` | 200, stored `Content-Type`, `Content-Disposition: attachment; filename="…"` | 404 `artifact_not_found` |
| `GET /v1/artifacts/{id}/meta` | 200 + full ref | 404 `artifact_not_found` |

## Module placement and config

- `ollamarun_lib/artifacts.py` lands as stage 22 (after `server.py`,
  before `openai_api.py`): it imports `constants`, `helpers` and
  `server` (`ApiError`) only; `openai_api`, `agent_api` and `cli` import
  it; `tools.py` never does (the `ctx.artifacts` handle keeps the one-way
  dependency graph). AGENTS/CLAUDE numbering shifts `openai_api` 22→23,
  `agent_api` 23→24, `cli` 24→25.
- Config (`DEFAULT_CONFIG`): `serve_artifact_root` (`ARTIFACTS_DIR`),
  `serve_artifact_max_bytes` (16 MiB), `serve_artifact_quota_bytes`
  (2 GiB), `serve_artifact_ttl` (604800 s).

## Tests

Fold into `test_ollamarun.py`, one `test_*` per area, wired into the
total sum:

| Test | Proves |
|---|---|
| `test_artifact_store` | register/get/meta, hash+size from stored bytes, immutability, id validation, traversal-shaped ids rejected, filename sanitization |
| `test_artifact_quota_and_ttl` | concurrent reservations cannot overshoot, 507, TTL sweep drops only expired, startup reconciliation of orphans |
| `test_artifact_upload_api` | `POST /v1/files` matrix: shape, 201, unknown fields, bad base64/media type, oversize before decode, 507 |
| `test_artifact_fetch_api` | download bytes + headers, `/meta`, 404 for missing/expired |
| `test_artifact_content_parts` | both surfaces: text/file/image resolution, 32 KiB bound + truncation note, binary note, missing id 404, vision parity |
| `test_artifact_remote_fetch` | the helper against a stub server: redirects, timeouts, cap, scheme, media type |
| `test_tool_artifact_registration` | write/edit/copy/move destination refs, dirs skipped, failure noted not fatal, `publish_artifact` gating/jail/size, absent without a store |
| `test_artifact_events` | `artifact.created` ordering and full-ref payload, `tool.result` ids, `TurnResult`, `turn_body` |
| `test_artifact_session_persistence` | session accumulation, message ids, old session JSON loads, `persist: false` artifacts fetchable |
| `test_serve_approval_timeout_zero` | the housekeeping fix |

## Delivery order

1. Housekeeping: `serve_approval_timeout` fix + test.
2. `artifacts.py` store + config keys + store tests.
3. `ToolHandlerResult` + `execute_tool` enrichment + the four file tools
   + `publish_artifact` + tests.
4. Runtime events + `TurnResult`/session persistence + `turn_body` + tests.
5. Shared part resolution + remote fetch + `openai_api` integration +
   the agent message schema + tests.
6. HTTP endpoints + `serve_main` wiring (store construction, startup
   sweep) + tests.
7. Docs (README, AGENTS/CLAUDE, config list, tool table), VERSION bump +
   changelog, phase index; move this plan to `done/` at closure.

## Acceptance

- Clients can upload images/files and fetch generated artifacts by opaque
  id; provenance (`user`/`tool`) is preserved end to end.
- No path escapes: ids are opaque and validated, lookups are index-driven,
  filenames are display-only.
- Sizes, quota and retention are enforced (including concurrent uploads),
  with the body-cap relationship honored.
- Both API surfaces resolve file references identically, with one
  capability answer per model.
- Events, turn results and sessions carry artifact references; older
  session files keep loading; `persist: false` artifacts remain fetchable.
- The CLI is unchanged: no store, no `publish_artifact`, no behavior
  change; the full suite stays green at every commit.

## Handoffs

- **Phase 6**: chunked request bodies (uploads beyond the current 32 MiB
  cap), `serve_agent`/`serve_models` gating of the artifact routes, SSRF
  hardening if any.
- **Phase 7**: fold the artifact matrix into the numbered test plan;
  pyflakes gate.
- **Phase 8**: README sections + example client for upload/download.
- **Phase 9**: verify the OpenAI surface on the phase-5 remote-image
  rules; embeddings, `n > 1` and `parallel_tool_calls` unchanged.
