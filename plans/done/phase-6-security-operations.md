# Phase 6 — Security and operations

Operational hardening and configuration integration, not an API redesign.
Phase 10 landed first (v1.55), so its context-control policy keys are folded
into this phase's matrix.

Status: implemented and landed at v1.56. The contracts below are what
shipped; the deviations and follow-ups are recorded under "Recorded
implementation decisions".

Network exposure is a privileged control surface. **Locked: LAN-accessible,
no authentication in this stage.**

## Baseline: what already exists (audited at v1.55)

Verified in the tree; the phase builds on these and does not rebuild them:

- `HttpServer` already accepts `body_limit`, `ping_interval` and
  `request_timeout` (`server.py`) — but `serve_main()` constructs it with
  bind/port only, so the transport runs on hard-coded defaults.
- One shared `ApiError` envelope; unhandled exceptions become a sanitized
  500 with the traceback going to the `--debug` log only. No filesystem
  paths in error responses.
- Per-request `OllamaClient` instances with `abort_stream`; tool
  subprocesses are cancellable through `ToolContext.cancel` (phases 3–5
  tests cover both).
- Approval (`serve_approval_timeout`, explicit-0 disables), turn
  (`serve_turn_timeout`, 0 = no cap) and turn-manager
  (`serve_max_active_turns`, `serve_turn_event_buffer`, `serve_turn_event_ttl`)
  keys exist and are read by `AgentApi`.
- Turn and fit shutdown already exist (`AgentApi.shutdown` resolves pending
  approvals as cancelled, then joins the registries), and
  `ManagerClosed`/`FitsClosed` already map to `503 server_shutting_down`.
- The artifact store enforces per-file cap (0 = unlimited), quota (0 =
  unlimited), TTL (0 = forever) and validates manifest rel-paths lexically
  (`_safe_rel`: relative, no `..`, no backslashes).
- Fit policy keys `serve_fit` / `serve_max_active_fits` / `serve_fit_ttl`
  ship from Phase 10 (the TTL already uses explicit-None defaulting).
- `@file` expansion is structurally unreachable from the API surfaces
  (only `run.py`/`repl.py` call `expand_file_mentions`).
- The OpenAI mapper range-checks its mapped parameters (`temperature`,
  `top_p`, penalties, `max_tokens`; `openai_api._options_from`).

## Gaps this phase closes

1. **Transport limits are not wired or validated** (§1) — and the serve
   config has accumulated enough keys to need one matrix.
2. **Chunked request bodies** answer `411 length_required` (§2).
3. **No aggregate input limits**: `resolve_content_parts` bounds each part
   only (10 MiB per image, 32 KiB per inlined text file). Message counts,
   part counts, total text and cumulative image bytes are unbounded — a
   32 MiB JSON body of remote-URL or `file_id` image parts can resolve to
   gigabytes and hundreds of fetches.
4. **Native parameters are barely validated**: turn `settings.parameters`
   accepts any scalar/list per key, `POST /v1/sessions` accepts any dict,
   `PATCH` cannot touch them (§3).
5. **No workspace boundary**: `AgentApi._cwd` feeds policy suggestions
   only; file tools use the REPL's cwd and `run_shell` takes a free `cwd`
   (§4). The artifact store has no resolved-root containment invariant.
6. **No `serve_models` / `serve_agent` switches** (§4).
7. **Shutdown has no HTTP-handler drain and no unload**: `daemon_threads`
   + `block_on_close=False` mean handler threads are never joined; serve
   never unloads (the CLI's `unload_on_exit` is not wired here) (§5).

## 1. Configuration matrix and startup validation

One validator, `serve_settings(config)` in `helpers.py`, returns a
normalized copy of the config (serve keys defaulted and type-checked; other
keys untouched). `serve_main()` calls it first, prints
`[err]serve_* config ...[/err]` and exits 2 on `ValueError`, then passes the
normalized config downstream and the transport values to `HttpServer`
(`body_limit`, `request_timeout`, `ping_interval`). This replaces the
current ad-hoc artifact-number try/except.

| Key | Type | Default | Validation | Zero / special |
|---|---|---|---|---|
| `serve_bind` | IP string | `0.0.0.0` | non-empty string | — |
| `serve_port` | int | 11440 | 0–65535 | 0 = ephemeral (tests) |
| `serve_agent` | bool | `true` | JSON boolean | — |
| `serve_models` | list[str] | `[]` | list of non-empty strings | empty = all models |
| `serve_workspace` | path | startup cwd | exists, is a directory | — |
| `serve_body_limit` | int bytes | 32 MiB | ≥ 1024 | — |
| `serve_request_timeout` | float s | 30 | > 0 | per-operation, not a request deadline |
| `serve_sse_ping_interval` | float s | 15 | > 0 | keeps live streams alive |
| `serve_shutdown_timeout` | float s | 10 | > 0 | bounds each shutdown stage |
| `serve_unload_on_exit` | bool | `true` | JSON boolean | `--keep-loaded` overrides |
| `serve_max_messages` | int | 16384 | ≥ 1 | cumulative per request |
| `serve_max_content_parts` | int | 4096 | ≥ 1 | cumulative per request |
| `serve_max_message_bytes` | int bytes | 8 MiB | ≥ 1 | UTF-8, one message |
| `serve_max_text_bytes` | int bytes | 32 MiB | ≥ 1 | total resolved text per request |
| `serve_max_images` | int | 64 | ≥ 1 | cumulative per request |
| `serve_max_image_bytes` | int bytes | 64 MiB | ≥ 1 | cumulative after resolution |
| `serve_max_active_turns` | int | 4 | ≥ 1 | **fixes `or`-conflation of 0** |
| `serve_turn_event_buffer` | int | 10000 | ≥ 1 | **fixes `or`-conflation** |
| `serve_turn_event_ttl` | float s | 86400 | ≥ 0 | 0 = keep completed turns forever |
| `serve_approval_timeout` | float s | 300 | ≥ 0 | 0 = disabled (already) |
| `serve_turn_timeout` | float s | 0 | ≥ 0 | 0 = no cap (already) |
| `serve_artifact_root` | path | `ARTIFACTS_DIR` | string | created on startup |
| `serve_artifact_max_bytes` | int bytes | 16 MiB | ≥ 0 | 0 = unlimited (already) |
| `serve_artifact_quota_bytes` | int bytes | 2 GiB | ≥ 0 | 0 = unlimited (already) |
| `serve_artifact_ttl` | float s | 604800 | ≥ 0 | 0 = forever (already) |
| `serve_fit` | enum | `probe` | off/estimate/probe | — |
| `serve_max_active_fits` | int | 2 | ≥ 1 | — |
| `serve_fit_ttl` | float s | 86400 | ≥ 0 | 0 = forever (already) |

Defaulting rule throughout: an omitted key uses the default; a present key
is validated — zero is legal only where the table defines it. The
`or`-defaulting sites in `AgentApi.__init__` become explicit-None checks.
`serve_turn_event_ttl: 0` additionally needs a `completed_ttl > 0` guard in
`TurnManager.sweep` so "keep forever" is real (today the sweep would drop
finished turns immediately).

## 2. Transport hardening: chunked request bodies

`server.py:_read_body()` decodes `Transfer-Encoding: chunked` instead of
rejecting it. Both API surfaces inherit the behavior; nothing changes in
`openai_api.py` / `agent_api.py`.

- Only the `chunked` coding is accepted; any other coding (gzip, multiple
  codings) → 400 `unsupported_value` naming the header value.
- `Transfer-Encoding` together with `Content-Length` is ambiguous framing
  → 400 `invalid_header` (anti-smuggling; never guess which is right).
- Chunk-size lines are hex with optional `;extensions` (ignored);
  cumulative **decoded** size is capped at `body_limit` and enforced while
  reading — over → 413 `body_too_large` before the body is buffered.
- Malformed sizes, missing CRLFs, a truncated chunk or a stream that ends
  before the terminal zero-chunk → 400 `invalid_body`.
- Trailers after the zero-chunk are drained until the blank line with a
  small bound (8 KiB) and ignored.
- Tests exercise the parser through a real socket, not only the unit path.

## 3. Shared input and parameter validation

**Aggregate input budget.** One per-request budget object (in
`artifacts.py`, next to `resolve_content_parts`) carries the six
`serve_max_*` limits and is charged by `resolve_content_parts` while parts
resolve. Both mappers hand it in: openai `_messages_from` and the agent
turn parser (new-session seed and existing-session delta alike). Counts
(messages, parts, images) are also pre-checked where cheap so a request
fails before any fetch or decode. Violations are 413 `input_too_large`
with the offending `param` (`messages[2].content[0].image_url.url`) and the
limit name/value in the error's extra fields. Per-part caps stay as they
are (10 MiB image, 32 KiB inline); the cumulative caps are what make
amplification bounded. Limits are config-file keys — no per-limit CLI
flags.

**Native Ollama option validator (decision A: bounded pass-through).**
One validator, used by `POST /v1/agent/turns` settings, `POST /v1/sessions`
and the fit request builder (fit reads only validated values; the
dedicated `num_ctx`/`gpu_layers` paths keep their Phase 10 model-aware
checks and precedence):

- Known options get type + range validation against the curated table
  (temperature [0,2], top_p (0,1], top_k ≥0, min_p [0,1], typical_p,
  tfs_z, repeat_penalty, repeat_last_n, presence/frequency_penalty
  [-2,2], seed, num_predict ≥-1, num_ctx ≥1, num_gpu/main_gpu ≥0,
  num_thread/num_batch ≥1, num_keep ≥0, mirostat 0|1|2, mirostat_eta/tau,
  stop as string or list, use_mmap/use_mlock booleans — the exact table
  lands with the implementation and mirrors the OpenAI mapper where they
  overlap).
- `parameters.num_ctx` keeps CLI-parity type/range validation only; the
  model-aware trained-maximum check stays on the dedicated `num_ctx`
  field.
- Non-finite floats (`NaN`, `Infinity`), nested dicts and bools in numeric
  slots are rejected.
- Unknown option names are forwarded (forward-compatibility with new
  Ollama options) but bounded: scalars or lists of scalars, strings ≤4 KiB,
  lists ≤64 items, whole `parameters` dict ≤64 KiB serialized.
- Errors: 400 `unsupported_value` naming `settings.parameters.<key>` (or
  the equivalent path).

## 4. Boundaries: workspace, artifact root, model list, route groups

**Workspace boundary (decision E).** `serve_workspace` (default: the
directory serve started in) is resolved once and set on the API tool
context only — the CLI passes nothing and is unchanged. A shared helper in
`tools.py` applies it to the path-argument tools (`read_file`,
`write_file`, `edit_file`, `list_dir`, `make_dir`, `move_file` and
`copy_file` source **and** destination, `delete_file`,
`publish_artifact`) and to the `cwd` argument of `run_shell`/`run_python`:
expanduser → relative paths join the root → `Path.resolve()` (following
symlinks) → must stay under the resolved root, escapes denied with a clear
tool error. It is **not** subprocess confinement: a shell command can still
reach anything the process user can; the contract governs path arguments
and default working directories, and the README says exactly that. The
serve banner prints the resolved workspace root.

**Artifact root containment.** `ArtifactStore` resolves its root once and
asserts the resolved-containment invariant — `(root / rel).resolve()` must
stay under the resolved root — on manifest adoption, reads and sweep. The
index and manifests are treated as untrusted on load: entries that fail are
skipped and logged (the startup sweep already reconciles), so a symlinked
directory under the root can never redirect a read outside it.

**`serve_models` (decision C).** Optional allowlist of fnmatch patterns
matched against `canonical_model_name`; empty/unset = all installed models.
Literal patterns are tag-normalized (`qwen3` matches `qwen3:latest`); glob
patterns match the canonical name as written. One helper (in `helpers.py`)
is used everywhere a model is selected, listed or used: `GET /v1/models`
(filters the list), `GET /v1/models/{id}`, chat completions, session
create/model PATCH, agent turns — including an existing session whose
stored model is no longer served, which must fail before the turn runs —
and both fit surfaces. Outside the list → 404 `model_not_found`, identical
to unknown (never reveal installed-but-forbidden).

**`serve_agent` (decision D).** Route registration splits into three groups:

| Route group | Controlled by |
|---|---|
| `/v1/models*`, `/v1/chat/completions` | always registered |
| `/v1/files`, `/v1/artifacts/{id}`, `/v1/artifacts/{id}/meta` | registered whenever a store exists — OpenAI `file`/image parts and downloads need them with the agent off |
| turns, approvals, sessions, fits | `serve_agent` (default true) |

`add_agent_routes()` splits into `add_artifact_routes()` and
`add_agent_routes()` sharing one `AgentApi`; with `serve_agent=false` the
agent group answers 404 (route absent), which also takes the fit endpoints
and turn-start fitting out of the surface. No module renumbering; the
artifact routes upload/download are unchanged.

## 5. Lifecycle: drain, shutdown, unload

Shutdown becomes an explicit sequence, bounded by `serve_shutdown_timeout`:

1. **Stop accepting** — signal → `run_forever` stops the accept loop
   (existing).
2. **Refuse new work** — `AgentApi.begin_shutdown()` (split from
   `shutdown()` on `TurnManager`/`FitService`: set closed first, join
   later) so in-flight handlers cannot start turns/fits; the existing
   503 `server_shutting_down` mapping does the rest.
3. **Cancel and join** — resolve pending approvals as cancelled, cancel
   active turns and fits, join workers within the deadline. Sessions are
   written by the turn path as today; the tests assert the on-disk session
   contents, not just turn status.
4. **Drain handlers** — `HttpServer` tracks active handler threads
   (counter + condition in `_ThreadedServer`); `drain(timeout)` waits for
   them after cancels have unblocked event streams, and proceeds at the
   deadline regardless (handler threads are daemons; a process exit is
   never held hostage by a stalled client).
5. **Unload (decision F)** — `serve_unload_on_exit` defaults to **true**
   and unloads only the distinct models this server instance actually used
   (a small tracked set noted by both surfaces; embedding-capability models
   skipped, matching the house rule). `--keep-loaded` disables it. A
   parallel CLI session's model is never evicted.
6. **Close** — `server.close()` last.

The serve parser gains `--keep-loaded`; all other new settings are
config-file keys.

## 6. Documentation and surface changes

- No new endpoints. Surface changes: 413 `input_too_large` for the
  aggregate caps, 400s for parameter/chunked-validation failures, 404
  `model_not_found` extended through `serve_models`, and the
  `serve_agent` route gating.
- SSRF (recorded decision H): no IP filtering — fetching LAN/internal
  image URLs is inherent to the trusted-network contract and filtering
  would break LAN media servers. Redirect/timeout/size caps stay as
  Phase 5 shipped them.
- Audit items with regressions where feasible: `@file` stays unreachable
  from the API; request content never reaches the log without `--debug`;
  no filesystem paths in error envelopes (tool results may legitimately
  name paths the model itself requested — documented).
- README: the consolidated `serve_*` matrix replaces the scattered lists;
  the banner adds the workspace root and, with `serve_agent=false`, the
  reduced surface. AGENTS.md/CLAUDE.md: architecture/test notes, `serve`
  flag, config key list. VERSION bump + changelog entry on landing (Phase
  10 landed at 1.55, so this is 1.56).

## Locked decisions

| # | Decision |
|---|---|
| A | Unknown Ollama options: bounded pass-through (known keys range-checked; bounds as in §3). |
| B | Input limits: looser defaults — 16384 messages, 4096 parts, 8 MiB/message, 32 MiB text, 64 images, 64 MiB cumulative image bytes — all config-file keys. |
| C | `serve_models`: fnmatch on canonical names, empty = all, one helper, 404 `model_not_found` everywhere including stored sessions. |
| D | `serve_agent`: default true; three route groups; artifacts stay with the agent off. |
| E | `serve_workspace`: startup cwd by default, overridable; resolved-prefix jail on path arguments; not subprocess confinement; CLI untouched. |
| F | `serve_unload_on_exit`: default **true**, tracked-used set only, embeddings skipped, `--keep-loaded` opt-out; `serve_shutdown_timeout` default 10. |
| G | Transport keys: `serve_body_limit` 32 MiB, `serve_request_timeout` 30 s, `serve_sse_ping_interval` 15 s; validated centrally. |
| H | SSRF: no IP filtering; caps and timeouts only. |

## Tests

| Test | Proves |
|---|---|
| `test_serve_config_matrix` | Every key: defaults, malformed values, zero semantics, explicit-vs-omitted, `--keep-loaded` precedence; the transport values reach `HttpServer` |
| `test_chunked_bodies` | Real-socket parity: `Content-Length` and chunked enforce the same decoded cap; exact limit; over limit (413); malformed size / truncated / missing terminal chunk (400 `invalid_body`); `TE`+`Content-Length` ambiguity (400 `invalid_header`); non-chunked coding rejected; trailers drained |
| `test_input_limits` | Message/part/image counts, per-message bytes, total text, cumulative image bytes — both surfaces, 413 `input_too_large` with the right `param`; counts fail before any fetch |
| `test_native_parameters` | Known-matrix types/ranges, non-finite rejection, booleans/lists bounds, unknown pass-through bounds, dict size cap; create/turn-seed/fit entry points |
| `test_workspace_jail` | Relative/absolute paths, symlink escape, move/copy destinations, `run_shell`/`run_python` cwd; CLI unaffected |
| `test_artifact_root_containment` | Symlinked dir under the root cannot redirect read/sweep; failing index entries skipped and logged |
| `test_serve_model_gate` | List filter, lookup, chat, session create/model PATCH, turn on a stored session, fit endpoints — consistent 404 |
| `test_serve_agent_switch` | With `serve_agent=false`: files/artifacts usable (201/200), agent paths 404 |
| `test_serve_shutdown` | Begin-refuses-new (503), parked approval resolved cancelled, turn/fit cancelled, handler drain bounded, in-flight request completed or refused cleanly; session file contents on disk |
| `test_serve_unload_on_exit` | Only the tracked used models unloaded, embeddings skipped, `--keep-loaded`/config off; never raises on a dead backend |
| existing suite | All CLI regression tests stay green; OpenAI streaming, artifact retrieval, turn replay and fitting unchanged |

## Delivery order

1. Config matrix: `serve_settings()` in `helpers.py`, new keys, transport
   wiring, `or`-conflation fixes (`serve_turn_event_ttl: 0` guard in
   `TurnManager.sweep`), `--keep-loaded`; tests.
2. Transport: chunked decoding + framing tests.
3. Shared budget + native parameter validator; tests.
4. Workspace jail, artifact resolution containment, `serve_models`,
   route split for `serve_agent`; tests.
5. Shutdown split (`begin`/`join`), handler drain, tracked-model unload;
   tests.
6. Docs (README matrix, banner, AGENTS/CLAUDE), VERSION 1.56 + changelog,
   move this plan to `done/`, index row.

## Acceptance

- The service is LAN-accessible by default with no credentials,
  documented as trusted-network-only; tool safeguards survive HTTP
  unchanged.
- Oversized, malformed and ambiguous requests fail safely on both
  surfaces — including resolved-input amplification — and chunked uploads
  interoperate under the same cap as `Content-Length` bodies.
- The API has one validated configuration surface: every `serve_*` key has
  a documented type, range and zero convention, enforced at startup.
- The API tool context is bounded by the configured workspace; the
  artifact store is contained under its resolved root; the model list and
  feature switches apply consistently across every route.
- Shutdown leaves sessions consistent on disk, resolves or cancels every
  pending operation within the deadline, and unloads only what it loaded.
- Every step lands with tests; the full suite (`python3 test_ollamarun.py`)
  is green at every commit.

## Handoffs

- **Phase 7**: roll-up — the phase ships its own tests; Phase 7 re-gates
  the whole suite, adds the pyflakes gate and the folded coverage matrix.
- **Phase 8**: full API documentation pass and example clients (Phase 6
  updates the config matrix, banner and AGENTS/CLAUDE as behavior moves).
- **Phase 9**: unchanged — `/v1/embeddings`, `n > 1`,
  `parallel_tool_calls`, remote-image verification and the recorded
  closures; `serve_models` and `serve_agent` get documented in the Phase 8
  pass.

## Recorded implementation decisions

- **`PATCH /v1/sessions/{id}` gained `parameters`.** The review listed
  PATCH as a validator entry point but the field did not exist; adding it
  (a validated merge into the session, rejection before any write) was the
  smallest coherent way to make that true.
- **Chunked framing codes**: over the decoded cap → 413 `body_too_large`
  (checked before each chunk is read); `Transfer-Encoding` plus
  `Content-Length` → 400 `invalid_header`; a coding other than `chunked`
  → 400 `unsupported_value`; malformed sizes, missing CRLF, truncation, a
  missing terminal chunk and oversized trailers → 400 `invalid_body`.
  Trailers are drained with an 8 KiB bound and ignored, and a client that
  vanishes mid-error no longer produces a server traceback
  (`_send_json_error` swallows broken pipes).
- **`serve_turn_event_ttl: 0` needed a code change**, not just defaulting:
  `TurnManager.sweep` gained a `completed_ttl > 0` guard, so 0 really
  keeps completed turns. The `or`-defaulting sites in `AgentApi.__init__`
  (`serve_max_active_turns`, `serve_turn_event_buffer`, `serve_turn_event_ttl`,
  `serve_max_active_fits`, `serve_turn_timeout`) became explicit-None
  checks.
- **The input budget lives in `artifacts.py`** (`InputLimits`,
  `input_limits(config)`, `InputBudget`) — the shared content layer both
  surfaces already resolve through. `resolve_content_parts` charges parts,
  text bytes and images; the mappers charge message counts and tool-message
  text; counts fail before any fetch. Image bytes are charged at the
  decoded size (base64 length × 3/4, conservative by the padding).
- **Known-option table** mirrors the OpenAI mapper's ranges where they
  overlap (temperature 0–2, top_p/min_p/typical_p 0–1, penalties −2–2);
  `parameters.num_ctx` gets type/range validation only — the model-aware
  trained-maximum check stays on the dedicated `num_ctx` field, which
  still wins over `parameters`.
- **Workspace resolution**: `serve_settings` keeps `serve_workspace: null`
  meaning "the startup cwd, resolved by the API"; `AgentApi` always has a
  workspace (`cwd or os.getcwd()`) and sets it on the tool context per
  turn. `publish_artifact`'s base moved from `ctx.cwd` to
  `ctx.workspace or ctx.cwd`; `run_shell`/`run_python` follow the same
  rule. The helper governs path arguments and default working
  directories only — subprocess confinement is explicitly not claimed.
- **Artifact containment treats bad entries as absent**: reads/resolve
  return None and the sweep drops the entry **without chasing the file
  outside the root** (nothing outside the root is ever unlinked).
- **`serve_models` matching**: literal patterns are tag-normalized (a
  bare `qwen3` matches `qwen3:latest`), glob patterns match the canonical
  name as written (`qwen3*` matches every tag), empty/unset serves every
  model. A stored session whose model fell out of the list is refused by
  the same `_ensure_model` the turn path already calls.
- **Route registration was reshaped**: `build_agent_api()` constructs the
  one `AgentApi`, `add_artifact_routes()` registers files/artifacts and
  `add_agent_routes()` registers turns/sessions/fits. The old
  constructor-style `add_agent_routes` signature was replaced (callers
  updated: `serve_main` and the test helper).
- **Unload tracking**: `helpers.ModelTracker` records chat-completion
  models, agent-turn models and probing fit jobs (estimates never load).
  Embedding-capability models are skipped and only resident tracked models
  are unloaded; `serve_unload_on_exit` defaults to true and
  `--keep-loaded` disables it.
- **Shutdown**: `begin_shutdown()` on both registries refuses new work
  before the cancel/join (racing handlers get the existing 503);
  `HttpServer.drain(timeout)` waits for tracked handler threads and
  proceeds at the deadline. One documented consequence: an idle keep-alive
  connection occupies its handler thread for up to `serve_request_timeout`
  (the socket timeout), so a drain may consume its full window at exit —
  handler threads are daemons, so this never blocks process exit.

## Closure

Closed at v1.56 with the full suite green (`python3 test_ollamarun.py`)
and these regression groups: `test_serve_config_matrix`,
`test_chunked_bodies`, `test_input_limits`, `test_native_parameters`,
`test_workspace_jail`, `test_artifact_root_containment`,
`test_serve_model_gate`, `test_serve_agent_switch`, `test_serve_shutdown`,
`test_serve_unload_on_exit`, plus the existing Phase 3–5/10 suites
unchanged (`test_turn_expired` was updated for the new `completed_ttl: 0`
semantics).
