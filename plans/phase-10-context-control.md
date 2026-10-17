# Phase 10 — Context and placement control over the API

The native API can only *guess* a context today: `settings.parameters` passes
`num_ctx`/`num_gpu` through as raw Ollama options (unvalidated, undocumented),
`gpu_layers` is not reachable at all, and no surface can ask the server — the
side with the machine access — to *fit* a context. This phase makes explicit
adjustment first-class and adds fitting: a synchronous estimate endpoint, and
asynchronous probe jobs that verify the boundary by loading the model.

**Scope: the native agent surface only.** The OpenAI-compatible surface stays
a stateless translation proxy (one request = one `/api/chat` call, no
session, unknown fields rejected): there is no session to attach a
`num_ctx`/`gpu_layers` setting to, and fitting per model turn would mean
load/fit churn per request. Locked here.

## A. First-class adjustment (cheap, synchronous)

- `PATCH /v1/sessions/{id}` gains `num_ctx` (positive int or null) and
  `gpu_layers` (the CLI spec grammar: `N`, `P%`, `+N`/`-N`, `all`, `cpu`,
  `auto`, or null).
- Validation reuses the CLI's parsers (`parse_gpu_layers_spec`,
  `resolve_gpu_layers` against the model's `/api/show` block count,
  `resolve_num_ctx`-style capping knowledge); `%`/delta specs resolve at
  PATCH time so a bad spec is a structured 400 rather than a payload-time
  surprise, and an out-of-range `num_ctx` is rejected naming the model's
  trained maximum (the API is strict where the CLI caps with a warning).
- A `gpu_layers` change unloads a resident model, mirroring `/gpu-layers`
  ("the next request reloads with the new placement"); a `num_ctx` change
  needs no explicit unload (Ollama reloads on mismatch).
- Agent turns accept the same fields in `settings` (session mutations, like
  `model`/`system`/`parameters`), plus the estimate strings from section D.
- The existing 409 `session_busy` guard covers edits during turns; section C
  extends it to active fit jobs.

## B. Synchronous estimate — `POST /v1/agent/fit`

```json
{"model": "…", "session_id": "…?", "mode": "gpu"|"cpu",
 "cpu_percent": 70, "gpu_layers": "50%", "apply": false}
```

- Estimate only: `/api/show` + `/api/tags` + `/api/ps` + machine state
  (local nvidia-smi/unit env, or one ssh probe when serve targets
  `--remote`); it never loads the model.
- 200 with the fit result (section E); `apply: true` (requires `session_id`)
  writes `num_ctx` into the session in the same call.
- Errors: 400 `fit_failed` for "cannot size this model" cases, 403
  `fit_disabled` when `serve_fit` is `off`, 404 for an unknown model, 409
  `session_busy` when applying to a session with an active turn/fit.

## C. Asynchronous probe jobs — `/v1/agent/fits`

- `POST /v1/agent/fits` validates a fit request (`probe` defaults true) and
  answers **202** with
  `{id, object, status, session_id?, model, mode, created_at, status_url,
  cancel_url, warnings: [...]}`.
- `GET /v1/agent/fits/{id}` returns status
  (`queued|running|completed|failed|cancelled`), `progress` (stage, current
  candidate context, attempts), and on completion the fit result.
- `POST /v1/agent/fits/{id}/cancel` sets the job's token; a probe stops at
  its next checkpoint and reports `cancelled`.
- `apply: true` writes the fitted `num_ctx` when the job completes (the
  session is held by the fit while it runs).
- **Warnings on submit**, honest about cost: other `/api/ps` residents or
  active turns → "the probe will be slower"; existing calibration
  measurements for this digest/host/placement → "starting from N prior
  measurements (fewer loads)".
- **Caching**: probes seed from `CtxCalibrationStore` and record what they
  measure — a re-fit, or a fit by another client, is faster; the cache is
  per machine/digest/placement, exactly as the CLI's.
- **Exclusivity**: a fit job with a `session_id` holds that session — a
  turn, a PATCH, or a second fit for the same session is refused 409
  `session_busy` (naming `active_turn_id` / `active_fit_id`); fits on other
  sessions are allowed and merely slower. A global cap
  `serve_max_active_fits` (default 2) bounds machine churn and answers 429
  `rate_limit_error`.
- **Retention**: jobs live in memory (like turn records), swept after
  `serve_fit_ttl` (default 86400 s); unknown and expired ids are the same
  404 `fit_not_found` (no 410 tombstones — a fit is cheap to re-run).

## D. Turn-start estimate (never a blocking probe)

- `settings.num_ctx` accepts, besides the int: `"max-gpu"`, or `"max-cpu"`
  with `settings.cpu_percent` (1–100; required for the cpu mode).
- Resolved at turn start through the existing `on_started` seam (the context
  warning / auto-compaction step) — **estimate only**: turns never wait
  minutes for a fit. Probing stays an explicit job (section C).
- The result is persisted (`session.parameters["num_ctx"]`), emitted as the
  typed `session.fit` event on the turn's stream, and carried as
  `result.fit` on the turn status/result. A failed fit leaves the context
  unchanged and reports the failure on the event/result rather than failing
  the turn.
- `serve_fit: "off"` rejects the fit strings at turn submit (400
  `fit_disabled`).

## E. The fit result shape

```json
{"model": "…", "mode": "gpu"|"cpu", "num_ctx": 32768,
 "source": "estimate"|"probe", "gpu_layers": "50%"|null,
 "bound": "max_gpu"|"max_cpu", "percent": 70,
 "candidates": [{"ctx": 16384, "ok": true}, …],
 "report": {"weights_bytes": …, "kv_bytes_per_token": …, "budget_bytes": …,
            "calibration_points": 3, "profile": "…"}}
```

Numbers and labels only — no paths, no host internals; `report` carries the
same facts the CLI's fit report prints, as data.

## F. Provision refactors

- **`ctxfit.py` compute/presentation split.** `max_gpu_context()` /
  `max_cpu_context()` print rich reports and return `Optional[int]`;
  the API needs structure. Introduce a `FitOutcome` (num_ctx, source, bound,
  report dict, error, cancelled) and a `fit_context()` entry point; the
  existing max-* functions keep their signatures by printing from the
  outcome, so the CLI/REPL paths are behavior-unchanged.
- **Cancellation and progress.** `FitRequest` gains `cancel`
  (a `CancelToken`) and `on_progress` (a callback); estimates and probe
  loops check the token at each checkpoint (raise `FitCancelled`), call
  `on_progress` per candidate, and pass the token into the machine probes
  (the helpers' ssh/nvidia-smi probes already take one).
- **`fits.py` (new stage 23).** The `FitService`: worker thread per job,
  status/progress/result, cancel, per-session and global caps, TTL sweep,
  shutdown, and the job body shape. Nothing here is reachable from the
  OpenAI surface.
- **`runtime.py` / `agent_api.py`.** `TurnResult.fit` (optional dict) and
  `turn_body()` grow `result.fit`; the `session.fit` event joins the
  vocabulary (`events.py`) and both consumers (the Repl event printer and
  `HeadlessRenderer`) stay generic — the API emits it from its `on_started`
  closure, so no front-end change is required beyond the vocabulary entry.
- **Module numbering**: `fits.py` lands as 23; `openai_api` 24,
  `agent_api` 25, `cli` 26 (AGENTS/CLAUDE update with the phase).

## Policy and caps

- `serve_fit`: `"off" | "estimate" | "probe"` (default `"probe"` — the
  trusted-network contract, opt-out for operators). `off` refuses the
  endpoints and the turn-start fit strings; `estimate` refuses probe jobs
  (403 `probe_disabled`).
- `serve_max_active_fits`: 2. `serve_fit_ttl`: 86400 (seconds).
- These ship with this phase; **Phase 6 folds them into its consolidated
  policy matrix** (values, caps and their tests) when it lands.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/agent/fit` | synchronous estimate; optional `apply` |
| `POST /v1/agent/fits` | start an async fit job (probe by default); 202 + warnings |
| `GET /v1/agent/fits/{id}` | status, progress, result |
| `POST /v1/agent/fits/{id}/cancel` | cancel a running job |
| `PATCH /v1/sessions/{id}` | extended with `num_ctx` / `gpu_layers` |
| `POST /v1/agent/turns` | `settings.num_ctx` / `gpu_layers` / `cpu_percent` |

## Tests

| Test | Proves |
|---|---|
| `test_ctxfit_outcome` | `fit_context()` returns the structured outcome (estimate modes, error cases); the CLI max-* wrappers still print and return |
| `test_ctxfit_cancel_progress` | a pre-cancelled token raises `FitCancelled` with no model loads; progress fires per candidate |
| `test_fit_service` | job lifecycle (queued→running→completed), progress, cancel, TTL sweep, shutdown, caps (per-session + global 429) |
| `test_session_context_fields` | PATCH `num_ctx`/`gpu_layers` validation (strict 400s, ranges, spec resolution), unload parity, 409 while busy, persistence |
| `test_fit_estimate_api` | `POST /v1/agent/fit` matrix: modes, percent, apply, policy gate, unknown model, failure envelope |
| `test_fit_job_api` | 202 + warnings, status/progress, cancel, apply on completion, 404 unknown/expired, 429 cap, policy gates |
| `test_turn_start_fit` | `settings.num_ctx: "max-gpu"` estimates before round one, persists, emits `session.fit`, carries `result.fit`; failures leave the context unchanged |
| `test_fit_policy_gates` | `serve_fit` off/estimate/probe across endpoints and turn settings |
| existing suite | CLI/REPL paths unchanged; all green |

Fit tests are GPU-free: `FitRequest.machine`/`ram` injection plus the
existing `/api/ps`-verified fake backend.

## Delivery order

1. Plan docs (this file, the index row, the Phase 6 policy line).
2. `ctxfit.py`: `FitOutcome`/`fit_context()` + cancel/progress threading +
   tests (CLI behavior unchanged).
3. `fits.py` (stage 23, renumbering) + service tests.
4. PATCH/turn-settings fields (`num_ctx`, `gpu_layers`) + tests.
5. Estimate endpoint, fit jobs (routes, warnings, policy), turn-start
   estimate (`session.fit`, `result.fit`) + tests.
6. README/AGENTS/CLAUDE, VERSION bump + changelog.

## Acceptance

- A client can set an explicit `num_ctx`/`gpu_layers`, get an estimate, or
  run a probe asynchronously and apply its result — all without the CLI.
- Probes are honest about cost (warnings), cancellable, capped, cached and
  session-exclusive; another client's probe can only make one slower, never
  wrong.
- Turn-start fitting is estimate-only and never blocks a turn; the typed
  event and `result.fit` make it observable.
- The OpenAI surface is untouched; the CLI/REPL behavior is unchanged; the
  full suite is green at every commit.

## Handoffs

- **Phase 6**: fold `serve_fit` / `serve_max_active_fits` / `serve_fit_ttl`
  into the consolidated policy matrix (caps, values, tests) alongside
  `serve_agent`/`serve_models`.
- **Phase 7**: roll-up (the phase ships its own tests; Phase 7 re-gates the
  whole suite).
- **Phase 8**: README/AGENTS/CLAUDE updates land with the phase; Phase 8
  keeps the API documentation pass.
- **Phase 9**: unchanged — the OpenAI surface deliberately gains no context
  fields (locked in this plan).
