# Security and limits

!!! warning "Trusted-network-only"

    The service has no authentication, and the bind address is its only
    boundary. Keep it on the LAN/tailnet and never expose the port to the
    public internet.

## Deployment and auth

`ollamarun serve` is meant to run on the trusted network under the user's own
service management (the `serve_*` keys cover the listener, limits and
shutdown; the startup banner prints the surface in effect). There is
deliberately no authentication yet — the bind address is the only boundary,
so the port must never reach the public internet. When authentication lands
it wraps these routes without reshaping them: it is a transport concern, and
never a substitute for the tool-approval gate.

## Limits

The JSON body cap is `serve_body_limit` (default 32 MiB), and it applies to
`Content-Length` and chunked bodies alike: a chunked upload is decoded with
the cumulative decoded size checked before each chunk is read,
`Transfer-Encoding` together with `Content-Length` is an ambiguity error, and
malformed framing (bad sizes, missing CRLF, truncation, a missing terminal
chunk, a non-chunked coding) is a structured 400.

Resolving content parts is bounded separately, because a small body can name
many large inputs: `serve_max_messages`, `serve_max_content_parts`,
`serve_max_message_bytes`, `serve_max_text_bytes`, `serve_max_images` and
`serve_max_image_bytes` (cumulative after resolving data URIs, remote fetches
and file inlines; counts fail before any fetch). Embeddings inputs are charged
to the same budget (input count, per-input bytes, total text). Violations are
`413 input_too_large` naming the limit.

`serve_request_timeout` (default 30 s) is a per-operation socket timeout — a
streaming response that keeps writing (heartbeats included) is never cut off
by it — and `serve_sse_ping_interval` (default 15 s) is the SSE heartbeat.

## Model allowlist

`serve_models` (default: empty) restricts which models the API exposes and
accepts: names or fnmatch patterns matched against the installed models after
tag normalization (`qwen3` matches `qwen3:latest`; `qwen3*` matches every
tag). The list applies everywhere a model is selected, listed or used — the
model list, lookup, chat completions, session creation and model edits, agent
turns (a stored session whose model fell out of the list is refused before it
runs) and the fit endpoints — and a model outside it answers exactly like an
unknown one (`404 model_not_found`).

## Tool workspace

Serve runs its tool context jailed to `serve_workspace` (default: the
directory serve started in, printed in the banner): relative paths in the
file tools resolve under it, resolved escapes — symlinks included — are
refused, and the `run_shell`/`run_python` working directories follow the same
rule. This is an argument and default-working-directory policy, **not**
subprocess confinement: an approved command can still reach whatever the
process user can. The CLI keeps its previous behavior.

## Feature switches

`serve_agent: false` runs the process as an inference server: the native
routes (turns, approvals, sessions, fits) are not registered at all (404),
while models, chat completions and the `/v1/files` + `/v1/artifacts/*`
endpoints keep serving — the OpenAI surface can still send `file` parts. One
model policy applies to every route: see the allowlist above.

Every key with its default, range and zero semantics is listed in
[Config file](../config/file.md#serve-keys).
