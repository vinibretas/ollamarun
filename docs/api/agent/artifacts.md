# Artifacts

Files in and out of the API: uploads become artifacts, tools publish their
results automatically, and both sides of the protocol reference the same
opaque `art_…` ids.

| Endpoint | What it does |
| --- | --- |
| `POST /v1/files` | Upload one file as an artifact (`{filename, media_type, data_base64}`, strict JSON). |
| `GET /v1/artifacts/{id}` | Download the bytes (with the recorded media type). |
| `GET /v1/artifacts/{id}/meta` | The artifact's `ArtifactRef`. |

## Upload, inspect, download

```console
$ B64=$(base64 -w0 report.txt)
$ curl -s http://127.0.0.1:11440/v1/files -H 'Content-Type: application/json' \
    -d "{\"filename\": \"report.txt\", \"media_type\": \"text/plain\", \"data_base64\": \"$B64\"}"
```

```json
{
  "id": "art_a6d3394b39b25a3c46dc1a0f3ed7b363",
  "media_type": "text/plain",
  "filename": "report.txt",
  "size": 39,
  "sha256": "ed65e35aba78802e9b12af061950641db032e9c5ab938fe01750134801816608",
  "source": "user",
  "turn_id": null,
  "session_id": null,
  "created_at": "2026-10-18T13:23:03.002384-03:00",
  "url": "/v1/artifacts/art_a6d3394b39b25a3c46dc1a0f3ed7b363"
}
```

`GET /meta` returns the same ref; `GET` the artifact returns the bytes:

```console
$ curl -s http://127.0.0.1:11440/v1/artifacts/art_a6d3394b… -o report.txt
$ head -c 60 report.txt
Quarterly report draft.
Revenue is up.
```

A `sha256` mismatch is impossible by construction — the hash and size come
from the stored bytes, not the request.

## Using an artifact in a message

Messages reference ids with a `file` content part — in agent turns and in
OpenAI messages alike:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Summarize this report in one sentence."},
  {"type": "file", "file_id": "art_a6d3394b39b25a3c46dc1a0f3ed7b363"}
]}
```

Text-ish files are inlined as a `<file name="…">` block (the live run above
answered "The company's quarterly report indicates a significant increase in
revenue."); images go through the vision gate; other binaries are described
but never faked. A missing or expired id is a structured
`404 artifact_not_found` **before** inference.

## Publication

Tool outputs publish automatically in serve: the file tools snapshot their
result (regular files only) and scripts can call the approval-gated
`publish_artifact` tool. Every publication emits `artifact.created` with the
full ref between the approval decision and the `tool.result`, and the turn
result lists the refs:

```json
{"artifacts": [{"id": "art_2c008b58d34477b314406314cce2ec96",
  "media_type": "text/plain", "filename": "approve-demo.txt", "size": 15,
  "sha256": "a441a9bd…", "source": "tool", "turn_id": "turn_7cc9871fcf7d",
  "session_id": "20261018-131737-3773", "created_at": "…",
  "url": "/v1/artifacts/art_2c008b58d34477b314406314cce2ec96"}]}
```

A `persist: false` turn still registers and serves its artifacts — retention
is TTL-only.

## Storage and retention

- **Immutable snapshots** under `~/etc/data/ollamarun/artifacts/` (config
  `serve_artifact_root`), index-driven lookup, sanitized display names, and a
  resolved-root containment invariant: every index/manifest path must
  resolve inside the store root.
- **Retention:** `serve_artifact_ttl` (default 7 days; `0` keeps forever).
  The sweep also reaps stale reservations and debris at startup. Deleting a
  session never deletes its artifacts early.
- **Caps:** `serve_artifact_max_bytes` (default 16 MiB — the base64-in-JSON
  request must fit the 32 MiB body cap), `serve_artifact_quota_bytes`
  (default 2 GiB, reserved under a lock so concurrent uploads cannot
  overshoot) → `507 artifact_quota_exceeded`; oversize uploads are a `413`.
- The endpoints stay registered even with `serve_agent: false` (checked
  live), so the OpenAI surface can always send `file` parts.
