# Artifacts

| Endpoint | What it does |
| --- | --- |
| `POST /v1/files` | upload one file (`{filename, media_type, data_base64}`) as an artifact |
| `GET /v1/artifacts/{id}` | download an artifact's bytes (recorded media type) |
| `GET /v1/artifacts/{id}/meta` | the artifact's `ArtifactRef` |

`POST /v1/files` uploads one file as an artifact
(`{filename, media_type, data_base64}`, strict JSON) and returns the full
record:

```json
{"id": "art_…", "media_type": "text/plain", "filename": "report.txt",
 "size": 1842, "sha256": "…", "source": "user", "turn_id": null,
 "session_id": null, "created_at": "…", "url": "/v1/artifacts/art_…"}
```

```console
$ B64=$(base64 -w0 report.txt)
$ curl -s http://127.0.0.1:11440/v1/files \
    -H 'Content-Type: application/json' \
    -d "{\"filename\": \"report.txt\", \"media_type\": \"text/plain\",
         \"data_base64\": \"$B64\"}"
{"id": "art_…", "url": "/v1/artifacts/art_…", …}

$ curl -s -o report.txt http://127.0.0.1:11440/v1/artifacts/art_…   # download
$ curl -s http://127.0.0.1:11440/v1/artifacts/art_…/meta           # the ref
```

Messages reference it as `{"type": "file", "file_id": "art_…"}` — in
agent-turn content arrays and in OpenAI messages alike; a missing or
expired id is a structured `404 artifact_not_found` before inference:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Summarize this report"},
  {"type": "file", "file_id": "art_…"}
]}
```

## Storage and retention

Artifacts are **immutable snapshots** (the hash and size come from the
stored bytes), live under `~/etc/data/ollamarun/artifacts/` (opaque ids,
index-driven lookup, sanitized display names only, and a resolved-root
containment invariant: every index/manifest path must resolve inside the
store root, so a symlinked directory under it can never redirect a read or
a sweep) and expire after config `serve_artifact_ttl` (default 7 days; the
sweep also reaps stale reservations and debris at startup). The per-file cap
is `serve_artifact_max_bytes` (default 16 MiB — the base64-in-JSON request
must fit the 32 MiB body cap) and `serve_artifact_quota_bytes` (default
2 GiB) is reserved under a lock so concurrent uploads cannot overshoot a
`507 artifact_quota_exceeded`.

## Publication

Tool outputs publish automatically (`write_file`/`edit_file`/`copy_file`/
`move_file` snapshot their result, regular files only); scripts publish
explicitly through the approval-gated
[`publish_artifact`](../../cli/tools/reference.md) tool. Every publication
emits `artifact.created` with the full record before `tool.result`, which
references the ids; turn results and session metadata list the refs. A
`persist: false` turn still registers and serves its artifacts — retention
is TTL-only and shared with session-deleted conversations (deleting a
session never deletes artifacts early).
