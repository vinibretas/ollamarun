# Phase 5 — Files, images, and artifacts

Adopt your provenance split and keep binary data out of message bodies.

**Input:**

- OpenAI content parts: `image_url` data URIs (and http(s) URLs with a size cap; document the SSRF caveat), mapped to Ollama `images` after capability validation.
- Native file upload: `POST /v1/files` with `{filename, media_type, data_base64}` (JSON keeps stdlib parsing simple; multipart can come later). Returns an opaque `file_id`, size, sha256.
- Messages reference files via a documented content part (`{"type": "file", "file_id": "..."}`). No arbitrary filesystem paths.

**Output:**

- `ArtifactRef`: `{id, media_type, filename, size, sha256, source: "model"|"tool"|"user", turn_id, session_id, created_at, url}`.
- Storage under `~/etc/data/ollamarun/artifacts/<session>/<turn>/`; metadata in the turn result and session; configurable root, max size, TTL, total quota; cleanup on startup/periodically.
- `GET /v1/artifacts/{id}` and `/meta`; opaque IDs; path-jailed; validated content types and sanitized filenames. (Auth is deferred per locked decision 3; opaque IDs and the path jail are the only protections in this stage.)
- **Locked decision**: auto-register outputs from file tools (`write_file`, `edit_file`, `copy_file`, `move_file`) by extending `ToolOutcome` with `artifacts`; add an explicit, approval-gated `publish_artifact` tool for files created by `run_python`/`run_shell`. No workspace snapshots or filesystem diffs.
- Provenance: `source: "tool"` for tool/script-created files, `source: "model"` if a backend ever returns model-generated images, `source: "user"` for uploads. `publish_artifact` validates the path (cwd/workspace jail) and the size before registering.
- Do not claim image support just because the API accepts it: check `capabilities` on the selected model.

**Acceptance:** clients can upload images/files and fetch generated artifacts by opaque ID; provenance is preserved; no path escapes; sizes/retention enforced.
