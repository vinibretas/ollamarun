# Config file

| Path | What |
| --- | --- |
| `~/etc/config_files/ollamarun.json` | Config (created from defaults on first run); `--config FILE` overrides |
| `~/etc/config_files/ollamarun-defs/{tools,prompts}/` | Per-user tool/prompt overrides |

Selected config keys: `render`, `math`, `thinking`, `ui`, `show_stats`,
`tps_footer`, `max_render_chars`, `pager`, `auto_title`, `auto_compact`,
`image_storage`, `max_file_kb`, `options`, `unload_on_exit`, `gpu_layers`,
`tools*`, `tool_colors`, `tts*`, `chem_*`, `remote*`, `max_gpu_*`,
`max_cpu_*`, `timeout`, plus the `serve_*` matrix below. See `DEFAULT_CONFIG`
in `ollamarun_lib/constants.py` for the full set.

## Serve keys

`ollamarun serve` validates every `serve_*` key at startup: an invalid
value names itself and exits 2, an omitted key uses the default, and a `0`
means exactly what the row says.

| Key | Default | Meaning / zero |
| --- | --- | --- |
| `serve_bind`, `serve_port` | `0.0.0.0`, 11440 | listener; port 0 = ephemeral |
| `serve_agent` | `true` | native agent routes; `false` = inference + files only |
| `serve_models` | `[]` | allowlist names/globs; empty = every model |
| `serve_workspace` | startup cwd | tool path-jail root |
| `serve_body_limit` | 32 MiB | request body cap (chunked included) |
| `serve_request_timeout` | 30 s | per-operation socket timeout |
| `serve_sse_ping_interval` | 15 s | SSE heartbeat |
| `serve_shutdown_timeout` | 10 s | bounds each shutdown stage |
| `serve_unload_on_exit` | `true` | unload the used models on exit |
| `serve_max_messages` | 16384 | messages per request |
| `serve_max_content_parts` | 4096 | content parts per request |
| `serve_max_message_bytes` | 8 MiB | one message's UTF-8 bytes |
| `serve_max_text_bytes` | 32 MiB | total resolved text per request |
| `serve_max_images` | 64 | images per request |
| `serve_max_image_bytes` | 64 MiB | cumulative decoded image bytes |
| `serve_max_active_turns` | 4 | concurrent turns (429 past it) |
| `serve_turn_event_buffer` | 10000 | per-turn event ring |
| `serve_turn_event_ttl` | 86400 s | completed-turn retention; 0 = keep |
| `serve_approval_timeout` | 300 s | parked approval deadline; 0 = wait forever |
| `serve_turn_timeout` | 0 | wall-clock turn cap; 0 = no cap |
| `serve_artifact_root` | `~/etc/data/ollamarun/artifacts/` | store root |
| `serve_artifact_max_bytes` | 16 MiB | per-file cap; 0 = unlimited |
| `serve_artifact_quota_bytes` | 2 GiB | store quota; 0 = unlimited |
| `serve_artifact_ttl` | 7 days | retention; 0 = keep forever |
| `serve_fit` | `probe` | `off` / `estimate` / `probe` |
| `serve_max_active_fits` | 2 | concurrent fits (429 past it) |
| `serve_fit_ttl` | 86400 s | completed-fit retention; 0 = keep |
