# Config file

One JSON file holds every persistent setting:

| Path | What |
| --- | --- |
| `~/etc/config_files/ollamarun.json` | The config (created from the defaults on first run); `--config FILE` uses another file instead. |
| `~/etc/config_files/ollamarun-defs/{tools,prompts}/` | Per-user [tool](prompts.md#tool-wording) and [prompt](prompts.md) overrides. |

User values merge over the defaults key by key, so a config file only needs
the keys you actually change (and `--config` demonstrates it: a file
containing just `{"render": "raw", "keepalive": "2h", "show_stats": false}`
made a one-shot print only `pong`).

## Every key

Defaults are `DEFAULT_CONFIG` in `ollamarun_lib/constants.py`. `null` means
"not set" wherever a parameter is optional.

### Model, server and prompt

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | `null` | Ollama address; `OLLAMA_HOST` and `--host` override it, `--remote` wins outright. |
| `remote` | `null` | A `--remote` NAME to make the default. |
| `remote_hosts` | `{}` | Your own short names for tailnet machines. |
| `remote_ssh_user` | `null` | ssh user for the remote probes (disk/GPU/RAM). |
| `model` | `null` | Default model, used when none is given. |
| `system` | `null` | System prompt; `null` = built-in default, `""` = none, text = replaces. |

### Rendering and input

| Key | Default | Meaning |
| --- | --- | --- |
| `render` | `"markdown"` | `auto`/`markdown`/`plain`/`raw` — see [Streaming markdown](../cli/rendering/markdown.md). |
| `math` | `true` | LaTeX → Unicode conversion. |
| `thinking` | `"collapsed"` | Thinking display: `collapsed`/`expanded`/`hidden`. |
| `tps_footer` | `true` | The live tok/s + ctx footer. |
| `show_stats` | `true` | The `⏱` timings line (on a terminal; piped needs `--verbose`). |
| `pager` | `false` | Buffer responses into `$PAGER`. |
| `max_render_chars` | `12000` | Past this, the live view streams raw instead of re-rendering markdown. |
| `ui` | `"inline"` | `inline` (default) / `tui` (deprecated) / `auto`. |
| `nowordwrap` *(flag only)* | — | `--nowordwrap` disables wrapping; no config key. |

### Sessions and files

| Key | Default | Meaning |
| --- | --- | --- |
| `auto_title` | `false` | Ask the model for a title after the first turn. |
| `auto_compact` | `"ask"` | `ask` = warn at 80% only; `auto` = also compact at 85%. |
| `max_file_kb` | `256` | `@file` mention size cap. |
| `image_storage` | `"embed"` | `embed` keeps image bytes in the session; other values keep the path for files over 8 MiB. |
| `screenshots_dir` | `"~/Pictures/Screenshots"` | The virtual `/` root for `/image`: `/NAME` and Tab completion read it first; other paths keep their normal meaning (`~` expands). |

### Runtime

| Key | Default | Meaning |
| --- | --- | --- |
| `options` | `{}` | Default model options (`--option` maps here). |
| `think` | `null` | Default thinking mode. |
| `keepalive` | `null` | Default `keep_alive`. |
| `gpu_layers` | `null` | Default placement spec. |
| `timeout` | `600` | Request timeout, seconds. |
| `unload_on_exit` | `true` | Unload the session model when the REPL exits. |

### Background sessions

See [Background sessions](../cli/sessions/background.md) for the lifecycle.

| Key | Default | Meaning |
| --- | --- | --- |
| `background_sessions` | `true` | Run the inline REPL in a detached session host. |
| `background_approval_timeout` | `0` | Seconds a detached approval waits before denial; `0` waits forever. |
| `background_event_buffer` | `20000` | Replay buffer size for a running turn. |
| `background_event_ttl` | `300` | Completed-turn retention in the host, seconds. |

### Tools and speech

| Key | Default | Meaning |
| --- | --- | --- |
| `tools` | `true` | Offer the tool harness. |
| `tools_policy` | `"allowlist"` | `allowlist`/`ask`/`auto` — see [Approvals](../cli/tools/approvals.md). |
| `tools_shell_allow` | `[]` | Extra commands for the shell allowlist. |
| `tools_timeout` | `120` | Per-tool subprocess timeout. |
| `tools_max_output` | `4000` | Chars of tool output that go back to the model. |
| `tools_max_rounds` | `8` | Tool rounds per turn. |
| `tool_colors` | `{}` | Per-tool panel colours, e.g. `{"run_shell": "green"}`. |
| `tts_auto`, `tts_voice`, `tts_lang` | `false`, `null`, `null` | Auto-speak and its defaults. |
| `tts_remote` | `"tts-remote"` | The speech CLI to invoke. |
| `tts_tools` | `true` | Offer `speak`/`list_voices`. |
| `chem_inventory` | `"chemInventoryAPI"` | The inventory CLI path. |
| `chem_tools` | `true` | Offer the `chem_*` tools. |
| `pdf_vision` | `true` | Let `read_pdf` read scanned pages with a vision model. |
| `pdf_vision_model` | `null` | Pin the vision model; `null` picks the best installed one. |
| `pdf_vision_switch` | `true` | Allow unloading the session model to load the vision one. |
| `pdf_vision_pages` | `10` | Most pages one `read_pdf` call rasterizes. |

### Context fits

| Key | Default | Meaning |
| --- | --- | --- |
| `max_gpu_margin_mb` | `1024` | llama.cpp's fit margin, subtracted from free VRAM. |
| `max_gpu_overhead_mb` | `0` | Extra GPU reserve beyond the margin. |
| `max_gpu_probes` | `10` | Load attempts a `--probe` may spend. |
| `max_cpu_overhead_mb` | `256` | Host-RAM allowance beyond weights + KV. |
| `max_cpu_probes` | `10` | CPU probe attempts. |

### Serve keys

`ollamarun serve` **validates** every `serve_*` key at startup: an invalid
value names itself and exits 2 —

```console
$ ollamarun serve --config bad.json      # {"serve_port": 999999}
serve_port must be <= 65535
$ echo $?
2
```

— an omitted (`null`) key uses the default, and a `0` means exactly what the
row says.

| Key | Default | Meaning / zero |
| --- | --- | --- |
| `serve_bind`, `serve_port` | `0.0.0.0`, 11440 | listener; port 0 = ephemeral |
| `serve_agent` | `true` | native agent routes; `false` = inference + files only |
| `serve_models` | `[]` | names/globs; empty = every model |
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

What those boundaries mean in practice is on
[Security and limits](../api/security.md), the fits on
[Context fits](../api/agent/fits.md).
