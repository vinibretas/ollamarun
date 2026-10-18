# Usage and flags

Every `ollama run` flag (ollama 0.34) is accepted, and the harness adds its
own on top. Three invocation shapes:

```console
$ ollamarun [flags] MODEL [PROMPT]      # flags first
$ ollamarun MODEL [PROMPT] [flags]      # or after the prompt
$ ollamarun                             # no model: picker, then the REPL
```

With a prompt the run is one generation and exits; without one it is the
interactive [REPL](../cli/repl.md) (a piped stdin counts as the prompt too).
Flags may sit anywhere relative to `MODEL` and `PROMPT` — `ollamarun
llama3.2:3b --no-tools "hi"` runs exactly like the forms above.

The reference below is the same surface as `ollamarun --help`, grouped by
what the flags are for.

## Model and server

| Flag | Default | Meaning |
| --- | --- | --- |
| `MODEL` (positional) | — | Model name; omit it to open the picker. |
| `-m MODEL`, `--model MODEL` | — | Same as the positional, for scripts that build arguments up. |
| `--host HOST[:PORT]` | `OLLAMA_HOST` or the `host` config key, then `127.0.0.1:11434` | The Ollama server to talk to. |
| `--remote NAME` | — | Run on a tailnet machine (`pi`, `kiddo`, `g3`, `optiplex7070`/`optiplex`, `guri`, or any `host[:port]`). See [Remote inference](../cli/remote.md). |
| `--pull` | off | Pull the model when it is missing without asking (free-disk line and per-layer progress). See [Pulling models](../cli/models/pulling.md). |
| `--sort KEY` | `modified` | Order for the model picker and lists: `modified`, `name`, `size`, `parameters`, `context` (aliases like `recent`/`newest` work). See [Picker and sorting](../cli/models/picker.md). |
| `--insecure` | off | Skip TLS verification for an HTTPS host (the same escape hatch `ollama` has). |

## Prompt input

| Flag | Default | Meaning |
| --- | --- | --- |
| `--system TEXT` | built-in default | Replace the system prompt for this run. See [Prompts](../config/prompts.md). |
| `--system-file FILE` | — | Read the replacement from a file. |
| `--no-system` | — | Send no system prompt at all. |
| `--option KEY=VALUE` | — | A model option (`-o temperature=0.7`); repeatable. |
| `--options-json JSON` | — | Model options as one JSON object. |
| `--image PATH` | — | Attach an image for a vision model; repeatable. See [Images and vision](../cli/images.md). |

`@path` mentions inside the prompt inline files — see
[@file mentions](../cli/file-mentions.md).

## Context and placement

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-ctx [N\|max-gpu]` | model/server default | Set the context window, or ask for the largest one that stays fully on the GPU. Capped to the model's trained maximum with a warning. |
| `--max-cpu N` | — | With `--num-ctx`: raise the window until the model's host RAM fills N% (1-100) of the available DRAM. |
| `--probe` | off | With `--num-ctx max-gpu`/`--max-cpu`: verify by loading the model and reading the split back from `/api/ps`. |
| `--gpu-layers SPEC` | `auto` | `N` layers, `P%` of them, `+N`/`-N` adjustments, `all`, `cpu` or `auto`. |

All of this (and how the fits are computed) lives in
[Context and placement](../cli/context/overview.md).

## Output and rendering

| Flag | Default | Meaning |
| --- | --- | --- |
| `--render {auto,markdown,plain,raw}` | `auto` | `auto` = markdown on a terminal, plain when piped (an explicit `markdown` degrades to plain off-terminal too). |
| `--no-math` | off | Disable LaTeX → Unicode conversion. See [2D math](../cli/rendering/math.md). |
| `--show-thinking` | collapsed | Start with the thinking panel expanded. |
| `--hidethinking` | — | Hide the thinking output. |
| `--no-tps` | off | Hide the live tok/s + context footer. |
| `--nowordwrap` | off | Don't wrap long lines to the terminal width. |
| `--pager` | off | Buffer the answer into `$PAGER` (rich's pager) instead of streaming it in place. See [Pager](../cli/pager.md). |
| `--raw` | — | Write only the answer text to stdout (no panels, no stats). |
| `--json` | — | With a chat run: print the final Ollama response as JSON (model, message with `thinking`, usage, timings). With `--list`/`--search`: JSON output. |
| `--verbose` | off | Obsolete for a terminal (timings always show there); adds the `⏱` timings line to piped/redirected output. |

## Tools and speech

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tools` | on (config `tools`) | Offer the built-in tool definitions to the model. |
| `--no-tools` | — | Disable the tool harness for this run. |
| `-y`, `--yes` | off | Auto-approve tool runs (and `--delete`). Use with care — see [Approvals and policy](../cli/tools/approvals.md). |
| `--speak` | config `tts_auto` | Read every final answer aloud through `tts-remote` (fire-and-forget). |
| `--no-speak` | — | Never auto-speak for this run. |

## Sessions

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c [SESSION_ID]`, `--continue [SESSION_ID]` | — | Continue the latest session, or the given one (exact id, id prefix, or exact title with a prompt). |
| `-r`, `--resume` | — | Browse saved sessions and pick one. |
| `--session NAME` | — | Continue the session saved under `NAME`, or create it. |
| `--new` | — | Force a fresh session even with `-c`/`--session`. |
| `--title TEXT` | — | Title for a new session. |
| `--tag TAG` | — | Tag for the session; repeatable. |
| `--auto-title` | config `auto_title` | Ask the model for a title after the first turn. |
| `--no-save` | off | Keep this run out of the session store entirely. |

Sessions and their files are covered in
[Sessions](../cli/sessions/storage.md).

## Session inspection

These all share one output convention — a rich table on a terminal, or
`--json` / `--csv` / `--markdown` with `--out FILE` for scripts:

| Flag | Meaning |
| --- | --- |
| `--list` | List saved sessions. |
| `--search TEXT` | Search messages across saved sessions. |
| `--show SESSION_ID` | Replay a saved transcript the way it looked live. |
| `--export SESSION_ID` | Export a session. |
| `--export-format {md,json,html}` | Export format (default `md`). |
| `--delete SESSION_ID` | Move a session to the trash (`trash-put`; asks first unless `-y`). |
| `--reindex` | Rebuild the session index from the session files. |
| `--out FILE` | Write the inspection output to a file instead of stdout. |

See [Search and export](../cli/sessions/search-export.md).

## Runtime and housekeeping

| Flag | Default | Meaning |
| --- | --- | --- |
| `--timeout SECONDS` | `600` | Per-request timeout. |
| `--keepalive DURATION` | config `keepalive` | How long Ollama keeps the model loaded (`5m`, `2h`, …). |
| `--think [STR]` | config `think` | Thinking mode: `true`/`false` or `high`/`medium`/`low` for supported models. |
| `--unload` | off | Unload the model when the program ends (one-shot included). |
| `--keep-loaded` | — | Never unload on exit (overrides config `unload_on_exit`). |
| `--ui MODE` | `inline` | `inline`, `tui` (deprecated) or `auto`. See [TUI](../tui/index.md). |
| `--config FILE` | `~/etc/config_files/ollamarun.json` | Use a different config file. |
| `--debug` | off | Log to `~/etc/logs/ollamarun.log`. |
| `--version` | — | Print the version and exit. |

## Ollama-compatible pass-through

Accepted for parity with `ollama run` and mapped onto the harness: `--think`,
`--keepalive`, `--hidethinking`, `--nowordwrap`, `--verbose` (see above),
plus:

| Flag | Meaning |
| --- | --- |
| `--dimensions int` | Truncate output embeddings to this dimension (embedding models). |
| `--truncate [bool]` | For embedding models: truncate inputs over the context window (default `true`; `--truncate=false` errors instead). |
| `--format string` | Response format — `json`, or an inline JSON schema (Ollama's `format`). With a format set, the tool harness steps aside. |
| `--insecure` | Skip TLS verification (HTTPS hosts). |

## Environment

| Variable | Meaning |
| --- | --- |
| `OLLAMA_HOST` | Server address when neither `--host` nor `--remote` is given. |
| `OLLAMA_EDITOR` | Editor for Ctrl+G and `/edit` (falls back to `$EDITOR`, then `vi`). See [Editor](../cli/editor.md). |
| `OLLAMA_NOHISTORY` | Do not keep the prompt input history. |

## Exit codes

| Code | When |
| --- | --- |
| 0 | Success — including a declined delete (`kept`) and a one-shot whose answer arrived. |
| 1 | Runtime failures: unknown session, connection/HTTP errors, a run that could not complete. |
| 2 | Usage and configuration errors: bad arguments, an invalid `serve_*` key, an unusable `--config`. |
| 130 | The run was cancelled (Ctrl+C). |
