# Tool reference

| Tool | What it does | Needs |
| --- | --- | --- |
| `run_shell` | Run a shell command (`bash -lc`), return output and exit code | — |
| `run_python` | Execute a short Python 3 program; the model is told to compute with it, never mentally | — |
| `read_file` | Read a text file, optionally a line range | — |
| `write_file` | Write or append text, creating the file | — |
| `edit_file` | Replace one exact fragment (unique match unless `replace_all`), atomic write keeping the mode; the call panel shows a coloured diff before approval | — |
| `list_dir` | Directory listing with sizes and modification times | — |
| `make_dir` | `mkdir -p` | — |
| `move_file` / `copy_file` | Move/copy; refuse an existing destination unless `overwrite=true` | — |
| `delete_file` | Move to the trash (`trash-put`) — never a permanent unlink | — |
| `publish_artifact` | Publish a generated file (from `run_shell`/`run_python`) as a downloadable artifact; the file tools publish their own results automatically | `ollamarun serve` artifact store |
| `fetch_url` | Fetch one exact URL as readable content — HTML to text plus the page's links, JSON pretty-printed, PDFs text-extracted; a URL already fetched in the session is answered from cache, and failures come back as one actionable line | — |
| `web_search` | Search the web for titles/URLs/snippets (cached; search-engine `q=` URLs are rerouted here) | `ddgs` package |
| `search_sessions` | Search your saved ollamarun conversations (titles, models, messages, thinking) | — |
| `list_models` | Installed models; loaded ones marked with VRAM and unload time (`loaded_only` for just the residents) | — |
| `system_status` | RAM, GPU VRAM and model-disk space of the machine running Ollama (ssh-probed under `--remote`; `target="local"` for this machine) | — |
| `context_status` | The context budget in use: window (labelled source), tokens the last request carried, free space, message counts, wrap-up note past 80% | — |
| `session_info` | Date/time, session id/title, model size/quantization/capabilities, server + cwd, parameters, thinking/render modes, system-prompt status, tool policy and remaining rounds | — |
| `speak` / `list_voices` | Speak text aloud through tts-remote (fire-and-forget), or list its voices | tts-remote |
| `chem_search` / `chem_container` / `chem_locations` | Read-only chemical-inventory lookups through the `chemInventoryAPI` CLI | chemInventoryAPI |

`web_search` only appears when `ddgs` is installed, the speech tools when
`tts-remote` is available, the chem tools when `chemInventoryAPI` resolves
(config `chem_inventory` names a different path, `chem_tools: false` hides
them), and `publish_artifact` when `ollamarun serve` runs an artifact store
(the CLI has none). A tool's `requires` capability gates it the same way for
user-added definitions.
