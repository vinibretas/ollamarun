# Load, unload and residency

Unloading is deliberate, not left to the server:

- `/model` and `/load --model` unload the current model before switching.
- `/load SESSION` prompts current-vs-last model and unloads when switching.
- REPL exit unloads (`unload_on_exit` config, `--keep-loaded` disables,
  `--unload` also unloads after one-shots).
- Embedding models are never unloaded.

`/ps` shows the resident models (ollama ps).
