# Usage and flags

```console
$ ollamarun MODEL [PROMPT] [--host HOST | --remote NAME] [--ui MODE]
                  [--pull] [--sort KEY] [--gpu-layers SPEC]
                  [--num-ctx [N | max-gpu | --max-cpu N] [--probe]]
                  [--think [STR]] [--render MODE] [--option K=V]
                  [--image PATH] [--system TEXT | --system-file FILE | --no-system]
                  [-c [ID] | --session NAME | -r] [--new] [--title TEXT] [--tag TAG]
                  [--tools | --no-tools] [-y] [--speak | --no-speak]
                  [--list | --show ID | --search TEXT | --export ID] [--export-format FMT]
                  [--delete ID [-y]] [--reindex] [--out FILE] [--no-save]
                  [--json | --csv | --markdown | --raw] [--verbose]
                  [--show-thinking | --hidethinking] [--no-math] [--no-tps]
                  [--unload | --keep-loaded] [--pager] [--timeout SECONDS] [--debug]
                  # ollama-compatible: --dimensions --format --hidethinking --insecure
                  #   --keepalive --nowordwrap --think --truncate --verbose
                  # Ctrl+T expands/collapses the thinking panel while streaming
```

With no `PROMPT`, `ollamarun` starts the interactive [REPL](repl.md); with
one, it is a single generation that then exits. A piped stdin works as the
prompt too. Where each flag leads:

- model and host: `--host`, `--remote` — [Remote inference](remote.md)
- placement and context: `--gpu-layers`, `--num-ctx`, `--probe` —
  [Context and placement](context/overview.md)
- output: `--render`, `--pager`, `--no-math`, `--no-tps`,
  `--show-thinking`/`--hidethinking`, `--json`/`--raw`/`--verbose` —
  [Rendering](rendering/markdown.md)
- tools: `--tools`/`--no-tools`, `-y` — [Tools](tools/index.md)
- sessions and inspection: `-c`, `--session`, `-r`, `--new`, `--title`,
  `--tag`, `--list`, `--show`, `--search`, `--export`, `--delete`,
  `--reindex`, `--no-save`, `--out` — [Sessions](sessions/storage.md)
- speech: `--speak`/`--no-speak` — [Slash commands](commands.md)
- housekeeping: `--pull`, `--sort`, `--unload`/`--keep-loaded` —
  [Models](models/picker.md)
