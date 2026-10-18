# Search and export

- Inspect: `--list`, `--show ID`, `--search TEXT`, `--export ID`
  (`--export-format md|json|html`), `--delete ID [-y]`, `--reindex` — all with
  the shared output convention: rich table by default, `--json`/`--csv`/
  `--markdown` plus `--out FILE` to redirect.

`--show ID` replays the transcript the way it looked live — see
[Transcript replay](../rendering/transcript.md). In the REPL, `/sessions
[filter]` browses and switches, and `/export [--format md|json|html] [FILE]`
exports the current session.
