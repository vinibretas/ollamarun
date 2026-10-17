# ollamarun Phase 1b — completion plan

Phase 1 (the `AgentRuntime` extraction, the event vocabulary and the
turn-safety fixes) is implemented and committed. This file closes every
item that was still open in the Phase 1 report: nothing stays
unresolved — each item is either a small Phase 1b task (finished before
Phase 2 starts), a numbered future phase, or a locked decision.

Phase 1b deliberately does not change the CLI/TUI surface, except for
the `@file` bug fix and restoring mid-turn `/render` parity. Every task
keeps the full suite green.

## Closure table

| # | Item | Disposition |
|---|---|---|
| 1 | `_check_tool_support` still on `Repl` | **Phase 1b** — move into `AgentRuntime` with a per-host "models checked" set in `TurnHooks`; emit `turn.warning` after `turn.started` so the ordering contract holds; test it fires once across turns |
| 2 | `approval.required` emitted by the approver, not the runtime | **Phase 1b** — lock the contract in code docs and test it (approver owns `approval.required`; runtime owns `tool.call`/`approval.resolved`; asserted order `tool.call → approval.required → approval.resolved → tool.result`) |
| 3 | Cancellation only reaches `run_shell`/`run_python` | **Phase 1b** — move `run_cancellable` into `helpers.py` (no cycle; tools imports helpers), thread `cancel` through the ssh probes (`ssh_free_disk`, `ssh_gpu_probe`, `ssh_ram_probe`), the chem CLI runner, `run_tts_say`, and `trash-put`; tests for the chem/TTS/ssh/kill paths |
| 4 | Python 3.10 not verified on this box (`/bin/python3.10` → 3.12) | **Phase 1b** — AST guard test `ast.parse(..., feature_version=(3, 10))` over every module + entry script, so no 3.11+ syntax can land. **Phase 7** — real-interpreter smoke if any fleet host that will run `serve` is genuinely 3.10 |
| 5 | `TurnRequest`/`TurnResult`/`to_turn_result()` untested | **Phase 1b** — mapping test (completed/failed/cancelled, usage, rounds, error) |
| 6 | `HeadlessRenderer.context_total` is `None` | **Phase 2** — the turn-metadata block (`context: {used, total, source}`) goes on `turn.started`/`turn.completed`; the renderer field stops being the source of truth |
| 7 | `turn.warning` wasn't in the approved vocabulary | **Locked** — it stays (non-terminal notice); added to the Phase 2 event table |
| 8 | Store lock is in-process only | **Phase 1b** — `fcntl.flock` on `index.json.lock` around the index read-modify-write, plus unique per-writer temp names (`<id>.<pid>.tmp`) so a CLI and `serve` can save different sessions at once; two-`SessionStore` concurrency test |
| 9 | TUI keeps `StreamSink` frames, no `RuntimeEvent` | **Locked** — intentional: events are the semantic layer for non-terminal front ends; the TUI's rich frames + `Ui` approvals stay |
| 10 | `@file` expansion result was discarded (pre-existing bug) | **Phase 1b** — store the expanded text in `_add_user_message`/`run_once` and test `@file` really inlines |
| 11 | `quiet_tools` snapshotted at turn start (mid-turn `/render` parity) | **Phase 1b** — make it a live callable like `tools_enabled`, used by both the loop and the event printer |
| 12 | `mode="model"` defined but unimplemented | **Phase 3** — the `AgentRuntime` model-turn path (single inference, client tools passthrough, no execution) is that phase's deliverable |
| 13 | Server turns never trigger `_context_warning`/auto-compaction | **Phase 4** — the agent-turn controller calls the headless-safe warning/compact step after a turn when configured |
| 14 | pyflakes nits (`ui.py` unused imports, two placeholder-less f-strings) | **Phase 7** — add a pyflakes-clean pass over `ollamarun_lib/` to the test gate and fix them there |
| 15 | `Task was destroyed…` warning at test exit (pre-existing Textual timer) | **Phase 7** — test hygiene: silence it or prove it harmless at suite exit |
| 16 | `HeadlessRenderer.after_tools`/`_committed_final` unused | **Locked** — kept for API symmetry with `Renderer`; documented as such |

## Phase 1b tasks (blocks Phase 2)

1. **Move `_check_tool_support` into `AgentRuntime`.** Call it from
   `run()` after emitting `turn.started`; `TurnHooks` gains a
   `tool_support_checked: set[str]` the host owns (Repl keeps its set,
   the server gets its own). Emit `turn.warning` and keep the console
   print only for non-headless. Test: two turns on a tool-less model
   warn once, and `turn.started` precedes the warning.
2. **`quiet_tools` as a live callable.** `TurnSettings.quiet_tools`
   becomes `Callable[[], bool]`; both the loop's
   `execute_tool_calls(quiet=...)` and `Repl._event_printer` evaluate it
   per use, restoring mid-turn `/render raw` behavior.
3. **Full cancellation coverage.** Move `run_cancellable()` to
   `helpers.py` (raising a helper-level cancellation error that
   `execute_tool` maps to a cancelled `ToolOutcome`). Thread the token
   through the ssh probe helpers, `_chem_run`, `run_tts_say` and the
   `trash-put` call. Tests: ssh/chem/TTS handlers with a pre-cancelled
   token return cancelled outcomes, and a killed shell still reports
   promptly.
4. **Cross-process store lock.** In `SessionStore`, take
   `fcntl.flock` on `index.json.lock` around the index
   read-modify-write (in addition to the existing `RLock`; no-op
   fallback if `fcntl` is unavailable), and write session JSON through a
   per-process temp name (`<id>.<pid>.tmp`) so two writers cannot share
   a temp file. Test: two independent `SessionStore` objects saving
   concurrently keep every index entry.
5. **`TurnRequest`/`TurnResult` tests.** Cover `to_turn_result()`
   mapping for completed, failed and cancelled turns (status, final
   message, usage, rounds, error).
6. **`approval.required` contract.** Document in `runtime.py` (approver
   emits `approval.required`; runtime emits `tool.call` and
   `approval.resolved`) and test the ordering with a stub approver.
7. **Python 3.10 syntax guard.** A test that `ast.parse`s every
   `ollamarun_lib/*.py` and the `ollamarun` entry script with
   `feature_version=(3, 10)`.
8. **`@file` fix.** `_add_user_message` and `run_once` store the
   expanded text, not the raw prompt; test that a message mentioning
   `@file` reaches the session (and the model payload) with the
   `<file …>` block.

## Locked decisions

- `turn.warning` is part of the vocabulary (non-terminal notices).
- The TUI keeps consuming `rich` frames through `StreamSink`; the event
  stream is the semantic layer for non-terminal front ends.
- The approver implementation owns emitting `approval.required` at the
  point it parks the turn; the runtime owns `tool.call`,
  `approval.resolved` and every other event.
- `HeadlessRenderer` keeps `after_tools`/`_committed_final` for API
  symmetry with `Renderer`.

## Moved to future phases

- **Phase 2**: context metadata (`used`, `total`, `source`) on the turn
  events; `TurnManager`, envelopes, `seq`, replay and buffer semantics.
- **Phase 3**: `mode="model"` in `AgentRuntime` for
  `/v1/chat/completions`.
- **Phase 4**: server-side warning/auto-compaction after turns;
  approvals resolved through the API.
- **Phase 7**: real 3.10 interpreter smoke if a fleet host runs
  `serve`; pyflakes-clean gate; Textual pending-timer test-exit
  hygiene.

## Acceptance

- Each Phase 1b task lands with its test; `python3 test_ollamarun.py`
  passes after every task.
- CLI/TUI behavior is unchanged apart from the `@file` fix and the
  restored mid-turn `/render` parity.
- Report files changed, tests run and any new findings before starting
  Phase 2.
