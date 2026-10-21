# Background sessions

Every interactive inline session runs in its own **session host**: a detached
process that owns the conversation, the slash commands, the agent turn and its
tool loop. The terminal is a thin client attached to it. Closing the window,
losing SSH, or detaching with `Ctrl+C` does not interrupt a turn — the host
keeps working, and `ollamarun -c` picks the session back up.

The host is per session and per machine. It starts on demand, survives the
terminal, and ends only on an explicit close:

- `/bye`
- `Ctrl+D` at the prompt
- `ollamarun --close-session ID`
- `ollamarun --delete ID` (stops the host before trashing the session)
- `SIGTERM`/system shutdown

## Detaching while work is running

While a turn or a command is running, `Ctrl+C` asks once:

```text
still running — press Ctrl+C again within 10s to detach; the session keeps
going in the background
```

A second `Ctrl+C` inside those 10 seconds closes the terminal and prints the
reattach command:

```text
detached · the session keeps running · reattach with `ollamarun -c 20261021-074254-e164`
```

The turn keeps streaming in the host. Nothing is cancelled and the response is
saved when it finishes. At an idle prompt `Ctrl+C` keeps its old job: it
clears the input line. `Ctrl+D` and `/bye` are the explicit **close** actions —
they stop the turn and shut the host down (unloading the model per
`unload_on_exit`).

## Reattaching

`ollamarun -c` (or `-c ID`, `--session NAME`, `-r`) attaches to the host when
one is running for that session:

- a turn still streaming is replayed from the start and continues live;
- a turn that finished while no terminal was attached is shown under
  “while you were away”, with its thinking and timings;
- a command that was running (`/pull`, `/summarize`, a probe, …) replays its
  buffered output and continues;
- a pending tool approval or model/context question is asked again.

Closing the terminal at the prompt simply leaves the host waiting; the next
attach gets the same prompt. A half-typed line is the only thing lost.

## Taking over

A second `ollamarun -c ID` **takes over**: the new terminal becomes the
attached client, and the old one prints `session moved to another terminal`
and exits. The session, its buffered output and any in-flight turn or
question move to the new terminal. This is how you continue from another
machine's terminal, or recover when the original window is unreachable.

## Commands run in the host

Because the whole REPL lives in the host, commands are not tied to the
terminal either: `/pull`, `/model`, `/summarize` (including automatic
compaction), `/save`, `/load`, `/branch`, `/export`, `/tools`, `/tts`, context
fitting and the rest keep running after a detach. Their output is buffered
and replayed on the next attach. The same applies to the auto-title call and
the model preload.

A tool approval interrupts the turn until someone answers. With no terminal
attached the turn parks in `waiting_approval` and waits; reattach and the
question is shown again. `background_approval_timeout` can turn that into a
denial after N seconds; the default `0` waits indefinitely.

## Turning it off

```console
$ ollamarun --no-background MODEL
```

runs the inline REPL in the foreground exactly as before. The config key
`background_sessions: false` and the `OLLAMARUN_NO_BACKGROUND=1` environment
variable do the same. `--no-save` runs cannot be resumed, so they stay in the
foreground too.

One-shot prompts and the Textual TUI are unchanged by this feature. If a
background host is already serving a session, a one-shot or TUI run for that
session refuses to start (it would be a second writer) and points at
`ollamarun -c` or `--close-session`.

## Inspecting and closing

The host keeps a small socket and metadata file under
`~/.cache/ollamarun/session-hosts/` (mode `0700`/`0600`, same user only). The
session itself still lives in the normal store. To close a session without
attaching:

```console
$ ollamarun --close-session 20261021-074254-e164
closed 20261021-074254-e164
```

`--delete ID` stops a running host first, then moves the session to the
trash. A host is per machine and per login session: a reboot or a full logout
ends it, but the completed turns are already in the session file, and the
next `-c` starts a fresh host for it.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `background_sessions` | `true` | Use a detached session host for the inline REPL. |
| `background_approval_timeout` | `0` | Seconds a detached tool approval waits before being denied; `0` waits forever. |
| `background_event_buffer` | `20000` | Replay buffer size for a running turn (coalesced deltas). |
| `background_event_ttl` | `300` | How long a finished turn stays replayable in the host. |

`--no-background` overrides the config for one run. The host also honours the
normal `unload_on_exit`/`--keep-loaded`/`--unload` model housekeeping at
explicit close.
