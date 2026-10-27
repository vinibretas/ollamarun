# Waiting for the model

A request that has not produced a token yet can be doing several very
different things, and they used to look identical:

```text
⠋ waiting for qwen3:8b…
```

Now the line names the stage:

```text
⠋ loading qwen3:8b · GPU 2.0 GiB + RAM 1.1 GiB of ~5.2 GiB
⠋ waiting for room · llama3:8b resident
⠋ making room · unloaded llama3:8b
⠋ reloading qwen3:8b · 4,096 → 8,192
⠋ prefill · ~1,234 tok
```

It shows up in three places, all fed by the same string: the spinner above a
streaming answer, the right-side hint on the prompt while a background
preload runs, and the status bar of the deprecated [TUI](../../tui/index.md).

## Startup is not a guess

`ollamarun` preloads the session's model in the background so the prompt
appears at once. That load is something the REPL started itself, so the line
says `loading qwen3:8b` from the first frame — before the first hardware
sample answers, and even when VRAM is allocated in one burst with nothing to
watch between samples. The elapsed suffix (`loading qwen3:8b · 7s`) is the
progress signal; the footprint (`GPU … + RAM … of ~…`) appears as soon as the
runner process can be matched.

## The idle note

When the resident runner's window or placement differs from what the next
request will ask for, `/num-ctx` and `/gpu-layers` print
`resident at 4,096 · reloads at 8,192 on send` and the prompt keeps showing it
(without a spinner — nothing is running yet). The note disappears when the
request is actually sent; the sampler snapshots the pending change first, so
the wait itself still says `reloading qwen3:8b · 4,096 → 8,192`.

## What each stage means

| Stage | Reading | What is happening |
| --- | --- | --- |
| `loading` | not in `/api/ps`; its runner process exists, or this REPL started the load | The runner is up and reading weights; `GPU … + RAM … of ~…` is the footprint so far against the model's size (RSS is touched pages, so the total is a floor). A preload is known before any probe answers, so the line is there from the first frame. |
| `waiting` | not in `/api/ps`, no runner of ours | Ollama has not started our runner: it is either about to (`· starting runner`) or blocked behind another model (`· <model> resident`, or `· another model is loading` when a foreign runner just spawned). |
| `evicting` | not in `/api/ps`, another resident vanished | Room is being made: a model that was resident when the wait started is gone. |
| `reloading` | in `/api/ps`, a request is open, the loaded config differs | The runner is being replaced at a new window or GPU split. |
| `prefill` | in `/api/ps`, request open, config matches | The model is warm and the prompt is being processed; `~N tok` is what the request carries. |
| `reload` | in `/api/ps`, idle, config differs | Nothing is in flight yet: this is the note `/num-ctx` and `/gpu-layers` print (`resident at 4,096 · reloads at 8,192 on send`). |
| `unreachable` | `/api/ps` failed | The host did not answer; the line says so instead of pretending to know. |

## Where the evidence comes from

1. **`/api/ps`** is the residency truth. An entry exists only once the runner
   is loaded and carries `context_length`, so "resident at 4,096 while the
   session asks for 8,192" is exact, not a guess.
2. **The runner process.** Ollama starts one `llama-server` per model. The
   blob paths `/api/show` prints in the generated Modelfile
   (`FROM …/blobs/sha256-…`) also appear on that runner's command line, which
   is what ties the process to *your* model; its per-process VRAM comes from
   `nvidia-smi --query-compute-apps` and its RSS from `/proc`. A load shows
   progress here long before `/api/ps` lists anything.
3. **The machine** — free VRAM/RAM and CPU busy — is a tiebreaker only. A GPU
   allocation happens in one burst, so sampling system-wide deltas misses it
   almost every time.

Placement changes (`/gpu-layers`) are not visible in `/api/ps`, so the REPL
remembers that the stored value differs from what the resident runner was
loaded with and reports it until the next request lands.

## Cost and switches

Sampling is one `/api/ps` plus one hardware probe per interval, on a daemon
thread, and only while a wait is active. It stops at the first token — a
running generation is not a wait — and headless `ollamarun serve` turns never
sample at all.

| Key | Default | What |
| --- | --- | --- |
| `wait_stages` | `true` | Turn the sampling (and the stage text) on or off. |
| `wait_stage_sample_seconds` | `1.0` | Local cadence (`/proc` + nvidia-smi). |
| `wait_stage_remote_seconds` | `5.0` | `--remote` cadence — each sample is one ssh round trip. |

Headless `ollamarun serve` turns never sample, and neither does piped output
with no terminal attached.
