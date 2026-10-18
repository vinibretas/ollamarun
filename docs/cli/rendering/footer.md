# Footer

## Live tok/s and context

```text
~24.0 tok/s (avg 21.4) · 45 tok · ctx 12.345k/32.768k (38.123%)
```

The 5s windowed rate, its cumulative average, the streamed token count and a
context counter that ticks with every token. Its denominator is the window the
model is actually loaded with (`/api/ps`; a session `num_ctx` wins, the
architectural maximum is only the fallback), and counts read raw below 1k /
3-decimal k above. The TUI status bar shows the same rate, average and live
ctx. `--no-tps`/`/tps` switch the footer off.

## Timings

`⏱ load … · prompt … · eval … · total …` prints under every answer on a
terminal; `--verbose` is obsolete there and now only adds the line to
piped/redirected output. `--raw`/`--json` never get it, and config
`show_stats: false` turns it off.
