# Streaming markdown

Output streams through a `rich.Live` view with a throttled markdown re-render
(raw streaming past `max_render_chars`, default 12000); `--pager` buffers into
the pager, `--render markdown|plain|raw` picks the mode.

- [2D math](math.md) — the display-math layout engine.
- [Thinking panel](thinking.md) — the collapsible reasoning preview.
- [Footer](footer.md) — tok/s, the context counter and timings.
- [Transcript replay](transcript.md) — saved sessions rendered the way they
  looked live.
