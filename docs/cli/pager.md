# Pager

`--pager` (config `pager: true`) buffers the response and opens it in the
system pager when it is complete, instead of streaming it into the
scrollback:

```console
$ PAGER=cat ollamarun --no-tools --pager llama3.2:3b "Reply with a two-item markdown bullet list of primary colors."
 • Red
 • Blue
```

The pager is the standard one: rich hands the content to `pydoc`'s pager,
which uses `$PAGER` (commonly `less`; `PAGER=cat` above prints it straight
through, which is also what happens when the output is piped). Inside the
pager you scroll and search the way that tool works; quitting it returns to
the prompt.

`/pager [on|off]` toggles it mid-session:

```text
llama3.2:3b › /pager on
pager: on
```

What lands in the pager is the finished answer with the same rendering —
including the thinking panel when it is visible, and the `response` divider
when a tool loop preceded it. What you lose is the live view: while the
model generates, the terminal stays quiet and the [footer](rendering/footer.md)
does not tick. For long answers in a short terminal that trade is worth it;
for watching progress it is not.

- [Streaming markdown](rendering/markdown.md) — the live path.
- [Transcript replay](rendering/transcript.md) — scrolling a session after
  the fact instead of during.
