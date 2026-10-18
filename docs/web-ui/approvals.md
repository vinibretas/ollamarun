# Approvals (planned)

> **Planned, not built.**

Approvals are the one place the web UI must interrupt the flow, so the plan
here is deliberately close to the terminal's behaviour — same engine, same
choices, same record.

## The interaction

1. The turn parks (`waiting_approval`) and the stream emits
   `approval.required` carrying the call, the reason and **suggested rules**
   — everything needed to decide, no round trip.
2. The affected tool card in the transcript becomes a decision prompt with
   buttons mirroring the CLI's numbered choices:
   run once · remember exactly this call · one button per suggested rule ·
   allow everything this turn · deny.
3. `POST /v1/agent/approvals/{id}` resolves it; the first decision wins
   (duplicates `409`), and `approval.resolved` arrives on the stream for
   every subscriber.
4. The tool result is the next `tool.result` — a denial renders as a
   not-run card, not an error.

## Policy mode

With `"approval": "policy"` (the default for non-interactive clients) there
is nothing to click: allowlisted and rule-covered calls run, everything else
is **denied with a note**. The planned UI will surface the note inline, so a
denial is visible instead of looking like a model failure — the same way the
terminal's result panel does it.

## Timeouts and edge cases

- An unanswered approval denies after `serve_approval_timeout` (default
  300 s), with the tool result saying so; the page can show a countdown from
  the event's timestamp.
- Cancelling a turn or shutting the server resolves parked approvals as
  cancelled — the UI must handle browsers that close mid-decision.
- Rules and remembered calls persist on the session, so the next turn's
  stream simply stops emitting `approval.required` for covered calls.

The engine itself — rule kinds, matching, the floor no rule covers — is
documented once, for both front ends, in
[Approvals and policy](../cli/tools/approvals.md).
