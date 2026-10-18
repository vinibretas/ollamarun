# Approvals

`settings.approval` decides who answers a tool call:

- `"policy"` (the default) applies the configured tool policy — allowlisted
  and rule-covered calls run, everything else is denied with a note. The
  serve policy is never `auto`; `--yes` remains the only CLI auto path.
- `"interactive"` parks genuinely undecided calls: the turn enters
  `waiting_approval` and emits `approval.required` with the `approval_id`,
  the call, the reason and suggested rules.

## The parked event

A real capture (`write_file` in interactive mode):

```json
{
  "name": "write_file",
  "args": {"path": "approve-demo.txt", "append": false, "content": "approved-by-api"},
  "detail": "approve-demo.txt (15 chars)",
  "fingerprint": "write_file:{\"append\": false, \"content\": \"approved-by-api\", \"path\": \"approve-demo.txt\"}",
  "reason": "write_file modifies state",
  "suggestions": [
    {"kind": "path", "pattern": "approve-demo.txt", "base": "/home/you/serve-dir", "created": "…"},
    {"kind": "path", "pattern": "**", "base": "/home/you/serve-dir", "created": "…"}
  ],
  "approval_id": "apr_83764cec6535"
}
```

The park is preceded on the same stream by `tool.call` (the call panel
equivalent) and followed, on decision, by `approval.resolved`. The
`suggestions` are the same generalized rules the CLI offers; the fingerprint
powers "remember exactly this call".

## Deciding

`POST /v1/agent/approvals/{id}` — the first decision wins; duplicates get
`409`.

| Request body | Effect |
| --- | --- |
| `{"action": "run"}` | Run this call once. |
| `{"action": "remember"}` | Run it, remembering the exact call for the session. |
| `{"action": "rule", "rule": {"kind": "path", "pattern": "reports/**"}}` | Run it and store a generalized rule (`call`, `shell`, `path` or `python`). |
| `{"action": "auto"}` | Allow everything for the rest of this turn. |
| `{"action": "deny"}` | Refuse it; the model gets a not-run note. |

```console
$ curl -s http://127.0.0.1:11440/v1/agent/approvals/apr_83764cec6535 \
    -H 'Content-Type: application/json' -d '{"action": "run"}'
{"id": "apr_83764cec6535", "object": "approval", "turn_id": "turn_7cc9871fcf7d",
 "action": "run", "run": true, "note": "confirmed",
 "resolved_at": "2026-10-18T13:20:09.745442-03:00"}
```

After `run`, the same turn went on to `tool.result` (`ok`), published the
written file as an [artifact](artifacts.md) and completed with the model's
confirmation. After `deny`:

```json
{"id": "apr_783809fe4258", "object": "approval", "turn_id": "turn_82bf7ce2fbc2",
 "action": "deny", "run": false, "note": "denied by the user", "resolved_at": "…"}
```

…the tool result is a not-run note and the file is **not** written (checked
on disk). The model still answered in that run — models sometimes claim
success anyway; the harness's record (`tool.result`, the session) shows what
actually happened, which is why clients should decide from events, not prose.

## Timeouts and edge cases

- An unanswered approval times out (`serve_approval_timeout`, default 300 s)
  as **denied**: the tool result says `not run: approval timed out` and the
  model continues.
- A turn cancelled or shut down while parked resolves the approval as
  cancelled (`200` with the resolved state), so clients never hang.
- The decision endpoint is per-approval; there is no bulk form — `auto` is
  the per-turn escalation.

The rules engine behind the decisions is the CLI's — see
[Approvals and policy](../../cli/tools/approvals.md) for kinds, matching and
the floor no rule can cover.
