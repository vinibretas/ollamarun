# Approvals

`settings.approval` decides who answers a tool call:

- `"policy"` (the default) applies the configured tool policy — allowlisted
  and rule-covered calls run, everything else is denied with a note. The
  serve policy is never `auto`; `--yes` remains the only CLI auto path.
- `"interactive"` parks genuinely undecided calls: the turn enters
  `waiting_approval` and emits `approval.required` with the `approval_id`,
  the call, the reason and suggested rules.

`POST /v1/agent/approvals/{id}` resolves a parked decision. The first decision
wins; duplicates get `409`. An unanswered approval times out (config
`serve_approval_timeout`, default 300 s) as **denied**: the tool result says
`not run: approval timed out` and the model continues.

The parked event carries everything needed to decide (the client already saw
it on the stream, as `approval.required`):

```json
{"name": "run_shell", "args": {"command": "rm -rf build"},
 "detail": "rm -rf build", "fingerprint": "…", "reason": "catastrophic rm",
 "approval_id": "apr_…",
 "suggestions": [{"kind": "shell", "pattern": "rm -rf build",
                  "base": "", "created": "…"}]}
```

The decision is one POST — the body's `action` picks the outcome:

```console
$ curl -s http://127.0.0.1:11440/v1/agent/approvals/apr_… \
    -H 'Content-Type: application/json' -d '{"action": "run"}'
{"id": "apr_…", "object": "approval", "turn_id": "trn_…", "session_id": "ses_…",
 "name": "run_shell", "status": "resolved", "action": "run", "run": true,
 "note": "confirmed", "resolved_at": "…"}
```

| Request body | Effect |
| --- | --- |
| `{"action": "run"}` | Run this call once |
| `{"action": "remember"}` | Run it, remembering the exact call for the session |
| `{"action": "rule", "rule": {"kind": "shell", "pattern": "make :*"}}` | Run it and store a generalized rule (`call`, `shell`, `path` or `python`) |
| `{"action": "auto"}` | Allow everything for the rest of this turn |
| `{"action": "deny"}` | Refuse it; the model gets a not-run note |

The rules are the same engine the CLI uses — see
[Approvals and policy](../../cli/tools/approvals.md).
