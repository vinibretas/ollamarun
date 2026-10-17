# Phase 7 — Testing and compatibility

Fold into `test_ollamarun.py` (add `test_*` functions and wire them into the total). If it grows too large, add `test_ollamarun_serve.py` and call it from the same command.

**Runtime:** normal inference, multi-step tools, tool failure, model failure, approval-required, approval-timeout, session persistence/recovery, cancellation, interrupted streams, model vs agent mode isolation.

**Events:** serialization determinism, seq/lifecycle, replay, subscriber disconnect, ring-buffer truncation.

**Lifecycle:** end-to-end turn state transitions (created → running
→ waiting_approval → completed/failed/cancelled), replay/`stream.gap`
recovery through the turn status endpoint, disconnect/reconnect
policies for model-only vs native turns, approval timeout/duplicate
resolution, and shutdown while waiting.

**HTTP:** schemas, SSE framing and termination, tool-call serialization, error status/body, no-auth behavior (trusted-network mode), body limits, concurrency and session isolation, static path traversal.

**Artifacts:** upload, fetch, path/ID validation, path traversal, provenance, TTL/quota.

**Compatibility:** an optional test using the official `openai` SDK (skip if the package is absent; it’s a test-only dependency, not a runtime one). Record unsupported fields and edge cases in the README rather than claiming full compatibility.

**CLI regression:** all existing tests, plus a headless runtime test proving no stdout and a clean event sequence.

**Acceptance:** automated tests cover the critical execution and
security paths; the compatible API works with at least one existing
OpenAI SDK client; the CLI keeps passing its tests; streaming is
verified for both normal completions and failures.
