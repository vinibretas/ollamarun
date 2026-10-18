# Example clients

Two reference clients under `examples/` double as integration checks — both
print a `PASS`/`SKIP`/`FAIL` line per step and exit non-zero on a failure:

- `examples/openai_client.py` — the official `openai` SDK against the `/v1`
  surface: the model list, a completion, streaming with usage, embeddings
  (the SDK's default base64 encoding), `n = 2` and a tools request. Needs
  the SDK (`pip install --user openai`); pass a chat model:

  ```console
  $ python3 examples/openai_client.py --model qwen3:8b
  ```

- `examples/agent_client.py` — the native agent flow over plain HTTP:
  submit a turn, follow the typed SSE stream, resolve a parked approval,
  read the session summary. `--mode interactive --approve run` exercises
  the approval path; `--session-id` continues an existing session:

  ```console
  $ python3 examples/agent_client.py --prompt "List the files in /tmp"
  ```

Both take `--base-url` for a remote or non-default listener (the OpenAI one
wants the `/v1` suffix).
