# Example clients

Two reference clients under `examples/` double as integration checks against
a running `ollamarun serve`. Both print a `PASS`/`SKIP`/`FAIL` line per step
and exit non-zero when a step fails.

## `examples/openai_client.py`

The official `openai` SDK against the `/v1` surface: the model list, a
non-streaming completion, streaming with usage, embeddings, `n = 2`, and a
tools request that the server must accept without executing. Run against a
chat model:

```console
$ python3 examples/openai_client.py --base-url http://127.0.0.1:38573/v1 --model llama3.2:3b
openai_client · SDK 2.29.0 · http://127.0.0.1:38573/v1
PASS  the model list answers
      using model llama3.2:3b
PASS  a non-streaming completion returns content and usage
PASS  streaming deltas arrive and usage closes the stream
SKIP  embeddings return one vector per input — llama3.2:3b is not an embedding model
PASS  n=2 returns two indexed choices
PASS  a tools request is accepted (the server never executes it)
      the model asked for get_weather({"city": "Sao Paulo"})

all checks passed
```

The embeddings step needs an embedding model, so a chat-model run reports
`SKIP` there — by design. Point it at an embedding model to exercise that
step; the chat steps then fail with the sanitized backend 502 (an embedding
model cannot chat), which is the expected outcome of that trade:

```console
$ python3 examples/openai_client.py --base-url … --model nomic-embed-text
PASS  embeddings return one vector per input
FAIL  a non-streaming completion returns content and usage
      InternalServerError("Error code: 502 - {'error': {'message': 'the model
      backend failed to complete the request', …}}")
```

Options: `--base-url` (the `/v1` suffix included), `--model` (defaults to the
first listed), `--api-key` (sent and ignored), `--timeout`.

## `examples/agent_client.py`

The native agent flow over plain HTTP: submit a turn, follow the typed event
stream, resolve a parked approval, print the session summary. The default
policy mode runs allowlisted/read-only tools and denies the rest:

```console
$ python3 examples/agent_client.py --base-url http://127.0.0.1:38573 \
      --model llama3.2:3b --prompt "List the files in the current directory and tell me how many there are."
turn turn_07f31de8504b · session 20261018-133348-fc8d · model llama3.2:3b · approval policy
tool · list_dir {"path": "."}
  → ok ['/tmp/…/serve-ws (2 entries)']
There are 2 files in the current directory.
status completed · rounds 1 · tokens 642
answer There are 2 files in the current directory.
```

`--mode interactive --approve run` exercises the approval path (the client
parks, prints the decision request, posts the decision, and continues):

```console
$ python3 examples/agent_client.py --base-url … --model llama3.2:3b \
      --mode interactive --approve run \
      --prompt "Use write_file to write the text from-the-client to client-demo.txt, then confirm."
turn turn_459cb29e505b · session 20261018-133515-cdb4 · model llama3.2:3b · approval interactive
tool · write_file {"append": false, "content": "from-the-client", "path": "client-demo.txt"}
approval required write_file modifies state
      resolved as run (200)
  → ok ['wrote 15 chars to /tmp/…/serve-ws/client-demo.txt']
```

(The file really was written — checked on disk afterwards.) Other options:
`--session-id` to continue an existing session instead of creating one,
`--approve {run,remember,rule,auto,deny}`, `--no-persist` for an ephemeral
turn, `--json` for machine output, `--timeout`.

Both clients are covered by the test suite (`test_examples`, against a fake
stack), so they cannot rot silently — and they are the shortest way to check
a live deployment by hand.
