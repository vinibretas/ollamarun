# Parameters

Session parameters are Ollama options that ride every request:

- `/set parameter NAME VALUE` sets one, `/params` shows them, and
  `--option K=V` seeds them at launch.
- They live in the session, so they persist across resume, `/save` and the
  API's session records (`settings.parameters`).
- `num_gpu` set here is overridden by a pinned
  [gpu-layers](../context/gpu-layers.md) value.

`/show parameters` shows the model's own parameters. System prompts are set
separately — see [Prompts](../../config/prompts.md).
