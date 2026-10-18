# Images and vision

Images are attached to the next message as vision input:

- `--image PATH` (repeatable) at launch, or `/image PATH… | clear` in the
  REPL.
- Under [`--remote`](remote.md), sessions, files and images stay local — the
  image is sent with the request.
- Config `image_storage` and `max_file_kb` control how they are kept and
  bounded.

The API surface takes images as `image_url` or `file` parts — see
[Vision](../api/openai/vision.md).
