# Vision

`image_url` content parts accept `data:` URIs or http(s) URLs (fetched with
explicit timeouts, at most 3 redirects and a 10 MiB cap — the SSRF caveat is
the locked trusted-network boundary). A `{"type": "file", "file_id":
"art_…"}` part resolves an uploaded artifact instead; text-ish files are
inlined as a `<file name="…">` block, images go through the same vision
gate, and other binaries are described but never pretended to be processed —
see [Artifacts](../agent/artifacts.md).

A real request and answer (red square, `minicpm-v4.6:1b`):

```console
$ B64=$(base64 -w0 red.png)
$ curl -s http://127.0.0.1:11440/v1/chat/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"minicpm-v4.6:1b\",
    \"messages\": [{\"role\": \"user\", \"content\": [
      {\"type\": \"text\", \"text\": \"What color fills this image? Reply with one word.\"},
      {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}]}],
    \"max_tokens\": 256}"
```

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "red",
      "reasoning_content": "Got it, let's look at the image. The image is filled with a red color. …"
    },
    "finish_reason": "stop"
  }],
  …
}
```

Two practical notes from that run: the model's thinking came back as
`reasoning_content` (non-standard, ignorable), and 32 max tokens were not
enough — the thinking consumed the budget and `content` came back empty with
`finish_reason: "length"`. Give a multimodal thinking model headroom.

## The capability gate

Content parts are validated against the model's `capabilities`, so a
text-only model is refused before inference:

```console
$ … -d '{"model": "llama3.2:3b", "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Color?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]}]}'
{"error": {"message": "this model does not support image input",
 "type": "invalid_request_error", "param": "messages",
 "code": "model_does_not_support_images"}}          # HTTP 400
```

The CLI takes images too — `--image PATH` / `/image PATH… | clear` — see
[Images and vision(../../cli/images.md).
