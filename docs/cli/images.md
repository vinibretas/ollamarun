# Images and vision

Images ride along with a message as vision input. They are attached to the
**next** message only — once it is sent, the attachment is spent, and
`/image clear` throws pending ones away.

## Attaching

| Way | When |
| --- | --- |
| `--image PATH` (repeatable) | At launch, for a one-shot or the REPL's first message. |
| `/image PATH…` | In the REPL, for the next message. |
| `/image /NAME…` | The same, read from the screenshots folder (below). |
| `/image clear` | Drop everything pending. |

The command answers with what it queued:

```text
minicpm-v4.6:1b › /image /tmp/red.png
attached red.png
minicpm-v4.6:1b › What color fills the attached image? Answer with one word.
red
```

## The screenshots shortcut

`/` is a virtual directory over the screenshots folder — config
`screenshots_dir`, default `~/Pictures/Screenshots`. Tab complete it like a
real path:

```text
llama3.2:3b › /image /          # lists the folder, newest first
llama3.2:3b › /image /Screenshot from 2026-10-27 10-31-51.png
attached Screenshot from 2026-10-27 10-31-51.png
```

- `/image /` lists the folder (subdirectories end in `/`); typing narrows
  the menu, and the completed name is inserted whole, spaces included.
- `/image /saved/x.png` resolves to `<screenshots_dir>/saved/x.png`.
- A prefix with no entry in that folder — `/tmp/x`, `~`, `./x` — is an
  ordinary path: absolute, home-relative or cwd-relative, completed as usual.
- The whole argument is tried as one virtual path first, so a screenshot
  name with spaces attaches unquoted. Several images at once still use
  quotes. `--image` takes real paths only.

The model must advertise **vision**. Asking a text-only model makes the
backend refuse the request, and the harness surfaces that as an ordinary
error:

```console
$ ollamarun --no-tools --no-save --image /tmp/red.png llama3.2:3b "What color is this image?"
llama3.2:3b ›
HTTP 400 on /api/chat: Multimodal data provided, but model does not support
multimodal requests.
```

A missing file exits 2 with a message naming the problem and the failing
path (an empty or unreadable file reports the path and the reason too):

```console
$ ollamarun --no-save --image /tmp/does-not-exist.png llama3.2:3b "hi"
image not found: /tmp/does-not-exist.png
$ echo $?
2
```

## How they are sent

Images are base64-embedded into the chat request (`images` on the Ollama
message), so nothing about the target machine matters: with
[`--remote`](remote.md) the file stays local and travels with the request.

- Config `image_storage` (default `"embed"`) keeps the bytes in the pending
  record. With any other value, files **over 8 MiB** are kept as a path and
  re-read when the request is built instead of being embedded twice.
- Everything is read from disk at attach time, so editing the file
  afterwards does not change what was sent (in `embed` mode).

## The API side

`ollamarun serve` takes images as OpenAI `image_url` parts (data URIs or
http(s) URLs) or as uploaded `file` parts, with its own capability check and
limits — see [Vision](../api/openai/vision.md) and
[Artifacts](../api/agent/artifacts.md).
