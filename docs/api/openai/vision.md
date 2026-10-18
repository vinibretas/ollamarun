# Vision

`image_url` content parts accept `data:` URIs or http(s) URLs (fetched with
explicit timeouts, at most 3 redirects and a 10 MiB cap — the SSRF caveat is
the locked trusted-network boundary), are validated against the model's
`capabilities` and refused with `model_does_not_support_images` when vision is
absent.

A `{"type": "file", "file_id": "art_…"}` part resolves an uploaded artifact:
text-ish files are inlined as a `<file name="…">` block, images go through the
same vision gate, and other binaries are described but never pretended to be
processed — see [Artifacts](../agent/artifacts.md).

The CLI takes images too — `--image PATH` / `/image PATH… | clear`.
