# Acknowledgements

This project exists mostly because of other people's work:

- **[Ollama](https://ollama.com)** — the local server every request goes
  through. The whole harness is an unaffiliated wrapper around its HTTP API,
  and a very happy user of it.
- **The OpenAI API** — the `ollamarun serve` `/v1` surface reimplements
  OpenAI's HTTP schema (chat completions, embeddings, SSE streaming, tool
  calls) so unmodified OpenAI clients work against local models.
  "OpenAI-compatible" is descriptive only — no affiliation with, or
  endorsement by, OpenAI is implied.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — its KV-cache
  geometry and fit arithmetic are what the context/placement fitting is
  modelled on, and it performs the actual inference behind Ollama.
- **The Python ecosystem** — [rich](https://github.com/Textualize/rich)
  (rendering, tables, progress),
  [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)
  (the inline REPL), [Textual](https://github.com/Textualize/textual) (the
  TUI), [pylatexenc](https://github.com/phfaist/pylatexenc) (inline-math
  fallback), [ddgs](https://github.com/deedy5/ddgs) (web search),
  [pypdf](https://github.com/py-pdf/pypdf) (PDF text extraction),
  [pyperclip](https://github.com/asweigart/pyperclip) (clipboard) and
  [requests](https://github.com/psf/requests) (HTTP) — plus the official
  [openai](https://github.com/openai/openai-python) SDK, which the test suite
  round-trips as the compatibility contract.
- **The system tools it leans on** — [Tailscale](https://tailscale.com)
  (`--remote`), [trash-cli](https://github.com/andreafrancia/trash-cli)
  (`trash-put` deletes), [poppler](https://poppler.freedesktop.org)
  (`pdftotext`, and `pdftoppm` for the pages `read_pdf` sends to a vision
  model) and NVIDIA's `nvidia-smi` (GPU probes).
