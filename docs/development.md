# Development

## Where to change what

--8<-- "AGENTS.md:where-to-change"

## Testing notes

--8<-- "AGENTS.md:testing"

## Building these docs

These pages are assembled from the repository's own files through
`pymdownx.snippets` includes, so they cannot drift from their sources:

| Page | Source |
| --- | --- |
| `index.md` | `README.md` |
| `architecture.md` | the marked `architecture` section of `AGENTS.md` |
| `development.md` | the marked `where-to-change` and `testing` sections of `AGENTS.md` |
| `design/api-layer.md` | the marked `api-layer` section of `ollamarunAPILayerPlan.md` |

Install the tooling (`mkdocs-material` pulls in `mkdocs` and
`pymdown-extensions`; on a PEP 668 box add `--break-system-packages`, which
still only touches the user site) and build:

```console
$ pip install --user mkdocs-material
$ mkdocs serve            # live preview on http://127.0.0.1:8000
$ mkdocs build --strict   # warnings fail the build; run before deploying
```

Deploying is `mkdocs gh-deploy`: it builds the site and pushes the `gh-pages`
branch, which GitHub Pages serves at <https://vinibretas.github.io/ollamarun/>.

To add a page, write a new `docs/NAME.md` and register it in `mkdocs.yml`'s
`nav` — prefer an include over copying text.
