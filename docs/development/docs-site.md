# Docs site

The site is MkDocs + Material: `mkdocs.yml` is the whole configuration and
`docs/` holds one page per feature, grouped by surface — `cli/`, `api/`,
`tui/`, `web-ui/` — with global material in `getting-started/`, `config/`,
`project/`, `development/` and `design/`. Collections that are one feature
(the slash commands, the tools) get a single reference page with a section per
item.

Two kinds of pages:

- Feature pages mirror the README's material page by page — the README is the
  GitHub front page, the site is the browsable version — so a behavior change
  updates both.
- A few pages are snippet includes of the repository's own files, so they
  cannot drift from their sources:

| Page | Source |
| --- | --- |
| `development/architecture.md` | the marked `architecture` section of `AGENTS.md` |
| `development/contributing.md` | the marked `where-to-change` and `testing` sections of `AGENTS.md` |
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

To add a page, put it under its surface (or `getting-started/`/`config/` for
global material), register it in `mkdocs.yml`'s `nav`, and prefer an include
over copying text. Feature pages created as stubs carry a single
`*This page is not written yet.*` line; replace it when the page is written.
