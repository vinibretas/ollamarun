# Docs site

The site is MkDocs + Material: `mkdocs.yml` is the whole configuration and
`docs/` holds one page per feature, grouped by surface — `cli/`, `api/`,
`tui/`, `web-ui/` — with global material in `getting-started/`, `config/`,
`project/`, `development/` and `design/`. Collections that are one feature
(the slash commands, the tools) get a single reference page with a section
per item.

## How the pages relate to the repository

- **Feature pages are the browsable reference.** `README.md` stays the
  GitHub front page — short, scannable, linking here for depth. When a
  behavior changes, update both: the README paragraph and the site page.
- **A few pages are snippet includes** of the repository's own files, so
  they cannot drift:

  | Page | Source |
  | --- | --- |
  | `development/architecture.md` | the marked `architecture` section of `AGENTS.md` |
  | `development/contributing.md` | the marked `where-to-change` and `testing` sections of `AGENTS.md` |
  | `design/api-layer.md` | the marked `api-layer` section of `ollamarunAPILayerPlan.md` |

  Edit the source files, never a generated page, and keep the snippet
  markers (`<!-- --8<-- [start:name] -->` … `[end:name]`) intact.
- **Command and tool text lives in the repository**, not duplicated: the
  slash-command reference mirrors `COMMAND_INFO`, the tool reference mirrors
  `defs/tools/*.json`, and the config reference mirrors `DEFAULT_CONFIG`
  (`ollamarun_lib/constants.py`). Change those files and the pages must
  follow.

## Examples are live-verified

Every example block in this site was executed against a running harness
before it landed — commands, flags, API calls and their outputs. When you
change behavior, re-run the examples on the affected pages; a stale example
is a bug in the docs, same as a broken link.

Small conventions that keep the examples honest:

- `console` blocks show real commands; output is trimmed with `…` where a
  full capture would be noise, never invented.
- Outputs that depend on machine state (model names, timings, ids) use
  representative values and say so where it matters.
- Long animated terminal moments (the live view) are shown as the final
  frame, not the redraw storm.

## Building and deploying

Install the tooling (`mkdocs-material` pulls in `mkdocs` and
`pymdown-extensions`; on a PEP 668 box add `--break-system-packages`, which
still only touches the user site):

```console
$ pip install --user mkdocs-material
$ mkdocs serve            # live preview on http://127.0.0.1:8000
$ mkdocs build --strict   # warnings fail the build; run before deploying
```

Deploying is `mkdocs gh-deploy`: it builds the site and pushes the
`gh-pages` branch, which GitHub Pages serves at
<https://vinibretas.github.io/ollamarun/>.

## Adding a page

1. Put the file under its surface (`docs/cli/`, `docs/api/`, …) or the
   global directories.
2. Register it in `mkdocs.yml`'s `nav`.
3. Prefer a snippet include over copied text when the repository already has
   the source; otherwise write the page in the feature style above.
4. `mkdocs build --strict` — the link/anchor validation fails on broken
   references, which is the point.
