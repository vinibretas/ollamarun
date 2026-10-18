# Picker and sorting

The model picker (a model-less `ollamarun` run, `/model`) sorts by `modified`
(newest first), `name`, `size`, `parameters` or `context`; `--sort KEY` seeds
it and the picker prompt takes `--sort KEY` in place. `context` costs one lazy
`/api/show` per model, cached per host.

In the REPL, `/model`/`/models [NAME|--list|--sort KEY]` switches model or
lists and sorts the installed models, and `/ps` shows the resident ones.
