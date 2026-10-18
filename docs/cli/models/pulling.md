# Pulling models

A model missing from the target machine (local or `--remote`) offers to pull
it after printing that machine's free disk space (local `disk_usage`, remote
via a BatchMode ssh `df` probe; degrades to unknown), with live per-layer
progress. `--pull` pulls without asking, `/pull [MODEL]` does it from the
REPL, and declining the prompt opens the model picker.
