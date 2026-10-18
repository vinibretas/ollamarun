# Compaction

`/summarize` compacts older messages into a summary, dropping the originals
from the context while keeping the session's recent turns. Config
`auto_compact` does the same automatically when the context fills up.

`/context` shows window usage, and the `context_status` tool reports the
budget, free space and a wrap-up note past 80%.
