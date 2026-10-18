# The tool loop

Calls and results are kept in the session (so `/show`/`--show` replay them),
a finished loop is separated from the committed answer by a centered
`response` divider, and every call/result panel is colour-coded — one fixed
colour per tool, a random palette colour (remembered for the run) for anything
unmapped, with config `tool_colors` overrides. The result box wears the call's
colour, prints ok/failed green/red in its title, and the `speak` tool's box
names where the voice is generated and where it will be heard
(`gpu-box → desk`, `(cached)` for a phrase served from disk).

Invented tool names (`search`, `curl`, `mkdir`, ...) and call-id-prefixed ones
(`call_fetch_url`) resolve onto the real tools through per-tool alias lists.

The loop runs in rounds, bounded by config `tools_max_rounds` (default 8);
approval rules decide what runs unattended — see
[Approvals and policy](approvals.md).
