# Slash commands

Dispatched in the REPL and the TUI; `/help` also lists the shortcuts (`/?`).

| Command | What it does |
| --- | --- |
| `/help, /? [topic]` | Show commands (topic: shortcuts) |
| `/set parameter NAME VALUE \| system TEXT` | Session parameters / system prompt |
| `/show [info\|license\|modelfile\|parameters\|system\|template]` | Model details |
| `/load [MODEL\|SESSION]` | Load a saved session or switch model |
| `/save NAME` | Create an Ollama model from this session |
| `/pull [MODEL]` | Pull a model onto this host (updates if present) |
| `/clear` | Forget the context, keep settings, start a new session |
| `/new` | Start a brand new session |
| `/sessions [filter]` | Browse and switch saved sessions |
| `/title [TEXT\|--auto]` | Show or set the session title |
| `/tag [list\|add TAG…\|rm TAG]` | Session tags |
| `/model, /models [NAME\|--list\|--sort KEY]` | Switch model, list or sort the installed models |
| `/ps` | Show resident models (ollama ps) |
| `/num-ctx [N \| --max-gpu [--probe] \| --max-cpu N [--probe]]` | Context window: N, the largest that keeps the GPU, or N% of DRAM |
| `/gpu-layers [N\|N%\|+N\|-N\|all\|cpu\|auto]` | GPU layers for this model |
| `/params` | Show session parameters |
| `/tools [on\|off\|policy MODE\|enable\|disable NAME…\|NAME on\|off\|NAME help\|NAME call ARGS\|allow\|rules\|help]` | Tool harness, per-tool control (aliases resolve), generated help pages and direct calls |
| `/tts [on\|off\|voice NAME\|lang CODE\|tools on\|off\|voices]` | Auto-speak via tts-remote |
| `/think [on\|off\|low\|medium\|high]` | Thinking mode |
| `/thinking [show\|collapse\|hide\|toggle\|full]` | Thinking display (Ctrl+T while streaming) |
| `/system [TEXT\|clear\|off]` | Show or set the system prompt (built-in default when unset) |
| `/undo` | Drop the last exchange |
| `/retry` | Regenerate the last response |
| `/edit` | Edit the last message in `$EDITOR` and resend |
| `/branch [TITLE]` | Fork this conversation into a new session |
| `/context` | Context-window usage |
| `/summarize` | Compact older messages into a summary |
| `/image PATH… \| clear` | Attach images to the next message |
| `/render [markdown\|plain\|raw]` | Response rendering mode |
| `/math [on\|off]` | LaTeX→Unicode math conversion |
| `/pager [on\|off]` | Page long transcripts |
| `/tps [on\|off]` | Tokens/s footer while streaming |
| `/copy [N]` | Copy the last response (or its Nth code block) |
| `/export [--format md\|json\|html] [FILE]` | Export this session |
| `/stats` | Session and last-generation stats |
| `/queue [list\|clear\|rm N]` | Messages and commands typed while busy |
| `/bye, /exit` | Leave |

Auto-speak: `--speak` / config `tts_auto` / `/tts on` read every final answer
aloud through `tts-remote say --markdown` (fire-and-forget; the `speaking`
line ends with the same `X → Y` as the tool's result box).
