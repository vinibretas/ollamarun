"""The slash-command tables: names, usage and the `/tools` help rows.

A leaf module: completion, help and the dispatch registry all read it, so
it must not import a command module (that would close an import cycle).
"""
from __future__ import annotations


COMMAND_INFO: dict[str, tuple[str, str]] = {
    "/help, /?": ("[topic]", "show commands (topic: shortcuts)"),
    "/set": ("parameter NAME VALUE | system TEXT", "session parameters / system prompt"),
    "/show": ("[info|license|modelfile|parameters|system|template]", "model details"),
    "/load": ("[MODEL|SESSION]", "load a saved session or switch model"),
    "/save": ("NAME", "create an Ollama model from this session"),
    "/pull": ("[MODEL]", "pull a model onto this host (updates if present)"),
    "/clear": ("", "forget the context, keep settings, start a new session"),
    "/new": ("", "start a brand new session"),
    "/sessions": ("[filter]", "browse and switch saved sessions"),
    "/title": ("[TEXT|--auto]", "show or set the session title"),
    "/tag": ("[list|add TAG…|rm TAG]", "session tags"),
    "/model, /models": ("[NAME|--list|--sort KEY|fuzzy TEXT]",
                        "switch model, list or sort the installed models"),
    "/ps": ("", "show resident models (ollama ps)"),
    "/num-ctx": ("[N | --max-gpu [--probe] | --max-cpu N [--probe]]",
                 "context window: N, the largest that keeps the GPU, or N% of DRAM"),
    "/gpu-layers": ("[N|N%|+N|-N|all|cpu|auto]", "GPU layers for this model"),
    "/params": ("", "show session parameters"),
    "/tools": ("[on|off|policy MODE|enable|disable NAME…|NAME on|off|NAME help|NAME call ARGS|allow|rules|help]",
               "tool harness, per-tool control and help"),
    "/tts": ("[on|off|voice NAME|lang CODE|tools on|off|voices]", "auto-speak via tts-remote"),
    "/think": ("[on|off|low|medium|high]", "thinking mode"),
    "/thinking": ("[show|collapse|hide|toggle|full]", "thinking display (Ctrl+T while streaming)"),
    "/system": ("[TEXT|clear|off]", "show or set the system prompt (built-in default when unset)"),
    "/undo": ("", "drop the last exchange"),
    "/retry": ("", "regenerate the last response"),
    "/edit": ("", "edit the last message in $EDITOR and resend"),
    "/branch": ("[TITLE]", "fork this conversation into a new session"),
    "/context": ("", "context-window usage"),
    "/summarize": ("", "compact older messages into a summary"),
    "/image": ("PATH… | clear", "attach images to the next message"),
    "/render": ("[markdown|plain|raw]", "response rendering mode"),
    "/math": ("[on|off]", "LaTeX→Unicode math conversion"),
    "/pager": ("[on|off]", "page long transcripts"),
    "/tps": ("[on|off]", "tokens/s footer while streaming"),
    "/copy": ("[N]", "copy the last response (or its Nth code block)"),
    "/export": ("[--format md|json|html] [FILE]", "export this session"),
    "/stats": ("", "session and last-generation stats"),
    "/queue": ("[list|clear|rm N]", "messages and commands typed while busy"),
    "/bye, /exit": ("", "leave"),
}


#: `/tools` subcommands (the first token), in help/table order.
TOOLS_SUBCOMMANDS: tuple[str, ...] = (
    "on", "off", "policy", "enable", "disable", "allow", "rules", "help",
    "list", "status")

#: actions after a tool name: `/tools NAME <action>`.
TOOLS_TOOL_ACTIONS: tuple[str, ...] = ("on", "off", "help", "call")

#: `/tools policy` modes (same values `approval_decision` knows).
TOOLS_POLICIES: tuple[str, ...] = ("auto", "allowlist", "ask")

#: `/tools help` reference rows: syntax -> what it does.
TOOLS_HELP_ROWS: tuple[tuple[str, str], ...] = (
    ("(none) · list · status", "table of tools with state and approval mode"),
    ("on | off", "enable or disable the whole tool harness"),
    ("policy auto|allowlist|ask", "global approval policy"),
    ("enable NAME… | disable NAME…", "toggle several tools at once"),
    ("NAME on | off", "toggle one tool, e.g. `/tools web_search off`"),
    ("NAME help", "description, parameters (required/optional), aliases"),
    ("NAME call ARGS", 'run it yourself: call {"query": "x"} · call query=x'),
    ("help [NAME]", "this reference, or one tool's page"),
    ("allow", "safe shell command allowlist"),
    ("rules [add|rm]", "stored approval rules"),
)
