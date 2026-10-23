"""The slash-command tables: names, usage and the `/tools` help rows.

The handler functions live in the command modules; this file is the one place
the REPL's completion, help and dispatch read the command set from.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .config import (cmd_math, cmd_pager, cmd_params, cmd_render, cmd_set,
                     cmd_show, cmd_system, cmd_think, cmd_thinking, cmd_tps,
                     cmd_tts)
from .context import cmd_context, cmd_summarize
from .help import cmd_help
from .media import cmd_copy, cmd_export, cmd_image
from .model import (cmd_gpu_layers, cmd_model, cmd_num_ctx, cmd_ps,
                    cmd_pull)
from .sessions import (cmd_branch, cmd_clear, cmd_edit, cmd_load, cmd_new,
                       cmd_retry, cmd_save, cmd_sessions, cmd_tag,
                       cmd_title, cmd_undo)
from .stats import cmd_bye, cmd_queue, cmd_stats
from .tools import cmd_tools

if TYPE_CHECKING:
    from ..core import Repl


COMMAND_HANDLERS: dict[str, Callable[[Repl, str], Any]] = {
    "/help": cmd_help, "/?": cmd_help,
    "/set": cmd_set,
    "/show": cmd_show,
    "/load": cmd_load,
    "/save": cmd_save,
    "/pull": cmd_pull,
    "/clear": cmd_clear,
    "/new": cmd_new,
    "/sessions": cmd_sessions,
    "/title": cmd_title,
    "/tag": cmd_tag,
    "/model": cmd_model, "/models": cmd_model,
    "/ps": cmd_ps,
    "/num-ctx": cmd_num_ctx,
    "/gpu-layers": cmd_gpu_layers,
    "/params": cmd_params,
    "/tools": cmd_tools,
    "/tts": cmd_tts,
    "/think": cmd_think,
    "/thinking": cmd_thinking,
    "/system": cmd_system,
    "/undo": cmd_undo,
    "/retry": cmd_retry,
    "/edit": cmd_edit,
    "/branch": cmd_branch,
    "/context": cmd_context,
    "/summarize": cmd_summarize,
    "/image": cmd_image,
    "/render": cmd_render,
    "/math": cmd_math,
    "/pager": cmd_pager,
    "/tps": cmd_tps,
    "/copy": cmd_copy,
    "/export": cmd_export,
    "/stats": cmd_stats,
    "/queue": cmd_queue,
    "/bye": cmd_bye, "/exit": cmd_bye,
}
