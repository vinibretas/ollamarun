"""REPL: `Repl` drives the interactive session; each slash command
(`/set`, `/show`, `/load`, `/save`, `/undo`, `/retry`, `/edit`, `/branch`,
`/context`, `/summarize`, `/image`, `/render`, `/math`, `/pager`, `/ps`,
`/num-ctx`, ...) is a standalone `cmd_NAME(repl, argument)` function
dispatched by name through `COMMAND_HANDLERS`.

The pieces: `core.py` (`Repl`), `completion.py` (Tab completion),
`queue.py` (what was typed while busy), `commands/` (the commands, one module
per topic, plus the name/help tables and the dispatch registry), `view.py`
(shared read-only table/format helpers) and `session.py` (switching the active
session). This module is the package's public surface; add a new slash
command by adding one function to a `commands/` module and wiring it into
`commands/registry.py`'s `COMMAND_HANDLERS`.
"""
from __future__ import annotations

import types

from .core import Repl
from .completion import (SlashCompleter, completion_candidates,
                         completion_fragment, tool_name_candidates)
from .queue import MessageQueue, QueuedItem
from .session import _switch_session
from .view import ps_table
from .commands.config import (cmd_math, cmd_pager, cmd_params, cmd_render,
                              cmd_set, cmd_show, cmd_system, cmd_think,
                              cmd_thinking, cmd_tps, cmd_tts, system_label)
from .commands.context import cmd_context, cmd_summarize
from .commands.help import cmd_help
from .commands.media import cmd_copy, cmd_export, cmd_image
from .commands.model import (cmd_gpu_layers, cmd_model, cmd_num_ctx, cmd_ps,
                             cmd_pull, resolve_num_ctx)
from .commands.registry import COMMAND_HANDLERS
from .commands.sessions import (cmd_branch, cmd_clear, cmd_edit, cmd_load,
                                cmd_new, cmd_retry, cmd_save, cmd_sessions,
                                cmd_tag, cmd_title, cmd_undo)
from .commands.stats import cmd_bye, cmd_queue, cmd_stats
from .commands.tables import (COMMAND_INFO, TOOLS_HELP_ROWS, TOOLS_POLICIES,
                              TOOLS_SUBCOMMANDS, TOOLS_TOOL_ACTIONS)
from .commands.tools import cmd_tools
from ..ui import _prompt_approval  # compatibility re-export


#: Public surface re-exported by `ollamarun_lib/__init__.py`, minus the
#: submodules themselves.
__all__ = [name for name, value in globals().items()
           if not name.startswith("_") and not isinstance(value, types.ModuleType)]
