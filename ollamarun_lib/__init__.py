"""ollamarun_lib - the implementation behind the `ollamarun` CLI, split into
one module per pipeline stage (see each module's docstring, and the
Architecture section of CLAUDE.md, for what lives where).

This package re-exports every module's public names at the top level so
`import ollamarun_lib as orun` behaves like the old single-file script did
-- existing callers and `test_ollamarun.py` don't need to know which
submodule a given name actually lives in.
"""
from __future__ import annotations

from .constants import *  # noqa: F401,F403
from .helpers import *  # noqa: F401,F403
from .mathrender import *  # noqa: F401,F403
from .files import *  # noqa: F401,F403
from .images import *  # noqa: F401,F403
from .renderer import *  # noqa: F401,F403
from .client import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .session import *  # noqa: F401,F403
from .emit import *  # noqa: F401,F403
from .ui import *  # noqa: F401,F403
from .interactive import *  # noqa: F401,F403
from .ctxfit import *  # noqa: F401,F403
from .repl import *  # noqa: F401,F403
from .tui import *  # noqa: F401,F403
from .run import *  # noqa: F401,F403
from .cli import *  # noqa: F401,F403

from .repl import _switch_session  # noqa: F401
from .interactive import _CODE_BLOCK_RE  # noqa: F401
