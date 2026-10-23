"""Tool registry, safety policy and executor for the agent loop.

`ToolSpec` describes one tool (name, description, JSON-schema parameters and
a handler); `default_registry()` builds the built-in set from the definition
files in `defs/tools/`, one JSON per tool, with
`~/etc/config_files/ollamarun-defs/tools/` overriding the shipped files by
name. `approval_decision()` classifies a call as auto-approved or needing
user confirmation (read-only tools and provably-harmless shell/python calls
are safe; anything else asks), and `execute_tool()` runs the handler with a
timeout and caps the text that goes back to the model.

The confirmation prompt can store generalized, session-scoped `ToolRule`s
-- `shell: mkdir -p :*`, `path: ~/proj/**`, `python: imports os,sys` --
suggested by `suggest_rules()` from the call being asked about and tested
by `rule_matches()`. `dangerous_command()` is the floor no rule ever
covers: substitutions, redirections, privilege/wrapper/interpreter
commands, disk-level tools, catastrophic `rm` and pipes into an
interpreter always prompt again.

Every responsibility has its own module: `spec` (types and output
plumbing), `policy`/`rules` (the approval gate), one module per tool
(`run_shell`, `read_file`, `fetch_url`, ...), `registry` (definition loading
and the default set) and `executor` (the guarded call path). This module is
the package's public surface: everything the rest of the codebase imports
from `ollamarun_lib.tools` is re-exported here.
"""
from __future__ import annotations

import types

from .spec import (MAX_PDF_VISION_PAGES, TRUNCATE_TAIL, Decision, ToolCancelled,
                   ToolContext, ToolError, ToolHandlerResult, ToolOutcome,
                   ToolSpec, truncate_output)
from .policy import (DEFAULT_SHELL_ALLOW, GIT_CONFIG_SAFE, GIT_READONLY,
                     PYTHON_DANGEROUS_ATTRS, PYTHON_DANGEROUS_CALLS,
                     PYTHON_DANGEROUS_NAMES, SAFE_PYTHON_MODULES,
                     approval_decision, python_import_roots, python_safety,
                     shell_safety)
from .rules import (PATH_RULE_TOOLS, RULE_FLOOR_COMMANDS, RULE_KINDS,
                    SHELL_INTERPRETERS, PolicyCheck, ToolRule,
                    dangerous_command, matching_rule, policy_check, rule_label,
                    rule_matches, suggest_rules)
from .args import call_detail, call_fingerprint, normalize_arguments
from .html import MAX_PAGE_LINKS, extract_html_text
from .pdf import (MAX_PDF_BYTES, MAX_PDF_CHARS, MAX_PDF_PAGES, PDF_VISION_DPI,
                  PDF_VISION_PREFERENCE, PDF_VISION_PROMPT,
                  PDF_VISION_TIMEOUT, extract_pdf_text, parse_page_spec,
                  pdf_page_count, pick_vision_model, render_pdf_page,
                  select_pdf_pages, vision_models_installed, vision_transcribe)
from .net import (BOT_WALL_MARKERS, BROWSER_HEADERS, DEFAULT_SEARCH_RESULTS,
                  FETCH_TIMEOUT, JS_SHELL_MARKERS, MAX_ERROR_DETAIL,
                  MAX_SEARCH_RESULTS, SEARCH_ENGINE_HOSTS, SEARCH_TIMEOUT,
                  SOFT_BLOCK_MARKERS, format_fetch_response,
                  search_engine_query)
from .web_search import (DDGS, format_search_results, run_web_search,
                         search_available)
from .read_file import MAX_READ_CHARS
from .list_dir import MAX_LIST_ENTRIES
from .chem_cli import (chem_inventory_available, find_chem_inventory,
                       run_chem_inventory)
from .tts import find_tts_remote, run_tts_say, speech_hosts
from .registry import (HANDLERS, REQUIREMENT_HINTS, REQUIREMENTS,
                       ToolDefinition, default_registry, load_tool_definitions,
                       resolve_tool_alias, tool_aliases,
                       tool_definition_dirs, tool_definitions, tool_schemas)
from .executor import ABANDON_ON_CANCEL, ToolRunner, execute_tool


#: Public surface re-exported by `ollamarun_lib/__init__.py`, minus the
#: submodules themselves.
__all__ = [name for name, value in globals().items()
           if not name.startswith("_") and not isinstance(value, types.ModuleType)]
