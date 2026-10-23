"""Model commands: `/model(s)`, `/show`, `/ps`, `/pull`, `/num-ctx` and `/gpu-layers`."""
from __future__ import annotations

from typing import TYPE_CHECKING
import shlex
from typing import Optional
from rich.markup import escape
from ...client import OllamaClient, OllamaError
from ...constants import console
from ...helpers import is_local_host, remote_node
from ...ctxfit import (
    FitRequest,
    NUM_CTX_HELP,
    machine_state_for,
    max_cpu_context,
    max_gpu_context,
    parse_fill_percent,
    parse_num_ctx_arg,
)
from ...interactive import (
    GPU_LAYERS_USAGE,
    MODEL_SORT_KEYS,
    disk_free_line,
    ensure_model,
    fetch_context_lengths,
    model_context_length,
    model_table,
    parse_gpu_layers_spec,
    parse_model_sort,
    pick_model,
    pull_with_progress,
    resolve_gpu_layers,
    sort_models,
)
from ..view import ps_table

if TYPE_CHECKING:
    from ..core import Repl


def cmd_ps(repl: Repl, argument: str) -> None:
    try:
        entries = repl.client.ps()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if not entries:
        console.print("[dim]no models currently loaded[/dim]")
        return
    console.print(ps_table(entries))


def resolve_num_ctx(client: OllamaClient, model: str, requested: int) -> tuple[int, Optional[int]]:
    """Caps `requested` to the model's max context length (from /api/show).

    Returns (value_to_use, max_ctx_or_None). Never raises: an unreachable
    server just means the max is unknown and the request passes through.
    """
    try:
        show_data = client.show(model)
    except OllamaError:
        return requested, None
    max_ctx = model_context_length(show_data, {})
    if max_ctx and requested > max_ctx:
        return max_ctx, max_ctx
    return requested, max_ctx


def cmd_num_ctx(repl: Repl, argument: str) -> None:
    usage = "usage: /num-ctx N | --max-gpu [--probe] | --max-cpu N [--probe]"
    argument = argument.strip()
    if not argument:
        current = repl.session.parameters.get("num_ctx")
        try:
            max_ctx = model_context_length(repl.client.show(repl.session.model), {})
        except OllamaError:
            max_ctx = None
        bits = [f"num_ctx = {current if current is not None else '(model default)'}"]
        if max_ctx:
            bits.append(f"model max {max_ctx}")
        console.print("[key]context size[/key] " + " · ".join(bits))
        console.print("[dim]--max-gpu picks the largest context that still fits 100% "
                      "on the GPU; --max-cpu N fills N% of the DRAM available to the "
                      "model (--probe verifies by loading)[/dim]")
        return
    tokens = argument.split()
    if tokens[0] in {"--help", "-h", "help"}:
        console.print(NUM_CTX_HELP)
        return

    def fit_request() -> FitRequest:
        ssh_target = remote_node(repl.session.remote, repl.config) \
            if repl.session.remote else None
        ssh_user = repl.config.get("remote_ssh_user")
        base = getattr(repl.client, "base", "") or ""
        host_label = repl.session.remote or (None if is_local_host(base) else base)
        return FitRequest(
            client=repl.client, model=repl.session.model, config=repl.config,
            host_label=host_label, ssh_target=ssh_target, ssh_user=ssh_user,
            pinned_spec=repl.session.gpu_layers,
            num_gpu=repl.session.parameters.get("num_gpu"),
            base_options={key: val for key, val in (repl.session.parameters or {}).items()
                          if key != "num_ctx"},
            keepalive=repl.session.keepalive or getattr(repl.args, "keepalive", None),
            refresh_gpu=lambda: machine_state_for(base, ssh_target=ssh_target,
                                                  ssh_user=ssh_user),
        )

    def apply(ctx: Optional[int], *, probe: bool) -> None:
        if ctx is None:
            return
        repl.session.parameters["num_ctx"] = ctx
        repl.session.touch()
        note = " [dim]· model is loaded at that context[/dim]" if probe else ""
        console.print(f"[ok]num_ctx set to {ctx}[/ok]{note}")

    head = tokens[0].lower()
    rest = tokens[1:]
    max_cpu: Optional[str] = None
    if head in {"--max-cpu", "max-cpu"}:
        if not rest:
            console.print(f"[err]{usage} — --max-cpu needs a percentage[/err]")
            return
        max_cpu, rest = rest[0], rest[1:]
    elif head.startswith("--max-cpu="):
        max_cpu = tokens[0].split("=", 1)[1]
    if max_cpu is not None:
        percent = parse_fill_percent(max_cpu)
        if percent is None:
            console.print("[err]--max-cpu takes a percentage between 1 and 100[/err]")
            return
        probe = False
        for token in rest:
            flag = token.lower()
            if flag in {"--probe", "probe", "-p"}:
                probe = True
            elif flag in {"--help", "-h", "help"}:
                console.print(NUM_CTX_HELP)
                return
            else:
                console.print(f"[err]{usage}[/err]")
                return
        apply(max_cpu_context(fit_request(), percent=percent, probe=probe), probe=probe)
        return

    kind, value = parse_num_ctx_arg(tokens[0])
    if kind == "max-gpu":
        probe = False
        for token in rest:
            flag = token.lower()
            if flag in {"--probe", "probe", "-p"}:
                probe = True
            elif flag in {"--help", "-h", "help"}:
                console.print(NUM_CTX_HELP)
                return
            else:
                console.print(f"[err]{usage}[/err]")
                return
        apply(max_gpu_context(fit_request(), probe=probe), probe=probe)
        return
    if kind == "error":
        console.print(f"[err]{usage}[/err]")
        return
    requested = int(value or 0)
    if requested <= 0:
        console.print("[err]--num-ctx must be a positive integer[/err]")
        return
    applied, max_ctx = resolve_num_ctx(repl.client, repl.session.model, requested)
    if max_ctx and requested > max_ctx:
        console.print(f"[warn]{requested} exceeds {escape(repl.session.model)}'s max context "
                      f"({max_ctx}) — capped to {max_ctx}[/warn]")
    repl.session.parameters["num_ctx"] = applied
    repl.session.touch()
    console.print(f"[ok]num_ctx set to {applied}[/ok]")


def cmd_gpu_layers(repl: Repl, argument: str) -> None:
    argument = argument.strip()
    if not argument:
        console.print(f"[key]gpu layers[/key] {escape(repl.gpu_layers_summary())}")
        console.print(f"[dim]{GPU_LAYERS_USAGE}[/dim]")
        return
    try:
        kind, number = parse_gpu_layers_spec(argument)
    except ValueError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    if kind == "auto":
        new_spec: Optional[str] = None
    elif kind == "cpu":
        new_spec = "cpu"
    elif kind == "all":
        new_spec = "all"
    else:
        current = repl.gpu_layers_value()
        try:
            resolved = resolve_gpu_layers(argument, repl._model_block_count(), current)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[err]{escape(str(exc))}[/err]")
            return
        # Deltas are normalized to an absolute value; percentages stay relative
        # so switching models re-resolves them against the new layer count.
        new_spec = f"{number}%" if kind == "pct" else str(resolved)
    if new_spec == repl.session.gpu_layers:
        console.print(f"[dim]already set[/dim] — {escape(repl.gpu_layers_summary())}")
        return
    repl.session.gpu_layers = new_spec
    repl.session.touch()
    console.print(f"[ok]gpu layers[/ok] {escape(repl.gpu_layers_summary())}")
    if repl.unload_model():
        console.print("[dim]the model will reload with the new setting[/dim]")



def _pick_model(repl: Repl, current: Optional[str] = None) -> Optional[str]:
    """The model picker ordered by the session's sort; a sort chosen inside
    the picker sticks for the rest of the session."""
    return pick_model(repl.client, current, ui=repl.ui, sort=repl.model_sort,
                      on_sort=lambda key: setattr(repl, "model_sort", key))


def _print_model_list(repl: Repl, sort: str) -> None:
    try:
        models = repl.client.tags()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    contexts = fetch_context_lengths(repl.client, models) if sort == "context" else {}
    console.print(model_table(sort_models(models, sort, contexts=contexts),
                              sort=sort, contexts=contexts))


def cmd_pull(repl: Repl, argument: str) -> None:
    name = argument.strip() or _pick_model(repl)
    if not name:
        return
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    summary = disk_free_line(repl.client, repl.config, ssh_target=ssh_target,
                             host_label=repl.session.remote)
    console.print(f"[dim]{escape(summary)}[/dim]")
    if repl.client.model_exists(name):
        console.print(f"[dim]{escape(name)} is installed — checking for updates[/dim]")
    try:
        pull_with_progress(repl.client, name, quiet=not console.is_terminal)
    except OllamaError as exc:
        console.print(f"[err]pull failed: {escape(str(exc))}[/err]")
        return
    repl.refresh_models()
    console.print(f"[ok]pulled[/ok] {escape(name)}")



def cmd_model(repl: Repl, argument: str) -> None:
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]unparsable arguments: {escape(str(exc))}[/err]")
        return
    wants_list = False
    sort: Optional[str] = None
    name_parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--sort" or token.startswith("--sort="):
            inline = "=" in token
            value = token.split("=", 1)[1] if inline else (
                tokens[index + 1] if index + 1 < len(tokens) else "")
            consumed = 1 if inline else 2
            if value.strip().lower() == "fuzzy":
                query_at = index + consumed
                if query_at >= len(tokens) or tokens[query_at].startswith("-"):
                    console.print("[err]fuzzy sort needs a search string — use "
                                  "/model --sort fuzzy TEXT[/err]")
                    return
                value = "fuzzy " + tokens[query_at]
                consumed += 1
            if not value.strip():
                console.print("[err]usage: /model --sort "
                              + "|".join(MODEL_SORT_KEYS) + "|fuzzy TEXT[/err]")
                return
            try:
                sort = parse_model_sort(value)
            except ValueError as exc:
                console.print(f"[err]{escape(str(exc))}[/err]")
                return
            index += consumed
            continue
        if token == "--list":
            wants_list = True
        else:
            name_parts.append(token)
        index += 1
    if sort is not None:
        repl.model_sort = sort
    if wants_list:
        _print_model_list(repl, repl.model_sort)
        return
    name = " ".join(name_parts).strip() or _pick_model(repl, repl.session.model)
    if not name:
        return
    ssh_target = remote_node(repl.session.remote, repl.config) if repl.session.remote else None
    chosen = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                          host_label=repl.session.remote, ui=repl.ui)
    if not chosen:
        return
    name = chosen
    if name == repl.session.model:
        console.print("[dim]already using that model[/dim]")
        return
    if repl.session.messages and not repl.ui.confirm(
            f"switch to [key]{escape(name)}[/key] and keep context?", default=False):
        repl.session.messages = []
    repl.unload_model()
    repl.session.model = name
    repl.session.touch()
    repl.refresh_models()
    console.print(f"[ok]model[/ok] {escape(name)}")


