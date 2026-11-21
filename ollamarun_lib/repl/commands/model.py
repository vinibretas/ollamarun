"""Model commands: `/model(s)`, `/show`, `/ps`, `/pull`, `/num-ctx` and `/gpu-layers`."""
from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Optional

from rich.markup import escape

from ...benchmarks import load_benchmarks, performance_index
from ...bestmachine import (choice_line, choose_machine, machine_table,
                            parse_metric, probe_hosts)
from ...client import OllamaClient, OllamaError
from ...constants import REMOTE_PROBE_TIMEOUT, console
from ...ctxfit import (NUM_CTX_HELP, FitRequest, machine_state_for,
                       max_cpu_context, max_gpu_context, parse_fill_percent,
                       parse_num_ctx_arg, resolve_num_ctx_delta)
from ...helpers import human_duration, is_local_host, remote_node
from ...interactive import (GPU_LAYERS_USAGE, MODEL_SORT_KEYS, disk_free_line,
                            ensure_model, fetch_context_lengths,
                            model_context_length, model_table,
                            parse_gpu_layers_spec, parse_model_sort,
                            pick_model, pull_with_progress, resolve_gpu_layers,
                            sort_models)
from ...inventory import (inventory_age, inventory_stale,
                          load_cluster_inventory)
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
    usage = ("usage: /num-ctx N | +N | -N | +P% | -P% | --max-gpu [--probe] "
             "| --max-cpu N [--probe]")
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
        console.print("[dim]+N/-N adjust by tokens, +P%/-P% by a share of the "
                      "current window[/dim]")
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

    def apply(ctx: Optional[int]) -> None:
        """Store the window; the next request reloads the runner at it.

        `num_ctx` is a load-time property, so the value rides in the session
        and the next message's payload makes Ollama reload the runner at it.
        """
        if ctx is None:
            return
        repl.session.parameters["num_ctx"] = ctx
        repl.session.touch()
        console.print(f"[ok]num_ctx set to {ctx}[/ok]")
        # The runner still answers with the old window until this goes out;
        # note the mismatch so /ps and the waiting line can say so.
        repl.mark_load_pending("ctx")

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
        apply(max_cpu_context(fit_request(), percent=percent, probe=probe))
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
        apply(max_gpu_context(fit_request(), probe=probe))
        return
    if kind == "error":
        console.print(f"[err]{usage}[/err]")
        return
    if kind in {"delta", "pct"}:
        number = int(value or 0)
        if number == 0:
            console.print("[err]a relative change needs a non-zero amount "
                          "(e.g. +1000 or -50%)[/err]")
            return
        if kind == "pct" and number < -100:
            console.print("[err]a percentage decrease cannot exceed 100%[/err]")
            return
        # The baseline is the window actually in force: a session num_ctx, else
        # the loaded runner, else the model's own default, else the ollama
        # service environment's OLLAMA_CONTEXT_LENGTH.
        baseline = repl.context_length()
        if baseline is None:
            console.print("[err]no context window is known yet — set an absolute "
                          "value first, e.g. /num-ctx 8192[/err]")
            return
        delta = number if kind == "delta" else round(baseline * number / 100)
        target = resolve_num_ctx_delta(baseline, kind, number)
        rendered = (f"{baseline} {number:+d}" if kind == "delta"
                    else f"{baseline} {number:+d}%")
        console.print(f"[dim]{rendered} → {target}[/dim]")
        if baseline + delta < 1:
            console.print("[warn]a relative decrease cannot go below 1 token[/warn]")
        applied, max_ctx = resolve_num_ctx(repl.client, repl.session.model, target)
        if max_ctx and target > max_ctx:
            console.print(f"[warn]{target} exceeds {escape(repl.session.model)}'s max "
                          f"context ({max_ctx}) — capped to {max_ctx}[/warn]")
        apply(applied)
        return
    requested = int(value or 0)
    if requested <= 0:
        console.print("[err]--num-ctx must be a positive integer[/err]")
        return
    applied, max_ctx = resolve_num_ctx(repl.client, repl.session.model, requested)
    if max_ctx and requested > max_ctx:
        console.print(f"[warn]{requested} exceeds {escape(repl.session.model)}'s max context "
                      f"({max_ctx}) — capped to {max_ctx}[/warn]")
    apply(applied)


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
    # The placement is a load-time property: the next request's num_gpu
    # makes Ollama reload the runner at the new split.
    repl.mark_load_pending("gpu")



def _performance_index(repl: Repl):
    """The fleet index only when the session's sort needs it."""
    return performance_index(repl.config) if repl.model_sort == "performance" else None


def _pick_model(repl: Repl, current: Optional[str] = None) -> Optional[str]:
    """The model picker ordered by the session's sort; a sort chosen inside
    the picker sticks for the rest of the session."""
    performance = _performance_index(repl)
    if performance is None:
        return pick_model(repl.client, current, ui=repl.ui, sort=repl.model_sort,
                          on_sort=lambda key: setattr(repl, "model_sort", key))
    return pick_model(repl.client, current, ui=repl.ui, sort=repl.model_sort,
                      on_sort=lambda key: setattr(repl, "model_sort", key),
                      performance=performance)


def _print_model_list(repl: Repl, sort: str) -> None:
    try:
        models = repl.client.tags()
    except OllamaError as exc:
        console.print(f"[err]{escape(str(exc))}[/err]")
        return
    contexts = fetch_context_lengths(repl.client, models) if sort == "context" else {}
    performance = performance_index(repl.config) if sort == "performance" else None
    if performance is not None and not performance:
        console.print("[warn]no benchmark performance data"
                      + (f": {escape(performance.error)}"
                         if performance.error else "") + "[/warn]")
    console.print(model_table(sort_models(models, sort, contexts=contexts,
                                          performance=performance),
                              sort=sort, contexts=contexts,
                              performance=performance))


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
    performance = _performance_index(repl)
    ensure_kwargs = {"performance": performance} if performance is not None else {}
    chosen = ensure_model(repl.client, name, repl.config, ssh_target=ssh_target,
                          host_label=repl.session.remote, ui=repl.ui,
                          **ensure_kwargs)
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




BEST_MACHINE_USAGE = ("usage: /best-machine [show] [--switch|-s] "
                      "[--metric|-m d|p|c]")


def cmd_best_machine(repl: Repl, argument: str) -> None:
    """Rank benchmarked machines and optionally move this session to one.

    Bare/`show` is read-only; `--switch`/`-s` resolves the machine to the
    host currently holding it, makes sure the model is available there and
    moves the session. `--metric`/`-m` picks decode (default), prefill or a
    normalized composite.
    """
    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        console.print(f"[err]unparsable arguments: {escape(str(exc))}[/err]")
        return
    show = False
    switch = False
    metric = "decode"
    index = 0
    while index < len(tokens):
        token = tokens[index]
        low = token.lower()
        if token in {"--help", "-h"} or low == "help":
            console.print(f"[dim]{escape(BEST_MACHINE_USAGE)}[/dim]")
            console.print("[dim]ranks benchmarked machines for the current "
                          "model; --switch moves this session to the host "
                          "holding the winner[/dim]")
            return
        if low == "show":
            show = True
        elif token in {"--switch", "-s"}:
            switch = True
        elif token in {"--metric", "-m"}:
            if index + 1 >= len(tokens):
                console.print(f"[err]{escape(BEST_MACHINE_USAGE)}[/err]")
                return
            try:
                metric = parse_metric(tokens[index + 1])
            except ValueError as exc:
                console.print(f"[err]{escape(str(exc))}[/err]")
                return
            index += 1
        elif low.startswith("--metric="):
            try:
                metric = parse_metric(token.split("=", 1)[1])
            except ValueError as exc:
                console.print(f"[err]{escape(str(exc))}[/err]")
                return
        else:
            try:
                metric = parse_metric(token)
            except ValueError:
                console.print(f"[err]{escape(BEST_MACHINE_USAGE)}[/err]")
                return
        index += 1
    if show and switch:
        console.print("[err]use either 'show' or --switch, not both[/err]")
        return

    inventory = load_cluster_inventory(repl.config)
    if inventory.error:
        console.print(f"[err]{escape(inventory.error)}[/err]")
        return
    max_age = repl.config.get("cluster_hosts_max_age", 86400)
    if inventory_stale(inventory, max_age):
        age = inventory_age(inventory)
        console.print(f"[warn]cluster inventory is {human_duration(age)} old — "
                      "the host/machine binding may be stale[/warn]")
    for warning in inventory.warnings[:3]:
        console.print(f"[dim]{escape(warning)}[/dim]")
    fleet = load_benchmarks(config=repl.config, inventory=inventory, refresh=True)
    if fleet.error:
        console.print(f"[err]{escape(fleet.error)}[/err]")
        return
    for warning in fleet.warnings[:3]:
        console.print(f"[dim]{escape(warning)}[/dim]")
    keys = [host.key for host in inventory.hosts.values()
            if host.machine and inventory.active(host.machine)]
    probes = probe_hosts(keys, config=repl.config, inventory=inventory)
    current = repl.current_machine()
    current_id = current.id if current is not None else None
    choice = choose_machine(
        repl.session.model, fleet=fleet, probes=probes, metric=metric,
        current=current_id,
        ask=lambda prompt, choices, default: repl.ui.ask(
            prompt, choices=choices, default=default))
    if choice.chosen is None:
        console.print(f"[err]no machine available: "
                      f"{escape(choice.reason)}[/err]")
        return
    console.print(machine_table(
        choice.ranked, chosen=choice.chosen.machine, current=current_id,
        metric=metric, model=repl.session.model))
    fastest = choice.ranked[0] if choice.ranked else None
    if fastest is not None and not fastest.available \
            and fastest.machine != choice.chosen.machine:
        console.print(f"[dim]fastest {escape(fastest.label)} is "
                      f"{escape(fastest.state_text(current_id))} — skipped[/dim]")
    console.print(f"[ok]{escape(choice_line(choice, metric))}[/ok]")
    if not switch:
        console.print("[dim]use /best-machine --switch (or -s) to move this "
                      "session[/dim]")
        return
    if choice.chosen.machine == current_id:
        console.print(f"[ok]already on the best machine[/ok] · host "
                      f"{escape(choice.host_key)}")
        if choice.chosen.has_model:
            return
        # The fastest machine is the current one but the model is missing;
        # `p` means pull it here, not "move to the same host again".
        chosen_model = ensure_model(
            repl.client, repl.session.model, repl.config,
            ssh_target=repl.tool_ctx.ssh_target,
            host_label=repl.session.remote, ui=repl.ui,
            sort=repl.model_sort, performance=_performance_index(repl))
        if chosen_model and chosen_model != repl.session.model:
            repl.session.model = chosen_model
            repl.session.touch()
            repl.refresh_models()
        return
    if not choice.host_key:
        console.print("[err]that machine has no connected host[/err]")
        return
    candidate = OllamaClient(choice.base,
                             insecure=getattr(repl.client, "insecure", False),
                             timeout=repl.timeout)
    try:
        version = candidate.version(timeout=REMOTE_PROBE_TIMEOUT)
    except OllamaError as exc:
        console.print(f"[err]cannot reach {escape(choice.host_key)}: "
                      f"{escape(str(exc))}[/err]")
        return
    local = bool(choice.chosen.probe and choice.chosen.probe.local)
    chosen_model = ensure_model(
        candidate, repl.session.model, repl.config,
        ssh_target=None if local else choice.ssh_target,
        host_label=choice.host_key,
        ui=repl.ui, sort=repl.model_sort,
        performance=_performance_index(repl))
    if not chosen_model:
        return
    if chosen_model != repl.session.model:
        repl.session.model = chosen_model
        repl.session.touch()
    repl.switch_host(choice.host_key, choice.base, ssh_target=choice.ssh_target,
                     local=local, client=candidate)
    console.print(f"[ok]now running on[/ok] {escape(choice.host_key)} "
                  f"· machine {escape(choice.chosen.machine)} "
                  f"· ollama {escape(version)}")
