"""Pick the best machine for a model and the host currently holding it.

`benchmarks.py` ranks hardware; this module turns that into an actionable
host. It probes every candidate endpoint (`/api/version` + `/api/tags`),
joins each machine to the host in the cluster inventory, and applies the
availability/model policy:

- only machines whose current host answers are eligible;
- a retired/maintenance machine, or a machine with no host, is shown but
  skipped;
- the fastest available machine wins even if it needs a pull, except when
  the session is already on the fastest machine and a lower-ranked machine
  already has the model -- then the caller is asked to pull on the fastest
  or use the one that has it.

The ranking metric is decode tok/s by default (`d`), prefill (`p`) or a
normalized composite (`c`). Nothing here prints; the CLI and the `/best-machine`
command own presentation.
"""
from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from rich import box
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from .benchmarks import (FleetBenchmarks, MachineBench, ModelBench,
                         parse_parameter_count, stamp)
from .client import OllamaClient
from .constants import DEFAULT_HOST, REMOTE_PROBE_TIMEOUT
from .helpers import (canonical_model_name, host_is_self, normalize_host,
                      remote_node, resolve_remote, tailscale_peer_state)
from .inventory import ClusterInventory, Host

METRIC_ALIASES = {
    "d": "decode", "decode": "decode",
    "p": "prefill", "prefill": "prefill",
    "c": "composite", "composite": "composite",
}
METRIC_LABELS = {
    "decode": "decode tok/s",
    "prefill": "prefill tok/s",
    "composite": "composite score",
}
#: Estimated rows lose a little against a direct measurement when ranked.
ESTIMATE_DISCOUNT = 0.9


def parse_metric(value: Any) -> str:
    """Canonical metric name; ValueError names the accepted values."""
    text = str(value or "").strip().lower()
    if not text:
        return "decode"
    metric = METRIC_ALIASES.get(text)
    if metric is None:
        raise ValueError("machine metric must be d/decode, p/prefill or "
                         "c/composite")
    return metric


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _ago(value: str) -> str:
    """'6d ago' from an ISO timestamp; the raw date when unparsable."""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value or "")[:16].replace("T", " ")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(stamp.tzinfo) - stamp).total_seconds()))
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


@dataclass
class HostProbe:
    """One endpoint's live answer."""

    key: str
    remote_name: str
    base: str
    local: bool = False
    online: bool = False
    version: str = ""
    error: str = ""
    tags: tuple[dict[str, Any], ...] = ()
    installed: dict[str, dict[str, Any]] = field(default_factory=dict)
    ssh_target: Optional[str] = None
    site: Optional[str] = None
    last_seen_up: Optional[str] = None
    last_seen_source: Optional[str] = None
    tailscale_state: str = ""

    def has_model(self, model: str, digest: str = "") -> bool:
        if not model:
            return True
        if canonical_model_name(model).lower() in self.installed:
            return True
        if digest:
            for tag in self.installed.values():
                if _digest(tag.get("digest")) == digest:
                    return True
        return False


def _factory_client(factory: Callable[..., Any], base: str, timeout: float) -> Any:
    try:
        return factory(base, timeout=timeout)
    except TypeError:
        return factory(base)


def probe_hosts(host_keys: Any, *, config: Optional[dict[str, Any]] = None,
                inventory: ClusterInventory,
                timeout: float = REMOTE_PROBE_TIMEOUT,
                client_factory: Callable[..., Any] = OllamaClient
                ) -> dict[str, HostProbe]:
    """Probe every endpoint in parallel; never raises for a dead host."""
    config = dict(config or {})
    keys = list(dict.fromkeys(str(key) for key in host_keys if key))
    if not keys:
        return {}
    workers = max(1, min(8, len(keys)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda key: _probe_one(key, config, inventory, timeout, client_factory),
            keys))
    return {probe.key: probe for probe in results}


def _probe_one(key: str, config: dict[str, Any], inventory: ClusterInventory,
               timeout: float, factory: Callable[..., Any]) -> HostProbe:
    entry = inventory.hosts.get(key)
    remote_name = inventory.remote_name(key)
    ssh_target = inventory.ssh_target(key)
    local = bool(host_is_self(key))
    if entry is not None:
        local = local or any(host_is_self(value) for value in (
            entry.ssh_alias, entry.tailscale, entry.static, entry.wifi_static)
            if value)
    bases: list[str] = []
    if local:
        local_base = (config.get("host") or os.environ.get("OLLAMA_HOST")
                      or DEFAULT_HOST)
        bases.append(normalize_host(local_base))
    try:
        resolved = resolve_remote(remote_name, config)
    except ValueError:
        resolved = ""
    if resolved and resolved not in bases:
        bases.append(resolved)
    tailscale_state = "this machine" if local else _tailscale_state(
        remote_name, config)
    errors: list[str] = []
    for base in bases:
        try:
            client = _factory_client(factory, base, timeout)
            version = client.version(timeout=timeout)
            tags = client.tags()
        except Exception as exc:  # a dead endpoint must not break the table
            errors.append(str(exc) or exc.__class__.__name__)
            continue
        installed: dict[str, dict[str, Any]] = {}
        clean_tags: list[dict[str, Any]] = []
        for tag in tags or []:
            if not isinstance(tag, dict):
                continue
            clean_tags.append(tag)
            name = str(tag.get("name") or "")
            if name:
                installed[canonical_model_name(name).lower()] = tag
        return HostProbe(
            key=key, remote_name=remote_name, base=base, local=local,
            online=True, version=str(version or ""),
            tags=tuple(clean_tags), installed=installed,
            ssh_target=ssh_target,
            site=entry.site if entry else None,
            last_seen_up=entry.last_seen_up if entry else None,
            last_seen_source=entry.last_seen_source if entry else None,
            tailscale_state=tailscale_state)
    return HostProbe(
        key=key, remote_name=remote_name, base=bases[0] if bases else "",
        local=local, online=False,
        error="; ".join(errors) or "no reachable address",
        ssh_target=ssh_target,
        site=entry.site if entry else None,
        last_seen_up=entry.last_seen_up if entry else None,
        last_seen_source=entry.last_seen_source if entry else None,
        tailscale_state=tailscale_state)


def _tailscale_state(remote_name: str, config: dict[str, Any]) -> str:
    try:
        return tailscale_peer_state(remote_node(remote_name, config))
    except ValueError:
        return ""


@dataclass
class MachinePerf:
    """One machine's ranking row for a model and metric."""

    machine: str
    box: str = ""
    status: str = "active"
    host: Optional[Host] = None
    probe: Optional[HostProbe] = None
    decode: Optional[float] = None
    prefill: Optional[float] = None
    composite: Optional[float] = None
    factor: float = 1.0
    method: str = "none"
    profile: str = ""
    runs: int = 0
    last_run: str = ""
    proxy: str = ""
    has_model: bool = False
    online: bool = False
    available: bool = False

    @property
    def host_key(self) -> str:
        return self.host.key if self.host is not None else ""

    @property
    def ssh_target(self) -> Optional[str]:
        if self.probe is not None:
            return self.probe.ssh_target
        return None

    @property
    def base(self) -> str:
        return self.probe.base if self.probe is not None else ""

    @property
    def site(self) -> Optional[str]:
        return self.host.site if self.host is not None else None

    @property
    def label(self) -> str:
        return f"{self.machine} ({self.box})" if self.box else self.machine

    def value(self, metric: str) -> Optional[float]:
        if metric == "decode":
            return self.decode
        if metric == "prefill":
            return self.prefill
        return self.composite

    def rank_score(self, metric: str) -> Optional[float]:
        value = self.value(metric)
        if value is None and self.method == "factor":
            value = self.factor
        if value is None:
            return None
        return value * (ESTIMATE_DISCOUNT
                        if self.method in {"proxy_factor", "factor"} else 1.0)

    def state_text(self, current: Optional[str] = None) -> str:
        status = self.status or "active"
        if status not in {"active", ""}:
            return status
        if self.host is None:
            return "no host"
        if self.online:
            if current and self.machine == current:
                return "this host"
            if self.probe is not None and self.probe.local:
                return "online · local"
            return "online"
        seen = ""
        if self.host.last_seen_up:
            seen = f" · last seen {_ago(self.host.last_seen_up)}"
            if self.host.last_seen_source:
                seen += f" ({self.host.last_seen_source})"
        return "offline" + seen


def _target_identity(model: str, probes: dict[str, HostProbe],
                     fleet: FleetBenchmarks,
                     digest: str = "", params: Optional[float] = None,
                     family: str = "", quant: str = "") -> tuple[str, Optional[float], str, str]:
    canonical = canonical_model_name(model).lower() if model else ""
    for probe in probes.values():
        tag = probe.installed.get(canonical)
        if not tag:
            continue
        details = tag.get("details") or {}
        return (_digest(tag.get("digest")),
                params if params is not None else parse_parameter_count(
                    details.get("parameter_size")),
                family or str(details.get("family") or ""),
                quant or str(details.get("quantization_level") or ""))
    if model:
        for machine in fleet.machines.values():
            bench = machine.bench(model=model)
            if bench is not None:
                return (digest or bench.digest,
                        params if params is not None else bench.params,
                        family or bench.family,
                        quant or bench.quant)
    return _digest(digest), params, family, quant


def _digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text[:12]


def rank_machines(model: str, *, fleet: FleetBenchmarks,
                  probes: dict[str, HostProbe],
                  metric: str = "decode", digest: str = "",
                  params: Optional[float] = None, family: str = "",
                  quant: str = "") -> list[MachinePerf]:
    """Every measured machine, ordered by the requested metric."""
    metric = parse_metric(metric)
    digest, params, family, quant = _target_identity(
        model, probes, fleet, digest, params, family, quant)
    proxy = fleet.nearest(params, family, quant) if model else None
    rows: list[MachinePerf] = []
    for machine_id, machine in fleet.machines.items():
        row = _rank_one(machine_id, machine, model=model, digest=digest,
                        fleet=fleet, probes=probes, metric=metric,
                        proxy=proxy)
        rows.append(row)
    _add_composite(rows)
    rows.sort(key=lambda row: (
        row.rank_score(metric) is None,
        -(row.rank_score(metric) or 0.0),
        not row.available,
        row.machine))
    return rows


def _rank_one(machine_id: str, machine: MachineBench, *, model: str,
              digest: str, fleet: FleetBenchmarks,
              probes: dict[str, HostProbe], metric: str,
              proxy: Optional[ModelBench]) -> MachinePerf:
    host = fleet.host_for_machine(machine_id)
    probe = probes.get(host.key) if host is not None else None
    status = fleet.status(machine_id)
    bench = machine.bench(digest=digest, model=model)
    method = "none"
    profile = ""
    runs = 0
    last_run = ""
    proxy_name = ""
    decode: Optional[float] = None
    prefill: Optional[float] = None
    factor = fleet.machine_factor(machine_id, "decode")
    if bench is not None:
        decode = bench.tps
        prefill = bench.prefill
        profile = bench.profile
        runs = bench.runs
        last_run = bench.last_run
        if digest and bench.digest == digest:
            method = "measured"
        elif model and bench.has_name(model):
            method = "tag"
        else:
            method = "measured"
    elif proxy is not None:
        proxy_name = ", ".join(proxy.names)
        pbench = machine.models.get(proxy.key)
        if pbench is not None:
            decode = pbench.tps
            prefill = pbench.prefill
            profile = pbench.profile
            runs = pbench.runs
            last_run = pbench.last_run
            method = "proxy"
        else:
            base_decode = fleet.median_for(proxy.key, "decode")
            base_prefill = fleet.median_for(proxy.key, "prefill")
            if base_decode:
                decode = base_decode * fleet.machine_factor(machine_id, "decode")
            if base_prefill:
                prefill = base_prefill * fleet.machine_factor(machine_id, "prefill")
            method = "proxy_factor"
            profile = proxy.profile
    else:
        if metric == "prefill":
            factor = fleet.machine_factor(machine_id, "prefill")
        method = "factor"
    online = bool(probe and probe.online)
    available = (status in {"active", ""} and host is not None and online)
    return MachinePerf(
        machine=machine_id, box=fleet.box(machine_id), status=status,
        host=host, probe=probe, decode=decode, prefill=prefill,
        factor=factor, method=method, profile=profile, runs=runs,
        last_run=last_run, proxy=proxy_name,
        has_model=bool(probe and probe.has_model(model, digest)) if model else True,
        online=online, available=available)


def _add_composite(rows: list[MachinePerf]) -> None:
    max_decode = max((row.decode for row in rows if row.decode), default=0.0)
    max_prefill = max((row.prefill for row in rows if row.prefill), default=0.0)
    for row in rows:
        parts: list[tuple[float, float]] = []
        if row.decode and max_decode:
            parts.append((0.8, row.decode / max_decode))
        if row.prefill and max_prefill:
            parts.append((0.2, row.prefill / max_prefill))
        if parts:
            weight = sum(part[0] for part in parts)
            row.composite = sum(part[0] * part[1] for part in parts) / weight
        elif row.method == "factor":
            row.composite = row.factor


@dataclass
class MachineChoice:
    """The selected machine plus the rows/notes behind it."""

    chosen: Optional[MachinePerf]
    ranked: list[MachinePerf] = field(default_factory=list)
    available: list[MachinePerf] = field(default_factory=list)
    reason: str = ""
    note: str = ""
    only_one: bool = False

    @property
    def skipped(self) -> list[MachinePerf]:
        return [row for row in self.ranked if not row.available]

    @property
    def host_key(self) -> str:
        return self.chosen.host_key if self.chosen else ""

    @property
    def base(self) -> str:
        return self.chosen.base if self.chosen else ""

    @property
    def ssh_target(self) -> Optional[str]:
        return self.chosen.ssh_target if self.chosen else None


def choose_machine(model: str, *, fleet: FleetBenchmarks,
                   probes: dict[str, HostProbe],
                   metric: str = "decode", current: Optional[str] = None,
                   ask: Optional[Callable[[str, list[str], str], str]] = None,
                   assume_yes: bool = False,
                   digest: str = "", params: Optional[float] = None,
                   family: str = "", quant: str = "") -> MachineChoice:
    """Rank, filter to available machines and apply the model policy."""
    ranked = rank_machines(model, fleet=fleet, probes=probes, metric=metric,
                           digest=digest, params=params, family=family,
                           quant=quant)
    available = [row for row in ranked if row.available]
    if not available:
        return MachineChoice(
            chosen=None, ranked=ranked, available=[],
            reason=_no_available_text(ranked))
    best = available[0]
    note = ""
    if len(available) == 1:
        note = f"only {best.machine} is available — nothing to compare"
    if not model or best.has_model:
        return MachineChoice(best, ranked, available, "best available machine",
                             note, len(available) == 1)
    alternatives = [row for row in available[1:] if row.has_model]
    if current and best.machine == current and alternatives:
        alternative = alternatives[0]
        default = "p" if assume_yes else "s"
        prompt = (f"{best.label} is the fastest but does not have {model}; "
                  f"pull it there (p) or use {alternative.label} which has "
                  f"it (s)?")
        choice = default
        if ask is not None and not assume_yes:
            try:
                choice = ask(prompt, ["p", "s"], default) or default
            except (EOFError, KeyboardInterrupt):
                choice = default
        if str(choice).strip().lower().startswith("s"):
            return MachineChoice(alternative, ranked, available,
                                 "second best that already has the model",
                                 note, len(available) == 1)
        return MachineChoice(best, ranked, available,
                             "pull the model on the fastest machine",
                             note, len(available) == 1)
    return MachineChoice(best, ranked, available,
                         "fastest available machine", note,
                         len(available) == 1)


def _no_available_text(rows: list[MachinePerf]) -> str:
    if not rows:
        return "no benchmarked machines found"
    parts = []
    for row in rows:
        if row.status not in {"active", ""}:
            parts.append(f"{row.machine} {row.status}")
        elif row.host is None:
            parts.append(f"{row.machine} has no host")
        else:
            parts.append(f"{row.host.key} offline")
    return "no active machine with an online host (" + "; ".join(parts[:5]) + ")"


def format_value(row: MachinePerf, metric: str) -> str:
    """The metric cell: `16.1`, `1,456`, `0.93`, `~1.23`, `1.40×` or `–`."""
    value = row.value(metric)
    if value is None:
        if row.method == "factor":
            return f"{row.factor:.2f}×"
        return "–"
    prefix = "~" if row.method in {"proxy_factor", "factor"} else ""
    if metric == "decode":
        return f"{prefix}{value:.1f}"
    if metric == "prefill":
        return f"{prefix}{value:,.0f}"
    return f"{prefix}{value:.2f}"


def choice_line(choice: MachineChoice, metric: str) -> str:
    """One-line decision summary (used by the CLI and `/best-machine`)."""
    if choice.chosen is None:
        return f"no machine available: {choice.reason}"
    row = choice.chosen
    value = format_value(row, metric)
    if row.method == "factor":
        unit = ""
    else:
        unit = "tok/s" if metric in {"decode", "prefill"} else "score"
    runs = f", {row.runs} run{'s' if row.runs != 1 else ''}" if row.runs else ""
    text = f"best machine: {row.label} → host {row.host_key or '?'} · {value}"
    if unit:
        text += f" {unit}"
    text += runs
    if row.method == "proxy" and row.proxy:
        text += f" · estimated from {row.proxy}"
    elif row.method in {"proxy_factor", "factor"}:
        text += " · estimated from the machine speed factor"
    if choice.note:
        text += f" · {choice.note}"
    return text


def machine_table(rows: list[MachinePerf], *, chosen: Optional[str] = None,
                  current: Optional[str] = None, metric: str = "decode",
                  model: str = "") -> Table:
    """The ranking table shared by `--best-machine` and `/best-machine`."""
    title = "best machine"
    if model:
        title += f" · {escape(model)}"
    title += f" · {metric_label(metric)}"
    table = Table(title=title, box=box.ROUNDED, header_style="bold magenta",
                  row_styles=["", "dim"])
    table.add_column("", justify="center", width=2)
    table.add_column("machine")
    table.add_column("host")
    table.add_column("state")
    table.add_column("tok/s" if metric != "composite" else "score",
                     justify="right")
    table.add_column("profile", justify="center")
    table.add_column("runs", justify="right")
    table.add_column("newest")
    for row in rows:
        marker = "→" if chosen == row.machine else (
            "*" if current == row.machine else "")
        machine = Text(row.machine)
        if row.box:
            machine.append(f" {row.box}", style="dim")
        if row.host is not None:
            host = Text(row.host.key)
            if row.site:
                host.append(f" · {row.site}", style="dim")
            if row.host.ssh_alias and row.host.ssh_alias != row.host.key:
                host.append(f" · ssh {row.host.ssh_alias}", style="dim")
        else:
            host = Text("—")
        profile = row.profile or "–"
        if row.method in {"proxy", "proxy_factor"} and row.proxy:
            profile += " ~"
        table.add_row(
            marker, machine, host, row.state_text(current),
            format_value(row, metric), profile,
            str(row.runs) if row.runs else "–",
            stamp(row.last_run) if row.last_run else "–")
    return table


__all__ = [
    "HostProbe", "MachineChoice", "MachinePerf", "choice_line", "choose_machine",
    "format_value", "machine_table", "metric_label", "parse_metric",
    "probe_hosts", "rank_machines",
]
