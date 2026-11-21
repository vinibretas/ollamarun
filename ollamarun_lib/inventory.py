"""The cluster inventory: machines (hardware) and hosts (named endpoints).

`cluster-hosts.json` is the fleet's canonical inventory. Benchmarks score
*hardware*, but every request goes to a *host* (an SSH/Tailscale endpoint),
and a hostname can move to different hardware over time. This module is the
one place that knows that indirection:

- a `Machine` is a physical box (`i7-9700`, `r7-2700-1660`, ...) with a
  hardware `fingerprint` (the string profilebench logs in `Machine`) and the
  hostname windows it has used (`tower` Sep 15-19 on the Ryzen box, `tower`
  from Nov 20 on the OptiPlex box, ...);
- a `Host` is a named endpoint (`tower`, `pi`, `kiddo`, ...) with its SSH
  alias, tailnet name, addresses and the `machine` it currently holds;
- `hosts[*].machine` is the current binding, while `machines[*].hostnames`
  windows are the historical labels used to attribute old benchmark runs.

The file is read-only here. `last_seen_up`/`last_seen_source` are
observations written by `uptimes --record`; they are never treated as live
state. A missing `machines` section falls back to the old `host == machine`
world with a warning.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import CLUSTER_HOSTS_PATH, REMOTE_NODES
from .helpers import host_is_self, normalize_host, remote_node

#: Machine statuses that may run a request. Anything else is shown but never
#: selected; `retired` keeps the historical benchmark numbers visible.
ACTIVE_STATUSES = frozenset({"active", "", None})


def _norm(text: Any) -> str:
    """Case- and whitespace-insensitive form used for every lookup."""
    return " ".join(str(text or "").strip().lower().split())


def _date_of(value: Any) -> Optional[date]:
    """The date part of an ISO timestamp/date, or None."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@dataclass
class HostnameWindow:
    """One hostname's tenure on a machine (`last_seen` None = still open)."""

    name: str
    first_seen: str = ""
    last_seen: Optional[str] = None

    def contains(self, when: Optional[date]) -> bool:
        if when is None:
            return self.last_seen is None
        start = _date_of(self.first_seen)
        end = _date_of(self.last_seen) if self.last_seen else None
        if start is not None and when < start:
            return False
        if end is not None and when > end:
            return False
        return True


@dataclass
class Host:
    """A named endpoint (the thing `--remote`/SSH connects to)."""

    key: str
    machine: Optional[str] = None
    site: Optional[str] = None
    ssh_alias: Optional[str] = None
    tailscale: Optional[str] = None
    static: Optional[str] = None
    wifi_static: Optional[str] = None
    tailscale_ip: Optional[str] = None
    aliases: tuple[str, ...] = ()
    last_seen_up: Optional[str] = None
    last_seen_source: Optional[str] = None
    note: str = ""

    def names(self) -> set[str]:
        """Every name this endpoint answers to."""
        names = {self.key}
        for value in (self.ssh_alias, self.tailscale, self.static,
                      self.wifi_static, self.tailscale_ip, *self.aliases):
            if value:
                names.add(str(value))
        return {name.lower() for name in names}

    def last_seen_text(self) -> str:
        if not self.last_seen_up:
            return ""
        source = f" ({self.last_seen_source})" if self.last_seen_source else ""
        return f"{self.last_seen_up[:16].replace('T', ' ')}{source}"


@dataclass
class Machine:
    """One physical box and the hostname windows it has hosted."""

    id: str
    fingerprint: str = ""
    fingerprints: tuple[str, ...] = ()
    box: str = ""
    board: str = ""
    status: str = "active"
    note: str = ""
    hostnames: tuple[HostnameWindow, ...] = ()
    host_key: Optional[str] = None
    host: Optional[Host] = None
    synthetic: bool = False

    def specs(self) -> tuple[str, ...]:
        values = [self.fingerprint, *self.fingerprints]
        return tuple(value for value in values if value)

    def label(self) -> str:
        return f"{self.id} ({self.box})" if self.box else self.id

    def window_for(self, when: Optional[date]) -> Optional[HostnameWindow]:
        for window in self.hostnames:
            if window.contains(when):
                return window
        return None


@dataclass
class ClusterInventory:
    """The parsed inventory plus the lookups every consumer shares."""

    path: Path
    exists: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    sites: dict[str, dict[str, Any]] = field(default_factory=dict)
    machines: dict[str, Machine] = field(default_factory=dict)
    hosts: dict[str, Host] = field(default_factory=dict)
    order: tuple[str, ...] = ()
    updated: str = ""
    mtime: float = 0.0
    legacy: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    # ── basic lookups ─────────────────────────────────────────────────────
    def machine(self, machine_id: Optional[str]) -> Optional[Machine]:
        if not machine_id:
            return None
        return self.machines.get(machine_id)

    def host(self, key: Optional[str]) -> Optional[Host]:
        if not key:
            return None
        return self.hosts.get(key)

    def host_for_machine(self, machine_id: Optional[str]) -> Optional[Host]:
        machine = self.machine(machine_id)
        if machine is None:
            return None
        if machine.host is not None:
            return machine.host
        return self.hosts.get(machine.host_key or "")

    def active(self, machine_id: Optional[str]) -> bool:
        machine = self.machine(machine_id)
        return machine is not None and _norm(machine.status) in ACTIVE_STATUSES

    def status(self, machine_id: Optional[str]) -> str:
        machine = self.machine(machine_id)
        if machine is None:
            return "active"
        return _norm(machine.status) or "active"

    def box(self, machine_id: Optional[str]) -> str:
        machine = self.machine(machine_id)
        return machine.box if machine else ""

    def host_for_ref(self, ref: Optional[str]) -> Optional[Host]:
        """The endpoint behind a hostname, base URL, SSH alias or address."""
        raw = str(ref or "").strip()
        if not raw:
            return None
        candidates = {raw.lower()}
        candidates.add(remote_node(raw, self.config).lower())
        try:
            parsed = urllib.parse.urlsplit(
                raw if "://" in raw else "http://" + raw)
            if parsed.hostname:
                candidates.add(parsed.hostname.lower())
        except ValueError:
            pass
        for host in self.hosts.values():
            if candidates & host.names():
                return host
        # A local base/resume can carry no useful name; match this machine.
        try:
            if host_is_self(normalize_host(raw)):
                local = socket.gethostname().lower()
                for host in self.hosts.values():
                    if host.names() & {local, local.split(".")[0]}:
                        return host
        except ValueError:
            pass
        return None

    def machine_for_host(self, ref: Optional[str]) -> Optional[Machine]:
        host = self.host_for_ref(ref)
        if host is None:
            return None
        return self.machine(host.machine)

    def local_host(self) -> Optional[Host]:
        """The inventory endpoint that is this machine (hostname/addresses)."""
        local_name = socket.gethostname().lower()
        for host in self.hosts.values():
            if local_name in host.names() or local_name.split(".")[0] in host.names():
                return host
            for value in (host.key, host.ssh_alias, host.tailscale,
                          host.static, host.wifi_static, host.tailscale_ip):
                if value and host_is_self(value):
                    return host
        return None

    def remote_name(self, host_key: str) -> str:
        """The name `--remote`/`resolve_remote` should use for a host key."""
        host = self.hosts.get(host_key)
        if host is None:
            return host_key
        key = host_key.lower()
        configured = {str(name).lower()
                      for name in (self.config.get("remote_hosts") or {})}
        if key in REMOTE_NODES or key in configured:
            return host_key
        return host.tailscale or host.ssh_alias or host_key

    def ssh_target(self, host_key: str) -> Optional[str]:
        """The name SSH probes should use (the inventory's ssh alias)."""
        host = self.hosts.get(host_key)
        if host is None:
            return remote_node(host_key, self.config)
        return host.ssh_alias or self.remote_name(host_key)

    def hostnames_for(self, machine_id: str) -> list[str]:
        machine = self.machine(machine_id)
        if machine is None:
            return []
        return [window.name for window in machine.hostnames]

    # ── run attribution ───────────────────────────────────────────────────
    def machine_for_run(self, hostnames: Iterable[str] = (),
                        fingerprint: str = "", started: Any = None,
                        label: str = "") -> Optional[Machine]:
        """The machine a benchmark run belongs to.

        Order: the `Machine` hardware fingerprint first (exact, then a unique
        containment), then the run's date against the hostname windows,
        then an explicit machine id in the label/names. Never the current
        `hosts[*].machine` binding: old runs happened on the old hardware.
        """
        names = [str(name).strip() for name in hostnames if str(name or "").strip()]
        fp = _norm(fingerprint)
        if fp:
            exact = [machine for machine in self.machines.values()
                     if fp in {_norm(spec) for spec in machine.specs()}]
            if not exact:
                exact = [machine for machine in self.machines.values()
                         if any(len(_norm(spec)) >= 6
                                and (_norm(spec) in fp or fp in _norm(spec))
                                for spec in machine.specs())]
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                windowed = self._window_match(names, started)
                if windowed is not None and windowed in exact:
                    return windowed

        windowed = self._window_match(names, started)
        if windowed is not None:
            return windowed

        for token in (label, *names):
            norm = _norm(token)
            if not norm:
                continue
            exact = [machine for machine in self.machines.values()
                     if _norm(machine.id) == norm]
            if len(exact) == 1:
                return exact[0]

        if names:
            candidates = {machine.id: machine for machine in self.machines.values()
                          if any(_norm(window.name) in {_norm(name) for name in names}
                                 for window in machine.hostnames)}
            if len(candidates) == 1:
                return next(iter(candidates.values()))
        return None

    def _window_match(self, names: list[str], started: Any) -> Optional[Machine]:
        when = _date_of(started)
        wanted = {_norm(name) for name in names}
        if not wanted:
            return None
        matches: list[tuple[str, Machine]] = []
        for machine in self.machines.values():
            for window in machine.hostnames:
                if _norm(window.name) not in wanted:
                    continue
                if not window.contains(when):
                    continue
                matches.append((str(window.first_seen or ""), machine))
        if not matches:
            return None
        matches.sort(key=lambda pair: pair[0], reverse=True)
        newest = matches[0][0]
        candidates = {machine.id: machine for first, machine in matches
                      if first == newest}
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        return None


def inventory_path(config: Optional[dict[str, Any]] = None,
                   path: Optional[str] = None) -> Path:
    """Where the inventory lives: explicit path > env > config > default."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get("CLUSTER_HOSTS_FILE")
    if env:
        return Path(env).expanduser()
    configured = (config or {}).get("cluster_hosts")
    if configured:
        return Path(str(configured)).expanduser()
    return Path(CLUSTER_HOSTS_PATH).expanduser()


def load_cluster_inventory(config: Optional[dict[str, Any]] = None,
                           path: Optional[str] = None) -> ClusterInventory:
    """Read the inventory (no cache: it is updated at runtime by the fleet)."""
    config = dict(config or {})
    target = inventory_path(config, path)
    inventory = ClusterInventory(path=target, config=config)
    if not target.exists():
        inventory.error = (f"cluster inventory not found at {target} "
                           "(set cluster_hosts or CLUSTER_HOSTS_FILE)")
        inventory.warnings.append(inventory.error)
        return inventory
    try:
        inventory.mtime = target.stat().st_mtime
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        inventory.error = f"cannot read cluster inventory {target}: {exc}"
        inventory.warnings.append(inventory.error)
        return inventory
    if not isinstance(raw, dict):
        inventory.error = f"cluster inventory {target} is not a JSON object"
        inventory.warnings.append(inventory.error)
        return inventory
    inventory.exists = True
    inventory.updated = str(raw.get("updated") or "")
    inventory.sites = {str(key): value for key, value in
                       (raw.get("sites") or {}).items() if isinstance(value, dict)}
    inventory.order = tuple(str(item) for item in (raw.get("order") or []) if item)
    _parse_machines(inventory, raw.get("machines") or {})
    _parse_hosts(inventory, raw.get("hosts") or {})
    if not inventory.machines:
        _legacy_machines(inventory)
    _bind_hosts(inventory)
    return inventory


def _parse_machines(inventory: ClusterInventory, raw: Any) -> None:
    if not isinstance(raw, dict):
        inventory.warnings.append("cluster inventory: 'machines' is not an object")
        return
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        windows = []
        for window in entry.get("hostnames") or []:
            if isinstance(window, dict) and window.get("name"):
                windows.append(HostnameWindow(
                    name=str(window["name"]),
                    first_seen=str(window.get("first_seen") or ""),
                    last_seen=(str(window["last_seen"])
                               if window.get("last_seen") else None)))
        inventory.machines[str(key)] = Machine(
            id=str(key),
            fingerprint=str(entry.get("fingerprint") or ""),
            fingerprints=tuple(_as_list(entry.get("fingerprints"))),
            box=str(entry.get("box") or ""),
            board=str(entry.get("board") or ""),
            status=str(entry.get("status") or "active"),
            note=str(entry.get("note") or ""),
            hostnames=tuple(windows),
        )


def _parse_hosts(inventory: ClusterInventory, raw: Any) -> None:
    if not isinstance(raw, dict):
        inventory.warnings.append("cluster inventory: 'hosts' is not an object")
        return
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        inventory.hosts[str(key)] = Host(
            key=str(key),
            machine=str(entry["machine"]) if entry.get("machine") else None,
            site=str(entry["site"]) if entry.get("site") else None,
            ssh_alias=str(entry["ssh_alias"]) if entry.get("ssh_alias") else None,
            tailscale=str(entry["tailscale"]) if entry.get("tailscale") else None,
            static=str(entry["static"]) if entry.get("static") else None,
            wifi_static=(str(entry["wifi_static"])
                         if entry.get("wifi_static") else None),
            tailscale_ip=(str(entry["tailscale_ip"])
                          if entry.get("tailscale_ip") else None),
            aliases=tuple(_as_list(entry.get("aliases"))),
            last_seen_up=(str(entry["last_seen_up"])
                          if entry.get("last_seen_up") else None),
            last_seen_source=(str(entry["last_seen_source"])
                              if entry.get("last_seen_source") else None),
            note=str(entry.get("note") or ""),
        )


def _legacy_machines(inventory: ClusterInventory) -> None:
    """Inventory without a `machines` section: host name is the machine id."""
    if inventory.hosts:
        inventory.legacy = True
        inventory.warnings.append(
            "cluster inventory has no 'machines' section; using host names "
            "as machine ids (moved hardware may be misattributed)")
    for key, host in inventory.hosts.items():
        machine = Machine(id=key, status="active", host_key=key, host=host,
                          synthetic=True)
        inventory.machines[key] = machine
        host.machine = key if host.machine is None else host.machine


def _bind_hosts(inventory: ClusterInventory) -> None:
    """Wire `hosts[*].machine` into the machines, warning on bad claims."""
    seen: dict[str, str] = {}
    ordered = [key for key in inventory.order if key in inventory.hosts]
    ordered += [key for key in inventory.hosts if key not in ordered]
    for key in ordered:
        host = inventory.hosts[key]
        machine_id = host.machine
        if machine_id is None:
            if not inventory.legacy and key in inventory.machines:
                machine_id = key
                host.machine = key
            else:
                inventory.warnings.append(
                    f"host {key} has no machine binding; it cannot be selected")
                continue
        machine = inventory.machines.get(machine_id)
        if machine is None:
            machine = Machine(id=machine_id, status="active",
                              host_key=key, host=host, synthetic=True)
            inventory.machines[machine_id] = machine
            inventory.warnings.append(
                f"host {key} points at unknown machine {machine_id}; "
                "keeping it as an unlisted machine")
        previous = seen.get(machine_id)
        if previous:
            inventory.warnings.append(
                f"machines {machine_id} is claimed by hosts {previous} and "
                f"{key}; keeping {previous}")
            continue
        seen[machine_id] = key
        machine.host_key = key
        machine.host = host

    # Machines with no host are unconnected; nothing to bind.
    for machine in inventory.machines.values():
        if machine.host is None and machine.host_key:
            machine.host = inventory.hosts.get(machine.host_key)


def inventory_age(inventory: ClusterInventory) -> Optional[float]:
    """Seconds since the inventory file was written (None if unknown)."""
    if not inventory.mtime:
        return None
    return max(0.0, time.time() - inventory.mtime)


def inventory_stale(inventory: ClusterInventory, max_age: Optional[float]) -> bool:
    """Whether the file is older than `max_age` seconds (0/None disables)."""
    if not max_age or max_age <= 0:
        return False
    age = inventory_age(inventory)
    return age is not None and age > float(max_age)
