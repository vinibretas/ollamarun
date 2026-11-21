"""profilebench results: measured throughput per hardware machine.

The benchmark tree under `benchmarks_dir` (`~/llms/benchmarks/profilebench`
by default) records one run directory per sweep. Each run's `matrix_full.csv`
has one row per model with the decode/prefill throughput of every offload
profile plus the `Machine` hardware fingerprint; very old runs only have
`matrix.csv` and a `state.json`.

This module turns that tree into two things:

- `FleetBenchmarks`, per-machine measurements (median across runs, merged
  across historical labels through `inventory.py`), with the hostname windows
  attributing old runs to the hardware they actually ran on;
- `PerformanceIndex`, the model -> best measured throughput map behind
  `--sort performance` (active machines first, retired hardware only as a
  fallback).

It never talks to Ollama. Availability/ranking lives in `bestmachine.py`.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .constants import BENCHMARKS_DIR
from .helpers import canonical_model_name, short_stamp
from .inventory import ClusterInventory, load_cluster_inventory

#: The profiles profilebench sweeps, in fallback order. `G` is the full
#: offload; a model that does not fit still gets an `auto` split cell.
PROFILES: tuple[str, ...] = ("C", "H1", "H2", "H3", "G")
VALID_STATES = frozenset({"ok", "auto"})
_RUN_DIR_RE = re.compile(r"^run_\d{8}_\d{6}_(?P<label>.+)$")
_PARAM_RE = re.compile(
    r"^\s*(?:(?P<count>\d+(?:\.\d+)?)\s*[x×]\s*)?"
    r"(?P<each>\d+(?:\.\d+)?)\s*(?P<unit>[bmk])\b", re.IGNORECASE)
_PARAM_UNITS = {"b": 1.0, "m": 0.001, "k": 0.000001}


def parse_parameter_count(text: Any) -> Optional[float]:
    """Parameter count in billions from `7.6B`, `134.52M`, `8x7B`, ..."""
    raw = str(text or "").strip()
    match = _PARAM_RE.match(raw)
    if not match:
        return None
    count = float(match.group("each")) * _PARAM_UNITS[match.group("unit").lower()]
    if match.group("count"):
        count *= float(match.group("count"))
    return count


def _num(value: Any) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return str(value or "").strip()


def benchmarks_root(config: Optional[dict[str, Any]] = None,
                    root: Optional[str] = None) -> Path:
    if root:
        return Path(root).expanduser()
    configured = (config or {}).get("benchmarks_dir")
    return Path(str(configured or BENCHMARKS_DIR)).expanduser()


@dataclass(frozen=True)
class BenchSample:
    """One model measured in one run on one machine."""

    machine: str
    model: str
    digest: str
    params: Optional[float]
    family: str
    quant: str
    tps: Optional[float]
    prefill: Optional[float]
    profile: str
    prefill_profile: str
    run: str
    started: str
    source: str


@dataclass
class ModelBench:
    """A model's measurements on one machine, aggregated across runs."""

    machine: str
    key: str
    digest: str
    names: tuple[str, ...]
    params: Optional[float]
    family: str
    quant: str
    tps: float
    tps_min: float
    tps_max: float
    prefill: Optional[float]
    profile: str
    runs: int
    last_run: str
    samples: tuple[BenchSample, ...]

    def has_name(self, name: str) -> bool:
        wanted = canonical_model_name(name).lower()
        return any(canonical_model_name(item).lower() == wanted
                   for item in self.names)


@dataclass
class MachineBench:
    """Everything measured on one hardware machine."""

    machine: str
    models: dict[str, ModelBench] = field(default_factory=dict)
    by_digest: dict[str, ModelBench] = field(default_factory=dict)
    by_name: dict[str, ModelBench] = field(default_factory=dict)
    runs: set[str] = field(default_factory=set)
    last_run: str = ""
    decode_factor: float = 1.0
    prefill_factor: float = 1.0

    def bench(self, *, key: str = "", digest: str = "",
              model: str = "") -> Optional[ModelBench]:
        if key and key in self.models:
            return self.models[key]
        if digest:
            wanted = _digest_key(digest)
            if wanted in self.by_digest:
                return self.by_digest[wanted]
        if model:
            return self.by_name.get(canonical_model_name(model).lower())
        return None


@dataclass(frozen=True)
class FleetPerf:
    """A model's best measured throughput on a machine (performance class)."""

    tps: float
    machine: str
    box: str
    runs: int
    last_run: str
    profile: str
    historical: bool = False


class PerformanceIndex:
    """Model lookup for `--sort performance` (digest first, then tag)."""

    def __init__(self, path: Optional[Path] = None, error: str = "") -> None:
        self.path = path
        self.error = error
        self.by_digest: dict[str, FleetPerf] = {}
        self.by_name: dict[str, FleetPerf] = {}

    def __bool__(self) -> bool:
        return bool(self.by_digest or self.by_name)

    def of(self, model: dict[str, Any]) -> Optional[FleetPerf]:
        digest = _digest_key(model.get("digest") or "")
        if digest:
            found = self.by_digest.get(digest)
            if found is not None:
                return found
        name = canonical_model_name(str(model.get("name") or "")).lower()
        found = self.by_name.get(name)
        if found is not None:
            return found
        if ":" in name:
            return self.by_name.get(name.rsplit(":", 1)[0])
        return None


@dataclass
class FleetBenchmarks:
    """The parsed tree plus the per-machine lookup every consumer shares."""

    root: Path
    inventory: ClusterInventory
    machines: dict[str, MachineBench] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def __bool__(self) -> bool:
        return bool(self.machines)

    def status(self, machine_id: str) -> str:
        return self.inventory.status(machine_id) if self.inventory.machines else "active"

    def box(self, machine_id: str) -> str:
        return self.inventory.box(machine_id)

    def active(self, machine_id: str) -> bool:
        return self.status(machine_id) in {"active", ""}

    def host_for_machine(self, machine_id: str):
        return self.inventory.host_for_machine(machine_id)

    def bench(self, machine_id: str, *, digest: str = "", model: str = "",
              key: str = "") -> Optional[ModelBench]:
        machine = self.machines.get(machine_id)
        if machine is None:
            return None
        return machine.bench(key=key, digest=digest, model=model)

    def nearest(self, params: Optional[float], family: str = "",
                quant: str = "") -> Optional[ModelBench]:
        """The closest measured model by parameter count (same family first)."""
        if not params or params <= 0:
            return None
        best: Optional[tuple[float, bool, bool, ModelBench]] = None
        seen: set[str] = set()
        for machine in self.machines.values():
            for bench in machine.models.values():
                if bench.key in seen or not bench.params or bench.params <= 0:
                    continue
                seen.add(bench.key)
                ratio = max(params, bench.params) / min(params, bench.params)
                if ratio > 4.0:
                    continue
                score = abs(math.log10(params / bench.params))
                if family and bench.family and family.lower() != bench.family.lower():
                    score += 0.25
                if quant and bench.quant and quant.lower() != bench.quant.lower():
                    score += 0.05
                entry = (score, bool(family and bench.family.lower() == family.lower()),
                         bool(quant and bench.quant.lower() == quant.lower()), bench)
                if best is None or entry[:3] < best[:3]:
                    best = entry
        return best[3] if best else None

    def median_for(self, key: str, metric: str = "decode") -> Optional[float]:
        values: list[float] = []
        for machine in self.machines.values():
            bench = machine.models.get(key)
            if bench is None:
                continue
            value = bench.tps if metric == "decode" else bench.prefill
            if value:
                values.append(float(value))
        return statistics.median(values) if values else None

    def machine_factor(self, machine_id: str, metric: str = "decode") -> float:
        machine = self.machines.get(machine_id)
        if machine is None:
            return 1.0
        return machine.decode_factor if metric == "decode" else machine.prefill_factor

    def labels(self) -> list[str]:
        return [str(window.name) for machine in self.inventory.machines.values()
                for window in machine.hostnames]

    def performance_index(self) -> PerformanceIndex:
        index = PerformanceIndex(self.root, self.error)
        for machine in self.machines.values():
            for bench in machine.models.values():
                historical = not self.active(machine.machine)
                perf = FleetPerf(
                    tps=bench.tps, machine=machine.machine,
                    box=self.box(machine.machine), runs=bench.runs,
                    last_run=bench.last_run, profile=bench.profile,
                    historical=historical)
                if bench.digest:
                    index.by_digest[bench.digest] = _prefer(index.by_digest.get(bench.digest), perf)
                for name in bench.names:
                    key = canonical_model_name(name).lower()
                    index.by_name[key] = _prefer(index.by_name.get(key), perf)
        return index


def _prefer(current: Optional[FleetPerf], candidate: FleetPerf) -> FleetPerf:
    """Active beats historical; then the higher measured throughput wins."""
    if current is None:
        return candidate
    if current.historical != candidate.historical:
        return candidate if current.historical else current
    return candidate if candidate.tps > current.tps else current


def _digest_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text[:12]


_CACHE: dict[tuple[Any, ...], FleetBenchmarks] = {}


def load_benchmarks(config: Optional[dict[str, Any]] = None,
                    root: Optional[str] = None,
                    inventory: Optional[ClusterInventory] = None,
                    refresh: bool = False) -> FleetBenchmarks:
    """Parse (or reuse) the profilebench tree for `root`/`config`."""
    config = dict(config or {})
    path = benchmarks_root(config, root)
    if inventory is None:
        inventory = load_cluster_inventory(config)
    fingerprint = _tree_fingerprint(path, inventory)
    cached = _CACHE.get(fingerprint)
    if cached is not None and not refresh:
        return cached
    fleet = _parse_tree(path, inventory)
    _CACHE[fingerprint] = fleet
    # Bound the cache: one entry per distinct tree generation is enough.
    while len(_CACHE) > 8:
        _CACHE.pop(next(iter(_CACHE)))
    return fleet


def _tree_fingerprint(path: Path, inventory: ClusterInventory) -> tuple[Any, ...]:
    entries: list[tuple[str, int, int, int, int]] = []
    results = path / "results"
    try:
        run_dirs = sorted(entry for entry in results.iterdir()
                          if entry.is_dir() and entry.name.startswith("run_"))
    except OSError:
        run_dirs = []
    for run in run_dirs:
        sizes = []
        for name in ("matrix_full.csv", "matrix.csv", "state.json"):
            try:
                stat = (run / name).stat()
                sizes.append(stat.st_size)
            except OSError:
                sizes.append(0)
        try:
            mtime = run.stat().st_mtime_ns
        except OSError:
            mtime = 0
        entries.append((run.name, mtime, *sizes))
    inv_mtime = 0
    try:
        inv_mtime = inventory.path.stat().st_mtime_ns
    except OSError:
        pass
    return (str(path), tuple(entries), str(inventory.path), inv_mtime)


def _parse_tree(path: Path, inventory: ClusterInventory) -> FleetBenchmarks:
    fleet = FleetBenchmarks(root=path, inventory=inventory)
    results = path / "results"
    if not results.is_dir():
        fleet.error = (f"no profilebench results at {results} "
                       "(set benchmarks_dir)")
        fleet.warnings.append(fleet.error)
        return fleet
    samples: list[BenchSample] = []
    run_dirs = sorted(entry for entry in results.iterdir()
                      if entry.is_dir() and entry.name.startswith("run_"))
    if not run_dirs:
        fleet.error = f"no run_* directories under {results}"
        fleet.warnings.append(fleet.error)
        return fleet
    for run_dir in run_dirs:
        try:
            samples.extend(_parse_run(run_dir, inventory, fleet.warnings))
        except (OSError, ValueError, csv.Error) as exc:
            fleet.warnings.append(f"skipping {run_dir.name}: {exc}")
    if not samples:
        fleet.error = f"no usable benchmark rows under {results}"
        fleet.warnings.append(fleet.error)
        return fleet
    _aggregate(fleet, samples)
    _compute_factors(fleet)
    return fleet


def _parse_run(run_dir: Path, inventory: ClusterInventory,
               warnings: list[str]) -> list[BenchSample]:
    state: dict[str, Any] = {}
    try:
        raw_state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        if isinstance(raw_state, dict):
            state = raw_state
    except (OSError, ValueError):
        state = {}
    label_match = _RUN_DIR_RE.match(run_dir.name)
    label = label_match.group("label") if label_match else run_dir.name
    matrix_full = run_dir / "matrix_full.csv"
    matrix = run_dir / "matrix.csv"
    if matrix_full.is_file():
        return _parse_matrix_full(run_dir, matrix_full, state, inventory,
                                  label, warnings)
    if matrix.is_file():
        return _parse_matrix(run_dir, matrix, state, inventory, label, warnings)
    return []


def _run_hostnames(state: dict[str, Any], row: Optional[dict[str, Any]] = None) -> list[str]:
    names: list[str] = []
    if row:
        for key in ("Selected", "Hostname"):
            value = _text(row.get(key))
            if value:
                names.append(value)
    for key in ("hostname", "selected"):
        value = _text(state.get(key))
        if value:
            names.append(value)
    machine = _text(state.get("machine"))
    if machine and " " not in machine:
        names.append(machine)
    return list(dict.fromkeys(names))


def _run_fingerprint(state: dict[str, Any], row: Optional[dict[str, Any]] = None) -> str:
    if row:
        value = _text(row.get("Machine"))
        if value:
            return value
    return _text(state.get("machine"))


def _machine_for(run_dir: Path, state: dict[str, Any],
                 row: Optional[dict[str, Any]],
                 inventory: ClusterInventory, label: str,
                 warnings: list[str]) -> str:
    hostnames = _run_hostnames(state, row)
    machine = inventory.machine_for_run(
        hostnames=hostnames,
        fingerprint=_run_fingerprint(state, row),
        started=(row or {}).get("Started") or state.get("started"),
        label=label)
    if machine is not None:
        return machine.id
    # No inventory match: keep the run visible under its hardware fingerprint
    # (or its historical hostname) so the tool still works without a map.
    fallback = _run_fingerprint(state, row) or (hostnames[0] if hostnames else label)
    warnings.append(f"{run_dir.name}: no machine match for {fallback!r}")
    return fallback or "unknown"


def _best_cell(row: dict[str, Any], *, prefill: bool = False) -> tuple[Optional[float], str]:
    best_value: Optional[float] = None
    best_profile = ""
    for profile in PROFILES:
        state = _text(row.get(f"{profile}_state")).lower()
        if state and state not in VALID_STATES:
            continue
        suffix = "prefill_tps" if prefill else "tps"
        value = _num(row.get(f"{profile}_{suffix}"))
        if value is None and not prefill:
            # Legacy `matrix.csv` uses bare profile columns (`C`, `H1`, ...).
            value = _num(row.get(profile))
        if value is None or value <= 0:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_profile = profile
    return best_value, best_profile


def _sample(machine: str, model: str, digest: str, params: Optional[float],
            family: str, quant: str, tps: float, prefill: Optional[float],
            profile: str, prefill_profile: str, run: str, started: str,
            source: str) -> BenchSample:
    return BenchSample(
        machine=machine, model=model, digest=digest, params=params,
        family=family, quant=quant, tps=tps, prefill=prefill,
        profile=profile, prefill_profile=prefill_profile, run=run,
        started=started, source=source)


def _parse_matrix_full(run_dir: Path, path: Path, state: dict[str, Any],
                       inventory: ClusterInventory, label: str,
                       warnings: list[str]) -> list[BenchSample]:
    samples: list[BenchSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = _text(row.get("Model"))
            if not model:
                continue
            machine = _machine_for(run_dir, state, row, inventory, label, warnings)
            tps = _num(row.get("BestTps"))
            profile = _text(row.get("BestProfile"))
            if tps is None or tps <= 0:
                tps, profile = _best_cell(row)
            if tps is None or tps <= 0:
                continue
            prefill, prefill_profile = _best_cell(row, prefill=True)
            digest = _digest_key(row.get("Digest"))
            started = _text(row.get("Started")) or _text(state.get("started"))
            samples.append(_sample(
                machine, model, digest,
                parse_parameter_count(row.get("Params")),
                _text(row.get("Family")), _text(row.get("Quant")),
                tps, prefill, profile or "", prefill_profile or "",
                run_dir.name, started, "matrix_full"))
    return samples


def _parse_matrix(run_dir: Path, path: Path, state: dict[str, Any],
                  inventory: ClusterInventory, label: str,
                  warnings: list[str]) -> list[BenchSample]:
    machine = _machine_for(run_dir, state, None, inventory, label, warnings)
    started = _text(state.get("started"))
    samples: list[BenchSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = _text(row.get("Model"))
            if not model:
                continue
            tps, profile = _best_cell(row)
            if tps is None or tps <= 0:
                continue
            prefill, prefill_profile = _best_cell(row, prefill=True)
            samples.append(_sample(
                machine, model, "", None, "", "", tps, prefill,
                profile or "", prefill_profile or "",
                run_dir.name, started, "matrix"))
    return samples


def _aggregate(fleet: FleetBenchmarks, samples: list[BenchSample]) -> None:
    # Digest-bearing rows are the good data. A legacy `matrix.csv` row has no
    # digest, so it is folded into the digest bench with the same tag when
    # there is exactly one; otherwise it stays a name-only bench.
    grouped: dict[tuple[str, str], list[BenchSample]] = {}
    by_name: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        if not sample.digest:
            continue
        key = sample.digest
        grouped.setdefault((sample.machine, key), []).append(sample)
        by_name.setdefault(
            (sample.machine, canonical_model_name(sample.model).lower()),
            set()).add(key)
    for sample in samples:
        if sample.digest:
            continue
        name_key = (sample.machine, canonical_model_name(sample.model).lower())
        keys = by_name.get(name_key) or set()
        key = next(iter(keys)) if len(keys) == 1 else (
            f"name:{canonical_model_name(sample.model).lower()}")
        grouped.setdefault((sample.machine, key), []).append(sample)
    for (machine_id, key), items in grouped.items():
        if any(item.source == "matrix_full" for item in items):
            # A partial/older `matrix.csv` run must not drag down a cell the
            # full-profile run measured. Rows only the legacy run has stay.
            items = [item for item in items if item.source == "matrix_full"]
        machine = fleet.machines.setdefault(
            machine_id, MachineBench(machine=machine_id))
        tps_values = [item.tps for item in items if item.tps is not None]
        if not tps_values:
            continue
        prefills = [item.prefill for item in items if item.prefill is not None]
        profiles = [item.profile for item in items if item.profile]
        params = next((item.params for item in reversed(items)
                       if item.params is not None), None)
        family = next((item.family for item in reversed(items) if item.family), "")
        quant = next((item.quant for item in reversed(items) if item.quant), "")
        names = tuple(sorted({item.model for item in items}))
        digest = next((item.digest for item in items if item.digest), "")
        bench = ModelBench(
            machine=machine_id, key=key, digest=digest, names=names,
            params=params, family=family, quant=quant,
            tps=statistics.median(tps_values),
            tps_min=min(tps_values), tps_max=max(tps_values),
            prefill=statistics.median(prefills) if prefills else None,
            profile=statistics.mode(profiles) if profiles else "",
            runs=len({item.run for item in items}),
            last_run=max((item.started for item in items), default=""),
            samples=tuple(items))
        machine.models[key] = bench
        if digest:
            machine.by_digest[digest] = bench
        for name in names:
            machine.by_name[canonical_model_name(name).lower()] = bench
        machine.runs.update(item.run for item in items)
        machine.last_run = max((machine.last_run, bench.last_run), default=bench.last_run)


def _compute_factors(fleet: FleetBenchmarks) -> None:
    """Each machine's median speed ratio against the fleet, per metric."""
    for metric in ("decode", "prefill"):
        ratios: dict[str, list[float]] = {machine_id: []
                                          for machine_id in fleet.machines}
        keys = {bench.key for machine in fleet.machines.values()
                for bench in machine.models.values()}
        for key in keys:
            values: dict[str, float] = {}
            for machine_id, machine in fleet.machines.items():
                bench = machine.models.get(key)
                if bench is None:
                    continue
                value = bench.tps if metric == "decode" else bench.prefill
                if value:
                    values[machine_id] = float(value)
            if len(values) < 2:
                continue
            median = statistics.median(values.values())
            if median <= 0:
                continue
            for machine_id, value in values.items():
                ratios[machine_id].append(value / median)
        for machine_id, machine in fleet.machines.items():
            factor = statistics.median(ratios[machine_id]) if ratios[machine_id] else 1.0
            if metric == "decode":
                machine.decode_factor = factor
            else:
                machine.prefill_factor = factor


#: Process-lifetime cache for `performance_index()` callers that only have a
#: config (the picker/list commands), keyed by the tree generation.
_PERF_CACHE: dict[tuple[Any, ...], PerformanceIndex] = {}


def performance_index(config: Optional[dict[str, Any]] = None,
                      root: Optional[str] = None,
                      inventory: Optional[ClusterInventory] = None,
                      refresh: bool = False) -> PerformanceIndex:
    """The fleet-best model lookup for `--sort performance` (never raises)."""
    config = dict(config or {})
    path = benchmarks_root(config, root)
    if inventory is None:
        inventory = load_cluster_inventory(config)
    key = _tree_fingerprint(path, inventory)
    if not refresh and key in _PERF_CACHE:
        return _PERF_CACHE[key]
    fleet = load_benchmarks(config, root, inventory, refresh=refresh)
    index = fleet.performance_index()
    _PERF_CACHE[key] = index
    while len(_PERF_CACHE) > 8:
        _PERF_CACHE.pop(next(iter(_PERF_CACHE)))
    return index


def best_machine_rows(fleet: FleetBenchmarks) -> list[tuple[str, str]]:
    """(machine id, 'id (box)') for help/debug output."""
    return [(machine_id, machine.label() if machine else machine_id)
            for machine_id, machine in fleet.inventory.machines.items()]


def stamp(value: str) -> str:
    """Short display stamp for a run's started timestamp."""
    return short_stamp(value)[:10]


__all__ = [
    "BenchSample", "FleetBenchmarks", "FleetPerf", "MachineBench", "ModelBench",
    "PerformanceIndex", "benchmarks_root", "load_benchmarks",
    "parse_parameter_count", "performance_index", "stamp",
]
