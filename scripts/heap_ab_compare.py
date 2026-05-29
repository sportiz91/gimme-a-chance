#!/usr/bin/env python3
"""Compare heap metrics across allocators from JSONL logs.

Reads logs/*.jsonl in the project root, filters `target=heap` entries, groups
them by `allocator` field, and prints a side-by-side report.

Usage:
    python3 scripts/heap_ab_compare.py                    # all .jsonl in logs/
    python3 scripts/heap_ab_compare.py logs/specific.jsonl
    python3 scripts/heap_ab_compare.py --skip-warmup 10   # drop first 10 samples
                                                           # per allocator

Designed to work with log entries produced by the `counting-alloc` feature.
Older logs (without `peak_live_bytes`, `live_buckets`, `allocator`) are
included in a separate "legacy" group so you can still see the totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


BUCKET_LABELS = [
    "≤16B", "≤32B", "≤64B", "≤128B", "≤256B", "≤512B", "≤1KB", "≤2KB",
    "≤4KB", "≤8KB", "≤16KB", "≤32KB", "≤64KB", "≤256KB", "≤1MB", ">1MB",
]


def fmt_bytes(n: float) -> str:
    """Human-readable bytes. Matches the frontend's fmtBytes roughly."""
    units = [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]
    for label, size in units:
        if n >= size:
            return f"{n / size:.2f} {label}"
    return f"{int(n)} B"


def fmt_delta(a: float, b: float) -> str:
    """Percentage delta from a → b. Returns e.g. '+12.3%' or '—' if a is 0."""
    if a == 0:
        return "—" if b == 0 else "(new)"
    pct = (b - a) / a * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def parse_bucket_field(raw: Any) -> list[int] | None:
    """Parse a bucket field that may come as a list, a string like '[1, 2, 3]',
    or None. Returns a list of ints or None if unparseable."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, str):
        # `?live_buckets` in tracing renders as Debug, e.g. "[0, 3, 12, ...]".
        stripped = raw.strip().lstrip("[").rstrip("]")
        if not stripped:
            return []
        return [int(x.strip()) for x in stripped.split(",") if x.strip()]
    return None


def load_heap_entries(paths: list[Path]) -> list[dict]:
    """Read JSONL files, return list of heap-target entries as dicts."""
    entries = []
    for path in paths:
        if not path.exists():
            print(f"  warning: {path} not found, skipping", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  warning: {path}:{line_no} bad json: {e}", file=sys.stderr)
                    continue
                if rec.get("target") != "heap":
                    continue
                # tracing nests structured fields under "fields"
                fields = rec.get("fields", {})
                merged = {
                    "timestamp": rec.get("timestamp"),
                    "message": fields.get("message"),
                    "live_bytes": fields.get("live_bytes"),
                    "peak_live_bytes": fields.get("peak_live_bytes"),
                    "total_allocated_bytes": fields.get("total_allocated_bytes"),
                    "total_deallocated_bytes": fields.get("total_deallocated_bytes"),
                    "allocator": fields.get("allocator"),
                    "live_buckets": parse_bucket_field(fields.get("live_buckets")),
                    "total_buckets": parse_bucket_field(fields.get("total_buckets")),
                    "source_file": path.name,
                }
                entries.append(merged)
    return entries


def group_by_allocator(entries: list[dict]) -> dict[str, list[dict]]:
    """Group entries by allocator name. Entries without an allocator field go
    to a 'legacy (pre-instrumentation)' bucket."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        key = e.get("allocator") or "legacy (pre-instrumentation)"
        groups[key].append(e)
    return groups


def summarize(group: list[dict], skip_warmup: int) -> dict:
    """Compute summary stats for one allocator group, dropping first N samples."""
    usable = group[skip_warmup:] if len(group) > skip_warmup else group
    if not usable:
        return {"samples": 0}

    live = [e["live_bytes"] for e in usable if e["live_bytes"] is not None]
    peak = [e["peak_live_bytes"] for e in usable if e["peak_live_bytes"] is not None]
    total_alloc = [e["total_allocated_bytes"] for e in usable if e["total_allocated_bytes"] is not None]
    total_dealloc = [e["total_deallocated_bytes"] for e in usable if e["total_deallocated_bytes"] is not None]

    # For bucket distributions, take the LAST sample — that's the steady state.
    last_live_buckets = None
    last_total_buckets = None
    for e in reversed(usable):
        if e["live_buckets"] and last_live_buckets is None:
            last_live_buckets = e["live_buckets"]
        if e["total_buckets"] and last_total_buckets is None:
            last_total_buckets = e["total_buckets"]
        if last_live_buckets is not None and last_total_buckets is not None:
            break

    return {
        "samples": len(usable),
        "live_mean": mean(live) if live else None,
        "live_median": median(live) if live else None,
        "live_max": max(live) if live else None,
        "peak_max": max(peak) if peak else None,
        "total_allocated_final": total_alloc[-1] if total_alloc else None,
        "total_deallocated_final": total_dealloc[-1] if total_dealloc else None,
        "live_buckets": last_live_buckets,
        "total_buckets": last_total_buckets,
        "source_files": sorted({e["source_file"] for e in usable}),
        "first_timestamp": usable[0]["timestamp"],
        "last_timestamp": usable[-1]["timestamp"],
    }


def print_scalar_row(label: str, stats_per_allocator: dict[str, dict], key: str,
                     formatter=str, baseline_key: str | None = None) -> None:
    """Print one row of the comparison table. First column is label, then one
    column per allocator. If baseline_key is given, show delta vs. that one."""
    cells = [f"{label:<30}"]
    baseline_val = None
    if baseline_key and baseline_key in stats_per_allocator:
        baseline_val = stats_per_allocator[baseline_key].get(key)
    for name, stats in stats_per_allocator.items():
        val = stats.get(key)
        cell = formatter(val) if val is not None else "—"
        if baseline_val is not None and val is not None and name != baseline_key:
            cell += f"  ({fmt_delta(baseline_val, val)})"
        cells.append(f"{cell:<28}")
    print("  ".join(cells))


def print_bucket_table(stats_per_allocator: dict[str, dict], key: str,
                       title: str) -> None:
    """Render the 16-bucket histogram side by side."""
    print(f"\n{title}")
    print(f"  {'bucket':<10}", end="")
    for name in stats_per_allocator:
        print(f"{name[:25]:<28}", end="")
    print()
    print(f"  {'-'*8:<10}", end="")
    for _ in stats_per_allocator:
        print(f"{'-'*26:<28}", end="")
    print()

    # For percentage calc per allocator
    totals: dict[str, int] = {}
    for name, stats in stats_per_allocator.items():
        buckets = stats.get(key)
        totals[name] = sum(buckets) if buckets else 0

    for i, label in enumerate(BUCKET_LABELS):
        print(f"  {label:<10}", end="")
        for name, stats in stats_per_allocator.items():
            buckets = stats.get(key)
            if buckets and i < len(buckets):
                count = buckets[i]
                pct = (count / totals[name] * 100) if totals[name] else 0
                cell = f"{count:>7}  ({pct:5.1f}%)"
            else:
                cell = "—"
            print(f"{cell:<28}", end="")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare heap metrics across allocators from JSONL logs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="JSONL log files to analyze. Default: all logs/*.jsonl",
    )
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=2,
        help="Drop the first N samples per allocator to skip startup allocs. "
        "Default: 2 (≈20s at 10s cadence)",
    )
    args = parser.parse_args()

    # Find the project root by walking up from this script.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    logs_dir = project_root / "logs"

    if args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        paths = sorted(logs_dir.glob("*.jsonl"))

    if not paths:
        print(f"No .jsonl logs found in {logs_dir}", file=sys.stderr)
        return 1

    print(f"Reading {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p}")
    print()

    entries = load_heap_entries(paths)
    if not entries:
        print("No heap entries found. Did you run with --features counting-alloc?",
              file=sys.stderr)
        return 1

    groups = group_by_allocator(entries)
    stats_per_allocator = {
        name: summarize(group, args.skip_warmup)
        for name, group in groups.items()
    }

    # Pick a baseline for delta computation: prefer "system" variants, else
    # whatever has the most samples (usually the richer one).
    baseline_key = None
    for candidate in stats_per_allocator:
        if "system" in candidate.lower() or "heapalloc" in candidate.lower():
            baseline_key = candidate
            break
    if baseline_key is None:
        baseline_key = max(stats_per_allocator, key=lambda k: stats_per_allocator[k]["samples"])

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 88)
    print(f"{'HEAP A/B COMPARISON':^88}")
    print("=" * 88)
    print(f"\nBaseline for delta (Δ) calculation: {baseline_key}")
    print(f"Skipping first {args.skip_warmup} samples per allocator (warmup).\n")

    # Labels (col headers)
    cells = [f"{'metric':<30}"]
    for name in stats_per_allocator:
        cells.append(f"{name[:25]:<28}")
    print("  ".join(cells))
    cells = [f"{'-'*28:<30}"]
    for _ in stats_per_allocator:
        cells.append(f"{'-'*26:<28}")
    print("  ".join(cells))

    # Session info
    print_scalar_row("samples", stats_per_allocator, "samples", str)
    print_scalar_row("first ts", stats_per_allocator, "first_timestamp",
                     lambda v: v[:19] if v else "—")
    print_scalar_row("last ts", stats_per_allocator, "last_timestamp",
                     lambda v: v[:19] if v else "—")
    print()

    # The meat
    print_scalar_row("peak live bytes", stats_per_allocator, "peak_max",
                     fmt_bytes, baseline_key)
    print_scalar_row("max live (samples)", stats_per_allocator, "live_max",
                     fmt_bytes, baseline_key)
    print_scalar_row("mean live", stats_per_allocator, "live_mean",
                     fmt_bytes, baseline_key)
    print_scalar_row("median live", stats_per_allocator, "live_median",
                     fmt_bytes, baseline_key)
    print_scalar_row("total bytes allocated", stats_per_allocator,
                     "total_allocated_final", fmt_bytes, baseline_key)
    print_scalar_row("total bytes freed", stats_per_allocator,
                     "total_deallocated_final", fmt_bytes, baseline_key)

    # Bucket histograms — only if anyone has them
    have_live = any(s.get("live_buckets") for s in stats_per_allocator.values())
    have_total = any(s.get("total_buckets") for s in stats_per_allocator.values())
    if have_live:
        print_bucket_table(stats_per_allocator, "live_buckets",
                           "Live allocations at end of session (count, % of live):")
    if have_total:
        print_bucket_table(stats_per_allocator, "total_buckets",
                           "Total allocations ever (count, % of total):")

    # ── Interpretation hints ──────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("INTERPRETATION HINTS")
    print("=" * 88)
    hints = [
        "• If peak live bytes differs >10% between allocators, the smaller one "
        "is handling your workload better.",
        "• If live_buckets is dominated by small sizes (≤64B, ≤128B), a pool "
        "allocator for those sizes might help more than switching allocator.",
        "• If total allocs differs >30% between runs, the workloads were not "
        "actually identical — redo the comparison.",
        "• If total_allocated ≈ total_deallocated, there's no leak. If "
        "total_allocated grows without bound and deallocated lags, suspect a leak.",
        "• HeapAlloc on Windows 10+ uses LFH automatically for small sizes — "
        "mimalloc's gains are typically marginal for allocation-light workloads.",
    ]
    for h in hints:
        print(f"\n{h}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
