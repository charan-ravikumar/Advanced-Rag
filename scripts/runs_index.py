"""Build the run registry: runs/index.jsonl, one JSON row per run folder.

    python scripts/runs_index.py            # write runs/index.jsonl
    python scripts/runs_index.py --check    # exit 1 if the committed file differs from a fresh build

A run folder is runs/<kind>/<run id>/ with at least one file (the generation caches are not runs). Every field comes from the files in the
run folders: no clock, no git, no machine details, so building twice gives the same bytes. A value a run did not record is null.
Reads saved files only (standard library); runs nothing and sends nothing.

Both modes fail if README.md or docs/CLAIMS_LOG.md name a run folder that is not in the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "runs/index.jsonl"
NOT_RUNS = frozenset({"gen/cache", "gen/judge_cache"})
METADATA_FILES = ("config.json", "manifest.json", "results.json", "parity.json", "dry_run.json", "categories.json")  # first present wins, per field
CLAIM_DOCS = ("README.md", "docs/CLAIMS_LOG.md")
RUN_REFERENCE = re.compile(r"runs/[a-z_]+/(?:[a-z_]+_)?\d{8}[A-Za-z0-9_]*")


def _files(folder: Path) -> list[Path]:
    found = (p for p in folder.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts))
    return sorted(found, key=lambda p: p.relative_to(folder).as_posix())


def run_folders(root: Path) -> list[Path]:
    runs = root / "runs"
    out = []
    for kind in sorted(runs.iterdir() if runs.is_dir() else [], key=lambda p: p.name):
        if not kind.is_dir() or kind.name.startswith("."):
            continue
        for folder in sorted(kind.iterdir(), key=lambda p: p.name):
            if folder.is_dir() and not folder.name.startswith(".") and f"{kind.name}/{folder.name}" not in NOT_RUNS and _files(folder):
                out.append(folder)
    return out


def _load(path: Path) -> dict:
    """The JSON object in ``path``, or {} if there is no such file or it holds something else. Broken JSON raises."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path.as_posix()} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _content_sha256(folder: Path) -> str:
    """One hash over every file of the run (names and bytes): any edit to a saved number changes the row."""
    digest = hashlib.sha256()
    for path in _files(folder):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _timestamp(meta: list[dict], name: str) -> str | None:
    for doc in meta:
        for key in ("created_utc", "started_utc"):
            if doc.get(key):
                return str(doc[key])
    found = re.search(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2}))?", name)  # only what the folder name says
    if not found:
        return None
    y, mo, d, h, mi, s = found.groups()
    return f"{y}-{mo}-{d}" + (f"T{h}:{mi}:{s}+00:00" if h else "")


def _split(root: Path, meta: list[dict]) -> str | None:
    """The run's own split, else the split of the run it was derived from (``source_run`` / ``generation_run``)."""
    for doc in meta:
        if isinstance(doc.get("split"), str):
            return doc["split"]
    for doc in meta:
        source = doc.get("source_run") or doc.get("generation_run")
        source = source.get("path") if isinstance(source, dict) else source
        if isinstance(source, str):
            parts = source.replace("\\", "/").rstrip("/").split("/")
            if "runs" in parts:  # recorded paths may be absolute or use Windows separators
                split = _load(root.joinpath(*parts[len(parts) - 1 - parts[::-1].index("runs"):], "config.json")).get("split")
                if isinstance(split, str):
                    return split
    return None


def _numbers(value, prefix: str = "") -> dict:
    """Every number and boolean under ``value``, by dotted path (lists are per-item detail and are left in the run folder)."""
    if isinstance(value, dict):
        out: dict = {}
        for key, child in value.items():
            out.update(_numbers(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, (int, float)) and math.isfinite(value):
        return {prefix: value}
    return {}


def key_metrics(folder: Path) -> dict:
    summary, manifest = _load(folder / "summary.json"), _load(folder / "manifest.json")
    if "arms" in summary:  # retrieval evaluation
        out = {}
        for arm, data in summary["arms"].items():
            out[f"{arm}.n_questions"] = data.get("n_questions")
            out.update({f"{arm}.{m}": v for m, v in data.get("means", {}).items() if not m.startswith("old_")})  # old_: the superseded relevance rule
            for stage in ("total", "rerank"):
                for q in ("p50", "p95"):
                    value = ((data.get("retrieval_latency_ms") or {}).get(stage) or {}).get(q)
                    if value is not None:
                        out[f"{arm}.latency_ms.{stage}.{q}"] = value
        return out
    if summary:  # generation evaluation and judge scoring
        wanted = ("short_mode", "long_mode", "metrics", "n_scored_rows")
        return _numbers({k: summary[k] for k in wanted if k in summary})
    if "ingest_signature" in manifest:
        return _numbers({k: manifest[k] for k in ("docs", "chunks", "verification") if k in manifest})
    if "counts" in manifest:  # export
        return _numbers({"counts": manifest["counts"]})
    skip = {"preregistration", "generator", "sample"}  # settings, not results
    for name in ("results.json", "parity.json", "dry_run.json", "categories.json"):
        data = _load(folder / name)
        if data:
            return _numbers({k: v for k, v in data.items() if k not in skip})
    return {}


def row(root: Path, folder: Path) -> dict:
    meta = [doc for doc in (_load(folder / name) for name in METADATA_FILES) if doc]

    def first(key: str):
        return next((doc[key] for doc in meta if doc.get(key) is not None), None)

    config_hash, source = None, None
    if (folder / "config.json").is_file():
        config_hash, source = hashlib.sha256((folder / "config.json").read_bytes()).hexdigest(), "config.json"
    elif first("ingest_signature"):
        config_hash, source = first("ingest_signature"), "ingest_signature"
    return {
        "run_id": folder.name,
        "kind": folder.parent.name,
        "path": folder.relative_to(root).as_posix(),
        "timestamp": _timestamp(meta, folder.name),
        "code_fingerprint": first("code_fingerprint"),
        "config_hash": config_hash,
        "config_hash_source": source,
        "split": _split(root, meta),
        "eval_mode": first("eval_mode"),
        "label": first("label"),
        "complete": first("complete"),
        "metrics": key_metrics(folder),
        "files": [p.relative_to(folder).as_posix() for p in _files(folder)],
        "content_sha256": _content_sha256(folder),
    }


def build(root: Path = ROOT) -> bytes:
    rows = [row(root, folder) for folder in run_folders(root)]
    ids = [r["run_id"] for r in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"run ids must be unique: {', '.join(duplicates)}")
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode("utf-8")


def unknown_references(root: Path, index: bytes) -> list[str]:
    known = {json.loads(line)["path"] for line in index.decode("utf-8").split("\n") if line}
    missing = []
    for doc in CLAIM_DOCS:
        path = root / doc
        if path.is_file():
            missing += [f"{doc} names {ref}" for ref in sorted(set(RUN_REFERENCE.findall(path.read_text(encoding="utf-8")))) if ref not in known]
    return missing


def differences(committed: bytes, fresh: bytes) -> list[str]:
    def by_path(data: bytes) -> dict:
        out = {}
        for line in data.decode("utf-8", "replace").split("\n"):
            try:
                out[json.loads(line)["path"]] = line
            except (ValueError, KeyError, TypeError):
                continue
        return out

    old, new = by_path(committed), by_path(fresh)
    found = [f"not in the index: {p}" for p in sorted(new.keys() - old.keys())]
    found += [f"in the index but not on disk: {p}" for p in sorted(old.keys() - new.keys())]
    found += [f"changed: {p}" for p in sorted(old.keys() & new.keys()) if old[p] != new[p]]
    return found or ["same rows, different bytes (order, spacing, line endings or a hand edit)"]


def main(argv=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail if the committed index is out of date")
    args = ap.parse_args(argv)
    fresh = build(root)
    missing = unknown_references(root, fresh)
    for line in missing:
        print(f"UNKNOWN RUN: {line}, which has no run folder")
    path = root / INDEX
    if args.check:
        committed = path.read_bytes() if path.is_file() else b""
        if committed != fresh:
            print(f"STALE: {INDEX} is out of date. Run `python scripts/runs_index.py` and commit the result.")
            for line in differences(committed, fresh):
                print(f"  {line}")
            return 1
        if not missing:
            print(f"{INDEX} is up to date")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fresh)
        print(f"wrote {INDEX}: {fresh.count(10)} runs")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
