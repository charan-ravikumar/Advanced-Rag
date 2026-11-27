"""The official Qasper TEST split as a held-out set: downloaded once, pinned by sha256, opened at most once.

    python -m eval.heldout fetch     # download once (public data), hash, store read-only, write the manifest; never re-downloads
    python -m eval.heldout verify    # check the stored files against the manifest (reads bytes only, parses nothing)
    python -m eval.heldout status    # secured? frozen? opened?
    python -m eval.heldout freeze --arms D --preregistration eval/PREREGISTRATION_<name>.md --frozen-by "<who and when>"
                                     # ONLY when the owner asks: pins data, arms, code and pre-registration for the one opening

This is NOT ``split.json``'s "test" split (a 60% slice of the official DEV papers, guarded by ``eval/test_freeze.json`` and already run once).

What is stored where
* The data lives OUTSIDE the repository, under ``%LOCALAPPDATA%/advrag/heldout/qasper_test_v0.3/`` (``~/.local/share/...`` off Windows), read-only.
  It is not part of the code fingerprint and cannot be committed by accident.
* ``eval/heldout_manifest.json`` (in the repository) pins the URL, sizes and sha256 of the archive and of every file in it.

How it is opened: ``open_heldout`` is the only function that returns the path of the test data for evaluation. It needs explicit confirmation, a
freeze file whose data hash, arms, code fingerprint and pre-registration all match, unchanged data, and no earlier opening. It records the opening
(the ledger in the repository AND a marker next to the data) BEFORE it returns, so a run that crashes has still used the set. ``fetch`` and
``verify`` hash bytes; they never parse a question, an answer or a paper.

What this cannot do: it is a procedural guard, not access control. Anyone with the machine can read the files, or delete the ledger and the
marker. Both are meant to be noticed (the ledger is committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = "qasper-test-v0.3"
URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"  # the URL in the official allenai/qasper loader
LICENSE = "CC BY 4.0 (Dasigi et al., 2021)"
ARCHIVE_NAME = "qasper-test-and-evaluator-v0.3.tgz"
TEST_FILE = "qasper-test-v0.3.json"
DEFAULT_DIR = "%LOCALAPPDATA%/advrag/heldout/qasper_test_v0.3"
OPENED_MARKER = "OPENED.json"
MANIFEST_FILE = ROOT / "eval" / "heldout_manifest.json"
FREEZE_FILE = ROOT / "eval" / "heldout_freeze.json"
LEDGER_FILE = ROOT / "eval" / "heldout_ledger.jsonl"


class HeldoutRefused(SystemExit):
    pass


def data_dir() -> Path:
    from db.store import resolve_store_path

    return resolve_store_path(DEFAULT_DIR)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _entry(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: Path, payload: dict) -> None:
    """Create ``path``; never overwrite (mode "x")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---- secure ------------------------------------------------------------------------------------------------------------------------

def _tls_context() -> ssl.SSLContext:
    """Certificate verification ON: the system's roots plus certifi's (pinned in the lock; some Python installs ship without system roots)."""
    import certifi

    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=120, context=_tls_context()) as response, open(target, "xb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)


def _extract(archive: Path, into: Path) -> list[Path]:
    """Every regular file of the archive, flattened into ``into``. Links, devices and two members with one name are refused. Nothing is executed."""
    out: list[Path] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            name = Path(member.name).name
            if not member.isfile() or name in ("", ".", "..", ARCHIVE_NAME, OPENED_MARKER) or (into / name).exists():
                raise HeldoutRefused(f"REFUSED: unexpected member in the archive: {member.name!r}")
            source = tar.extractfile(member)
            with open(into / name, "xb") as f:
                for block in iter(lambda: source.read(1 << 20), b""):
                    f.write(block)
            out.append(into / name)
    return out


def verify(*, directory: Path, manifest_path: Path = MANIFEST_FILE) -> dict:
    """The manifest, if every file it names is present in ``directory`` with the recorded size and sha256; otherwise ``HeldoutRefused``."""
    if not Path(manifest_path).exists():
        raise HeldoutRefused(f"REFUSED: no manifest at {manifest_path}: the set has not been secured (python -m eval.heldout fetch).")
    manifest = _read_json(manifest_path)
    for name, expected in {ARCHIVE_NAME: manifest["archive"], **manifest["files"]}.items():
        path = Path(directory) / name
        if not path.is_file():
            raise HeldoutRefused(f"REFUSED: {name} is missing from {directory}.")
        if _entry(path) != expected:
            raise HeldoutRefused(f"REFUSED: {name} does not match the manifest (size or sha256 changed).")
    return manifest


def fetch(*, directory: Path, manifest_path: Path = MANIFEST_FILE, download: Callable[[str, Path], None] = _download,
          now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> tuple[str, dict]:
    """Secure the set. ``("already secured", manifest)`` without any network access when the files are there and match. Without a manifest: download,
    hash, store, write the manifest. With a manifest but no files (another machine): download and accept ONLY an identical archive and files."""
    directory, manifest_path = Path(directory), Path(manifest_path)
    present = directory.exists() and any(directory.iterdir())
    if present:
        return "already secured", verify(directory=directory, manifest_path=manifest_path)  # never re-downloads or overwrites
    pinned = _read_json(manifest_path) if manifest_path.exists() else None
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        staging = Path(tmp) / "set"
        staging.mkdir()
        archive = staging / ARCHIVE_NAME
        download(URL, archive)
        if pinned is not None and _entry(archive) != pinned["archive"]:
            raise HeldoutRefused("REFUSED: the downloaded archive differs from the pinned manifest; nothing was stored.")
        files = {p.name: _entry(p) for p in sorted(_extract(archive, staging))}
        if TEST_FILE not in files:
            raise HeldoutRefused(f"REFUSED: the archive does not contain {TEST_FILE}.")
        if pinned is not None and files != pinned["files"]:
            raise HeldoutRefused("REFUSED: the extracted files differ from the pinned manifest; nothing was stored.")
        manifest = pinned or {"dataset": DATASET, "url": URL, "license": LICENSE, "downloaded_utc": now().isoformat(), "archive": _entry(archive), "files": files}
        for path in staging.iterdir():
            os.chmod(path, 0o444)
        if directory.exists():
            directory.rmdir()  # empty (checked above)
        staging.rename(directory)
    if pinned is None:
        _write_new(manifest_path, manifest)
    return "secured", manifest


# ---- freeze and open ---------------------------------------------------------------------------------------------------------------

def _pinned_facts(*, arms: Sequence[str], top_k: int, candidate_k: int, preregistration: Path, manifest: dict, fingerprint: str) -> dict:
    from eval.run_retrieval_eval import arm_config, select_arms

    preregistration = Path(preregistration)
    if not preregistration.is_file():
        raise HeldoutRefused(f"REFUSED: no pre-registration at {preregistration}: write it before the set is frozen.")
    config = arm_config(select_arms(",".join(arms)), top_k=top_k, candidate_k=candidate_k)
    return {"dataset": DATASET, "data_sha256": manifest["files"][TEST_FILE]["sha256"], **config, "code_fingerprint": fingerprint,
            "preregistration_sha256": hashlib.sha256(preregistration.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def _fingerprint() -> str:
    from utils.fingerprint import code_fingerprint

    return code_fingerprint()


def freeze(*, arms: Sequence[str], preregistration: Path, frozen_by: str, directory: Path, manifest_path: Path = MANIFEST_FILE,
           freeze_path: Path = FREEZE_FILE, top_k: int | None = None, candidate_k: int | None = None, fingerprint: str | None = None,
           now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """Write the freeze file (never overwrites one). Run only when the owner asks."""
    from eval.run_retrieval_eval import CANDIDATE_K, TOP_K

    if not frozen_by.strip():
        raise HeldoutRefused("REFUSED: --frozen-by must say who asked for the freeze.")
    if Path(freeze_path).exists():
        raise HeldoutRefused(f"REFUSED: {freeze_path} exists already; a freeze is never edited or recreated.")
    manifest = verify(directory=directory, manifest_path=manifest_path)
    facts = _pinned_facts(arms=arms, top_k=top_k or TOP_K, candidate_k=candidate_k or CANDIDATE_K, preregistration=preregistration, manifest=manifest,
                          fingerprint=fingerprint or _fingerprint())
    frozen = {**facts, "frozen_by": frozen_by, "created_utc": now().isoformat()}
    _write_new(Path(freeze_path), frozen)
    return frozen


def openings(*, directory: Path, ledger_path: Path = LEDGER_FILE) -> list[dict]:
    """Every recorded opening of this dataset: the ledger's lines and the marker next to the data. Either one means the set is used."""
    found = []
    if Path(ledger_path).exists():
        found += [e for e in (json.loads(line) for line in Path(ledger_path).read_text(encoding="utf-8").splitlines() if line.strip()) if e.get("dataset") == DATASET]
    marker = Path(directory) / OPENED_MARKER
    if marker.exists():
        found.append(_read_json(marker))
    return found


def open_heldout(*, confirmed: bool, arms: Sequence[str], top_k: int, candidate_k: int, preregistration: Path, purpose: str, directory: Path,
                 manifest_path: Path = MANIFEST_FILE, freeze_path: Path = FREEZE_FILE, ledger_path: Path = LEDGER_FILE, fingerprint: str | None = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> Path:
    """The path of the test JSON, ONCE. All checks first, then the opening is recorded, then the path is returned."""
    if not confirmed:
        raise HeldoutRefused("REFUSED: opening the held-out set is final. It needs the explicit confirmation (--i-understand-this-is-final).")
    if not purpose.strip():
        raise HeldoutRefused("REFUSED: say what this one opening is for.")
    if not Path(freeze_path).exists():
        raise HeldoutRefused(f"REFUSED: no frozen configuration at {freeze_path}. It is created only when the owner asks.")
    if openings(directory=directory, ledger_path=ledger_path):
        raise HeldoutRefused("REFUSED: the held-out set has been opened before. It is no longer held out and is not opened again.")
    manifest = verify(directory=directory, manifest_path=manifest_path)
    fingerprint = fingerprint or _fingerprint()
    facts = _pinned_facts(arms=arms, top_k=top_k, candidate_k=candidate_k, preregistration=preregistration, manifest=manifest, fingerprint=fingerprint)
    frozen = _read_json(freeze_path)
    for key, value in facts.items():
        if frozen.get(key) != value:
            raise HeldoutRefused(f"REFUSED: {key} differs from the frozen configuration.")
    entry = {"dataset": DATASET, "data_sha256": facts["data_sha256"], "code_fingerprint": fingerprint, "arms": sorted(facts["arms"]),
             "purpose": purpose, "utc": now().isoformat()}
    try:
        _write_new(Path(directory) / OPENED_MARKER, entry)  # exclusive create: of two concurrent openings exactly one gets past this line
    except FileExistsError:
        raise HeldoutRefused("REFUSED: the held-out set has been opened before. It is no longer held out and is not opened again.") from None
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return Path(directory) / TEST_FILE


def refuse_heldout_inputs(paths: Sequence[Path | str], *, directory: Path | None = None, manifest_path: Path = MANIFEST_FILE) -> None:
    """For the evaluation runners: refuse an input file that IS the held-out data (inside its folder, or a byte-identical copy elsewhere)."""
    directory = Path(directory) if directory is not None else data_dir()
    manifest = _read_json(manifest_path) if Path(manifest_path).exists() else None
    pinned = {} if manifest is None else {e["sha256"]: e["bytes"] for e in (manifest["archive"], *manifest["files"].values())}
    for raw in paths:
        path = Path(raw)
        if path.resolve().is_relative_to(directory.resolve()) or (path.is_file() and path.stat().st_size in pinned.values() and _sha256(path) in pinned):
            raise HeldoutRefused(f"REFUSED: {path} is the held-out official test set. It is opened only through eval.heldout.open_heldout.")


# ---- command line ------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify", "status"):
        sub.add_parser(name)
    fr = sub.add_parser("freeze", help="only when the owner asks")
    fr.add_argument("--arms", required=True, help="arm names, comma-separated")
    fr.add_argument("--preregistration", required=True)
    fr.add_argument("--frozen-by", required=True)
    args = ap.parse_args(argv)
    directory = data_dir()
    if args.command == "fetch":
        state, manifest = fetch(directory=directory)
    elif args.command == "freeze":
        freeze(arms=args.arms.split(","), preregistration=Path(args.preregistration), frozen_by=args.frozen_by, directory=directory)
        print(f"frozen: {FREEZE_FILE}")
        return 0
    elif args.command == "status" and not MANIFEST_FILE.exists():
        print("secured: no")
        return 0
    else:
        state, manifest = "verified", verify(directory=directory)
    print(f"{state}: {manifest['dataset']} in {directory}")
    print(f"  archive  {manifest['archive']['sha256']}  {manifest['archive']['bytes']} bytes")
    for name, entry in manifest["files"].items():
        print(f"  {name}  {entry['sha256']}  {entry['bytes']} bytes")
    if args.command == "status":
        print(f"frozen: {'yes' if FREEZE_FILE.exists() else 'no'}")
        print(f"opened: {len(openings(directory=directory)) > 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
