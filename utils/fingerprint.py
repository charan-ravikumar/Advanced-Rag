"""Code fingerprint: a sha256 over all .py and .yaml files, used in place of a git SHA.

Files are sorted by their POSIX-style relative path. For each file the hash
absorbs the relative path, a NUL separator, the content (CRLF normalised to LF
so Windows/Linux checkouts agree) and another NUL. Virtualenvs, caches, vector
stores and __pycache__ directories are excluded.

CLI: python -m utils.fingerprint [root]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXTENSIONS = frozenset({".py", ".yaml"})
EXCLUDED_DIRS = frozenset(
    {
        ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", ".cache", ".chroma", "node_modules",
        "site-packages", "build", "dist",
    }
)


def _iter_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        rel_parts = path.relative_to(root).parts
        if any(p in EXCLUDED_DIRS or p.endswith(".egg-info") for p in rel_parts[:-1]):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def code_fingerprint(root: str | Path | None = None) -> str:
    """Return the hex sha256 fingerprint of all .py/.yaml files under ``root``.

    ``root`` defaults to the repository root (the parent of ``utils/``).
    """
    root_path = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    root_path = root_path.resolve()
    digest = hashlib.sha256()
    for path in _iter_files(root_path):
        rel = path.relative_to(root_path).as_posix()
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(code_fingerprint(sys.argv[1] if len(sys.argv) > 1 else None))
