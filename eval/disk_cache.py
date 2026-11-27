"""A small on-disk key-value cache shared by the generation runner and the RAGAS judge: one JSON file per key, written atomically. A corrupt or unreadable file
is a miss, never an error. Pure stdlib."""

from __future__ import annotations

import os
from pathlib import Path


class DiskCache:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        import json

        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: dict) -> None:
        import json

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.rglob("*.json")) if self.dir.exists() else 0
