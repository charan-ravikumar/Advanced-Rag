"""CI ONLY: download the pinned embedding model (config.yaml, public) into HF_HOME, for the Docker smoke test.

Refuses to run unless CI=true: on a developer machine models are never downloaded without the owner's approval (CLAUDE.md rule 11).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    if os.environ.get("CI") != "true":
        print("REFUSED: this script downloads a model and runs only in CI (CI=true).", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    from config import cfg

    path = snapshot_download(cfg.embedding.model, revision=cfg.embedding.revision,
                             allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"])
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
