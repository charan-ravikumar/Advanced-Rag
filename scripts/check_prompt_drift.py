"""Drift check: are the production prompts in Langfuse the same as the templates in code? (Not part of the hermetic test suite: it needs the network.)

    python scripts/check_prompt_drift.py

Exit code 0: both identical. 1: a difference, a missing prompt or a failed fetch (the output says which). 2: no Langfuse keys (nothing contacted).
A difference is not necessarily a bug (someone may have improved the prompt in the Langfuse UI): it means the code, the tests and the evaluation
no longer describe what production serves. Reconcile by copying the Langfuse text back into rag_core/prompts.py (and bumping PROMPT_VERSION)
or by creating a new version from the code with scripts/seed_prompts.py --apply --update-differing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from rag_core.env import load_env
    from rag_core.langfuse_client import get_langfuse_client
    from rag_core.prompt_sync import compare_all

    load_env()
    client = get_langfuse_client()
    if client is None:
        print("SKIPPED: Langfuse keys are not available (environment or .env); nothing was contacted.")
        return 2
    results = compare_all(client)
    for r in results:
        print(f"  {r.name:<18} {r.state:<10} local={r.local_sha256[:12]} remote={(r.remote_sha256 or '-')[:12]} remote_version={r.remote_version}"
              + (f"  ({r.detail})" if r.detail else ""))
    drift = [r.name for r in results if r.state != "identical"]
    print("NO DRIFT" if not drift else f"DRIFT in: {', '.join(drift)}")
    return 0 if not drift else 1


if __name__ == "__main__":
    sys.exit(main())
