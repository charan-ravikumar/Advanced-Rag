"""Seed the two answer prompts into Langfuse Prompt Management from the templates in rag_core/prompts.py.

    python scripts/seed_prompts.py                        # DRY RUN: reports missing / identical / differs, changes nothing
    python scripts/seed_prompts.py --apply                # creates ONLY the prompts that are missing (type chat, label production)
    python scripts/seed_prompts.py --apply --update-differing   # also creates a NEW version where Langfuse differs from the code

Idempotent by content hash (a second run creates nothing). Never edits or deletes an existing version. What is sent to Langfuse: the prompt
templates, which contain no data and no secrets. Keys come from the environment or .env (allowlisted names; values never printed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="create the missing prompts (default: dry run)")
    parser.add_argument("--update-differing", action="store_true", help="with --apply: also create a new version where Langfuse differs from the code")
    args = parser.parse_args(argv)
    if args.update_differing and not args.apply:
        parser.error("--update-differing only makes sense with --apply")

    from rag_core.env import load_env
    from rag_core.langfuse_client import get_langfuse_client
    from rag_core.prompt_sync import seed

    load_env()
    client = get_langfuse_client()
    if client is None:
        print("SKIPPED: Langfuse keys are not available (environment or .env); nothing was contacted.")
        return 0
    results = seed(client, apply=args.apply, update_differing=args.update_differing)
    print("MODE:", "APPLY" if args.apply else "DRY RUN (nothing is changed)")
    for r in results:
        print(f"  {r.name:<18} state={r.state:<10} local={r.local_sha256[:12]} remote={(r.remote_sha256 or '-')[:12]} "
              f"remote_version={r.remote_version} action={r.action or '-'}" + (f"  ({r.detail})" if r.detail else ""))
    if args.apply and any(r.action == "created" for r in results):
        client.flush()
    needs = [r.name for r in results if r.state in {"missing", "differs"} and r.action != "created"]
    if needs and not args.apply:
        print("Re-run with --apply to create what is missing.")
    return 1 if any(r.state in {"error", "unreadable"} for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
