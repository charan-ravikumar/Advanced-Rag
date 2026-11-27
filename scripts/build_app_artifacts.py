"""Build the data the app displays: app_data/results.json and app_data/papers.json (see eval/app_results.py for what is read and how it is validated).

    python scripts/build_app_artifacts.py            # write both files
    python scripts/build_app_artifacts.py --check    # exit 1 if the committed files are stale (differ from a fresh build, ignoring the build time)

Reads saved run folders and docs/CLAIMS_LOG.md only. Runs no retrieval and no LLM; sends nothing anywhere. Refuses to build if any number lacks a source run folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail if the committed files differ from a fresh build")
    args = ap.parse_args(argv)
    from eval.app_results import build, strip_volatile, write

    results, papers = build(root)
    if args.check:
        stale = []
        for name, fresh in (("results.json", strip_volatile(results)), ("papers.json", papers)):
            path = Path(root) / "app_data" / name
            if not path.exists():
                stale.append(f"{name} is missing")
                continue
            have = json.loads(path.read_text(encoding="utf-8"))
            have = strip_volatile(have) if name == "results.json" else have
            if have != json.loads(json.dumps(fresh, ensure_ascii=False)):
                stale.append(f"{name} is stale")
        print("STALE: " + "; ".join(stale) if stale else "app_data is up to date")
        return 1 if stale else 0
    r_path, p_path = write(root, results, papers)
    print(f"wrote {r_path} and {p_path}")
    print(f"sections: {[k for k in results if k not in ('schema', 'built_utc', 'sources')]}; papers: {len(papers['papers'])}; examples: {len(papers['examples']['items'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
