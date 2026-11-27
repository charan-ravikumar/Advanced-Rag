#!/usr/bin/env bash
# Smoke test of the compose stack: index a few papers, start API and UI, require GET /v1/ready to say ready.
# Needs the image advrag:local and the pinned embedding model under HF_CACHE_DIR. Contacts no external service (no keys).
# It uses its OWN compose project (and volume), so a developer's index (project "advrag") is never touched.
set -euo pipefail
cd "$(dirname "$0")/.."

export API_PORT="${SMOKE_API_PORT:-18000}" UI_PORT="${SMOKE_UI_PORT:-18501}"
compose() { docker compose -p advrag-smoke --env-file /dev/null "$@"; }
cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then compose logs --no-color --tail 100 || true; fi
  compose --profile tools down -v >/dev/null 2>&1 || true   # removes only the advrag-smoke volume created here
  exit "$status"
}
trap cleanup EXIT

compose run --rm --no-deps ingest python scripts/ingest.py --strategy recursive --limit-docs 5
compose up -d --no-build --wait --wait-timeout 300

ready=$(curl -fsS "http://127.0.0.1:${API_PORT}/v1/ready")
echo "$ready"
echo "$ready" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["ready"] is True else "the API is not ready")'
curl -fsS -X POST "http://127.0.0.1:${API_PORT}/v1/retrieve" -H 'Content-Type: application/json' -d '{"question":"What datasets are used?"}' |
  python3 -c 'import json,sys; n=len(json.load(sys.stdin)["sources"]); print("retrieved", n); sys.exit(0 if n > 0 else "retrieval returned nothing")'
curl -fsS "http://127.0.0.1:${UI_PORT}/_stcore/health"; echo
echo "smoke test passed"
