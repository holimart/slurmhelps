#!/usr/bin/env bash
# Label each running lane session with its lane id. Source of truth: slurmhelps fleet/label-lanes.sh.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lane-common.sh
. "$HERE/lane-common.sh"
PROJECT="${PROJECT:-$(alloc_value PROJECT)}"
PROJECT="${PROJECT:-${AGENT_PROJECT:?no PROJECT in alloc.env and AGENT_PROJECT unset}}"
cd "$PROJECT" || exit 1
lane_env_setup || exit 1
langywrap steward lane-map --project . --json 2>/dev/null | python3 -c '
import sys,json
raw=sys.stdin.read(); i=raw.find("[")
rows=json.loads(raw[i:]) if i>=0 else []
for r in rows:
    if r.get("state") in ("running","waiting") and r.get("session"):
        print(r["lane"], r["session"])
' | while read -r lane sid; do
  langywrap session label "$sid" "$lane" --path . >/dev/null 2>&1 && echo "labelled $lane"
done
