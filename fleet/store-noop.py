#!/usr/bin/env python3
"""store-noop.py — stands in for store-sync.py when langywrap owns the node-local store.

Source of truth: slurmhelps fleet/store-noop.py. supervisor.sbatch selects it when the
project's .langywrap/lanes.json has a top-level ``store.node_local`` block (langywrap's
built-in node-local store: the daemon itself stages the databases in, snapshots them
every ``snapshot_interval_s`` and once on stop, by default into
``<project>/.langywrap/node-local-persist``). Running store-sync.py as well would keep a
second persisted copy and, at stage-in, could restore an older copy that langywrap would
then keep ("an existing local file wins"). Override: fleet.session_store.engine in
lanes.json ("store-sync" forces the old path).
"""
import sys
import time

cmd = sys.argv[1] if len(sys.argv) > 1 else "?"
print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} store-sync: {cmd} skipped: "
      f"the session store is langywrap's (lanes.json store.node_local)", flush=True)
sys.exit(0)
