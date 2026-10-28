#!/bin/bash
# Record a Metal System Trace of the dev engine's decode sweep. Settings: tools/env.sh.
#   tools/trace.sh OUT.trace [KEYS]
set -euo pipefail
OUT="$1"
KEYS="${2:-nobar}"
. "$(dirname "$0")/env.sh"
BIN="${BIN:-${DEV:?set DEV or BIN}}"
TMP="$(mktemp -d)"
wait_idle
trap 'pkill -f tf-dsv41-forward || true; server_start; unload_window "end trace"; rm -rf "$TMP"' EXIT
unload_window "start trace $KEYS"
server_stop
rm -rf "$OUT"
timeout -k 10 300 xcrun xctrace record --template 'Metal System Trace' --output "$OUT" --time-limit 120s \
	--env DS_SWEEP="$KEYS" --launch -- "$BIN" "$MODEL" "$IDS" "$TMP" 2>&1 | tail -3
