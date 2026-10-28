#!/bin/bash
# Time DeepSeek's decode with the dev engine's DS_SWEEP. Settings: tools/env.sh.
#   tools/sweep.sh [KEYS]      KEYS: DS_SWEEP's (default 1: every pipeline, "!", "nobar", "serial")
# Needs ~210 GiB: SERVER_STOP frees the memory first and SERVER_START restores it afterwards.
set -euo pipefail
KEYS="${1:-1}"
. "$(dirname "$0")/env.sh"
BIN="${BIN:-${DEV:?set DEV or BIN}}"
OUT="$(mktemp -d)"
wait_idle
trap 'pkill -f tf-dsv41-forward || true; server_start; unload_window "end sweep"; rm -rf "$OUT"' EXIT
unload_window "start sweep $KEYS"
server_stop
DS_SWEEP="$KEYS" "$BIN" "$MODEL" "$IDS" "$OUT" 2>&1 | grep -E '^sweep|^kbench|^lat|^replay|^rx|error|^\||^===|rows\[|ds_'
