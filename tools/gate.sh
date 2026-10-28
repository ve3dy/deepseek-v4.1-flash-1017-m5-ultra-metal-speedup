#!/bin/bash
# Exactness and speed gate for a dev engine against the reference engine, on a DeepSeek-V4.1 checkpoint:
#   1. a decode row's logits, byte for byte (DS_SWEEP's DS_REF), and its time;
#   2. 128 greedy tokens, plain and DSpark-drafted, against the reference's (DS_DRAFT);
#   3. four streams' shared windows against each alone, logits and drafts (DS_SHARED).
# The reference's outputs are made once into $REFDIR. Settings: tools/env.sh.
#   tools/gate.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"
: "${DEV:?set DEV}" "${REF:?set REF}"
REFDIR="${REFDIR:-$WORK/ref}"
TMP="$(mktemp -d)"
wait_idle
trap 'pkill -f tf-dsv41-forward || true; server_start; unload_window "end gate"; rm -rf "$TMP"' EXIT
mkdir -p "$REFDIR"
unload_window "start gate"
server_stop
run() { timeout -k 10 600 "$@" 2>&1; }
if [[ ! -f "$REFDIR/row.bin" || ! -f "$REFDIR/greedy.bin" ]]; then
	echo "== reference"
	DS_SWEEP=- DS_REF="$REFDIR/row.bin" run "$REF" "$MODEL" "$IDS" "$TMP" | grep -E '^sweep'
	DS_DRAFT=128 run "$REF" "$MODEL" "$IDS" "$TMP" | grep -E 'plain|error'
	cp "$TMP/greedy.bin" "$REFDIR/greedy.bin"
fi
echo "== dev"
fail=0
out=$(DS_SWEEP=- DS_REF="$REFDIR/row.bin" run "$DEV" "$MODEL" "$IDS" "$TMP" | grep -E '^sweep' || true)
echo "$out"
grep -q 'reference .*: logits equal' <<<"$out" || { echo "GATE: decode row logits missing or differ"; fail=1; }
rm -f "$TMP/greedy.bin"
DS_DRAFT=128 run "$DEV" "$MODEL" "$IDS" "$TMP" | grep -E 'plain|error' || true
if [[ -f "$TMP/greedy.bin" ]] && cmp -s "$TMP/greedy.bin" "$REFDIR/greedy.bin"; then echo "greedy 128: equal to the reference's"; else echo "GATE: greedy 128 missing or differs"; fail=1; fi
# streams of up to 600 tokens (the shared check takes prompts 250 shorter than the input)
out=$(DS_SHARED=8 run "$DEV" "$MODEL" "$IDS_LONG" "$TMP" | grep -E 'shared|error' || true)
echo "$out"
grep -qE 'logits equal for ([0-9]+)/\1 rows, drafts equal ([0-9]+)/\2' <<<"$out" || { echo "GATE: shared streams missing or differ"; fail=1; }
((fail == 0)) && echo "GATE: pass" || { echo "GATE: FAIL"; exit 1; }
