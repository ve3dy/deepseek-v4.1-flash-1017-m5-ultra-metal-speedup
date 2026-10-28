# Sourced by gate.sh, sweep.sh and trace.sh. Settings come from the environment or from tools/local.env (not in git):
#   MODEL         the checkpoint directory (required)
#   DEV, REF      tf-dsv41-forward binaries: the engine under test and the reference (gate.sh needs both)
#   WORK          inputs, reference outputs and the unload log (default ~/dsv41-work)
#   IDS, IDS_LONG token-id inputs (default $WORK/prose256.npy, $WORK/long600.npy; tools/make_ids.py makes the first)
#   SERVER_STOP   command that frees the GPU memory first, e.g. stops your model server (default: none)
#   SERVER_START  command that brings it back afterwards (default: none)
# local.env may also define busy_lines(): print anything while a client is mid-request, and wait_idle waits until it
# prints nothing (up to IDLE_WAIT seconds, default 600, then exits 3), so a run never cuts off a live request.
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$ENV_DIR/local.env" ]] && . "$ENV_DIR/local.env"
: "${MODEL:?set MODEL to the checkpoint directory (environment or tools/local.env)}"
WORK="${WORK:-$HOME/dsv41-work}"
IDS="${IDS:-$WORK/prose256.npy}"
IDS_LONG="${IDS_LONG:-$WORK/long600.npy}"
declare -F busy_lines >/dev/null || busy_lines() { :; }
server_stop() { if [[ -n "${SERVER_STOP:-}" ]]; then eval "$SERVER_STOP"; sleep 5; fi; }
server_start() { if [[ -n "${SERVER_START:-}" ]]; then eval "$SERVER_START"; fi; }
wait_idle() {
	local waited=0 busy
	while :; do
		busy=$(busy_lines)
		[[ -z "$busy" ]] && return 0
		if ((waited >= ${IDLE_WAIT:-600})); then
			echo "still busy after ${IDLE_WAIT:-600} s; not unloading. Last activity:" >&2
			echo "$busy" | tail -3 >&2
			exit 3
		fi
		((waited == 0)) && echo "a client is mid-request; waiting until it is idle" >&2
		sleep 10
		waited=$((waited + 10))
	done
}
unload_window() {
	mkdir -p "$WORK"
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >>"$WORK/unload-windows.log"
}
