#!/usr/bin/env bash
# run_sweep.sh — launch sweep_features.py in the background; survives SSH logout.
#
# Usage:
#   ./run_sweep.sh                                       # default: change_in_weight
#   ./run_sweep.sh --edges-col change_in_adjusted_weight
#   ./run_sweep.sh --quarters 2024Q1,2024Q2,2024Q3,2024Q4
#   ./run_sweep.sh --data-dir ~/13Fgnn/data --epochs 80
#
# After launching, you can:
#   - close the terminal / log out / close your laptop
#   - reconnect later and `tail -f logs/sweep_*.log` to monitor
#   - `kill <PID>` (PID is printed below) to stop early

set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/sweep_${TS}.log"
PID_FILE="logs/sweep_${TS}.pid"

# nohup: ignore HUP signal so process survives terminal close
# &: run in background
# disown: detach from shell job table so closing the shell doesn't kill it
nohup python -u sweep_features.py "$@" > "$LOG" 2>&1 &
PID=$!
disown
echo "$PID" > "$PID_FILE"

echo "started sweep_features.py"
echo "  args:        $*"
echo "  PID:         $PID"
echo "  log file:    $LOG"
echo "  PID file:    $PID_FILE"
echo
echo "monitor:       tail -f $LOG"
echo "stop:          kill \$(cat $PID_FILE)"
echo "list jobs:     ps -p $PID -o pid,etime,cmd"
echo
echo "you can close this terminal / SSH session / laptop now."
