#!/usr/bin/env bash
# run_trader.sh — called by cron every 6 hours
# Logs to logs/paper_trader.log with timestamps

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/logs/paper_trader.log"
MAX_LOG_BYTES=5242880  # 5 MB — rotate when exceeded

# Rotate log if too large
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
    mv "$LOG" "${LOG}.1"
fi

echo "" >> "$LOG"
echo "======================================" >> "$LOG"
echo "RUN: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
echo "======================================" >> "$LOG"

cd "$DIR"
# Use the prediction-market-analysis venv (has all required dependencies)
PYTHON="/Users/samgrumet/Dropbox/Sam Grumet's Files/VSCODE/prediction-market-analysis/.venv/bin/python3"
"$PYTHON" paper_trader.py --forward --live >> "$LOG" 2>&1
EXIT_CODE=$?

echo "EXIT: $EXIT_CODE  at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
exit $EXIT_CODE
