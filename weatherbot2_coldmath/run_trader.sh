#!/usr/bin/env bash
# run_trader.sh — called by cron every 15 minutes
# Logs to logs/paper_trader.log with timestamps

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$DIR/logs/paper_trader.log"
MAX_LOG_BYTES=5242880  # 5 MB — rotate when exceeded

mkdir -p "$DIR/logs"

# Rotate log if too large
if [ -f "$LOG" ] && [ "$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
    mv "$LOG" "${LOG}.1"
fi

echo "" >> "$LOG"
echo "======================================" >> "$LOG"
echo "RUN: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
echo "======================================" >> "$LOG"

# ── VPN guard ──────────────────────────────────────────────────────────────
# ProtonVPN (and other macOS VPNs) create a utun interface when connected.
# Skip the run if no utun interface has an assigned IPv4 address — this
# prevents hitting Polymarket from a US IP.
VPN_UP=$(ifconfig 2>/dev/null | awk '/^utun/{iface=$1} iface && /inet /{print iface; iface=""}' | head -1)
if [ -z "$VPN_UP" ]; then
    echo "SKIP: VPN not connected (no active utun interface). Connect ProtonVPN and retry." >> "$LOG"
    exit 0
fi
echo "VPN: active on $VPN_UP" >> "$LOG"
# ───────────────────────────────────────────────────────────────────────────

cd "$DIR"
# Use the prediction-market-analysis venv (has all required dependencies)
PYTHON="/Users/samgrumet/Dropbox/Sam Grumet's Files/VSCODE/prediction-market-analysis/.venv/bin/python3"
"$PYTHON" paper_trader.py --forward --live >> "$LOG" 2>&1
EXIT_CODE=$?

echo "EXIT: $EXIT_CODE  at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
exit $EXIT_CODE
