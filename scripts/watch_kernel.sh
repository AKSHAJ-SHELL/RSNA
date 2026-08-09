#!/usr/bin/env bash
# Watch a Kaggle kernel from the terminal until it finishes, then print its log.
#
#   scripts/watch_kernel.sh [kernel-slug] [poll-seconds]
#   scripts/watch_kernel.sh akshajshandilya/rsna-knee-build-pixel-cache 60
#
# Kaggle does not stream logs for a running kernel over the API — `kernels output` returns
# nothing until the run ends, and `kernels status` reports only RUNNING/COMPLETE/ERROR. So this
# tracks state and elapsed time, and dumps the log the moment it becomes available. For live
# per-cell output while it runs, the browser is the only option:
#   https://www.kaggle.com/code/<slug>/log

set -uo pipefail

KERNEL="${1:-akshajshandilya/rsna-knee-build-pixel-cache}"
POLL="${2:-60}"
LOGDIR="$(mktemp -d)"
START=$(date +%s)

printf 'watching %s (polling every %ss, Ctrl-C to stop)\n' "$KERNEL" "$POLL"
printf 'live cell output: https://www.kaggle.com/code/%s/log\n\n' "$KERNEL"

while :; do
  RAW="$(kaggle kernels status "$KERNEL" 2>&1)"
  STATUS="$(printf '%s' "$RAW" | sed -n 's/.*KernelWorkerStatus\.\([A-Z_]*\).*/\1/p' | head -1)"
  [ -z "$STATUS" ] && STATUS="UNKNOWN"

  NOW=$(date +%s); ELAPSED=$((NOW - START))
  printf '\r[%02d:%02d:%02d] %-12s' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$STATUS"

  case "$STATUS" in
    RUNNING|QUEUED|UNKNOWN) sleep "$POLL" ;;
    *) break ;;
  esac
done

printf '\n\n=== %s after %dm ===\n' "$STATUS" $(( ($(date +%s) - START) / 60 ))

if kaggle kernels output "$KERNEL" -p "$LOGDIR" >/dev/null 2>&1; then
  LOG="$(find "$LOGDIR" -name '*.log' | head -1)"
  if [ -n "$LOG" ]; then
    # The log is JSON records, not plain text; print the streams in order.
    python3 - "$LOG" <<'PY'
import json, sys
for entry in json.load(open(sys.argv[1])):
    text = entry.get("data", "").rstrip()
    if text:
        print(text)
PY
  fi
  echo
  echo "output files:"
  find "$LOGDIR" -type f -exec ls -lh {} \; | awk '{print "  ", $5, $9}'
fi

[ "$STATUS" = "COMPLETE" ] && exit 0 || exit 1
