#!/usr/bin/env bash
# Progress bar for a run_pipeline.sh batch: pipeline_progress.sh <clip>... (LOG_DIR [/tmp/e2h_pipeline], HAND [sharpa]).
# Use with watch: watch -n 30 scripts/pipeline_progress.sh 0 1 10 100
set -uo pipefail
LOG_DIR=${LOG_DIR:-/tmp/e2h_pipeline}
HAND=${HAND:-sharpa}
total=$#
done_n=0
lines=()
for clip in "$@"; do
  log="$LOG_DIR/${HAND}_${clip}.log"
  if [ ! -f "$log" ]; then
    lines+=("$(printf '%-5s queued' "$clip")")
    continue
  fi
  res=$(grep -h "result:" "$log" | tail -1 | sed 's/^ *result: //')
  step=$(grep -h "^\[time\]" "$log" | tail -1 | sed 's/^\[time\] //; s/:.*(total/ (total/')
  age=$(( $(date +%s) - $(stat -c %Y "$log") ))
  if [ -n "$res" ]; then
    done_n=$((done_n + 1))
    lines+=("$(printf '%-5s done    %s' "$clip" "$res")")
  else
    lines+=("$(printf '%-5s running last: %s (log idle %ss)' "$clip" "${step:-starting}" "$age")")
  fi
done
width=30
filled=$(( total > 0 ? done_n * width / total : 0 ))
bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((width - filled)) '' | tr ' ' '-')
echo "[$bar] $done_n/$total clips done"
printf '%s\n' "${lines[@]}"
