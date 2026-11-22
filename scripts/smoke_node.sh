#!/usr/bin/env bash
# Smoke-test a built node image: scripts/smoke_node.sh <preflight|replay> [tag]
set -euo pipefail
node="${1:?usage: smoke_node.sh <node> [tag]}"
tag="${2:-ego2humanoid/${node}:dev}"
out="$(mktemp -d)"
trap 'rm -rf "${out}"' EXIT
case "${node}" in
  preflight) docker run --rm "${tag}" --help >/dev/null ;;
  replay)
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "${out}:/out" "${tag}" synthetic-run /out
    test -s "${out}/replay/render.mp4" && test -s "${out}/replay/qa_report.json"
    ;;
  *) echo "unknown node ${node}" >&2; exit 2 ;;
esac
echo "smoke ok: ${node}"
