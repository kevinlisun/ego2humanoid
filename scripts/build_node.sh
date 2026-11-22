#!/usr/bin/env bash
# Build a node image from the repository root: scripts/build_node.sh <node> [tag]
set -euo pipefail
node="${1:?usage: build_node.sh <node> [tag]}"
tag="${2:-ego2humanoid/${node}:dev}"
cd "$(dirname "$0")/.."
docker build -f "nodes/${node}/Dockerfile" -t "${tag}" .
echo "built ${tag}"
