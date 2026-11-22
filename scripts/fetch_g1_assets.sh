#!/usr/bin/env bash
# Fetch the pinned Unitree G1 (29-DoF + Dex3-1 hands, rev 1.0) MJCF and meshes (BSD-3-Clause).
# The MJCF is tracked in git; meshes (~31 MB) are downloaded and git-ignored (HARNESS §5).
set -euo pipefail

REPO="https://github.com/unitreerobotics/unitree_ros"
SHA="725b59b6de39a26c425632120c89f23c49a36f1e"
SRC="robots/g1_description"
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/robots/g1_dex3"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin "$REPO"
git -C "$TMP" fetch -q --depth 1 origin "$SHA"
git -C "$TMP" checkout -q FETCH_HEAD -- "$SRC/g1_29dof_with_hand_rev_1_0.xml" "$SRC/meshes" LICENSE

mkdir -p "$DEST/meshes"
cp "$TMP/$SRC/meshes/"* "$DEST/meshes/"
cp "$TMP/LICENSE" "$DEST/LICENSE.unitree_ros"
cmp -s "$TMP/$SRC/g1_29dof_with_hand_rev_1_0.xml" "$DEST/g1_29dof_with_hand_rev_1_0.xml" \
  || cp "$TMP/$SRC/g1_29dof_with_hand_rev_1_0.xml" "$DEST/g1_29dof_with_hand_rev_1_0.xml"
echo "G1 Dex3 assets at $DEST (unitree_ros@$SHA)"
