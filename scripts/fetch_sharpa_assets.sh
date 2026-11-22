#!/usr/bin/env bash
# Fetch the pinned Sharpa Wave five-finger tactile hand MJCF and meshes (Apache-2.0) from MuJoCo Menagerie.
# The MJCF is tracked in git; meshes (~15 MB) are downloaded and git-ignored (HARNESS §5).
set -euo pipefail

REPO="https://github.com/google-deepmind/mujoco_menagerie"
SHA="370c43965323c0944b433b202aac5058b5f907a9"
SRC="sharpa_wave"
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/robots/sharpa_wave"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin "$REPO"
git -C "$TMP" fetch -q --depth 1 origin "$SHA"
git -C "$TMP" checkout -q FETCH_HEAD -- "$SRC"

mkdir -p "$DEST"
cp -r "$TMP/$SRC/assets" "$DEST/"
cp "$TMP/$SRC/LICENSE" "$TMP/$SRC/README.md" "$DEST/"
for f in left_hand.xml right_hand.xml; do
  cmp -s "$TMP/$SRC/$f" "$DEST/$f" || cp "$TMP/$SRC/$f" "$DEST/$f"
done
echo "Sharpa Wave assets at $DEST (mujoco_menagerie@$SHA)"
