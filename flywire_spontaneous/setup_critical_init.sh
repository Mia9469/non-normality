#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

URL="https://github.com/MouseLand/critical_init.git"
COMMIT="2c15edf4e770165fc3962dcc3920c8bcaf555bed"
TARGET="critical_init_official"

# Fast path: if the checkout already exists at the pinned commit and is clean,
# do NOT touch the network (servers may have no GitHub access).
if [[ -d "${TARGET}/.git" ]] \
   && [[ "$(git -C "${TARGET}" rev-parse HEAD 2>/dev/null)" == "${COMMIT}" ]] \
   && git -C "${TARGET}" diff --quiet -- powerlaw.py lyapun.py \
   && git -C "${TARGET}" diff --cached --quiet -- powerlaw.py lyapun.py; then
  echo "official MouseLand/critical_init already at ${COMMIT} (offline, skipping fetch)"
  exit 0
fi

# Otherwise we need the network to obtain/repair the pinned commit.
if [[ ! -d "${TARGET}/.git" ]]; then
  git clone "${URL}" "${TARGET}"
fi
git -C "${TARGET}" fetch origin "${COMMIT}" --depth 1
git -C "${TARGET}" checkout --detach "${COMMIT}"
test "$(git -C "${TARGET}" rev-parse HEAD)" = "${COMMIT}"
git -C "${TARGET}" diff --quiet -- powerlaw.py lyapun.py
git -C "${TARGET}" diff --cached --quiet -- powerlaw.py lyapun.py
echo "official MouseLand/critical_init ready at ${COMMIT}"
