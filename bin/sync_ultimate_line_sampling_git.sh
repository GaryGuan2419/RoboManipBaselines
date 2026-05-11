#!/usr/bin/env bash
# Merge only the ultimate-line 9-camera replay sampling files from a git ref
# into the current working tree. Does not switch branches or create a worktree.
#
# Typical server usage (inside an existing RoboManipBaselines checkout).
# First time only: this script did not exist on your branch until 7f6e275, so
# fetch the ref then materialize the script from it, then run it:
#   git fetch myfork ultimate-line-9cam-sampling
#   git checkout myfork/ultimate-line-9cam-sampling -- bin/sync_ultimate_line_sampling_git.sh
#   chmod +x bin/sync_ultimate_line_sampling_git.sh
#   ./bin/sync_ultimate_line_sampling_git.sh
#
# Later updates:
#   git fetch myfork ultimate-line-9cam-sampling
#   ./bin/sync_ultimate_line_sampling_git.sh
#
# Override ref / remote:
#   ./bin/sync_ultimate_line_sampling_git.sh myfork/ultimate-line-9cam-sampling
#   GIT_SYNC_REMOTE=myfork GIT_SYNC_BRANCH=ultimate-line-9cam-sampling ./bin/sync_ultimate_line_sampling_git.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REF="${1:-}"
REMOTE="${GIT_SYNC_REMOTE:-myfork}"
BRANCH="${GIT_SYNC_BRANCH:-ultimate-line-9cam-sampling}"

if [[ -z "$REF" ]]; then
  REF="${REMOTE}/${BRANCH}"
fi

PATHS=(
  bin/replay_ultimate_line_keyframes.py
  bin/sync_ultimate_line_sampling_git.sh
  robo_manip_baselines/envs/mujoco/hsr/MujocoDualHsrDemoEnv.py
  robo_manip_baselines/envs/mujoco/hsr/MujocoDualHsrUltimateLineEnv.py
  robo_manip_baselines/envs/assets/mujoco/envs/hsr/env_dual_hsr_ultimate_line.xml
)

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not inside a git repository: $REPO_ROOT" >&2
  exit 1
fi

if [[ "$REF" == */* ]]; then
  REMOTE_NAME="${REF%%/*}"
  BRANCH_NAME="${REF#*/}"
  if ! git rev-parse --verify --quiet "$REF^{commit}" >/dev/null; then
    echo "[sync] Fetching ${REMOTE_NAME} ${BRANCH_NAME}..."
    git fetch "$REMOTE_NAME" "$BRANCH_NAME"
  fi
else
  if ! git rev-parse --verify --quiet "$REF^{commit}" >/dev/null; then
    echo "error: ref not found: $REF" >&2
    exit 1
  fi
fi

echo "[sync] Checking out sampling files from $REF"
git checkout "$REF" -- "${PATHS[@]}"

echo "[sync] Updated files:"
for path in "${PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    echo "  $path"
  else
    echo "  missing: $path" >&2
    exit 1
  fi
done

echo "[sync] Done. Review with: git status -- ${PATHS[*]}"
