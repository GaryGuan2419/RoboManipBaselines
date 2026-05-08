#!/usr/bin/env bash
# 后台批量 Rollout：hsr_side_pick（MujocoHsrTidyup），无渲染、无 plot，自动连跑多局。
#
# 用法（在仓库根目录）:
#   chmod +x bin/batch_rollout_hsr_side_pick.sh
#   ./bin/batch_rollout_hsr_side_pick.sh
#
# 环境变量（可选）:
#   EPISODES=200          局数（默认 200）
#   CKPT=.../policy_best.ckpt   checkpoint 路径
#   OUT_DIR=.../runs      日志与 yaml 输出目录
#   SKIP=2                与训练一致的 skip
#   EXTRA_ARGS='...'      追加传给 Rollout 的参数
#
# 输出:
#   $OUT_DIR/rollout_${EPISODES}_${STAMP}.log
#   $OUT_DIR/rollout_${EPISODES}_${STAMP}.yaml   # success/reward/duration 列表

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPISODES="${EPISODES:-200}"
CKPT="${CKPT:-$REPO_ROOT/robo_manip_baselines/checkpoint/ManiFlowPolicy/hsr_side_pick/policy_best.ckpt}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/hsr_pick_rollout_batch}"
SKIP="${SKIP:-2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_DIR/rollout_${EPISODES}_${STAMP}.log"
RES="$OUT_DIR/rollout_${EPISODES}_${STAMP}.yaml"

mkdir -p "$OUT_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

# 后台: nohup + 日志重定向；PYTHONUNBUFFERED 便于 tail -f 看进度
nohup env PYTHONUNBUFFERED=1 python -m robo_manip_baselines.bin.Rollout \
  ManiFlowPolicy MujocoHsrTidyup \
  --checkpoint "$CKPT" \
  --world_idx 0 \
  --world_idx_repeat_count "$EPISODES" \
  --skip "$SKIP" \
  --no_render \
  --no_plot \
  --auto_exit \
  --max_duration 120 \
  --result_filename "$RES" \
  ${EXTRA_ARGS:-} \
  >"$LOG" 2>&1 &

echo "Started background Rollout"
echo "  PID=$!"
echo "  log=$LOG"
echo "  result_yaml=$RES"
echo "  episodes=$EPISODES"
echo "Follow: tail -f $LOG"
