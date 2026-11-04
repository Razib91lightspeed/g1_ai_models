#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/rhasan/Desktop/razib/ai_models
CODE="$ROOT/simulations/unitree_g1/parallel_rl/experiments/g1_goal_navigation_training/scripts/train_nav_ppo.py"
BASE="$ROOT/simulations/unitree_g1/parallel_rl/experiments/g1_goal_navigation_training"
RUN="$BASE/overnight_20261103"
TEMP="$BASE/temp"
IMAGE='nvcr.io/nvidia/pytorch:25.11-py3'
NAME='g1-nav-proxy-overnight-20261103'
STOP_TIME="${STOP_TIME:-2026-11-04T08:00}"

cd "$ROOT"
[[ -f "$CODE" ]] || { echo "STOP: Missing $CODE"; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "STOP: Docker image $IMAGE not present (do not silently pull another image)"; exit 1; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "STOP: container $NAME already exists"; exit 1
fi
mkdir -p "$RUN" "$TEMP" "$BASE/smoke"
# Explicitly test the training script in a disposable, four-environment smoke run.
echo '===== FOUR-ENVIRONMENT GPU SMOKE TEST ====='
docker run --rm --network none --log-driver none \
  --device nvidia.com/gpu=all \
  -v "$CODE:/workspace/train_nav_ppo.py:ro" \
  -v "$BASE/smoke:/workspace/output:rw" \
  -v "$TEMP:/workspace/tmp:rw" \
  -e HOME=/workspace/tmp -e TMPDIR=/workspace/tmp \
  -e XDG_CACHE_HOME=/workspace/tmp -e TORCH_HOME=/workspace/tmp \
  "$IMAGE" bash -lc \
  'python -u -B /workspace/train_nav_ppo.py --smoke --envs 4 --max-updates 2 --deadline 2026-11-04T12:00 --output /workspace/output > /workspace/output/smoke.log 2>&1'
cat "$BASE/smoke/smoke.log"
[[ -s "$BASE/smoke/latest.pt" ]] || { echo 'STOP: smoke checkpoint missing'; exit 1; }

echo "===== STARTING DETACHED TRAINING UNTIL $STOP_TIME (HELSINKI) ====="
docker run -d --name "$NAME" --restart no --network none --log-driver none \
  --device nvidia.com/gpu=all \
  -v "$CODE:/workspace/train_nav_ppo.py:ro" \
  -v "$RUN:/workspace/output:rw" \
  -v "$TEMP:/workspace/tmp:rw" \
  -e HOME=/workspace/tmp -e TMPDIR=/workspace/tmp \
  -e XDG_CACHE_HOME=/workspace/tmp -e TORCH_HOME=/workspace/tmp \
  -e STOP_TIME="$STOP_TIME" \
  "$IMAGE" bash -lc \
  'exec python -u -B /workspace/train_nav_ppo.py --envs 64 --deadline "$STOP_TIME" --output /workspace/output > /workspace/output/train.log 2>&1'
echo "Container: $NAME"
echo "Live log: $RUN/train.log"
echo "Metrics: $RUN/metrics.csv"
echo "Checkpoint: $RUN/latest.pt"
echo "You may disconnect from SSH; Docker stays running."
