#!/bin/bash
# Submits one stage from a Leonardo login node, env or benchmark:
#   scripts/leonardo/submit.sh env
#   scripts/leonardo/submit.sh benchmark <group>/<name> [--finetune] [key=value ...]
# Downloads here what a benchmark reads from W&B, queues the offline GPU job, then the
# sync job on lrd_all_serial once it succeeds. Everything the jobs write sits in one
# folder under $WORK/jobs, removed by the sync; SLURM logs go to $WORK/logs. The tiles
# it reads are cut once per patch setup by the MAE's scripts/leonardo/submit.sh cut.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a
source .env
set +a
stage=$1
test -f "scripts/leonardo/$stage.sbatch"
mkdir -p "$WORK/jobs" "$WORK/logs"
export UV_PROJECT_ENVIRONMENT=$WORK/envs/mars-multisensor-finetune
logs=$WORK/logs/%x-%j.out
if [[ $stage == env ]]; then
    sbatch --job-name=env-finetune --output="$logs" scripts/leonardo/env.sbatch
    exit
fi
run_name=$2
if [[ " ${*:3} " == *" --evaluate "* && " ${*:3} " != *" --finetune "* ]]; then
    echo "--evaluate alone reopens the W&B run of the heads, which no offline job can" >&2
    exit 1
fi
flat=${run_name//\//.}
MAE_OUTPUT_ROOT=$(mktemp -d "$WORK/jobs/$stage-$flat.XXXXXX")
export MAE_OUTPUT_ROOT
export WANDB_DIR=$MAE_OUTPUT_ROOT
export WANDB_DATA_DIR=$MAE_OUTPUT_ROOT/wandb-data
export WANDB_CACHE_DIR=$MAE_OUTPUT_ROOT/wandb-cache
trap 'rm -rf "$MAE_OUTPUT_ROOT"' ERR
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/leonardo/prepare.py "$run_name"
job=$(sbatch --parsable --job-name="$stage-$flat" --output="$logs" "scripts/leonardo/$stage.sbatch" "${@:2}")
trap - ERR
sync=$(sbatch --parsable --job-name="sync-$flat" --output="$logs" --dependency="afterok:$job" --kill-on-invalid-dep=yes scripts/leonardo/sync.sbatch "$MAE_OUTPUT_ROOT")
echo "$stage $job, sync $sync, folder $MAE_OUTPUT_ROOT"
