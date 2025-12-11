#!/bin/bash
#SBATCH --job-name=TabPFN-Wide-analysis
#SBATCH --array=0-5
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=analysis_job_%A_%a.out
#SBATCH --error=analysis_job_%A_%a.err

# Source init script
source training/init.sh

# Get list of checkpoint directories
# We assume checkpoints are in $BASE_DIR_LOCAL/checkpoints
CHECKPOINT_ROOT="${BASE_DIR_LOCAL}/checkpoints"
CHECKPOINT_DIRS=($(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort))

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "No checkpoint directories found in $CHECKPOINT_ROOT"
    exit 1
fi

CURRENT_CHECKPOINT_DIR=${CHECKPOINT_DIRS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$CURRENT_CHECKPOINT_DIR" ]; then
    echo "Invalid array index $SLURM_ARRAY_TASK_ID for ${#CHECKPOINT_DIRS[@]} checkpoints"
    exit 1
fi

# Find the latest checkpoint file
CHECKPOINT_PATH=$(ls -t "$CURRENT_CHECKPOINT_DIR"/*.pt | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "No checkpoint found in $CURRENT_CHECKPOINT_DIR"
    exit 1
fi

OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/$(basename "$CURRENT_CHECKPOINT_DIR")"

echo "Running analysis for checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

RUN_HDLSS_ONLY=${RUN_HDLSS_ONLY:-false}
bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR" "$RUN_HDLSS_ONLY"
