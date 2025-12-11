#!/bin/bash

#SBATCH --job-name=TabPFN-Wide-analysis
#SBATCH --array=0-5

#SBATCH --cpus-per-task=100
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=100G

#SBATCH --gres=gpu:2

#SBATCH --time=72:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

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

RUN_HDLSS_ONLY=true
#RUN_HDLSS_ONLY=${RUN_HDLSS_ONLY:-false}
bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR" "$RUN_HDLSS_ONLY"

# If this is the last task (assuming array 0-5), run the comparison script
if [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    echo "Task ID 5: Running default model analysis..."
    OUTPUT_DIR_DEFAULT="${BASE_DIR_LOCAL}/analysis_results/TabPFNv2.5_default"
    bash analysis/run_analysis.sh "default" "$OUTPUT_DIR_DEFAULT" "$RUN_HDLSS_ONLY"

    echo "Task ID 5: Running comparison plotting..."
    bash analysis/compare_results.sh "${BASE_DIR_LOCAL}/analysis_results"
fi
