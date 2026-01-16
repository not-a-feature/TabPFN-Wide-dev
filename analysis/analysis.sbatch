#!/bin/bash

#SBATCH --job-name=TabPFN-Wide-analysis
#SBATCH --array=0-10

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

# Handle default models and specific checkpoints
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    CHECKPOINT_PATH="default_n1g1"
    OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/default_n1g1"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    CHECKPOINT_PATH="default_n8g3"
    OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/default_n8g3"
else
    # Define specific checkpoints
    CHECKPOINT_NAMES=(
        "0_AddFeat1500_NEst1_Group1"
        "1_AddFeat5000_NEst1_Group1"
        "2_AddFeat8000_NEst1_Group1"
        "3_AddFeat1500_NEst8_Group3"
        "4_AddFeat5000_NEst8_Group3"
        "5_AddFeat8000_NEst8_Group3"
    )

    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CHECKPOINT_NAMES[@]}" ]; then
         echo "Invalid array index $SLURM_ARRAY_TASK_ID"
         exit 1
    fi

    CHECKPOINT_NAME=${CHECKPOINT_NAMES[$SLURM_ARRAY_TASK_ID]}
    CURRENT_CHECKPOINT_DIR="${BASE_DIR_LOCAL}/checkpoints/${CHECKPOINT_NAME}"

    if [ ! -d "$CURRENT_CHECKPOINT_DIR" ]; then
        echo "Checkpoint directory not found: $CURRENT_CHECKPOINT_DIR"
        exit 1
    fi

    # Find the latest checkpoint file
    CHECKPOINT_PATH=$(ls -t "$CURRENT_CHECKPOINT_DIR"/*.pt | head -n 1)

    if [ -z "$CHECKPOINT_PATH" ]; then
        echo "No checkpoint found in $CURRENT_CHECKPOINT_DIR"
        exit 1
    fi

    OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/${CHECKPOINT_NAME}"
fi

echo "Running analysis for checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

RUN_HDLSS_ONLY=false
#RUN_HDLSS_ONLY=${RUN_HDLSS_ONLY:-false}
bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR" "$RUN_HDLSS_ONLY" 500

echo "Running analysis for checkpoint (no subsampling): $CHECKPOINT_PATH"
OUTPUT_DIR_NOSUB="${OUTPUT_DIR}_nosub"

echo "Output directory (no subsampling): $OUTPUT_DIR_NOSUB"
bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR_NOSUB" "$RUN_HDLSS_ONLY" 50000

# If this is the last task (assuming array 0-10), run the comparison script
if [ "$SLURM_ARRAY_TASK_ID" -eq 10 ]; then
    echo "Task ID 10: Running comparison plotting..."
    bash analysis/compare_results.sh "${BASE_DIR_LOCAL}/analysis_results"
fi
