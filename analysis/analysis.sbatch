#!/bin/bash

#SBATCH --job-name=TabPFN-Wide-analysis
#SBATCH --array=0-4

#SBATCH --cpus-per-task=100
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=100G

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

# Source init script
source training/init.sh

# Define Jobs: Type:Value
# TODO Uppdate array size
JOBS=(
    "MODEL:v2"
    "MODEL:wide-v2-1.5k-nocat"
    "MODEL:wide-v2-5k-nocat"
    "MODEL:wide-v2-8k-nocat"
    "MODEL:wide-v2-5k"
    #"CHECKPOINT:0_AddFeat1500_NEst1_Group1"
    #"CHECKPOINT:1_AddFeat5000_NEst1_Group1"
    #"CHECKPOINT:2_AddFeat8000_NEst1_Group1"
)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#JOBS[@]}" ]; then
    echo "Invalid array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

JOB_INFO=${JOBS[$SLURM_ARRAY_TASK_ID]}
JOB_TYPE=${JOB_INFO%%:*}
JOB_VALUE=${JOB_INFO#*:}

if [ "$JOB_TYPE" == "MODEL" ]; then
    CHECKPOINT_PATH="$JOB_VALUE"
    OUTPUT_DIR="${BASE_DIR_LOCAL}/analysis_results/${JOB_VALUE}"
elif [ "$JOB_TYPE" == "CHECKPOINT" ]; then
    CHECKPOINT_NAME="$JOB_VALUE"
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
else
    echo "Unknown job type: $JOB_TYPE"
    exit 1
fi

echo "Running analysis for: $JOB_INFO"
echo "Checkpoint/Model: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

RUN_HDLSS_ONLY=false
# RUN_HDLSS_ONLY=${RUN_HDLSS_ONLY:-false}

bash analysis/run_analysis.sh "$CHECKPOINT_PATH" "$OUTPUT_DIR" "$RUN_HDLSS_ONLY" 50000

# If this is the last task, run the comparison script
if [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    echo "Task ID 6: Running comparison plotting..."
    bash analysis/compare_results.sh "${BASE_DIR_LOCAL}/analysis_results"
fi
