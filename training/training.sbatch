#!/bin/bash 

#SBATCH --job-name=TabPFN-Wide-training
#SBATCH --array=0-8

#SBATCH --cpus-per-task=100
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=20G

#SBATCH --gres=gpu:1

#SBATCH --time=72:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

source /weka/pfeifer/ppu738/Evo2-human-v2/init.sh
python /weka/pfeifer/ppu738/Evo2-human-v2/evaluate.py --container "$EVO2_CONTAINER" --project_root "/weka/pfeifer/ppu738/Evo2-human-v2" --task_id $SLURM_ARRAY_TASK_ID --no_external_models --total_tasks 16 
