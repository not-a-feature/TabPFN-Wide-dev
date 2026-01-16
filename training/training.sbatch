#!/bin/bash 

#SBATCH --job-name=TabPFN-Wide-training
#SBATCH --array=0-2

#SBATCH --cpus-per-task=100
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti # --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=50G

#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

source /weka/pfeifer/ppu738/TabPFN-Wide/training/init.sh
source /weka/pfeifer/ppu738/TabPFN-Wide/training/train.sh $SLURM_ARRAY_TASK_ID