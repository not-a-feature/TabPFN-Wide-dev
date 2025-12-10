#!/bin/bash 

#SBATCH --job-name=TabPFN-grouping-analysis

#SBATCH --cpus-per-task=200
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti #  SBATCH --partition=h100-ferranti h100-preemptable-ferranti
#SBATCH --mem=500G

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --error=job.%J.err 
#SBATCH --output=job.%J.out 

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jules.kreuer@uni-tuebingen.de 

source /weka/pfeifer/ppu738/TabPFN-Wide/training/init.sh
python /weka/pfeifer/ppu738/TabPFN-Wide/analysis/grouping_benchmark.py 457 --extract_embeddings