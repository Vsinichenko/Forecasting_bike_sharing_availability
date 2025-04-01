#!/bin/bash

#SBATCH --ntasks=1                   # #SBATCH lines request resources and
#SBATCH --cpus-per-task=15                 # #SBATCH lines request resources and
#SBATCH --mem=200G
#SBATCH --time=10:00:00               # specify Slurm options
#SBATCH --job-name=sarima_all_multi          # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=tmp/sarima_all_multi-%j.out    # after the shebang line
#SBATCH --error=tmp/sarima_all_multi-%j.err     # Comments start with # and do not count as interruptions

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/16b SARIMAX wth weather and calendar for all hex.py"