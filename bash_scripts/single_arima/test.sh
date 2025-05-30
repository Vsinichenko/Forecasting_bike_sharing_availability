#!/bin/bash

#SBATCH --ntasks=1                   # #SBATCH lines request resources and
#SBATCH --cpus-per-task=15                 # #SBATCH lines request resources and
#SBATCH --mem=100G
#SBATCH --time=01:00:00               # specify Slurm options
#SBATCH --job-name=sarima_all          # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=tmp/sarima_all-%j.out    # after the shebang line
#SBATCH --error=tmp/sarima_all-%j.err     # Comments start with # and do not count as interruptions

#source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

#conda activate FBSfor_arima

python "python_scripts/test_logging.py"