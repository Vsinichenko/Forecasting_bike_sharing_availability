#!/bin/bash

#SBATCH --ntasks=1                   # #SBATCH lines request resources and
#SBATCH --cpus-per-task=4                 # #SBATCH lines request resources and
#SBATCH --mem=200G
#SBATCH --time=01:00:00               # specify Slurm options
#SBATCH --job-name=arima_one_hexagon           # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=tmp/simulation-%j.out    # after the shebang line
#SBATCH --error=tmp/simulation-%j.err     # Comments start with # and do not count as interruptions

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/12c SARIMAX for some hexagons, part 1.py"