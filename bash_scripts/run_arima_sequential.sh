#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=100                
#SBATCH --mem=100G
#SBATCH --time=10:00:00              
#SBATCH --job-name=sarimax_calendar_FB          
#SBATCH --output=logs/sarimax_calendar_FB-%j.out    
#SBATCH --error=logs/sarimax_calendar_FB-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/16c SARIMAX wth weather and calendar for all hex.py"