#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=100                
#SBATCH --mem=100G
#SBATCH --time=10:00:00              
#SBATCH --job-name=sarimax_calendar_          
#SBATCH --output=logs/sarimax_calendar_-%j.out    
#SBATCH --error=logs/sarimax_calendar_-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/16b_SARIMAX_with_calendar_all_hex.py" --dep_var demand