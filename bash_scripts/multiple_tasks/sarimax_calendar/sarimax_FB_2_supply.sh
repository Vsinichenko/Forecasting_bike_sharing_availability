#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=100                
#SBATCH --mem=10G
#SBATCH --time=10:00:00              
#SBATCH --job-name=FB_2_supply          
#SBATCH --output=logs/sarimax_calendar_FB_2_supply_-%j.out    
#SBATCH --error=logs/sarimax_calendar_FB_2_supply_-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/16b_SARIMAX_with_calendar_all_hex.py"  --city FB --part 2 --depvar supply