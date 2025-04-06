#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=100                
#SBATCH --mem=10G
#SBATCH --time=10:00:00              
#SBATCH --job-name=sarimax_calendar_weather_FB_1_demand          
#SBATCH --output=logs/sarimax_calendar_weather_FB_1_demand_-%j.out    
#SBATCH --error=logs/sarimax_calendar_weather_FB_1_demand_-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda activate FBSfor_arima

python "python_scripts/17a_SARIMAX_calendar_weather.py"  --city FB --part 1 --depvar demand