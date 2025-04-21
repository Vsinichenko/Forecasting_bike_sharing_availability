#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=10                
#SBATCH --mem=10G
#SBATCH --time=10:00:00              
#SBATCH --job-name=clustering_FB_less_complex
#SBATCH --output=logs/clustering_FB_less_complex-%j.out    
#SBATCH --error=logs/clustering_FB_less_complex-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda FBS_cluster

python "location_clustering/python_scripts/19g_clustering_FB_less_complex.py" 