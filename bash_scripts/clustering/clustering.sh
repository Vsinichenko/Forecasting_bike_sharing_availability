#!/bin/bash

#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=200                
#SBATCH --mem=200G
#SBATCH --time=10:00:00              
#SBATCH --job-name=clustering_FB
#SBATCH --output=logs/clustering_FB-%j.out    
#SBATCH --error=logs/clustering_FB-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda FBS_cluster

python "location_clustering/python_scripts/19d_clustering.py" 