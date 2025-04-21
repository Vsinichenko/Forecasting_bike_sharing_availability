#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=1     
#SBATCH --gres=gpu:1           
#SBATCH --mem=100G
#SBATCH --time=10:00:00              
#SBATCH --job-name=clustering_FB_less_complex_GPU
#SBATCH --output=logs/clustering_FB_less_complex_GPU-%j.out    
#SBATCH --error=logs/clustering_FB_less_complex_GPU-%j.err     

source /home/vasi018e/miniconda3/etc/profile.d/conda.sh

conda FBS_cluster

python "location_clustering/python_scripts/19g_clustering_FB_less_complex.py" 