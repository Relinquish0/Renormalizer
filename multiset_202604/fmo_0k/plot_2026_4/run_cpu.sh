#!/bin/bash
#SBATCH --job-name=fmo_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p CPU
#SBATCH --output=fmo_plot.log
#SBATCH --error=fmo_plot.log

source "$HOME/.bashrc"
source /software/envs/anaconda3.env
conda activate reno
export PYTHONUNBUFFERED=1


python plot.py\

exit
