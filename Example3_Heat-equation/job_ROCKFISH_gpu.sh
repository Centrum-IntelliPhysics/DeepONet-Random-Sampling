#!/bin/bash -l

#SBATCH
#SBATCH --job-name=check
#SBATCH --time=03-00:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -A mshiel10_gpu
#SBATCH --mail-user=skarumu1@jh.edu
#SBATCH --mail-type=ALL


ml anaconda
conda activate py311


### Place your python script run command here
### Example :

python -u DeepONet_analysis.py $1 $2 $3 

