#!/bin/bash -l

#SBATCH
#SBATCH --job-name=check
#SBATCH --time=20:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -A mshiel10
#SBATCH --mail-user=skarumu1@jh.edu
#SBATCH --mail-type=ALL


ml anaconda
conda activate py311


### Place your python script run command here
### Example :

python -u DeepONet_analysis.py $1 $2 $3 

