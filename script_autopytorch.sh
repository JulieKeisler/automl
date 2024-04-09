#!/bin/bash

#SBATCH --time=25:00:00
#SBATCH --job-name="autopytorch"
#SBATCH --error=autopytorch_%j.out
#SBATCH --output=autopytorch_%j.out
#SBATCH --nodes=1

module load CUDA
module load impi/2021.7.0


srun python3 main.py --config automl_config --mh autopytorch --target conso_rte --MPI --save_dir autopytorch --filename dataset/data.csv

exit 1
