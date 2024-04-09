#!/bin/bash

#SBATCH --ntasks=21
#SBATCH --time=24:00:00
#SBATCH --job-name="dragon"
#SBATCH --error=dragon_rte_%j.out
#SBATCH --output=dragon_rte_%j.out
#SBATCH --nodes=5

module load CUDA
module load impi/2021.7.0

mpiexec -np 21 python3 main.py --config automl_config --mh SSEA --target conso_rte --MPI --save_dir dragon_rte --filename dataset/data.csv

exit 1