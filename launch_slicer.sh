#!/bin/bash
#SBATCH --job-name=SliceInserter
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH -p low

mpirun ./split_model.py
