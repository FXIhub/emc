#!/bin/bash
#SBATCH --job-name=SplitModel
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH -p low

#mpirun /home/ekeberg/Work/programs/emc/split_model.py -o /scratch/fhgfs/ekeberg/emc/mimi/multiple_run4/split_41/ /home/ekeberg/Work/programs/emc/my_emc_runs/mimi/multi_run4/run_41/emc.conf 99
mpirun /home/ekeberg/Work/programs/emc/split_model.py
