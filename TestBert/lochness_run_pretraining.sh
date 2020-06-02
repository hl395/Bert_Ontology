#!/bin/sh -l
#
# NOTE the -l (login) flag!
# set partition
#SBATCH -p datasci -n 2
# Merge the standard out and standard error to one file
#SBATCH -J run_pretraining
#SBATCH -o run_pretraining.output
#SBATCH -e run_pretraining.output
# Default in slurm
#SBATCH --mail-user hl395@njit.edu
#SBATCH --mail-type=ALL
#SBATCH --nodelist=node[437]

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
# Full path to executable
python run_pretraining.py
