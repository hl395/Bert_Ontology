#!/bin/sh -l
#
# NOTE the -l (login) flag!
# set partition
#SBATCH -p datasci -n 2
# Merge the standard out and standard error to one file
#SBATCH -J create_pretraining_data
#SBATCH -o create_pretraining_data.output
#SBATCH -e create_pretraining_data.output
# Default in slurm
#SBATCH --mail-user hl395@njit.edu
#SBATCH --mail-type=ALL

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
# Full path to executable
python create_pretraining_data.py
