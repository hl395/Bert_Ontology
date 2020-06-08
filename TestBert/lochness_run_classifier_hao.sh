#!/bin/bash
#
#SBATCH --partition=datasci
# Merge the standard out and standard error to one file
#SBATCH -J run_classifier_hao
#SBATCH -o %x.%j.output       # expand to job name.jobid.output
#SBATCH --mail-user hl395@njit.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:TitanRtx:1  # if you need 2 GPUs gpu:TitanRtx:2
#SBATCH --mem=24G

. /opt/Modules/init/bash

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
# Full path to executable
python run_classifier_hao.py
