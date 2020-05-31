#!/bin/bash
#
# set the job name
#$ -N my_job_name
# set the output file name
#$ -o test.output
# default the current working directory as root directory
#$ -cwd
# set the email address for notification
#$ -M ucid@njit.edu
# set memory requirement
#$ -l mem=4G
# set which queue to run the job
#$ -q datasci
# set which node to run the job
#$ -node=437

python my_file.py