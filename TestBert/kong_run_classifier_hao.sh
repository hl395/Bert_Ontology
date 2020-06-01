#!/bin/sh
#
# Usage: mytest.sh
# Change job name and email address as needed 
#        
 
# -- our name ---
#$ -N run_bert_classifier_test1
# bert_procedure_pre_training_simple_50000_10_area
# Make sure that the .e and .o file arrive in the
#working directory
#$ -cwd
#Merge the standard out and standard error to one file
#$ -j y
# Send mail at submission and completion of script
#$ -m be
#$ -M hl395@njit.edu
# Specify datasci queue
#$ -l hostname=node437
#$ -q datasci
. /opt/Modules/init/bash

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
 
#Full path to executable
python run_classifier_hao.py
