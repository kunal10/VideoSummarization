#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=12:00:00 
#SBATCH --error=../logs/preprocess.err 
#SBATCH --output=../logs/preprocess.out
#SBATCH --job-name=Preprocessing
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to VideoSummarization directory.
cd ..
# Preprocess images
luajit preprocess.lua

echo "\nFinished with exit code $? at: `date`"

