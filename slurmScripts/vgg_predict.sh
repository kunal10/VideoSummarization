#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=00:10:00 
#SBATCH --error=../logs/vggPredict10000.err 
#SBATCH --output=../logs/vggPredict10000.out
#SBATCH --job-name=EvalVgg
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to VideoSummarization directory.
cd ..

luajit vggPredict.lua -model models/vgg10000.t7 -output_file results/vgg_predictions10000.txt 

echo "\nFinished with exit code $? at: `date`"

