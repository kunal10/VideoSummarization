#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=6:00:00 
#SBATCH --error=../logs/lstm_mse.err 
#SBATCH --output=../logs/lstm_mse.out
#SBATCH --job-name=LSTM_MSE
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# cd to VideoSummarization directory.
cd ..
# Train BLSTM
luajit lstm.lua -num_iterations 10000 -num_batches 200 -model_prefix models/lstm_mse  -train_data data/tvsum50/train_data200.t7 -train_targets data/tvsum50/train_data_labels200.t7 

echo "\nFinished with exit code $? at: `date`"

