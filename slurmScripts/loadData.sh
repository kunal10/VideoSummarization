#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --time=00:30:00 
#SBATCH --error=job.predictbackward.err 
#SBATCH --output=job.predictbackward.out
#SBATCH --job-name=PredictBackwardFeatures
#SBATCH --mail-user=kunal.lad@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

luajit loadData.lua -train_labels=data/tvsum50/trainLabels.txt -test_label=data/tvsum50/testLabels.txt -cnn_proto=models/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model=models/VGG_ILSVRC_19_layers.caffemodel 

echo "\nFinished with exit code $? at: `date`"

