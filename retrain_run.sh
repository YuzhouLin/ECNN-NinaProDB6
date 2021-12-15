#!/bin/bash
#SBATCH -o ./logs/%j.log
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=yl339@kent.ac.uk
THEANO_FLAGS='device=cuda,floatX=float32'

date -d -30days

python /cluster/home/cug/yl339/TCN-proj/src/retrain.py 0

date -d -30days

