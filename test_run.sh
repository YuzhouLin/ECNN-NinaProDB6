#!/bin/bash
#SBATCH -o ./logs/%j.log
#SBATCH --mem-per-cpu=8G
THEANO_FLAGS='device=cuda,floatX=float32'

date -d -30days


#python /cluster/home/cug/yl339/TCN-proj/src/testing.py
python /cluster/home/cug/yl339/TCN-proj/src/testing.py --tcn


date -d -30days
