#!/bin/bash
#SBATCH -o ./logs/%j.log
THEANO_FLAGS='device=cuda,floatX=float32'

date -d -30days

python /cluster/home/cug/yl339/current_proj/src/train.py 3

date -d -30days
