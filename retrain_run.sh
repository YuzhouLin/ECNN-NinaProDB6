#!/bin/bash
#SBATCH -o ./logs/%j.log
THEANO_FLAGS='device=cuda,floatX=float32'

date -d -30days

python /cluster/home/cug/yl339/current_proj/src/retrain.py 0
python /cluster/home/cug/yl339/current_proj/src/retrain.py 1
python /cluster/home/cug/yl339/current_proj/src/retrain.py 2
python /cluster/home/cug/yl339/current_proj/src/retrain.py 3

date -d -30days
