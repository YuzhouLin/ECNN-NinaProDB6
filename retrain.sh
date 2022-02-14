#!/bin/bash

python src/retrain.py
python src/retrain.py --edl 1
python src/retrain.py --edl 2
python src/retrain.py --edl 3
python src/retrain.py --tcn
python src/retrain.py --edl 1 --tcn
python src/retrain.py --edl 2 --tcn
python src/retrain.py --edl 3 --tcn
