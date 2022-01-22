#!/bin/bash

#python src/hp_study.py --edl 2
#python src/hp_study.py --edl 3
#python src/hp_study.py --edl 1 --tcn
#python src/hp_study.py
#python shutdown.py
python src/testing.py --edl 2
python src/testing.py --edl 3
python src/testing.py --tcn
python src/testing.py --edl 1 --tcn
python src/testing.py --edl 2 --tcn
python src/testing.py --edl 3 --tcn
