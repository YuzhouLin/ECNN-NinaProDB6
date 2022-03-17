#!/bin/bash

# 运行程序
# Step1:
python src/testing.py --tcn

# 上传github
git add .
git commit -m "HPO TCN sb2 training and testing completed"
git push origin cloudtest
