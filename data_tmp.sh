#!/bin/bash

# 迁移实例的时候使用
# Step1: 复制数据到hy-tmp
cp ../../hy-public/Ninapro/Data6/DB6_s2_*.zip ../../hy-tmp
# Step2: 解压成mat文件
unzip ../../hy-tmp/DB6_s2_a.zip -d ../../hy-tmp/
unzip ../../hy-tmp/DB6_s2_b.zip -d ../../hy-tmp/
# Step3: 将mat文件移动至 hy-tmp/Data6/s1
mkdir ../../hy-tmp/Data6
mkdir ../../hy-tmp/Data6/s2
mkdir ../../hy-tmp/Data6/Processed

mv ../../hy-tmp/DB6_s2_a/*.mat ../../hy-tmp/Data6/s2
mv ../../hy-tmp/DB6_s2_b/*.mat ../../hy-tmp/Data6/s2
# Step4：数据预处理，保存成pkl的形式存入 hy-tmp/Data6/Processed
python data/data_pre.py --data_path '../../hy-tmp/Data6'
# Step5: 删除多余文件
rm -r ../../hy-tmp/DB6_s2_a/
rm -r ../../hy-tmp/DB6_s2_b/
rm -r ../../hy-tmp/__MACOSX
rm ../../hy-tmp/DB6_s2_*.zip
