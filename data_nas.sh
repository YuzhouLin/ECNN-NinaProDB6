#!/bin/bash

# 迁移实例的时候使用
# Step1: 复制数据到hy-nas
cp ../../hy-public/Ninapro/Data6/DB6_s2_*.zip ../../hy-nas
# Step2: 解压成mat文件
unzip ../../hy-nas/DB6_s2_a.zip -d ../../hy-nas/
unzip ../../hy-nas/DB6_s2_b.zip -d ../../hy-nas/
# Step3: 将mat文件移动至 hy-nas/Data6/s1
mkdir ../../hy-nas/Data6
mkdir ../../hy-nas/Data6/s2
mkdir ../../hy-nas/Data6/Processed

mv ../../hy-nas/DB6_s2_a/*.mat ../../hy-nas/Data6/s2
mv ../../hy-nas/DB6_s2_b/*.mat ../../hy-nas/Data6/s2
# Step4：数据预处理，保存成pkl的形式存入 hy-nas/Data6/Processed
python data/data_pre.py --data_path '../../hy-nas/Data6'
# Step5: 删除多余文件
rm -r ../../hy-nas/DB6_s2_a/
rm -r ../../hy-nas/DB6_s2_b/
rm -r ../../hy-nas/__MACOSX
rm ../../hy-nas/DB6_s2_*.zip
