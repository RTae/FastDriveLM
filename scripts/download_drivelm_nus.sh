#!/bin/bash

# Download mapping files for DriveLM-NUS dataset
wget https://cloud.tsinghua.edu.cn/f/768fdde7e79a4c96a0ae/?dl=1 -O ./datasets/v1_1_train_nus.json 
# Download image files for DriveLM-NUS dataset
wget https://cloud.tsinghua.edu.cn/f/3c943f91efc9460999a6/?dl=1 -O ./datasets/nus_images_train.zip
# Unzip the downloaded image files
unzip ./datasets/nus_images_train.zip -d ./datasets/nus_images_train