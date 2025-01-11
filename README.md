# U-Net Project

## 项目简介
本项目实现了在PyTorch环境下的U-Net模型，并包含了用于图像分割的训练和数据处理代码。

## 文件结构

├── unet
│   ├── unet_model.py  # 定义U-Net模型
│   └── unet_parts.py       # 定义U-Net组件
├── data           # 数据文件
├── AHMTrain.py        # 通过自适应硬掩码训练U-Net
├── SegTrain.py        # 训练U-Net进行图像分割
└── README.md          # 项目说明文件

## 使用说明
训练U-Net模型

1. 
通过自适应硬掩码训练U-Net：

python AHMTrain.py

1. 
训练U-Net进行图像分割：

python SegTrain.py