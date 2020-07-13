# 实验结果报告
## 版本
* 基于镜像 tensorflow/tensorflow:latest-gpu-py3
* Python@3.6.9
* Tensorflow@2.1.0-gpu
* Keras@2.3.1
* numpy@1.18.1
* matplotlib@2.2.5

## 1. Image Classification
* 数据集 CIFAR-10
    * 训练集规模：50000
    * 测试集规模：10000
    * 图像尺寸：32 x 32 x 3
    * 输出标签共 10 类
* 数据处理
    * 输入图像只做缩放处理，除以255，取值范围控制在 [0, 1]
    * 输出的预测标签进行 one-hot 编码处理，扩展维度
* 模型 VGG16
* 超参数设置
    * batch_size，训练时 64， 测试时 128
    * 训练轮数 30
    * 优化器 Adam，学习率 0.0001
    * 损失函数 categorical crossentropy

|Training time/minutes|Training acc|Test time/seconds| Test acc|
|--|--|--|--|
|12.95|98.31%|0.714|78.82%|
|12.89|98.72%|0.679|76.65%|
|12.77|98.60%|0.706|77.92%|
|-|-|-|-|
|12.87|98.54%|0.700|77.80%|

(注：最后一行为平均值) 
 
<img src="imgs/dlf1.png" width="600">
