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
### 模型 VGG16
* 超参数设置
    * batch_size，训练时 64， 测试时 128
    * 训练轮数 30
    * 优化器 Adam，学习率 1e-4
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

## 2. Text Sentimnet Analysis
* 数据集 IMDB
    * 训练集规模：25k
    * 测试集规模：25k
    * 二分类，输出结果是影评的情感为正面/负面
* 数据处理
    * 输入数据是单词索引组成的元组，对其采取词嵌入编码 word embedding

### 模型 Bi-LSTM
* 超参数设置
    * batch_size 64
    * 训练轮数 20
    * 优化器 Adam，学习率 1e-4
    * 单词总数 32k
    * 词嵌入维度 64
    * 验证集占训练集比例 0.2
    * dropout=0，即不采取随机失活
    * 评论长度 MAXLEN，超出此长度的被截断，这里并未给出，采取和下个模型相同的参数 128
* 模型结构
    * Embedding
    * Bi-LSTM
    * Bi-LSTM
    * FC
    * FC

|Training time/minutes|Training acc|Validation acc|Test time/seconds| Test acc|
|--|--|--|--|--|
|34.19|99.78%|84.46%|19.787|82.06%|
|32.62|99.85%|83.50%|19.060|82.99%|
|32.84|99.92%|84.30%|18.703|82.93%|
|-|-|-|-|
|33.21|99.85%|84.08%|19.183|82.66%|
<br>
<img src="imgs/dlf2.png" width="600"><br>

注：虚线为训练集，实线为验证集；  
从图像来看，存在过拟合问题，可能因为没有采取正则化；  
LSTM适于分析全局的长期性结构，在情感分析问题上表现得差一些。

### 模型 Bert Large Uncased 768-hidden
* 用于预训练 NLP，无监督、深度双向模型
    * per-training，基于大量文本做的无监督学习
    * fine-tuning，用于具体任务时需要微调
* 超参数设置
    * batch-size 32
    * 训练轮数 3
    * 优化器 AdamW，学习率 1e-5，衰减参数 1e-6
    * MAX_SEQ_LENGTH 128

<img src="imgs/bert1.png" width="480">

* 网络结构：由于 Pytorch 直接使用了 huggingface 的模型，BertForSequenceClassification，此处对于 Tensorflow，采取上面所示的，自己设置的模型
    * 加载预训练的 bert 分类器
    * 展开，加上一层全连接层
    * 应用 Dropout，取需要丢弃的比例为 0.3
    * 最后加上输出层，应用 sigmoid 激活函数，输出二分类结果

* 训练结果如下。如果应用 Adam 而非 AdamW 优化器，模型表现还会略有上升。

|Training time/minutes|Training acc|Test time/seconds| Test acc|
|--|--|--|--|
|37.58|85.30%|57.25|92.74%|
|34.73|85.74%|58.49|91.82%|
|33.48|85.71%|58.08|91.53%|
|-|-|-|-|
|35.26|85.58%|57.49|92.03%|

* 参考
    * 基于预训练部分做编码 encoding [链接](https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/)
    * 数据[下载](https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git)


## 3. Image Generation
* 数据集 MNIST 手写数字识别
    * 图像尺寸：28 x 28 x 1
* 数据处理
    * 归一化到 [-0.5, 0.5]
### 模型 Dc-Gan
* GANs，生成对抗网络，两个模型通过对抗过程同时训练
    * 生成器，学习生成看起来真实的图像
    * 判别器，学习区分真假图像
    * 训练过程中，二者的能力都逐渐增强，直到判别器无法区分真实图片和伪造图片时，训练达到平衡。
* 超参数设置
    * batch-size 64
    * 训练轮数 20
    * 优化器 Adam，学习率 2e-4，betas (0.5, 0.999)
* 生成器模型

<img src="imgs/gan_gen.png" width="400">

* 判别器模型

<img src="imgs/gan_disc.png" width="400">

* 问题
    * dcgan_tutorial.py：目前在colab运行，远程机报错  Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR, Aborted (core dumped)
    * dlf4_dcgan.py，训练存在问题，不收敛
* 参考
    * 官方教程，[链接](https://www.tensorflow.org/tutorials/generative/dcgan?hl=zh-cn)
    * 《Python 深度学习》第8章

20轮训练以后得到的生成图像  
<img src="imgs/dcgan.png">

## 4. Image Translation
* 数据集 horse and zebra from ImageNet
    * 939张马的图片，1177张斑马的图片
    * 图像尺寸：3 x 256 x 256
### 模型 Cycle GAN
* 超参数设置
    * batch-size 8
    * 训练轮数 100
    * 学习率 2e-4
    * Regularization term 10