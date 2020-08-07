# Deep Learning Frameworks Performance Evaluation
This project is targeted to evaluate the performances of three deep learning frameworks on some different and typical deep nerual networks, including:
1. Google TensorFlow
2. Facebook PyTorch
3. Baidu PaddlePaddle

These three frameworks are evaluated by the following tasks:
* task1: image classification on CIFAR10 by VGG16
* task2: text sentimental classification on IMDB movie reviews by bi-lstm
* task3: text sentimental classification on IMDB movie reviews by bert
* task4: image generation on MINST dataset by DCGAN
* task5: image translation on horse2zebra dataset by CycleGAN (Paddle implementation not included)


# Project Structure
All tasks implemented by these three frameworks are placed in three directories named by the framework. Three directories are:
1. Tensorflow
2. Pytorch
3. Paddle

In each framework's directory, each task is implemented in directoy as:
* task1 in directory: imageclassification
* task2 in directory: textclassify_bilstm
* task3 in directory: textclassify_bert
* task4 in directory: DCGAN
* task5 in directory: CycleGAN

# Run the task
Please refer the readme in each framework's directory

