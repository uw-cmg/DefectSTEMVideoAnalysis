# Defect STEM Video Analysis

This project provides the source code for the paper ["Deep Learning based Automatic Defect Analysis System for In-situ Ion Irradiations TEM Videos"]().

## Installation

Our code is modified from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3). It is build on `Python 3.6` with `Keras` and `Tensorflow` as backend engine. It was tested on `CUDA 9.0` with `cudnn` and `cudatoolkit`.

## Run

You first need to download the pretrained weights or using the weights we trained.

* [pretrained weights for YOLO]()
* [our trained weights for the paper]()

You can run the training with the following command

```

python3 train.py

```

You can run the testing with the following command

```

python3 yolo_IoU_test.py

```