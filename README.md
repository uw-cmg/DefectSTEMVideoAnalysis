# Defect STEM Video Analysis

This project provides the source code for the paper  [A Deep Learning Based Automatic Defect Analysis Framework for In-situ TEM Ion Irradiations]() 

## Installation

Our code is modified from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3). It is build on `Python 3.6` with `Keras` and `Tensorflow` as backend engine. It was tested on `CUDA 9.0` with `cudnn` and `cudatoolkit`.

## Run

You first need to download the weights we trained. If you want to train for original YOLO weights, following instructions described [here](https://github.com/uw-cmg/DefectSTEMVideoAnalysis/tree/master/codes)

* [our trained weights for the paper](https://github.com/uw-cmg/DefectSTEMVideoAnalysis/tree/master/weights)

You can run the training with the following command

```

python3 train.py

```
You can run the demo of the code in this [Notebook]() via Google Colab.

