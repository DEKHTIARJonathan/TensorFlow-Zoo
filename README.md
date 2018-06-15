# Deep Learning Model Zoo using Tensorflow and Tensorlayer

[![Build Status](https://travis-ci.org/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo.svg?branch=master)](https://travis-ci.org/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo)
[![Updates](https://pyup.io/repos/github/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/shield.svg)](https://pyup.io/repos/github/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/)
[![Python 3](https://pyup.io/repos/github/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/python-3-shield.svg)](https://pyup.io/repos/github/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/)

This project is compatible and tested with the following versions of:
  * [TensorFlow](https://github.com/tensorflow/tensorflow): 1.6.0+
  * [TensorLayer](https://github.com/tensorlayer/tensorlayer): 1.8.6+

## Model Zoo

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[Inception V4](https://arxiv.org/abs/1602.07261)|[Code](https://github.com/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/blob/master/models/inceptionV4.py)|[inception_v4.ckpt](http://www.smarter-engineering.com/models/inception_v4.ckpt)|80.2|95.2|
[VGG 16](https://arxiv.org/abs/1409.1556)|[Code](https://github.com/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/blob/master/models/vgg16.py)|[vgg16.ckpt](http://www.smarter-engineering.com/models/vgg16.ckpt)|71.5|89.8|
[MobileNet_v1_1.0_224](https://arxiv.org/abs/1704.04861)|[Code](https://github.com/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/blob/master/models/mobilenet.py)|[mobilenet.ckpt](http://www.smarter-engineering.com/models/mobilenet.ckpt)|70.9|89.9|
[SqueezeNet](https://arxiv.org/abs/1602.07360)|[Code](https://github.com/DEKHTIARJonathan/TensorFlow-TensorLayer-Zoo/blob/master/models/squeezenet.py)|[squeezenet.ckpt](http://www.smarter-engineering.com/models/squeezenet.ckpt)|Unknown|Unknown|

## Project Installation

If you are using a machine without an Nividia GPU:
```shell
pip install -r requirements.txt
```

If you are using a machine with an Nividia GPU:
```shell
pip install -r requirements-gpu.txt
```

## Testing the models

Running all the tests one by one (no output will be displayed, except errors):
```shell
pytest
```

Running the models one by one:
```shell
python -m tests.test_inceptionV4
python -m tests.test_vgg16
python -m tests.test_mobilenet
python -m tests.test_squeezenet
```